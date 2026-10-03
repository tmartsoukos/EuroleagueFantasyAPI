"""Integration tests του pipeline: cache (fixtures) -> clean -> scoring -> SQLite, και του verify.

Δεν γίνεται καμία κλήση δικτύου: το --no-fetch διαβάζει από ένα cache που φτιάχνεται από τα
πραγματικά fixtures και το fetch προσομοιώνεται με πηγή που απαντά από τα ίδια fixtures.
"""

import logging
from datetime import date, datetime

import pandas as pd
import pytest
import requests
from sqlalchemy import func, select, update

from elfantasy.db import models
from elfantasy.db.session import get_engine
from elfantasy.ingest import clean, pipeline, verify
from elfantasy.ingest.fetch import FetchError, RawCache

SEASONS = [2016, 2023, 2024, 2025, 2026]
EXPECTED_COUNTS = {"teams": 15, "players": 152, "games": 10, "player_games": 167, "predictions": 0}


class FixtureSource:
    """Πηγή που απαντά από τα fixtures αντί για το δίκτυο και καταγράφει τις κλήσεις."""

    def __init__(self, boxscores, results, schedule):
        self._boxscores = boxscores
        self._results = results
        self._schedule = schedule
        self.calls: list[tuple] = []
        self.failing: set[tuple[int, int]] = set()

    def results(self, season):
        self.calls.append(("results", season))
        return self._results[season]

    def schedule(self, season):
        self.calls.append(("schedule", season))
        return self._schedule[season]

    def boxscore(self, season, gamecode):
        self.calls.append(("boxscore", season, gamecode))
        if (season, gamecode) in self.failing:
            response = requests.Response()
            response.status_code = 404
            raise requests.HTTPError("404", response=response)
        rows = self._boxscores
        return rows[(rows["Season"] == season) & (rows["Gamecode"] == gamecode)].reset_index(
            drop=True
        )


@pytest.fixture
def fixture_source(raw_boxscores, raw_results, raw_schedule) -> FixtureSource:
    return FixtureSource(raw_boxscores, raw_results, raw_schedule)


@pytest.fixture
def data_dir(fixture_cache):
    return fixture_cache.root.parent


@pytest.fixture
def db_url(tmp_path) -> str:
    return f"sqlite:///{(tmp_path / 'test.db').as_posix()}"


@pytest.fixture
def restore_logging():
    """Το main() ρυθμίζει τον root logger: τον επαναφέρουμε και κλείνουμε το αρχείο του log."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)


def run_from_cache(data_dir, db_url, seasons=SEASONS, **options):
    return pipeline.run_pipeline(seasons, db_url=db_url, data_dir=data_dir, fetch=False, **options)


def read_table(db_url, table) -> pd.DataFrame:
    engine = get_engine(db_url)
    try:
        with engine.connect() as conn:
            return pd.read_sql_query(select(table), conn)
    finally:
        engine.dispose()


def counts_in(db_url) -> dict[str, int]:
    engine = get_engine(db_url)
    try:
        return pipeline.table_counts(engine)
    finally:
        engine.dispose()


def query(db_url, statement):
    engine = get_engine(db_url)
    try:
        with engine.connect() as conn:
            return conn.execute(statement).mappings().all()
    finally:
        engine.dispose()


class TestPipelineFromCache:
    def test_row_counts(self, data_dir, db_url):
        result = run_from_cache(data_dir, db_url)
        assert result.table_counts == EXPECTED_COUNTS
        assert result.rows_sent == {"teams": 15, "players": 152, "games": 10, "player_games": 167}
        assert (result.pir_matches, result.pir_rows) == (167, 167)
        assert result.quality_counts == {}
        assert result.missing_games == 0 and result.failed_seasons == []

    def test_fantasy_scores_of_the_documented_players(self, data_dir, db_url):
        run_from_cache(data_dir, db_url)
        pg = models.player_games.c
        rows = query(
            db_url,
            select(models.player_games).where(pg.season == 2025, pg.gamecode == 1),
        )
        by_id = {row["player_id"]: row for row in rows}
        larkin, hoard = by_id["P007200"], by_id["P006835"]
        hazer, beaubois = by_id["P011201"], by_id["P006590"]
        assert (larkin["pir"], larkin["valuation"], larkin["won"], larkin["fantasy_score"]) == (
            12,
            12,
            True,
            13.2,
        )
        assert (hoard["pir"], hoard["won"], hoard["fantasy_score"]) == (22, False, 22.0)
        assert (hazer["pir"], hazer["won"], hazer["fantasy_score"]) == (-4, True, -4.4)
        assert (beaubois["dnp"], beaubois["minutes"], beaubois["fantasy_score"]) == (True, 0.0, 0.0)
        assert larkin["team_code"] == "IST" and larkin["opp_code"] == "TEL" and larkin["home"]

    def test_every_row_follows_the_scoring_rule(self, data_dir, db_url):
        run_from_cache(data_dir, db_url)
        rows = read_table(db_url, models.player_games)
        assert (rows["pir"] == rows["valuation"]).all()
        expected = rows["pir"].where(~rows["won"], rows["pir"] * 11 / 10)
        assert (rows["fantasy_score"] == expected).all()
        assert rows["dnp"].sum() == 11
        dnp = rows[rows["dnp"]]
        assert (dnp["fantasy_score"] == 0).all() and (dnp["minutes"] == 0).all()

    def test_players_teams_and_names(self, data_dir, db_url):
        run_from_cache(data_dir, db_url)
        players = read_table(db_url, models.players).set_index("player_id")
        assert players.loc["P003469", "name"] == "VEZENKOV, SASHA"
        assert players.loc["P003469", ["first_season", "last_season"]].tolist() == [2016, 2026]
        assert {"PLRU", "P012711"} <= set(players.index)  # διαφορετικά IDs δεν συγχωνεύονται
        assert (players.index == players.index.str.strip()).all()
        teams = read_table(db_url, models.teams).set_index("team_code")["name"]
        assert teams["RED"] == "CRVENA ZVEZDA MERIDIANBET BELGRADE"

    def test_future_games_from_the_schedule_are_stored(self, data_dir, db_url):
        run_from_cache(data_dir, db_url)
        games = models.games.c
        future = query(
            db_url,
            select(models.games)
            .where(games.played.is_(False))
            .order_by(games.game_date, games.gamecode),
        )
        assert [row["gamecode"] for row in future] == [31, 32, 33]
        first = future[0]
        assert (first["home_code"], first["away_code"], first["phase"], first["round"]) == (
            "PRS",
            "ASV",
            "RS",
            4,
        )
        assert first["game_date"] == date(2026, 10, 7)
        assert first["tipoff_utc"] == datetime(2026, 10, 7, 18, 45)  # 20:45 CEST
        assert first["home_score"] is None and first["winner_code"] is None
        played = query(db_url, select(func.count()).select_from(models.games).where(games.played))
        assert played[0]["count_1"] == 7

    def test_second_run_is_idempotent(self, data_dir, db_url):
        first = run_from_cache(data_dir, db_url)
        tables = {
            table.name: read_table(db_url, table)
            for table in (models.teams, models.players, models.games, models.player_games)
        }
        second = run_from_cache(data_dir, db_url)
        assert second.table_counts == first.table_counts == EXPECTED_COUNTS
        assert second.rows_sent == first.rows_sent
        for table in (models.teams, models.players, models.games, models.player_games):
            pd.testing.assert_frame_equal(read_table(db_url, table), tables[table.name])

    def test_changed_source_data_is_updated_not_duplicated(self, data_dir, db_url, fixture_cache):
        run_from_cache(data_dir, db_url)
        # Διόρθωση στατιστικών στην πηγή: ο Larkin έχει μία ασίστ περισσότερη.
        box = fixture_cache.load_boxscores(2025)
        mask = (box["Gamecode"] == 1) & (box["Player_ID"].str.strip() == "P007200")
        box.loc[mask, "Assistances"] += 1
        box.loc[mask, "Valuation"] += 1
        total = (box["Gamecode"] == 1) & (box["Player_ID"] == "Total") & (box["Team"] == "IST")
        box.loc[total, ["Assistances", "Valuation"]] += 1
        fixture_cache.append_boxscores(2025, [box])
        result = run_from_cache(data_dir, db_url)
        assert result.table_counts == EXPECTED_COUNTS
        pg = models.player_games.c
        row = query(
            db_url,
            select(models.player_games).where(pg.player_id == "P007200", pg.season == 2025),
        )[0]
        assert (row["assists"], row["pir"], row["valuation"], row["fantasy_score"]) == (
            6,
            13,
            13,
            14.3,
        )

    def test_running_an_older_subset_does_not_shrink_players_or_revert_names(
        self, data_dir, db_url
    ):
        run_from_cache(data_dir, db_url)
        run_from_cache(data_dir, db_url, seasons=[2016])
        players = read_table(db_url, models.players).set_index("player_id")
        assert players.loc["P003469", "name"] == "VEZENKOV, SASHA"
        assert players.loc["P003469", ["first_season", "last_season"]].tolist() == [2016, 2026]
        teams = read_table(db_url, models.teams).set_index("team_code")["name"]
        assert teams["RED"] == "CRVENA ZVEZDA MERIDIANBET BELGRADE"  # όχι «MTS BELGRADE» του 2016
        assert teams["MIL"] == "ARMANI OLIMPIA MILAN"
        assert counts_in(db_url) == EXPECTED_COUNTS

    def test_loading_seasons_one_by_one_widens_the_range_and_updates_the_name(
        self, data_dir, db_url
    ):
        run_from_cache(data_dir, db_url, seasons=[2016])
        players = read_table(db_url, models.players).set_index("player_id")
        assert players.loc["P003469", "name"] == "VEZENKOV, ALEKSANDAR"
        assert players.loc["P003469", ["first_season", "last_season"]].tolist() == [2016, 2016]
        run_from_cache(data_dir, db_url, seasons=[2026])
        players = read_table(db_url, models.players).set_index("player_id")
        assert players.loc["P003469", "name"] == "VEZENKOV, SASHA"
        assert players.loc["P003469", ["first_season", "last_season"]].tolist() == [2016, 2026]
        teams = read_table(db_url, models.teams).set_index("team_code")["name"]
        assert teams["RED"] == "CRVENA ZVEZDA MTS BELGRADE"  # η σεζόν 2026 δεν έχει αγώνα της RED

    def test_reports_are_written(self, data_dir, db_url):
        run_from_cache(data_dir, db_url)
        anomalies = pd.read_csv(data_dir / "reports" / "name_anomalies.csv")
        assert sorted(anomalies["kind"]) == [
            "multiple_names_for_id",
            "multiple_names_for_id",
            "same_name_multiple_ids",
        ]
        issues = pd.read_csv(data_dir / "reports" / "quality_issues.csv")
        assert list(issues.columns) == clean.ISSUE_COLUMNS and issues.empty

    def test_a_quality_error_stops_the_load_and_leaves_a_report(
        self, data_dir, db_url, fixture_cache
    ):
        box = fixture_cache.load_boxscores(2025)
        duplicate = box[(box["Gamecode"] == 1) & (box["Player_ID"].str.strip() == "P007200")]
        # Το cache αντικαθιστά γραμμές ανά αγώνα, άρα γράφουμε απευθείας το αρχείο.
        broken = pd.concat([box, duplicate], ignore_index=True)
        broken.to_parquet(fixture_cache.path("boxscores", 2025), index=False)
        with pytest.raises(clean.DataQualityError, match="duplicate_key"):
            run_from_cache(data_dir, db_url)
        issues = pd.read_csv(data_dir / "reports" / "quality_issues.csv")
        assert "duplicate_key" in set(issues["check"])
        assert not (data_dir.parent / "test.db").exists()  # δεν δημιουργήθηκε καν η βάση

    def test_blocking_checks_can_be_allowed_explicitly(self, data_dir, db_url, fixture_cache):
        box = fixture_cache.load_boxscores(2025)
        mask = (box["Gamecode"] == 1) & (box["Player_ID"].str.strip() == "P007200")
        box.loc[mask, "FieldGoalsMade2"] += 1  # Points != 2·FGM2 + 3·FGM3 + FTM
        box.to_parquet(fixture_cache.path("boxscores", 2025), index=False)
        with pytest.raises(clean.DataQualityError, match="points_formula"):
            run_from_cache(data_dir, db_url)
        result = run_from_cache(data_dir, db_url, allow_quality_issues=["points_formula"])
        assert result.table_counts["player_games"] == 167
        assert result.quality_counts["points_formula"] == 1

    def test_no_cached_boxscores_is_an_error(self, tmp_path, db_url):
        with pytest.raises(FetchError, match="no cached boxscores"):
            run_from_cache(tmp_path / "empty", db_url)

    def test_a_season_with_games_but_no_boxscores_only_loads_the_games(
        self, data_dir, db_url, fixture_cache, caplog
    ):
        caplog.set_level(logging.WARNING, logger="elfantasy.ingest.pipeline")
        fixture_cache.path("boxscores", 2024).unlink()
        result = run_from_cache(data_dir, db_url)
        assert result.table_counts["games"] == 10  # ο αγώνας 2024/175 μένει στον πίνακα games
        assert result.table_counts["player_games"] == 167 - 24
        assert any("no boxscores in cache" in record.getMessage() for record in caplog.records)

    def test_seasons_without_any_cache_are_skipped(self, data_dir, db_url):
        result = run_from_cache(data_dir, db_url, seasons=[2015, 2025])
        assert result.table_counts["games"] == 1 and result.table_counts["player_games"] == 24


class TestPipelineWithFetch:
    def test_fetch_then_process(self, tmp_path, db_url, fixture_source):
        data_dir = tmp_path / "fresh"
        result = pipeline.run_pipeline(
            SEASONS,
            db_url=db_url,
            data_dir=data_dir,
            rps=1.5,
            source=fixture_source,
            sleep=lambda seconds: None,
        )
        assert result.table_counts == EXPECTED_COUNTS
        assert [(r.season, r.expected, r.fetched, r.missing) for r in result.fetch_reports] == [
            (2016, 2, 2, []),
            (2023, 1, 1, []),
            (2024, 1, 1, []),
            (2025, 1, 1, []),
            (2026, 2, 2, []),
        ]
        assert len([c for c in fixture_source.calls if c[0] == "boxscore"]) == 7
        cache = RawCache(data_dir / "raw")
        assert cache.boxscore_gamecodes(2016) == {1, 62}
        assert all(entry["missing"] == [] for entry in cache.load_missing().values())
        assert (data_dir / "raw" / "results_2026.parquet").exists()

    def test_second_run_only_refreshes_the_season_in_progress(
        self, tmp_path, db_url, fixture_source
    ):
        options = {"db_url": db_url, "data_dir": tmp_path / "fresh", "source": fixture_source}
        options["sleep"] = lambda seconds: None
        pipeline.run_pipeline(SEASONS, **options)
        fixture_source.calls.clear()
        second = pipeline.run_pipeline(SEASONS, **options)
        # Οι ολοκληρωμένες σεζόν δεν χρειάζονται δίκτυο, η 2026 έχει μελλοντικούς αγώνες.
        assert fixture_source.calls == [("results", 2026), ("schedule", 2026)]
        assert second.table_counts == EXPECTED_COUNTS

    def test_update_mode_refreshes_the_given_season_even_if_it_is_complete(
        self, tmp_path, db_url, fixture_source
    ):
        options = {"db_url": db_url, "data_dir": tmp_path / "fresh", "source": fixture_source}
        options["sleep"] = lambda seconds: None
        pipeline.run_pipeline(SEASONS, **options)
        fixture_source.calls.clear()
        result = pipeline.run_pipeline([2025], update=True, **options)
        assert fixture_source.calls == [("results", 2025), ("schedule", 2025)]
        assert result.table_counts == EXPECTED_COUNTS

    def test_a_missing_game_is_reported_and_everything_else_is_loaded(
        self, tmp_path, db_url, fixture_source
    ):
        fixture_source.failing.add((2016, 62))
        result = pipeline.run_pipeline(
            SEASONS,
            db_url=db_url,
            data_dir=tmp_path / "fresh",
            source=fixture_source,
            sleep=lambda seconds: None,
        )
        assert result.missing_games == 1
        report = next(r for r in result.fetch_reports if r.season == 2016)
        assert (report.expected, report.fetched, report.missing) == (2, 1, [62])
        assert result.table_counts["games"] == 10  # ο αγώνας υπάρχει στο schedule
        assert result.table_counts["player_games"] == 167 - 23  # αλλά όχι οι γραμμές του
        missing = RawCache(tmp_path / "fresh" / "raw").load_missing()
        assert missing["2016"]["missing"] == [62]
        assert "HTTP 404" in missing["2016"]["reasons"]["62"]


class TestCommandLine:
    def test_no_fetch_run(self, data_dir, db_url, restore_logging, capsys):
        code = pipeline.main(
            ["--seasons", "2016-2026", "--no-fetch", "--db", db_url, "--data-dir", str(data_dir)]
        )
        assert code == 0
        log = (data_dir / "logs" / "ingest.log").read_text(encoding="utf-8")
        assert "Starting ingestion" in log
        assert "PIR from raw statistics equals Valuation in 167 of 167 rows" in log
        assert "Row counts in the database" in log
        assert counts_in(db_url) == EXPECTED_COUNTS
        assert "Pipeline finished" in capsys.readouterr().out

    def test_returns_1_when_there_is_nothing_to_process(self, tmp_path, db_url, restore_logging):
        code = pipeline.main(["--no-fetch", "--db", db_url, "--data-dir", str(tmp_path / "empty")])
        assert code == 1

    def test_returns_2_when_games_are_missing(self, monkeypatch, restore_logging, tmp_path):
        report = pipeline.SeasonReport(season=2025, expected=3, fetched=2, missing=[3])
        stub = pipeline.PipelineResult(seasons=[2025], fetch_reports=[report])
        monkeypatch.setattr(pipeline, "run_pipeline", lambda *args, **kwargs: stub)
        assert pipeline.main(["--seasons", "2025", "--data-dir", str(tmp_path)]) == 2

    def test_returns_2_when_a_season_failed(self, monkeypatch, restore_logging, tmp_path):
        stub = pipeline.PipelineResult(
            seasons=[2025], fetch_reports=[pipeline.SeasonReport(season=2025, error="boom")]
        )
        monkeypatch.setattr(pipeline, "run_pipeline", lambda *args, **kwargs: stub)
        assert pipeline.main(["--seasons", "2025", "--data-dir", str(tmp_path)]) == 2

    def test_update_mode_uses_only_the_newest_season(self, monkeypatch, restore_logging, tmp_path):
        calls = []

        def fake_run(seasons, **kwargs):
            calls.append((list(seasons), kwargs))
            return pipeline.PipelineResult(seasons=list(seasons))

        monkeypatch.setattr(pipeline, "run_pipeline", fake_run)
        monkeypatch.setattr(pipeline, "current_season", lambda: 2026)
        assert pipeline.main(["--update", "--data-dir", str(tmp_path)]) == 0
        assert (
            pipeline.main(["--update", "--seasons", "2020-2024", "--data-dir", str(tmp_path)]) == 0
        )
        assert calls[0][0] == [2026] and calls[0][1]["update"] is True
        assert calls[1][0] == [2024]

    def test_default_seasons_run_from_2016_to_the_current_season(
        self, monkeypatch, restore_logging, tmp_path
    ):
        calls = []
        monkeypatch.setattr(
            pipeline,
            "run_pipeline",
            lambda seasons, **kwargs: (
                calls.append((list(seasons), kwargs))
                or pipeline.PipelineResult(seasons=list(seasons))
            ),
        )
        monkeypatch.setattr(pipeline, "current_season", lambda: 2026)
        pipeline.main(
            [
                "--rps",
                "0.8",
                "--allow-quality-issues",
                "points_formula, rebounds_sum",
                "--data-dir",
                str(tmp_path),
            ]
        )
        seasons, kwargs = calls[0]
        assert seasons == list(range(2016, 2027))
        assert kwargs["rps"] == 0.8 and kwargs["fetch"] is True and kwargs["update"] is False
        assert kwargs["allow_quality_issues"] == ["points_formula", "rebounds_sum"]

    @pytest.mark.parametrize(
        "error",
        [clean.DataQualityError("bad data"), FetchError("network"), RuntimeError("unexpected")],
    )
    def test_errors_give_exit_code_1(self, error, monkeypatch, restore_logging, tmp_path):
        def failing(*args, **kwargs):
            raise error

        monkeypatch.setattr(pipeline, "run_pipeline", failing)
        assert pipeline.main(["--seasons", "2025", "--data-dir", str(tmp_path)]) == 1


class TestVerify:
    @pytest.fixture
    def loaded(self, data_dir, db_url):
        run_from_cache(data_dir, db_url)
        engine = get_engine(db_url)
        yield engine
        engine.dispose()

    def test_clean_dataset_passes(self, loaded, data_dir):
        result = verify.verify(loaded, data_dir / "reports", expected_games={})
        assert result.failures == []
        assert (result.rows, result.pir_matches) == (167, 167)
        assert result.pir_match_rate == 1.0
        assert result.dnp_rows == 11 and result.dnp_rate == pytest.approx(11 / 167)
        assert result.stored_pir_differs == 0 and result.fantasy_differs == 0
        assert result.games_without_one_winner == 0 and result.games_without_rows.empty
        summary = result.season_summary.set_index("season")
        assert summary["games_played"].to_dict() == {2016: 2, 2023: 1, 2024: 1, 2025: 1, 2026: 2}
        assert summary["games_with_rows"].to_dict() == summary["games_played"].to_dict()
        assert summary["rows"].to_dict() == {2016: 47, 2023: 24, 2024: 24, 2025: 24, 2026: 48}
        assert (summary["games_missing"] == 0).all()
        assert result.pir_min == result.season_summary["pir_min"].min()
        mismatches = pd.read_csv(data_dir / "reports" / "pir_mismatches.csv")
        assert mismatches.empty and "diff" in mismatches.columns
        text = verify.format_report(result)
        assert "167 of 167 rows (100.0000%)" in text and "RESULT: OK" in text

    def test_pir_mismatches_are_reported_and_not_hidden(self, loaded, data_dir):
        with loaded.begin() as conn:
            conn.execute(
                update(models.player_games)
                .where(
                    models.player_games.c.player_id == "P007200",
                    models.player_games.c.season == 2025,
                )
                .values(valuation=15)
            )
        result = verify.verify(loaded, data_dir / "reports", expected_games={})
        assert result.pir_matches == 166
        assert any("pir != valuation" in failure for failure in result.failures)
        mismatches = pd.read_csv(data_dir / "reports" / "pir_mismatches.csv")
        assert mismatches[
            ["season", "gamecode", "player_id", "valuation", "pir", "diff"]
        ].values.tolist() == [[2025, 1, "P007200", 15, 12, -3]]
        text = verify.format_report(result)
        assert "RESULT: FAILED" in text and "by season: {2025: 1}" in text and "P007200" in text

    def test_stored_pir_that_is_not_recomputable_is_detected(self, loaded, data_dir):
        with loaded.begin() as conn:
            conn.execute(
                update(models.player_games)
                .where(models.player_games.c.player_id == "P007200")
                .values(pir=99)
            )
        result = verify.verify(loaded, data_dir / "reports", expected_games={})
        assert result.stored_pir_differs == 1
        assert any("stored pir is not recomputable" in failure for failure in result.failures)

    def test_inconsistent_fantasy_score_is_detected(self, loaded, data_dir):
        with loaded.begin() as conn:
            conn.execute(
                update(models.player_games)
                .where(models.player_games.c.player_id == "P007200")
                .values(fantasy_score=12.0)  # νίκη αλλά χωρίς το ×1,1
            )
        result = verify.verify(loaded, data_dir / "reports", expected_games={})
        assert result.fantasy_differs == 1
        assert any("fantasy_score" in failure for failure in result.failures)

    def test_games_without_player_rows_are_listed(self, loaded, data_dir):
        with loaded.begin() as conn:
            conn.execute(
                models.player_games.delete().where(
                    models.player_games.c.season == 2016, models.player_games.c.gamecode == 62
                )
            )
        result = verify.verify(loaded, data_dir / "reports", expected_games={})
        assert result.games_without_rows[["season", "gamecode"]].values.tolist() == [[2016, 62]]
        assert any("have no player rows" in failure for failure in result.failures)
        assert (data_dir / "reports" / "games_without_rows.csv").exists()
        text = verify.format_report(result)
        assert "first 30 (all in games_without_rows.csv)" in text and "2016" in text
        assert result.season_summary.set_index("season").loc[2016, "games_missing"] == 1

    def delete_game_rows(self, engine, season: int, gamecode: int) -> None:
        with engine.begin() as conn:
            conn.execute(
                models.player_games.delete().where(
                    models.player_games.c.season == season,
                    models.player_games.c.gamecode == gamecode,
                )
            )

    def test_accepted_gaps_are_reported_but_do_not_fail(self, loaded, data_dir):
        self.delete_game_rows(loaded, 2016, 62)
        result = verify.verify(
            loaded, data_dir / "reports", expected_games={}, accepted_missing={(2016, 62)}
        )
        assert result.failures == []
        assert result.games_without_rows.empty
        assert result.accepted_missing[["season", "gamecode"]].values.tolist() == [[2016, 62]]
        listed = pd.read_csv(data_dir / "reports" / "games_without_rows.csv")
        assert listed[["season", "gamecode"]].values.tolist() == [[2016, 62]]
        text = verify.format_report(result)
        assert "plus 1 accepted gaps at the source" in text and "RESULT: OK" in text
        assert "accepted gaps (--accept-missing)" in text
        # Αποδεκτό κενό που δεν λείπει στην πραγματικότητα: δεν επηρεάζει τίποτα.
        other = verify.verify(
            loaded, data_dir / "reports", expected_games={}, accepted_missing={(2025, 1)}
        )
        assert len(other.games_without_rows) == 1 and other.accepted_missing.empty

    def test_reasons_from_the_cache_are_attached_to_missing_games(self, loaded, data_dir):
        self.delete_game_rows(loaded, 2016, 62)
        cache = RawCache(data_dir / "raw")
        cache.record_season(
            pipeline.SeasonReport(
                2016, expected=2, fetched=1, missing=[62], reasons={62: "empty skeleton"}
            )
        )
        reasons = verify.load_missing_reasons(data_dir / "raw")
        assert reasons == {(2016, 62): "empty skeleton"}
        result = verify.verify(
            loaded, data_dir / "reports", expected_games={}, missing_reasons=reasons
        )
        assert result.games_without_rows["reason"].tolist() == ["empty skeleton"]
        assert "empty skeleton" in verify.format_report(result)

    def test_missing_reasons_without_a_cache_are_empty(self, tmp_path):
        assert verify.load_missing_reasons(tmp_path / "nothing") == {}

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("2018/21", {(2018, 21)}),
            ("2018/21, 2021/7,", {(2018, 21), (2021, 7)}),
            ("", set()),
        ],
    )
    def test_parse_game_keys(self, text, expected):
        assert verify.parse_game_keys(text) == expected

    @pytest.mark.parametrize("text", ["2018", "2018-21", "x/21", "2018/y", "2018/"])
    def test_invalid_game_keys(self, text):
        with pytest.raises(ValueError, match="SEASON/GAMECODE"):
            verify.parse_game_keys(text)

    def test_command_line_accepts_known_gaps(self, loaded, data_dir, db_url, monkeypatch, capsys):
        monkeypatch.setattr(verify, "EXPECTED_GAMES", {})
        self.delete_game_rows(loaded, 2016, 62)
        arguments = ["--db", db_url, "--data-dir", str(data_dir)]
        assert verify.main(arguments) == 1
        assert "RESULT: FAILED" in capsys.readouterr().out
        assert verify.main([*arguments, "--accept-missing", "2016/62"]) == 0
        assert "RESULT: OK" in capsys.readouterr().out
        with pytest.raises(SystemExit) as error:
            verify.main([*arguments, "--accept-missing", "oops"])
        assert error.value.code == 2

    def test_deviation_from_the_expected_number_of_games(self, loaded, data_dir):
        result = verify.verify(loaded, data_dir / "reports", expected_games={2016: 259, 2025: 1})
        assert result.failures == ["season 2016: 2 games in the database, 259 expected"]

    def test_a_game_with_two_winners_is_detected(self, loaded, data_dir):
        with loaded.begin() as conn:
            conn.execute(
                update(models.player_games)
                .where(
                    models.player_games.c.season == 2025,
                    models.player_games.c.gamecode == 1,
                    models.player_games.c.team_code == "TEL",
                )
                .values(won=True)
            )
        result = verify.verify(loaded, data_dir / "reports", expected_games={})
        assert result.games_without_one_winner == 1
        assert any("exactly one winning team" in failure for failure in result.failures)

    def test_empty_database_fails(self, tmp_path):
        engine = get_engine(f"sqlite:///{tmp_path / 'empty.db'}")
        from elfantasy.db.session import create_all

        create_all(engine)
        result = verify.verify(engine, tmp_path / "reports", expected_games={})
        assert result.failures == ["player_games is empty"]
        engine.dispose()

    def test_command_line_exit_codes(self, loaded, data_dir, db_url, monkeypatch, capsys):
        arguments = ["--db", db_url, "--data-dir", str(data_dir)]
        # Με τα πραγματικά αναμενόμενα πλήθη της Φάσης 1 το μικρό fixture αποκλίνει.
        assert verify.main(arguments) == 1
        assert "RESULT: FAILED" in capsys.readouterr().out
        monkeypatch.setattr(verify, "EXPECTED_GAMES", {})
        assert verify.main(arguments) == 0
        assert "RESULT: OK" in capsys.readouterr().out
