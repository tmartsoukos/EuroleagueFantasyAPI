"""Tests του model/record_predictions.py: καταγραφή των προβλέψεων στον πίνακα `predictions`."""

import math
from datetime import UTC, date, datetime, timedelta

import pytest
from api_support import FakePredictor, make_prediction
from recorded_support import make_legacy_predictions
from sqlalchemy import func, inspect, select, text

from elfantasy.db import models
from elfantasy.db.session import get_engine
from elfantasy.model import record_predictions as recorder
from elfantasy.model.artifact import ModelLoadError
from elfantasy.model.predict import NextGame, Predictor
from elfantasy.model.record_predictions import (
    KEY_COLUMNS,
    prediction_rows,
    record_predictions,
)

SECRET = "S3cr3t-Pa55"
T1 = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)
T2 = T1 + timedelta(hours=5)


def stored(engine) -> list[dict]:
    with engine.connect() as connection:
        rows = connection.execute(select(models.predictions).order_by(models.predictions.c.id))
        return [dict(row) for row in rows.mappings()]


def count(engine) -> int:
    with engine.connect() as connection:
        return connection.execute(select(func.count()).select_from(models.predictions)).scalar_one()


def upcoming_game(league, index: int = 0) -> NextGame:
    """Ένας πραγματικός μελλοντικός αγώνας της βάσης του συνθετικού πρωταθλήματος."""
    row = league.schedule.iloc[index]
    return NextGame(
        season=int(row["season"]),
        gamecode=int(row["gamecode"]),
        game_date=row["game_date"].date(),
        tipoff_utc=row["tipoff_utc"].to_pydatetime(),
        team_code=row["home_code"],
        opp_code=row["away_code"],
        home=True,
    )


def naive(moment: datetime) -> datetime:
    return moment.replace(tzinfo=None)


class TestPredictionRows:
    def test_only_players_with_a_game_and_finite_values_become_rows(self):
        game = NextGame(2026, 31, date(2026, 10, 7), None, "AAA", "BBB", True)
        predictions = [
            make_prediction("P1", 12.5, pir=11.0, next_game=game),
            make_prediction("P2", 9.0),  # χωρίς επόμενο αγώνα
            make_prediction("P3", math.nan, next_game=game),
            make_prediction("P4", 5.0, pir=math.inf, next_game=game),
            make_prediction("P5", -3.25, pir=-4.0, next_game=game),
        ]
        rows, without_game, invalid = prediction_rows(predictions, date(2026, 10, 3), T1)
        assert (without_game, invalid) == (1, 2)
        assert rows == [
            {
                "player_id": "P1",
                "season": 2026,
                "gamecode": 31,
                "predicted_fantasy": 12.5,
                "predicted_pir": 11.0,
                "model_version": "fake-v1",
                "as_of": date(2026, 10, 3),
                "created_at": T1,
            },
            {
                "player_id": "P5",
                "season": 2026,
                "gamecode": 31,
                "predicted_fantasy": -3.25,
                "predicted_pir": -4.0,
                "model_version": "fake-v1",
                "as_of": date(2026, 10, 3),
                "created_at": T1,
            },
        ]

    def test_no_predictions(self):
        assert prediction_rows([], date(2026, 10, 3), T1) == ([], 0, 0)

    def test_the_key_columns_match_the_unique_constraint(self):
        constraint = next(
            c for c in models.predictions.constraints if c.__class__.__name__ == "UniqueConstraint"
        )
        assert KEY_COLUMNS == [column.name for column in constraint.columns]


class TestRecordPredictions:
    def test_the_predictions_of_the_model_are_stored(self, league_db, league_predictor, api_today):
        result = record_predictions(league_db, league_predictor, as_of=api_today, now=lambda: T1)
        predictions = league_predictor.predict_all(as_of=api_today, active_only=True)
        with_game = [p for p in predictions if p.next_game is not None]
        assert with_game, "το συνθετικό πρωτάθλημα έχει μελλοντικούς αγώνες"
        assert result.as_of == api_today
        assert result.model_version == "test-xgb-v1"
        assert result.predicted == len(predictions)
        assert result.recorded == len(with_game)
        assert result.skipped_without_game == len(predictions) - len(with_game)
        assert result.skipped_invalid == 0

        rows = {row["player_id"]: row for row in stored(league_db)}
        assert set(rows) == {p.player_id for p in with_game}
        for prediction in with_game:
            row = rows[prediction.player_id]
            assert (row["season"], row["gamecode"]) == (
                prediction.next_game.season,
                prediction.next_game.gamecode,
            )
            assert row["predicted_fantasy"] == pytest.approx(prediction.predicted_fantasy)
            assert row["predicted_pir"] == pytest.approx(prediction.predicted_pir)
            assert row["model_version"] == "test-xgb-v1" and row["as_of"] == api_today
            assert naive(row["created_at"]) == naive(T1)

    def test_rerunning_the_same_day_does_not_duplicate_rows(
        self, league_db, league_predictor, api_today
    ):
        first = record_predictions(league_db, league_predictor, as_of=api_today, now=lambda: T1)
        before = stored(league_db)
        second = record_predictions(league_db, league_predictor, as_of=api_today, now=lambda: T2)
        after = stored(league_db)
        assert first.recorded == second.recorded == len(before) == len(after)
        assert [r["id"] for r in after] == [r["id"] for r in before]  # ενημέρωση, όχι νέες γραμμές
        assert all(naive(r["created_at"]) == naive(T2) for r in after)
        for old, new in zip(before, after, strict=True):
            assert {k: v for k, v in old.items() if k != "created_at"} == {
                k: v for k, v in new.items() if k != "created_at"
            }

    def test_three_runs_still_give_one_row_per_player(self, league_db, league_predictor, api_today):
        for _ in range(3):
            record_predictions(league_db, league_predictor, as_of=api_today)
        keys = [tuple(row[column] for column in KEY_COLUMNS) for row in stored(league_db)]
        assert len(keys) == len(set(keys)) == count(league_db)

    def test_a_different_day_adds_a_new_snapshot(self, league_db, league_predictor, api_today):
        first = record_predictions(league_db, league_predictor, as_of=api_today)
        second = record_predictions(
            league_db, league_predictor, as_of=api_today + timedelta(days=1)
        )
        assert count(league_db) == first.recorded + second.recorded
        assert {row["as_of"] for row in stored(league_db)} == {
            api_today,
            api_today + timedelta(days=1),
        }

    def test_a_different_model_version_adds_its_own_rows(
        self, league_db, league_predictor, tiny_ridge_bundle, api_today
    ):
        record_predictions(league_db, league_predictor, as_of=api_today)
        ridge = Predictor(tiny_ridge_bundle, league_db, today=lambda: api_today)
        record_predictions(league_db, ridge, as_of=api_today)
        versions = [row["model_version"] for row in stored(league_db)]
        assert set(versions) == {"test-xgb-v1", "test-ridge-v1"}
        assert versions.count("test-xgb-v1") == versions.count("test-ridge-v1")

    def test_all_players_is_a_superset_of_the_active_players(
        self, league_db, league_predictor, api_today
    ):
        active = record_predictions(league_db, league_predictor, as_of=api_today, active_only=True)
        everyone = record_predictions(
            league_db, league_predictor, as_of=api_today, active_only=False
        )
        assert everyone.recorded >= active.recorded
        assert (
            count(league_db) == everyone.recorded
        )  # οι ενεργοί ενημερώθηκαν, οι υπόλοιποι προστέθηκαν

    def test_players_without_an_upcoming_game_are_not_recorded(self, league_db, synthetic_league):
        game = upcoming_game(synthetic_league)
        players = synthetic_league.players["player_id"].tolist()
        fake = FakePredictor(
            [
                make_prediction(players[0], 10.0, next_game=game),
                make_prediction(players[1], 8.0),  # offseason: χωρίς αγώνα
                make_prediction(players[2], 7.0),
            ]
        )
        result = record_predictions(league_db, fake, as_of=date(2026, 10, 3))
        assert (result.predicted, result.recorded, result.skipped_without_game) == (3, 1, 2)
        assert [row["player_id"] for row in stored(league_db)] == [players[0]]

    def test_no_predictions_at_all_is_fine(self, league_db):
        result = record_predictions(league_db, FakePredictor([]), as_of=date(2026, 10, 3))
        assert (result.predicted, result.recorded) == (0, 0)
        assert count(league_db) == 0

    def test_the_default_day_is_today_in_utc_and_the_default_clock_is_now(
        self, league_db, synthetic_league
    ):
        game = upcoming_game(synthetic_league)
        player = synthetic_league.players["player_id"].iloc[0]
        fake = FakePredictor([make_prediction(player, 5.0, next_game=game)])
        before = datetime.now(UTC)
        result = record_predictions(league_db, fake)
        after = datetime.now(UTC)
        assert result.as_of == datetime.now(UTC).date() or result.as_of == before.date()
        (row,) = stored(league_db)
        assert (
            naive(before) - timedelta(seconds=1)
            <= naive(row["created_at"])
            <= naive(after) + timedelta(seconds=1)
        )
        assert fake.predict_all_calls == [
            {"as_of": result.as_of, "team_code": None, "active_only": True}
        ]

    def test_the_active_only_flag_reaches_the_predictor(self, league_db, synthetic_league):
        fake = FakePredictor([])
        record_predictions(league_db, fake, as_of=date(2026, 10, 3), active_only=False)
        assert fake.predict_all_calls[0]["active_only"] is False

    def test_an_unknown_game_is_rejected_by_the_foreign_key(self, league_db, synthetic_league):
        ghost = NextGame(2099, 1, date(2099, 1, 1), None, "T00", "T01", True)
        player = synthetic_league.players["player_id"].iloc[0]
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            record_predictions(
                league_db, FakePredictor([make_prediction(player, 5.0, next_game=ghost)])
            )
        assert count(league_db) == 0  # τίποτα δεν γράφτηκε (μία συναλλαγή)

    def test_the_table_is_created_in_sqlite_when_missing(self, league_db, synthetic_league):
        with league_db.begin() as connection:
            connection.execute(text("DROP TABLE predictions"))
        game = upcoming_game(synthetic_league)
        player = synthetic_league.players["player_id"].iloc[0]
        record_predictions(league_db, FakePredictor([make_prediction(player, 5.0, next_game=game)]))
        assert count(league_db) == 1

    def test_the_legacy_empty_table_is_replaced(self, league_db, synthetic_league):
        make_legacy_predictions(league_db)
        game = upcoming_game(synthetic_league)
        player = synthetic_league.players["player_id"].iloc[0]
        record_predictions(league_db, FakePredictor([make_prediction(player, 5.0, next_game=game)]))
        assert "as_of" in {c["name"] for c in inspect(league_db).get_columns("predictions")}
        assert count(league_db) == 1


class TestCli:
    @pytest.fixture
    def db_url(self, league_db):
        return league_db.url.render_as_string(hide_password=False)

    def test_a_run_records_and_reports(
        self, db_url, league_db, league_predictor, api_today, monkeypatch, capsys
    ):
        calls = []

        def fake_load(cls, model_path=None, engine=None, **kwargs):
            calls.append((model_path, engine is not None))
            return league_predictor

        monkeypatch.setattr(Predictor, "load", classmethod(fake_load))
        assert recorder.main(["--db", db_url, "--as-of", api_today.isoformat()]) == 0
        out = capsys.readouterr().out
        assert f"as_of={api_today.isoformat()}" in out and "model test-xgb-v1" in out
        assert "Recorded " in out and "players without an upcoming game" in out
        assert count(league_db) > 0
        assert calls == [(None, True)]  # το μοντέλο από τις ρυθμίσεις και η δική του engine

    def test_the_model_option_and_the_all_flag(self, db_url, monkeypatch, capsys):
        fake = FakePredictor([])
        paths = []

        def fake_load(cls, model_path=None, engine=None, **kwargs):
            paths.append(model_path)
            return fake

        monkeypatch.setattr(Predictor, "load", classmethod(fake_load))
        assert (
            recorder.main(
                ["--db", db_url, "--model", "other/model.joblib", "--all", "--as-of", "2026-10-03"]
            )
            == 0
        )
        assert paths == ["other/model.joblib"]
        assert fake.predict_all_calls == [
            {"as_of": date(2026, 10, 3), "team_code": None, "active_only": False}
        ]
        capsys.readouterr()

    def test_active_only_is_the_default_and_the_flags_are_exclusive(self, db_url, monkeypatch):
        fake = FakePredictor([])
        monkeypatch.setattr(Predictor, "load", classmethod(lambda cls, *a, **k: fake))
        assert recorder.main(["--db", db_url, "--as-of", "2026-10-03", "--active-only"]) == 0
        assert fake.predict_all_calls[-1]["active_only"] is True
        with pytest.raises(SystemExit) as excinfo:
            recorder.main(["--all", "--active-only"])
        assert excinfo.value.code == 2

    def test_an_invalid_date_is_a_usage_error(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            recorder.main(["--as-of", "03/10/2026"])
        assert excinfo.value.code == 2
        assert "YYYY-MM-DD" in capsys.readouterr().err

    def test_the_default_database_comes_from_the_settings(
        self, league_db, db_url, monkeypatch, capsys
    ):
        monkeypatch.setenv("DATABASE_URL", db_url)
        recorder.get_settings.cache_clear()
        monkeypatch.setattr(Predictor, "load", classmethod(lambda cls, *a, **k: FakePredictor([])))
        assert recorder.main(["--as-of", "2026-10-03"]) == 0
        assert "Recorded 0 predictions" in capsys.readouterr().out

    def test_a_missing_model_exits_with_1(self, db_url, monkeypatch, capsys):
        def failing(cls, *args, **kwargs):
            raise ModelLoadError("artifact not found")

        monkeypatch.setattr(Predictor, "load", classmethod(failing))
        assert recorder.main(["--db", db_url]) == 1
        assert "the model could not be loaded" in capsys.readouterr().out

    def test_a_database_failure_exits_with_1_and_hides_the_password(
        self, db_url, monkeypatch, capsys
    ):
        def failing(cls, *args, **kwargs):
            raise RuntimeError(f"server said no to postgresql://u:{SECRET}@h/db")

        monkeypatch.setattr(Predictor, "load", classmethod(failing))
        assert recorder.main(["--db", db_url]) == 1
        out = capsys.readouterr().out
        assert "recording the predictions failed" in out and SECRET not in out

    def test_an_invalid_url_exits_with_2(self, capsys):
        assert recorder.main(["--db", f"postgresql://u:{SECRET}@h:badport/db"]) == 2
        assert SECRET not in capsys.readouterr().out

    def test_the_engine_and_the_predictor_are_closed(self, db_url, monkeypatch):
        closed = []
        fake = FakePredictor([])
        fake.close = lambda: closed.append("predictor")
        monkeypatch.setattr(Predictor, "load", classmethod(lambda cls, *a, **k: fake))
        assert recorder.main(["--db", db_url, "--as-of", "2026-10-03"]) == 0
        assert closed == ["predictor"]


def test_the_cli_works_against_a_database_it_creates_the_table_in(
    tmp_path, synthetic_league, league_template, monkeypatch
):
    """Μια βάση χωρίς πίνακα predictions (παλιά τοπική βάση) τον αποκτά στην πρώτη εκτέλεση."""
    import shutil

    copy = tmp_path / "old.db"
    shutil.copyfile(league_template, copy)
    engine = get_engine(f"sqlite:///{copy.as_posix()}")
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE predictions"))
    game = upcoming_game(synthetic_league)
    player = synthetic_league.players["player_id"].iloc[0]
    fake = FakePredictor([make_prediction(player, 5.0, next_game=game)])
    monkeypatch.setattr(Predictor, "load", classmethod(lambda cls, *a, **k: fake))
    assert recorder.main(["--db", f"sqlite:///{copy.as_posix()}", "--as-of", "2026-10-03"]) == 0
    assert count(engine) == 1
    engine.dispose()
