"""Tests του ingest/clean.py: boxscore, ημερομηνίες και ώρες, αγώνες, ονόματα, έλεγχοι."""

import locale
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from elfantasy.ingest import clean

GREEK_LOCALES = ["el_GR.UTF-8", "el_GR", "Greek_Greece.1253", "Greek_Greece.utf8", "Greek"]


def player_mask(raw: pd.DataFrame, season: int, gamecode: int, player_id: str) -> pd.Series:
    ids = raw["Player_ID"].str.strip()
    return (raw["Season"] == season) & (raw["Gamecode"] == gamecode) & (ids == player_id)


def run_clean(raw, results, schedule) -> clean.CleanData:
    return clean.clean_all(raw, results, schedule)


@pytest.fixture
def dataset(fixture_dataset) -> clean.CleanData:
    return fixture_dataset


class TestParseDate:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Sep 30, 2025", date(2025, 9, 30)),
            ("Oct 05, 2023", date(2023, 10, 5)),
            ("Jan 1, 2026", date(2026, 1, 1)),
            ("May 24, 2026", date(2026, 5, 24)),
            ("  Dec 31, 2016 ", date(2016, 12, 31)),
        ],
    )
    def test_valid_dates(self, text, expected):
        assert clean.parse_date(text) == expected

    def test_all_month_abbreviations(self):
        names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for number, name in enumerate(names, start=1):
            assert clean.parse_date(f"{name} 15, 2025") == date(2025, number, 15)

    @pytest.mark.parametrize(
        "text", ["30/09/2025", "Foo 30, 2025", "Sep 31, 2025", "", None, "2025-09-30"]
    )
    def test_invalid_dates(self, text):
        with pytest.raises(ValueError, match="Invalid date"):
            clean.parse_date(text)

    def test_does_not_depend_on_the_system_locale(self):
        original = locale.setlocale(locale.LC_TIME)
        for name in GREEK_LOCALES:
            try:
                locale.setlocale(locale.LC_TIME, name)
            except locale.Error:
                continue
            try:
                # Η αιτία του ρητού πίνακα μηνών: με ελληνικό locale το %b περιμένει «Σεπ».
                with pytest.raises(ValueError):
                    datetime.strptime("Sep 30, 2025", "%b %d, %Y")
                assert clean.parse_date("Sep 30, 2025") == date(2025, 9, 30)
                assert clean.parse_date("Mar 08, 2026") == date(2026, 3, 8)
            finally:
                locale.setlocale(locale.LC_TIME, original)
            return
        pytest.skip("no Greek locale installed on this machine")


class TestTipoff:
    def test_summer_time_is_utc_plus_two(self):
        # E2026_31: 07/10/2026 20:45 CEST -> 18:45 UTC (docs/DATA_SOURCES.md, ενότητα 8)
        assert clean.tipoff_to_utc(date(2026, 10, 7), "20:45") == datetime(2026, 10, 7, 18, 45)

    def test_winter_time_is_utc_plus_one(self):
        # Επαληθεύτηκε με το utcDate του v2: 06/01/2026 19:30 -> 18:30 UTC
        assert clean.tipoff_to_utc(date(2026, 1, 6), "19:30") == datetime(2026, 1, 6, 18, 30)

    @pytest.mark.parametrize(
        ("day", "expected_hour"),
        [
            (date(2026, 3, 28), 19),  # την παραμονή της αλλαγής ώρας (CET, UTC+1)
            (date(2026, 3, 29), 18),  # μέρα αλλαγής: 20:00 ώρα CEST (UTC+2)
            (date(2026, 10, 24), 18),  # CEST
            (date(2026, 10, 25), 19),  # μέρα επιστροφής σε CET
        ],
    )
    def test_daylight_saving_changes(self, day, expected_hour):
        assert clean.tipoff_to_utc(day, "20:00") == datetime(
            day.year, day.month, day.day, expected_hour
        )

    def test_result_is_naive_utc(self):
        assert clean.tipoff_to_utc(date(2025, 9, 30), "19:45").tzinfo is None

    @pytest.mark.parametrize("value", [None, np.nan, "", "  ", "TBD", "25:00", "20:75"])
    def test_missing_or_invalid_time_gives_none(self, value):
        assert clean.tipoff_to_utc(date(2026, 1, 1), value) is None


class TestMinutes:
    @pytest.mark.parametrize(
        ("text", "minutes", "dnp"),
        [
            ("33:21", 33 + 21 / 60, False),
            ("07:41", 7 + 41 / 60, False),
            ("00:20", 20 / 60, False),
            ("00:00", 0.0, False),
            ("", 0.0, False),  # 2022/313: παίκτες με φάουλ και κενό «Minutes»
            ("  ", 0.0, False),
            ("DNP", 0.0, True),
            ("dnp", 0.0, True),
            (" 10:30 ", 10.5, False),
            ("200:00", 200.0, False),
            ("225:00", 225.0, False),
        ],
    )
    def test_valid_values(self, text, minutes, dnp):
        value, is_dnp = clean.parse_minutes(text)
        assert value == pytest.approx(minutes)
        assert is_dnp is dnp

    @pytest.mark.parametrize("value", ["7:5", "07:75", "abc", None, np.nan, 12])
    def test_invalid_values(self, value):
        with pytest.raises(ValueError, match="Invalid Minutes"):
            clean.parse_minutes(value)

    def test_column_version_matches_scalar_and_keeps_the_index(self):
        series = pd.Series(["33:21", "DNP", "00:20", "200:00"], index=[10, 11, 12, 13])
        minutes, dnp = clean.parse_minutes_column(series)
        assert list(minutes.index) == [10, 11, 12, 13]
        assert minutes.tolist() == pytest.approx([33.35, 0.0, 20 / 60, 200.0])
        assert dnp.tolist() == [False, True, False, False]

    def test_column_version_treats_blank_text_as_zero_minutes_without_dnp(self):
        series = pd.Series(["", "DNP", " ", "10:30"])
        minutes, dnp = clean.parse_minutes_column(series)
        assert minutes.tolist() == pytest.approx([0.0, 0.0, 0.0, 10.5])
        assert dnp.tolist() == [False, True, False, False]
        for position, text in enumerate(series):
            assert clean.parse_minutes(text) == (minutes.iloc[position], dnp.iloc[position])

    def test_column_version_rejects_invalid_values(self):
        with pytest.raises(clean.DataQualityError, match="invalid Minutes"):
            clean.parse_minutes_column(pd.Series(["10:00", None, "xx"]))


class TestBoxscoreSplit:
    def test_split_strips_ids_and_removes_team_and_total(self, raw_boxscores):
        players, team, total = clean.split_boxscore(raw_boxscores)
        assert (len(players), len(team), len(total)) == (167, 14, 14)
        assert not players["Player_ID"].isin(["Team", "Total"]).any()
        assert (players["Player_ID"] == players["Player_ID"].str.strip()).all()
        assert "P007200" in set(players["Player_ID"])  # ο Larkin, χωρίς trailing spaces
        # Τα Total.Points κρατιούνται για έλεγχο του σκορ.
        ist_total = total[
            (total["Season"] == 2025) & (total["Gamecode"] == 1) & (total["Team"] == "IST")
        ]
        assert ist_total["Points"].tolist() == [85]

    def test_clean_players_columns_and_types(self, raw_boxscores):
        players = clean.clean_players(clean.split_boxscore(raw_boxscores)[0])
        assert "IsPlaying" not in players.columns and "is_playing" not in players.columns
        assert {"player_id", "team_code", "minutes", "dnp", "is_starter", "valuation"} <= set(
            players
        )
        assert players["is_starter"].dtype == bool
        assert players["dnp"].dtype == bool
        assert players["minutes"].dtype == float
        starters = players.groupby(["season", "gamecode", "team_code"])["is_starter"].sum()
        assert (starters == 5).all()

    def test_dnp_rows_have_zero_minutes_and_zero_statistics(self, raw_boxscores):
        players = clean.clean_players(clean.split_boxscore(raw_boxscores)[0])
        dnp = players[players["dnp"]]
        assert len(dnp) == 11
        assert (dnp["minutes"] == 0).all()
        assert (dnp[clean.INTEGER_STAT_COLUMNS] == 0).all().all()
        assert not (players.loc[~players["dnp"], "minutes"] == 0).any()

    def test_missing_columns_are_reported(self, raw_boxscores):
        players = clean.split_boxscore(raw_boxscores)[0].drop(columns=["Assistances"])
        with pytest.raises(clean.DataQualityError, match="Assistances"):
            clean.clean_players(players)

    def test_plusminus_column_is_optional(self, raw_boxscores):
        players = clean.split_boxscore(raw_boxscores)[0].drop(columns=["Plusminus"])
        assert clean.clean_players(players)["plus_minus"].isna().all()


class TestGames:
    def test_clean_results_ignores_the_played_flag_of_the_package(self, raw_results):
        results = raw_results[2025].copy()
        results["played"] = False  # το πακέτο δίνει True για κάθε μη κενό κείμενο: το αγνοούμε
        frame = clean.clean_results(results, 2025)
        assert frame["season"].tolist() == [2025]
        assert frame[["home_code", "away_code", "home_score", "away_score"]].iloc[0].tolist() == [
            "IST",
            "TEL",
            85,
            78,
        ]
        assert "played" not in frame.columns
        games, _ = clean.build_games(frame, clean.clean_schedule(None, 2025))
        assert games["played"].tolist() == [True]  # θετικό σκορ => παιγμένος

    def test_clean_schedule_reads_played_as_text(self, raw_schedule):
        frame = clean.clean_schedule(raw_schedule[2026], 2026)
        by_game = frame.set_index("gamecode")
        assert by_game.loc[5, "played"]
        assert not by_game.loc[31, "played"]
        assert by_game.loc[31, "tipoff_utc"] == pd.Timestamp("2026-10-07 18:45")
        assert by_game.loc[31, "round"] == 4
        assert by_game.loc[31, "game_date"] == date(2026, 10, 7)

    def test_build_games_combines_results_and_schedule(self, raw_results, raw_schedule):
        results = clean.clean_results(raw_results[2026], 2026)
        schedule = clean.clean_schedule(raw_schedule[2026], 2026)
        games, conflicts = clean.build_games(results, schedule)
        assert conflicts.empty
        assert sorted(games["gamecode"]) == [5, 8, 31, 32, 33]
        by_game = games.set_index("gamecode")
        assert by_game.loc[[5, 8], "played"].all()
        assert not by_game.loc[[31, 32, 33], "played"].any()
        assert by_game.loc[5, ["home_score", "away_score"]].tolist() == [89, 106]
        assert by_game.loc[31, ["home_score", "away_score"]].isna().all()
        assert by_game.loc[31, ["home_code", "away_code"]].tolist() == ["PRS", "ASV"]

    def test_build_games_reports_differences_between_results_and_schedule(
        self, raw_results, raw_schedule
    ):
        schedule = raw_schedule[2026].copy()
        schedule.loc[schedule["game"] == "5", "date"] = "Sep 26, 2026"  # μετάθεση
        results = clean.clean_results(raw_results[2026], 2026)
        games, conflicts = clean.build_games(results, clean.clean_schedule(schedule, 2026))
        assert conflicts[["season", "gamecode"]].values.tolist() == [[2026, 5]]
        # Για παιγμένο αγώνα υπερισχύει η ημερομηνία των results
        assert games.set_index("gamecode").loc[5, "game_date"] == date(2026, 9, 24)

    def test_build_games_from_results_only(self, raw_results):
        results = clean.clean_results(raw_results[2025], 2025)
        games, _ = clean.build_games(results, clean.clean_schedule(None, 2025))
        assert games["played"].tolist() == [True]

    def test_build_games_from_schedule_only(self, raw_schedule):
        schedule = clean.clean_schedule(raw_schedule[2026], 2026)
        games, _ = clean.build_games(clean.clean_results(None, 2026), schedule)
        assert games["played"].sum() == 2 and len(games) == 5
        assert games["home_score"].isna().all()

    def test_build_games_tolerates_inputs_without_the_optional_columns(self, raw_results):
        results = clean.clean_results(raw_results[2025], 2025)
        games, _ = clean.build_games(results, pd.DataFrame())  # schedule χωρίς στήλες
        assert games["played"].tolist() == [True]
        schedule = clean.clean_schedule(None, 2025)
        games, _ = clean.build_games(pd.DataFrame(), schedule)  # results χωρίς στήλες
        assert games.empty

    def test_results_row_with_zero_score_is_not_played(self, raw_results):
        results = raw_results[2025].copy()
        results["homescore"] = 0
        results["awayscore"] = 0
        games, _ = clean.build_games(
            clean.clean_results(results, 2025), clean.clean_schedule(None, 2025)
        )
        assert games.empty

    def test_scores_fall_back_to_boxscore_totals_when_results_are_missing(
        self, raw_boxscores, raw_schedule
    ):
        # Χωρίς results η νίκη προκύπτει από τα Total.Points του boxscore.
        data = run_clean(raw_boxscores, {}, raw_schedule)
        game = data.games[(data.games["season"] == 2025) & (data.games["gamecode"] == 1)].iloc[0]
        assert (game["home_score"], game["away_score"], game["winner_code"]) == (85, 78, "IST")
        rows = data.player_games[
            (data.player_games["season"] == 2025) & (data.player_games["gamecode"] == 1)
        ]
        assert rows.groupby("team_code")["won"].first().to_dict() == {"IST": True, "TEL": False}

    def test_games_are_sorted_by_date_and_time(self, dataset):
        games = dataset.games
        order = games[["game_date", "tipoff_utc"]].apply(tuple, axis=1).tolist()
        assert order == sorted(order)
        assert games["game_date"].is_monotonic_increasing

    def test_winner_code_for_played_and_future_games(self, dataset):
        by_game = dataset.games.set_index(["season", "gamecode"])
        assert by_game.loc[(2025, 1), "winner_code"] == "IST"
        assert by_game.loc[(2016, 62), "winner_code"] == "MAD"  # εκτός έδρας νίκη
        assert by_game.loc[(2026, 31), "winner_code"] is None

    def test_utc_tipoff_times_in_both_seasons_of_the_year(self, dataset):
        by_game = dataset.games.set_index(["season", "gamecode"])
        assert by_game.loc[(2025, 1), "tipoff_utc"] == pd.Timestamp("2025-09-30 17:45")  # CEST
        assert by_game.loc[(2024, 175), "tipoff_utc"] == pd.Timestamp("2025-01-09 19:05")  # CET
        assert by_game.loc[(2016, 62), "tipoff_utc"] == pd.Timestamp("2016-11-18 20:00")  # CET

    def test_no_results_and_no_schedule_is_an_error(self, raw_boxscores):
        with pytest.raises(clean.DataQualityError, match="no results or schedule"):
            run_clean(raw_boxscores, {}, {})


class TestPlayerGames:
    def test_fixture_is_clean(self, dataset):
        assert dataset.quality.issues.empty
        dataset.quality.raise_if_errors()
        assert (len(dataset.teams), len(dataset.players), len(dataset.games)) == (15, 152, 10)
        assert len(dataset.player_games) == 167

    def test_home_opponent_and_won_of_documented_players(self, dataset):
        rows = dataset.player_games
        game = rows[(rows["season"] == 2025) & (rows["gamecode"] == 1)].set_index("player_id")
        assert game.loc["P007200", ["team_code", "opp_code", "home", "won"]].tolist() == [
            "IST",
            "TEL",
            True,
            True,
        ]  # Larkin: νίκη
        assert game.loc["P006835", ["team_code", "opp_code", "home", "won"]].tolist() == [
            "TEL",
            "IST",
            False,
            False,
        ]  # Hoard: ήττα
        assert game.loc["P011201", "won"]  # Hazer
        assert game.loc["P006590", "dnp"] and game.loc["P006590", "minutes"] == 0  # Beaubois

    def test_team_and_opponent_are_the_two_teams_of_the_game(self, dataset):
        rows = dataset.player_games
        teams = rows.apply(lambda row: {row["team_code"], row["opp_code"]}, axis=1)
        expected = rows.apply(lambda row: {row["home_code"], row["away_code"]}, axis=1)
        assert (teams == expected).all()
        assert (rows["home"] == (rows["team_code"] == rows["home_code"])).all()

    def test_exactly_one_winning_team_per_game(self, dataset):
        winners = dataset.player_games.groupby(["season", "gamecode", "team_code"])["won"].first()
        assert (winners.groupby(["season", "gamecode"]).sum() == 1).all()

    def test_rows_are_in_chronological_order(self, dataset):
        dates = dataset.player_games["game_date"].tolist()
        assert dates == sorted(dates)

    def test_player_ids_are_stripped(self, dataset):
        assert (
            dataset.player_games["player_id"] == dataset.player_games["player_id"].str.strip()
        ).all()


class TestNamesAndTeams:
    def test_canonical_name_is_the_most_recent_one(self, dataset):
        players = dataset.players.set_index("player_id")
        assert players.loc["P003469", "name"] == "VEZENKOV, SASHA"  # 2016: ALEKSANDAR
        assert players.loc["P013250", "name"] == "DEJULIUS, DAVID"  # 2024: DeJULIUS
        assert players.loc["P003469", ["first_season", "last_season"]].tolist() == [2016, 2026]

    def test_canonical_name_does_not_depend_on_row_order(self, dataset):
        shuffled = dataset.player_games.sample(frac=1, random_state=7)
        pd.testing.assert_frame_equal(clean.canonical_players(shuffled), dataset.players)

    def test_different_ids_are_never_merged(self, dataset):
        players = dataset.players.set_index("player_id")
        assert {"PLRU", "P012711"} <= set(players.index)
        assert players.loc["PLRU", "name"] == players.loc["P012711", "name"] == "SIMONOVIC, MARKO"
        assert players.index.is_unique

    def test_name_anomalies_report(self, dataset):
        report = dataset.name_anomalies.set_index(["kind", "key"])
        names_for_id = report.loc["multiple_names_for_id"]
        assert set(names_for_id.index) == {"P003469", "P013250"}
        assert names_for_id.loc["P003469", "names"] == "VEZENKOV, ALEKSANDAR | VEZENKOV, SASHA"
        assert names_for_id.loc["P003469", "canonical_name"] == "VEZENKOV, SASHA"
        ids_for_name = report.loc["same_name_multiple_ids"]
        assert ids_for_name.loc["SIMONOVIC, MARKO", "player_ids"] == "P012711 | PLRU"

    def test_name_key_ignores_case_accents_and_extra_spaces(self):
        assert clean.name_key("González,  Hugo") == clean.name_key("GONZALEZ, HUGO")

    def test_same_name_with_accents_is_reported_as_multiple_ids(self):
        rows = pd.DataFrame(
            {
                "player_id": ["P1", "P2"],
                "player_name": ["GONZÁLEZ, HUGO", "GONZALEZ, HUGO"],
                "season": [2022, 2023],
                "gamecode": [1, 1],
                "game_date": [date(2022, 10, 1), date(2023, 10, 1)],
                "tipoff_utc": pd.to_datetime(["2022-10-01 18:00", "2023-10-01 18:00"]),
            }
        )
        report = clean.name_anomalies(rows)
        assert report["kind"].tolist() == ["same_name_multiple_ids"]
        assert report.loc[0, "player_ids"] == "P1 | P2"

    def test_no_anomalies_gives_an_empty_report(self):
        rows = pd.DataFrame(
            {
                "player_id": ["P1"],
                "player_name": ["A, B"],
                "season": [2022],
                "gamecode": [1],
                "game_date": [date(2022, 10, 1)],
                "tipoff_utc": pd.to_datetime(["2022-10-01 18:00"]),
            }
        )
        assert clean.name_anomalies(rows).empty

    def test_canonical_team_names_come_from_the_latest_data(self, dataset):
        names = dataset.teams.set_index("team_code")["name"]
        assert names["RED"] == "CRVENA ZVEZDA MERIDIANBET BELGRADE"  # 2016: «MTS BELGRADE»
        assert names["TEL"] == "MACCABI RAPYD TEL AVIV"  # 2024: «PLAYTIKA»
        assert (
            names["MIL"] == "ARMANI OLIMPIA MILAN"
        )  # από το schedule 2026 (2025: «EA7 EMPORIO ...»)

    def test_every_team_of_games_and_rows_is_in_the_teams_table(self, dataset):
        codes = set(dataset.teams["team_code"])
        assert set(dataset.games["home_code"]) | set(dataset.games["away_code"]) <= codes
        assert (
            set(dataset.player_games["team_code"]) | set(dataset.player_games["opp_code"]) <= codes
        )

    def test_build_teams_without_data(self):
        assert clean.build_teams([]).empty


class TestQualityChecks:
    def run(self, raw, results, schedule) -> clean.CleanData:
        return run_clean(raw, results, schedule)

    def checks(self, data: clean.CleanData) -> set[str]:
        return set(data.quality.issues["check"])

    def test_duplicate_rows_fail(self, raw_boxscores, raw_results, raw_schedule):
        mask = player_mask(raw_boxscores, 2025, 1, "P007200")
        raw = pd.concat([raw_boxscores, raw_boxscores[mask]], ignore_index=True)
        data = self.run(raw, raw_results, raw_schedule)
        assert "duplicate_key" in self.checks(data)
        with pytest.raises(clean.DataQualityError, match="duplicate_key"):
            data.quality.raise_if_errors()
        data.quality.raise_if_errors(allow=["duplicate_key", "team_total_mismatch"])  # δεν σηκώνει

    def test_nan_in_numeric_columns_fails(self, raw_boxscores, raw_results, raw_schedule):
        raw = raw_boxscores.copy()
        raw["Points"] = raw["Points"].astype(float)
        raw.loc[player_mask(raw, 2025, 1, "P007200"), "Points"] = np.nan
        data = self.run(raw, raw_results, raw_schedule)
        issues = data.quality.issues
        missing = issues[issues["check"] == "missing_values"]
        assert len(missing) == 1 and missing.iloc[0]["detail"] == "NaN in points"
        assert missing.iloc[0]["key"] == "P007200"
        with pytest.raises(clean.DataQualityError, match="missing_values"):
            data.quality.raise_if_errors()

    def test_points_that_do_not_match_the_shots_fail(
        self, raw_boxscores, raw_results, raw_schedule
    ):
        raw = raw_boxscores.copy()
        raw.loc[player_mask(raw, 2025, 1, "P007200"), "FieldGoalsMade2"] += 1
        data = self.run(raw, raw_results, raw_schedule)
        issues = data.quality.issues
        bad = issues[issues["check"] == "points_formula"]
        assert bad["key"].tolist() == ["P007200"]
        assert bad.iloc[0]["detail"] == "points=14, 2*FGM2+3*FGM3+FTM=16"
        assert (bad["severity"] == "error").all()

    def test_rebounds_that_do_not_add_up_fail(self, raw_boxscores, raw_results, raw_schedule):
        raw = raw_boxscores.copy()
        raw.loc[player_mask(raw, 2025, 1, "P007200"), "TotalRebounds"] += 1
        issues = self.run(raw, raw_results, raw_schedule).quality.issues
        bad = issues[issues["check"] == "rebounds_sum"]
        assert bad["key"].tolist() == ["P007200"]
        assert bad.iloc[0]["detail"] == "total_reb=5, off_reb+def_reb=4"

    def test_warnings_do_not_block(self, raw_boxscores, raw_results, raw_schedule):
        raw = raw_boxscores.copy()
        mask = player_mask(raw, 2025, 1, "P006590")  # Beaubois, DNP
        raw.loc[mask, "Assistances"] = 1  # DNP με στατιστικά
        negative = player_mask(raw, 2025, 1, "P007200")
        raw.loc[negative, "FieldGoalsMade3"] = (
            1  # εύστοχα (1) περισσότερα από τις προσπάθειες... (4)
        )
        raw.loc[negative, "FieldGoalsAttempted3"] = 0
        data = self.run(raw, raw_results, raw_schedule)
        checks = self.checks(data)
        assert {"dnp_with_stats", "made_exceeds_attempted", "team_total_mismatch"} <= checks
        warnings_only = data.quality.warnings["check"].unique().tolist()
        assert "dnp_with_stats" in warnings_only and "made_exceeds_attempted" in warnings_only

    def test_blank_minutes_count_as_zero_minutes_and_are_reported(
        self, raw_boxscores, raw_results, raw_schedule
    ):
        # 2022/313: παίκτης με ένα φάουλ και κενό «Minutes» (χωρίς ένδειξη DNP)
        raw = raw_boxscores.copy()
        mask = player_mask(raw, 2025, 1, "P007200")
        raw.loc[mask, "Minutes"] = ""
        data = self.run(raw, raw_results, raw_schedule)
        issues = data.quality.issues
        found = issues[issues["check"] == "zero_minutes_not_dnp"]
        assert found["key"].tolist() == ["P007200"]
        assert (found["severity"] == "warning").all()
        data.quality.raise_if_errors()  # μόνο προειδοποίηση: δεν σταματά το pipeline
        row = data.player_games[
            (data.player_games["season"] == 2025)
            & (data.player_games["gamecode"] == 1)
            & (data.player_games["player_id"] == "P007200")
        ].iloc[0]
        assert row["minutes"] == 0.0 and not row["dnp"]

    def test_negative_statistics_are_reported(self, raw_boxscores, raw_results, raw_schedule):
        raw = raw_boxscores.copy()
        raw.loc[player_mask(raw, 2025, 1, "P007200"), "Steals"] = -1
        assert "negative_stat" in self.checks(self.run(raw, raw_results, raw_schedule))

    def test_team_totals_must_equal_players_plus_team(
        self, raw_boxscores, raw_results, raw_schedule
    ):
        raw = raw_boxscores.copy()
        raw.loc[player_mask(raw, 2025, 1, "P007200"), "Assistances"] += 1
        issues = self.run(raw, raw_results, raw_schedule).quality.issues
        bad = issues[issues["check"] == "team_total_mismatch"]
        assert bad[["season", "gamecode", "key"]].values.tolist() == [[2025, 1, "IST"]]
        assert bad.iloc[0]["detail"] == "Assistances: -1"
        assert (bad["severity"] == "warning").all()

    @pytest.mark.parametrize(
        ("minutes", "flagged"),
        [
            ("200:00", False),
            ("225:00", False),  # μία παράταση
            ("250:00", False),  # δύο παρατάσεις
            ("199:01", False),  # μικρές αποκλίσεις υπάρχουν στην πηγή
            ("201:00", False),
            ("220:03", True),  # λείπουν ~5 λεπτά από μία παράταση
            ("176:43", True),  # 2017/14: λείπει παίκτης από το boxscore
        ],
    )
    def test_team_minutes_must_be_200_plus_25_per_overtime(
        self, raw_boxscores, raw_results, raw_schedule, minutes, flagged
    ):
        raw = raw_boxscores.copy()
        total = (raw["Season"] == 2025) & (raw["Gamecode"] == 1) & (raw["Player_ID"] == "Total")
        raw.loc[total & (raw["Team"] == "IST"), "Minutes"] = minutes
        issues = self.run(raw, raw_results, raw_schedule).quality.issues
        found = issues[issues["check"] == "team_minutes_mismatch"]
        if flagged:
            assert found[["season", "gamecode", "key", "severity"]].values.tolist() == [
                [2025, 1, "IST", "warning"]
            ]
            assert found.iloc[0]["detail"].startswith(f"Total minutes {minutes}, expected ")
        else:
            assert found.empty

    def test_team_minutes_check_without_data(self, raw_boxscores):
        total = clean.split_boxscore(raw_boxscores)[2]
        assert clean.check_team_minutes(total.iloc[0:0]).empty

    def test_total_points_must_match_the_result(self, raw_boxscores, raw_results, raw_schedule):
        results = {season: frame.copy() for season, frame in raw_results.items()}
        results[2025].loc[:, "homescore"] = 86
        issues = self.run(raw_boxscores, results, raw_schedule).quality.issues
        bad = issues[issues["check"] == "score_mismatch"]
        assert bad["key"].tolist() == ["IST"]
        assert bad.iloc[0]["detail"] == "Total.Points=85, results score=86"

    def test_home_flag_must_agree_with_the_game(self, raw_boxscores, raw_results, raw_schedule):
        raw = raw_boxscores.copy()
        in_game = (raw["Season"] == 2025) & (raw["Gamecode"] == 1)
        raw.loc[in_game, "Home"] = 1 - raw.loc[in_game, "Home"]
        data = self.run(raw, raw_results, raw_schedule)
        flagged = data.quality.issues[data.quality.issues["check"] == "home_flag_mismatch"]
        assert len(flagged) == 24 and set(flagged["season"]) == {2025}
        assert not data.quality.errors.shape[0]

    def test_game_without_results_or_schedule_fails(self, raw_boxscores, raw_results, raw_schedule):
        results = {season: frame.copy() for season, frame in raw_results.items()}
        schedule = {season: frame.copy() for season, frame in raw_schedule.items()}
        results[2016] = results[2016][results[2016]["gameCode"] != 62]
        schedule[2016] = schedule[2016][schedule[2016]["game"] != "62"]
        data = self.run(raw_boxscores, results, schedule)
        issues = data.quality.issues
        assert set(issues.loc[issues["check"] == "missing_game", "gamecode"]) == {62}
        with pytest.raises(clean.DataQualityError, match="missing_game"):
            data.quality.raise_if_errors()

    def test_team_not_in_game_fails(self, raw_boxscores, raw_results, raw_schedule):
        raw = raw_boxscores.copy()
        raw.loc[player_mask(raw, 2025, 1, "P007200"), "Team"] = "XXX"
        data = self.run(raw, raw_results, raw_schedule)
        issues = data.quality.issues
        assert issues.loc[issues["check"] == "team_not_in_game", "key"].tolist() == ["P007200"]

    def test_quality_counts_summary(self, raw_boxscores, raw_results, raw_schedule):
        raw = raw_boxscores.copy()
        raw.loc[player_mask(raw, 2025, 1, "P007200"), "Steals"] = -1
        counts = self.run(raw, raw_results, raw_schedule).quality.counts()
        assert counts["negative_stat"] == 1
        assert counts["team_total_mismatch"] == 1

    def test_check_team_totals_on_the_untouched_fixture(self, raw_boxscores):
        players, team, total = clean.split_boxscore(raw_boxscores)
        assert clean.check_team_totals(players, team, total).empty
        assert clean.check_team_totals(players, team, total.iloc[0:0]).empty

    def test_invalid_minutes_stop_the_cleaning(self, raw_boxscores, raw_results, raw_schedule):
        raw = raw_boxscores.copy()
        raw.loc[player_mask(raw, 2025, 1, "P007200"), "Minutes"] = "33:99"
        with pytest.raises(clean.DataQualityError, match="invalid Minutes"):
            self.run(raw, raw_results, raw_schedule)

    def test_results_and_schedule_conflicts_are_reported_by_clean_all(
        self, raw_boxscores, raw_results, raw_schedule
    ):
        schedule = {season: frame.copy() for season, frame in raw_schedule.items()}
        schedule[2025].loc[schedule[2025]["game"] == "1", "startime"] = "21:00"  # άλλη ώρα έναρξης
        data = self.run(raw_boxscores, raw_results, schedule)
        issues = data.quality.issues
        conflict = issues[issues["check"] == "results_schedule_conflict"]
        assert conflict[["season", "gamecode", "severity"]].values.tolist() == [
            [2025, 1, "warning"]
        ]
        data.quality.raise_if_errors()  # προειδοποίηση: δεν σταματά το pipeline

    def test_rows_that_are_all_incomplete_skip_the_value_checks(self, raw_boxscores):
        players = clean.clean_players(clean.split_boxscore(raw_boxscores)[0])
        players[clean.INTEGER_STAT_COLUMNS] = float("nan")
        issues = clean.check_player_rows(players)
        assert set(issues["check"]) == {"missing_values"}
        assert len(issues) == len(players)

    def test_total_scores_check_without_data(self, raw_boxscores):
        total = clean.split_boxscore(raw_boxscores)[2]
        assert clean.check_total_scores(total.iloc[0:0], pd.DataFrame()).empty
        assert clean.check_total_scores(total, pd.DataFrame()).empty
