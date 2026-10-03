"""Tests του feature engineering (`elfantasy.features.build`).

Τρεις ομάδες ελέγχων:

* χειροποίητα μικρά παραδείγματα με αριθμούς που υπολογίζονται με το χέρι (rolling μέσοι,
  dnp_streak, ξεκούραση, κλίση λεπτών, PIR που δέχεται ο αντίπαλος, πρώτος αγώνας, όρια σεζόν,
  ταξινόμηση κατά ώρα έναρξης και όχι κατά gamecode, αγώνας χωρίς boxscore, ταυτόχρονοι αγώνες),
* έλεγχοι διαρροής (leakage): η αλλαγή ενός αγώνα δεν αλλάζει τα features του ίδιου αγώνα ή
  προηγούμενων, αλλά αλλάζει τα επόμενα, και η online διαδρομή (history + upcoming) δίνει τα
  ίδια features με το batch,
* διασταύρωση με αργή υλοποίηση αναφοράς (`feature_reference.Reference`) σε τυχαίες γραμμές.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from feature_reference import Reference
from synthetic_league import make_league

from elfantasy.features.build import (
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    ID_COLUMNS,
    META_COLUMNS,
    TARGET_COLUMNS,
    UPCOMING_COLUMNS,
    build_features,
)
from elfantasy.scoring import fantasy_score

# --------------------------------------------------------------------------------------
# Βοηθητικά για χειροποίητα δεδομένα
# --------------------------------------------------------------------------------------


def hrow(
    player,
    team,
    opp,
    gamecode,
    day,
    *,
    season=2020,
    hour=19,
    home=True,
    minutes=20.0,
    pir=10,
    dnp=False,
    starter=False,
    score=80,
    opp_score=70,
):
    """Μία γραμμή ιστορικού (παίκτης × αγώνας)."""
    won = score > opp_score
    return {
        "season": season,
        "gamecode": gamecode,
        "player_id": player,
        "team_code": team,
        "opp_code": opp,
        "home": home,
        "game_date": pd.Timestamp(day),
        "tipoff_utc": pd.NaT if hour is None else pd.Timestamp(f"{day} {hour:02d}:00"),
        "is_starter": starter,
        "minutes": 0.0 if dnp else float(minutes),
        "dnp": dnp,
        "pir": 0 if dnp else pir,
        "fantasy_score": 0.0 if dnp else fantasy_score(pir, won),
        "team_score": float(score),
        "opp_score": float(opp_score),
    }


def urow(player, team, opp, gamecode, day, *, season=2020, hour=19, home=True):
    """Μία γραμμή αγώνα που δεν έχει παιχτεί (upcoming)."""
    return {
        "player_id": player,
        "season": season,
        "gamecode": gamecode,
        "team_code": team,
        "opp_code": opp,
        "home": home,
        "game_date": pd.Timestamp(day),
        "tipoff_utc": pd.NaT if hour is None else pd.Timestamp(f"{day} {hour:02d}:00"),
    }


def frame(rows):
    return pd.DataFrame(rows)


def box_game(
    gamecode, day, home, away, home_pirs, away_pirs, score_home, score_away, *, season=2020, hour=19
):
    """Αγώνας με πλήρες boxscore: 5 παίκτες ανά ομάδα με 40 λεπτά ο καθένας (σύνολο 200)."""
    rows = []
    for team, opp, is_home, pirs, score, opp_score in (
        (home, away, True, home_pirs, score_home, score_away),
        (away, home, False, away_pirs, score_away, score_home),
    ):
        for index, pir in enumerate(pirs):
            rows.append(
                hrow(
                    f"{team}{index}",
                    team,
                    opp,
                    gamecode,
                    day,
                    season=season,
                    hour=hour,
                    home=is_home,
                    minutes=40.0,
                    pir=pir,
                    score=score,
                    opp_score=opp_score,
                )
            )
    return rows


def team_upcoming(gamecode, day, home, away, *, season=2020, hour=19):
    """Γραμμές upcoming για τους 5 παίκτες κάθε ομάδας (όπως στο `box_game`)."""
    rows = []
    for team, opp, is_home in ((home, away, True), (away, home, False)):
        for index in range(5):
            rows.append(
                urow(
                    f"{team}{index}",
                    team,
                    opp,
                    gamecode,
                    day,
                    season=season,
                    hour=hour,
                    home=is_home,
                )
            )
    return rows


def row_of(result, player, gamecode, season=2020):
    selected = result[
        (result["player_id"] == player)
        & (result["gamecode"] == gamecode)
        & (result["season"] == season)
    ]
    assert len(selected) == 1, f"expected exactly one row for {player} {season}/{gamecode}"
    return selected.iloc[0]


def ewm(values_most_recent_first):
    values = np.asarray(values_most_recent_first, dtype=float)[:12]
    weights = 0.5 ** (np.arange(len(values)) / 3.0)
    return float((weights * values).sum() / weights.sum())


# --------------------------------------------------------------------------------------
# Δομή εξόδου
# --------------------------------------------------------------------------------------


class TestOutputStructure:
    def test_columns_and_order(self, batch):
        assert list(batch.columns) == [
            *ID_COLUMNS,
            *META_COLUMNS,
            *TARGET_COLUMNS,
            *FEATURE_COLUMNS,
        ]

    def test_feature_list_is_documented_unique_and_numeric(self, batch):
        grouped = [name for names in FEATURE_GROUPS.values() for name in names]
        assert grouped == FEATURE_COLUMNS
        assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)
        assert all(batch[name].dtype == np.float64 for name in FEATURE_COLUMNS)

    def test_no_leaky_columns_among_the_features(self):
        forbidden = {"minutes", "points", "won", "pir", "fantasy_score", "team_score", "opp_score"}
        assert forbidden.isdisjoint(FEATURE_COLUMNS)

    def test_one_row_per_input_row_sorted_by_time(self, league, batch):
        assert len(batch) == len(league.history)
        order = batch[["tipoff_utc", "season", "gamecode", "player_id"]].reset_index(drop=True)
        assert order.equals(
            order.sort_values(list(order.columns), kind="stable").reset_index(drop=True)
        )

    def test_targets_only_for_appearances(self, batch):
        appearance = batch["is_appearance"]
        assert batch.loc[appearance, TARGET_COLUMNS].notna().all().all()
        assert batch.loc[~appearance, TARGET_COLUMNS].isna().all().all()
        assert (~appearance).sum() > 0  # το δείγμα περιέχει DNP

    def test_inputs_are_not_modified(self, league):
        history = league.history.copy()
        games = league.games.copy()
        build_features(league.history, games=league.games)
        pd.testing.assert_frame_equal(league.history, history)
        pd.testing.assert_frame_equal(league.games, games)

    def test_row_order_of_the_input_does_not_matter(self, league, batch):
        shuffled = league.history.sample(frac=1.0, random_state=3).reset_index(drop=True)
        again = build_features(shuffled, games=league.games)
        pd.testing.assert_frame_equal(batch, again)

    def test_empty_upcoming_is_the_same_as_none(self, league, batch):
        again = build_features(
            league.history, upcoming=league.schedule.iloc[0:0], games=league.games
        )
        pd.testing.assert_frame_equal(batch, again)

    def test_only_upcoming_rows_work_without_history(self):
        up = frame([urow("P1", "A", "B", 1, "2020-10-01")])
        result = build_features(frame([hrow("X", "A", "B", 0, "2020-09-01")]).iloc[0:0], up)
        row = result.iloc[0]
        assert row["is_upcoming"]
        assert row["games_played_total"] == 0
        assert row["home"] == 1.0
        assert np.isnan(row["pir_mean_5"])


# --------------------------------------------------------------------------------------
# Χειροποίητα παραδείγματα: φόρμα και λεπτά
# --------------------------------------------------------------------------------------


class TestFormAndMinutes:
    PIRS = [4, 8, 12, 16, 20, 24]
    MINUTES = [10.0, 14.0, 18.0, 22.0, 26.0, 30.0]
    STARTERS = [False, False, True, True, True, True]
    SCORES = [(80, 70), (70, 80), (80, 70), (70, 80), (80, 70), (70, 80)]  # νίκη στους περιττούς
    DAYS = ["2020-10-01", "2020-10-05", "2020-10-09", "2020-10-13", "2020-10-17", "2020-10-21"]

    @pytest.fixture
    def history(self):
        return frame(
            [
                hrow(
                    "P1",
                    "A",
                    "B",
                    index + 1,
                    self.DAYS[index],
                    pir=self.PIRS[index],
                    minutes=self.MINUTES[index],
                    starter=self.STARTERS[index],
                    score=self.SCORES[index][0],
                    opp_score=self.SCORES[index][1],
                )
                for index in range(6)
            ]
        )

    @pytest.fixture
    def result(self, history):
        return build_features(history, frame([urow("P1", "A", "B", 7, "2020-10-25")]))

    def test_rolling_means_of_pir_and_fantasy(self, history, result):
        row = row_of(result, "P1", 7)
        pir = np.array(self.PIRS, dtype=float)
        fantasy = history["fantasy_score"].to_numpy()
        assert row["pir_mean_3"] == pytest.approx(pir[-3:].mean())  # 20
        assert row["pir_mean_5"] == pytest.approx(pir[-5:].mean())  # 16
        assert row["pir_mean_10"] == pytest.approx(pir.mean())  # λιγότεροι από 10: όλοι
        assert row["pir_mean_20"] == pytest.approx(pir.mean())
        assert row["fantasy_mean_3"] == pytest.approx(fantasy[-3:].mean())
        assert row["fantasy_mean_5"] == pytest.approx(fantasy[-5:].mean())
        assert row["fantasy_mean_10"] == pytest.approx(fantasy.mean())
        assert (row["pir_mean_3"], row["pir_mean_5"], row["pir_mean_10"]) == pytest.approx(
            (20.0, 16.0, 14.0)
        )

    def test_the_fantasy_score_includes_the_win_bonus(self, history):
        # νίκη στους περιττούς αγώνες: PIR 4 -> 4.4, PIR 12 -> 13.2, PIR 20 -> 22.0
        assert history["fantasy_score"].tolist() == pytest.approx(
            [4.4, 8.0, 13.2, 16.0, 22.0, 24.0]
        )

    def test_exponentially_weighted_mean(self, history, result):
        row = row_of(result, "P1", 7)
        assert row["pir_ewm"] == pytest.approx(ewm([24, 20, 16, 12, 8, 4]))
        assert row["fantasy_ewm"] == pytest.approx(ewm(history["fantasy_score"].to_numpy()[::-1]))

    def test_volatility_and_pir_per_minute(self, result):
        row = row_of(result, "P1", 7)
        assert row["pir_std_5"] == pytest.approx(np.std([8, 12, 16, 20, 24], ddof=1))  # √40
        assert row["pir_per_min_5"] == pytest.approx(80 / 110)

    def test_minutes_features(self, result):
        row = row_of(result, "P1", 7)
        assert row["min_mean_3"] == pytest.approx(26.0)
        assert row["min_mean_5"] == pytest.approx(22.0)
        assert row["min_last"] == pytest.approx(30.0)
        assert row["min_trend_5"] == pytest.approx(4.0)  # +4 λεπτά ανά αγώνα
        assert row["starter_rate_5"] == pytest.approx(0.8)
        assert row["min_season_mean"] == pytest.approx(np.mean(self.MINUTES))

    def test_season_to_date_and_counts(self, result):
        row = row_of(result, "P1", 7)
        assert row["pir_season_mean"] == pytest.approx(14.0)
        assert row["games_played_season"] == 6
        assert row["games_played_total"] == 6
        assert row["days_since_last_appearance"] == 4  # 21 -> 25 Οκτωβρίου

    def test_features_of_earlier_rows_only_use_earlier_games(self, result):
        # Στον 3ο αγώνα ο παίκτης έχει 2 προηγούμενες συμμετοχές: PIR 4 και 8
        row = row_of(result, "P1", 3)
        assert row["pir_mean_5"] == pytest.approx(6.0)
        assert row["games_played_total"] == 2
        assert row["pir_std_5"] == pytest.approx(np.std([4, 8], ddof=1))
        assert np.isnan(row["min_trend_5"])  # λιγότερες από 3 συμμετοχές

    def test_a_trend_needs_three_appearances(self, result):
        assert np.isnan(row_of(result, "P1", 3)["min_trend_5"])
        assert row_of(result, "P1", 4)["min_trend_5"] == pytest.approx(4.0)

    def test_the_first_game_has_no_form_features(self, result):
        row = row_of(result, "P1", 1)
        form_and_minutes = [*FEATURE_GROUPS["form"], *FEATURE_GROUPS["minutes"]]
        assert row[form_and_minutes].isna().all()
        assert row["games_played_total"] == 0
        assert row["games_played_season"] == 0
        assert row["dnp_streak"] == 0
        assert np.isnan(row["days_since_last_appearance"])

    def test_windows_larger_than_the_history_use_what_exists(self, history):
        result = build_features(history.iloc[:2], frame([urow("P1", "A", "B", 3, "2020-10-09")]))
        row = row_of(result, "P1", 3)
        assert row["pir_mean_20"] == pytest.approx(6.0)
        assert row["pir_ewm"] == pytest.approx(ewm([8, 4]))

    def test_pir_per_minute_needs_enough_minutes(self):
        history = frame([hrow("P1", "A", "B", 1, "2020-10-01", minutes=3.0, pir=9)])
        result = build_features(history, frame([urow("P1", "A", "B", 2, "2020-10-05")]))
        assert np.isnan(row_of(result, "P1", 2)["pir_per_min_5"])


# --------------------------------------------------------------------------------------
# Συμμετοχές, DNP και διαθεσιμότητα
# --------------------------------------------------------------------------------------


class TestAvailability:
    def test_dnp_rows_do_not_enter_the_form_statistics(self):
        history = frame(
            [
                hrow("P1", "A", "B", 1, "2020-10-01", pir=10, minutes=20),
                hrow("P1", "A", "B", 2, "2020-10-05", dnp=True),
                hrow("P1", "A", "B", 3, "2020-10-09", pir=20, minutes=30),
                hrow("P1", "A", "B", 4, "2020-10-13", dnp=True),
            ]
        )
        result = build_features(history, frame([urow("P1", "A", "B", 5, "2020-10-17")]))
        row = row_of(result, "P1", 5)
        assert row["pir_mean_5"] == pytest.approx(15.0)  # μόνο οι δύο συμμετοχές
        assert row["games_played_total"] == 2
        assert row["min_last"] == pytest.approx(30.0)
        assert row["days_since_last_appearance"] == 8  # από τον αγώνα 3 (9 -> 17 Οκτωβρίου)

    def test_dnp_streak_counts_consecutive_dnp_rows_before_the_game(self):
        history = frame(
            [
                hrow("P1", "A", "B", 1, "2020-10-01"),
                hrow("P1", "A", "B", 2, "2020-10-05", dnp=True),
                hrow("P1", "A", "B", 3, "2020-10-09", dnp=True),
                hrow("P1", "A", "B", 4, "2020-10-13", dnp=True),
                hrow("P1", "A", "B", 5, "2020-10-17"),
                hrow("P1", "A", "B", 6, "2020-10-21", dnp=True),
            ]
        )
        result = build_features(history, frame([urow("P1", "A", "B", 7, "2020-10-25")]))
        streaks = {gc: row_of(result, "P1", gc)["dnp_streak"] for gc in range(1, 8)}
        assert streaks == {1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 0, 7: 1}

    def test_a_zero_minute_row_without_dnp_is_not_an_appearance(self):
        history = frame(
            [
                hrow("P1", "A", "B", 1, "2020-10-01", pir=10, minutes=20),
                hrow("P1", "A", "B", 2, "2020-10-05", pir=-1, minutes=0.0),
            ]
        )
        result = build_features(history, frame([urow("P1", "A", "B", 3, "2020-10-09")]))
        zero_row = row_of(result, "P1", 2)
        assert not zero_row["is_appearance"]
        assert np.isnan(zero_row["fantasy_score"]) and np.isnan(zero_row["pir"])
        row = row_of(result, "P1", 3)
        assert row["games_played_total"] == 1
        assert row["pir_mean_5"] == pytest.approx(10.0)
        assert row["dnp_streak"] == 1

    def test_days_since_last_appearance_is_capped(self):
        history = frame([hrow("P1", "A", "B", 1, "2018-10-01")])
        result = build_features(
            history, frame([urow("P1", "A", "B", 2, "2020-10-01", season=2020)])
        )
        assert row_of(result, "P1", 2)["days_since_last_appearance"] == 365

    def test_missed_games_among_the_last_ten_of_the_team(self):
        # Η ομάδα A παίζει 12 αγώνες. Ο P1 παίζει στους 1-3, δεν έχει γραμμή στους 4-6, έχει DNP
        # στον 7, παίζει στους 8-9, δεν έχει γραμμή στον 10 και παίζει στον 11.
        # Ο P2 έρχεται στον αγώνα 8: οι προηγούμενοι αγώνες δεν του χρεώνονται.
        days = pd.date_range("2020-10-01", periods=12, freq="3D").strftime("%Y-%m-%d")
        rows = []
        for gamecode in range(1, 12):
            rows.append(hrow("P0", "A", "B", gamecode, days[gamecode - 1]))
        appears_p1 = {1, 2, 3, 8, 9, 11}
        appears_p2 = {8, 10, 11}
        for gamecode in appears_p1:
            rows.append(hrow("P1", "A", "B", gamecode, days[gamecode - 1]))
        rows.append(hrow("P1", "A", "B", 7, days[6], dnp=True))
        for gamecode in appears_p2:
            rows.append(hrow("P2", "A", "B", gamecode, days[gamecode - 1]))
        up = frame([urow(p, "A", "B", 12, days[11]) for p in ("P0", "P1", "P2")])
        result = build_features(frame(rows), up)
        assert row_of(result, "P0", 12)["missed_last10"] == 0
        # Παράθυρο αγώνων 2..11: ο P1 έπαιξε στους 2, 3, 8, 9, 11 -> έχασε τους 4, 5, 6, 7, 10
        assert row_of(result, "P1", 12)["missed_last10"] == 5
        # Ο P2 μετρά από τον αγώνα 8 (πρώτη γραμμή του): έχασε μόνο τον 9
        assert row_of(result, "P2", 12)["missed_last10"] == 1
        # Την ώρα του πρώτου αγώνα του κανείς δεν έχει χάσει κάτι
        assert row_of(result, "P1", 1)["missed_last10"] == 0
        assert row_of(result, "P2", 8)["missed_last10"] == 0

    def test_missed_games_ignore_games_without_any_row_of_the_team(self):
        # Ο αγώνας 2 δεν έχει καμία γραμμή της ομάδας A (ελλιπές boxscore) αλλά υπάρχει στο games.
        rows = [
            hrow("P1", "A", "B", 1, "2020-10-01"),
            hrow("P1", "A", "B", 3, "2020-10-09"),
        ]
        games = frame(
            [
                {
                    "season": 2020,
                    "gamecode": gc,
                    "game_date": pd.Timestamp(day),
                    "tipoff_utc": pd.Timestamp(f"{day} 19:00"),
                    "home_code": "A",
                    "away_code": "B",
                    "home_score": 80.0,
                    "away_score": 70.0,
                }
                for gc, day in ((1, "2020-10-01"), (2, "2020-10-05"), (3, "2020-10-09"))
            ]
        )
        up = frame([urow("P1", "A", "B", 4, "2020-10-13")])
        with_games = build_features(frame(rows), up, games=games)
        row = row_of(with_games, "P1", 4)
        assert row["missed_last10"] == 0  # ο αγώνας 2 δεν μετρά
        assert row["team_games_season"] == 3  # αλλά ο αγώνας έγινε και μετρά στην ομάδα
        without = build_features(frame(rows), up)  # χωρίς games: ο αγώνας 2 είναι άγνωστος
        assert row_of(without, "P1", 4)["team_games_season"] == 2

    def test_games_played_counts_and_season_boundaries(self):
        history = frame(
            [
                hrow("P1", "A", "B", 1, "2020-10-01", season=2020, pir=10, minutes=20),
                hrow("P1", "A", "B", 2, "2020-10-05", season=2020, pir=20, minutes=30),
                hrow("P1", "A", "B", 3, "2020-10-09", season=2020, pir=30, minutes=25),
            ]
        )
        up = frame([urow("P1", "A", "B", 1, "2021-10-02", season=2021)])
        row = row_of(build_features(history, up), "P1", 1, season=2021)
        # Η φόρμα μεταφέρεται στη νέα σεζόν...
        assert row["pir_mean_5"] == pytest.approx(20.0)
        assert row["min_last"] == pytest.approx(25.0)
        assert row["games_played_total"] == 3
        # ...αλλά τα στατιστικά «της σεζόν μέχρι τώρα» ξεκινούν από την αρχή.
        assert row["games_played_season"] == 0
        assert np.isnan(row["pir_season_mean"])
        assert np.isnan(row["fantasy_season_mean"])
        assert np.isnan(row["min_season_mean"])
        assert row["days_since_last_appearance"] == 358
        assert row["team_games_season"] == 0

    def test_transfers_use_the_team_of_each_row(self):
        history = frame(
            [
                hrow("P1", "A", "B", 1, "2020-10-01"),
                hrow("P1", "C", "B", 2, "2020-10-05"),
            ]
        )
        up = frame([urow("P1", "C", "B", 3, "2020-10-09")])
        row = row_of(build_features(history, up), "P1", 3)
        assert row["games_played_total"] == 2
        assert row["team_games_season"] == 1  # ένας προηγούμενος αγώνας της C


# --------------------------------------------------------------------------------------
# Πλαίσιο αγώνα: ξεκούραση, φόρτος
# --------------------------------------------------------------------------------------


class TestScheduleContext:
    DAYS = ["2020-10-01", "2020-10-04", "2020-10-06", "2020-10-13", "2021-01-20"]

    @pytest.fixture
    def result(self):
        rows = [hrow("Q", "A", "B", i + 1, day) for i, day in enumerate(self.DAYS)]
        up = frame([urow("Q", "A", "B", 6, "2021-01-22")])
        return build_features(frame(rows), up)

    def test_rest_days_and_short_rest(self, result):
        rest = {gc: row_of(result, "Q", gc)["team_rest_days"] for gc in range(1, 7)}
        assert np.isnan(rest[1])  # πρώτος αγώνας της ομάδας στα δεδομένα
        assert [rest[2], rest[3], rest[4]] == [3, 2, 7]
        assert rest[5] == 14  # 99 ημέρες: περιορίζεται στο όριο
        assert rest[6] == 2
        short = {gc: row_of(result, "Q", gc)["team_short_rest"] for gc in range(2, 7)}
        assert short == {2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 1.0}

    def test_games_in_the_last_seven_days(self, result):
        last7 = {gc: row_of(result, "Q", gc)["team_games_last7"] for gc in range(1, 7)}
        # Ο αγώνας 7 ημέρες πριν δεν μετρά (ανοικτό όριο): 13 Οκτωβρίου - 6 Οκτωβρίου.
        assert last7 == {1: 0, 2: 1, 3: 2, 4: 0, 5: 0, 6: 1}

    def test_team_games_in_the_season(self, result):
        played = {gc: row_of(result, "Q", gc)["team_games_season"] for gc in range(1, 7)}
        assert played == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

    def test_the_opponent_context_is_computed_for_the_other_team(self):
        rows = [
            hrow("Q", "A", "B", 1, "2020-10-02"),
            hrow("Z", "B", "A", 1, "2020-10-02", home=False),
            hrow("Z", "B", "C", 2, "2020-10-05", home=True),
            hrow("Q", "A", "D", 3, "2020-10-06"),
        ]
        up = frame([urow("Q", "A", "B", 4, "2020-10-08")])
        row = row_of(build_features(frame(rows), up), "Q", 4)
        assert row["team_rest_days"] == 2  # A: 6 -> 8 Οκτωβρίου
        assert row["opp_rest_days"] == 3  # B: 5 -> 8 Οκτωβρίου
        assert row["opp_games_last7"] == 2
        assert row["team_games_last7"] == 2

    def test_home_flag(self):
        history = frame(
            [
                hrow("Q", "A", "B", 1, "2020-10-01", home=True),
                hrow("Q", "A", "B", 2, "2020-10-05", home=False),
            ]
        )
        up = frame([urow("Q", "A", "B", 3, "2020-10-09", home=False)])
        result = build_features(history, up)
        assert [row_of(result, "Q", gc)["home"] for gc in (1, 2, 3)] == [1.0, 0.0, 0.0]


# --------------------------------------------------------------------------------------
# Ισχύς ομάδων και αμυντική αξία αντιπάλου
# --------------------------------------------------------------------------------------


class TestTeamStrength:
    def games(self):
        rows = []
        rows += box_game(1, "2020-10-01", "A", "B", [10] * 5, [8] * 5, 90, 80)  # A: +10
        rows += box_game(2, "2020-10-05", "B", "A", [9] * 5, [11] * 5, 75, 70)  # A: -5
        rows += box_game(3, "2020-10-09", "A", "B", [12] * 5, [7] * 5, 85, 60)  # A: +25
        return rows

    def test_win_percentage_and_point_difference(self):
        result = build_features(
            frame(self.games()), frame(team_upcoming(4, "2020-10-13", "A", "B"))
        )
        a = row_of(result, "A0", 4)
        assert a["team_win_pct_5"] == pytest.approx(2 / 3)
        assert a["team_pd_5"] == pytest.approx((10 - 5 + 25) / 3)
        assert a["team_win_pct_season"] == pytest.approx(2 / 3)
        assert a["team_pd_season"] == pytest.approx(10.0)
        # Από την πλευρά της B οι ίδιοι αγώνες είναι: ήττα, νίκη, ήττα
        assert a["opp_win_pct_5"] == pytest.approx(1 / 3)
        assert a["opp_pd_5"] == pytest.approx(-10.0)
        b = row_of(result, "B0", 4)
        assert b["team_win_pct_5"] == pytest.approx(1 / 3)
        assert b["opp_win_pct_5"] == pytest.approx(2 / 3)

    def test_the_current_game_result_is_not_used(self):
        # Ο αγώνας 3 νικιέται από την A με +25, αλλά τα features του αγώνα 3 βλέπουν μόνο τους 1-2.
        result = build_features(frame(self.games()))
        row = row_of(result, "A0", 3)
        assert row["team_win_pct_5"] == pytest.approx(0.5)
        assert row["team_pd_5"] == pytest.approx(2.5)

    def test_rolling_window_is_five_games_but_the_season_mean_is_cumulative(self):
        rows = []
        for gc in range(
            1, 9
        ):  # 8 αγώνες: A νικά τους 6 πρώτους με +1 και χάνει τους 2 τελευταίους με -3
            home_score, away_score = (81, 80) if gc <= 6 else (77, 80)
            day = f"2020-10-{gc * 3:02d}"
            rows += box_game(gc, day, "A", "B", [10] * 5, [9] * 5, home_score, away_score)
        up = team_upcoming(9, "2020-10-27", "A", "B")
        row = row_of(build_features(frame(rows), frame(up)), "A0", 9)
        assert row["team_win_pct_5"] == pytest.approx(3 / 5)  # αγώνες 4-8: ΝΝΝ ΗΗ
        assert row["team_pd_5"] == pytest.approx((1 + 1 + 1 - 3 - 3) / 5)
        assert row["team_win_pct_season"] == pytest.approx(6 / 8)
        assert row["team_pd_season"] == pytest.approx((6 - 6) / 8)

    def test_season_to_date_resets_but_the_rolling_window_carries_over(self):
        rows = box_game(1, "2020-10-01", "A", "B", [10] * 5, [8] * 5, 90, 80, season=2020)
        up = team_upcoming(1, "2021-10-02", "A", "B", season=2021)
        result = build_features(frame(rows), frame(up))
        row = row_of(result, "A0", 1, season=2021)
        assert row["team_win_pct_5"] == pytest.approx(1.0)  # η φόρμα μεταφέρεται
        assert np.isnan(row["team_win_pct_season"])  # η σεζόν ξεκινά από την αρχή
        assert row["team_games_season"] == 0

    def test_a_game_without_a_boxscore_still_counts_through_the_games_table(self):
        history = frame(self.games()[:10])  # μόνο ο πρώτος αγώνας
        games = frame(
            [
                {
                    "season": 2020,
                    "gamecode": gc,
                    "game_date": pd.Timestamp(day),
                    "tipoff_utc": pd.Timestamp(f"{day} 19:00"),
                    "home_code": "A",
                    "away_code": "B",
                    "home_score": float(hs),
                    "away_score": float(aw),
                }
                for gc, day, hs, aw in ((1, "2020-10-01", 90, 80), (2, "2020-10-05", 70, 80))
            ]
        )
        up = frame(team_upcoming(3, "2020-10-09", "A", "B"))
        with_games = row_of(build_features(history, up, games=games), "A0", 3)
        assert with_games["team_win_pct_5"] == pytest.approx(0.5)  # ο αγώνας 2 (ήττα) μετρά
        assert with_games["team_games_season"] == 2
        without = row_of(build_features(history, up), "A0", 3)
        assert without["team_win_pct_5"] == pytest.approx(1.0)
        assert without["team_games_season"] == 1


class TestOpponentDefense:
    def test_pir_allowed_is_normalised_by_the_league_average(self):
        rows = []
        rows += box_game(1, "2020-10-01", "A", "B", [10] * 5, [8] * 5, 90, 80)  # A: 50, B: 40
        rows += box_game(2, "2020-10-05", "B", "A", [9] * 5, [11] * 5, 75, 70)  # B: 45, A: 55
        up = team_upcoming(3, "2020-10-09", "A", "B")
        result = build_features(frame(rows), frame(up))
        # Η B δέχτηκε 50 και 55 PIR (το PIR της A): μέσος 52,5. Η A δέχτηκε 40 και 45: 42,5.
        # Μέσος όρος λίγκας πριν τον αγώνα 3: (40 + 50 + 55 + 45) / 4 = 47,5.
        a = row_of(result, "A0", 3)
        assert a["opp_def_pir_5"] == pytest.approx(52.5 / 47.5)
        assert a["opp_def_pir_10"] == pytest.approx(52.5 / 47.5)
        b = row_of(result, "B0", 3)
        assert b["opp_def_pir_5"] == pytest.approx(42.5 / 47.5)

    def test_only_the_last_games_enter_the_rolling_window(self):
        rows = []
        for gc in range(1, 8):  # η B δέχεται 40, 40, ..., και μετά 70 στον 7ο αγώνα
            allowed = 70 if gc == 7 else 40
            rows += box_game(
                gc, f"2020-10-{gc * 3:02d}", "A", "B", [allowed // 5] * 5, [8] * 5, 80, 70
            )
        up = team_upcoming(8, "2020-10-27", "A", "B")
        result = build_features(frame(rows), frame(up))
        a = row_of(result, "A0", 8)
        # Μέσος όρος λίγκας: 14 γεγονότα (η B δέχεται 40 έξι φορές και 70 μία, η A δέχεται πάντα 40)
        league_mean = np.mean([40] * 13 + [70])
        assert a["opp_def_pir_5"] == pytest.approx(np.mean([40, 40, 40, 40, 70]) / league_mean)
        assert a["opp_def_pir_10"] == pytest.approx(np.mean([40] * 6 + [70]) / league_mean)

    def test_an_incomplete_boxscore_is_ignored_and_does_not_crash(self):
        rows = box_game(1, "2020-10-01", "A", "B", [10] * 5, [8] * 4, 90, 80)  # η B έχει 160 λεπτά
        up = team_upcoming(2, "2020-10-05", "B", "A")
        result = build_features(frame(rows), frame(up))
        # Το boxscore της B είναι ελλιπές: η A δεν έχει έγκυρη τιμή PIR που δέχεται (NaN).
        assert np.isnan(row_of(result, "B0", 2)["opp_def_pir_5"])
        # Η B δέχεται το PIR της A (πλήρες boxscore): έχει τιμή. Μέσος λίγκας: μόνο αυτή η τιμή.
        assert row_of(result, "A0", 2)["opp_def_pir_5"] == pytest.approx(1.0)

    def test_a_game_without_boxscore_gives_nan_not_a_crash(self):
        history = frame([hrow("P1", "A", "B", 1, "2020-10-01")])
        games = frame(
            [
                {
                    "season": 2020,
                    "gamecode": 2,
                    "game_date": pd.Timestamp("2020-10-05"),
                    "tipoff_utc": pd.Timestamp("2020-10-05 19:00"),
                    "home_code": "A",
                    "away_code": "B",
                    "home_score": 70.0,
                    "away_score": 60.0,
                }
            ]
        )
        up = frame([urow("P1", "A", "B", 3, "2020-10-09")])
        row = row_of(build_features(history, up, games=games), "P1", 3)
        assert np.isnan(row["opp_def_pir_5"])  # δεν υπάρχει καμία τιμή PIR που δέχεται η B
        assert row["opp_win_pct_5"] == pytest.approx(0.0)  # η B έχασε και τους δύο αγώνες

    def test_league_average_falls_back_to_the_previous_season_early_in_a_season(self):
        rows = box_game(1, "2020-10-01", "A", "B", [10] * 5, [8] * 5, 90, 80, season=2020)
        rows += box_game(2, "2020-10-05", "B", "A", [9] * 5, [11] * 5, 75, 70, season=2020)
        rows += box_game(1, "2021-10-01", "A", "B", [10] * 5, [10] * 5, 90, 80, season=2021)
        up = team_upcoming(2, "2021-10-05", "A", "B", season=2021)
        result = build_features(frame(rows), frame(up))
        # Λίγα γεγονότα στη νέα σεζόν (< 20): ο μέσος όρος λίγκας είναι της προηγούμενης σεζόν:
        # (40 + 50 + 55 + 45) / 4 = 47,5. Η B δέχτηκε στη νέα σεζόν 50 (PIR της A): μέσος των
        # τελευταίων 5 αγώνων της (50, 55, 50) = 51,667.
        a = row_of(result, "A0", 2, season=2021)
        assert a["opp_def_pir_5"] == pytest.approx(np.mean([50, 55, 50]) / 47.5)


# --------------------------------------------------------------------------------------
# Ταξινόμηση και ταυτόχρονοι αγώνες
# --------------------------------------------------------------------------------------


class TestOrdering:
    def test_games_are_ordered_by_tipoff_not_by_gamecode(self):
        # Ο αγώνας με gamecode 2 μετατέθηκε και παίχτηκε ΜΕΤΑ τον αγώνα με gamecode 3.
        history = frame(
            [
                hrow("P1", "A", "B", 1, "2020-10-01", minutes=10.0, pir=4),
                hrow("P1", "A", "B", 3, "2020-10-08", minutes=30.0, pir=8),
                hrow("P1", "A", "B", 2, "2020-10-15", minutes=20.0, pir=12),
            ]
        )
        result = build_features(history, frame([urow("P1", "A", "B", 4, "2020-10-22")]))
        row = row_of(result, "P1", 4)
        assert row["min_last"] == pytest.approx(20.0)  # ο τελευταίος ΧΡΟΝΙΚΑ αγώνας
        assert row["min_trend_5"] == pytest.approx(5.0)  # (10, 30, 20) κατά χρονολογική σειρά
        assert row["team_rest_days"] == 7
        assert row["pir_ewm"] == pytest.approx(ewm([12, 8, 4]))
        # Ο αγώνας με gamecode 2 βλέπει και τους δύο προηγούμενους (1 και 3):
        assert row_of(result, "P1", 2)["games_played_total"] == 2
        assert row_of(result, "P1", 3)["games_played_total"] == 1

    def test_two_games_on_the_same_day_are_ordered_by_tipoff(self):
        history = frame(
            [
                hrow("P1", "A", "B", 5, "2020-10-01", hour=17, pir=10),
                hrow("P2", "C", "D", 6, "2020-10-01", hour=20, pir=10),
            ]
        )
        up = frame(
            [
                urow("P1", "A", "B", 7, "2020-10-01", hour=21),
                urow("P2", "C", "D", 8, "2020-10-01", hour=18),
            ]
        )
        result = build_features(history, up)
        # Ο P1 παίζει στις 21:00: έχει πάρει το πρώτο του παιχνίδι των 17:00 ως προηγούμενο.
        assert row_of(result, "P1", 7)["games_played_total"] == 1
        # Ο P2 παίζει στις 18:00: το παιχνίδι του των 20:00 είναι ΜΕΤΑ.
        assert row_of(result, "P2", 8)["games_played_total"] == 0

    def test_games_without_a_tipoff_time_sort_to_the_end_of_their_day(self):
        history = frame(
            [
                hrow("P1", "A", "B", 1, "2020-10-01", hour=None, pir=10),
                hrow("P1", "A", "B", 2, "2020-10-02", hour=19, pir=20),
            ]
        )
        result = build_features(history)
        assert row_of(result, "P1", 2)["games_played_total"] == 1
        assert row_of(result, "P1", 1)["games_played_total"] == 0

    def test_simultaneous_rows_of_the_same_player_do_not_see_each_other(self):
        history = frame(
            [
                hrow("P1", "A", "B", 1, "2020-10-01", pir=10),
                hrow("P1", "A", "B", 2, "2020-10-05", pir=30),
                hrow(
                    "P1", "A", "C", 3, "2020-10-05", pir=50
                ),  # ίδια ώρα έναρξης (σφάλμα δεδομένων)
            ]
        )
        result = build_features(history)
        first, second = row_of(result, "P1", 2), row_of(result, "P1", 3)
        assert first["games_played_total"] == second["games_played_total"] == 1
        assert first["pir_mean_5"] == second["pir_mean_5"] == pytest.approx(10.0)

    def test_simultaneous_games_do_not_leak_into_the_league_average(self):
        # Δύο ταυτόχρονοι αγώνες (ίδια ώρα): ο ένας δεν επηρεάζει το μέσο όρο λίγκας του άλλου.
        base = []
        base += box_game(1, "2020-10-01", "A", "B", [10] * 5, [8] * 5, 90, 80)
        simultaneous = box_game(2, "2020-10-05", "A", "B", [10] * 5, [8] * 5, 90, 80)
        simultaneous += box_game(3, "2020-10-05", "C", "D", [20] * 5, [20] * 5, 90, 80)
        changed = box_game(2, "2020-10-05", "A", "B", [10] * 5, [8] * 5, 90, 80)
        changed += box_game(3, "2020-10-05", "C", "D", [1] * 5, [1] * 5, 90, 80)
        first = build_features(frame(base + simultaneous))
        second = build_features(frame(base + changed))
        for player in ("A0", "B0"):
            before = row_of(first, player, 2)[FEATURE_COLUMNS].to_numpy(float)
            after = row_of(second, player, 2)[FEATURE_COLUMNS].to_numpy(float)
            np.testing.assert_array_equal(before, after)

    def test_an_earlier_game_on_the_same_day_does_enter_the_league_average(self):
        def run(c_pir):
            rows = box_game(1, "2020-10-01", "A", "B", [10] * 5, [8] * 5, 90, 80, hour=21)
            rows += box_game(2, "2020-10-01", "C", "D", [c_pir] * 5, [c_pir] * 5, 90, 80, hour=17)
            rows += box_game(3, "2020-09-20", "A", "B", [10] * 5, [8] * 5, 90, 80)
            return build_features(frame(rows))

        low = row_of(run(2), "A0", 1)["opp_def_pir_5"]
        high = row_of(run(30), "A0", 1)["opp_def_pir_5"]
        assert low != pytest.approx(high)  # ο αγώνας των 17:00 μετρά για αυτόν των 21:00


# --------------------------------------------------------------------------------------
# Ουδέτερο πλαίσιο και έλεγχοι εισόδου
# --------------------------------------------------------------------------------------


class TestNeutralContextAndValidation:
    def history(self):
        return frame(
            [
                hrow("P1", "A", "B", 1, "2020-10-01", pir=10, minutes=20),
                hrow("P1", "A", "B", 2, "2020-10-05", pir=20, minutes=30),
            ]
        )

    def test_a_row_without_a_scheduled_game_has_neutral_context(self):
        neutral = frame(
            [
                {
                    "player_id": "P1",
                    "season": 2020,
                    "gamecode": -1,
                    "team_code": "A",
                    "opp_code": None,
                    "home": np.nan,
                    "game_date": pd.Timestamp("2020-10-10"),
                    "tipoff_utc": pd.Timestamp("2020-10-11"),
                }
            ]
        )
        row = row_of(build_features(self.history(), neutral), "P1", -1)
        context = [
            *FEATURE_GROUPS["context"],
            *FEATURE_GROUPS["team_strength"],
            *FEATURE_GROUPS["opp_defense"],
            "team_games_season",
            "missed_last10",
        ]
        assert row[context].isna().all()
        # Τα features του ίδιου του παίκτη υπολογίζονται κανονικά
        assert row["pir_mean_5"] == pytest.approx(15.0)
        assert row["games_played_total"] == 2
        assert row["days_since_last_appearance"] == 5

    def test_a_neutral_row_sees_games_played_on_its_own_date(self):
        history = frame([hrow("P1", "A", "B", 1, "2020-10-10", hour=21, pir=10)])
        neutral = frame(
            [
                {
                    "player_id": "P1",
                    "season": 2020,
                    "gamecode": -1,
                    "team_code": "A",
                    "opp_code": None,
                    "home": np.nan,
                    "game_date": pd.Timestamp("2020-10-10"),
                    "tipoff_utc": pd.Timestamp("2020-10-11 00:00"),
                }
            ]
        )
        assert row_of(build_features(history, neutral), "P1", -1)["games_played_total"] == 1

    def test_missing_columns_are_reported(self):
        with pytest.raises(ValueError, match="history is missing required columns"):
            build_features(self.history().drop(columns=["minutes"]))
        with pytest.raises(ValueError, match="upcoming is missing required columns"):
            build_features(
                self.history(),
                frame([urow("P1", "A", "B", 3, "2020-10-09")]).drop(columns=["home"]),
            )

    def test_duplicate_rows_are_rejected(self):
        duplicated = pd.concat([self.history(), self.history().iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            build_features(duplicated)
        up = frame([urow("P1", "A", "B", 3, "2020-10-09")] * 2)
        with pytest.raises(ValueError, match="duplicate"):
            build_features(self.history(), up)

    def test_upcoming_rows_must_not_overlap_the_history(self):
        with pytest.raises(ValueError, match="overlap"):
            build_features(self.history(), frame([urow("P1", "A", "B", 2, "2020-10-05")]))

    def test_upcoming_rows_get_no_target(self):
        result = build_features(self.history(), frame([urow("P1", "A", "B", 3, "2020-10-09")]))
        row = row_of(result, "P1", 3)
        assert row["is_upcoming"] and not row["is_appearance"]
        assert np.isnan(row["fantasy_score"]) and np.isnan(row["pir"])

    def test_upcoming_columns_constant(self):
        assert set(UPCOMING_COLUMNS) == {
            "player_id",
            "season",
            "gamecode",
            "team_code",
            "opp_code",
            "home",
            "game_date",
            "tipoff_utc",
        }


# --------------------------------------------------------------------------------------
# Διαρροή (leakage) και συνέπεια διαδρομών
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def league():
    return make_league()


@pytest.fixture(scope="module")
def batch(league):
    return build_features(league.history, games=league.games)


def _index(frame_):
    return frame_.set_index(["player_id", "season", "gamecode"])


def _equal(a, b):
    np.testing.assert_allclose(
        a[FEATURE_COLUMNS].to_numpy(float),
        b[FEATURE_COLUMNS].to_numpy(float),
        rtol=1e-9,
        atol=1e-9,
        equal_nan=True,
    )


@pytest.fixture(scope="module")
def target_game(league):
    """Ένας αγώνας στη μέση της δεύτερης σεζόν, με παίκτες και των δύο ομάδων."""
    games = league.games[league.games["season"] == league.seasons[1]].sort_values("tipoff_utc")
    return games.iloc[len(games) // 2]


class TestNoLeakage:
    def test_changing_a_game_changes_later_features_but_not_its_own_or_earlier_ones(
        self, league, batch, target_game
    ):
        season, gamecode = int(target_game["season"]), int(target_game["gamecode"])
        in_game = (league.history["season"] == season) & (league.history["gamecode"] == gamecode)
        history = league.history.copy()
        history.loc[in_game, "pir"] += 9
        history.loc[in_game, "fantasy_score"] += 9.9
        history.loc[in_game & ~history["dnp"], "minutes"] += 4.0
        history.loc[in_game, "is_starter"] = ~history.loc[in_game, "is_starter"]
        history.loc[in_game, "team_score"] += 25.0
        history.loc[in_game, "won"] = ~history.loc[in_game, "won"]
        games = league.games.copy()
        games_mask = (games["season"] == season) & (games["gamecode"] == gamecode)
        games.loc[games_mask, "home_score"] += 25.0
        changed = build_features(history, games=games)

        time = target_game["tipoff_utc"]
        before = batch["tipoff_utc"] < time
        same_game = (batch["season"] == season) & (batch["gamecode"] == gamecode)
        _equal(batch[before], changed[before])  # όλα τα προηγούμενα: ίδια
        _equal(batch[same_game], changed[same_game])  # ο ίδιος ο αγώνας: ίδια features
        after = batch["tipoff_utc"] > time
        # Μετά τον αγώνα αλλάζουν features: οι παίκτες του αγώνα έχουν διαφορετική φόρμα
        players = set(league.history.loc[in_game & ~league.history["dnp"], "player_id"])
        later = batch[after & batch["player_id"].isin(players)]
        later_changed = changed.loc[later.index]
        assert not np.allclose(
            later["pir_mean_3"].to_numpy(float),
            later_changed["pir_mean_3"].to_numpy(float),
            equal_nan=True,
        )

    def test_changing_the_result_changes_the_team_strength_of_later_games_only(
        self, league, batch, target_game
    ):
        season, gamecode = int(target_game["season"]), int(target_game["gamecode"])
        games = league.games.copy()
        mask = (games["season"] == season) & (games["gamecode"] == gamecode)
        games.loc[mask, "home_score"] += 40.0
        history = league.history.copy()
        in_game = (history["season"] == season) & (history["gamecode"] == gamecode)
        history.loc[in_game & history["home"], "team_score"] += 40.0
        history.loc[in_game & ~history["home"], "opp_score"] += 40.0
        changed = build_features(history, games=games)
        time = target_game["tipoff_utc"]
        not_after = batch["tipoff_utc"] <= time
        _equal(batch[not_after], changed[not_after])
        home_team = target_game["home_code"]
        later_home = (batch["tipoff_utc"] > time) & (batch["team_code"] == home_team)
        assert (
            batch.loc[later_home, "team_pd_5"].to_numpy(float)
            != changed.loc[later_home, "team_pd_5"].to_numpy(float)
        ).any()

    @pytest.mark.parametrize("fraction", [0.25, 0.5, 0.8])
    def test_the_future_is_never_used(self, league, batch, fraction):
        """Η εξάλειψη όλων των αγώνων μετά από ένα σημείο δεν αλλάζει κανένα προηγούμενο feature."""
        cut = league.games["tipoff_utc"].sort_values().iloc[int(len(league.games) * fraction)]
        history = league.history[league.history["tipoff_utc"] <= cut]
        games = league.games[league.games["tipoff_utc"] <= cut]
        truncated = build_features(history, games=games)
        _equal(
            _index(batch[batch["tipoff_utc"] <= cut]).sort_index(), _index(truncated).sort_index()
        )

    def test_no_feature_of_a_game_depends_on_that_games_own_statistics(self, league, batch):
        """Αλλάζουν όλα τα στατιστικά όλων των αγώνων από μια σεζόν και μετά: τα features του
        πρώτου αγώνα της σεζόν (που βλέπει μόνο το παρελθόν) δεν αλλάζουν."""
        season = league.seasons[2]
        first = league.games[league.games["season"] == season].sort_values("tipoff_utc").iloc[0]
        history = league.history.copy()
        mask = history["season"] >= season
        history.loc[mask, ["pir", "fantasy_score"]] = 0.0
        history.loc[mask & ~history["dnp"], "minutes"] = 1.0
        history.loc[mask, ["team_score", "opp_score"]] = 50.0
        games = league.games.copy()
        games.loc[games["season"] >= season, ["home_score", "away_score"]] = 50.0
        changed = build_features(history, games=games)
        rows = (batch["season"] == season) & (batch["gamecode"] == first["gamecode"])
        _equal(batch[rows], changed[rows])


class TestOnlinePathMatchesBatch:
    @staticmethod
    def _games_to_check(league):
        games = league.games.sort_values("tipoff_utc").reset_index(drop=True)
        chosen = {0, len(games) // 3, 2 * len(games) // 3, len(games) - 1}
        for season in league.seasons[1:]:
            of_season = games[games["season"] == season]
            chosen.add(int(of_season.index[0]))  # πρώτος αγώνας σεζόν
        # ένας μετατεθειμένος αγώνας: παίζεται αργότερα από αγώνα με μεγαλύτερο gamecode
        postponed = None
        for season in league.seasons:
            by_code = games[games["season"] == season].sort_values("gamecode")
            later_min = by_code["game_date"][::-1].cummin()[::-1].shift(-1)
            late = by_code.index[by_code["game_date"] > later_min]
            if len(late):
                postponed = int(late[0])
                break
        assert postponed is not None, "the synthetic league must contain a postponed game"
        chosen.add(postponed)
        return games.loc[sorted(chosen), ["season", "gamecode", "tipoff_utc"]]

    def test_history_plus_upcoming_gives_the_features_of_the_batch(self, league, batch):
        checks = self._games_to_check(league)
        assert len(checks) >= 6
        for season, gamecode, tipoff in checks.itertuples(index=False):
            rows = league.history[
                (league.history["season"] == season) & (league.history["gamecode"] == gamecode)
            ]
            upcoming = rows[list(UPCOMING_COLUMNS)]
            history = league.history[league.history["tipoff_utc"] < tipoff]
            games = league.games[league.games["tipoff_utc"] < tipoff]
            online = build_features(history, upcoming, games=games)
            online = online[online["is_upcoming"]]
            expected = batch[(batch["season"] == season) & (batch["gamecode"] == gamecode)]
            assert len(online) == len(expected) == len(rows)
            _equal(_index(expected).sort_index(), _index(online).sort_index())

    def test_the_online_path_without_the_games_table_also_matches(self, league, batch):
        # Χωρίς `games`, οι αγώνες προκύπτουν από τις γραμμές του ιστορικού (εδώ δεν λείπει αγώνας).
        season, gamecode, tipoff = self._games_to_check(league).iloc[3]
        rows = league.history[
            (league.history["season"] == season) & (league.history["gamecode"] == gamecode)
        ]
        history = league.history[league.history["tipoff_utc"] < tipoff]
        online = build_features(history, rows[list(UPCOMING_COLUMNS)])
        online = online[online["is_upcoming"]]
        expected = batch[(batch["season"] == season) & (batch["gamecode"] == gamecode)]
        _equal(_index(expected).sort_index(), _index(online).sort_index())


class TestAgainstTheReferenceImplementation:
    def test_random_rows_match_the_naive_definition(self, league, batch):
        reference = Reference(league.history, league.games)
        sample = pd.concat(
            [
                batch.sample(28, random_state=1),
                batch[~batch["is_appearance"]].sample(6, random_state=2),  # γραμμές DNP
                batch.sort_values("tipoff_utc")
                .groupby("player_id")
                .head(1)
                .sample(8, random_state=3),
            ]
        ).drop_duplicates(["player_id", "season", "gamecode"])
        for row in sample.itertuples(index=False):
            expected = reference.features(
                {
                    "player_id": row.player_id,
                    "season": row.season,
                    "gamecode": row.gamecode,
                    "team_code": row.team_code,
                    "opp_code": row.opp_code,
                    "home": row.home,
                    "game_date": row.game_date,
                    "tipoff_utc": row.tipoff_utc,
                }
            )
            for name in FEATURE_COLUMNS:
                got, want = getattr(row, name), expected[name]
                if np.isnan(want):
                    assert np.isnan(got), (row.player_id, row.season, row.gamecode, name, got)
                else:
                    assert got == pytest.approx(want, rel=1e-9, abs=1e-9), (
                        row.player_id,
                        row.season,
                        row.gamecode,
                        name,
                    )
