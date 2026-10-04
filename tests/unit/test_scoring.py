"""Tests του scoring.py: PIR και fantasy score, με στόχο 100% ακρίβεια."""

import re
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from elfantasy import scoring
from elfantasy.scoring import (
    CLEAN_COLUMNS,
    RAW_COLUMNS,
    fantasy_score,
    fantasy_score_frame,
    pir,
    pir_frame,
    pir_from_row,
)

# Παραδείγματα του FANTASY_RULES.md, ενότητα 7 (πραγματικός αγώνας E2025_1, IST 85 - 78 TEL).
LARKIN = {  # νίκη, PIR 12
    "points": 14,
    "rebounds": 4,
    "assists": 5,
    "steals": 0,
    "blocks_favour": 0,
    "fouls_received": 4,
    "fg2_made": 5,
    "fg2_attempted": 8,
    "fg3_made": 0,
    "fg3_attempted": 4,
    "ft_made": 4,
    "ft_attempted": 6,
    "turnovers": 2,
    "blocks_against": 1,
    "fouls_committed": 3,
}
HOARD = {  # ήττα, PIR 22
    "points": 15,
    "rebounds": 9,
    "assists": 2,
    "steals": 0,
    "blocks_favour": 0,
    "fouls_received": 2,
    "fg2_made": 3,
    "fg2_attempted": 6,
    "fg3_made": 2,
    "fg3_attempted": 3,
    "ft_made": 3,
    "ft_attempted": 4,
    "turnovers": 1,
    "blocks_against": 0,
    "fouls_committed": 0,
}
HAZER = {  # νίκη, PIR -4
    "points": 0,
    "rebounds": 4,
    "assists": 0,
    "steals": 1,
    "blocks_favour": 0,
    "fouls_received": 0,
    "fg2_made": 0,
    "fg2_attempted": 1,
    "fg3_made": 0,
    "fg3_attempted": 2,
    "ft_made": 0,
    "ft_attempted": 0,
    "turnovers": 4,
    "blocks_against": 1,
    "fouls_committed": 1,
}
BEAUBOIS = dict.fromkeys(LARKIN, 0)  # δεν αγωνίστηκε (DNP): όλα τα στατιστικά 0


class TestPir:
    @pytest.mark.parametrize(
        ("stats", "expected"),
        [(LARKIN, 12), (HOARD, 22), (HAZER, -4), (BEAUBOIS, 0)],
        ids=["Larkin", "Hoard", "Hazer", "Beaubois-DNP"],
    )
    def test_documented_examples(self, stats, expected):
        assert pir(**stats) == expected
        assert isinstance(pir(**stats), int)

    def test_each_term_has_the_documented_sign(self):
        base = dict.fromkeys(LARKIN, 0)
        for positive in (
            "points",
            "rebounds",
            "assists",
            "steals",
            "blocks_favour",
            "fouls_received",
        ):
            assert pir(**{**base, positive: 3}) == 3, positive
        # Τα αρνητικά: αστοχημένα σουτ, αστοχημένες βολές, λάθη, μπλοκ που δέχεται, φάουλ που κάνει.
        assert pir(**{**base, "fg2_attempted": 2}) == -2
        assert pir(**{**base, "fg3_attempted": 3}) == -3
        assert pir(**{**base, "ft_attempted": 4}) == -4
        assert pir(**{**base, "turnovers": 5}) == -5
        assert pir(**{**base, "blocks_against": 2}) == -2
        assert pir(**{**base, "fouls_committed": 5}) == -5

    def test_made_shots_do_not_count_as_misses(self):
        stats = {**dict.fromkeys(LARKIN, 0), "fg2_made": 4, "fg2_attempted": 4, "ft_made": 2}
        stats["ft_attempted"] = 2
        assert (
            pir(**stats) == 0
        )  # τα εύστοχα σουτ δεν δίνουν ούτε πλην (οι πόντοι δίνονται χωριστά)

    def test_only_negatives(self):
        stats = {
            **dict.fromkeys(LARKIN, 0),
            "fg2_attempted": 6,
            "fg3_attempted": 5,
            "ft_attempted": 4,
            "turnovers": 3,
            "blocks_against": 2,
            "fouls_committed": 5,
        }
        assert pir(**stats) == -25

    def test_very_large_values(self):
        stats = {**dict.fromkeys(LARKIN, 0), "points": 10**9, "rebounds": 10**9}
        assert pir(**stats) == 2 * 10**9

    def test_keyword_only(self):
        with pytest.raises(TypeError):
            pir(*LARKIN.values())

    def test_rejects_non_integer_result(self):
        with pytest.raises(ValueError, match="integer"):
            pir(**{**LARKIN, "points": 14.5})

    def test_accepts_integral_floats_and_numpy_ints(self):
        assert pir(**{key: float(value) for key, value in LARKIN.items()}) == 12
        assert pir(**{key: np.int64(value) for key, value in LARKIN.items()}) == 12

    def test_from_row_with_clean_and_raw_column_names(self):
        clean_row = {CLEAN_COLUMNS[argument]: value for argument, value in LARKIN.items()}
        raw_row = {RAW_COLUMNS[argument]: value for argument, value in LARKIN.items()}
        assert pir_from_row(clean_row) == 12
        assert pir_from_row(pd.Series(raw_row), columns=RAW_COLUMNS) == 12

    def test_column_mappings_cover_all_arguments(self):
        assert set(CLEAN_COLUMNS) == set(RAW_COLUMNS) == set(LARKIN)


class TestFantasyScore:
    def test_documented_examples(self):
        assert fantasy_score(12, True) == 13.2  # Larkin, νίκη
        assert fantasy_score(12, False) == 12  # ίδιος αγώνας αν είχε χάσει
        assert fantasy_score(22, False) == 22  # Hoard, ήττα
        # Hazer, νίκη με αρνητικό PIR: το μπόνους προστίθεται θετικό (FANTASY_RULES.md, 3.4).
        assert fantasy_score(-4, True) == -3.6
        assert fantasy_score(-4, False) == -4  # ίδιος αγώνας αν είχε χάσει
        assert fantasy_score(0, True) == 0  # Beaubois, DNP
        assert fantasy_score(0, False) == 0

    def test_results_are_floats_without_float_noise(self):
        assert repr(fantasy_score(12, True)) == "13.2"  # το 12 * 1.1 θα έδινε 13.200000000000001
        assert repr(fantasy_score(-4, True)) == "-3.6"
        assert isinstance(fantasy_score(12, False), float)
        assert isinstance(fantasy_score(12, True), float)
        assert isinstance(fantasy_score(-4, True), float)

    def test_bonus_constants(self):
        # Το μπόνους νίκης είναι το 1/10 του |PIR|.
        assert (scoring.WIN_BONUS_NUM, scoring.WIN_BONUS_DEN) == (1, 10)

    def test_edge_cases(self):
        assert fantasy_score(0, True) == 0.0
        assert fantasy_score(10**9, True) == 1_100_000_000
        assert fantasy_score(-(10**9), True) == -900_000_000
        assert fantasy_score(-100, True) == -90.0
        assert fantasy_score(-100, False) == -100.0
        assert fantasy_score(-1, True) == -0.9
        assert fantasy_score(1, True) == 1.1

    def test_negative_pir_in_a_win_gets_a_positive_bonus(self):
        # Ο παλιός κανόνας ×1,1 έδινε −4,4: τα επίσημα σύνολα βαθμών τον αποκλείουν (§3.4, §9).
        assert fantasy_score(-4, True) != -4.4
        assert fantasy_score(-4, True) == round(-4 + 0.1 * 4, 1)
        assert fantasy_score(-10, True) == -9.0
        assert fantasy_score(-13, True) == -11.7  # το χαμηλότερο PIR του dataset

    def test_accepts_numpy_integers_and_numpy_bools(self):
        assert fantasy_score(np.int64(12), True) == 13.2
        assert fantasy_score(np.int8(-4), True) == -3.6
        assert fantasy_score(12, np.True_) == 13.2
        assert fantasy_score(12, np.False_) == 12.0
        assert fantasy_score(np.int64(-4), np.bool_(True)) == -3.6

    @pytest.mark.parametrize("value", range(-100, 151))
    def test_win_bonus_matches_rounding_and_has_no_float_noise(self, value):
        won = fantasy_score(value, True)
        assert won == round(value + abs(value) * 0.1, 1)
        assert re.fullmatch(r"-?\d+(\.\d)?", repr(won)), f"{value}: {won!r} has float noise"
        assert fantasy_score(value, False) == value

    @pytest.mark.parametrize("value", range(-100, 151))
    def test_win_bonus_is_exact_in_decimal_arithmetic(self, value):
        # Ακριβής έλεγχος χωρίς κινητή υποδιαστολή: PIR + |PIR|/10, δηλαδή ×1,1 (PIR ≥ 0) ή ×0,9.
        won = Decimal(repr(fantasy_score(value, True)))
        assert won == Decimal(value) + abs(Decimal(value)) / 10
        assert won == Decimal(value) * (Decimal("1.1") if value >= 0 else Decimal("0.9"))

    def test_monotonic_in_pir(self):
        # Με τον νέο κανόνα η συνάρτηση παραμένει γνήσια αύξουσα και σε νίκη (κλίση 0,9 και 1,1).
        for won in (True, False):
            scores = [fantasy_score(value, won) for value in range(-100, 151)]
            assert all(a < b for a, b in zip(scores, scores[1:], strict=False))

    def test_win_never_hurts(self):
        # Το μπόνους είναι πάντα θετικό: η νίκη βελτιώνει το score για κάθε PIR ≠ 0.
        for value in range(1, 151):
            assert fantasy_score(value, True) > fantasy_score(value, False)
        for value in range(-100, 0):
            assert fantasy_score(value, True) > fantasy_score(value, False)
        assert fantasy_score(0, True) == fantasy_score(0, False) == 0.0

    def test_bonus_is_one_tenth_of_the_absolute_pir(self):
        for value in range(-100, 151):
            with_win = Decimal(repr(fantasy_score(value, True)))
            without_win = Decimal(repr(fantasy_score(value, False)))
            assert with_win - without_win == abs(Decimal(value)) / 10


class TestInputValidation:
    """Αυστηρός έλεγχος εισόδου: μια ελλιπής ή λάθος τιμή δεν πρέπει να δώσει σιωπηλά μπόνους."""

    @pytest.mark.parametrize(
        "won",
        ["False", "True", "", "no", float("nan"), None, 1, 0, 1.0, 0.0, np.int64(1), pd.NA, [True]],
        ids=lambda value: f"{type(value).__name__}-{value!r}",
    )
    def test_won_must_be_a_bool(self, won):
        with pytest.raises(TypeError, match="won must be a bool"):
            fantasy_score(10, won)

    @pytest.mark.parametrize("pir_value", [True, False, np.True_, np.False_])
    def test_pir_must_not_be_a_bool(self, pir_value):
        with pytest.raises(TypeError, match="not a bool"):
            fantasy_score(pir_value, True)

    @pytest.mark.parametrize("pir_value", [12.0, 12.5, float("nan"), None, "12", pd.NA])
    def test_pir_must_be_an_integer(self, pir_value):
        with pytest.raises(TypeError):
            fantasy_score(pir_value, True)

    def test_a_rejected_input_is_rejected_for_both_outcomes(self):
        with pytest.raises(TypeError):
            fantasy_score(12.5, False)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            fantasy_score(True, False)

    @staticmethod
    def pir_series() -> pd.Series:
        return pd.Series([12, 12, -4, -4, 0])

    @pytest.mark.parametrize(
        "won",
        [
            pd.Series([True, np.nan, True, False, True]),  # NaN: object ή float
            pd.Series([1.0, 0.0, 1.0, 0.0, 1.0]),
            pd.Series([1, 0, 1, 0, 1]),
            pd.Series(["True", "False", "True", "False", "True"]),
            pd.Series([True, None, True, False, True], dtype=object),
            pd.Series([True, False, True, False, 1], dtype=object),
            pd.Series([True, False, True, False, "True"], dtype=object),
            pd.Series([np.nan] * 5),
        ],
        ids=[
            "bool-with-nan",
            "float",
            "int",
            "str",
            "object-none",
            "object-int",
            "object-str",
            "nan",
        ],
    )
    def test_frame_rejects_a_non_bool_won(self, won):
        with pytest.raises(TypeError, match="won must"):
            fantasy_score_frame(self.pir_series(), won)

    def test_frame_rejects_missing_values_in_a_nullable_boolean(self):
        won = pd.Series([True, pd.NA, True, False, True], dtype="boolean")
        with pytest.raises(ValueError, match="missing"):
            fantasy_score_frame(self.pir_series(), won)

    def test_frame_does_not_turn_a_missing_result_into_a_win(self):
        # Το `won.astype(bool)` θα έκανε το NaN νίκη και θα έδινε σιωπηλά μπόνους 10%.
        won = pd.Series([True, np.nan, True, False, True])
        assert won.astype(bool).tolist() == [True, True, True, False, True]
        with pytest.raises(TypeError):
            fantasy_score_frame(self.pir_series(), won)

    @pytest.mark.parametrize("dtype", [bool, "boolean", object])
    def test_frame_accepts_every_bool_representation(self, dtype):
        won = pd.Series([True, False, True, False, True], dtype=dtype)
        result = fantasy_score_frame(self.pir_series(), won)
        assert result.tolist() == [13.2, 12.0, -3.6, -4.0, 0.0]

    def test_frame_accepts_numpy_bools_in_an_object_series(self):
        won = pd.Series([np.True_, np.False_, np.True_, np.False_, np.True_], dtype=object)
        assert fantasy_score_frame(self.pir_series(), won).tolist() == [13.2, 12.0, -3.6, -4.0, 0.0]

    def test_frame_rejects_a_bool_pir(self):
        with pytest.raises(TypeError, match="not bool"):
            fantasy_score_frame(pd.Series([True, False]), pd.Series([True, True]))

    @pytest.mark.parametrize(
        "pir_values",
        [
            pd.Series([12.0, 12.0, -4.0, -4.0, 0.0]),
            pd.Series([12.5, 12.0, -4.0, -4.0, 0.0]),
            pd.Series([12, 12, -4, -4, np.nan]),
            pd.Series(["12", "12", "-4", "-4", "0"]),
            pd.Series([12, 12, -4, -4, None], dtype=object),
        ],
        ids=["float", "fractional", "nan", "str", "object-none"],
    )
    def test_frame_rejects_a_non_integer_pir(self, pir_values):
        with pytest.raises(TypeError, match="pir must be an integer"):
            fantasy_score_frame(pir_values, pd.Series([True] * 5))

    def test_frame_rejects_missing_values_in_a_nullable_integer_pir(self):
        pir_values = pd.Series([12, 12, -4, -4, pd.NA], dtype="Int64")
        with pytest.raises(ValueError, match="missing"):
            fantasy_score_frame(pir_values, pd.Series([True] * 5))

    def test_frame_requires_the_same_index(self):
        won = pd.Series([True] * 5, index=range(1, 6))
        with pytest.raises(ValueError, match="same index"):
            fantasy_score_frame(self.pir_series(), won)

    def test_frame_keeps_the_index(self):
        index = pd.Index([10, 20, 30, 40, 50], name="row")
        pir_values = pd.Series([12, 12, -4, -4, 0], index=index)
        won = pd.Series([True, False, True, False, True], index=index)
        result = fantasy_score_frame(pir_values, won)
        assert result.index.equals(index)
        assert result.tolist() == [13.2, 12.0, -3.6, -4.0, 0.0]

    @pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "Int64"])
    def test_frame_accepts_every_integer_dtype_without_overflow(self, dtype):
        # Σε int8 το 53 * 10 θα ξεχείλιζε (max 127): το frame αναβαθμίζει σε int64 πριν υπολογίσει.
        values = pd.Series([53, 12, 0, 1], dtype=dtype)
        result = fantasy_score_frame(values, pd.Series([True] * 4))
        assert result.tolist() == [58.3, 13.2, 0.0, 1.1]
        assert result.dtype == float

    def test_frame_negative_values_in_small_integer_dtypes(self):
        values = pd.Series([-13, -4, -1], dtype="int8")
        result = fantasy_score_frame(values, pd.Series([True] * 3))
        assert result.tolist() == [-11.7, -3.6, -0.9]

    def test_frame_empty_input(self):
        result = fantasy_score_frame(pd.Series([], dtype=float), pd.Series([], dtype=bool))
        assert result.empty
        assert result.dtype == float
        empty_objects = pd.Series([], dtype=object)
        assert fantasy_score_frame(empty_objects, empty_objects).empty


class TestVectorized:
    @staticmethod
    def random_stats(rows: int = 400) -> pd.DataFrame:
        generator = np.random.default_rng(12345)
        columns = {name: generator.integers(0, 12, rows) for name in CLEAN_COLUMNS.values()}
        frame = pd.DataFrame(columns)
        # Οι προσπάθειες δεν είναι ποτέ λιγότερες από τα εύστοχα.
        for made, attempted in (
            ("fg2_made", "fg2_attempted"),
            ("fg3_made", "fg3_attempted"),
            ("ft_made", "ft_attempted"),
        ):
            frame[attempted] = frame[made] + generator.integers(0, 6, rows)
        return frame

    def test_pir_frame_equals_scalar(self):
        frame = self.random_stats()
        expected = [pir_from_row(row) for _, row in frame.iterrows()]
        assert pir_frame(frame).tolist() == expected

    def test_pir_frame_with_raw_columns(self):
        frame = self.random_stats()
        raw = frame.rename(
            columns={clean: RAW_COLUMNS[arg] for arg, clean in CLEAN_COLUMNS.items()}
        )
        assert pir_frame(raw, RAW_COLUMNS).tolist() == pir_frame(frame).tolist()

    def test_fantasy_score_frame_equals_scalar(self):
        values = pd.Series(range(-100, 151))
        for won_value in (True, False):
            won = pd.Series([won_value] * len(values))
            vectorized = fantasy_score_frame(values, won)
            scalar = [fantasy_score(int(v), won_value) for v in values]
            assert vectorized.tolist() == scalar
            assert vectorized.dtype == float

    def test_fantasy_score_frame_mixed_outcomes(self):
        values = pd.Series([12, 12, -4, -4, 0])
        won = pd.Series([True, False, True, False, True])
        assert fantasy_score_frame(values, won).tolist() == [13.2, 12.0, -3.6, -4.0, 0.0]

    def test_fantasy_score_frame_equals_scalar_on_random_rows(self):
        generator = np.random.default_rng(2026)
        values = pd.Series(generator.integers(-13, 54, 20_000))
        won = pd.Series(generator.random(20_000) < 0.5)
        vectorized = fantasy_score_frame(values, won)
        scalar = [fantasy_score(int(v), bool(w)) for v, w in zip(values, won, strict=True)]
        assert vectorized.tolist() == scalar  # ακριβής ισότητα, όχι ανοχή
        assert all(re.fullmatch(r"-?\d+(\.\d)?", repr(x)) for x in vectorized.tolist())


class TestRealBoxscoreFixture:
    """Πραγματικό boxscore (7 αγώνες): το PIR από τα στατιστικά ισούται με το Valuation."""

    def test_pir_equals_valuation_in_every_row(self, raw_boxscores):
        computed = pir_frame(raw_boxscores, RAW_COLUMNS)
        mismatches = raw_boxscores[computed != raw_boxscores["Valuation"]]
        assert mismatches.empty, mismatches[["Season", "Gamecode", "Player_ID", "Valuation"]]
        assert len(raw_boxscores) == 195  # 167 παίκτες + 14 γραμμές Team + 14 γραμμές Total

    def test_total_is_sum_of_players_plus_team(self, raw_boxscores):
        columns = [
            "Points",
            "FieldGoalsMade2",
            "FieldGoalsAttempted2",
            "FieldGoalsMade3",
            "FieldGoalsAttempted3",
            "FreeThrowsMade",
            "FreeThrowsAttempted",
            "OffensiveRebounds",
            "DefensiveRebounds",
            "TotalRebounds",
            "Assistances",
            "Steals",
            "Turnovers",
            "BlocksFavour",
            "BlocksAgainst",
            "FoulsCommited",
            "FoulsReceived",
            "Valuation",
        ]
        ids = raw_boxscores["Player_ID"].str.strip()
        players = raw_boxscores[~ids.isin(["Team", "Total"])]
        team = raw_boxscores[ids == "Team"]
        total = raw_boxscores[ids == "Total"]
        key = ["Season", "Gamecode", "Team"]
        expected = players.groupby(key)[columns].sum() + team.set_index(key)[columns]
        actual = total.set_index(key)[columns]
        pd.testing.assert_frame_equal(actual.sort_index(), expected.sort_index(), check_dtype=False)
        assert len(actual) == 14

    def test_fixture_contains_the_documented_special_cases(self, raw_boxscores):
        ids = raw_boxscores["Player_ID"].str.strip()
        players = raw_boxscores[~ids.isin(["Team", "Total"])]
        assert (players["Minutes"] == "DNP").sum() == 11
        assert (players["Valuation"] < 0).sum() == 14
        old_style = ids[~ids.isin(["Team", "Total"]) & ~ids.str.match(r"^P\d{6}$")]
        assert {"PLRU", "PAAX"} <= set(old_style)
        # Τα ακατέργαστα IDs των παικτών έχουν trailing spaces (10 χαρακτήρες)
        assert (raw_boxscores.loc[players.index, "Player_ID"].str.len() == 10).all()

    def test_documented_examples_exist_in_the_fixture(self, raw_boxscores):
        game = raw_boxscores[(raw_boxscores["Season"] == 2025) & (raw_boxscores["Gamecode"] == 1)]
        by_id = game.assign(pid=game["Player_ID"].str.strip()).set_index("pid")
        assert by_id.loc["P007200", "Valuation"] == 12  # Larkin
        assert by_id.loc["P006835", "Valuation"] == 22  # Hoard
        assert by_id.loc["P011201", "Valuation"] == -4  # Hazer
        assert by_id.loc["P006590", "Minutes"] == "DNP"  # Beaubois
        assert by_id.loc["P006590", "Valuation"] == 0
