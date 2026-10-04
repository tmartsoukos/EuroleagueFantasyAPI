"""Υπολογισμός του PIR και του fantasy score ενός παίκτη (ο τύπος ορίζεται στο FANTASY_RULES.md).

Οι συναρτήσεις είναι καθαρές και δεν εξαρτώνται από pandas (το numpy χρειάζεται μόνο για να
αναγνωρίζεται ο `np.bool_`). Οι εκδόσεις `*_frame` δουλεύουν σε στήλες DataFrame χρησιμοποιώντας
μόνο αριθμητικούς τελεστές και μεθόδους των Series (vectorized), χωρίς import του pandas σε αυτό
το αρχείο.

Τύπος (FANTASY_RULES.md, ενότητα 3):

    PIR = πόντοι + ριμπάουντ + ασίστ + κλεψίματα + μπλοκ που κάνει + φάουλ που δέχεται
          - αστοχημένα δίποντα - αστοχημένα τρίποντα - αστοχημένες βολές
          - λάθη - μπλοκ που δέχεται - φάουλ που κάνει
    fantasy = PIR + |PIR| / 10 αν η ομάδα του παίκτη κερδίσει, αλλιώς PIR

Το μπόνους νίκης είναι το 10% του ΑΠΟΛΥΤΟΥ PIR και προστίθεται πάντα θετικό: PIR 12 → 13,2 (×1,1)
και PIR −4 → −3,6 (×0,9, όχι −4,4). Ο κανόνας για το αρνητικό PIR δεν προκύπτει από κείμενο
κανονισμού αλλά από τα επίσημα σύνολα βαθμών των νικητών των αγωνιστικών του Fantasy Challenge
(FANTASY_RULES.md, ενότητες 3.4 και 9).
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import Any

import numpy as np

# Μπόνους νίκης: το 1/10 του |PIR| (FANTASY_RULES.md, ενότητα 3.2), προστίθεται πάντα θετικό.
# Υπολογίζεται με ακέραια αριθμητική ως `(pir * 10 + |pir|) / 10`, δηλαδή `pir * 11 / 10` για
# PIR ≥ 0 και `pir * 9 / 10` για PIR < 0. Το `pir * 1.1` δίνει θόρυβο κινητής υποδιαστολής
# (12 * 1.1 = 13.200000000000001), ενώ η διαίρεση δύο ακεραίων δίνει το πλησιέστερο double στην
# ακριβή τιμή, δηλαδή πάντα το «καθαρό» δεκαδικό με το πολύ ένα ψηφίο (FANTASY_RULES.md, 3.3).
WIN_BONUS_NUM = 1
WIN_BONUS_DEN = 10

# Αντιστοίχιση ορισμάτων του `pir()` σε στήλες των πινάκων του project (db/models.py).
CLEAN_COLUMNS: dict[str, str] = {
    "points": "points",
    "rebounds": "total_reb",
    "assists": "assists",
    "steals": "steals",
    "blocks_favour": "blocks_favour",
    "fouls_received": "fouls_received",
    "fg2_made": "fg2_made",
    "fg2_attempted": "fg2_attempted",
    "fg3_made": "fg3_made",
    "fg3_attempted": "fg3_attempted",
    "ft_made": "ft_made",
    "ft_attempted": "ft_attempted",
    "turnovers": "turnovers",
    "blocks_against": "blocks_against",
    "fouls_committed": "fouls_committed",
}

# Η ίδια αντιστοίχιση για τις ακατέργαστες στήλες του boxscore του euroleague_api
# (FANTASY_RULES.md, ενότητα 4). Οι ορθογραφικές ιδιαιτερότητες είναι του πακέτου.
RAW_COLUMNS: dict[str, str] = {
    "points": "Points",
    "rebounds": "TotalRebounds",
    "assists": "Assistances",
    "steals": "Steals",
    "blocks_favour": "BlocksFavour",
    "fouls_received": "FoulsReceived",
    "fg2_made": "FieldGoalsMade2",
    "fg2_attempted": "FieldGoalsAttempted2",
    "fg3_made": "FieldGoalsMade3",
    "fg3_attempted": "FieldGoalsAttempted3",
    "ft_made": "FreeThrowsMade",
    "ft_attempted": "FreeThrowsAttempted",
    "turnovers": "Turnovers",
    "blocks_against": "BlocksAgainst",
    "fouls_committed": "FoulsCommited",
}


def pir(
    *,
    points: int,
    rebounds: int,
    assists: int,
    steals: int,
    blocks_favour: int,
    fouls_received: int,
    fg2_made: int,
    fg2_attempted: int,
    fg3_made: int,
    fg3_attempted: int,
    ft_made: int,
    ft_attempted: int,
    turnovers: int,
    blocks_against: int,
    fouls_committed: int,
) -> int:
    """Υπολογίζει το PIR από τα ακατέργαστα στατιστικά ενός παίκτη σε έναν αγώνα.

    Τα ορίσματα είναι μόνο ονομαστικά (keyword-only), ώστε να μην μπερδεύονται 15 ακέραιοι.
    Τα αστοχημένα σουτ εντός πεδιάς είναι τα αστοχημένα δίποντα και τρίποντα μαζί, ενώ οι
    αστοχημένες βολές μετρούν χωριστά (FANTASY_RULES.md, ενότητα 3.1).

    Επιστρέφει ακέραιο. Αν τα στατιστικά δεν δίνουν ακέραιο (π.χ. δεκαδικά ορίσματα),
    σηκώνει `ValueError` αντί να στρογγυλοποιήσει σιωπηλά.
    """
    positive = points + rebounds + assists + steals + blocks_favour + fouls_received
    missed_field_goals = (fg2_attempted - fg2_made) + (fg3_attempted - fg3_made)
    missed_free_throws = ft_attempted - ft_made
    negative = (
        missed_field_goals + missed_free_throws + turnovers + blocks_against + fouls_committed
    )
    total = positive - negative
    result = int(total)
    if result != total:
        raise ValueError(f"PIR must be an integer, got {total!r}")
    return result


def pir_from_row(row: Mapping[str, Any], columns: Mapping[str, str] = CLEAN_COLUMNS) -> int:
    """Υπολογίζει το PIR από μια γραμμή (dict ή pandas Series) με τις στήλες του `columns`.

    Προεπιλογή είναι τα ονόματα στηλών του project (`CLEAN_COLUMNS`). Για ακατέργαστο
    boxscore περνάμε `columns=RAW_COLUMNS`.
    """
    return pir(**{argument: row[column] for argument, column in columns.items()})


def pir_frame(df: Any, columns: Mapping[str, str] = CLEAN_COLUMNS) -> Any:
    """Vectorized έκδοση του `pir()`: επιστρέφει Series με το PIR κάθε γραμμής του `df`.

    Χρησιμοποιεί μόνο αριθμητικούς τελεστές πάνω στις στήλες, άρα δίνει ακριβώς τα ίδια
    αποτελέσματα με το `pir()` (ακέραια αριθμητική, χωρίς στρογγυλοποιήσεις).
    """
    c = columns
    positive = (
        df[c["points"]]
        + df[c["rebounds"]]
        + df[c["assists"]]
        + df[c["steals"]]
        + df[c["blocks_favour"]]
        + df[c["fouls_received"]]
    )
    missed_field_goals = (df[c["fg2_attempted"]] - df[c["fg2_made"]]) + (
        df[c["fg3_attempted"]] - df[c["fg3_made"]]
    )
    missed_free_throws = df[c["ft_attempted"]] - df[c["ft_made"]]
    negative = (
        missed_field_goals
        + missed_free_throws
        + df[c["turnovers"]]
        + df[c["blocks_against"]]
        + df[c["fouls_committed"]]
    )
    return positive - negative


def fantasy_score(pir: int, won: bool) -> float:
    """Fantasy score παίκτη: `PIR + |PIR| / 10` αν η ομάδα του κέρδισε, αλλιώς το PIR.

    Το μπόνους νίκης είναι το 10% του απόλυτου PIR και προστίθεται πάντα θετικό: PIR 12 → 13,2
    και PIR −4 → −3,6 (FANTASY_RULES.md, ενότητες 3.2 και 3.4). Η νίκη δεν μειώνει λοιπόν ποτέ το
    score. Ο υπολογισμός γίνεται με ακέραια αριθμητική (`pir * 11 / 10` για PIR ≥ 0, `pir * 9 / 10`
    για PIR < 0), οπότε το αποτέλεσμα έχει πάντα το πολύ ένα δεκαδικό και δεν έχει θόρυβο κινητής
    υποδιαστολής. Παίκτης που δεν αγωνίστηκε (DNP) έχει PIR 0 και άρα fantasy score 0.

    Η είσοδος ελέγχεται αυστηρά, ώστε μια ελλιπής ή λάθος τιμή να μη δώσει σιωπηλά μπόνους:

    - το `pir` πρέπει να είναι ακέραιο (δέχονται και οι ακέραιοι της numpy) και όχι bool
      (`TypeError` για float, NaN, None, bool),
    - το `won` πρέπει να είναι bool ή `np.bool_` (`TypeError` για str, NaN, None, ακεραίους).

    Επιστρέφει πάντα float.
    """
    if isinstance(pir, bool | np.bool_):
        raise TypeError(f"pir must be an integer, not a bool, got {pir!r}")
    if not isinstance(won, bool | np.bool_):
        raise TypeError(f"won must be a bool, got {won!r} ({type(won).__name__})")
    pir_value = operator.index(pir)
    if won:
        return (pir_value * WIN_BONUS_DEN + abs(pir_value) * WIN_BONUS_NUM) / WIN_BONUS_DEN
    return float(pir_value)


def _checked_pir_values(pir_values: Any) -> Any:
    """Επικυρώνει το PIR ενός frame και το επιστρέφει ως Series int64 (χωρίς overflow)."""
    kind = pir_values.dtype.kind
    if kind == "b":
        raise TypeError("pir must be an integer Series, not bool")
    if kind not in "iu":
        raise TypeError(f"pir must be an integer Series, got dtype {pir_values.dtype}")
    if pir_values.isna().any():
        raise ValueError("pir contains missing values")
    # Το upcast αποτρέπει overflow μικρών ακεραίων τύπων (π.χ. int8) στον πολλαπλασιασμό με το 10.
    return pir_values.astype("int64")


def _checked_won_mask(won: Any) -> Any:
    """Επικυρώνει το `won` ενός frame και το επιστρέφει ως Series bool (χωρίς σιωπηλό astype)."""
    kind = won.dtype.kind
    if kind == "b":
        if won.isna().any():  # nullable boolean με ελλιπείς τιμές
            raise ValueError("won contains missing values")
        return won.astype(bool)
    if kind == "O":
        is_bool = won.map(lambda value: isinstance(value, bool | np.bool_)).astype(bool)
        if not is_bool.all():
            first_bad = won[~is_bool].iloc[0]
            raise TypeError(
                f"won must contain only bool values, found {int((~is_bool).sum())} other "
                f"value(s), e.g. {first_bad!r}"
            )
        return won.astype(bool)
    raise TypeError(f"won must be a bool Series, got dtype {won.dtype}")


def fantasy_score_frame(pir_values: Any, won: Any) -> Any:
    """Vectorized έκδοση του `fantasy_score()`.

    `pir_values`: Series με ακέραιο PIR, `won`: Series με bool (ίδιο index). Επιστρέφει Series
    τύπου float, με τις ίδιες τιμές που θα έδινε το `fantasy_score()` γραμμή προς γραμμή.

    Η είσοδος ελέγχεται με τους ίδιους αυστηρούς κανόνες με το `fantasy_score()`: το PIR πρέπει να
    είναι ακέραιου τύπου (όχι bool ή float, χωρίς ελλιπείς τιμές) και το `won` τύπου bool
    (`TypeError` για NaN, str, ακεραίους, None, και `ValueError` για ελλιπείς τιμές nullable
    boolean). Δεν γίνεται σιωπηλό `astype(bool)`, που θα μετέτρεπε μια ελλιπή τιμή σε νίκη.
    Τα δύο Series πρέπει να έχουν το ίδιο index (`ValueError` αλλιώς). Κενή είσοδος δίνει κενό
    Series float.
    """
    if len(pir_values) == 0 and len(won) == 0:
        return pir_values.astype(float)
    values = _checked_pir_values(pir_values)
    won_mask = _checked_won_mask(won)
    if not values.index.equals(won_mask.index):
        raise ValueError("pir and won must have the same index")
    with_bonus = (values * WIN_BONUS_DEN + values.abs() * WIN_BONUS_NUM) / WIN_BONUS_DEN
    return with_bonus.where(won_mask, values.astype(float))
