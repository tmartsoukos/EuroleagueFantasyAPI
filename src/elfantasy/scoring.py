"""Υπολογισμός του PIR και του fantasy score ενός παίκτη (ο τύπος ορίζεται στο FANTASY_RULES.md).

Οι συναρτήσεις είναι καθαρές και δεν εξαρτώνται από pandas. Οι εκδόσεις `*_frame` δουλεύουν
σε στήλες DataFrame χρησιμοποιώντας μόνο αριθμητικούς τελεστές (vectorized), χωρίς import
του pandas σε αυτό το αρχείο.

Τύπος (FANTASY_RULES.md, ενότητα 3):

    PIR = πόντοι + ριμπάουντ + ασίστ + κλεψίματα + μπλοκ που κάνει + φάουλ που δέχεται
          - αστοχημένα δίποντα - αστοχημένα τρίποντα - αστοχημένες βολές
          - λάθη - μπλοκ που δέχεται - φάουλ που κάνει
    fantasy = PIR × 1,1 αν η ομάδα του παίκτη κερδίσει, αλλιώς PIR
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import Any

# Μπόνους νίκης: ×1,1 (FANTASY_RULES.md, ενότητα 3.2), σε ακέραια αριθμητική ως 11/10.
# Το `pir * 1.1` δίνει θόρυβο κινητής υποδιαστολής (12 * 1.1 = 13.200000000000001), ενώ το
# `pir * 11 / 10` διαιρεί δύο ακέραιους και δίνει το πλησιέστερο double στην ακριβή τιμή,
# δηλαδή πάντα το «καθαρό» δεκαδικό με το πολύ ένα ψηφίο (FANTASY_RULES.md, ενότητα 3.3).
WIN_BONUS_NUM = 11
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
    """Fantasy score παίκτη: `PIR × 1,1` αν η ομάδα του κέρδισε, αλλιώς το PIR.

    Ο υπολογισμός γίνεται με ακέραια αριθμητική (`pir * 11 / 10`), οπότε το αποτέλεσμα έχει
    πάντα το πολύ ένα δεκαδικό και δεν έχει θόρυβο κινητής υποδιαστολής. Το αρνητικό PIR
    σε νίκη πολλαπλασιάζεται κατά γράμμα (−4 → −4,4), όπως ορίζει το FANTASY_RULES.md, ενότητα
    3.4. Παίκτης που δεν αγωνίστηκε (DNP) έχει PIR 0 και άρα fantasy score 0.

    Το `pir` πρέπει να είναι ακέραιο (δέχονται και οι ακέραιοι της numpy). Επιστρέφει πάντα float.
    """
    pir_value = operator.index(pir)
    if won:
        return pir_value * WIN_BONUS_NUM / WIN_BONUS_DEN
    return float(pir_value)


def fantasy_score_frame(pir_values: Any, won: Any) -> Any:
    """Vectorized έκδοση του `fantasy_score()`.

    `pir_values`: Series με ακέραιο PIR, `won`: Series με bool (ίδιο index). Επιστρέφει Series
    τύπου float, με τις ίδιες τιμές που θα έδινε το `fantasy_score()` γραμμή προς γραμμή.
    """
    with_bonus = pir_values * WIN_BONUS_NUM / WIN_BONUS_DEN
    return with_bonus.where(won.astype(bool), pir_values.astype(float))
