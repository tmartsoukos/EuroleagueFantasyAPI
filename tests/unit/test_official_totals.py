"""Επίσημα σύνολα βαθμών του EuroLeague Fantasy Challenge: regression test του τύπου.

Κάθε αγωνιστική το EuroLeague δημοσιεύει στο euroleaguebasketball.net το άρθρο «EuroLeague
Fantasy Challenge Round N winner» με τη νικήτρια ομάδα της αγωνιστικής: captain, 4 παίκτες της
πεντάδας, έκτο παίκτη, 4 παίκτες του πάγκου, την ομάδα του προπονητή και το σύνολο βαθμών
(π.χ. 239,75). Το `tests/fixtures/official_rounds.csv` έχει 29 τέτοιες αγωνιστικές: 23 της σεζόν
2025-26 (`season` 2025 στη βάση) και 6 της 2024-25 (`season` 2024).

Πηγή: τα άρθρα στο https://www.euroleaguebasketball.net/en/euroleague/news/ (η στήλη `article`
του CSV έχει το τελευταίο τμήμα του URL· τα slugs δεν έχουν ενιαία μορφή). Τα ονόματα, οι ομάδες
και τα σύνολα μεταγράφηκαν με το χέρι και ελέγχθηκαν ξανά και για τις 29 αγωνιστικές σε πραγματικό
browser στις 2026-10-04 (σειρά ονομάτων, ομάδες, σύνολο, προπονητής· δύο επώνυμα γράφονται αλλιώς
στα άρθρα: «Farried» για τον Faried και «Anglola» για τον Angola). Από τα άρθρα χρησιμοποιούνται
μόνο γεγονότα (ονόματα, ομάδες, σύνολο), όχι το κείμενό τους. Το PIR και το αποτέλεσμα (`won`)
κάθε παίκτη προέρχονται από τη βάση του project (`data/elfantasy.db`). Παίκτης της σύνθεσης που
δεν αγωνίστηκε (`status` dnp ή no_row) έχει PIR 0.

Το σύνολο μιας αγωνιστικής είναι το άθροισμα των `fantasy_score` των 10 παικτών με βάρη: captain
×2 (2025-26) ή ×1,5 (2024-25, όπως γράφει το επίσημο άρθρο κανόνων), πεντάδα και έκτος ×1, πάγκος
×0,5, συν τους πόντους του προπονητή. **Οι πόντοι προπονητή είναι υπόθεση**: υπολογίζονται με τον
πίνακα του fan wiki (νίκη με διαφορά 1–10: +10, 11–20: +20, άνω των 20: +25) και η υπόθεση
επιβεβαιώνεται μόνο για νίκες, γιατί και οι 29 προπονητές είναι της ομάδας που κέρδισε (το μέρος
του πίνακα για ήττες δεν ελέγχεται).

Αποτέλεσμα: ο τύπος `PIR + |PIR|/10` σε νίκη, αλλιώς PIR, αναπαράγει ακριβώς 24 από τις 29
αγωνιστικές (20 από 23 και 4 από 6). Οι υπόλοιπες 5 (`UNEXPLAINED`) δεν εξηγούνται και
καταγράφονται εδώ ρητά με το υπόλοιπό τους, χωρίς να υποστηρίζεται κάποια αιτία. Ο παλιός κανόνας
`PIR × 1,1` σε νίκη αποτυγχάνει επιπλέον ακριβώς στις αγωνιστικές που έχουν παίκτη πάγκου με
αρνητικό PIR σε νίκη (FANTASY_RULES.md, ενότητες 3.4 και 9).
"""

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest

from elfantasy.scoring import fantasy_score

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "official_rounds.csv"

# Πολλαπλασιαστής captain ανά σεζόν της βάσης: 2025-26 ×2 και 2024-25 ×1,5. Το fan wiki γράφει ×2
# και για το 2024-25, αλλά με ×2 κανένα επίσημο σύνολο της σεζόν αυτής δεν ταιριάζει.
CAPTAIN_MULTIPLIER = {2025: Fraction(2), 2024: Fraction(3, 2)}
ROLE_WEIGHT = {"S": Fraction(1), "6": Fraction(1), "B": Fraction(1, 2)}

# Αγωνιστικές που ο τύπος ΔΕΝ αναπαράγει: υπόλοιπο = επίσημο σύνολο − υπολογισμένο σύνολο. Η αιτία
# δεν έχει εξακριβωθεί. Το test αποτυγχάνει αν αλλάξει το υπόλοιπο μιας από αυτές ή αν αποτύχει
# οποιαδήποτε άλλη αγωνιστική.
UNEXPLAINED = {
    (2025, 1): Fraction("8.80"),
    (2025, 3): Fraction("-2.20"),
    (2025, 11): Fraction("5.00"),
    (2024, 2): Fraction("0.50"),
    (2024, 4): Fraction("-1.00"),
}

# Αγωνιστικές όπου ο παλιός κανόνας `PIR × 1,1` δίνει διαφορετικό σύνολο από τον τρέχοντα, ενώ ο
# τρέχων ταιριάζει: όλες έχουν παίκτη πάγκου με αρνητικό PIR σε νίκη. Η (2025, 11) έχει επίσης
# τέτοιον παίκτη, αλλά είναι από τις μη εξηγούμενες (το υπόλοιπό της μικραίνει από 5,20 σε 5,00).
OLD_RULE_FAILS_WHERE_NEW_MATCHES = {
    (2025, 2),
    (2025, 5),
    (2025, 9),
    (2025, 14),
    (2024, 10),
    (2024, 29),
}


@dataclass(frozen=True)
class Member:
    role: str  # C captain, S πεντάδα, 6 έκτος, B πάγκος
    player: str
    team_code: str
    pir: int
    won: bool
    status: str  # played, dnp (γραμμή DNP στη βάση) ή no_row (καμία γραμμή στη βάση)


@dataclass(frozen=True)
class OfficialRound:
    season: int
    number: int
    members: tuple[Member, ...]
    official_total: Fraction
    coach_team: str
    coach_margin: int
    coach_points: int
    article: str

    @property
    def key(self) -> tuple[int, int]:
        return (self.season, self.number)


def load_rounds() -> dict[tuple[int, int], OfficialRound]:
    with FIXTURE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["season"]), int(row["round"]))].append(row)
    rounds = {}
    for (season, number), group in grouped.items():
        first = group[0]
        for row in group:  # τα στοιχεία της αγωνιστικής επαναλαμβάνονται σε κάθε γραμμή
            for column in (
                "official_total",
                "coach_team",
                "coach_margin",
                "coach_points",
                "article",
            ):
                assert row[column] == first[column], (season, number, column)
        members = tuple(
            Member(
                role=row["role"],
                player=row["player"],
                team_code=row["team_code"],
                pir=int(row["pir"]),
                won=bool(int(row["won"])),
                status=row["status"],
            )
            for row in group
        )
        rounds[(season, number)] = OfficialRound(
            season=season,
            number=number,
            members=members,
            official_total=Fraction(first["official_total"]),
            coach_team=first["coach_team"],
            coach_margin=int(first["coach_margin"]),
            coach_points=int(first["coach_points"]),
            article=first["article"],
        )
    return rounds


ROUNDS = load_rounds()


# ---- Κανόνες που συγκρίνονται: κάθε ένας δίνει το fantasy score ενός παίκτη ως ακριβές κλάσμα ----


def rule_current(pir: int, won: bool) -> Fraction:
    """Ο τύπος του project: `scoring.fantasy_score` (PIR + |PIR|/10 σε νίκη)."""
    return Fraction(repr(fantasy_score(pir, won)))


def rule_old_times_1_1(pir: int, won: bool) -> Fraction:
    """Ο ΠΑΛΙΟΣ κανόνας του project: ×1,1 σε νίκη, και για αρνητικό PIR (−4 → −4,4)."""
    return Fraction(pir) * Fraction(11, 10) if won else Fraction(pir)


def rule_no_bonus_on_negative(pir: int, won: bool) -> Fraction:
    """Παραλλαγή: μπόνους ×1,1 μόνο σε θετικό PIR, κανένα μπόνους σε αρνητικό (−4 → −4)."""
    return Fraction(pir) * Fraction(11, 10) if won and pir > 0 else Fraction(pir)


def half_up(value: Fraction) -> Fraction:
    return Fraction(math.floor(value + Fraction(1, 2)))


def captain_weight(season: int, captain_multiplier: Fraction | None = None) -> Fraction:
    return CAPTAIN_MULTIPLIER[season] if captain_multiplier is None else captain_multiplier


def round_total(
    official_round: OfficialRound,
    rule,
    *,
    player_rounding=None,
    total_rounding=None,
    captain_multiplier: Fraction | None = None,
) -> Fraction:
    """Υπολογισμένο σύνολο της αγωνιστικής: παίκτες με βάρη ρόλου, συν πόντοι προπονητή."""
    total = Fraction(0)
    for member in official_round.members:
        weight = (
            captain_weight(official_round.season, captain_multiplier)
            if member.role == "C"
            else ROLE_WEIGHT[member.role]
        )
        score = rule(member.pir, member.won)
        if player_rounding is not None:
            score = player_rounding(score)
        total += weight * score
    if total_rounding is not None:
        total = total_rounding(total)
    return total + official_round.coach_points


def residual(official_round: OfficialRound, rule=rule_current, **options) -> Fraction:
    """Επίσημο σύνολο μείον υπολογισμένο (0 = ακριβής ταύτιση)."""
    return official_round.official_total - round_total(official_round, rule, **options)


def exact_matches(rule, season: int, **options) -> int:
    return sum(
        residual(item, rule, **options) == 0 for item in ROUNDS.values() if item.season == season
    )


def negative_pir_winners(official_round: OfficialRound) -> list[Member]:
    return [member for member in official_round.members if member.won and member.pir < 0]


MATCHING = sorted(key for key in ROUNDS if key not in UNEXPLAINED)


def round_id(key: tuple[int, int]) -> str:
    season, number = key
    return f"{season}-{str(season + 1)[2:]}-R{number}"


class TestFixture:
    def test_has_the_29_documented_rounds(self):
        assert len(ROUNDS) == 29
        assert sorted(item.number for item in ROUNDS.values() if item.season == 2025) == [
            *range(1, 4),
            *range(5, 15),
            *range(16, 21),
            *range(22, 25),
            26,
            27,
        ]
        assert sorted(item.number for item in ROUNDS.values() if item.season == 2024) == [
            2,
            4,
            10,
            17,
            29,
            30,
        ]

    def test_every_round_has_a_full_lineup(self):
        for item in ROUNDS.values():
            roles = sorted(member.role for member in item.members)
            assert roles == ["6", "B", "B", "B", "B", "C", "S", "S", "S", "S"], item.key

    def test_every_round_has_an_article_and_a_two_decimal_total(self):
        for item in ROUNDS.values():
            assert (
                item.article.startswith("euroleague-fantasy-challenge-")
                and "winner" in item.article
            )
            assert item.article.endswith("/")
            assert (item.official_total * 20).denominator == 1, item.key  # πολλαπλάσιο του 0,05
            assert 150 < item.official_total < 300, item.key

    def test_all_coaches_won_so_the_loss_part_of_the_coach_table_is_untested(self):
        assert all(item.coach_margin > 0 for item in ROUNDS.values())
        assert {item.coach_points for item in ROUNDS.values()} == {10, 20, 25}
        for item in ROUNDS.values():
            expected = 10 if item.coach_margin <= 10 else 20 if item.coach_margin <= 20 else 25
            assert item.coach_points == expected, item.key

    def test_members_without_a_game_count_zero(self):
        absent = [m for item in ROUNDS.values() for m in item.members if m.status != "played"]
        assert len(absent) == 10 and {m.status for m in absent} == {"dnp", "no_row"}
        assert all(m.pir == 0 for m in absent)


class TestCurrentRuleReproducesTheOfficialTotals:
    @pytest.mark.parametrize("key", MATCHING, ids=round_id)
    def test_the_computed_total_equals_the_official_one(self, key):
        item = ROUNDS[key]
        computed = round_total(item, rule_current)
        assert computed == item.official_total
        assert abs(float(computed) - float(item.official_total)) < 1e-9

    def test_20_of_23_rounds_of_2025_26_and_4_of_6_of_2024_25_match(self):
        assert exact_matches(rule_current, 2025) == 20
        assert exact_matches(rule_current, 2024) == 4
        assert len(MATCHING) == 24

    def test_the_failing_rounds_are_exactly_the_documented_known_exceptions(self):
        failing = {key: residual(item) for key, item in ROUNDS.items() if residual(item) != 0}
        assert failing == UNEXPLAINED

    def test_rounds_with_members_who_did_not_play_match_too(self):
        # Το DNP = 0 επιβεβαιώνεται: 8 αγωνιστικές με απόντες ταιριάζουν όλες ακριβώς.
        with_absent = {
            key for key, item in ROUNDS.items() if any(m.status != "played" for m in item.members)
        }
        assert with_absent == {
            (2025, 2),
            (2025, 16),
            (2025, 17),
            (2025, 18),
            (2025, 20),
            (2025, 22),
            (2025, 27),
            (2024, 17),
        }
        assert with_absent.isdisjoint(UNEXPLAINED)

    def test_the_old_rule_leaves_the_same_residual_in_the_known_exceptions_except_one(self):
        # Στις 5 μη εξηγούμενες το υπόλοιπο είναι ίδιο και με τον παλιό κανόνα, εκτός από την
        # (2025, 11).
        for key in UNEXPLAINED:
            old = residual(ROUNDS[key], rule_old_times_1_1)
            if key == (2025, 11):
                assert (old, UNEXPLAINED[key]) == (Fraction("5.20"), Fraction("5.00"))
            else:
                assert old == UNEXPLAINED[key], key


class TestOldRuleTimesOnePointOne:
    def test_the_old_rule_matches_only_18_rounds(self):
        assert exact_matches(rule_old_times_1_1, 2025) == 16
        assert exact_matches(rule_old_times_1_1, 2024) == 2

    def test_it_fails_exactly_where_a_bench_player_has_a_negative_pir_in_a_win(self):
        old_fails = {key for key in MATCHING if residual(ROUNDS[key], rule_old_times_1_1) != 0}
        assert old_fails == OLD_RULE_FAILS_WHERE_NEW_MATCHES
        for key in MATCHING:
            has_negative_winner = bool(negative_pir_winners(ROUNDS[key]))
            assert (key in old_fails) == has_negative_winner, key

    def test_every_negative_pir_winner_in_the_lineups_is_on_the_bench(self):
        cases = [
            (item.key, member) for item in ROUNDS.values() for member in negative_pir_winners(item)
        ]
        assert len(cases) == 7  # 7 ανεξάρτητες περιπτώσεις σε 7 αγωνιστικές
        assert {member.role for _, member in cases} == {"B"}
        assert {key for key, _ in cases} == OLD_RULE_FAILS_WHERE_NEW_MATCHES | {(2025, 11)}

    def test_the_old_rule_error_is_one_fifth_of_the_bench_weighted_negative_pir(self):
        # Παλιός κανόνας: −4 → −4,4 αντί για −3,6, δηλαδή 0,2 × |PIR| χαμηλότερα (×0,5 στον πάγκο).
        for key in OLD_RULE_FAILS_WHERE_NEW_MATCHES:
            item = ROUNDS[key]
            expected = sum(
                Fraction(1, 5) * abs(member.pir) * ROLE_WEIGHT[member.role]
                for member in negative_pir_winners(item)
            )
            assert residual(item, rule_old_times_1_1) == expected, key

    def test_the_current_rule_is_never_worse_than_the_old_one(self):
        for item in ROUNDS.values():
            assert abs(residual(item, rule_current)) <= abs(residual(item, rule_old_times_1_1))

    def test_no_bonus_on_negative_pir_is_rejected_too(self):
        # Μισό υπόλοιπο (όχι μηδέν): −4 → −4 αντί για −3,6 είναι 0,4 × βάρος πάγκου 0,5 = 0,2
        # (ο Jackson της 2025-26 R2).
        assert exact_matches(rule_no_bonus_on_negative, 2025) == 16
        assert exact_matches(rule_no_bonus_on_negative, 2024) == 2
        for key in OLD_RULE_FAILS_WHERE_NEW_MATCHES:
            item = ROUNDS[key]
            expected = sum(
                Fraction(1, 10) * abs(member.pir) * ROLE_WEIGHT[member.role]
                for member in negative_pir_winners(item)
            )
            assert residual(item, rule_no_bonus_on_negative) == expected, key


class TestRounding:
    """Δεν υπάρχει στρογγυλοποίηση σε ακέραιο: τα σύνολα έχουν δύο δεκαδικά (βήμα 0,05)."""

    def test_some_official_totals_are_not_integers(self):
        assert sum(item.official_total.denominator != 1 for item in ROUNDS.values()) >= 25

    def test_rounding_every_player_score_to_an_integer_does_not_reproduce_the_totals(self):
        assert exact_matches(rule_current, 2025, player_rounding=half_up) == 1
        assert exact_matches(rule_current, 2024, player_rounding=half_up) == 0

    def test_rounding_the_total_to_an_integer_does_not_reproduce_the_totals(self):
        assert exact_matches(rule_current, 2025, total_rounding=half_up) == 0
        assert exact_matches(rule_current, 2024, total_rounding=half_up) == 1


class TestCaptainMultiplier:
    def test_2024_25_uses_1_5_and_not_2(self):
        # Με ×2 σε όλες τις αγωνιστικές του 2024-25 το υπολογισμένο σύνολο ξεπερνά το επίσημο
        # κατά 13 έως 21 βαθμούς.
        residuals = [
            residual(item, captain_multiplier=Fraction(2))
            for item in ROUNDS.values()
            if item.season == 2024
        ]
        assert all(Fraction(-21) <= value <= Fraction(-13) for value in residuals), residuals

    def test_2025_26_uses_2_and_not_1_5(self):
        assert exact_matches(rule_current, 2025, captain_multiplier=Fraction(3, 2)) == 0
