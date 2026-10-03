"""Επαλήθευση του dataset που έχει φορτωθεί στη βάση (σε ΟΛΟΚΛΗΡΟ το dataset, όχι σε δείγμα).

    python -m elfantasy.ingest.verify [--db URL] [--data-dir data] [--accept-missing 2018/21,...]

Ελέγχοι:
- Ποσοστό γραμμών όπου το PIR που υπολογίζεται από τα στατιστικά (scoring.py) ισούται με τη στήλη
  `Valuation` του API. Οι αποκλίσεις γράφονται στο `data/reports/pir_mismatches.csv`.
- Συνέπεια του αποθηκευμένου `pir` και του `fantasy_score` (PIR × 1,1 σε νίκη) με τον υπολογισμό.
- Πλήθος αγώνων, παικτών και γραμμών ανά σεζόν σε σχέση με τα αναμενόμενα, και αγώνες χωρίς γραμμές
  παικτών (δηλαδή αγώνες που λείπουν).
- Ποσοστό DNP και εύρος PIR. Κάθε παιγμένος αγώνας πρέπει να έχει ακριβώς έναν νικητή.

Κωδικός εξόδου: 0 αν όλοι οι έλεγχοι περνούν, 1 αν υπάρχει οποιαδήποτε απόκλιση. Καμία απόκλιση
δεν κρύβεται. Αγώνες που λείπουν οριστικά από την πηγή (π.χ. το API επιστρέφει κενό boxscore)
μπορούν να δηλωθούν ρητά με `--accept-missing`: εμφανίζονται στην αναφορά ως αποδεκτά κενά και δεν
μετρούν ως αποτυχία.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, select

from elfantasy import scoring
from elfantasy.config import get_settings
from elfantasy.db import models
from elfantasy.db.session import get_engine

logger = logging.getLogger(__name__)

# Αγώνες ανά σεζόν στο API, όπως μετρήθηκαν στη Φάση 1 (docs/DATA_SOURCES.md, ενότητα 9).
# Η σεζόν 2026 συνεχίζεται, άρα δεν έχει σταθερό αναμενόμενο πλήθος.
EXPECTED_GAMES = {
    2016: 259,
    2017: 260,
    2018: 260,
    2019: 252,
    2020: 328,
    2021: 299,
    2022: 328,
    2023: 331,
    2024: 330,
    2025: 402,
}

MISMATCH_COLUMNS = [
    "season",
    "gamecode",
    "player_id",
    "team_code",
    "minutes",
    "valuation",
    "pir",
    "diff",
    *scoring.CLEAN_COLUMNS.values(),
]


@dataclass
class VerifyResult:
    """Αποτέλεσμα της επαλήθευσης."""

    season_summary: pd.DataFrame
    rows: int = 0
    pir_matches: int = 0
    pir_mismatches: pd.DataFrame = field(default_factory=pd.DataFrame)
    stored_pir_differs: int = 0
    fantasy_differs: int = 0
    games_without_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    accepted_missing: pd.DataFrame = field(default_factory=pd.DataFrame)
    games_without_one_winner: int = 0
    dnp_rows: int = 0
    pir_min: int = 0
    pir_max: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def pir_match_rate(self) -> float:
        return self.pir_matches / self.rows if self.rows else float("nan")

    @property
    def dnp_rate(self) -> float:
        return self.dnp_rows / self.rows if self.rows else float("nan")


def parse_game_keys(text: str) -> set[tuple[int, int]]:
    """Διαβάζει «2018/21,2021/7» (σεζόν/κωδικός αγώνα) και επιστρέφει σύνολο ζευγαριών."""
    keys = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        season, separator, gamecode = part.partition("/")
        if not separator or not season.strip().isdigit() or not gamecode.strip().isdigit():
            raise ValueError(f"Invalid game key {part!r}, expected SEASON/GAMECODE like 2018/21")
        keys.add((int(season), int(gamecode)))
    return keys


def load_missing_reasons(raw_dir: Path) -> dict[tuple[int, int], str]:
    """Οι λόγοι που καταγράφηκε ότι λείπει κάθε αγώνας, από το `missing.json` του cache."""
    path = Path(raw_dir) / "missing.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        (int(season), int(gamecode)): reason
        for season, entry in data.items()
        for gamecode, reason in entry.get("reasons", {}).items()
    }


def load_frames(engine: Engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Φορτώνει από τη βάση τους πίνακες `games` και `player_games`."""
    with engine.connect() as conn:
        games = pd.read_sql_query(select(models.games), conn)
        player_games = pd.read_sql_query(select(models.player_games), conn)
    return games, player_games


def pir_mismatch_table(player_games: pd.DataFrame) -> pd.DataFrame:
    """Γραμμές όπου το PIR από τα στατιστικά διαφέρει από το `valuation` του API."""
    recomputed = scoring.pir_frame(player_games)
    frame = player_games.assign(pir=recomputed, diff=recomputed - player_games["valuation"])
    return frame[frame["diff"] != 0][MISMATCH_COLUMNS].reset_index(drop=True)


def build_season_summary(
    games: pd.DataFrame, player_games: pd.DataFrame, expected: Mapping[int, int] | None = None
) -> pd.DataFrame:
    """Πίνακας ανά σεζόν: αγώνες, αγώνες με γραμμές παικτών, παίκτες, γραμμές, DNP, εύρος PIR."""
    played = games[games["played"]]
    per_game = player_games.groupby(["season", "gamecode"]).size().rename("rows").reset_index()
    played_keys = played[["season", "gamecode"]].merge(per_game, how="left")
    played_keys["rows"] = played_keys["rows"].fillna(0).astype(int)

    summary = pd.DataFrame(
        {
            "games_played": played.groupby("season").size(),
            "games_with_rows": played_keys[played_keys["rows"] > 0].groupby("season").size(),
            "players": player_games.groupby("season")["player_id"].nunique(),
            "rows": player_games.groupby("season").size(),
            "dnp_rows": player_games.groupby("season")["dnp"].sum(),
            "pir_min": player_games.groupby("season")["valuation"].min(),
            "pir_max": player_games.groupby("season")["valuation"].max(),
        }
    ).fillna(0)
    for column in ("games_with_rows", "players", "rows", "dnp_rows", "pir_min", "pir_max"):
        summary[column] = summary[column].astype(int)
    summary["games_missing"] = summary["games_played"] - summary["games_with_rows"]
    summary["rows_per_game"] = (
        summary["rows"] / summary["games_with_rows"].where(summary["games_with_rows"] > 0)
    ).round(2)
    summary["dnp_pct"] = (
        100 * summary["dnp_rows"] / summary["rows"].where(summary["rows"] > 0)
    ).round(2)
    expected_games = EXPECTED_GAMES if expected is None else expected
    summary["expected_games"] = [expected_games.get(season) for season in summary.index]
    return summary.reset_index().rename(columns={"index": "season"})


def verify(
    engine: Engine,
    reports_dir: Path,
    expected_games: Mapping[int, int] | None = None,
    accepted_missing: Collection[tuple[int, int]] = (),
    missing_reasons: Mapping[tuple[int, int], str] | None = None,
) -> VerifyResult:
    """Εκτελεί όλους τους ελέγχους, γράφει τις αναφορές CSV και επιστρέφει το αποτέλεσμα.

    - `expected_games`: σεζόν -> αριθμός αγώνων (προεπιλογή τα `EXPECTED_GAMES`).
    - `accepted_missing`: ζευγάρια (σεζόν, gamecode) αγώνων που λείπουν οριστικά από την πηγή.
      Αναφέρονται ως αποδεκτά κενά και δεν μετρούν ως αποτυχία.
    - `missing_reasons`: ο λόγος που λείπει κάθε αγώνας (από το `missing.json`), για την αναφορά.
    """
    expected = EXPECTED_GAMES if expected_games is None else expected_games
    games, player_games = load_frames(engine)
    if player_games.empty:
        return VerifyResult(season_summary=pd.DataFrame(), failures=["player_games is empty"])
    result = VerifyResult(season_summary=build_season_summary(games, player_games, expected))
    result.rows = len(player_games)

    recomputed = scoring.pir_frame(player_games)
    result.pir_matches = int((recomputed == player_games["valuation"]).sum())
    result.pir_mismatches = pir_mismatch_table(player_games)
    reports_dir.mkdir(parents=True, exist_ok=True)
    result.pir_mismatches.to_csv(reports_dir / "pir_mismatches.csv", index=False, encoding="utf-8")
    result.season_summary.to_csv(reports_dir / "season_summary.csv", index=False, encoding="utf-8")

    result.stored_pir_differs = int((recomputed != player_games["pir"]).sum())
    expected_fantasy = scoring.fantasy_score_frame(player_games["pir"], player_games["won"])
    result.fantasy_differs = int((expected_fantasy != player_games["fantasy_score"]).sum())

    summary = result.season_summary
    played_without_rows = games[games["played"]].merge(
        player_games[["season", "gamecode"]].drop_duplicates(), how="left", indicator=True
    )
    without_rows = played_without_rows[played_without_rows["_merge"] == "left_only"][
        ["season", "gamecode", "game_date", "home_code", "away_code"]
    ].reset_index(drop=True)
    reasons = missing_reasons or {}
    keys = list(zip(without_rows["season"], without_rows["gamecode"], strict=True))
    without_rows["reason"] = [reasons.get((int(s), int(g)), "") for s, g in keys]
    accepted = {(int(season), int(gamecode)) for season, gamecode in accepted_missing}
    is_accepted = pd.Series([(int(s), int(g)) in accepted for s, g in keys], dtype=bool)
    result.accepted_missing = without_rows[is_accepted.to_numpy()].reset_index(drop=True)
    result.games_without_rows = without_rows[~is_accepted.to_numpy()].reset_index(drop=True)
    without_rows.to_csv(reports_dir / "games_without_rows.csv", index=False, encoding="utf-8")

    won_per_game = player_games.groupby(["season", "gamecode", "team_code"])["won"].first()
    winners = won_per_game.groupby(["season", "gamecode"]).agg(["sum", "count"])
    result.games_without_one_winner = int(((winners["sum"] != 1) | (winners["count"] != 2)).sum())

    result.dnp_rows = int(player_games["dnp"].sum())
    result.pir_min = int(player_games["valuation"].min())
    result.pir_max = int(player_games["valuation"].max())

    if len(result.pir_mismatches):
        result.failures.append(
            f"{len(result.pir_mismatches)} rows with pir != valuation (see pir_mismatches.csv)"
        )
    if result.stored_pir_differs:
        result.failures.append(
            f"{result.stored_pir_differs} rows where stored pir is not recomputable"
        )
    if result.fantasy_differs:
        result.failures.append(f"{result.fantasy_differs} rows with inconsistent fantasy_score")
    if len(result.games_without_rows):
        result.failures.append(
            f"{len(result.games_without_rows)} played games have no player rows (not accepted)"
        )
    if result.games_without_one_winner:
        result.failures.append(
            f"{result.games_without_one_winner} games without exactly one winning team"
        )
    deviations = summary.dropna(subset=["expected_games"])
    deviations = deviations[deviations["games_played"] != deviations["expected_games"]]
    for row in deviations.itertuples():
        result.failures.append(
            f"season {row.season}: {row.games_played} games in the database, "
            f"{int(row.expected_games)} expected"
        )
    return result


def format_report(result: VerifyResult) -> str:
    """Κείμενο αναφοράς για την κονσόλα και το log."""
    lines = ["=== Ingestion verification ===", ""]
    lines.append(
        f"PIR (from statistics) == Valuation (API): {result.pir_matches} of {result.rows} rows "
        f"({100 * result.pir_match_rate:.4f}%)"
    )
    lines.append(f"Rows with pir != valuation: {len(result.pir_mismatches)}")
    if len(result.pir_mismatches):
        by_season = result.pir_mismatches.groupby("season").size().to_dict()
        lines.append(f"  by season: {by_season}")
        lines.append("  examples:")
        lines.append(result.pir_mismatches.head(10).to_string(index=False))
    lines.append(f"Stored pir differs from recomputed pir: {result.stored_pir_differs}")
    lines.append(f"Stored fantasy_score differs from pir/won rule: {result.fantasy_differs}")
    lines.append(f"Games without exactly one winning team: {result.games_without_one_winner}")
    lines.append(
        f"Played games without player rows: {len(result.games_without_rows)} "
        f"(plus {len(result.accepted_missing)} accepted gaps at the source)"
    )
    if len(result.games_without_rows):
        lines.append("  first 30 (all in games_without_rows.csv):")
        lines.append(result.games_without_rows.head(30).to_string(index=False))
    if len(result.accepted_missing):
        lines.append("  accepted gaps (--accept-missing):")
        lines.append(result.accepted_missing.to_string(index=False))
    lines.append(f"DNP rows: {result.dnp_rows} ({100 * result.dnp_rate:.2f}% of rows)")
    lines.append(f"PIR range (Valuation): {result.pir_min} to {result.pir_max}")
    lines.append("")
    lines.append("Per season:")
    lines.append(result.season_summary.to_string(index=False))
    lines.append("")
    lines.append("RESULT: " + ("OK" if not result.failures else "FAILED"))
    lines.extend(f"  - {failure}" for failure in result.failures)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m elfantasy.ingest.verify",
        description="Verify the ingested dataset (PIR vs Valuation, counts per season, DNP rate).",
    )
    parser.add_argument("--db", default=None, help="database URL (default: DATABASE_URL setting)")
    parser.add_argument("--data-dir", default=None, help="data folder (default: DATA_DIR setting)")
    parser.add_argument(
        "--accept-missing",
        default="",
        help="games missing for good at the source, as SEASON/GAMECODE list (e.g. 2018/21)",
    )
    args = parser.parse_args(argv)
    try:
        accepted = parse_game_keys(args.accept_missing)
    except ValueError as exc:
        parser.error(str(exc))

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    data_dir = Path(args.data_dir or get_settings().data_dir)
    engine = get_engine(args.db)
    try:
        result = verify(
            engine,
            data_dir / "reports",
            accepted_missing=accepted,
            missing_reasons=load_missing_reasons(data_dir / "raw"),
        )
    finally:
        engine.dispose()
    print(format_report(result))
    return 0 if not result.failures else 1


if __name__ == "__main__":
    sys.exit(main())
