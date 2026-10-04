"""Ροή ingestion: fetch (με cache) -> clean -> pir και fantasy score -> upsert στη βάση.

Εκτέλεση από τη ρίζα του repo:

    python -m elfantasy.ingest.pipeline --seasons 2016-2026
    python -m elfantasy.ingest.pipeline --update            # μόνο νέοι αγώνες της τρέχουσας σεζόν
    python -m elfantasy.ingest.pipeline --seasons 2016-2026 --no-fetch   # μόνο από το cache

Το pipeline είναι idempotent: οι εγγραφές στη βάση γίνονται με upsert, άρα ένα νέο τρέξιμο δεν
διπλασιάζει γραμμές. Το log γράφεται στο `data/logs/ingest.log` και στην κονσόλα.

Κωδικοί εξόδου: 0 επιτυχία, 1 σφάλμα (π.χ. αποτυχία ελέγχου ποιότητας), 2 ολοκληρώθηκε αλλά λείπουν
αγώνες (αναφέρονται στο `data/raw/missing.json`).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, case, func, select

from elfantasy import scoring
from elfantasy.config import get_settings
from elfantasy.db import models
from elfantasy.db.session import (
    LegacySchemaError,
    SchemaNotInitialisedError,
    ensure_schema,
    get_engine,
    upsert,
)
from elfantasy.db.urls import safe_url
from elfantasy.ingest import clean
from elfantasy.ingest.fetch import (
    DEFAULT_RPS,
    MAX_RPS,
    EuroleagueApiSource,
    FetchError,
    RateLimiter,
    RawCache,
    RequestRunner,
    SeasonReport,
    Source,
    fetch_seasons,
)

logger = logging.getLogger(__name__)

FIRST_SEASON = 2016  # η πρώτη σεζόν των δεδομένων του project (2016-17)


# ----------------------------------------------------------------------------------------------
# Βοηθητικά
# ----------------------------------------------------------------------------------------------


def current_season(today: date | None = None) -> int:
    """Η σεζόν που βρίσκεται σε εξέλιξη (έτος έναρξης): από τον Αύγουστο και μετά το τρέχον έτος."""
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


def parse_seasons(text: str) -> list[int]:
    """Διαβάζει «2016-2026» (εύρος), «2024,2025» (λίστα) ή «2025» (μία σεζόν)."""
    seasons: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            if start > end:
                raise ValueError(f"Invalid season range: {part!r}")
            seasons.extend(range(start, end + 1))
        elif part:
            seasons.append(int(part))
    if not seasons:
        raise ValueError(f"No seasons in {text!r}")
    return sorted(set(seasons))


def python_values(series: pd.Series) -> list:
    """Τιμές στήλης ως native τύποι Python, με None στη θέση των NaN, NaT και NA."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return [None if pd.isna(value) else value.to_pydatetime() for value in series]
    values = []
    for value in series.astype(object):
        if value is None or value is pd.NA or (isinstance(value, float) and value != value):
            values.append(None)
        else:
            values.append(value.item() if hasattr(value, "item") else value)
    return values


def to_records(frame: pd.DataFrame, columns: Sequence[str]) -> list[dict]:
    """Μετατρέπει τις στήλες ενός DataFrame σε λίστα dict για εισαγωγή στη βάση."""
    if frame.empty:
        return []
    columns = list(columns)
    column_values = [python_values(frame[column]) for column in columns]
    return [dict(zip(columns, row, strict=True)) for row in zip(*column_values, strict=True)]


def add_scores(player_games: pd.DataFrame) -> pd.DataFrame:
    """Προσθέτει το PIR (από τα στατιστικά) και το fantasy score (PIR + |PIR|/10 σε νίκη)."""
    result = player_games.copy()
    result["pir"] = scoring.pir_frame(result)
    result["fantasy_score"] = scoring.fantasy_score_frame(result["pir"], result["won"])
    return result


# ----------------------------------------------------------------------------------------------
# Φόρτωση από cache και εγγραφή στη βάση
# ----------------------------------------------------------------------------------------------


def load_from_cache(
    cache: RawCache, seasons: Iterable[int]
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """Διαβάζει από το cache boxscores, results και schedule των σεζόν που υπάρχουν."""
    boxscores, results, schedules = [], {}, {}
    for season in seasons:
        box = cache.load_boxscores(season)
        result = cache.load_results(season)
        schedule = cache.load_schedule(season)
        if result is None and schedule is None:
            logger.warning("Season %s: no results or schedule in cache, skipped", season)
            continue
        if result is not None:
            results[season] = result
        if schedule is not None:
            schedules[season] = schedule
        if box is None or box.empty:
            logger.warning("Season %s: no boxscores in cache (games only)", season)
        else:
            boxscores.append(box)
    if not boxscores:
        raise FetchError("There are no cached boxscores for the requested seasons")
    return pd.concat(boxscores, ignore_index=True), results, schedules


def _player_updates(excluded) -> dict:
    """Ενημέρωση παικτών σε σύγκρουση: το εύρος σεζόν μόνο διευρύνεται και το όνομα αλλάζει μόνο
    όταν τα νέα δεδομένα δεν είναι παλαιότερα από όσα υπάρχουν ήδη."""
    table = models.players.c
    return {
        "name": case((excluded.last_season >= table.last_season, excluded.name), else_=table.name),
        "first_season": case(
            (excluded.first_season < table.first_season, excluded.first_season),
            else_=table.first_season,
        ),
        "last_season": case(
            (excluded.last_season > table.last_season, excluded.last_season),
            else_=table.last_season,
        ),
    }


def load_to_db(engine: Engine, data: clean.CleanData, player_games: pd.DataFrame) -> dict[str, int]:
    """Εισάγει teams, players, games και player_games σε μία συναλλαγή (idempotent upsert).

    Επιστρέφει το πλήθος των γραμμών που στάλθηκαν ανά πίνακα. Τα ονόματα ομάδων ενημερώνονται
    μόνο αν τα δεδομένα που φορτώνονται δεν είναι παλαιότερα από τη νεότερη σεζόν της βάσης, ώστε
    η επανεκτέλεση παλιών σεζόν να μην αντικαθιστά πρόσφατα ονόματα με παλαιότερα.
    """

    def records(frame: pd.DataFrame, table) -> list[dict]:
        return to_records(frame, [column.name for column in table.columns])

    sent = {}
    with engine.begin() as conn:
        newest_in_db = conn.execute(select(func.max(models.games.c.season))).scalar()
        newest_loaded = int(data.games["season"].max())
        refresh_team_names = newest_in_db is None or newest_loaded >= newest_in_db

        sent["teams"] = upsert(
            conn,
            models.teams,
            records(data.teams, models.teams),
            ["team_code"],
            set_=None if refresh_team_names else {},
        )
        sent["players"] = upsert(
            conn,
            models.players,
            records(data.players, models.players),
            ["player_id"],
            set_=_player_updates,
        )
        sent["games"] = upsert(
            conn, models.games, records(data.games, models.games), ["season", "gamecode"]
        )
        sent["player_games"] = upsert(
            conn,
            models.player_games,
            records(player_games, models.player_games),
            ["season", "gamecode", "player_id"],
        )
    return sent


def table_counts(engine: Engine) -> dict[str, int]:
    """Πλήθος γραμμών κάθε πίνακα της βάσης."""
    with engine.connect() as conn:
        return {
            table.name: conn.execute(select(func.count()).select_from(table)).scalar_one()
            for table in (
                models.teams,
                models.players,
                models.games,
                models.player_games,
                models.predictions,
            )
        }


def write_reports(reports_dir: Path, data: clean.CleanData) -> None:
    """Γράφει τις αναφορές `name_anomalies.csv` και `quality_issues.csv` (UTF-8)."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    data.name_anomalies.to_csv(reports_dir / "name_anomalies.csv", index=False, encoding="utf-8")
    data.quality.issues.to_csv(reports_dir / "quality_issues.csv", index=False, encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Κύρια ροή
# ----------------------------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Περίληψη ενός τρεξίματος του pipeline."""

    seasons: list[int]
    fetch_reports: list[SeasonReport] = field(default_factory=list)
    rows_sent: dict[str, int] = field(default_factory=dict)
    table_counts: dict[str, int] = field(default_factory=dict)
    quality_counts: dict[str, int] = field(default_factory=dict)
    pir_matches: int = 0
    pir_rows: int = 0
    seconds: float = 0.0

    @property
    def missing_games(self) -> int:
        return sum(len(report.missing) for report in self.fetch_reports)

    @property
    def failed_seasons(self) -> list[int]:
        return [report.season for report in self.fetch_reports if report.error]


def run_pipeline(
    seasons: Sequence[int],
    *,
    db_url: str | None = None,
    data_dir: Path | str | None = None,
    rps: float = DEFAULT_RPS,
    fetch: bool = True,
    update: bool = False,
    source: Source | None = None,
    allow_quality_issues: Iterable[str] = (),
    checkpoint_every: int = 20,
    sleep: Callable[[float], None] = time.sleep,
) -> PipelineResult:
    """Εκτελεί ολόκληρο το pipeline για τις `seasons` και επιστρέφει περίληψη.

    - `fetch=False`: χωρίς δίκτυο, μόνο επεξεργασία από το cache (`--no-fetch`).
    - `update=True`: ανανεώνει results και schedule των `seasons` ακόμη κι αν υπάρχουν στο cache
      (η επιλογή `--update` περνάει εδώ μόνο την τρέχουσα σεζόν).
    - `source`: αντικαθιστά τις κλήσεις δικτύου (στα tests). Αλλιώς: euroleague_api.
    """
    started = time.monotonic()
    settings = get_settings()
    data_dir = Path(data_dir) if data_dir is not None else Path(settings.data_dir)
    cache = RawCache(data_dir / "raw")
    reports_dir = data_dir / "reports"
    result = PipelineResult(seasons=list(seasons))

    if fetch:
        limiter = RateLimiter(rps, sleep=sleep)
        runner = RequestRunner(limiter, sleep=sleep)
        result.fetch_reports = fetch_seasons(
            seasons,
            source or EuroleagueApiSource(),
            runner,
            cache,
            refresh_seasons=seasons if update else (),
            checkpoint_every=checkpoint_every,
            sleep=sleep,
        )

    boxscores, results_by_season, schedule_by_season = load_from_cache(cache, seasons)
    logger.info(
        "Cleaning %d raw boxscore rows of %d seasons", len(boxscores), len(results_by_season)
    )
    data = clean.clean_all(boxscores, results_by_season, schedule_by_season)
    write_reports(reports_dir, data)
    result.quality_counts = data.quality.counts()
    for check, count in result.quality_counts.items():
        logger.warning("Quality check %s: %d findings (see quality_issues.csv)", check, count)
    data.quality.raise_if_errors(allow=allow_quality_issues)

    player_games = add_scores(data.player_games)
    result.pir_rows = len(player_games)
    result.pir_matches = int((player_games["pir"] == player_games["valuation"]).sum())
    logger.info(
        "PIR from raw statistics equals Valuation in %d of %d rows",
        result.pir_matches,
        result.pir_rows,
    )

    engine = get_engine(db_url)
    try:
        ensure_schema(engine)
        result.rows_sent = load_to_db(engine, data, player_games)
        result.table_counts = table_counts(engine)
    finally:
        engine.dispose()
    result.seconds = time.monotonic() - started
    logger.info("Rows sent to the database: %s", result.rows_sent)
    logger.info("Row counts in the database: %s", result.table_counts)
    logger.info("Pipeline finished in %.0f s", result.seconds)
    return result


# ----------------------------------------------------------------------------------------------
# Γραμμή εντολών
# ----------------------------------------------------------------------------------------------


# Οι handlers που έχει εγκαταστήσει το configure_logging (αφαιρούνται όταν ξανακαλείται).
_installed_handlers: list[logging.Handler] = []


def configure_logging(logs_dir: Path, level: str = "INFO") -> Path:
    """Ρυθμίζει το logging: αρχείο `ingest.log` (UTF-8) και κονσόλα. Επιστρέφει τη διαδρομή του."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "ingest.log"
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    console_handler = logging.StreamHandler(sys.stdout)
    root = logging.getLogger()
    for handler in _installed_handlers:
        root.removeHandler(handler)
        handler.close()
    _installed_handlers.clear()
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        root.addHandler(handler)
        _installed_handlers.append(handler)
    root.setLevel(level)
    return log_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m elfantasy.ingest.pipeline",
        description="Euroleague ingestion: fetch (cached), clean, score and load into a database.",
    )
    parser.add_argument(
        "--seasons",
        default=None,
        help=f"seasons as start years: '2016-2026', '2024,2025' or '2025' "
        f"(default: {FIRST_SEASON}-<current season>)",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=DEFAULT_RPS,
        help=f"requests per second, at most {MAX_RPS} (default: {DEFAULT_RPS})",
    )
    parser.add_argument("--db", default=None, help="database URL (default: DATABASE_URL setting)")
    parser.add_argument(
        "--update",
        action="store_true",
        help="only the newest season: refresh results and schedule and fetch only new games",
    )
    parser.add_argument(
        "--no-fetch", action="store_true", help="no network access: process the raw cache only"
    )
    parser.add_argument("--data-dir", default=None, help="data folder (default: DATA_DIR setting)")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="games between cache checkpoints (default: 20)",
    )
    parser.add_argument(
        "--allow-quality-issues",
        default="",
        help="comma separated blocking checks to downgrade to warnings (e.g. points_formula)",
    )
    parser.add_argument("--log-level", default="INFO", help="logging level (default: INFO)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 < args.rps <= MAX_RPS:
        parser.error(f"--rps must be in (0, {MAX_RPS}]")
    try:
        seasons = (
            parse_seasons(args.seasons)
            if args.seasons
            else list(range(FIRST_SEASON, current_season() + 1))
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.update:
        seasons = [seasons[-1] if args.seasons else current_season()]

    settings = get_settings()
    data_dir = Path(args.data_dir or settings.data_dir)
    log_path = configure_logging(data_dir / "logs", args.log_level)
    shown = {**vars(args), "db": safe_url(args.db) if args.db else None}  # χωρίς κωδικό
    logger.info("Starting ingestion: seasons %s, arguments %s", seasons, shown)
    logger.info("Database: %s", safe_url(args.db or settings.database_url))
    allow = [name.strip() for name in args.allow_quality_issues.split(",") if name.strip()]
    try:
        result = run_pipeline(
            seasons,
            db_url=args.db,
            data_dir=data_dir,
            rps=args.rps,
            fetch=not args.no_fetch,
            update=args.update,
            allow_quality_issues=allow,
            checkpoint_every=args.checkpoint_every,
        )
    except (FetchError, clean.DataQualityError) as exc:
        logger.error("Ingestion failed: %s", exc)
        return 1
    except (SchemaNotInitialisedError, LegacySchemaError) as exc:
        # Στο Postgres το σχήμα δημιουργείται μόνο από τα migrations (docs/DATABASE.md).
        logger.error("Ingestion failed: %s", exc)
        return 1
    except Exception:
        logger.exception("Ingestion failed with an unexpected error (log: %s)", log_path)
        return 1

    if result.missing_games or result.failed_seasons:
        logger.error(
            "Finished with problems: %d missing games, failed seasons %s (see missing.json)",
            result.missing_games,
            result.failed_seasons,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
