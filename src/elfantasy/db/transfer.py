"""Μεταφορά των δεδομένων μιας βάσης σε άλλη (SQLite -> Postgres/Supabase) και επαλήθευσή τους.

Χρήση (τα URL δεν γράφονται ποτέ σε αρχείο· προτίμησε το `DATABASE_URL` του περιβάλλοντος για τον
προορισμό αντί του `--to`, ώστε ο κωδικός να μην μπει στο ιστορικό του shell)::

    # προορισμός: το DATABASE_URL
    python -m elfantasy.db.transfer --from sqlite:///data/elfantasy.db
    # μόνο το σχέδιο, καμία εγγραφή
    python -m elfantasy.db.transfer --from sqlite:///data/elfantasy.db --dry-run
    # μόνο σύγκριση πηγής και προορισμού
    python -m elfantasy.db.transfer --from sqlite:///data/elfantasy.db --verify-only

Ο προορισμός πρέπει να έχει ήδη το σχήμα (`python -m elfantasy.db.migrate`). Οι πίνακες
αντιγράφονται με σειρά foreign keys (`teams`, `players`, `games`, `player_games`, `predictions`,
`player_availability`), σε παρτίδες, με idempotent upsert: μια διακοπή ή επανάληψη δεν διπλασιάζει
γραμμές και συνεχίζει με ασφάλεια. Η πηγή διαβάζεται σε streaming (όχι ολόκληρη στη μνήμη) και,
όταν είναι αρχείο SQLite, ανοίγει μόνο για ανάγνωση (`mode=ro`): η εντολή δεν μπορεί να την
αλλάξει.

Στο τέλος (και μόνη της με `--verify-only`) γίνεται επαλήθευση: ίδια πλήθη γραμμών, ίδια
aggregates (count, sums, min/max ημερομηνιών, άθροισμα πόντων ανά σεζόν) και ίδιο «ψηφιακό
αποτύπωμα» ολόκληρου του περιεχομένου κάθε πίνακα, υπολογισμένο στην Python και στις δύο βάσεις
(ανεξάρτητο από τη σειρά των γραμμών και από το collation). Ο πίνακας των συγκρίσεων τυπώνεται
και ο κωδικός εξόδου είναι ≠ 0 σε οποιαδήποτε ασυμφωνία.

Προστασίες: η εντολή αρνείται να γράψει στην ίδια βάση που διαβάζει, και προορισμό που δεν είναι
Postgres εκτός αν δοθεί `--allow-non-postgres` (για tests).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Engine,
    Float,
    Integer,
    Table,
    UniqueConstraint,
    case,
    create_engine,
    func,
    inspect,
    select,
)
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError

from elfantasy.config import get_settings
from elfantasy.db import models
from elfantasy.db.cli import console_logging, report_failure, use_utf8_output
from elfantasy.db.session import get_engine, missing_tables, upsert
from elfantasy.db.urls import (
    DatabaseUrlError,
    backend_name,
    redact_secrets,
    safe_url,
    same_database,
    sqlite_database_path,
)

# Ρητό όνομα: με `python -m` το `__name__` είναι `__main__` και τα μηνύματα δεν θα έφταναν
# στον handler του πακέτου `elfantasy` (db/cli.py).
logger = logging.getLogger("elfantasy.db.transfer")

DEFAULT_BATCH_SIZE = 2000

# Πίνακες με «τεχνικό» αυτόματο κλειδί `id`: δεν αντιγράφεται (ο προορισμός δίνει δικό του) και η
# σύγκρουση του upsert ορίζεται από το unique constraint των φυσικών στηλών.
SURROGATE_KEYS = {"predictions": "id"}

_DIGEST_MASK = (1 << 64) - 1


class TransferError(Exception):
    """Αποτυχία μεταφοράς ή άκυρη διαμόρφωση. Το μήνυμα δεν περιέχει ποτέ το URL ή κωδικό."""


@dataclass
class TableResult:
    """Το αποτέλεσμα της μεταφοράς ενός πίνακα."""

    name: str
    source_rows: int = 0
    written: int = 0
    seconds: float = 0.0
    note: str = ""


@dataclass(frozen=True)
class Check:
    """Μία σύγκριση πηγής και προορισμού."""

    table: str
    metric: str
    source: Any
    target: Any
    ok: bool


@dataclass
class VerificationReport:
    """Οι συγκρίσεις της επαλήθευσης."""

    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if not check.ok]


# ----------------------------------------------------------------------------------------------
# Πίνακες, στήλες και τύποι
# ----------------------------------------------------------------------------------------------


def table_names() -> list[str]:
    """Τα ονόματα των πινάκων με τη σειρά των foreign keys."""
    return [table.name for table in models.TABLE_ORDER]


def select_tables(names: Sequence[str] | None) -> list[Table]:
    """Οι πίνακες που ζητήθηκαν, πάντα με τη σειρά των foreign keys (όχι της γραμμής εντολών)."""
    if not names:
        return list(models.TABLE_ORDER)
    unknown = sorted(set(names) - set(table_names()))
    if unknown:
        raise TransferError(
            f"unknown table(s): {', '.join(unknown)} (choose from {', '.join(table_names())})"
        )
    wanted = set(names)
    return [table for table in models.TABLE_ORDER if table.name in wanted]


def transfer_columns(table: Table) -> list[Column]:
    """Οι στήλες που αντιγράφονται: όλες εκτός από το τεχνικό `id` (βλ. `SURROGATE_KEYS`)."""
    skip = SURROGATE_KEYS.get(table.name)
    return [column for column in table.columns if column.name != skip]


def conflict_columns(table: Table) -> list[str]:
    """Οι στήλες που ορίζουν τη σύγκρουση του upsert: το primary key, ή (για πίνακες με τεχνικό
    `id`) το unique constraint των φυσικών στηλών."""
    if table.name in SURROGATE_KEYS:
        unique = next(c for c in table.constraints if isinstance(c, UniqueConstraint))
        return [column.name for column in unique.columns]
    return [column.name for column in table.primary_key.columns]


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _coerce_datetime(timezone_aware: bool) -> Callable[[Any], Any]:
    def coerce(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if isinstance(value, date) and not isinstance(value, datetime):
            value = datetime(value.year, value.month, value.day)
        # Οι ώρες της SQLite είναι naive και θεωρούνται UTC. Σε στήλη timestamptz μια naive ώρα θα
        # ερμηνευόταν στη ζώνη ώρας της συνεδρίας του server: γι' αυτό γίνεται ρητά UTC.
        return _to_utc(value) if timezone_aware else _to_utc(value).replace(tzinfo=None)

    return coerce


def _coerce_date(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    return value


def _nullable(function: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return lambda value: None if value is None else function(value)


def coercer_for(column: Column) -> Callable[[Any], Any]:
    """Η συνάρτηση που φέρνει μια τιμή της στήλης στον native τύπο Python του προορισμού:
    boolean (το 0/1 της SQLite), ακέραιος, δεκαδικός, ημερομηνία, ώρα (UTC) ή κείμενο· το NULL
    μένει NULL."""
    column_type = column.type
    if isinstance(column_type, Boolean):
        return _nullable(bool)
    if isinstance(column_type, DateTime):
        return _coerce_datetime(bool(column_type.timezone))
    if isinstance(column_type, Date):
        return _coerce_date
    if isinstance(column_type, Integer):
        return _nullable(int)
    if isinstance(column_type, Float):  # περιλαμβάνει το Double
        return _nullable(float)
    return _nullable(str)


# ----------------------------------------------------------------------------------------------
# Πηγή και προορισμός
# ----------------------------------------------------------------------------------------------


def open_source_engine(url: str) -> Engine:
    """Engine για την πηγή. Ένα αρχείο SQLite ανοίγει ΜΟΝΟ για ανάγνωση (`mode=ro`) και δεν
    δημιουργείται αν λείπει· κάθε άλλη βάση ανοίγει κανονικά."""
    path = sqlite_database_path(url)
    if path is None:
        return get_engine(url)
    if not path.is_file():
        raise TransferError("the source SQLite file does not exist")
    read_only = URL.create(
        "sqlite", database=f"file:{path.resolve().as_posix()}", query={"mode": "ro", "uri": "true"}
    )
    return create_engine(read_only)


def _source_state(source: Engine, table: Table) -> tuple[bool, str]:
    """(διαθέσιμος, σημείωση): αν ο πίνακας υπάρχει στην πηγή με όλες τις στήλες που χρειάζονται.

    Ένας πίνακας που λείπει ή έχει παλιά διάταξη (π.χ. το `predictions` πριν τη στήλη `as_of`)
    παραλείπεται όταν είναι κενός και είναι σφάλμα όταν έχει δεδομένα που δεν μεταφέρονται.
    """
    inspector = inspect(source)
    if not inspector.has_table(table.name):
        return False, "table does not exist in the source"
    present = {column["name"] for column in inspector.get_columns(table.name)}
    missing = [column.name for column in transfer_columns(table) if column.name not in present]
    if not missing:
        return True, ""
    with source.connect() as connection:
        rows = connection.execute(select(func.count()).select_from(table)).scalar_one()
    if rows == 0:
        return False, f"empty table with an old layout (missing columns: {', '.join(missing)})"
    raise TransferError(
        f"source table {table.name} has {rows} rows but lacks the columns: {', '.join(missing)}"
    )


def _count(engine: Engine, table: Table) -> int:
    with engine.connect() as connection:
        return connection.execute(select(func.count()).select_from(table)).scalar_one()


def _check_target(target: Engine, tables: Sequence[Table]) -> None:
    missing = missing_tables(target, tables)
    if missing:
        raise TransferError(
            f"the target database is missing tables: {', '.join(missing)}; create the schema first "
            "with: python -m elfantasy.db.migrate"
        )


# ----------------------------------------------------------------------------------------------
# Μεταφορά
# ----------------------------------------------------------------------------------------------


def _chunks(engine: Engine, table: Table, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    """Οι γραμμές του πίνακα σε παρτίδες `batch_size` (streaming, χωρίς ταξινόμηση)."""
    columns = transfer_columns(table)
    with engine.connect() as connection:
        result = connection.execution_options(stream_results=True).execute(select(*columns))
        for chunk in result.mappings().partitions(batch_size):
            yield [dict(row) for row in chunk]


def copy_table(
    source: Engine, target: Engine, table: Table, batch_size: int = DEFAULT_BATCH_SIZE
) -> TableResult:
    """Αντιγράφει έναν πίνακα με upsert, μία συναλλαγή ανά παρτίδα."""
    started = time.perf_counter()
    columns = transfer_columns(table)
    coercers = {column.name: coercer_for(column) for column in columns}
    keys = conflict_columns(table)
    updates = [name for name in coercers if name not in keys]
    total = _count(source, table)
    result = TableResult(table.name, source_rows=total)
    report_every = max(1, math.ceil(total / batch_size / 10))  # περίπου κάθε 10% της προόδου
    for number, chunk in enumerate(_chunks(source, table, batch_size), start=1):
        rows = [{name: coercers[name](row[name]) for name in coercers} for row in chunk]
        try:
            with target.begin() as connection:
                upsert(
                    connection,
                    table,
                    rows,
                    keys,
                    set_=lambda excluded: {name: excluded[name] for name in updates},
                )
        except IntegrityError as exc:
            reason = type(exc.orig).__name__
            hint = ""
            if "foreign key" in str(exc.orig).lower() or reason == "ForeignKeyViolation":
                hint = "; copy the parent tables first (teams, players, games)"
            # Το μήνυμα του driver δεν γράφεται: περιέχει τιμές των γραμμών.
            raise TransferError(
                f"{table.name}: the target rejected a batch ({reason}){hint}"
            ) from None
        result.written += len(rows)
        if number % report_every == 0 or result.written >= total:
            logger.info(
                "%s: %d/%d rows (%.0f%%), %.1f s",
                table.name,
                result.written,
                total,
                100 * result.written / total,
                time.perf_counter() - started,
            )
    result.seconds = time.perf_counter() - started
    return result


def transfer(
    source: Engine,
    target: Engine,
    tables: Sequence[str] | None = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
) -> list[TableResult]:
    """Μεταφέρει τους πίνακες `tables` (προεπιλογή: όλοι) από την πηγή στον προορισμό.

    Ο προορισμός ελέγχεται πριν από οποιαδήποτε εγγραφή (πρέπει να έχει όλους τους πίνακες).
    Πίνακες που λείπουν ή είναι κενοί στην πηγή παραλείπονται. Με `dry_run` δεν γράφεται τίποτα.
    """
    if batch_size < 1:
        raise TransferError("the batch size must be at least 1")
    selected = select_tables(tables)
    _check_target(target, selected)
    results: list[TableResult] = []
    for table in selected:
        available, note = _source_state(source, table)
        if not available:
            logger.info("%s: skipped (%s)", table.name, note)
            results.append(TableResult(table.name, note=f"skipped: {note}"))
            continue
        rows = _count(source, table)
        if rows == 0:
            logger.info("%s: skipped (the source table is empty)", table.name)
            results.append(TableResult(table.name, note="skipped: the source table is empty"))
            continue
        if dry_run:
            batches = math.ceil(rows / batch_size)
            logger.info(
                "%s: would copy %d rows in %d batches (the target has %d rows now)",
                table.name,
                rows,
                batches,
                _count(target, table),
            )
            results.append(TableResult(table.name, source_rows=rows, note="dry run"))
            continue
        logger.info("%s: copying %d rows", table.name, rows)
        results.append(copy_table(source, target, table, batch_size))
    return results


# ----------------------------------------------------------------------------------------------
# Επαλήθευση
# ----------------------------------------------------------------------------------------------


def _scalar_metrics(table: Table) -> list[tuple[str, Any]]:
    """Τα aggregates ενός πίνακα που υπολογίζονται από τη βάση (το πρώτο είναι πάντα το πλήθος)."""
    c = table.c
    metrics: list[tuple[str, Any]] = [("count", func.count())]
    if table.name == "players":
        metrics += [
            ("min(first_season)", func.min(c.first_season)),
            ("max(last_season)", func.max(c.last_season)),
        ]
    elif table.name == "games":
        metrics += [
            ("min(game_date)", func.min(c.game_date)),
            ("max(game_date)", func.max(c.game_date)),
            ("played games", func.sum(case((c.played.is_(True), 1), else_=0))),
            ("sum(home_score)", func.sum(c.home_score)),
            ("sum(away_score)", func.sum(c.away_score)),
        ]
    elif table.name == "player_games":
        metrics += [
            ("sum(pir)", func.sum(c.pir)),
            ("sum(valuation)", func.sum(c.valuation)),
            ("sum(fantasy_score)", func.sum(c.fantasy_score)),
            ("sum(minutes)", func.sum(c.minutes)),
            ("dnp rows", func.sum(case((c.dnp.is_(True), 1), else_=0))),
        ]
    elif table.name == "predictions":
        metrics += [
            ("sum(predicted_fantasy)", func.sum(c.predicted_fantasy)),
            ("sum(predicted_pir)", func.sum(c.predicted_pir)),
            ("min(as_of)", func.min(c.as_of)),
            ("max(as_of)", func.max(c.as_of)),
        ]
    return metrics


def _grouped_metrics(table: Table) -> list[tuple[str, Column, Any]]:
    """Aggregates ανά ομάδα: (όνομα, στήλη ομαδοποίησης, έκφραση)."""
    if table.name == "player_games":
        return [("sum(points)", table.c.season, func.sum(table.c.points))]
    if table.name == "player_availability":
        return [("count", table.c.status, func.count())]
    return []


def _aggregates(engine: Engine, table: Table) -> dict[str, Any]:
    """Όλα τα aggregates του πίνακα ως λεξικό (όνομα -> τιμή)."""
    values: dict[str, Any] = {}
    scalars = _scalar_metrics(table)
    with engine.connect() as connection:
        # Το select_from είναι απαραίτητο: χωρίς αυτό το `count(*)` ενός πίνακα που δεν αναφέρεται
        # σε καμία στήλη του (π.χ. teams) θα γινόταν `SELECT count(*)` χωρίς FROM και θα έδινε 1.
        statement = select(*(expression for _, expression in scalars)).select_from(table)
        row = connection.execute(statement).one()
        for (name, _), value in zip(scalars, row, strict=True):
            values[name] = value
        for name, group, expression in _grouped_metrics(table):
            statement = select(group, expression).group_by(group)
            for key, value in connection.execute(statement):
                values[f"{name} {group.name}={key}"] = value
    return values


def table_digest(engine: Engine, table: Table, batch_size: int = DEFAULT_BATCH_SIZE) -> str:
    """Ψηφιακό αποτύπωμα του περιεχομένου του πίνακα: `πλήθος:άθροισμα hash γραμμών (64 bit)`.

    Κάθε γραμμή κανονικοποιείται στους native τύπους του προορισμού (βλ. `coercer_for`) και
    κατακερματίζεται· τα hashes αθροίζονται, άρα το αποτέλεσμα ΔΕΝ εξαρτάται από τη σειρά των
    γραμμών (η σειρά και το collation διαφέρουν ανάμεσα σε SQLite και Postgres). Μια διαφορά σε
    οποιαδήποτε στήλη οποιασδήποτε γραμμής αλλάζει το αποτέλεσμα.
    """
    columns = transfer_columns(table)
    coercers = [(column.name, coercer_for(column)) for column in columns]
    rows = 0
    accumulator = 0
    for chunk in _chunks(engine, table, batch_size):
        for row in chunk:
            canonical = repr(tuple(coerce(row[name]) for name, coerce in coercers))
            hashed = hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).digest()
            accumulator = (accumulator + int.from_bytes(hashed, "big")) & _DIGEST_MASK
            rows += 1
    return f"{rows}:{accumulator:016x}"


def values_match(first: Any, second: Any) -> bool:
    """Ισότητα τιμών aggregate: τα NULL ισούνται μόνο με NULL και οι δεκαδικοί συγκρίνονται με
    ελάχιστη ανοχή (η σειρά άθροισης διαφέρει ανάμεσα στις βάσεις)."""
    if first is None or second is None:
        return first is None and second is None
    if isinstance(first, (float, Decimal)) or isinstance(second, (float, Decimal)):
        return math.isclose(float(first), float(second), rel_tol=1e-9, abs_tol=1e-6)
    return first == second


def verify(
    source: Engine,
    target: Engine,
    tables: Sequence[str] | None = None,
    *,
    row_digest: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> VerificationReport:
    """Συγκρίνει πηγή και προορισμό: πλήθη, aggregates και (προαιρετικά) αποτύπωμα περιεχομένου.

    Ένας πίνακας που λείπει από την πηγή ή έχει παλιά διάταξη συγκρίνεται ως κενός (ο προορισμός
    πρέπει τότε να είναι επίσης κενός).
    """
    report = VerificationReport()
    selected = select_tables(tables)
    _check_target(target, selected)
    for table in selected:
        available, _ = _source_state(source, table)
        source_values = _aggregates(source, table) if available else {}
        target_values = _aggregates(target, table)
        if not available:
            source_values = {"count": 0}
        for name in sorted(set(source_values) | set(target_values), key=_metric_order(table)):
            first, second = source_values.get(name), target_values.get(name)
            report.checks.append(
                Check(table.name, name, first, second, values_match(first, second))
            )
        if row_digest:
            first = table_digest(source, table, batch_size) if available else "0:0000000000000000"
            second = table_digest(target, table, batch_size)
            report.checks.append(
                Check(table.name, "content digest", first, second, first == second)
            )
    return report


def _metric_order(table: Table) -> Callable[[str], tuple[int, str]]:
    """Σειρά εμφάνισης: πρώτα τα scalar με τη σειρά ορισμού τους, μετά τα ανά ομάδα."""
    scalars = [name for name, _ in _scalar_metrics(table)]
    return lambda name: (scalars.index(name), "") if name in scalars else (len(scalars), name)


def _format_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (float, Decimal)):
        return format(float(value), ".10g")
    return str(value)


def format_report(report: VerificationReport) -> str:
    """Ο πίνακας των συγκρίσεων για την κονσόλα."""
    rows = [("table", "check", "source", "target", "result")]
    rows += [
        (
            check.table,
            check.metric,
            _format_value(check.source),
            _format_value(check.target),
            "OK" if check.ok else "MISMATCH",
        )
        for check in report.checks
    ]
    widths = [max(len(row[index]) for row in rows) for index in range(5)]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in rows
    ]
    lines.insert(1, "  ".join("-" * width for width in widths))
    verdict = (
        f"VERIFICATION OK ({len(report.checks)} checks)"
        if report.ok
        else f"VERIFICATION FAILED: {len(report.failures)} of {len(report.checks)} checks differ"
    )
    return "\n".join([*lines, "", verdict])


# ----------------------------------------------------------------------------------------------
# Γραμμή εντολών
# ----------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m elfantasy.db.transfer",
        description="Copy the data of one database into another (SQLite -> Postgres/Supabase) "
        "with idempotent upserts, then verify counts, aggregates and content digests.",
    )
    parser.add_argument(
        "--from",
        dest="source",
        required=True,
        help="source database URL, e.g. sqlite:///data/elfantasy.db (opened read-only)",
    )
    parser.add_argument(
        "--to",
        default=None,
        help="target database URL (default: the DATABASE_URL setting). Prefer the environment "
        "variable: a URL on the command line ends up in the shell history",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=None,
        metavar="TABLE",
        help=f"tables to copy (default: all, in foreign key order: {', '.join(table_names())})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"rows per batch and per transaction (default: {DEFAULT_BATCH_SIZE})",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="show the plan and write nothing to the target"
    )
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="copy nothing: only compare the source and the target",
    )
    parser.add_argument(
        "--no-row-digest",
        action="store_true",
        help="skip the content digest (faster): compare only counts and aggregates",
    )
    parser.add_argument(
        "--allow-non-postgres",
        action="store_true",
        help="allow a target that is not PostgreSQL (for tests only)",
    )
    return parser


def _tables_argument(raw: Sequence[str] | None) -> list[str] | None:
    if raw is None:
        return None
    return [name.strip() for item in raw for name in item.split(",") if name.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    """Κωδικοί εξόδου: 0 επιτυχία, 1 αποτυχία ή ασυμφωνία στην επαλήθευση, 2 άρνηση (άκυρο URL,
    ίδια βάση, προορισμός που δεν είναι Postgres, πηγή που δεν υπάρχει)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    use_utf8_output()
    source_url = args.source
    target_url = args.to or get_settings().database_url
    started = time.perf_counter()
    with console_logging():
        source = target = None
        try:
            tables = select_tables(_tables_argument(args.tables))
            if args.batch_size < 1:
                raise TransferError("the batch size must be at least 1")
            if same_database(source_url, target_url):
                raise TransferError(
                    "refusing to copy a database onto itself (same source and target)"
                )
            if backend_name(target_url) != "postgresql" and not args.allow_non_postgres:
                raise TransferError(
                    f"the target is not PostgreSQL ({backend_name(target_url)}): refusing "
                    "(use --allow-non-postgres only for tests)"
                )
            source = open_source_engine(source_url)
            target = get_engine(target_url)
        except (TransferError, DatabaseUrlError) as exc:
            logger.error("%s", exc)
            if source is not None:
                source.dispose()
            return 2
        logger.info("Source: %s (read-only)", safe_url(source_url))
        logger.info("Target: %s", safe_url(target_url))
        names = [table.name for table in tables]
        try:
            if not args.verify_only:
                results = transfer(
                    source, target, names, batch_size=args.batch_size, dry_run=args.dry_run
                )
                if args.dry_run:
                    planned = sum(result.source_rows for result in results)
                    logger.info("Dry run: %d rows would be copied, nothing was written", planned)
                    return 0
                written = sum(result.written for result in results)
                elapsed = time.perf_counter() - started
                logger.info(
                    "Copied %d rows in %.1f s (%.0f rows/s)",
                    written,
                    elapsed,
                    written / elapsed if elapsed else 0,
                )
            report = verify(
                source,
                target,
                names,
                row_digest=not args.no_row_digest,
                batch_size=args.batch_size,
            )
            print(format_report(report))
            logger.info("Total time: %.1f s", time.perf_counter() - started)
            return 0 if report.ok else 1
        except TransferError as exc:
            logger.error("%s", redact_secrets(str(exc), source_url, target_url))
            return 1
        except Exception as exc:
            report_failure("transfer failed", exc, source_url, target_url)
            return 1
        finally:
            source.dispose()
            target.dispose()


if __name__ == "__main__":
    sys.exit(main())
