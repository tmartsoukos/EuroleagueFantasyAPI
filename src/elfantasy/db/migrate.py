"""Migrations της βάσης Postgres: εφαρμογή αρχείων SQL με σειρά, καταγραφή και έλεγχος checksum.

Χρήση (από τη ρίζα του repo, με το `DATABASE_URL` στο περιβάλλον ή με `--db`)::

    python -m elfantasy.db.migrate --status      # τι έχει εφαρμοστεί και τι εκκρεμεί
    python -m elfantasy.db.migrate --dry-run     # τι θα εφαρμοζόταν (καμία εγγραφή στη βάση)
    python -m elfantasy.db.migrate               # εφαρμογή των εκκρεμών migrations
    python -m elfantasy.db.migrate --print-sql   # το DDL του 001_init.sql, από το db/models.py

Τα migrations είναι τα αρχεία `migrations/NNN_όνομα.sql` (UTF-8 χωρίς BOM). Εφαρμόζονται με
αύξουσα σειρά του αριθμού NNN. Κάθε migration τρέχει σε ΜΙΑ συναλλαγή μαζί με την εγγραφή του
στον πίνακα `schema_migrations(version, applied_at, checksum)`: το Postgres υποστηρίζει
transactional DDL, άρα ένα migration που αποτυγχάνει στη μέση δεν αφήνει τίποτα πίσω. Το τρέξιμο
είναι idempotent (ό,τι έχει εφαρμοστεί παραλείπεται) και δυνατά αυστηρό:

* αν ένα εφαρμοσμένο migration έχει αλλάξει ως αρχείο (άλλο checksum), ή το αρχείο του λείπει, ή
  εκκρεμεί migration με αριθμό μικρότερο από ήδη εφαρμοσμένο, ΔΕΝ εφαρμόζεται τίποτα·
* δύο ταυτόχρονα τρεξίματα δεν συγκρούονται (advisory lock του Postgres ανά migration).

Το εργαλείο δουλεύει ΜΟΝΟ σε Postgres (σε SQLite το σχήμα δημιουργείται με `create_all`). Η
κλάση `Migrator` δεν ελέγχει το dialect, ώστε η λογική (σειρά, checksum, idempotency) να
δοκιμάζεται offline σε SQLite με συνθετικά migrations· το CLI όμως αρνείται URL που δεν είναι
Postgres.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    Column,
    Connection,
    DateTime,
    Engine,
    MetaData,
    Table,
    Text,
    func,
    insert,
    inspect,
    select,
    text,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from elfantasy.config import get_settings
from elfantasy.db import models
from elfantasy.db.cli import console_logging, report_failure, use_utf8_output, write_text
from elfantasy.db.session import get_engine
from elfantasy.db.urls import DatabaseUrlError, backend_name, redact_secrets, safe_url

# Ρητό όνομα: με `python -m` το `__name__` είναι `__main__` και τα μηνύματα δεν θα έφταναν
# στον handler του πακέτου `elfantasy` (db/cli.py).
logger = logging.getLogger("elfantasy.db.migrate")

# Ο φάκελος των αρχείων SQL (μέσα στο πακέτο, ώστε να ακολουθεί το `pip install`).
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

SCHEMA_MIGRATIONS_TABLE = "schema_migrations"

# Όνομα αρχείου migration: τρία ή περισσότερα ψηφία, κάτω παύλα, όνομα με πεζά λατινικά, αριθμούς
# και κάτω παύλες, και κατάληξη .sql.
_MIGRATION_NAME = re.compile(r"^(?P<number>\d{3,})_(?P<slug>[a-z0-9][a-z0-9_]*)\.sql$")

# Κλειδί του advisory lock του Postgres που σειριοποιεί τα ταυτόχρονα τρεξίματα ("ELMI").
_ADVISORY_LOCK_KEY = 0x454C4D49

# Ο πίνακας καταγραφής ζει σε δικό του MetaData: δεν ανήκει στο σχήμα της εφαρμογής (`models`),
# άρα ούτε το `create_all` ούτε το `001_init.sql` τον περιλαμβάνουν. Τον δημιουργεί ο runner.
_bookkeeping = MetaData()
schema_migrations = Table(
    SCHEMA_MIGRATIONS_TABLE,
    _bookkeeping,
    Column("version", Text, primary_key=True),
    Column("applied_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("checksum", Text, nullable=False),
)

# ----------------------------------------------------------------------------------------------
# Σφάλματα
# ----------------------------------------------------------------------------------------------


class MigrationError(Exception):
    """Αποτυχία των migrations. Το μήνυμα δεν περιέχει ποτέ το URL ή κωδικό της βάσης."""


class MigrationFileError(MigrationError):
    """Άκυρος φάκελος ή αρχείο migration (όνομα, κωδικοποίηση, διπλός αριθμός, κενό αρχείο)."""


class MigrationConsistencyError(MigrationError):
    """Η βάση και ο φάκελος δεν συμφωνούν (άλλο checksum, αρχείο που λείπει, εκτός σειράς)."""


class NotPostgresError(MigrationError):
    """Το URL δεν δείχνει σε Postgres."""


# ----------------------------------------------------------------------------------------------
# Αρχεία migrations
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Migration:
    """Ένα αρχείο migration. Το `sql` έχει αλλαγές γραμμής LF· το `version` είναι το όνομα
    του αρχείου χωρίς την κατάληξη (π.χ. `001_init`) και καταγράφεται στο `schema_migrations`."""

    version: str
    number: int
    path: Path
    sql: str
    checksum: str


@dataclass(frozen=True)
class MigrationStatus:
    """Η κατάσταση ενός migration: `applied`, `pending`, `changed` (το αρχείο διαφέρει από αυτό
    που εφαρμόστηκε) ή `missing` (εφαρμόστηκε αλλά το αρχείο δεν υπάρχει πια)."""

    version: str
    state: str
    applied_at: datetime | None
    file_checksum: str | None
    recorded_checksum: str | None


def checksum_of(sql: str) -> str:
    """SHA-256 του κειμένου, με κανονικοποιημένες αλλαγές γραμμής (το ίδιο checksum σε Windows και
    Linux ακόμη κι αν το git αλλάξει τα CRLF/LF)."""
    normalized = sql.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _as_utc(moment: datetime) -> datetime:
    """Το `moment` ως datetime UTC με ζώνη (η SQLite επιστρέφει naive ώρες UTC, ενώ το Postgres
    τις επιστρέφει στη ζώνη της συνεδρίας του server)."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _version_number(version: str) -> int:
    match = re.match(r"\d+", version)
    return int(match.group()) if match else sys.maxsize


_DOLLAR_QUOTE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")


def _skip_quoted(script: str, start: int, quote: str) -> int:
    """Η θέση μετά το κλείσιμο του εισαγωγικού που ανοίγει στο `start` (το διπλό είναι escape)."""
    position = start + 1
    while position < len(script):
        if script[position] == quote:
            if script.startswith(quote * 2, position):
                position += 2
                continue
            return position + 1
        position += 1
    return len(script)


def split_sql(script: str) -> list[str]:
    """Σπάει ένα σενάριο SQL σε statements (χωρίς το τελικό `;`).

    Τα `;` μέσα σε σχόλια (`--` και `/* */`, με φωλιασμένα), σε αλφαριθμητικά (`'...'`), σε
    εισαγωγικά ονομάτων (`"..."`) και σε dollar-quoted τμήματα (`$$...$$`, `$tag$...$tag$`, π.χ.
    σώμα `DO`) δεν χωρίζουν statements. Τα σχόλια αφαιρούνται. Τα κενά statements παραλείπονται.
    Δεν υποστηρίζονται αλφαριθμητικά `E'...'` με backslash (δεν χρησιμοποιούνται στα migrations).
    """
    statements: list[str] = []
    current: list[str] = []
    position, length = 0, len(script)

    def flush() -> None:
        statement = "".join(current).strip()
        if statement:
            statements.append(statement)
        current.clear()

    while position < length:
        char = script[position]
        if script.startswith("--", position):
            newline = script.find("\n", position)
            position = length if newline == -1 else newline
        elif script.startswith("/*", position):
            depth, position = 1, position + 2
            while position < length and depth:
                if script.startswith("/*", position):
                    depth, position = depth + 1, position + 2
                elif script.startswith("*/", position):
                    depth, position = depth - 1, position + 2
                else:
                    position += 1
            current.append(" ")
        elif char in "'\"":
            end = _skip_quoted(script, position, char)
            current.append(script[position:end])
            position = end
        elif char == "$" and not (position and re.match(r"[A-Za-z0-9_$]", script[position - 1])):
            tag = _DOLLAR_QUOTE.match(script, position)
            if tag is None:
                current.append(char)
                position += 1
                continue
            close = script.find(tag.group(), tag.end())
            end = length if close == -1 else close + len(tag.group())
            current.append(script[position:end])
            position = end
        elif char == ";":
            flush()
            position += 1
        else:
            current.append(char)
            position += 1
    flush()
    return statements


def load_migrations(directory: Path | str | None = None) -> list[Migration]:
    """Διαβάζει τα migrations του φακέλου, ταξινομημένα κατά αριθμό.

    Σηκώνει `MigrationFileError` για αρχείο `.sql` με άκυρο όνομα, για κωδικοποίηση που δεν είναι
    UTF-8 ή BOM, για κενό αρχείο (χωρίς statements) και για δύο αρχεία με τον ίδιο αριθμό.
    """
    folder = Path(directory) if directory is not None else MIGRATIONS_DIR
    if not folder.is_dir():
        raise MigrationFileError(f"migrations folder not found: {folder}")
    migrations: dict[int, Migration] = {}
    for path in sorted(folder.glob("*.sql")):
        match = _MIGRATION_NAME.match(path.name)
        if match is None:
            raise MigrationFileError(
                f"invalid migration file name {path.name!r}: expected NNN_name.sql "
                "(lowercase letters, digits and underscores)"
            )
        try:
            raw = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            raise MigrationFileError(f"{path.name} is not valid UTF-8") from None
        if raw.startswith("\ufeff"):
            raise MigrationFileError(f"{path.name} starts with a byte order mark: save it as UTF-8")
        sql = raw.replace("\r\n", "\n").replace("\r", "\n")
        if not split_sql(sql):
            raise MigrationFileError(f"{path.name} contains no SQL statements")
        number = int(match.group("number"))
        if number in migrations:
            raise MigrationFileError(
                f"migrations {migrations[number].path.name} and {path.name} have the same number"
            )
        migrations[number] = Migration(
            version=path.stem, number=number, path=path, sql=sql, checksum=checksum_of(sql)
        )
    return [migrations[number] for number in sorted(migrations)]


# ----------------------------------------------------------------------------------------------
# Παραγωγή του 001_init.sql από το db/models.py
# ----------------------------------------------------------------------------------------------

# Περιγραφή κάθε πίνακα, που γράφεται ως σχόλιο πριν από το CREATE TABLE (ένα test ελέγχει ότι
# δεν λείπει κανένας πίνακας του `models.TABLE_ORDER`).
TABLE_DESCRIPTIONS = {
    "teams": "Ομάδες: κλειδί ο κωδικός της ομάδας (π.χ. IST), όνομα το πιο πρόσφατο.",
    "players": (
        "Παίκτες: κλειδί το player_id (π.χ. P007200 ή παλαιά μορφή PADF), όνομα το πιο πρόσφατο."
    ),
    "games": (
        "Αγώνες: παιγμένοι και μελλοντικοί του προγράμματος (played = false). Η ώρα έναρξης\n"
        "tipoff_utc είναι σε UTC, χωρίς ζώνη ώρας."
    ),
    "player_games": (
        "Στατιστικά παίκτη ανά αγώνα, μαζί με τις γραμμές DNP (dnp = true, όλα 0). Τα pir και\n"
        "fantasy_score υπολογίζονται από τα στατιστικά (scoring.py)· το valuation είναι η στήλη\n"
        "Valuation του API."
    ),
    "predictions": (
        "Καταγεγραμμένες προβλέψεις του μοντέλου (python -m elfantasy.model.record_predictions):\n"
        "μία γραμμή ανά (παίκτης, αγώνας, έκδοση μοντέλου, ημέρα πρόβλεψης as_of)."
    ),
    "player_availability": (
        "Διαθεσιμότητα παικτών (τραυματισμοί, απουσίες): χειροκίνητο override, γράφεται μόνο\n"
        "από το API."
    ),
}

INIT_HEADER = """\
-- 001_init.sql: αρχικό σχήμα της βάσης (PostgreSQL / Supabase).
--
-- ΠΑΡΑΓΕΤΑΙ ΑΥΤΟΜΑΤΑ από το src/elfantasy/db/models.py με την εντολή
--     python -m elfantasy.db.migrate --print-sql
-- και ένα test ελέγχει ότι το αρχείο ισούται με την έξοδό της. Μην το επεξεργάζεσαι με το χέρι.
--
-- Μετά την εφαρμογή του σε μια βάση το αρχείο ΔΕΝ αλλάζει ποτέ: το checksum του καταγράφεται στον
-- πίνακα schema_migrations και ο runner αρνείται να συνεχίσει αν διαφέρει. Κάθε επόμενη αλλαγή του
-- σχήματος γίνεται με ΝΕΟ αρχείο (003_..., 004_...).
--
-- Τύποι: text, integer, double precision, boolean, date, timestamp (η ώρα έναρξης tipoff_utc είναι
-- σε UTC, χωρίς ζώνη) και timestamptz (χρονικές στιγμές με ζώνη).
"""


def _format_ddl(statement: object) -> str:
    """Το DDL της SQLAlchemy χωρίς tabs και κενά στο τέλος των γραμμών, με τελικό `;`."""
    lines = str(statement).strip().replace("\t", "    ").splitlines()
    return "\n".join(line.rstrip() for line in lines) + ";"


def _comment(text_: str) -> str:
    return "\n".join(f"-- {line}" for line in text_.splitlines())


def generate_init_sql() -> str:
    """Το DDL του πλήρους σχήματος (`models.metadata`) για Postgres: ΤΟ περιεχόμενο του
    `migrations/001_init.sql`. Οι πίνακες έχουν τη σειρά των foreign keys και τα indexes
    αλφαβητική σειρά ονόματος (η σειρά του `table.indexes` δεν είναι σταθερή). Το αποτέλεσμα
    είναι ντετερμινιστικό, με αλλαγές γραμμής LF."""
    dialect = postgresql.dialect()
    parts = [INIT_HEADER]
    for table in models.TABLE_ORDER:
        parts.append(_comment(TABLE_DESCRIPTIONS[table.name]))
        parts.append(_format_ddl(CreateTable(table).compile(dialect=dialect)))
        for index in sorted(table.indexes, key=lambda candidate: candidate.name or ""):
            parts.append(_format_ddl(CreateIndex(index).compile(dialect=dialect)))
        parts.append("")
    return "\n".join(parts).rstrip("\n") + "\n"


# ----------------------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------------------


def _run_statement(connection: Connection, statement: str) -> None:
    """Εκτελεί ένα statement ΩΣ ΕΧΕΙ: χωρίς παραμέτρους, άρα χωρίς ερμηνεία των χαρακτήρων `%` και
    `:` (στο psycopg ο `%` θα ήταν placeholder). Παρακάμπτει τη SQLAlchemy μόνο ως προς την
    εκτέλεση: η συναλλαγή είναι η ίδια της σύνδεσης."""
    cursor = connection.connection.cursor()
    try:
        cursor.execute(statement)
    finally:
        cursor.close()


class Migrator:
    """Εφαρμόζει τα migrations ενός φακέλου σε μια βάση, με καταγραφή στο `schema_migrations`."""

    def __init__(self, engine: Engine, directory: Path | str | None = None):
        self.engine = engine
        self.directory = Path(directory) if directory is not None else MIGRATIONS_DIR

    # ----- κατάσταση -----
    def _has_bookkeeping(self) -> bool:
        return inspect(self.engine).has_table(SCHEMA_MIGRATIONS_TABLE)

    def _recorded(self) -> dict[str, tuple[str, datetime]]:
        """Όσα έχουν εφαρμοστεί: έκδοση -> (checksum, ώρα εφαρμογής)."""
        if not self._has_bookkeeping():
            return {}
        statement = select(
            schema_migrations.c.version,
            schema_migrations.c.checksum,
            schema_migrations.c.applied_at,
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement).all()
        return {row.version: (row.checksum, _as_utc(row.applied_at)) for row in rows}

    def status(self) -> list[MigrationStatus]:
        """Η κατάσταση κάθε migration (αρχεία και εγγραφές της βάσης), κατά αριθμό. Δεν γράφει
        τίποτα στη βάση (ούτε δημιουργεί τον πίνακα καταγραφής)."""
        migrations = load_migrations(self.directory)
        recorded = self._recorded()
        statuses: list[MigrationStatus] = []
        for migration in migrations:
            if migration.version not in recorded:
                statuses.append(
                    MigrationStatus(migration.version, "pending", None, migration.checksum, None)
                )
                continue
            checksum, applied_at = recorded[migration.version]
            state = "applied" if checksum == migration.checksum else "changed"
            statuses.append(
                MigrationStatus(migration.version, state, applied_at, migration.checksum, checksum)
            )
        known = {migration.version for migration in migrations}
        for version, (checksum, applied_at) in recorded.items():
            if version not in known:
                statuses.append(MigrationStatus(version, "missing", applied_at, None, checksum))
        statuses.sort(key=lambda item: (_version_number(item.version), item.version))
        return statuses

    @staticmethod
    def problems(statuses: Sequence[MigrationStatus]) -> list[str]:
        """Οι ασυνέπειες που εμποδίζουν την εφαρμογή: αλλαγμένα ή απόντα αρχεία και migrations που
        εκκρεμούν αλλά έχουν αριθμό μικρότερο από ήδη εφαρμοσμένο (εκτός σειράς)."""
        problems: list[str] = []
        for item in statuses:
            if item.state == "changed":
                problems.append(
                    f"{item.version}: the file changed after it was applied "
                    f"(recorded checksum {(item.recorded_checksum or '')[:12]}, "
                    f"file checksum {(item.file_checksum or '')[:12]}): never edit an applied "
                    "migration, add a new one instead"
                )
            elif item.state == "missing":
                problems.append(
                    f"{item.version}: applied to the database but its file does not exist"
                )
        applied = [item for item in statuses if item.state in ("applied", "changed", "missing")]
        if applied:
            newest = max(applied, key=lambda item: _version_number(item.version))
            for item in statuses:
                if item.state == "pending" and _version_number(item.version) < _version_number(
                    newest.version
                ):
                    problems.append(
                        f"{item.version}: pending but older than the applied {newest.version}: "
                        "renumber it after the latest migration"
                    )
        return problems

    # ----- εφαρμογή -----
    def _bootstrap(self) -> None:
        """Δημιουργεί τον πίνακα καταγραφής (αν λείπει). Στο Postgres ενεργοποιεί αμέσως Row Level
        Security: ο πίνακας ζει στο public και δεν πρέπει να είναι εκτεθειμένος στο REST API."""
        with self.engine.begin() as connection:
            if connection.dialect.name == "postgresql":
                # Το ταυτόχρονο `CREATE TABLE IF NOT EXISTS` δύο διεργασιών μπορεί να αποτύχει
                # στο Postgres: το ίδιο lock με των migrations τις σειριοποιεί.
                connection.execute(
                    text("SELECT pg_advisory_xact_lock(:key)"), {"key": _ADVISORY_LOCK_KEY}
                )
            connection.execute(CreateTable(schema_migrations, if_not_exists=True))
            if connection.dialect.name == "postgresql":
                connection.execute(
                    text(f"ALTER TABLE {SCHEMA_MIGRATIONS_TABLE} ENABLE ROW LEVEL SECURITY")
                )

    def apply(self, *, dry_run: bool = False) -> list[str]:
        """Εφαρμόζει τα εκκρεμή migrations με σειρά και επιστρέφει τις εκδόσεις που εφαρμόστηκαν
        (με `dry_run`: αυτές που θα εφαρμόζονταν, χωρίς καμία εγγραφή στη βάση).

        Σηκώνει `MigrationConsistencyError` πριν εφαρμοστεί οτιδήποτε αν υπάρχει ασυνέπεια, και
        `MigrationError` (με το migration και τον αριθμό του statement) αν ένα migration αποτύχει:
        τότε η συναλλαγή του ακυρώνεται και δεν καταγράφεται.
        """
        statuses = self.status()
        problems = self.problems(statuses)
        if problems:
            raise MigrationConsistencyError("; ".join(problems))
        pending = [item.version for item in statuses if item.state == "pending"]
        if dry_run or not pending:
            return pending
        self._bootstrap()
        by_version = {migration.version: migration for migration in load_migrations(self.directory)}
        applied = []
        for version in pending:
            if self._apply_one(by_version[version]):
                applied.append(version)
        return applied

    def _apply_one(self, migration: Migration) -> bool:
        """Εφαρμόζει ένα migration σε μία συναλλαγή. False αν το πρόλαβε άλλη διεργασία."""
        statements = split_sql(migration.sql)
        started = time.perf_counter()
        logger.info("Applying %s (%d statements)", migration.version, len(statements))
        with self.engine.begin() as connection:
            if connection.dialect.name == "postgresql":
                connection.execute(
                    text("SELECT pg_advisory_xact_lock(:key)"), {"key": _ADVISORY_LOCK_KEY}
                )
            already = connection.execute(
                select(schema_migrations.c.checksum).where(
                    schema_migrations.c.version == migration.version
                )
            ).first()
            if already is not None:
                logger.info(
                    "%s was applied meanwhile by another process: skipped", migration.version
                )
                return False
            for number, statement in enumerate(statements, start=1):
                try:
                    _run_statement(connection, statement)
                except Exception as exc:
                    summary = " ".join(statement.split())[:100]
                    reason = redact_secrets(str(exc).strip(), self.engine.url)[:600]
                    raise MigrationError(
                        f"{migration.version} failed at statement {number} of {len(statements)} "
                        f"({summary}): {reason}"
                    ) from None
            connection.execute(
                insert(schema_migrations).values(
                    version=migration.version, checksum=migration.checksum
                )
            )
        logger.info("Applied %s in %.2f s", migration.version, time.perf_counter() - started)
        return True


def require_postgres(url: str) -> None:
    """Σηκώνει `NotPostgresError` αν το URL δεν δείχνει σε Postgres (ή είναι άκυρο)."""
    try:
        backend = backend_name(url)
    except DatabaseUrlError as exc:
        raise NotPostgresError(str(exc)) from None
    if backend != "postgresql":
        raise NotPostgresError(
            f"the migrations are for PostgreSQL only (this URL is {backend}): SQLite databases get "
            "their tables from create_all, which the ingestion pipeline runs automatically"
        )


# ----------------------------------------------------------------------------------------------
# Γραμμή εντολών
# ----------------------------------------------------------------------------------------------


def format_status(statuses: Sequence[MigrationStatus]) -> str:
    """Πίνακας κατάστασης για την κονσόλα."""
    if not statuses:
        return "(no migrations)"
    width = max(len(item.version) for item in statuses)
    lines = [f"{'version':<{width}}  {'state':<8}  {'applied at (UTC)':<19}  checksum"]
    for item in statuses:
        applied = "-" if item.applied_at is None else item.applied_at.strftime("%Y-%m-%d %H:%M:%S")
        checksum = (item.file_checksum or item.recorded_checksum or "")[:12]
        lines.append(f"{item.version:<{width}}  {item.state:<8}  {applied:<19}  {checksum}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m elfantasy.db.migrate",
        description="Apply the SQL migrations to a PostgreSQL database (Supabase), in order, "
        "each in one transaction, recording them in the schema_migrations table.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="database URL (default: the DATABASE_URL setting). Prefer the environment variable: "
        "a URL on the command line ends up in the shell history",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--status", action="store_true", help="show which migrations are applied or pending"
    )
    mode.add_argument(
        "--dry-run", action="store_true", help="show what would be applied, change nothing"
    )
    mode.add_argument(
        "--print-sql",
        action="store_true",
        help="print the DDL of 001_init.sql generated from db/models.py (no database needed)",
    )
    parser.add_argument(
        "--migrations-dir", default=None, help="folder of the .sql files (default: the package's)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Κωδικοί εξόδου: 0 επιτυχία, 1 αποτυχία ή ασυνέπεια, 2 άκυρο URL ή URL όχι Postgres."""
    args = build_parser().parse_args(argv)
    use_utf8_output()
    if args.print_sql:
        write_text(generate_init_sql())
        return 0
    url = args.db or get_settings().database_url
    with console_logging():
        try:
            require_postgres(url)
        except NotPostgresError as exc:
            logger.error("%s", exc)
            return 2
        logger.info("Database: %s", safe_url(url))
        try:
            engine = get_engine(url)
        except DatabaseUrlError as exc:
            logger.error("%s", exc)
            return 2
        try:
            migrator = Migrator(engine, args.migrations_dir)
            if args.status:
                statuses = migrator.status()
                print(format_status(statuses))
                problems = migrator.problems(statuses)
                for problem in problems:
                    logger.error("%s", problem)
                return 1 if problems else 0
            applied = migrator.apply(dry_run=args.dry_run)
            if args.dry_run:
                print("Would apply: " + (", ".join(applied) if applied else "nothing (up to date)"))
            else:
                print("Applied: " + (", ".join(applied) if applied else "nothing (up to date)"))
            return 0
        except MigrationError as exc:
            logger.error("%s", redact_secrets(str(exc), url))
            return 1
        except Exception as exc:
            report_failure("migration failed", exc, url)
            return 1
        finally:
            engine.dispose()


if __name__ == "__main__":
    sys.exit(main())
