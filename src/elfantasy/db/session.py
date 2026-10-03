"""Σύνδεση στη βάση (engine), έλεγχος σχήματος και idempotent upsert.

Η ίδια λογική δουλεύει για SQLite (τοπικά, tests) και για Postgres (Supabase, παραγωγή): το
`upsert` επιλέγει το σωστό `insert ... on conflict do update` ανάλογα με το dialect της σύνδεσης.

Διαχείριση του σχήματος ανά βάση (docs/DATABASE.md):

* SQLite: `create_all` (και ο πίνακας `player_availability` δημιουργείται από το API αν λείπει).
* Postgres: ΜΟΝΟ από τα migrations (`python -m elfantasy.db.migrate`). Ο κώδικας του project δεν
  δημιουργεί ποτέ πίνακες σε Postgres, γιατί ένας πίνακας που δημιουργήθηκε εκτός migrations δεν
  έχει Row Level Security και θα ήταν εκτεθειμένος στο REST API του Supabase.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from sqlalchemy import Engine, Table, create_engine, event, inspect, text
from sqlalchemy.engine import URL, Connection
from sqlalchemy.exc import NoSuchTableError

from elfantasy.config import get_settings
from elfantasy.db.models import metadata, predictions
from elfantasy.db.urls import DatabaseUrlError, parse_database_url, sqlite_database_path
from elfantasy.db.urls import normalize_database_url as normalize_database_url

logger = logging.getLogger(__name__)

# Πόσες γραμμές στέλνονται ανά εκτέλεση. Το executemany δεν έχει όριο παραμέτρων ανά εντολή,
# το chunk περιορίζει μόνο τη μνήμη.
UPSERT_CHUNK_SIZE = 5000

# Ρυθμίσεις συνδέσεων Postgres. Το free tier του Supabase έχει λίγες συνδέσεις (και ο pooler έχει
# δικό του όριο), γι' αυτό το pool είναι μικρό: έως 5 + 5 συνδέσεις ανά διεργασία. Το
# `pool_recycle` κλείνει συνδέσεις που ο pooler ή ο firewall θα είχε ήδη κόψει σε αδράνεια, και το
# `pool_pre_ping` ελέγχει κάθε σύνδεση πριν τη χρήση (αντί να αποτύχει το αίτημα του χρήστη).
POSTGRES_POOL_SIZE = 5
POSTGRES_MAX_OVERFLOW = 5
POSTGRES_POOL_RECYCLE = 300  # δευτερόλεπτα
# Χωρίς χρονικό όριο το libpq περιμένει ώσπου να λήξει το TCP (πάνω από 1 λεπτό): μια παυμένη
# βάση (free tier) θα κρατούσε το startup του API «κολλημένο». Το όριο ισχύει μόνο αν το URL δεν
# δηλώνει δικό του `connect_timeout`.
POSTGRES_CONNECT_TIMEOUT = 10  # δευτερόλεπτα

# Ο transaction pooler του Supabase (Supavisor, πόρτα 6543) δεν υποστηρίζει prepared statements:
# ο driver psycopg 3 πρέπει να τα απενεργοποιεί (`prepare_threshold=None`).
TRANSACTION_POOLER_PORT = 6543
# Παράμετρος του URL που διαβάζει το get_engine (δεν περνά στον driver): `none` (ή `off`) για
# απενεργοποίηση των prepared statements, αλλιώς ακέραιος όπως στο psycopg.
PREPARE_THRESHOLD_PARAM = "prepare_threshold"
_PREPARE_DISABLED_WORDS = frozenset({"none", "off", "disable", "disabled", "false", "no"})


class SchemaNotInitialisedError(NoSuchTableError):
    """Λείπουν πίνακες από μια βάση Postgres: δεν έχουν εφαρμοστεί τα migrations.

    Κληρονομεί από την `NoSuchTableError` της SQLAlchemy, άρα το API την αντιμετωπίζει σαν κάθε
    άλλο σφάλμα βάσης (degraded κατάσταση, καθαρό μήνυμα στους clients).
    """


class LegacySchemaError(RuntimeError):
    """Η τοπική βάση SQLite έχει πίνακα `predictions` της παλιάς μορφής (πριν τη Φάση 5) με γραμμές.

    Ο πίνακας αντικαθίσταται αυτόματα μόνο όταν είναι κενός (κανένας κώδικας του project δεν τον
    είχε γράψει ποτέ)· αν περιέχει δεδομένα, ο χρήστης αποφασίζει τι θα γίνουν.
    """


def _parse_prepare_threshold(raw: str) -> int | None:
    value = raw.strip().lower()
    if value in _PREPARE_DISABLED_WORDS:
        return None
    try:
        number = int(value)
    except ValueError:
        number = -1
    if number < 0:
        raise DatabaseUrlError(
            f"invalid {PREPARE_THRESHOLD_PARAM} value {raw!r}: use 'none' or a non-negative integer"
        )
    return number


def postgres_engine_options(sa_url: URL) -> tuple[URL, dict[str, Any]]:
    """Οι ρυθμίσεις του engine για Postgres: επιστρέφει (URL χωρίς τις δικές μας παραμέτρους,
    ορίσματα του `create_engine`).

    * `prepare_threshold=None` στα connect args όταν η πόρτα είναι 6543 (transaction pooler) ή το
      URL έχει την παράμετρο `prepare_threshold` (τιμή `none` ή ακέραιος). Η παράμετρος
      αφαιρείται από το URL: αλλιώς η SQLAlchemy θα την έδινε στον driver ως κείμενο.
    * Το `sslmode` του URL περνά αυτούσιο στο libpq (το Supabase υποστηρίζει SSL και τεκμηριώνει
      τη σύσταση να χρησιμοποιείται· docs/DATABASE.md).
    * `connect_timeout` 10 s, αν το URL δεν ορίζει δικό του.
    """
    query = dict(sa_url.query)
    connect_args: dict[str, Any] = {}

    raw = query.pop(PREPARE_THRESHOLD_PARAM, None)
    if isinstance(raw, tuple):  # η παράμετρος δόθηκε πολλές φορές: ισχύει η τελευταία
        raw = raw[-1]
    if raw is not None:
        connect_args["prepare_threshold"] = _parse_prepare_threshold(raw)
        sa_url = sa_url.difference_update_query([PREPARE_THRESHOLD_PARAM])
    elif sa_url.port == TRANSACTION_POOLER_PORT:
        connect_args["prepare_threshold"] = None
    if "connect_timeout" not in query:
        connect_args["connect_timeout"] = POSTGRES_CONNECT_TIMEOUT

    options = {
        "pool_pre_ping": True,
        "pool_size": POSTGRES_POOL_SIZE,
        "max_overflow": POSTGRES_MAX_OVERFLOW,
        "pool_recycle": POSTGRES_POOL_RECYCLE,
        "connect_args": connect_args,
    }
    return sa_url, options


def get_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """Δημιουργεί engine από το `url` ή από το `DATABASE_URL` των ρυθμίσεων.

    Για αρχείο SQLite δημιουργεί τον φάκελο του αρχείου (π.χ. `data/`) αν δεν υπάρχει. Στο
    SQLite ενεργοποιούνται τα foreign keys (το Postgres τα επιβάλλει πάντα), ώστε λάθη στη
    σειρά εισαγωγής των πινάκων να φαίνονται και στην τοπική βάση.

    Για Postgres ισχύουν οι ρυθμίσεις του `postgres_engine_options` (μικρό pool, pre-ping,
    recycle, prepared statements εκτός λειτουργίας στον transaction pooler). Η δημιουργία του
    engine δεν συνδέεται στη βάση. Ένα άκυρο URL σηκώνει `DatabaseUrlError` χωρίς το URL στο
    μήνυμα· ο κωδικός δεν εμφανίζεται ποτέ σε log (το `repr` του engine τον κρύβει).
    """
    sa_url = parse_database_url(url or get_settings().database_url)
    backend = sa_url.get_backend_name()
    options: dict[str, Any] = {}
    if backend == "sqlite":
        path = sqlite_database_path(sa_url)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    elif backend == "postgresql":
        sa_url, options = postgres_engine_options(sa_url)

    engine = create_engine(sa_url, echo=echo, **options)
    if backend == "sqlite":

        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    elif backend == "postgresql":
        event.listen(engine, "connect", set_full_precision_floats)

    return engine


def set_full_precision_floats(dbapi_connection, _connection_record=None) -> None:
    """Ορίζει `extra_float_digits = 3` στη συνεδρία, ώστε τα `double precision` να διαβάζονται
    χωρίς απώλεια ακρίβειας.

    Το Supabase ορίζει `extra_float_digits = 0` στον server: τα floats στέλνονται ως κείμενο με 15
    σημαντικά ψηφία, π.χ. 19.983333333333334 → 19.9833333333333. Η αποθηκευμένη τιμή είναι
    σωστή, αλλά η ανάγνωση την αλλοιώνει (και η επαλήθευση της μεταφοράς δεν μπορεί να ταυτίσει το
    περιεχόμενο). Από το Postgres 12 η τιμή 3 δίνει τη συντομότερη δεκαδική μορφή που ξαναδιαβάζεται
    ακριβώς. Το SET γίνεται commit, αλλιώς η επαναφορά (rollback) του pool στο τέλος της πρώτης
    συναλλαγής θα το ακύρωνε. Σε transaction pooler (πόρτα 6543) η ρύθμιση συνεδρίας δεν είναι
    εγγυημένη: γι' αυτό προτιμάται ο session pooler (docs/DATABASE.md).
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("SET extra_float_digits = 3")
    finally:
        cursor.close()
    dbapi_connection.commit()


def create_all(engine: Engine) -> None:
    """Δημιουργεί όλους τους πίνακες και τα indexes που λείπουν (ασφαλές να ξανατρέξει).

    Μόνο για SQLite και tests. Στο Postgres το σχήμα το διαχειρίζονται τα migrations (βλ.
    `ensure_schema`): η κλήση σηκώνει `RuntimeError`, γιατί ένας πίνακας που δημιουργείται εκτός
    migrations δεν έχει Row Level Security.
    """
    if engine.dialect.name == "postgresql":
        raise RuntimeError(
            "create_all is not allowed on PostgreSQL: the schema is created by the migrations "
            "(python -m elfantasy.db.migrate)"
        )
    metadata.create_all(engine)


def schema_is_managed_by_migrations(engine: Engine) -> bool:
    """True αν η βάση έχει τον πίνακα `schema_migrations` (εφαρμόστηκαν migrations)."""
    return inspect(engine).has_table("schema_migrations")


def missing_tables(engine: Engine, tables: Iterable[Table]) -> list[str]:
    """Τα ονόματα των πινάκων που λείπουν από τη βάση."""
    inspector = inspect(engine)
    return [table.name for table in tables if not inspector.has_table(table.name)]


def _replace_empty_legacy_predictions(engine: Engine) -> bool:
    """Αντικαθιστά τον κενό πίνακα `predictions` της παλιάς μορφής (χωρίς τη στήλη `as_of`).

    Οι τοπικές βάσεις SQLite που δημιουργήθηκαν στις Φάσεις 2 έως 4 έχουν έναν κενό πίνακα
    `predictions` με παλιότερη διάταξη, που ο κώδικας του project δεν έγραψε ποτέ. Το `create_all`
    δεν αλλάζει υπάρχοντες πίνακες, άρα ο πίνακας διαγράφεται (μόνο όταν είναι κενός) και
    ξαναδημιουργείται με τη νέα διάταξη. Επιστρέφει True αν έγινε αντικατάσταση. Αν ο παλιός πίνακας
    έχει γραμμές σηκώνει `LegacySchemaError`, χωρίς να αγγίξει τίποτα.
    """
    inspector = inspect(engine)
    if not inspector.has_table(predictions.name):
        return False
    if "as_of" in {column["name"] for column in inspector.get_columns(predictions.name)}:
        return False
    with engine.begin() as connection:
        rows = connection.execute(text("SELECT COUNT(*) FROM predictions")).scalar_one()
        if rows:
            raise LegacySchemaError(
                f"the predictions table has the old layout (no as_of column) and contains {rows} "
                "rows: move or drop it manually before continuing"
            )
        connection.execute(text("DROP TABLE predictions"))
    logger.info("replaced the empty predictions table of the old layout")
    return True


def ensure_schema(engine: Engine, tables: Iterable[Table] | None = None) -> str:
    """Σιγουρεύει ότι υπάρχουν οι πίνακες `tables` (προεπιλογή: όλοι οι πίνακες του project).

    * SQLite: δημιουργεί όσους λείπουν (`create_all` με checkfirst) και επιστρέφει `"created"`.
      Ένας κενός πίνακας `predictions` της παλιάς μορφής αντικαθίσταται (βλ. παραπάνω).
    * Postgres: ΔΕΝ δημιουργεί ποτέ τίποτα. Αν υπάρχουν όλοι, επιστρέφει `"migrations"` (η βάση
      έχει `schema_migrations`) ή `"existing"` (οι πίνακες υπάρχουν αλλά η βάση δεν έχει
      `schema_migrations`). Αν λείπει κάποιος, σηκώνει `SchemaNotInitialisedError`.

    Δεν χρειάζεται δικαίωμα CREATE σε Postgres: μόνο ανάγνωση των καταλόγων του συστήματος.
    """
    wanted = list(tables) if tables is not None else list(metadata.sorted_tables)
    if engine.dialect.name != "postgresql":
        if predictions in wanted:
            _replace_empty_legacy_predictions(engine)
        metadata.create_all(engine, tables=wanted)
        return "created"
    missing = missing_tables(engine, wanted)
    if missing:
        raise SchemaNotInitialisedError(
            f"database tables are missing: {', '.join(missing)}; "
            "apply the migrations with: python -m elfantasy.db.migrate"
        )
    return "migrations" if schema_is_managed_by_migrations(engine) else "existing"


def _dialect_insert(connection: Connection):
    """Επιστρέφει το `insert` του dialect της σύνδεσης (με υποστήριξη on_conflict)."""
    name = connection.dialect.name
    if name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    elif name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        raise NotImplementedError(f"Το upsert δεν υποστηρίζεται για το dialect {name!r}")
    return insert


def upsert(
    conn: Connection,
    table: Table,
    rows: Sequence[Mapping[str, Any]],
    index_elements: Iterable[str],
    *,
    set_: Mapping[str, Any] | Callable[[Any], Mapping[str, Any]] | None = None,
) -> int:
    """Εισάγει τις γραμμές και ενημερώνει όσες υπάρχουν ήδη με το ίδιο κλειδί (idempotent).

    - `index_elements`: οι στήλες του κλειδιού (primary key ή unique) που ορίζουν τη σύγκρουση.
    - `set_`: προαιρετικά, ρητές εκφράσεις ενημέρωσης ανά στήλη (π.χ. με `case` για να μη
      μικραίνει το `last_season`). Μπορεί να είναι dict ή συνάρτηση που παίρνει το `excluded`
      (οι τιμές της νέας γραμμής) και επιστρέφει dict. Από προεπιλογή ενημερώνονται όλες οι
      στήλες εκτός κλειδιού με τις νέες τιμές. Με κενό `set_` οι υπάρχουσες γραμμές μένουν ως έχουν.

    Δουλεύει σε μία συναλλαγή της σύνδεσης που δίνει ο καλών. Επιστρέφει το πλήθος των γραμμών
    που στάλθηκαν (όχι το πλήθος των νέων).
    """
    if not rows:
        return 0
    key_columns = list(index_elements)
    insert = _dialect_insert(conn)
    statement = insert(table)
    if set_ is None:
        update = {
            column.name: statement.excluded[column.name]
            for column in table.columns
            if column.name not in key_columns
        }
    elif callable(set_):
        update = dict(set_(statement.excluded))
    else:
        update = dict(set_)
    if update:
        statement = statement.on_conflict_do_update(index_elements=key_columns, set_=update)
    else:
        statement = statement.on_conflict_do_nothing(index_elements=key_columns)

    for start in range(0, len(rows), UPSERT_CHUNK_SIZE):
        conn.execute(statement, list(rows[start : start + UPSERT_CHUNK_SIZE]))
    return len(rows)
