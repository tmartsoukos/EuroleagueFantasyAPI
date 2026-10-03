"""Σύνδεση στη βάση (engine), δημιουργία πινάκων και idempotent upsert.

Η ίδια λογική δουλεύει για SQLite και για Postgres: το `upsert` επιλέγει το σωστό
`insert ... on conflict do update` ανάλογα με το dialect της σύνδεσης.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, Table, create_engine, event
from sqlalchemy.engine import Connection, make_url

from elfantasy.config import get_settings
from elfantasy.db.models import metadata

# Πόσες γραμμές στέλνονται ανά εκτέλεση. Το executemany δεν έχει όριο παραμέτρων ανά εντολή,
# το chunk περιορίζει μόνο τη μνήμη.
UPSERT_CHUNK_SIZE = 5000


def normalize_database_url(url: str) -> str:
    """Προσθέτει τον driver `psycopg` (v3) σε URLs Postgres που δεν δηλώνουν driver.

    Η υπηρεσία Supabase δίνει URL της μορφής `postgresql://...` (ή `postgres://...`), αλλά το
    project χρησιμοποιεί το `psycopg` 3 και όχι το `psycopg2` που θα επέλεγε η SQLAlchemy.
    """
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def get_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """Δημιουργεί engine από το `url` ή από το `DATABASE_URL` των ρυθμίσεων.

    Για αρχείο SQLite δημιουργεί τον φάκελο του αρχείου (π.χ. `data/`) αν δεν υπάρχει. Στο
    SQLite ενεργοποιούνται τα foreign keys (το Postgres τα επιβάλλει πάντα), ώστε λάθη στη
    σειρά εισαγωγής των πινάκων να φαίνονται και στην τοπική βάση.
    """
    sa_url = make_url(normalize_database_url(url or get_settings().database_url))
    is_sqlite = sa_url.get_backend_name() == "sqlite"
    if is_sqlite:
        database = sa_url.database
        if database and database != ":memory:" and not database.startswith("file:"):
            Path(database).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(sa_url, echo=echo)
    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_all(engine: Engine) -> None:
    """Δημιουργεί όλους τους πίνακες και τα indexes που λείπουν (ασφαλές να ξανατρέξει)."""
    metadata.create_all(engine)


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
