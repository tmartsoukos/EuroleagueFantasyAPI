"""Βοηθητικά για tests που χρειάζονται ΠΡΑΓΜΑΤΙΚΟ Postgres (Φάση 5): ΜΟΝΟ τοπικός server.

Πηγές server, με σειρά:

1. `TEST_DATABASE_URL`: URL ενός ΤΟΠΙΚΟΥ server Postgres (localhost, 127.0.0.1, ::1 ή socket unix).
   Ο ρόλος πρέπει να μπορεί να δημιουργεί βάσεις. URL προς άλλον υπολογιστή ΑΠΟΡΡΙΠΤΕΤΑΙ (το test
   παραλείπεται): κανένα test δεν συνδέεται ποτέ σε remote βάση, π.χ. στο Supabase.
2. `pixeltable-pgserver`: ενσωματωμένος Postgres με binaries, που ξεκινά σε προσωρινό φάκελο και
   σταματά στο τέλος της συνεδρίας (αρκεί `pip install -r requirements-dev.txt`).

Αν δεν υπάρχει καμία πηγή, τα tests παραλείπονται, εκτός αν οριστεί `ELFANTASY_REQUIRE_POSTGRES=1`
(π.χ. στο CI): τότε αποτυγχάνουν, ώστε να μη χάνεται σιωπηλά η κάλυψη.

Κάθε test παίρνει δική του, κενή βάση (`CREATE DATABASE`), που διαγράφεται στο τέλος.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from elfantasy.db.urls import normalize_database_url

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@dataclass(frozen=True)
class PostgresServer:
    """Ένας τοπικός server Postgres: το URL διαχείρισης (βάση `postgres`) και αν είναι ο
    ενσωματωμένος (προσωρινός), οπότε επιτρέπεται η δημιουργία ρόλων σε επίπεδο server."""

    admin_url: str
    embedded: bool


def is_local_url(url: str) -> bool:
    """True αν το URL δείχνει ΜΟΝΟ σε τοπικό server (loopback ή socket unix).

    Εξετάζονται ο host του URL και οι παράμετροι `host` και `hostaddr` (το libpq τις δέχεται και
    με κενό host στο URL, π.χ. `postgresql://u@/db?host=remote.example.com`), καθώς και οι λίστες
    με κόμμα. Άκυρο URL θεωρείται μη τοπικό."""
    try:
        parsed = make_url(normalize_database_url(url))
    except Exception:
        return False
    candidates = [parsed.host or ""]
    for key in ("host", "hostaddr"):
        value = parsed.query.get(key)
        candidates.extend([value] if isinstance(value, str) else list(value or ()))
    hosts = [part.strip() for candidate in candidates for part in candidate.split(",")]
    return all(host in LOCAL_HOSTS or host == "" or host.startswith("/") for host in hosts)


def with_database(admin_url: str, name: str) -> str:
    """Το URL του server με άλλη βάση."""
    return make_url(admin_url).set(database=name).render_as_string(hide_password=False)


def with_query(url: str, **parameters: str) -> str:
    """Το URL με επιπλέον παραμέτρους (π.χ. `options`, `sslmode`). Σε server με socket unix το URL
    έχει ήδη παράμετρο (`?host=/tmp/…`): γι' αυτό δεν αρκεί η προσθήκη κειμένου `?…`."""
    return make_url(url).update_query_dict(parameters).render_as_string(hide_password=False)


def create_database(server: PostgresServer, prefix: str = "elfantasy_test") -> str:
    """Δημιουργεί μια κενή βάση και επιστρέφει το URL της."""
    name = f"{prefix}_{uuid.uuid4().hex[:12]}"
    admin = create_engine(server.admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        admin.dispose()
    return with_database(server.admin_url, name)


def drop_database(server: PostgresServer, url: str) -> None:
    """Διαγράφει τη βάση (και κλείνει τυχόν ανοιχτές συνδέσεις της)."""
    name = make_url(url).database
    admin = create_engine(server.admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    finally:
        admin.dispose()


def require_postgres() -> bool:
    """True όταν η απουσία Postgres πρέπει να αποτύχει αντί να παραλείψει τα tests."""
    return os.environ.get("ELFANTASY_REQUIRE_POSTGRES", "") not in ("", "0")
