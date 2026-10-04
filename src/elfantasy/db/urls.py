"""Βοηθητικά για URL βάσης: κανονικοποίηση, ασφαλής εμφάνιση χωρίς κωδικό και σύγκριση.

Τα URL βάσης περιέχουν τον κωδικό της βάσης. Οι κανόνες του project είναι:

* Ό,τι εμφανίζεται σε log, μήνυμα σφάλματος ή έξοδο εντολής περνά από το `safe_url` (κωδικός
  `***`) ή από το `redact_secrets` (αντικατάσταση του κωδικού μέσα σε οποιοδήποτε κείμενο).
* Τα σφάλματα ανάγνωσης URL (`DatabaseUrlError`) δεν περιέχουν ποτέ το URL ή μέρος του.
* Το URL δεν γράφεται ποτέ σε αρχείο από τον κώδικα του project.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import unquote

from sqlalchemy.engine import URL, make_url

# Ελάχιστο μήκος κωδικού για αντικατάσταση σε κείμενο: ένας πολύ μικρός κωδικός (π.χ. «a»)
# θα κατέστρεφε το κείμενο του μηνύματος χωρίς πραγματικό όφελος.
_MIN_REDACTED_LENGTH = 3

# Το `user:password@` ενός URL, ακόμη και όταν η SQLAlchemy δεν μπορεί να το διαβάσει.
_PASSWORD_IN_URL = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^:/@]*:([^@]*)@")

_DEFAULT_POSTGRES_PORT = 5432

# Παράμετροι του query (`?password=...`) που κρύβουν κωδικό. Το libpq δέχεται τον κωδικό και ως
# παράμετρο του URL, και η SQLAlchemy τον προωθεί αυτούσιο: το `hide_password` δεν τον πιάνει.
_SECRET_QUERY_KEYS = frozenset({"password", "sslpassword", "passphrase"})


class DatabaseUrlError(ValueError):
    """Άκυρο URL βάσης. Το μήνυμα δεν περιέχει ποτέ το URL (μπορεί να έχει κωδικό)."""


def normalize_database_url(url: str) -> str:
    """Προσθέτει τον driver `psycopg` (v3) σε URLs Postgres που δεν δηλώνουν driver.

    Η υπηρεσία Supabase δίνει URL της μορφής `postgresql://...` (ή `postgres://...`), αλλά το
    project χρησιμοποιεί το `psycopg` 3 και όχι το `psycopg2` που θα επέλεγε η SQLAlchemy.
    """
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def parse_database_url(url: str) -> URL:
    """Διαβάζει το URL (μετά την κανονικοποίηση του driver) ή σηκώνει `DatabaseUrlError`.

    Η αρχική εξαίρεση της SQLAlchemy δεν διατηρείται (`from None`): ορισμένα μηνύματά της
    περιέχουν τμήματα του URL, άρα και πιθανώς του κωδικού.
    """
    try:
        return make_url(normalize_database_url(url))
    except Exception:
        raise DatabaseUrlError(
            "invalid database URL (details are hidden because a URL may contain a password); "
            "expected e.g. postgresql://USER:PASSWORD@HOST:5432/postgres or sqlite:///path.db"
        ) from None


def safe_url(url: str | URL | None) -> str:
    """Το URL με κρυμμένο κωδικό, για logs και μηνύματα. Δεν σηκώνει ποτέ εξαίρεση."""
    if url is None:
        return "<none>"
    try:
        parsed = url if isinstance(url, URL) else make_url(normalize_database_url(url))
        return (
            _mask_query_secrets(parsed)
            .render_as_string(hide_password=True)
            .replace("%2A%2A%2A", "***")
        )
    except Exception:
        return "<invalid database URL>"


def _query_secrets(parsed: URL) -> set[str]:
    """Οι τιμές κωδικού που βρίσκονται στο query του URL (π.χ. `?password=...`)."""
    found: set[str] = set()
    for key, value in parsed.query.items():
        if key.lower() in _SECRET_QUERY_KEYS:
            found.update(value if isinstance(value, tuple) else (value,))
    return {item for item in found if item}


def _mask_query_secrets(parsed: URL) -> URL:
    """Αντιγράφει το URL με `***` στις παραμέτρους query που κρύβουν κωδικό."""
    if not _query_secrets(parsed):
        return parsed
    query = {
        key: (("***",) * len(value) if isinstance(value, tuple) else "***")
        if key.lower() in _SECRET_QUERY_KEYS
        else value
        for key, value in parsed.query.items()
    }
    return parsed.set(query=query)


def backend_name(url: str | URL) -> str:
    """Το όνομα του backend του URL: `sqlite`, `postgresql` κ.λπ. (σηκώνει `DatabaseUrlError`)."""
    parsed = url if isinstance(url, URL) else parse_database_url(url)
    return parsed.get_backend_name()


def sqlite_database_path(url: str | URL) -> Path | None:
    """Η διαδρομή του αρχείου SQLite, ή None για URL που δεν είναι αρχείο (μνήμη, `file:`)."""
    parsed = url if isinstance(url, URL) else parse_database_url(url)
    if parsed.get_backend_name() != "sqlite":
        return None
    database = parsed.database
    if not database or database == ":memory:" or database.startswith("file:"):
        return None
    return Path(database)


def sqlite_file_is_missing(url: str) -> bool:
    """True αν το URL δείχνει σε αρχείο SQLite που δεν υπάρχει.

    Η SQLite θα δημιουργούσε σιωπηλά ένα κενό αρχείο στην πρώτη σύνδεση· εδώ θέλουμε αντί γι'
    αυτό «degraded» κατάσταση (API) ή σαφές σφάλμα (εργαλεία) και καμία παρενέργεια στον δίσκο.
    """
    try:
        path = sqlite_database_path(url)
    except DatabaseUrlError:  # άκυρο URL: θα αναφερθεί από το get_engine
        return False
    return path is not None and not path.is_file()


def same_database(first: str, second: str) -> bool:
    """True αν τα δύο URL δείχνουν στην ίδια βάση (best effort).

    SQLite: το ίδιο αρχείο (μετά από resolve). Postgres: ίδιος host, port, βάση και χρήστης,
    ανεξάρτητα από driver, κωδικό και παραμέτρους. Βάσεις στη μνήμη θεωρούνται πάντα διαφορετικές.
    Δεν μπορεί να εντοπίσει διαφορετικά URL που καταλήγουν στην ίδια βάση (π.χ. direct και pooler
    του Supabase).
    """
    a, b = parse_database_url(first), parse_database_url(second)
    if a.get_backend_name() != b.get_backend_name():
        return False
    if a.get_backend_name() == "sqlite":
        path_a, path_b = sqlite_database_path(a), sqlite_database_path(b)
        if path_a is None or path_b is None:
            return False
        return os.path.normcase(str(path_a.resolve())) == os.path.normcase(str(path_b.resolve()))
    return (
        (a.host or "").lower() == (b.host or "").lower()
        and (a.port or _DEFAULT_POSTGRES_PORT) == (b.port or _DEFAULT_POSTGRES_PORT)
        and (a.database or "") == (b.database or "")
        and (a.username or "") == (b.username or "")
    )


def _password_candidates(url: str) -> set[str]:
    """Οι τιμές κωδικού που μπορεί να εμφανιστούν σε κείμενο: ο κωδικός όπως τον διαβάζει η
    SQLAlchemy και όπως γράφτηκε (με percent-encoding) στο URL."""
    candidates: set[str] = set()
    try:
        parsed = make_url(normalize_database_url(url))
        password = parsed.password
        candidates |= _query_secrets(parsed)
    except Exception:
        password = None
    if password:
        candidates.add(password)
    match = _PASSWORD_IN_URL.match(url.strip())
    if match:
        raw = match.group(1)
        candidates.update({raw, unquote(raw)})
    return {value for value in candidates if len(value) >= _MIN_REDACTED_LENGTH}


def redact_secrets(text: str, *urls: str | URL | None) -> str:
    """Αντικαθιστά με `***` τους κωδικούς των `urls` (και κάθε `user:password@` URL) στο `text`.

    Χρησιμοποιείται για κάθε μήνυμα εξαίρεσης που γράφεται σε log ή τερματικό: τα μηνύματα
    των drivers ή της SQLAlchemy μπορεί, σε σπάνιες περιπτώσεις, να περιέχουν τμήματα του URL.
    """
    secrets: set[str] = set()
    for url in urls:
        if url is None:
            continue
        if isinstance(url, URL):
            if url.password and len(url.password) >= _MIN_REDACTED_LENGTH:
                secrets.add(url.password)
            secrets |= {s for s in _query_secrets(url) if len(s) >= _MIN_REDACTED_LENGTH}
            continue
        secrets |= _password_candidates(url)
    # Πρώτα τα μεγαλύτερα, ώστε ένας κωδικός που περιέχει έναν άλλον να αντικαθίσταται ολόκληρος.
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, "***")
    # Οποιοδήποτε URL με κωδικό που τυχόν υπάρχει στο κείμενο.
    return re.sub(r"(://[^:/@\s]*:)[^@\s]+(@)", r"\1***\2", text)
