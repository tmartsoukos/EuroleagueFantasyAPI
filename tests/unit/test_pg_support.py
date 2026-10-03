"""Tests των βοηθητικών για πραγματικό Postgres (tests/pg_support.py): ΠΟΤΕ remote βάση.

Το `TEST_DATABASE_URL` γίνεται δεκτό μόνο αν δείχνει σε τοπικό server. Ο έλεγχος αυτός προστατεύει
από τυχαία εκτέλεση των tests (που δημιουργούν και διαγράφουν βάσεις) πάνω σε πραγματική βάση.
"""

import pytest
from pg_support import is_local_url, with_query


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://postgres@localhost:5432/postgres",
        "postgresql://postgres:secret@127.0.0.1:5432/postgres",
        "postgres://postgres@[::1]:5432/postgres",
        "postgresql+psycopg://postgres@localhost/db?sslmode=disable",
        "postgresql://postgres@/postgres?host=/tmp/pgsock",  # socket unix
        "postgresql://postgres@/postgres",  # προεπιλεγμένο socket
        "postgresql://postgres@/postgres?host=localhost,127.0.0.1",
    ],
)
def test_local_urls_are_accepted(url):
    assert is_local_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://postgres:secret@aws-0-eu.pooler.supabase.com:5432/postgres",
        "postgresql://postgres@db.abcdef.supabase.co:5432/postgres",
        "postgresql://postgres@10.0.0.5:5432/postgres",
        "postgresql://postgres@localhost.example.com/postgres",
        "postgresql://postgres@/postgres?host=remote.example.com",  # host μόνο στις παραμέτρους
        "postgresql://postgres@/postgres?hostaddr=203.0.113.7",
        "postgresql://postgres@localhost/postgres?host=remote.example.com",
        "postgresql://postgres@/postgres?host=localhost,remote.example.com",
        "not a url",
        "",
    ],
)
def test_remote_and_invalid_urls_are_rejected(url):
    assert not is_local_url(url)


def test_with_query_adds_parameters_to_urls_with_and_without_a_query():
    assert with_query("postgresql://u@h/db", options="-c TimeZone=UTC").endswith(
        "?options=-c+TimeZone%3DUTC"
    )
    url = with_query("postgresql://u@/db?host=/tmp/sock", sslmode="disable")
    assert url.count("?") == 1 and "host=%2Ftmp%2Fsock" in url and "sslmode=disable" in url
