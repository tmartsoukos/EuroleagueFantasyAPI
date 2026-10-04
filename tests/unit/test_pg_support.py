"""Tests των βοηθητικών για πραγματικό Postgres (tests/pg_support.py): ΠΟΤΕ remote βάση.

Το `TEST_DATABASE_URL` γίνεται δεκτό μόνο αν δείχνει σε τοπικό server. Ο έλεγχος αυτός προστατεύει
από τυχαία εκτέλεση των tests (που δημιουργούν και διαγράφουν βάσεις) πάνω σε πραγματική βάση.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pg_support import (
    is_disposable,
    is_forbidden_postgres_skip,
    is_local_url,
    require_postgres,
    with_query,
)

TESTS_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = TESTS_DIR.parent / "src"


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


# --------------------------------------------------------------------------------------
# Φάση 6: CI. Προσωρινός server και απαγόρευση των σιωπηλών παραλείψεων
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "enabled"),
    [(None, False), ("", False), ("0", False), ("1", True), ("true", True), ("yes", True)],
)
@pytest.mark.parametrize(
    ("variable", "function"),
    [
        ("ELFANTASY_PG_DISPOSABLE", is_disposable),
        ("ELFANTASY_REQUIRE_POSTGRES", require_postgres),
    ],
)
def test_the_opt_in_variables(monkeypatch, variable, function, value, enabled):
    monkeypatch.delenv(variable, raising=False)
    if value is not None:
        monkeypatch.setenv(variable, value)
    assert function() is enabled


@pytest.mark.parametrize(
    ("require", "skipped", "keywords", "forbidden"),
    [
        ("1", True, {"postgres": 1, "test_x": 1}, True),
        ("1", True, {"unit": 1}, False),  # παράλειψη test χωρίς το marker: επιτρέπεται
        ("1", False, {"postgres": 1}, False),  # δεν παραλείφθηκε
        ("0", True, {"postgres": 1}, False),  # χωρίς ELFANTASY_REQUIRE_POSTGRES επιτρέπεται
        ("", True, {"postgres": 1}, False),
    ],
)
def test_which_skips_are_forbidden(monkeypatch, require, skipped, keywords, forbidden):
    monkeypatch.setenv("ELFANTASY_REQUIRE_POSTGRES", require)
    report = SimpleNamespace(skipped=skipped, keywords=keywords)
    assert is_forbidden_postgres_skip(report) is forbidden


SAMPLE_CONFTEST = """\
from pg_support import ForbidSilentPostgresSkips


def pytest_configure(config):
    config.pluginmanager.register(ForbidSilentPostgresSkips(), "forbid-silent-postgres-skips")
"""

SAMPLE_TESTS = """\
import pytest


def test_passes():
    pass


def test_plain_skip():
    pytest.skip("not related to postgres")


@pytest.mark.postgres
def test_postgres_passes():
    pass


@pytest.mark.postgres
def test_postgres_skips():
    pytest.skip("pretend there is no server")
"""


def run_sample_session(folder: Path, *, require: bool, tests: str = SAMPLE_TESTS):
    """Μια ξεχωριστή εκτέλεση του pytest πάνω σε μικρά tests, με το ίδιο plugin του conftest."""
    (folder / "conftest.py").write_text(SAMPLE_CONFTEST, encoding="utf-8")
    (folder / "test_sample.py").write_text(tests, encoding="utf-8")
    (folder / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    postgres: sample\n", encoding="utf-8"
    )
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("ELFANTASY_")
    }
    environment["PYTHONPATH"] = os.pathsep.join([str(TESTS_DIR), str(SRC_DIR)])
    if require:
        environment["ELFANTASY_REQUIRE_POSTGRES"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-rs", str(folder)],
        cwd=folder,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_a_skipped_postgres_test_fails_the_session_when_postgres_is_required(tmp_path):
    result = run_sample_session(tmp_path, require=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "postgres tests skipped although they are required" in result.stdout
    assert "test_sample.py::test_postgres_skips" in result.stdout
    assert "test_sample.py::test_plain_skip" not in result.stdout.split("although they are")[-1]
    assert "2 passed, 2 skipped" in result.stdout  # όλα τα άλλα tests έτρεξαν κανονικά


def test_the_same_skip_is_fine_when_postgres_is_not_required(tmp_path):
    result = run_sample_session(tmp_path, require=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "postgres tests skipped" not in result.stdout


def test_other_skips_and_passing_postgres_tests_are_not_a_problem(tmp_path):
    tests = SAMPLE_TESTS.split("@pytest.mark.postgres\ndef test_postgres_skips")[0]
    result = run_sample_session(tmp_path, require=True, tests=tests)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed, 1 skipped" in result.stdout
