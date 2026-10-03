"""Tests του db/urls.py: κανονικοποίηση, κρυμμένοι κωδικοί, σύγκριση βάσεων."""

import pytest
from sqlalchemy.engine import make_url

from elfantasy.db import urls
from elfantasy.db.urls import (
    DatabaseUrlError,
    backend_name,
    normalize_database_url,
    parse_database_url,
    redact_secrets,
    safe_url,
    same_database,
    sqlite_database_path,
    sqlite_file_is_missing,
)

SECRET = "S3cr3t-Pa55"
PG_URL = f"postgresql://postgres.abcdef:{SECRET}@aws-0-eu.pooler.supabase.com:5432/postgres"


class TestNormalize:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("postgresql://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
            (
                "postgres://u:p@h:6543/db?sslmode=require",
                "postgresql+psycopg://u:p@h:6543/db?sslmode=require",
            ),
            ("postgresql+psycopg://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
            ("postgresql+psycopg2://u:p@h/db", "postgresql+psycopg2://u:p@h/db"),
            ("sqlite:///x.db", "sqlite:///x.db"),
            ("", ""),
        ],
    )
    def test_the_driver_is_added_only_when_missing(self, url, expected):
        assert normalize_database_url(url) == expected


class TestParse:
    def test_a_valid_url(self):
        parsed = parse_database_url(PG_URL)
        assert parsed.drivername == "postgresql+psycopg"
        assert parsed.port == 5432 and parsed.database == "postgres"
        assert parsed.username == "postgres.abcdef"

    @pytest.mark.parametrize(
        "bad", ["", "not a url", "://nothing", f"postgresql://u:{SECRET}@h:port/db"]
    )
    def test_an_invalid_url_raises_without_echoing_it(self, bad):
        with pytest.raises(DatabaseUrlError) as excinfo:
            parse_database_url(bad)
        message = str(excinfo.value)
        assert SECRET not in message
        assert not bad or bad not in message
        assert excinfo.value.__cause__ is None  # η αρχική εξαίρεση (με το URL) δεν διατηρείται

    def test_backend_name(self):
        assert backend_name(PG_URL) == "postgresql"
        assert backend_name("sqlite:///x.db") == "sqlite"
        assert backend_name(make_url("sqlite://")) == "sqlite"
        with pytest.raises(DatabaseUrlError):
            backend_name("garbage")


class TestSafeUrl:
    def test_the_password_is_hidden(self):
        shown = safe_url(PG_URL)
        assert SECRET not in shown
        assert shown == (
            "postgresql+psycopg://postgres.abcdef:***@aws-0-eu.pooler.supabase.com:5432/postgres"
        )

    def test_accepts_url_objects_and_keeps_the_query(self):
        shown = safe_url(make_url(PG_URL + "?sslmode=require"))
        assert SECRET not in shown and shown.endswith("?sslmode=require")

    def test_none_and_garbage_never_raise(self):
        assert safe_url(None) == "<none>"
        assert safe_url("definitely not a url") == "<invalid database URL>"
        assert SECRET not in safe_url(f"postgresql://u:{SECRET}@h:badport/db")

    def test_a_url_without_a_password_is_unchanged(self):
        assert safe_url("sqlite:///data/elfantasy.db") == "sqlite:///data/elfantasy.db"


class TestRedact:
    def test_the_password_is_replaced_anywhere_in_the_text(self):
        text = f"could not connect with {SECRET} to the host; retry {SECRET}"
        assert redact_secrets(text, PG_URL) == "could not connect with *** to the host; retry ***"

    def test_percent_encoded_passwords_are_found_in_both_forms(self):
        url = "postgresql://u:p%40ss%2Fword@h/db"  # κωδικός: p@ss/word
        text = "raw p%40ss%2Fword and decoded p@ss/word"
        assert redact_secrets(text, url) == "raw *** and decoded ***"

    def test_any_url_with_a_password_inside_the_text_is_redacted(self):
        text = "failed: postgresql://someone:other-secret@db.example.com:5432/postgres"
        redacted = redact_secrets(text)
        assert "other-secret" not in redacted and "someone:***@db.example.com" in redacted

    def test_url_objects_and_missing_urls(self):
        text = f"password {SECRET}"
        assert redact_secrets(text, make_url(PG_URL), None) == "password ***"
        assert redact_secrets("nothing to hide", None, "sqlite:///x.db") == "nothing to hide"

    def test_very_short_passwords_are_not_replaced_in_the_text(self):
        # ένας κωδικός «a» θα κατέστρεφε κάθε λέξη με «a» χωρίς πραγματικό όφελος
        assert redact_secrets("a banana", "postgresql://u:a@h/db") == "a banana"

    def test_the_longest_password_wins_when_one_contains_another(self):
        first = f"postgresql://u:{SECRET}@h/db"
        second = f"postgresql://u:{SECRET}-extra@h/db"
        assert redact_secrets(f"{SECRET}-extra", first, second) == "***"

    def test_an_unparsable_url_still_yields_the_raw_password(self):
        url = f"postgresql://u:{SECRET}@h:notaport/db"
        assert redact_secrets(f"error for {SECRET}", url) == "error for ***"


class TestSqlitePaths:
    def test_a_file_path(self, tmp_path):
        target = tmp_path / "x.db"
        assert sqlite_database_path(f"sqlite:///{target.as_posix()}") == target

    @pytest.mark.parametrize(
        "url",
        ["sqlite://", "sqlite:///:memory:", "sqlite:///file:memdb?mode=memory&uri=true", PG_URL],
    )
    def test_everything_else_has_no_path(self, url):
        assert sqlite_database_path(url) is None

    def test_missing_file_detection(self, tmp_path):
        existing = tmp_path / "there.db"
        existing.write_bytes(b"")
        assert sqlite_file_is_missing(f"sqlite:///{(tmp_path / 'absent.db').as_posix()}")
        assert not sqlite_file_is_missing(f"sqlite:///{existing.as_posix()}")
        assert not sqlite_file_is_missing("sqlite://")
        assert not sqlite_file_is_missing("garbage")  # θα αναφερθεί από το get_engine
        assert not sqlite_file_is_missing(PG_URL)


class TestSameDatabase:
    def test_the_same_sqlite_file_in_different_spellings(self, tmp_path, monkeypatch):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "a.db").write_bytes(b"")
        monkeypatch.chdir(tmp_path)
        absolute = f"sqlite:///{(tmp_path / 'data' / 'a.db').as_posix()}"
        assert same_database("sqlite:///data/a.db", absolute)
        assert same_database("sqlite:///data/../data/a.db", "sqlite:///data/a.db")
        assert not same_database("sqlite:///data/a.db", "sqlite:///data/b.db")

    def test_in_memory_databases_are_never_the_same(self):
        assert not same_database("sqlite://", "sqlite://")
        assert not same_database("sqlite:///:memory:", "sqlite:///:memory:")

    def test_postgres_ignores_driver_password_and_parameters(self):
        other = "postgres://postgres.abcdef:different@AWS-0-EU.pooler.supabase.com/postgres?sslmode=require"
        assert same_database(PG_URL, other)  # κεφαλαία στο host και προεπιλεγμένη πόρτα 5432

    @pytest.mark.parametrize(
        "other",
        [
            f"postgresql://postgres.abcdef:{SECRET}@aws-0-eu.pooler.supabase.com:6543/postgres",
            f"postgresql://postgres.abcdef:{SECRET}@other-host.com:5432/postgres",
            f"postgresql://postgres.abcdef:{SECRET}@aws-0-eu.pooler.supabase.com:5432/other",
            f"postgresql://someone:{SECRET}@aws-0-eu.pooler.supabase.com:5432/postgres",
            "sqlite:///data/a.db",
        ],
    )
    def test_any_difference_in_host_port_database_user_or_backend(self, other):
        assert not same_database(PG_URL, other)

    def test_invalid_urls_raise(self):
        with pytest.raises(DatabaseUrlError):
            same_database(PG_URL, "garbage")


def test_the_password_pattern_matches_only_userinfo():
    assert urls._PASSWORD_IN_URL.match("postgresql://u:pw@h/db").group(1) == "pw"
    assert urls._PASSWORD_IN_URL.match("postgresql://u@h/db") is None
    assert urls._PASSWORD_IN_URL.match("sqlite:///x.db") is None
