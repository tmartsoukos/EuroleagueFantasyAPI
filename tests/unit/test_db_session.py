"""Tests του db/session.py: ρυθμίσεις engine για Postgres, έλεγχος σχήματος και προστασίες."""

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import make_url

from elfantasy.db import models, session
from elfantasy.db.session import (
    LegacySchemaError,
    SchemaNotInitialisedError,
    create_all,
    ensure_schema,
    get_engine,
    missing_tables,
    postgres_engine_options,
    schema_is_managed_by_migrations,
)
from elfantasy.db.urls import DatabaseUrlError

SECRET = "S3cr3t-Pa55"


@pytest.fixture
def captured(monkeypatch):
    """Καταγράφει τα ορίσματα που περνούν στο `create_engine`, χωρίς να δημιουργεί engine."""
    calls = []

    def fake_create_engine(url, **kwargs):
        calls.append((url, kwargs))
        return create_engine("sqlite://")

    monkeypatch.setattr(session, "create_engine", fake_create_engine)
    return calls


class TestPostgresEngineOptions:
    def test_the_defaults_of_a_session_pooler(self):
        url, options = postgres_engine_options(
            make_url(f"postgresql+psycopg://u:{SECRET}@host:5432/postgres?sslmode=require")
        )
        assert options["pool_pre_ping"] is True
        assert options["pool_size"] == 5 and options["max_overflow"] == 5
        assert options["pool_recycle"] == 300
        assert options["connect_args"] == {"connect_timeout": 10}  # καμία αλλαγή στα prepared
        assert url.query == {"sslmode": "require"}  # το sslmode περνά αυτούσιο στο libpq

    def test_the_transaction_pooler_disables_prepared_statements(self):
        _, options = postgres_engine_options(
            make_url("postgresql+psycopg://u:p@host:6543/postgres")
        )
        assert options["connect_args"] == {"prepare_threshold": None, "connect_timeout": 10}

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("none", None), ("OFF", None), ("disable", None), ("0", 0), ("5", 5), (" 12 ", 12)],
    )
    def test_the_prepare_threshold_parameter_of_the_url(self, value, expected):
        url, options = postgres_engine_options(
            make_url(
                f"postgresql+psycopg://u:p@host:5432/db?prepare_threshold={value}&sslmode=require"
            )
        )
        assert options["connect_args"]["prepare_threshold"] == expected
        assert "prepare_threshold" not in url.query  # δεν περνά στον driver ως κείμενο
        assert url.query == {"sslmode": "require"}

    def test_the_parameter_wins_over_the_port(self):
        _, options = postgres_engine_options(
            make_url("postgresql+psycopg://u:p@host:6543/db?prepare_threshold=3")
        )
        assert options["connect_args"]["prepare_threshold"] == 3

    def test_a_repeated_parameter_uses_the_last_value(self):
        url = make_url(
            "postgresql+psycopg://u:p@host/db?prepare_threshold=1&prepare_threshold=none"
        )
        _, options = postgres_engine_options(url)
        assert options["connect_args"]["prepare_threshold"] is None

    @pytest.mark.parametrize("value", ["abc", "-1", "1.5", "none-ish"])
    def test_an_invalid_prepare_threshold_is_rejected_without_the_password(self, value):
        url = make_url(f"postgresql+psycopg://u:{SECRET}@host/db?prepare_threshold={value}")
        with pytest.raises(DatabaseUrlError) as excinfo:
            postgres_engine_options(url)
        assert SECRET not in str(excinfo.value)
        assert "prepare_threshold" in str(excinfo.value)

    def test_a_blank_prepare_threshold_is_ignored(self):
        # η SQLAlchemy δεν κρατά παραμέτρους χωρίς τιμή στο URL
        _, options = postgres_engine_options(
            make_url("postgresql+psycopg://u:p@host/db?prepare_threshold=")
        )
        assert "prepare_threshold" not in options["connect_args"]

    def test_a_connect_timeout_in_the_url_is_respected(self):
        url, options = postgres_engine_options(
            make_url("postgresql+psycopg://u:p@host/db?connect_timeout=3")
        )
        assert "connect_timeout" not in options["connect_args"]
        assert url.query == {"connect_timeout": "3"}


class TestGetEngine:
    def test_a_postgres_url_gets_the_pool_settings(self, captured):
        get_engine(
            f"postgresql://u:{SECRET}@aws-0-eu.pooler.supabase.com:5432/postgres?sslmode=require"
        )
        ((url, kwargs),) = captured
        assert url.drivername == "postgresql+psycopg"  # postgresql:// -> psycopg 3
        assert url.query == {"sslmode": "require"}
        assert kwargs["pool_pre_ping"] is True
        assert (kwargs["pool_size"], kwargs["max_overflow"], kwargs["pool_recycle"]) == (5, 5, 300)
        assert kwargs["connect_args"] == {"connect_timeout": 10}

    def test_port_6543_disables_prepared_statements(self, captured):
        get_engine("postgresql://u:p@aws-0-eu.pooler.supabase.com:6543/postgres")
        assert captured[0][1]["connect_args"]["prepare_threshold"] is None

    def test_sqlite_is_unchanged(self, captured, tmp_path):
        get_engine(f"sqlite:///{(tmp_path / 'sub' / 'x.db').as_posix()}")
        ((url, kwargs),) = captured
        assert url.get_backend_name() == "sqlite"
        assert set(kwargs) == {"echo"}  # ούτε pool ούτε connect_args του Postgres
        assert (tmp_path / "sub").is_dir()  # ο φάκελος του αρχείου δημιουργείται

    def test_creating_the_engine_does_not_connect(self):
        # η πόρτα 1 δεν ακούει: αν το get_engine συνδεόταν, θα σήκωνε σφάλμα εδώ
        engine = get_engine(f"postgresql://u:{SECRET}@127.0.0.1:1/db")
        assert engine.dialect.name == "postgresql" and engine.dialect.driver == "psycopg"
        engine.dispose()

    def test_a_postgres_engine_asks_for_full_precision_floats_on_every_new_connection(self):
        """Το Supabase ορίζει extra_float_digits = 0 (15 ψηφία): χωρίς το SET η ανάγνωση των
        `double precision` αλλοιώνει τις τιμές (19.983333333333334 → 19.9833333333333)."""
        calls = []

        class FakeCursor:
            def execute(self, sql):
                calls.append(("execute", sql))

            def close(self):
                calls.append(("close",))

        class FakeConnection:
            def cursor(self):
                return FakeCursor()

            def commit(self):
                calls.append(("commit",))

        engine = get_engine("postgresql://u:p@127.0.0.1:1/db")
        assert event.contains(engine, "connect", session.set_full_precision_floats)
        session.set_full_precision_floats(FakeConnection(), None)
        # commit μετά το SET: αλλιώς το rollback του pool θα το ακύρωνε
        assert calls == [("execute", "SET extra_float_digits = 3"), ("close",), ("commit",)]
        engine.dispose()

    def test_the_float_setting_is_not_applied_to_sqlite(self, tmp_path):
        engine = get_engine(f"sqlite:///{(tmp_path / 'f.db').as_posix()}")
        with engine.connect() as connection:  # θα έσκαγε με «near SET: syntax error»
            assert connection.execute(text("SELECT 1.5")).scalar_one() == 1.5
        engine.dispose()

    def test_the_cursor_is_closed_even_if_the_set_fails(self):
        closed = []

        class BrokenCursor:
            def execute(self, sql):
                raise RuntimeError("boom")

            def close(self):
                closed.append(True)

        class Connection:
            committed = False

            def cursor(self):
                return BrokenCursor()

            def commit(self):  # pragma: no cover - δεν πρέπει να κληθεί
                self.committed = True

        connection = Connection()
        with pytest.raises(RuntimeError, match="boom"):
            session.set_full_precision_floats(connection)
        assert closed == [True] and connection.committed is False

    def test_the_password_never_appears_in_the_repr_or_str_of_the_engine(self):
        engine = get_engine(f"postgresql://u:{SECRET}@127.0.0.1:1/db")
        assert SECRET not in repr(engine) and SECRET not in str(engine.url)
        assert "***" in repr(engine)
        engine.dispose()

    def test_an_invalid_url_raises_without_the_secret(self):
        with pytest.raises(DatabaseUrlError) as excinfo:
            get_engine(f"postgresql://u:{SECRET}@host:badport/db")
        assert SECRET not in str(excinfo.value)

    def test_an_invalid_prepare_threshold_raises_without_the_secret(self):
        with pytest.raises(DatabaseUrlError) as excinfo:
            get_engine(f"postgresql://u:{SECRET}@host/db?prepare_threshold=oops")
        assert SECRET not in str(excinfo.value)

    def test_the_default_url_comes_from_the_settings(self, captured, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db.local:6543/postgres")
        session.get_settings.cache_clear()
        get_engine()
        assert captured[0][0].host == "db.local"


class TestEnsureSchema:
    def test_sqlite_creates_the_missing_tables(self, tmp_path):
        engine = get_engine(f"sqlite:///{(tmp_path / 'a.db').as_posix()}")
        assert missing_tables(engine, models.TABLE_ORDER) == [t.name for t in models.TABLE_ORDER]
        assert ensure_schema(engine) == "created"
        assert missing_tables(engine, models.TABLE_ORDER) == []
        assert ensure_schema(engine) == "created"  # idempotent
        engine.dispose()

    def test_sqlite_creates_only_the_requested_tables(self, tmp_path):
        engine = get_engine(f"sqlite:///{(tmp_path / 'a.db').as_posix()}")
        ensure_schema(engine, [models.player_availability])
        assert inspect(engine).get_table_names() == ["player_availability"]
        engine.dispose()

    def test_schema_migrations_is_detected(self, tmp_path):
        engine = get_engine(f"sqlite:///{(tmp_path / 'a.db').as_posix()}")
        assert not schema_is_managed_by_migrations(engine)
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE schema_migrations (version TEXT)"))
        assert schema_is_managed_by_migrations(engine)
        engine.dispose()

    def test_create_all_refuses_postgres(self):
        engine = create_engine("postgresql+psycopg://u:p@127.0.0.1:1/db")
        with pytest.raises(RuntimeError, match="migrations"):
            create_all(engine)
        engine.dispose()

    def test_create_all_still_works_on_sqlite(self, tmp_path):
        engine = get_engine(f"sqlite:///{(tmp_path / 'a.db').as_posix()}")
        create_all(engine)
        assert len(inspect(engine).get_table_names()) == 6
        engine.dispose()

    # ----- Postgres: με ψεύτικο engine, χωρίς σύνδεση -----
    @staticmethod
    def _postgres(monkeypatch, *, missing, managed):
        monkeypatch.setattr(session, "missing_tables", lambda engine, tables: list(missing))
        monkeypatch.setattr(session, "schema_is_managed_by_migrations", lambda engine: managed)
        return SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

    def test_postgres_never_creates_anything(self, monkeypatch):
        engine = self._postgres(monkeypatch, missing=["players", "games"], managed=False)
        created = []
        monkeypatch.setattr(session.metadata, "create_all", lambda *a, **k: created.append(1))
        with pytest.raises(SchemaNotInitialisedError) as excinfo:
            ensure_schema(engine)
        assert "players, games" in str(excinfo.value)
        assert "python -m elfantasy.db.migrate" in str(excinfo.value)
        assert created == []

    def test_postgres_with_all_tables_and_migrations(self, monkeypatch):
        engine = self._postgres(monkeypatch, missing=[], managed=True)
        assert ensure_schema(engine) == "migrations"

    def test_postgres_with_all_tables_but_no_schema_migrations(self, monkeypatch):
        engine = self._postgres(monkeypatch, missing=[], managed=False)
        assert ensure_schema(engine, [models.player_availability]) == "existing"

    def test_the_error_is_a_database_error_for_the_api(self):
        from sqlalchemy.exc import SQLAlchemyError

        assert issubclass(SchemaNotInitialisedError, SQLAlchemyError)


class TestLegacyPredictionsTable:
    LEGACY = (
        "CREATE TABLE predictions (id INTEGER NOT NULL PRIMARY KEY, player_id VARCHAR NOT NULL, "
        "season INTEGER, gamecode INTEGER, predicted_fantasy FLOAT NOT NULL, predicted_pir FLOAT, "
        "model_version VARCHAR NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL)"
    )

    @pytest.fixture
    def engine(self, tmp_path):
        engine = get_engine(f"sqlite:///{(tmp_path / 'legacy.db').as_posix()}")
        with engine.begin() as connection:
            connection.execute(text(self.LEGACY))
            connection.execute(
                text("CREATE INDEX ix_predictions_player_id ON predictions (player_id)")
            )
        yield engine
        engine.dispose()

    def test_an_empty_legacy_table_is_replaced(self, engine):
        assert ensure_schema(engine) == "created"
        columns = {c["name"] for c in inspect(engine).get_columns("predictions")}
        assert "as_of" in columns
        assert [u["column_names"] for u in inspect(engine).get_unique_constraints("predictions")]

    def test_a_legacy_table_with_rows_is_never_touched(self, engine):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO predictions (player_id, predicted_fantasy, model_version) "
                    "VALUES ('P1', 1.0, 'v')"
                )
            )
        with pytest.raises(LegacySchemaError, match="1 rows"):
            ensure_schema(engine)
        columns = {c["name"] for c in inspect(engine).get_columns("predictions")}
        assert "as_of" not in columns
        with engine.connect() as connection:
            assert connection.execute(text("SELECT COUNT(*) FROM predictions")).scalar_one() == 1

    def test_a_table_with_the_new_layout_is_left_alone(self, tmp_path):
        engine = get_engine(f"sqlite:///{(tmp_path / 'new.db').as_posix()}")
        ensure_schema(engine)
        assert session._replace_empty_legacy_predictions(engine) is False
        engine.dispose()

    def test_no_table_at_all_is_not_a_legacy_table(self, tmp_path):
        engine = get_engine(f"sqlite:///{(tmp_path / 'none.db').as_posix()}")
        assert session._replace_empty_legacy_predictions(engine) is False
        engine.dispose()

    def test_unrelated_subsets_do_not_trigger_the_replacement(self, engine):
        ensure_schema(engine, [models.player_availability])
        assert "as_of" not in {c["name"] for c in inspect(engine).get_columns("predictions")}
