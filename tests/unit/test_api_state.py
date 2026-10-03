"""Unit tests της φόρτωσης της εφαρμογής (`elfantasy.api.state`): έλεγχος αρχείου SQLite,
startup/shutdown και η μετάφραση κάθε αποτυχίας σε degraded κατάσταση, χωρίς HTTP.
"""

from __future__ import annotations

import pytest
from api_support import FakePredictor, make_prediction, make_settings
from sqlalchemy.exc import OperationalError

from elfantasy.api import state as app_state
from elfantasy.api.state import (
    PROBLEM_DATABASE,
    PROBLEM_MODEL_FAILED,
    PROBLEM_MODEL_MISSING,
    AppState,
    sqlite_file_is_missing,
    start,
    stop,
)
from elfantasy.config import get_settings
from elfantasy.model.artifact import ModelLoadError
from elfantasy.model.predict import Predictor


class TestSqliteFileIsMissing:
    def test_a_missing_file(self, tmp_path):
        assert sqlite_file_is_missing(f"sqlite:///{(tmp_path / 'absent.db').as_posix()}") is True

    def test_a_relative_path_that_does_not_exist(self):
        assert sqlite_file_is_missing("sqlite:///data/definitely-not-here-123.db") is True

    def test_an_existing_file(self, tmp_path):
        existing = tmp_path / "there.db"
        existing.write_bytes(b"")
        assert sqlite_file_is_missing(f"sqlite:///{existing.as_posix()}") is False

    @pytest.mark.parametrize(
        "url",
        [
            "sqlite://",
            "sqlite:///:memory:",
            "sqlite:///file:memdb1?mode=memory&cache=shared&uri=true",
            "postgresql://user:password@localhost:5432/db",
            "postgres://user:password@localhost:6543/postgres",
            "postgresql+psycopg://user:password@localhost/db",
            "this is not a url",
            "",
        ],
    )
    def test_everything_else_is_not_reported_as_missing(self, url):
        assert sqlite_file_is_missing(url) is False


class TestStartAndStop:
    def test_start_with_injected_parts(self, api_engine, api_predictor, api_clock):
        state = AppState(
            clock=api_clock,
            settings=make_settings(),
            engine=api_engine,
            predictor=api_predictor,
        )
        start(state)
        assert state.started and state.problems == {}
        assert state.service is not None
        assert not state.owns_engine and not state.owns_predictor
        stop(state)
        assert state.service is None and not state.started
        assert state.engine is api_engine and state.predictor is api_predictor  # δεν κλείνουν

    def test_start_is_repeatable_and_forgets_old_problems(
        self, api_engine, api_predictor, api_clock
    ):
        state = AppState(
            clock=api_clock, settings=make_settings(), engine=api_engine, predictor=api_predictor
        )
        state.problems["model"] = "stale problem"
        start(state)
        start(state)
        assert state.problems == {}

    def test_the_settings_default_to_the_environment(self, api_engine, api_predictor, monkeypatch):
        monkeypatch.setenv("ADMIN_API_KEY", "from-env")
        get_settings.cache_clear()
        state = AppState(engine=api_engine, predictor=api_predictor)
        assert state.settings is None
        start(state)
        assert state.settings.admin_api_key == "from-env"

    def test_stop_without_start_is_harmless(self):
        state = AppState()
        stop(state)
        assert state.service is None and not state.started

    def test_stop_closes_only_what_the_application_created(self, api_engine, tmp_path, monkeypatch):
        closed = []
        fake = FakePredictor([make_prediction("P1", 1.0)])
        fake.close = lambda: closed.append("predictor")
        monkeypatch.setattr(Predictor, "load", classmethod(lambda cls, *a, **k: fake))
        url = api_engine.url.render_as_string(hide_password=False)
        state = AppState(settings=make_settings(database_url=url))
        # το FakePredictor δεν έχει τον πίνακα players του Predictor: χρησιμοποιείται η βάση του
        # api_engine, που περιέχει όλους τους πίνακες
        start(state)
        engine = state.engine
        assert state.owns_engine and state.owns_predictor and state.predictor is fake
        stop(state)
        assert closed == ["predictor"]
        assert state.engine is None and state.predictor is None
        assert engine is not api_engine


class TestDegradedStartup:
    @pytest.fixture
    def state(self, api_engine, api_clock):
        return AppState(
            clock=api_clock,
            settings=make_settings(
                database_url=api_engine.url.render_as_string(hide_password=False)
            ),
        )

    @pytest.mark.parametrize(
        ("error", "key", "message"),
        [
            (
                ModelLoadError("artifact not found: C:/secret/model.joblib"),
                "model",
                PROBLEM_MODEL_MISSING,
            ),
            (
                OperationalError("SELECT 1", {}, Exception("connection refused")),
                "database",
                PROBLEM_DATABASE,
            ),
            (RuntimeError("feature bug"), "model", PROBLEM_MODEL_FAILED),
            (KeyError("column"), "model", PROBLEM_MODEL_FAILED),
        ],
    )
    def test_the_failure_of_the_predictor_is_classified(
        self, state, monkeypatch, caplog, error, key, message
    ):
        def failing(cls, *args, **kwargs):
            raise error

        monkeypatch.setattr(Predictor, "load", classmethod(failing))
        with caplog.at_level("ERROR"):
            start(state)
        assert state.problems == {key: message}
        assert state.started and state.service is None and state.predictor is None
        assert caplog.records  # η λεπτομέρεια γράφεται στο log
        stop(state)

    def test_an_engine_that_cannot_be_created(self, state, monkeypatch):
        def failing(url, **kwargs):
            raise ModuleNotFoundError("No module named 'psycopg'")

        monkeypatch.setattr(app_state, "get_engine", failing)
        start(state)
        assert state.problems == {"database": PROBLEM_DATABASE}
        assert state.engine is None and not state.owns_engine and state.started

    def test_a_table_that_cannot_be_created(self, state, monkeypatch):
        def failing(engine):
            raise OperationalError("CREATE TABLE", {}, Exception("read-only database"))

        monkeypatch.setattr(app_state, "ensure_table", failing)
        start(state)
        assert state.problems == {"database": PROBLEM_DATABASE}
        assert state.service is None  # χωρίς βάση δεν γίνεται προσπάθεια φόρτωσης μοντέλου
        assert state.predictor is None
        stop(state)
        assert state.engine is None  # ο engine που δημιουργήθηκε έκλεισε

    def test_a_missing_sqlite_file_never_reaches_get_engine(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(app_state, "get_engine", lambda *a, **k: calls.append(a))
        state = AppState(
            settings=make_settings(
                database_url=f"sqlite:///{(tmp_path / 'no' / 'db.db').as_posix()}"
            )
        )
        start(state)
        assert calls == []
        assert state.problems == {"database": PROBLEM_DATABASE}
        assert not (tmp_path / "no").exists()

    def test_a_recovery_after_a_failed_start_works_on_the_next_start(
        self, state, api_predictor, monkeypatch
    ):
        monkeypatch.setattr(
            Predictor,
            "load",
            classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(ModelLoadError("missing"))),
        )
        start(state)
        assert "model" in state.problems
        stop(state)
        monkeypatch.setattr(Predictor, "load", classmethod(lambda cls, *a, **k: api_predictor))
        start(state)
        assert state.problems == {} and state.service is not None
        stop(state)


class TestPostgresStartup:
    """Στο Postgres το startup ΕΛΕΓΧΕΙ ότι ο πίνακας υπάρχει, αλλά δεν τον δημιουργεί ποτέ (τον
    δημιουργούν τα migrations, με RLS). Με ψεύτικο engine, χωρίς σύνδεση."""

    URL = "postgresql://postgres.abc:S3cr3t-Pa55@db.example.com:5432/postgres"

    @pytest.fixture
    def fake_engine(self, monkeypatch):
        from types import SimpleNamespace

        from elfantasy.db import session

        engine = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"), disposed=False)
        engine.dispose = lambda: setattr(engine, "disposed", True)
        created = []
        monkeypatch.setattr(app_state, "get_engine", lambda url, **kwargs: engine)
        monkeypatch.setattr(session.metadata, "create_all", lambda *a, **k: created.append(a))
        monkeypatch.setattr(session, "schema_is_managed_by_migrations", lambda e: True)
        engine.created = created
        return engine

    def test_an_existing_table_is_accepted_and_nothing_is_created(self, fake_engine, monkeypatch):
        from elfantasy.db import session

        monkeypatch.setattr(session, "missing_tables", lambda engine, tables: [])
        state = AppState(settings=make_settings(database_url=self.URL))
        app_state._start_database(state, state.settings)
        assert state.problems == {} and state.engine is fake_engine and state.owns_engine
        assert fake_engine.created == []

    def test_a_missing_table_makes_the_app_degraded_without_creating_it(
        self, fake_engine, monkeypatch, caplog
    ):
        from elfantasy.db import session

        monkeypatch.setattr(
            session, "missing_tables", lambda engine, tables: ["player_availability"]
        )
        state = AppState(settings=make_settings(database_url=self.URL))
        with caplog.at_level("ERROR"):
            app_state._start_database(state, state.settings)
        assert state.problems == {"database": PROBLEM_DATABASE}
        assert fake_engine.created == []
        assert "player_availability" in caplog.text and "S3cr3t-Pa55" not in caplog.text

    def test_start_stops_before_loading_the_model_when_the_schema_is_missing(
        self, fake_engine, monkeypatch
    ):
        from elfantasy.db import session

        monkeypatch.setattr(
            session, "missing_tables", lambda engine, tables: ["player_availability"]
        )
        loaded = []
        monkeypatch.setattr(Predictor, "load", classmethod(lambda cls, *a, **k: loaded.append(1)))
        state = AppState(settings=make_settings(database_url=self.URL))
        start(state)
        assert state.problems == {"database": PROBLEM_DATABASE} and loaded == []
        stop(state)
        assert fake_engine.disposed  # η engine που δημιούργησε η εφαρμογή κλείνει

    def test_an_unreachable_postgres_is_a_degraded_start_without_a_traceback_password(self, caplog):
        url = "postgresql://postgres:S3cr3t-Pa55@127.0.0.1:1/postgres?connect_timeout=1"
        state = AppState(settings=make_settings(database_url=url))
        with caplog.at_level("ERROR"):
            start(state)
        assert state.problems == {"database": PROBLEM_DATABASE} and state.service is None
        assert "S3cr3t-Pa55" not in caplog.text
        stop(state)
