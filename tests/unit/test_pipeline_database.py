"""Tests της διαδρομής βάσης του ingestion (Φάση 5): διαχείριση σχήματος και κρυμμένοι κωδικοί.

Σε SQLite το pipeline δημιουργεί τους πίνακες (`create_all`)· σε Postgres ποτέ τίποτα:
τους δημιουργούν τα migrations, με Row Level Security (docs/DATABASE.md).
"""

import logging

import pytest
from recorded_support import make_legacy_predictions
from sqlalchemy import inspect

from elfantasy.db.session import LegacySchemaError, SchemaNotInitialisedError, get_engine
from elfantasy.ingest import pipeline

SECRET = "S3cr3t-Pa55"
PG_URL = f"postgresql://postgres.abc:{SECRET}@aws-0-eu.pooler.supabase.com:5432/postgres"
SEASONS = [2016, 2023, 2024, 2025, 2026]


def stub_run(monkeypatch, result=None, error=None):
    def fake_run(seasons, **kwargs):
        if error is not None:
            raise error
        return result or pipeline.PipelineResult(seasons=list(seasons))

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run)


class TestRedaction:
    def test_the_arguments_are_logged_without_the_database_password(
        self, monkeypatch, restore_logging, tmp_path, capsys
    ):
        stub_run(monkeypatch)
        arguments = ["--seasons", "2025", "--no-fetch", "--db", PG_URL, "--data-dir", str(tmp_path)]
        assert pipeline.main(arguments) == 0
        log = (tmp_path / "logs" / "ingest.log").read_text(encoding="utf-8")
        out = capsys.readouterr().out
        assert SECRET not in log and SECRET not in out
        assert "postgres.abc:***@aws-0-eu.pooler.supabase.com:5432/postgres" in log
        assert "Database: postgresql+psycopg://postgres.abc:***@" in log

    def test_the_default_database_url_of_the_settings_is_logged_safely(
        self, monkeypatch, restore_logging, tmp_path
    ):
        monkeypatch.setenv("DATABASE_URL", PG_URL)
        pipeline.get_settings.cache_clear()
        stub_run(monkeypatch)
        assert pipeline.main(["--seasons", "2025", "--no-fetch", "--data-dir", str(tmp_path)]) == 0
        log = (tmp_path / "logs" / "ingest.log").read_text(encoding="utf-8")
        assert SECRET not in log and "Database: postgresql+psycopg://postgres.abc:***@" in log

    def test_without_a_db_argument_none_is_logged(self, monkeypatch, restore_logging, tmp_path):
        stub_run(monkeypatch)
        assert pipeline.main(["--seasons", "2025", "--no-fetch", "--data-dir", str(tmp_path)]) == 0
        log = (tmp_path / "logs" / "ingest.log").read_text(encoding="utf-8")
        assert "'db': None" in log


class TestMissingSchema:
    @pytest.mark.parametrize(
        "error",
        [
            SchemaNotInitialisedError("database tables are missing: teams; apply the migrations"),
            LegacySchemaError("the predictions table has the old layout"),
        ],
    )
    def test_main_reports_a_clean_failure_without_a_traceback(
        self, monkeypatch, restore_logging, tmp_path, caplog, error
    ):
        stub_run(monkeypatch, error=error)
        with caplog.at_level(logging.ERROR):
            code = pipeline.main(["--seasons", "2025", "--no-fetch", "--data-dir", str(tmp_path)])
        assert code == 1
        assert str(error) in caplog.text
        assert not any(record.exc_info for record in caplog.records)  # χωρίς traceback

    def test_nothing_is_written_when_postgres_has_no_schema(
        self, fixture_cache, tmp_path, monkeypatch
    ):
        """Προσομοίωση Postgres χωρίς migrations: το `ensure_schema` αρνείται, καμία εγγραφή."""
        engine = get_engine(f"sqlite:///{(tmp_path / 'sim.db').as_posix()}")

        def refuse(engine):
            raise SchemaNotInitialisedError("database tables are missing: teams")

        monkeypatch.setattr(pipeline, "get_engine", lambda url=None: engine)
        monkeypatch.setattr(pipeline, "ensure_schema", refuse)
        with pytest.raises(SchemaNotInitialisedError):
            pipeline.run_pipeline(SEASONS, db_url=PG_URL, data_dir=tmp_path / "data", fetch=False)
        assert inspect(engine).get_table_names() == []
        engine.dispose()

    def test_the_pipeline_uses_ensure_schema_and_never_create_all(
        self, fixture_cache, tmp_path, monkeypatch
    ):
        engine = get_engine(f"sqlite:///{(tmp_path / 'real.db').as_posix()}")
        calls = []
        real = pipeline.ensure_schema
        monkeypatch.setattr(pipeline, "get_engine", lambda url=None: engine)
        monkeypatch.setattr(pipeline, "ensure_schema", lambda e: calls.append(e) or real(e))
        assert not hasattr(pipeline, "create_all")
        result = pipeline.run_pipeline(SEASONS, data_dir=tmp_path / "data", fetch=False)
        assert calls == [engine]
        assert result.table_counts["player_games"] > 0
        engine.dispose()

    def test_an_old_empty_predictions_table_does_not_break_the_pipeline(
        self, fixture_cache, tmp_path
    ):

        url = f"sqlite:///{(tmp_path / 'old.db').as_posix()}"
        engine = get_engine(url)
        make_legacy_predictions(engine)
        result = pipeline.run_pipeline(SEASONS, db_url=url, data_dir=tmp_path / "data", fetch=False)
        assert result.table_counts["predictions"] == 0
        assert "as_of" in {c["name"] for c in inspect(engine).get_columns("predictions")}
        engine.dispose()
