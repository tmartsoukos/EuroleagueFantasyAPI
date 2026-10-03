"""Tests του config.py: προεπιλογές, μεταβλητές περιβάλλοντος και αρχείο .env."""

from pathlib import Path

import pytest

from elfantasy.config import DEFAULT_MAE_THRESHOLD, Settings, get_settings

VARIABLES = ("DATABASE_URL", "MODEL_PATH", "MAE_THRESHOLD", "ADMIN_API_KEY", "DATA_DIR")


@pytest.fixture
def clean_environment(monkeypatch, tmp_path):
    """Χωρίς μεταβλητές περιβάλλοντος και χωρίς αρχείο .env (ο τρέχων φάκελος είναι άδειος)."""
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_defaults(clean_environment):
    settings = Settings()
    assert settings.database_url == "sqlite:///data/elfantasy.db"
    assert settings.model_path == "models/model.joblib"
    assert settings.mae_threshold == DEFAULT_MAE_THRESHOLD == 6.0
    assert settings.admin_api_key == ""
    assert settings.data_dir == "data"


def test_environment_variables_override_the_defaults(clean_environment, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host/db")
    monkeypatch.setenv("MODEL_PATH", "other/model.joblib")
    monkeypatch.setenv("MAE_THRESHOLD", "4.25")
    monkeypatch.setenv("ADMIN_API_KEY", "secret")
    monkeypatch.setenv("DATA_DIR", "somewhere")
    settings = Settings()
    assert settings.database_url == "postgresql://user:pass@host/db"
    assert settings.model_path == "other/model.joblib"
    assert settings.mae_threshold == 4.25
    assert settings.admin_api_key == "secret"
    assert settings.data_dir == "somewhere"


def test_values_are_read_from_a_dotenv_file_and_the_environment_wins(
    clean_environment, monkeypatch
):
    (clean_environment / ".env").write_text(
        "DATABASE_URL=sqlite:///from_dotenv.db\nMAE_THRESHOLD=3.5\nUNRELATED=1\n", encoding="utf-8"
    )
    settings = Settings()
    assert settings.database_url == "sqlite:///from_dotenv.db"
    assert settings.mae_threshold == 3.5
    monkeypatch.setenv("MAE_THRESHOLD", "9")
    assert Settings().mae_threshold == 9.0


def test_derived_folders(clean_environment, monkeypatch):
    monkeypatch.setenv("DATA_DIR", "store")
    settings = Settings()
    assert settings.raw_dir == Path("store") / "raw"
    assert settings.reports_dir == Path("store") / "reports"
    assert settings.logs_dir == Path("store") / "logs"


def test_get_settings_is_cached(clean_environment, monkeypatch):
    first = get_settings()
    assert get_settings() is first
    monkeypatch.setenv("DATA_DIR", "changed")
    assert get_settings().data_dir == "data"  # ακόμη η παλιά τιμή
    get_settings.cache_clear()
    assert get_settings().data_dir == "changed"


def test_no_warning_for_the_model_prefix(clean_environment, recwarn):
    Settings()
    assert not [w for w in recwarn.list if "model_" in str(w.message)]
