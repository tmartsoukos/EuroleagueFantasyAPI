"""Tests του config.py: προεπιλογές, μεταβλητές περιβάλλοντος και αρχείο .env."""

from pathlib import Path

import pytest

from elfantasy.config import DEFAULT_MAE_THRESHOLD, Settings, get_settings

VARIABLES = (
    "DATABASE_URL",
    "MODEL_PATH",
    "MAE_THRESHOLD",
    "ADMIN_API_KEY",
    "DATA_DIR",
    "RENDER_GIT_COMMIT",
    "GIT_COMMIT",
)


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


def test_the_commit_is_empty_by_default(clean_environment):
    assert Settings().git_commit == ""


def test_the_commit_is_read_from_the_render_variable_first(clean_environment, monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "generic0")
    assert Settings().git_commit == "generic0"
    monkeypatch.setenv("RENDER_GIT_COMMIT", "render01")
    assert Settings().git_commit == "render01"  # η μεταβλητή του Render προηγείται


def test_the_commit_can_also_be_given_by_the_field_name(clean_environment):
    assert Settings(_env_file=None, git_commit="deadbeef").git_commit == "deadbeef"


def test_the_commit_can_come_from_the_dotenv_file(clean_environment):
    (clean_environment / ".env").write_text("RENDER_GIT_COMMIT=fromdotenv\n", encoding="utf-8")
    assert Settings().git_commit == "fromdotenv"


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


def test_the_representation_hides_the_database_password_and_the_admin_key(clean_environment):
    """Οι ρυθμίσεις μπορεί να τυπωθούν κατά λάθος σε log ή σε μήνυμα σφάλματος."""
    settings = Settings(
        _env_file=None,
        database_url="postgresql://postgres.abc:S3cr3t-Pa55@db.example.com:5432/postgres",
        admin_api_key="very-secret-key",
    )
    for text in (repr(settings), str(settings)):
        assert "S3cr3t-Pa55" not in text and "very-secret-key" not in text
        assert "postgres.abc:***@db.example.com" in text and "admin_api_key='***'" in text
    assert settings.database_url.endswith("S3cr3t-Pa55@db.example.com:5432/postgres")  # η τιμή ίδια


def test_the_representation_keeps_harmless_values_and_an_empty_key(clean_environment):
    text = repr(Settings(_env_file=None))
    assert "database_url='sqlite:///data/elfantasy.db'" in text
    assert "admin_api_key=''" in text and "mae_threshold=6.0" in text
