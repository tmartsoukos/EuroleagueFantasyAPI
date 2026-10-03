"""Οι εντολές της Φάσης 5 τρέχουν ως `python -m ...` και τα μηνύματά τους φαίνονται στην κονσόλα.

Με `python -m module` το `__name__` του module είναι `__main__`: ένας logger με όνομα `__name__`
δεν θα έφτανε στον handler του πακέτου `elfantasy` και τα μηνύματα (ακόμη και τα σφάλματα) θα
χάνονταν σιωπηλά. Τα tests τρέχουν πραγματικές υποδιεργασίες (offline, χωρίς καμία σύνδεση).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SECRET = "S3cr3t-Pa55"


def run_module(module: str, *arguments: str, tmp_path: Path) -> subprocess.CompletedProcess:
    env = {
        key: os.environ[key]
        for key in ("SYSTEMROOT", "PATH", "TEMP", "TMP", "COMSPEC")
        if key in os.environ
    }
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        DATABASE_URL=f"sqlite:///{(tmp_path / 'unused.db').as_posix()}",
    )
    return subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=tmp_path,  # κανένα .env στον φάκελο εργασίας
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def test_migrate_reports_a_non_postgres_url(tmp_path):
    result = run_module("elfantasy.db.migrate", "--db", "sqlite:///x.db", tmp_path=tmp_path)
    assert result.returncode == 2
    assert "PostgreSQL only" in result.stdout
    assert not (tmp_path / "x.db").exists()


def test_migrate_prints_the_init_ddl_with_utf8_and_lf(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "elfantasy.db.migrate", "--print-sql"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUTF8": "1"},
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0
    expected = (ROOT / "src" / "elfantasy" / "db" / "migrations" / "001_init.sql").read_bytes()
    assert result.stdout == expected  # byte προς byte, χωρίς CRLF ή BOM


def test_transfer_reports_a_missing_source(tmp_path):
    result = run_module(
        "elfantasy.db.transfer",
        "--from",
        f"sqlite:///{(tmp_path / 'missing.db').as_posix()}",
        "--to",
        f"sqlite:///{(tmp_path / 'target.db').as_posix()}",
        "--allow-non-postgres",
        tmp_path=tmp_path,
    )
    assert result.returncode == 2
    assert "does not exist" in result.stdout
    assert not (tmp_path / "missing.db").exists()


@pytest.mark.parametrize(
    "module", ["elfantasy.model.record_predictions", "elfantasy.model.evaluate_recorded"]
)
def test_model_tools_report_an_invalid_url_without_the_password(module, tmp_path):
    result = run_module(module, "--db", f"postgresql://u:{SECRET}@h:badport/db", tmp_path=tmp_path)
    assert result.returncode == 2
    assert "invalid database URL" in result.stdout
    assert SECRET not in result.stdout + result.stderr


def test_the_help_of_every_tool_works(tmp_path):
    for module in (
        "elfantasy.db.migrate",
        "elfantasy.db.transfer",
        "elfantasy.model.record_predictions",
        "elfantasy.model.evaluate_recorded",
    ):
        result = run_module(module, "--help", tmp_path=tmp_path)
        assert result.returncode == 0 and "usage:" in result.stdout, module
