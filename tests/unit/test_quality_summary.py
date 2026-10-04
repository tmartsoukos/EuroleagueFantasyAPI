"""Tests του `scripts/quality_summary.py`: η περίληψη του quality gate στη σελίδα του run.

Τρέχουν πάνω στο πραγματικό committed μοντέλο και στο held-out fixture (λιγότερο από ένα
δευτερόλεπτο το καθένα), ώστε το script να μη χαλάσει σιωπηλά (το CI το καλεί με
`if: !cancelled()` και δεν θα το πρόσεχε κανείς) και τα νούμερά του να ταυτίζονται με του gate.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import quality_summary as summary

from elfantasy.config import get_settings

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "quality_summary.py"


@pytest.fixture
def settings_env(monkeypatch):
    """Αλλαγή των ρυθμίσεων μέσω περιβάλλοντος (η cache των ρυθμίσεων αδειάζει πριν και μετά)."""

    def apply(**values: str):
        for name, value in values.items():
            monkeypatch.setenv(name, value)
        get_settings.cache_clear()

    yield apply
    get_settings.cache_clear()


def recorded_mae() -> float:
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
    return metrics["fantasy"]["test"]["mae"]


def test_the_summary_shows_the_gate_numbers():
    text = summary.build_summary()
    assert text.startswith("### Model quality gate\n")
    assert "**PASS**" in text
    value = float(re.search(r"fantasy MAE (\d+\.\d+) on the held-out", text).group(1))
    assert value == pytest.approx(recorded_mae(), abs=1e-3)  # το ίδιο με το metrics.json
    assert "threshold 6.00" in text and "| threshold (`MAE_THRESHOLD`) | 6.00 |" in text
    assert "baseline: naive season mean |" in text
    assert "Improvement over the best baseline (naive season mean)" in text
    assert "xgboost" in text and "numpy" in text  # οι εκδόσεις της εκπαίδευσης


def test_the_best_baseline_is_the_one_with_the_lowest_mae():
    text = summary.build_summary()
    rows = dict(re.findall(r"\| baseline: ([\w ]+) \| ([\d.]+) \|", text))
    assert len(rows) == 4
    best = min(rows, key=lambda label: float(rows[label]))
    assert f"({best})" in text


def test_a_threshold_below_the_mae_is_reported_as_a_failure(settings_env):
    settings_env(MAE_THRESHOLD="5.50")
    text = summary.build_summary()
    assert "**FAIL**" in text and "threshold 5.50." in text
    assert "margin" not in text


def test_a_threshold_that_is_not_set_is_a_failure(settings_env):
    settings_env(MAE_THRESHOLD="0")
    assert "**FAIL**" in summary.build_summary()


def test_a_missing_model_is_a_clean_error(settings_env, capsys, tmp_path):
    settings_env(MODEL_PATH=str(tmp_path / "missing.joblib"))
    assert summary.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: ModelLoadError" in captured.err and "Traceback" not in captured.err


def test_it_runs_as_a_script_from_another_folder(tmp_path):
    """Όπως στο CI (`python scripts/quality_summary.py >> $GITHUB_STEP_SUMMARY`), χωρίς `.env`."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("### Model quality gate")
    assert "**PASS**" in result.stdout
