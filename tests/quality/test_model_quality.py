"""Quality gate του μοντέλου (Φάση 6): το committed artifact πάνω στο committed held-out fixture.

Το test είναι offline και γρήγορο: φορτώνει το `models/model.joblib` και το
`tests/fixtures/holdout_2025.parquet` (όλες οι συμμετοχές της σεζόν test, με features, στόχους και
προβλέψεις των naive baselines), υπολογίζει το MAE και αποτυγχάνει αν:

* το MAE (fantasy) δεν είναι μικρότερο από το threshold (`MAE_THRESHOLD`, προεπιλογή στο config),
* το MAE δεν συμφωνεί με αυτό που καταγράφηκε στο `metrics.json` κατά την εκπαίδευση,
* το μοντέλο δεν κερδίζει κάθε naive baseline (και δεν είναι στατιστικά καλύτερο από το
  καλύτερο, με paired bootstrap ανά αγώνα).

Το fixture παράγεται με `python -m elfantasy.model.holdout` και το committed μοντέλο
εκπαιδεύεται σε σεζόν που δεν περιλαμβάνουν τη σεζόν του fixture, άρα το MAE είναι out-of-sample.
Αν το αρχείο του μοντέλου λείπει, το test ΑΠΟΤΥΓΧΑΝΕΙ (δεν παραλείπεται): ένα quality gate δεν
επιτρέπεται να περνά σιωπηλά.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from elfantasy.config import get_settings
from elfantasy.features.build import FEATURE_COLUMNS
from elfantasy.model.artifact import ModelBundle, load_bundle
from elfantasy.model.holdout import read_holdout
from elfantasy.model.metrics import mae, paired_bootstrap_mae_difference

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "holdout_2025.parquet"
BASELINES = [
    "baseline_global_mean",
    "baseline_global_median",
    "baseline_naive_rolling5",
    "baseline_naive_season_mean",
]
#: Ανοχή στη συμφωνία με το metrics.json (το MAE αποθηκεύεται με 6 δεκαδικά).
TOLERANCE = 1e-3


def model_path() -> Path:
    path = Path(get_settings().model_path)
    return path if path.is_absolute() else ROOT / path


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    path = model_path()
    assert path.is_file(), (
        f"the model artifact {path} is missing: train it with `python -m elfantasy.model.train`"
    )
    return load_bundle(path)


@pytest.fixture(scope="module")
def holdout():
    assert FIXTURE.is_file(), (
        f"the held-out fixture {FIXTURE} is missing: create it with "
        "`python -m elfantasy.model.holdout --out tests/fixtures/holdout_2025.parquet`"
    )
    frame, meta = read_holdout(FIXTURE)
    return frame, meta


@pytest.fixture(scope="module")
def predictions(bundle, holdout):
    frame, _ = holdout
    return bundle.predict(frame)


@pytest.fixture(scope="module")
def metrics(bundle) -> dict:
    assert bundle.metrics, "the artifact does not embed its training metrics"
    return bundle.metrics


def test_the_artifact_and_the_fixture_use_the_same_features(bundle, holdout):
    frame, meta = holdout
    assert bundle.feature_columns == FEATURE_COLUMNS
    assert meta["feature_columns"] == FEATURE_COLUMNS
    assert set(FEATURE_COLUMNS) <= set(frame.columns)
    assert len(frame) == meta["rows"] > 5000


def test_the_mae_is_below_the_threshold(predictions, holdout):
    """Το quality gate: MAE (fantasy) στο held-out fixture < MAE_THRESHOLD."""
    threshold = get_settings().mae_threshold
    assert threshold > 0, "MAE_THRESHOLD is not set (a value <= 0 means 'not set')"
    frame, _ = holdout
    value = mae(frame["fantasy_score"], predictions["fantasy"])
    assert value < threshold, f"fantasy MAE {value:.4f} on the held-out season >= {threshold}"


def test_the_mae_agrees_with_the_recorded_metrics(predictions, holdout, metrics):
    frame, _ = holdout
    protocol = metrics["protocol"]
    assert len(frame) == protocol["test"]["rows"]
    if not protocol["test_mae_is_out_of_sample"]:
        pytest.skip("the artifact was refit through the test season: MAE is not comparable")
    fantasy = mae(frame["fantasy_score"], predictions["fantasy"])
    pir = mae(frame["pir"], predictions["pir"])
    assert fantasy == pytest.approx(metrics["fantasy"]["test"]["mae"], abs=TOLERANCE)
    assert pir == pytest.approx(metrics["pir"]["test"]["mae"], abs=TOLERANCE)


def test_the_model_beats_every_naive_baseline(predictions, holdout):
    frame, _ = holdout
    model_mae = mae(frame["fantasy_score"], predictions["fantasy"])
    for column in BASELINES:
        assert column in frame.columns
        baseline_mae = mae(frame["fantasy_score"], frame[column])
        assert model_mae < baseline_mae, (
            f"the model ({model_mae:.4f}) loses to {column} ({baseline_mae:.4f})"
        )


def test_the_improvement_over_the_best_baseline_is_statistically_significant(predictions, holdout):
    frame, _ = holdout
    y = frame["fantasy_score"].to_numpy()
    best = min(BASELINES, key=lambda column: mae(y, frame[column]))
    games = (frame["season"].astype(str) + "/" + frame["gamecode"].astype(str)).to_numpy()
    result = paired_bootstrap_mae_difference(
        y, predictions["fantasy"].to_numpy(), frame[best].to_numpy(), games, iterations=400, seed=0
    )
    assert result["ci_high"] < 0, f"the 95% interval of (model - {best}) includes zero: {result}"


def test_the_threshold_is_below_the_best_naive_baseline(holdout):
    """Το όριο πρέπει να είναι αυστηρότερο από το καλύτερο naive baseline (αλλιώς το gate δεν
    εξασφαλίζει ότι το μοντέλο προσφέρει κάτι)."""
    frame, _ = holdout
    best = min(mae(frame["fantasy_score"], frame[column]) for column in BASELINES)
    assert get_settings().mae_threshold < best


def test_the_embedded_metrics_equal_the_metrics_file(metrics):
    path = model_path().parent / "metrics.json"
    assert path.is_file(), f"{path} is missing"
    assert json.loads(path.read_text(encoding="utf-8")) == metrics


def test_the_artifact_was_not_trained_on_the_fixture_season(metrics, holdout):
    _, meta = holdout
    protocol = metrics["protocol"]
    assert protocol["test"]["season"] == meta["test_season"]
    if protocol["test_mae_is_out_of_sample"]:
        assert max(protocol["final_fit"]["seasons"]) < meta["test_season"]


def test_the_predictions_are_finite_and_plausible(predictions):
    values = predictions.to_numpy()
    assert np.isfinite(values).all()
    assert values.min() > -15 and values.max() < 70
    assert predictions["fantasy"].std() > 1.0  # όχι σταθερή πρόβλεψη
