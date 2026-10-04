"""Tests του `check_out_of_sample` (tests/gate_support.py): το quality gate αποτυγχάνει, αντί να
παραλείπεται, όταν το committed artifact έχει δει τη σεζόν του held-out fixture (εύρημα m2 του
review της Φάσης 7: με `train --final-refit-through 2025` το gate περνούσε με in-sample MAE)."""

from __future__ import annotations

import copy

import pytest
from gate_support import REFIT_MESSAGE, check_out_of_sample

GOOD = {
    "protocol": {
        "test": {"season": 2025, "rows": 8741},
        "final_fit": {"seasons": [2016, 2017, 2018, 2024]},
        "test_mae_is_out_of_sample": True,
    }
}


def variant(**changes):
    metrics = copy.deepcopy(GOOD)
    for dotted, value in changes.items():
        *path, last = dotted.split("__")
        node = metrics["protocol"]
        for key in path:
            node = node[key]
        node[last] = value
    return metrics


def test_an_out_of_sample_artifact_passes():
    check_out_of_sample(GOOD, 2025)


def test_a_refit_artifact_fails_with_an_explicit_message():
    with pytest.raises(AssertionError) as error:
        check_out_of_sample(variant(test_mae_is_out_of_sample=False), 2025)
    assert str(error.value).startswith(REFIT_MESSAGE[:40])
    assert "--final-refit-through" in str(error.value)


def test_an_artifact_trained_on_the_fixture_season_fails_even_if_the_flag_says_otherwise():
    with pytest.raises(AssertionError, match="final-refit-through"):
        check_out_of_sample(variant(final_fit__seasons=[2016, 2024, 2025]), 2025)


def test_a_fixture_of_a_different_season_than_the_metrics_fails():
    with pytest.raises(AssertionError, match="the metrics were computed on season 2025"):
        check_out_of_sample(GOOD, 2026)


@pytest.mark.parametrize("flag", [False, None, 0, "yes"])
def test_only_the_boolean_true_counts_as_out_of_sample(flag):
    with pytest.raises(AssertionError):
        check_out_of_sample(variant(test_mae_is_out_of_sample=flag), 2025)
