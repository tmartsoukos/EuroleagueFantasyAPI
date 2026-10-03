"""Tests των μετρικών αξιολόγησης (`elfantasy.model.metrics`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elfantasy.model import metrics as M


class TestRegressionMetrics:
    def test_hand_computed_values(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        pred = np.array([12.0, 18.0, 33.0, 40.0])  # σφάλματα +2, -2, +3, 0
        result = M.regression_metrics(y, pred)
        assert result["n"] == 4
        assert result["mae"] == pytest.approx(7 / 4)
        assert result["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9 + 0) / 4))
        assert result["bias"] == pytest.approx(3 / 4)
        # R² = 1 - ΣΣφάλματος² / ΣΣολικό: μέσος 25, ΣΣ = 225 + 25 + 25 + 225 = 500
        assert result["r2"] == pytest.approx(1 - 17 / 500)

    def test_perfect_and_constant_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert M.regression_metrics(y, y) == {
            "n": 3,
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 1.0,
            "bias": 0.0,
        }
        constant = M.regression_metrics(y, np.full(3, y.mean()))
        assert constant["r2"] == pytest.approx(0.0)

    def test_a_constant_target_has_an_undefined_r2(self):
        assert np.isnan(M.regression_metrics(np.ones(3), np.zeros(3))["r2"])

    def test_mae_helper(self):
        assert M.mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)


class TestSegments:
    def test_minutes_buckets(self):
        minutes = pd.Series([np.nan, 5.0, 9.99, 10.0, 15.0, 20.0, 20.01, 30.0])
        labels = M.minutes_bucket(minutes).tolist()
        assert labels == [
            "no_history",
            "<10 min",
            "<10 min",
            "10-20 min",
            "10-20 min",
            "10-20 min",
            ">20 min",
            ">20 min",
        ]

    def test_history_buckets(self):
        labels = M.history_bucket(pd.Series([0, 4, 5, 100]))
        assert labels.tolist() == ["<5 prior", "<5 prior", ">=5 prior", ">=5 prior"]

    def test_phase_buckets(self):
        labels = M.phase_bucket(pd.Series(["RS", "PO", "FF", "PI", None]))
        assert labels.tolist() == [
            "RS",
            "playoffs/other",
            "playoffs/other",
            "playoffs/other",
            "unknown",
        ]

    def test_segment_report(self):
        y = np.array([10.0, 10.0, 20.0, 20.0])
        report = M.segment_report(
            {"group": pd.Series(["a", "a", "b", "b"])},
            y,
            {"good": np.array([10.0, 12.0, 20.0, 20.0]), "bad": np.array([0.0, 0.0, 0.0, 0.0])},
        )
        assert report["group"]["a"] == {"n": 2, "good": 1.0, "bad": 10.0}
        assert report["group"]["b"] == {"n": 2, "good": 0.0, "bad": 20.0}


class TestBootstrap:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(0)
        n_games, per_game = 120, 12
        groups = np.repeat(np.arange(n_games), per_game)
        y = rng.normal(8, 6, size=n_games * per_game)
        good = y + rng.normal(0, 3, size=y.size)  # σφάλμα ~3
        bad = y + rng.normal(0, 6, size=y.size)  # σφάλμα ~6
        return y, good, bad, groups

    def test_a_better_model_has_a_negative_interval(self, data):
        y, good, bad, groups = data
        result = M.paired_bootstrap_mae_difference(y, good, bad, groups, iterations=500, seed=1)
        assert result["difference"] == pytest.approx(M.mae(y, good) - M.mae(y, bad))
        assert result["ci_low"] <= result["difference"] <= result["ci_high"]
        assert result["ci_high"] < 0
        assert result["prob_better"] == 1.0
        assert result["n_games"] == 120 and result["iterations"] == 500

    def test_identical_predictions_give_a_zero_difference(self, data):
        y, good, _, groups = data
        result = M.paired_bootstrap_mae_difference(y, good, good, groups, iterations=200, seed=1)
        assert result["difference"] == result["ci_low"] == result["ci_high"] == 0.0

    def test_the_result_depends_only_on_the_seed(self, data):
        y, good, bad, groups = data
        first = M.paired_bootstrap_mae_difference(y, good, bad, groups, iterations=300, seed=5)
        second = M.paired_bootstrap_mae_difference(y, good, bad, groups, iterations=300, seed=5)
        other = M.paired_bootstrap_mae_difference(y, good, bad, groups, iterations=300, seed=6)
        assert first == second
        assert first["ci_low"] != other["ci_low"]

    def test_games_are_resampled_as_a_whole(self):
        # Ένας μόνο αγώνας: κάθε επαναδειγματοληψία δίνει το ίδιο αποτέλεσμα
        y = np.array([1.0, 2.0, 3.0])
        result = M.paired_bootstrap_mae_difference(
            y, y + 1.0, y + 2.0, np.array([7, 7, 7]), iterations=50, seed=1
        )
        assert result["ci_low"] == result["ci_high"] == pytest.approx(-1.0)

    def test_the_mae_interval_contains_the_estimate(self, data):
        y, good, _, groups = data
        result = M.bootstrap_mae_interval(y, good, groups, iterations=400, seed=2)
        assert result["ci_low"] < result["mae"] < result["ci_high"]
        assert result["mae"] == pytest.approx(M.mae(y, good))
        # Το διάστημα είναι στενό για 1440 γραμμές αλλά όχι μηδενικό
        assert 0.05 < result["ci_high"] - result["ci_low"] < 0.8

    def test_a_larger_sample_gives_a_narrower_interval(self):
        rng = np.random.default_rng(3)

        def width(n_games):
            groups = np.repeat(np.arange(n_games), 10)
            y = rng.normal(0, 5, size=groups.size)
            prediction = y + rng.normal(0, 4, size=groups.size)
            result = M.bootstrap_mae_interval(y, prediction, groups, iterations=300, seed=1)
            return result["ci_high"] - result["ci_low"]

        assert width(400) < width(40)
