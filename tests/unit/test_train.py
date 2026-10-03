"""Tests της εκπαίδευσης (`elfantasy.model.train`) σε μικρό συνθετικό dataset (χωρίς πραγματική
βάση).

Καλύπτουν: χωρισμό ανά σεζόν, naive baselines, ολόκληρο το backtest (επιλογή, refit, test),
τον κανόνα του threshold (έξοδος 0, 3, 1 και ατομική εγγραφή), επαναληψιμότητα με seed και ότι η
σεζόν test δεν επηρεάζει ούτε την επιλογή ούτε το τελικό μοντέλο.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from synthetic_league import make_league, write_to_database

from elfantasy.config import get_settings
from elfantasy.db.session import get_engine
from elfantasy.features.build import FEATURE_COLUMNS, build_features
from elfantasy.model import train as T
from elfantasy.model.artifact import (
    ModelBundle,
    RidgeModel,
    XGBoostModel,
    atomic_write_many,
    deserialize_bundle,
    load_bundle,
    serialize_bundle,
)

VAL, TEST = 2022, 2023


@pytest.fixture(scope="module")
def league():
    return make_league(seed=7)


@pytest.fixture(scope="module")
def prepared(league):
    frame = build_features(league.history, games=league.games)
    info = T.describe_dataset(frame, league.history, league.games)
    return frame, info


def make_config(out_dir, **overrides):
    options = {
        "val_season": VAL,
        "test_season": TEST,
        "tune": False,
        "out_dir": out_dir,
        "bootstrap_iterations": 60,
    }
    options.update(overrides)
    return T.TrainConfig(**options)


@pytest.fixture(scope="module")
def outcome(prepared, tmp_path_factory):
    frame, info = prepared
    config = make_config(tmp_path_factory.mktemp("out"))
    return T.run_training(frame, info, config, threshold=1000.0, progress=lambda message: None)


@pytest.fixture
def restore_logging():
    """Το main() ρυθμίζει τον root logger: τον επαναφέρουμε μετά το test."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)


def signature(model) -> bytes:
    """Ισοδυναμία δύο μοντέλων: τα bytes του XGBoost ή οι παράμετροι του Ridge."""
    if isinstance(model, XGBoostModel):
        return model.raw
    return b"".join(
        np.asarray(array, dtype=float).tobytes()
        for array in (model.medians, model.mean, model.scale, model.coef, [model.intercept])
    )


# --------------------------------------------------------------------------------------
# Χωρισμός και baselines
# --------------------------------------------------------------------------------------


class TestSplitsAndBaselines:
    def test_split_by_season(self, prepared, tmp_path):
        frame, _ = prepared
        splits = T.make_splits(frame, make_config(tmp_path))
        assert set(splits.train["season"]) == {2020, 2021}
        assert set(splits.val["season"]) == {VAL}
        assert set(splits.test["season"]) == {TEST}
        assert set(splits.final_fit["season"]) == {2020, 2021, VAL}
        assert len(splits.final_fit) == len(splits.train) + len(splits.val)
        assert splits.train["is_appearance"].all() and splits.test["is_appearance"].all()
        assert splits.train[["fantasy_score", "pir"]].notna().all().all()

    def test_later_seasons_are_never_used(self, prepared, tmp_path):
        frame, _ = prepared
        later = frame[frame["season"] == TEST].copy()
        later["season"] = TEST + 1
        extended = pd.concat([frame, later], ignore_index=True)
        splits = T.make_splits(extended, make_config(tmp_path))
        for part in (splits.train, splits.val, splits.test, splits.final_fit):
            assert (part["season"] <= TEST).all()
        assert len(splits.test) == len(frame[frame["is_appearance"] & (frame["season"] == TEST)])

    def test_empty_splits_and_bad_order_are_errors(self, prepared, tmp_path):
        frame, _ = prepared
        with pytest.raises(T.TrainingError, match="empty"):
            T.make_splits(frame, make_config(tmp_path, val_season=2019, test_season=TEST))
        with pytest.raises(T.TrainingError, match="later"):
            T.make_splits(frame, make_config(tmp_path, val_season=2023, test_season=2022))

    def test_naive_baselines_and_their_fallbacks(self):
        frame = pd.DataFrame(
            {
                "fantasy_mean_5": [10.0, np.nan, np.nan, 7.0],
                "fantasy_season_mean": [8.0, 6.0, np.nan, np.nan],
                "pir_mean_5": [1.0, 1.0, 1.0, 1.0],
                "pir_season_mean": [2.0, 2.0, 2.0, 2.0],
            }
        )
        predictions = T.baseline_predictions(frame, "fantasy", fit_mean=5.0, fit_median=4.0)
        assert predictions["global_mean"].tolist() == [5.0] * 4
        assert predictions["global_median"].tolist() == [4.0] * 4
        # rolling-5, αλλιώς μέσος σεζόν, αλλιώς καθολικός μέσος
        assert predictions["naive_rolling5"].tolist() == [10.0, 6.0, 5.0, 7.0]
        # μέσος σεζόν, αλλιώς rolling-5, αλλιώς καθολικός μέσος
        assert predictions["naive_season_mean"].tolist() == [8.0, 6.0, 5.0, 7.0]
        pir = T.baseline_predictions(frame, "pir", fit_mean=0.0, fit_median=0.0)
        assert pir["naive_rolling5"].tolist() == [1.0] * 4  # το PIR χρησιμοποιεί τις στήλες του PIR

    def test_the_candidate_search_stays_within_the_budget(self):
        tuned = T.candidate_specs(T.TrainConfig(tune=True))
        assert 30 <= len(tuned) <= 40
        assert len({spec.name for spec in tuned}) == len(tuned)
        assert {spec.family for spec in tuned} == {
            "ridge",
            "xgb_squarederror",
            "xgb_absoluteerror",
            "xgb_pseudohuber",
        }
        assert tuned == T.candidate_specs(T.TrainConfig(tune=True))  # ντετερμινιστικό
        quick = T.candidate_specs(T.TrainConfig(tune=False))
        assert [spec.family for spec in quick] == ["ridge", "xgb_absoluteerror"]


# --------------------------------------------------------------------------------------
# Ολόκληρο το backtest
# --------------------------------------------------------------------------------------


class TestBacktest:
    def test_metrics_structure(self, outcome, prepared):
        metrics = outcome.metrics
        protocol = metrics["protocol"]
        assert protocol["train"]["seasons"] == [2020, 2021]
        assert protocol["validation"]["season"] == VAL
        assert protocol["test"]["season"] == TEST
        assert protocol["final_fit"]["seasons"] == [2020, 2021, VAL]
        assert protocol["final_refit_through"] is None
        assert protocol["test_mae_is_out_of_sample"] is True
        rows = protocol["train"]["rows"] + protocol["validation"]["rows"]
        assert protocol["final_fit"]["rows"] == rows
        assert metrics["feature_columns"] == FEATURE_COLUMNS
        assert metrics["data"]["last_played_season"] == 2023
        assert metrics["data"]["n_player_rows"] == len(prepared[0])
        assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", metrics["model_version"])
        assert set(metrics["library_versions"]) >= {"python", "numpy", "pandas", "xgboost"}
        assert len(metrics["feature_importance_top15"]) == 15

    def test_all_models_and_baselines_are_compared_for_both_targets(self, outcome):
        comparison = outcome.metrics["comparison_mae"]
        expected = {
            "global_mean",
            "global_median",
            "naive_rolling5",
            "naive_season_mean",
            "ridge",
            "xgb_absoluteerror",
        }
        for target in ("fantasy", "pir"):
            assert set(comparison[target]) == expected
            for values in comparison[target].values():
                assert values["validation"] > 0 and values["test"] > 0

    def test_fantasy_and_pir_metrics(self, outcome):
        for target in ("fantasy", "pir"):
            for split in ("validation", "test"):
                values = outcome.metrics[target][split]
                assert set(values) == {"n", "mae", "rmse", "r2", "bias"}
                assert values["rmse"] >= values["mae"] > 0
        assert (
            outcome.metrics["fantasy"]["test"]["n"] == outcome.metrics["protocol"]["test"]["rows"]
        )

    def test_the_model_beats_the_best_naive_baseline_on_synthetic_data(self, outcome):
        check = outcome.metrics["threshold"]
        assert check["beats_best_naive_baseline"]
        assert check["best_naive_baseline"] in T.BASELINE_NAMES
        assert check["test_mae"] == outcome.metrics["fantasy"]["test"]["mae"]

    def test_segments_and_bootstrap(self, outcome):
        segments = outcome.metrics["segments"]["test"]
        assert set(segments) == {"minutes_mean5", "prior_appearances", "phase"}
        assert set(segments["prior_appearances"]) <= {"<5 prior", ">=5 prior"}
        assert (
            sum(part["n"] for part in segments["minutes_mean5"].values())
            == (outcome.metrics["protocol"]["test"]["rows"])
        )
        boot = outcome.metrics["bootstrap"]["test"]["model_minus_baseline"]
        assert boot["ci_low"] <= boot["difference"] <= boot["ci_high"]
        assert boot["iterations"] == 60
        assert boot["difference"] < 0  # το μοντέλο είναι καλύτερο από το baseline

    def test_metrics_are_json_serialisable_without_nan(self, outcome):
        text = json.dumps(outcome.metrics, allow_nan=False)
        assert json.loads(text)["model_version"] == outcome.metrics["model_version"]

    def test_the_bundle_predicts_both_targets(self, outcome, prepared):
        frame, _ = prepared
        sample = frame[frame["is_appearance"]].head(50)
        predictions = outcome.bundle.predict(sample)
        assert list(predictions.columns) == ["fantasy", "pir"]
        assert np.isfinite(predictions.to_numpy()).all()
        assert outcome.bundle.feature_columns == FEATURE_COLUMNS

    def test_selection_uses_only_the_validation_error(self, outcome):
        candidates = outcome.metrics["search"]["candidates"]
        best = min(candidates, key=lambda candidate: candidate["validation_mae"])
        assert outcome.metrics["selected_model"]["name"] == best["name"]
        assert outcome.metrics["selected_model"]["validation_mae"] == pytest.approx(
            best["validation_mae"], abs=1e-6
        )


# --------------------------------------------------------------------------------------
# Η σεζόν test δεν επηρεάζει την επιλογή ή το τελικό μοντέλο
# --------------------------------------------------------------------------------------


class TestTestSeasonIsolation:
    def test_changing_the_test_season_does_not_change_selection_or_the_final_model(
        self, prepared, outcome, tmp_path
    ):
        frame, info = prepared
        mutated = frame.copy()
        in_test = mutated["season"] == TEST
        mutated.loc[in_test, "fantasy_score"] = mutated.loc[in_test, "fantasy_score"] + 37.0
        mutated.loc[in_test, "pir"] = -mutated.loc[in_test, "pir"]
        mutated.loc[in_test, "pir_mean_5"] = 99.0
        mutated.loc[in_test, "min_mean_5"] = 1.0
        other = T.run_training(
            mutated, info, make_config(tmp_path), threshold=1000.0, progress=lambda message: None
        )
        assert other.metrics["selected_model"] == outcome.metrics["selected_model"]
        assert other.metrics["search"] == outcome.metrics["search"]
        for target in ("fantasy", "pir"):
            assert signature(other.bundle.models[target]) == signature(
                outcome.bundle.models[target]
            )
            assert other.metrics[target]["validation"] == outcome.metrics[target]["validation"]
        # Μόνο οι μετρικές του test αλλάζουν
        assert other.metrics["fantasy"]["test"]["mae"] != outcome.metrics["fantasy"]["test"]["mae"]

    def test_the_test_season_is_not_part_of_any_training_set(self, outcome):
        protocol = outcome.metrics["protocol"]
        assert TEST not in protocol["train"]["seasons"]
        assert TEST not in protocol["final_fit"]["seasons"]
        assert VAL not in protocol["train"]["seasons"]
        assert max(protocol["final_fit"]["seasons"]) == protocol["final_fit_through_season"] == VAL


# --------------------------------------------------------------------------------------
# Επαναληψιμότητα
# --------------------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_gives_identical_results(self, prepared, outcome, tmp_path):
        frame, info = prepared
        again = T.run_training(
            frame, info, make_config(tmp_path), threshold=1000.0, progress=lambda message: None
        )
        assert again.metrics["fantasy"] == outcome.metrics["fantasy"]
        assert again.metrics["pir"] == outcome.metrics["pir"]
        assert again.metrics["comparison_mae"] == outcome.metrics["comparison_mae"]
        assert again.metrics["selected_model"] == outcome.metrics["selected_model"]
        for target in ("fantasy", "pir"):
            assert signature(again.bundle.models[target]) == signature(
                outcome.bundle.models[target]
            )

    def test_the_seed_is_used_by_the_search(self):
        first = T.candidate_specs(T.TrainConfig(tune=True, seed=1))
        second = T.candidate_specs(T.TrainConfig(tune=True, seed=2))
        assert [s.params for s in first] != [s.params for s in second]

    def test_the_model_version_depends_on_the_data_and_parameters(
        self, prepared, outcome, tmp_path
    ):
        frame, info = prepared
        other_seed = T.run_training(
            frame, info, make_config(tmp_path, seed=7), threshold=1000.0, progress=lambda m: None
        )
        assert (
            other_seed.metrics["model_version"].split("-")[1]
            != (outcome.metrics["model_version"].split("-")[1])
        )


# --------------------------------------------------------------------------------------
# Κανόνας threshold και εγγραφή αρχείων
# --------------------------------------------------------------------------------------


class TestThresholdRule:
    def test_a_passing_model_writes_the_artifact_and_the_metrics(self, prepared, tmp_path):
        frame, info = prepared
        code = T.train_and_save(
            make_config(tmp_path, threshold=1000.0), frame, info, progress=lambda m: None
        )
        assert code == T.EXIT_OK
        assert sorted(path.name for path in tmp_path.iterdir()) == ["metrics.json", "model.joblib"]
        metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        bundle = load_bundle(tmp_path / "model.joblib")
        assert bundle.model_version == metrics["model_version"]
        assert bundle.metrics["threshold"]["passed"] is True
        assert metrics["threshold"]["value"] == 1000.0

    def test_a_failing_model_exits_with_3_and_leaves_existing_files_untouched(
        self, prepared, tmp_path
    ):
        frame, info = prepared
        model_path, metrics_path = tmp_path / "model.joblib", tmp_path / "metrics.json"
        model_path.write_bytes(b"old model bytes \x00\x01")
        metrics_path.write_bytes(b'{"old": true}')
        before = (model_path.read_bytes(), metrics_path.read_bytes())
        code = T.train_and_save(
            make_config(tmp_path, threshold=0.01), frame, info, progress=lambda m: None
        )
        assert code == T.EXIT_BELOW_THRESHOLD == 3
        assert (model_path.read_bytes(), metrics_path.read_bytes()) == before
        assert sorted(path.name for path in tmp_path.iterdir()) == ["metrics.json", "model.joblib"]

    def test_a_failing_model_does_not_create_files_either(self, prepared, tmp_path):
        frame, info = prepared
        out = tmp_path / "models"
        code = T.train_and_save(make_config(out, threshold=0.01), frame, info, lambda m: None)
        assert code == 3
        assert not out.exists()

    def test_the_threshold_is_strict(self, prepared, outcome, tmp_path):
        frame, info = prepared
        mae = outcome.metrics["fantasy"]["test"]["mae"]
        # Το όριο λίγο κάτω από το MAE αποτυγχάνει, λίγο πάνω περνά (ο κανόνας είναι MAE < όριο).
        below = make_config(tmp_path / "below", threshold=mae - 1e-3)
        assert T.train_and_save(below, frame, info, lambda m: None) == 3
        above = make_config(tmp_path / "above", threshold=mae + 1e-3)
        assert T.train_and_save(above, frame, info, lambda m: None) == 0

    @pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
    def test_an_unset_threshold_is_an_error(self, value):
        with pytest.raises(T.TrainingError, match="no MAE threshold"):
            T.resolve_threshold(value)

    def test_the_threshold_falls_back_to_the_setting(self, monkeypatch):
        monkeypatch.setenv("MAE_THRESHOLD", "4.25")
        get_settings.cache_clear()
        assert T.resolve_threshold(None) == 4.25
        assert T.resolve_threshold(3.0) == 3.0  # η γραμμή εντολών υπερισχύει
        monkeypatch.setenv("MAE_THRESHOLD", "0")
        get_settings.cache_clear()
        with pytest.raises(T.TrainingError, match="no MAE threshold"):
            T.resolve_threshold(None)

    def test_a_failure_while_writing_leaves_the_existing_files_intact(self, tmp_path):
        model_path, metrics_path = tmp_path / "model.joblib", tmp_path / "metrics.json"
        model_path.write_bytes(b"old model")
        blocked = tmp_path / "blocked"
        blocked.write_bytes(b"i am a file, not a folder")
        with pytest.raises(OSError):
            atomic_write_many({model_path: b"new model", blocked / "metrics.json": b"new metrics"})
        assert model_path.read_bytes() == b"old model"
        assert not metrics_path.exists()
        assert [path.name for path in tmp_path.iterdir() if path.name.endswith(".tmp")] == []

    def test_a_successful_write_replaces_the_files_and_leaves_no_temporary_ones(self, tmp_path):
        target = tmp_path / "model.joblib"
        target.write_bytes(b"old")
        atomic_write_many({target: b"new", tmp_path / "metrics.json": b"{}"})
        assert target.read_bytes() == b"new"
        assert sorted(path.name for path in tmp_path.iterdir()) == ["metrics.json", "model.joblib"]


class TestFinalRefit:
    def test_refit_through_a_later_season_is_recorded_and_changes_the_artifact(
        self, prepared, tmp_path
    ):
        frame, info = prepared
        honest_dir, refit_dir = tmp_path / "honest", tmp_path / "refit"
        options = {"threshold": 1000.0, "val_season": 2021, "test_season": 2022}
        honest_config = make_config(honest_dir, **options)
        refit_config = make_config(refit_dir, final_refit_through=2023, **options)
        assert T.train_and_save(honest_config, frame, info, lambda m: None) == 0
        assert T.train_and_save(refit_config, frame, info, lambda m: None) == 0
        metrics = json.loads((refit_dir / "metrics.json").read_text(encoding="utf-8"))
        protocol = metrics["protocol"]
        assert protocol["final_refit_through"] == 2023
        assert protocol["test_mae_is_out_of_sample"] is False
        assert protocol["final_fit_through_season"] == 2023
        assert "note" in protocol
        honest = load_bundle(honest_dir / "model.joblib")
        refit = load_bundle(refit_dir / "model.joblib")
        sample = frame[frame["is_appearance"]].head(30)
        assert not np.allclose(
            honest.predict(sample)["fantasy"].to_numpy(),
            refit.predict(sample)["fantasy"].to_numpy(),
        )

    def test_the_default_is_no_refit(self, outcome):
        assert outcome.metrics["protocol"]["final_refit_through"] is None
        assert outcome.metrics["protocol"]["test_mae_is_out_of_sample"] is True

    def test_the_refit_season_must_not_precede_the_test_season(self, prepared, tmp_path):
        frame, info = prepared
        config = make_config(tmp_path, threshold=1000.0, final_refit_through=VAL)
        with pytest.raises(T.TrainingError, match="final-refit-through"):
            T.train_and_save(config, frame, info, lambda m: None)

    def test_the_refit_is_skipped_when_the_threshold_fails(self, prepared, tmp_path):
        frame, info = prepared
        config = make_config(tmp_path / "x", threshold=0.01, final_refit_through=TEST)
        assert T.train_and_save(config, frame, info, lambda m: None) == 3
        assert not (tmp_path / "x").exists()


# --------------------------------------------------------------------------------------
# Ridge και artifact
# --------------------------------------------------------------------------------------


class TestModelsAndArtifact:
    def test_the_numpy_ridge_equals_the_sklearn_pipeline(self, prepared):
        frame, _ = prepared
        rows = frame[frame["is_appearance"]]
        features = rows[FEATURE_COLUMNS].to_numpy(float)
        target = rows["fantasy_score"].to_numpy(float)
        model = T.fit_ridge(features, target, alpha=25.0)
        reference = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            Ridge(alpha=25.0),
        ).fit(features, target)
        assert np.isnan(features).any()  # το δείγμα έχει NaN: ελέγχεται η συμπλήρωση και οι δείκτες
        np.testing.assert_allclose(model.predict(features), reference.predict(features), atol=1e-8)
        # Και σε γραμμή με NaN σε στήλη που στην εκπαίδευση δεν είχε NaN:
        probe = features[:3].copy()
        probe[:, FEATURE_COLUMNS.index("home")] = np.nan
        np.testing.assert_allclose(model.predict(probe), reference.predict(probe), atol=1e-8)

    def test_ridge_importance_is_normalised(self, prepared):
        frame, _ = prepared
        rows = frame[frame["is_appearance"]]
        model = T.fit_ridge(
            rows[FEATURE_COLUMNS].to_numpy(float), rows["pir"].to_numpy(float), 10.0
        )
        importance = model.importance(FEATURE_COLUMNS)
        assert sum(importance.values()) == pytest.approx(1.0)

    def test_the_artifact_roundtrips_and_keeps_the_predictions(self, outcome, prepared):
        frame, _ = prepared
        bundle = outcome.bundle
        bundle.metrics = outcome.metrics
        restored = deserialize_bundle(serialize_bundle(bundle))
        sample = frame.head(200)
        pd.testing.assert_frame_equal(bundle.predict(sample), restored.predict(sample))
        assert restored.model_version == bundle.model_version
        assert restored.feature_columns == FEATURE_COLUMNS
        assert restored.metrics["threshold"] == outcome.metrics["threshold"]

    def test_the_xgboost_model_is_stored_in_the_native_format(self, outcome):
        model = outcome.bundle.models["fantasy"]
        assert isinstance(model, XGBoostModel)
        assert model.raw[:1] == b"{"  # Universal Binary JSON του XGBoost
        assert model.n_estimators == outcome.metrics["selected_model"]["n_estimators"]
        assert isinstance(outcome.bundle, ModelBundle)

    def test_a_ridge_artifact_also_roundtrips(self, prepared):
        frame, _ = prepared
        rows = frame[frame["is_appearance"]]
        features = rows[FEATURE_COLUMNS].to_numpy(float)
        models = {
            "fantasy": T.fit_ridge(features, rows["fantasy_score"].to_numpy(float), 10.0),
            "pir": T.fit_ridge(features, rows["pir"].to_numpy(float), 10.0),
        }
        bundle = ModelBundle("v1", list(FEATURE_COLUMNS), models, {"python": "3"}, "now", {})
        restored = deserialize_bundle(serialize_bundle(bundle))
        assert isinstance(restored.models["fantasy"], RidgeModel)
        sample = frame.head(100)
        pd.testing.assert_frame_equal(bundle.predict(sample), restored.predict(sample))


# --------------------------------------------------------------------------------------
# Η γραμμή εντολών πάνω σε βάση SQLite
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_url(league, tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "league.db"
    url = f"sqlite:///{path.as_posix()}"
    engine = get_engine(url)
    write_to_database(engine, league)
    engine.dispose()
    return url


class TestCommandLine:
    def args(self, db_url, out_dir, *extra):
        return [
            "--db",
            db_url,
            "--val-season",
            str(VAL),
            "--test-season",
            str(TEST),
            "--out-dir",
            str(out_dir),
            "--no-tune",
            "--bootstrap",
            "50",
            *extra,
        ]

    def test_exit_code_0_and_artifacts(self, db_url, tmp_path, restore_logging):
        code = T.main(self.args(db_url, tmp_path, "--threshold", "1000"))
        assert code == 0
        assert (tmp_path / "model.joblib").stat().st_size > 0
        metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["data"]["last_played_season"] == 2023
        assert metrics["data"]["last_game_date"] is not None

    def test_exit_code_3_below_the_threshold(self, db_url, tmp_path, restore_logging, capsys):
        code = T.main(self.args(db_url, tmp_path / "out", "--threshold", "0.05"))
        assert code == 3
        assert not (tmp_path / "out").exists()
        assert "threshold NOT met" in capsys.readouterr().out

    @pytest.mark.parametrize("flag", [["--threshold", "0"], ["--threshold", "-2"]])
    def test_exit_code_1_when_the_threshold_is_not_set(
        self, db_url, tmp_path, restore_logging, capsys, flag
    ):
        code = T.main(self.args(db_url, tmp_path / "out", *flag))
        assert code == 1
        assert "no MAE threshold" in capsys.readouterr().out
        assert not (tmp_path / "out").exists()

    def test_the_setting_is_used_when_the_flag_is_missing(
        self, db_url, tmp_path, restore_logging, monkeypatch
    ):
        monkeypatch.setenv("MAE_THRESHOLD", "1000")
        get_settings.cache_clear()
        assert T.main(self.args(db_url, tmp_path)) == 0
        monkeypatch.setenv("MAE_THRESHOLD", "0")
        get_settings.cache_clear()
        assert T.main(self.args(db_url, tmp_path / "second")) == 1

    def test_exit_code_1_on_an_empty_database(self, tmp_path, restore_logging, capsys):
        url = f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"
        from elfantasy.db.session import create_all

        engine = get_engine(url)
        create_all(engine)
        engine.dispose()
        code = T.main(self.args(url, tmp_path / "out", "--threshold", "1000"))
        assert code == 1
        assert "no player rows" in capsys.readouterr().out

    def test_exit_code_1_on_bad_seasons(self, db_url, tmp_path, restore_logging):
        args = [
            "--db",
            db_url,
            "--threshold",
            "1000",
            "--val-season",
            "2030",
            "--test-season",
            "2031",
        ]
        assert T.main([*args, "--no-tune", "--out-dir", str(tmp_path / "out")]) == 1

    def test_parser_defaults(self):
        args = T.build_parser().parse_args([])
        assert (args.val_season, args.test_season, args.seed) == (2024, 2025, 42)
        assert args.threshold is None and not args.no_tune and args.final_refit_through is None
        assert str(args.out_dir) == "models"
