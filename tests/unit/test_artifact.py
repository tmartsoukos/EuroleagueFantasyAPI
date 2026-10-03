"""Tests του artifact του μοντέλου (`elfantasy.model.artifact`): αποθήκευση, φόρτωση, έλεγχοι.

Το artifact πρέπει να φορτώνεται από το ίδιο το αρχείο, να ελέγχει εκδόσεις βιβλιοθηκών και να
αποτυγχάνει με σαφές μήνυμα (`ModelLoadError`) όταν είναι κατεστραμμένο ή ασύμβατο.
"""

from __future__ import annotations

import io
import logging
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

from elfantasy.features.build import FEATURE_COLUMNS
from elfantasy.model import artifact as A
from elfantasy.model.artifact import (
    ModelBundle,
    ModelLoadError,
    ModelVersionWarning,
    XGBoostModel,
    deserialize_bundle,
    load_bundle,
    save_bundle,
    serialize_bundle,
)


@pytest.fixture
def saved(tiny_xgb_bundle, tmp_path):
    path = tmp_path / "model.joblib"
    save_bundle(tiny_xgb_bundle, path)
    return path


def payload_of(bundle: ModelBundle) -> dict:
    return bundle.to_payload()


def dump(payload, path):
    joblib.dump(payload, path, compress=("zlib", 3))


class TestRoundtrip:
    def test_save_and_load_keep_the_predictions(self, tiny_xgb_bundle, synthetic_frame, saved):
        loaded = load_bundle(saved)
        sample = synthetic_frame.head(120)
        pd.testing.assert_frame_equal(tiny_xgb_bundle.predict(sample), loaded.predict(sample))
        assert loaded.model_version == "test-xgb-v1"
        assert loaded.feature_columns == FEATURE_COLUMNS
        assert loaded.metrics["threshold"]["value"] == 9.99
        assert loaded.library_versions == tiny_xgb_bundle.library_versions

    def test_saving_is_atomic_and_leaves_no_temporary_file(self, tiny_xgb_bundle, tmp_path):
        target = tmp_path / "models" / "model.joblib"
        size = save_bundle(tiny_xgb_bundle, target)
        assert size == target.stat().st_size > 0
        assert [path.name for path in target.parent.iterdir()] == ["model.joblib"]

    def test_the_artifact_stores_plain_data_only(self, saved):
        payload = joblib.load(saved)
        assert payload["format"] == "elfantasy-model" and payload["format_version"] == 1
        model = payload["models"]["fantasy"]
        assert model["kind"] == "xgboost" and isinstance(model["raw"], bytes)
        assert model["format"] == "ubj"
        classes = {type(value).__module__.split(".")[0] for value in model.values()}
        assert "xgboost" not in classes and "sklearn" not in classes

    def test_a_missing_feature_column_is_a_clear_error(self, tiny_xgb_bundle, synthetic_frame):
        with pytest.raises(ValueError, match="missing model columns"):
            tiny_xgb_bundle.predict(synthetic_frame.drop(columns=["pir_mean_5"]))

    def test_the_size_of_the_tiny_artifact_is_small(self, saved):
        assert saved.stat().st_size < 500_000

    def test_library_versions_are_reported_without_importing_the_libraries(self):
        versions = A.library_versions()
        assert {"python", "numpy", "pandas", "scikit-learn", "xgboost"} <= set(versions)
        assert all(isinstance(value, str) for value in versions.values())


class TestCorruptedOrIncompatibleFiles:
    def test_a_missing_file(self, tmp_path):
        with pytest.raises(ModelLoadError, match="not found"):
            load_bundle(tmp_path / "missing.joblib")

    def test_random_bytes(self, tmp_path):
        path = tmp_path / "model.joblib"
        path.write_bytes(np.random.default_rng(0).bytes(5000))
        with pytest.raises(ModelLoadError, match="corrupted or unreadable"):
            load_bundle(path)

    def test_an_empty_file(self, tmp_path):
        path = tmp_path / "model.joblib"
        path.write_bytes(b"")
        with pytest.raises(ModelLoadError, match="corrupted or unreadable"):
            load_bundle(path)

    def test_a_truncated_file(self, saved):
        data = saved.read_bytes()
        saved.write_bytes(data[: len(data) // 2])
        with pytest.raises(ModelLoadError, match="corrupted or unreadable"):
            load_bundle(saved)

    def test_a_flipped_byte(self, saved):
        data = bytearray(saved.read_bytes())
        data[len(data) // 2] ^= 0xFF
        saved.write_bytes(bytes(data))
        with pytest.raises(ModelLoadError):
            load_bundle(saved)

    def test_another_kind_of_joblib_file(self, tmp_path):
        path = tmp_path / "other.joblib"
        dump({"hello": "world"}, path)
        with pytest.raises(ModelLoadError, match="not an 'elfantasy-model'"):
            load_bundle(path)
        dump([1, 2, 3], path)
        with pytest.raises(ModelLoadError, match="not an 'elfantasy-model'"):
            load_bundle(path)

    def test_an_unsupported_format_version(self, tiny_xgb_bundle, tmp_path):
        payload = payload_of(tiny_xgb_bundle)
        payload["format_version"] = 99
        dump(payload, tmp_path / "m.joblib")
        with pytest.raises(ModelLoadError, match="format version 99"):
            load_bundle(tmp_path / "m.joblib")

    @pytest.mark.parametrize("field", ["model_version", "feature_columns", "library_versions"])
    def test_missing_fields(self, tiny_xgb_bundle, tmp_path, field):
        payload = payload_of(tiny_xgb_bundle)
        del payload[field]
        dump(payload, tmp_path / "m.joblib")
        with pytest.raises(ModelLoadError, match="missing fields"):
            load_bundle(tmp_path / "m.joblib")

    def test_a_missing_target_model(self, tiny_xgb_bundle, tmp_path):
        payload = payload_of(tiny_xgb_bundle)
        del payload["models"]["pir"]
        dump(payload, tmp_path / "m.joblib")
        with pytest.raises(ModelLoadError, match="must contain models"):
            load_bundle(tmp_path / "m.joblib")

    def test_an_unknown_model_kind(self, tiny_xgb_bundle, tmp_path):
        payload = payload_of(tiny_xgb_bundle)
        payload["models"]["fantasy"]["kind"] = "lightgbm"
        dump(payload, tmp_path / "m.joblib")
        with pytest.raises(ModelLoadError, match="unknown model kind"):
            load_bundle(tmp_path / "m.joblib")

    def test_an_unreadable_xgboost_model(self, tiny_xgb_bundle, tmp_path):
        payload = payload_of(tiny_xgb_bundle)
        payload["models"]["fantasy"]["raw"] = b"this is not an xgboost model"
        dump(payload, tmp_path / "m.joblib")
        with pytest.raises(ModelLoadError, match="could not be loaded"):
            load_bundle(tmp_path / "m.joblib")

    def test_an_unsupported_xgboost_payload_format(self, tiny_xgb_bundle, tmp_path):
        payload = payload_of(tiny_xgb_bundle)
        payload["models"]["fantasy"]["format"] = "pickle"
        dump(payload, tmp_path / "m.joblib")
        with pytest.raises(ModelLoadError, match="unsupported XGBoost payload format"):
            load_bundle(tmp_path / "m.joblib")

    def test_the_error_message_tells_how_to_retrain(self, tmp_path):
        path = tmp_path / "model.joblib"
        path.write_bytes(b"garbage")
        with pytest.raises(ModelLoadError, match="elfantasy.model.train"):
            load_bundle(path)


class TestLibraryVersionChecks:
    def test_no_warning_when_the_versions_match(self, saved):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ModelVersionWarning)
            load_bundle(saved)

    def test_a_different_xgboost_minor_version_warns_clearly(
        self, tiny_xgb_bundle, tmp_path, monkeypatch, caplog
    ):
        payload = payload_of(tiny_xgb_bundle)
        payload["library_versions"] = {**payload["library_versions"], "xgboost": "1.7.6"}
        dump(payload, tmp_path / "m.joblib")
        with caplog.at_level(logging.WARNING, logger="elfantasy.model.artifact"):
            with pytest.warns(ModelVersionWarning, match=r"xgboost: model trained with 1\.7\.6"):
                bundle = load_bundle(tmp_path / "m.joblib")
        assert "Re-train" in caplog.text or "re-train" in caplog.text
        assert bundle.model_version == "test-xgb-v1"  # φορτώνει παρά την προειδοποίηση

    def test_a_different_scikit_learn_version_warns(self, tiny_xgb_bundle, tmp_path):
        payload = payload_of(tiny_xgb_bundle)
        payload["library_versions"] = {**payload["library_versions"], "scikit-learn": "0.24.2"}
        dump(payload, tmp_path / "m.joblib")
        with pytest.warns(ModelVersionWarning, match="scikit-learn"):
            load_bundle(tmp_path / "m.joblib")

    def test_a_patch_difference_does_not_warn(self, tiny_xgb_bundle, tmp_path):
        current = A.library_versions()["xgboost"]
        major, minor = current.split(".")[:2]
        payload = payload_of(tiny_xgb_bundle)
        payload["library_versions"] = {
            **payload["library_versions"],
            "xgboost": f"{major}.{minor}.0",
        }
        dump(payload, tmp_path / "m.joblib")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ModelVersionWarning)
            load_bundle(tmp_path / "m.joblib")

    def test_other_libraries_do_not_warn(self, tiny_xgb_bundle, tmp_path):
        payload = payload_of(tiny_xgb_bundle)
        payload["library_versions"] = {
            **payload["library_versions"],
            "numpy": "1.0.0",
            "pandas": "1.0",
        }
        dump(payload, tmp_path / "m.joblib")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ModelVersionWarning)
            load_bundle(tmp_path / "m.joblib")

    def test_version_mismatch_helper(self):
        saved = {"xgboost": "3.2.1", "scikit-learn": "1.8.0", "numpy": "1.0"}
        same = {"xgboost": "3.2.9", "scikit-learn": "1.8.4", "numpy": "9.9"}
        assert A.version_mismatches(saved, same) == []
        different = {"xgboost": "3.4.1", "scikit-learn": "1.9.1"}
        problems = A.version_mismatches(saved, different)
        assert len(problems) == 2 and "xgboost" in problems[0]
        assert (
            A.version_mismatches({}, different) == []
        )  # άγνωστη έκδοση εκπαίδευσης: καμία ειδοποίηση


class TestXGBoostModel:
    def test_predictions_are_deterministic_and_independent_of_the_row_order(self, tiny_xgb_bundle):
        model = tiny_xgb_bundle.models["fantasy"]
        assert isinstance(model, XGBoostModel)
        matrix = np.random.default_rng(1).normal(size=(40, len(FEATURE_COLUMNS)))
        matrix[::3, 5] = np.nan
        first = model.predict(matrix)
        np.testing.assert_array_equal(first, model.predict(matrix))
        np.testing.assert_allclose(model.predict(matrix[::-1]), first[::-1])

    def test_nan_features_give_finite_predictions(self, tiny_xgb_bundle, tiny_ridge_bundle):
        all_nan = np.full((3, len(FEATURE_COLUMNS)), np.nan)
        for bundle in (tiny_xgb_bundle, tiny_ridge_bundle):
            for target in ("fantasy", "pir"):
                assert np.isfinite(bundle.predict_target(target, all_nan)).all()

    def test_importance_is_normalised(self, tiny_xgb_bundle):
        importance = tiny_xgb_bundle.models["fantasy"].importance(FEATURE_COLUMNS)
        assert importance
        assert sum(importance.values()) == pytest.approx(1.0)
        assert set(importance) <= set(FEATURE_COLUMNS)

    def test_serialisation_is_stable_across_calls(self, tiny_xgb_bundle):
        # Δεν μπορούμε να υποθέσουμε ίδια bytes (το joblib μπορεί να βάζει μεταδεδομένα), αλλά
        # η αποσειριοποίηση πρέπει να δίνει πάντα το ίδιο μοντέλο.
        first = deserialize_bundle(serialize_bundle(tiny_xgb_bundle))
        second = deserialize_bundle(io.BytesIO(serialize_bundle(tiny_xgb_bundle)).getvalue())
        assert first.models["fantasy"].raw == second.models["fantasy"].raw
