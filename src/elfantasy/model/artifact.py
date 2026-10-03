"""Μορφή, αποθήκευση και φόρτωση του μοντέλου (artifact `models/model.joblib`).

Το artifact είναι ένα `dict` από απλούς τύπους Python, αριθμούς numpy και bytes, που
αποθηκεύεται με το joblib (συμπίεση zlib). ΔΕΝ περιέχει αντικείμενα του scikit-learn ή του
XGBoost, ώστε να μην εξαρτάται από εσωτερικά των βιβλιοθηκών ή από τα ονόματα των κλάσεών μας:

* XGBoost: το μοντέλο αποθηκεύεται στην εγγενή μορφή του (`Booster.save_raw("ubj")`, Universal
  Binary JSON), που διαβάζεται από μεταγενέστερες εκδόσεις.
* Ridge: αποθηκεύονται οι παράμετροι (διάμεσοι για συμπλήρωση NaN, δείκτες NaN, μέσος και
  κλίμακα, συντελεστές) και η πρόβλεψη υπολογίζεται με numpy.

Στη φόρτωση ελέγχονται η μορφή και οι εκδόσεις των βιβλιοθηκών. Αν οι εκδόσεις xgboost ή
scikit-learn διαφέρουν (major.minor) από όσες καταγράφηκαν στην εκπαίδευση, εκδίδεται
`ModelVersionWarning` και μήνυμα στο log, και το μοντέλο ελέγχεται με δοκιμαστική πρόβλεψη. Αν το
αρχείο είναι κατεστραμμένο ή ασύμβατο, σηκώνεται `ModelLoadError` με σαφές μήνυμα.

Το joblib (pickle) δεν είναι ασφαλές για αρχεία από άγνωστη πηγή: το artifact είναι αρχείο του
ίδιου του repo και φορτώνεται μόνο από αυτό.
"""

from __future__ import annotations

import io
import logging
import os
import platform
import warnings
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ARTIFACT_FORMAT = "elfantasy-model"
FORMAT_VERSION = 1
TARGETS = ("fantasy", "pir")
#: Βιβλιοθήκες των οποίων η διαφορά major.minor δίνει προειδοποίηση στη φόρτωση.
CHECKED_LIBRARIES = ("xgboost", "scikit-learn")
TRACKED_LIBRARIES = ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "joblib")


class ModelLoadError(RuntimeError):
    """Το αρχείο του μοντέλου λείπει, είναι κατεστραμμένο ή ασύμβατο με τον τρέχοντα κώδικα."""


class ModelVersionWarning(UserWarning):
    """Οι εκδόσεις βιβλιοθηκών της φόρτωσης διαφέρουν από αυτές της εκπαίδευσης."""


def library_versions() -> dict[str, str]:
    """Εκδόσεις Python και βιβλιοθηκών, χωρίς να εισάγονται (π.χ. χωρίς import του sklearn)."""
    versions = {"python": platform.python_version()}
    for name in TRACKED_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _major_minor(version: str) -> tuple[str, str] | None:
    parts = str(version).split(".")
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def version_mismatches(saved: dict[str, str], current: dict[str, str]) -> list[str]:
    """Μηνύματα για τις ελεγχόμενες βιβλιοθήκες που διαφέρουν σε major.minor."""
    problems = []
    for name in CHECKED_LIBRARIES:
        before, now = saved.get(name), current.get(name)
        if before is None or now is None:
            continue
        if _major_minor(before) != _major_minor(now):
            problems.append(f"{name}: model trained with {before}, loaded with {now}")
    return problems


# --------------------------------------------------------------------------------------
# Μοντέλα ανά στόχο
# --------------------------------------------------------------------------------------


@dataclass
class RidgeModel:
    """Ridge με συμπλήρωση NaN (διάμεσος) και δείκτες NaN, αποθηκευμένο ως παράμετροι numpy.

    Η πρόβλεψη: (1) τα NaN αντικαθίστανται με τον διάμεσο της εκπαίδευσης, (2) προστίθενται
    στήλες-δείκτες (1 όπου το feature έλειπε) για τα features που είχαν NaN στην εκπαίδευση,
    (3) τυποποίηση με μέσο και κλίμακα της εκπαίδευσης, (4) γραμμικός συνδυασμός. Ισοδυναμεί με
    `make_pipeline(SimpleImputer("median", add_indicator=True), StandardScaler(), Ridge(alpha))`.
    """

    medians: np.ndarray
    indicator_columns: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    alpha: float
    kind: str = field(default="ridge", init=False)

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        missing = np.isnan(values)
        filled = np.where(missing, self.medians, values)
        indicators = missing[:, self.indicator_columns].astype(float)
        design = np.hstack([filled, indicators])
        return ((design - self.mean) / self.scale) @ self.coef + self.intercept

    def importance(self, columns: list[str]) -> dict[str, float]:
        """Σχετική σημασία: |τυποποιημένος συντελεστής| κανονικοποιημένος στο 1."""
        magnitude = np.abs(self.coef[: len(columns)])
        total = float(magnitude.sum())
        if total == 0:
            return {}
        return {name: float(value / total) for name, value in zip(columns, magnitude, strict=True)}

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "ridge",
            "alpha": float(self.alpha),
            "medians": np.asarray(self.medians, dtype=float),
            "indicator_columns": np.asarray(self.indicator_columns, dtype=np.int64),
            "mean": np.asarray(self.mean, dtype=float),
            "scale": np.asarray(self.scale, dtype=float),
            "coef": np.asarray(self.coef, dtype=float),
            "intercept": float(self.intercept),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RidgeModel:
        return cls(
            medians=np.asarray(payload["medians"], dtype=float),
            indicator_columns=np.asarray(payload["indicator_columns"], dtype=np.int64),
            mean=np.asarray(payload["mean"], dtype=float),
            scale=np.asarray(payload["scale"], dtype=float),
            coef=np.asarray(payload["coef"], dtype=float),
            intercept=float(payload["intercept"]),
            alpha=float(payload["alpha"]),
        )


class XGBoostModel:
    """Μοντέλο XGBoost σε εγγενή μορφή (UBJ bytes). Το `Booster` φορτώνεται όταν χρειαστεί."""

    kind = "xgboost"

    def __init__(
        self,
        raw: bytes,
        feature_columns: list[str],
        params: dict[str, Any] | None = None,
        n_estimators: int | None = None,
    ):
        self.raw = bytes(raw)
        self.feature_columns = list(feature_columns)
        self.params = dict(params or {})
        self.n_estimators = n_estimators
        self._booster = None

    @property
    def booster(self):
        if self._booster is None:
            import xgboost as xgb

            booster = xgb.Booster()
            booster.load_model(bytearray(self.raw))
            self._booster = booster
        return self._booster

    def predict(self, features: np.ndarray) -> np.ndarray:
        import xgboost as xgb

        matrix = xgb.DMatrix(
            np.asarray(features, dtype=np.float32),
            feature_names=self.feature_columns,
            missing=np.nan,
        )
        return self.booster.predict(matrix)

    def importance(self, columns: list[str]) -> dict[str, float]:
        """Σχετική σημασία: κέρδος (gain) ανά feature, κανονικοποιημένο ώστε να αθροίζει στο 1."""
        scores = self.booster.get_score(importance_type="gain")
        total = float(sum(scores.values()))
        if total == 0:
            return {}
        return {name: float(scores.get(name, 0.0) / total) for name in columns if name in scores}

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "xgboost",
            "format": "ubj",
            "raw": self.raw,
            "params": self.params,
            "n_estimators": self.n_estimators,
            "feature_columns": self.feature_columns,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> XGBoostModel:
        if payload.get("format") != "ubj":
            raise ModelLoadError(f"unsupported XGBoost payload format {payload.get('format')!r}")
        return cls(
            raw=payload["raw"],
            feature_columns=payload["feature_columns"],
            params=payload.get("params"),
            n_estimators=payload.get("n_estimators"),
        )


TargetModel = RidgeModel | XGBoostModel


def model_from_payload(payload: dict[str, Any]) -> TargetModel:
    kind = payload.get("kind")
    if kind == "ridge":
        return RidgeModel.from_payload(payload)
    if kind == "xgboost":
        return XGBoostModel.from_payload(payload)
    raise ModelLoadError(f"unknown model kind {kind!r}")


# --------------------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------------------


@dataclass
class ModelBundle:
    """Το πλήρες artifact: μοντέλα fantasy και PIR, features, έκδοση και μετρικές."""

    model_version: str
    feature_columns: list[str]
    models: dict[str, TargetModel]
    library_versions: dict[str, str]
    trained_at: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def feature_matrix(self, features: pd.DataFrame) -> np.ndarray:
        """Πίνακας float64 με τις στήλες του μοντέλου, με τη σειρά της εκπαίδευσης."""
        missing = [name for name in self.feature_columns if name not in features.columns]
        if missing:
            raise ValueError(f"features are missing model columns: {missing}")
        return features[self.feature_columns].to_numpy(dtype=float)

    def predict_target(self, target: str, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        matrix = self.feature_matrix(features) if isinstance(features, pd.DataFrame) else features
        return self.models[target].predict(matrix)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Προβλέψεις `fantasy` και `pir` για κάθε γραμμή του `features`."""
        matrix = self.feature_matrix(features)
        return pd.DataFrame(
            {target: self.models[target].predict(matrix) for target in TARGETS},
            index=features.index,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": ARTIFACT_FORMAT,
            "format_version": FORMAT_VERSION,
            "model_version": self.model_version,
            "feature_columns": list(self.feature_columns),
            "models": {name: model.to_payload() for name, model in self.models.items()},
            "library_versions": dict(self.library_versions),
            "trained_at": self.trained_at,
            "metrics": self.metrics,
        }


def serialize_bundle(bundle: ModelBundle) -> bytes:
    """Το artifact ως bytes (joblib, συμπίεση zlib επιπέδου 3)."""
    buffer = io.BytesIO()
    joblib.dump(bundle.to_payload(), buffer, compress=("zlib", 3))
    return buffer.getvalue()


def deserialize_bundle(data: bytes, source: str = "<bytes>") -> ModelBundle:
    """Ανασυνθέτει και ελέγχει ένα artifact από bytes. Σηκώνει `ModelLoadError` σε πρόβλημα."""
    try:
        payload = joblib.load(io.BytesIO(data))
    except Exception as error:
        raise ModelLoadError(
            f"model artifact {source} is corrupted or unreadable "
            f"({type(error).__name__}: {error}). "
            "Re-train it with `python -m elfantasy.model.train`."
        ) from error
    return _bundle_from_payload(payload, source)


def _bundle_from_payload(payload: Any, source: str) -> ModelBundle:
    if not isinstance(payload, dict) or payload.get("format") != ARTIFACT_FORMAT:
        raise ModelLoadError(f"{source} is not an {ARTIFACT_FORMAT!r} model artifact")
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ModelLoadError(
            f"model artifact {source} has format version {version!r}, "
            f"this code reads version {FORMAT_VERSION}. Re-train the model."
        )
    required = ("model_version", "feature_columns", "models", "library_versions", "trained_at")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ModelLoadError(f"model artifact {source} is missing fields: {missing}")
    models_payload = payload["models"]
    if not isinstance(models_payload, dict) or any(t not in models_payload for t in TARGETS):
        raise ModelLoadError(f"model artifact {source} must contain models for {list(TARGETS)}")
    saved_versions = dict(payload["library_versions"])
    current_versions = library_versions()
    for problem in version_mismatches(saved_versions, current_versions):
        message = (
            f"Library version mismatch for model artifact {source} -> {problem}. "
            "Predictions may differ or loading may fail; re-train the model in this environment."
        )
        logger.warning(message)
        warnings.warn(ModelVersionWarning(message), stacklevel=3)
    try:
        models = {name: model_from_payload(models_payload[name]) for name in TARGETS}
        bundle = ModelBundle(
            model_version=str(payload["model_version"]),
            feature_columns=[str(name) for name in payload["feature_columns"]],
            models=models,
            library_versions=saved_versions,
            trained_at=str(payload["trained_at"]),
            metrics=dict(payload.get("metrics") or {}),
        )
        # Δοκιμαστική πρόβλεψη: ανιχνεύει μοντέλα που δεν διαβάζονται από την τρέχουσα έκδοση.
        probe = np.full((1, len(bundle.feature_columns)), np.nan)
        for target in TARGETS:
            value = bundle.predict_target(target, probe)
            if not np.isfinite(value).all():
                raise ModelLoadError(f"model {target!r} returned a non-finite probe prediction")
    except ModelLoadError:
        raise
    except Exception as error:
        raise ModelLoadError(
            f"model artifact {source} could not be loaded with the installed libraries "
            f"({type(error).__name__}: {error}). "
            "Re-train it with `python -m elfantasy.model.train`."
        ) from error
    return bundle


def load_bundle(path: str | Path) -> ModelBundle:
    """Φορτώνει και ελέγχει το artifact από αρχείο."""
    path = Path(path)
    if not path.is_file():
        raise ModelLoadError(
            f"model artifact not found: {path}. Train it with `python -m elfantasy.model.train`."
        )
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ModelLoadError(f"cannot read model artifact {path}: {error}") from error
    return deserialize_bundle(data, str(path))


def save_bundle(bundle: ModelBundle, path: str | Path) -> int:
    """Αποθηκεύει το artifact ατομικά (προσωρινό αρχείο και `os.replace`). Επιστρέφει το μέγεθος."""
    data = serialize_bundle(bundle)
    atomic_write_many({Path(path): data})
    return len(data)


def atomic_write_many(files: dict[Path, bytes]) -> None:
    """Γράφει όλα τα αρχεία ατομικά ως προς το καθένα.

    Πρώτα γράφονται όλα τα προσωρινά αρχεία (στον ίδιο φάκελο με τον προορισμό τους) και μόνο αν
    επιτύχουν όλα γίνεται `os.replace` στα τελικά ονόματα. Αν κάποια εγγραφή αποτύχει, τα τελικά
    αρχεία δεν αγγίζονται και τα προσωρινά διαγράφονται.
    """
    temporary: dict[Path, Path] = {}
    try:
        for target, data in files.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + ".tmp")
            temp.write_bytes(data)
            temporary[target] = temp
        for target, temp in temporary.items():
            os.replace(temp, target)
    finally:
        for temp in temporary.values():
            if temp.exists():
                temp.unlink()
