"""Εκπαίδευση, backtest και αποθήκευση του μοντέλου πρόβλεψης fantasy score (Φάση 3).

Εκτέλεση (από τη ρίζα του repo)::

    python -m elfantasy.model.train [--db URL] [--threshold X] [--val-season 2024] \\
        [--test-season 2025] [--out-dir models] [--no-tune] [--seed 42] \\
        [--final-refit-through ΣΕΖΟΝ]

Πρωτόκολλο (χρονολογικό, χωρίς ανακάτεμα, ανά σεζόν):

1. Εκπαίδευση: σεζόν < `val_season`. Επικύρωση (validation): `val_season`. Η επιλογή μοντέλου και
   υπερπαραμέτρων (το πολύ ~40 διαμορφώσεις, early stopping στο validation) γίνεται ΜΟΝΟ με το
   MAE του validation.
2. Το επιλεγμένο μοντέλο ξαναεκπαιδεύεται με τις ίδιες υπερπαραμέτρους σε όλες τις σεζόν ≤
   `val_season` (train + validation) και αξιολογείται ΜΙΑ φορά, στο τέλος, στη σεζόν
   `test_season`, που δεν έχει δει ποτέ. Αυτό είναι το «ειλικρινές» (honest) test MAE και
   συγκρίνεται με το threshold.
3. Το `models/model.joblib` και το `models/metrics.json` γράφονται ΜΟΝΟ αν test MAE < threshold.
   Αλλιώς κανένα υπάρχον αρχείο δεν αλλάζει και το πρόγραμμα τερματίζει με κωδικό 3.

Κωδικοί εξόδου: 0 επιτυχία και αποθήκευση, 1 σφάλμα (και threshold που δεν έχει οριστεί),
3 το μοντέλο δεν πέρασε το threshold.

Η εκπαίδευση είναι ντετερμινιστική στο ίδιο περιβάλλον: σταθερός σπόρος και σταθερός αριθμός
threads του XGBoost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elfantasy.config import get_settings
from elfantasy.features.build import FEATURE_COLUMNS, build_features
from elfantasy.model import metrics as M
from elfantasy.model.artifact import (
    ModelBundle,
    RidgeModel,
    XGBoostModel,
    atomic_write_many,
    library_versions,
    serialize_bundle,
)

logger = logging.getLogger("elfantasy.train")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_BELOW_THRESHOLD = 3

TARGET_COLUMN = {"fantasy": "fantasy_score", "pir": "pir"}
#: Στήλες (κυλιόμενος μέσος 5, μέσος σεζόν) που χρησιμοποιούν τα naive baselines ανά στόχο.
BASELINE_COLUMNS = {
    "fantasy": ("fantasy_mean_5", "fantasy_season_mean"),
    "pir": ("pir_mean_5", "pir_season_mean"),
}
BASELINE_NAMES = ("global_mean", "global_median", "naive_rolling5", "naive_season_mean")

NTHREAD = (
    4  # σταθερός αριθμός threads: ίδια αποτελέσματα ανεξάρτητα από τους πυρήνες του μηχανήματος
)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
XGB_OBJECTIVES = {
    "squarederror": "reg:squarederror",
    "absoluteerror": "reg:absoluteerror",
    "pseudohuber": "reg:pseudohubererror",
}
XGB_GRID = {
    "max_depth": [3, 4, 5, 6],
    "eta": [0.03, 0.05, 0.08],
    "min_child_weight": [5, 10, 20, 50],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9],
    "reg_lambda": [1.0, 5.0, 20.0],
}
HUBER_SLOPES = [2.0, 5.0, 10.0]
XGB_CONFIGS_PER_OBJECTIVE = 10
SEARCH = {"max_rounds": 2000, "early_stopping": 50}
NO_TUNE = {"max_rounds": 400, "early_stopping": 30}
DEFAULT_XGB = {
    "max_depth": 4,
    "eta": 0.1,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
}


class TrainingError(RuntimeError):
    """Σφάλμα εκπαίδευσης με μήνυμα προς τον χρήστη (κωδικός εξόδου 1)."""


@dataclass(frozen=True)
class TrainConfig:
    """Ρυθμίσεις μιας εκτέλεσης."""

    val_season: int = 2024
    test_season: int = 2025
    threshold: float | None = None
    out_dir: Path = Path("models")
    tune: bool = True
    seed: int = 42
    final_refit_through: int | None = None
    db_url: str | None = None
    bootstrap_iterations: int = M.BOOTSTRAP_ITERATIONS


@dataclass
class DataInfo:
    """Πληροφορίες για το dataset που χρησιμοποιήθηκε (καταγράφονται στο metrics.json)."""

    last_game_date: str | None = None
    last_played_season: int | None = None
    n_games_played: int = 0
    n_player_rows: int = 0
    n_appearances: int = 0
    fingerprint: str = ""


@dataclass
class ModelSpec:
    """Μία διαμόρφωση μοντέλου: οικογένεια, όνομα και υπερπαράμετροι."""

    name: str
    family: str  # "ridge", "xgb_squarederror", "xgb_absoluteerror", "xgb_pseudohuber"
    params: dict[str, Any]

    @property
    def kind(self) -> str:
        return "ridge" if self.family == "ridge" else "xgboost"


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    final_fit: pd.DataFrame


@dataclass
class TrainOutcome:
    """Αποτέλεσμα εκπαίδευσης: μετρικές, υποψήφιο artifact και απόφαση για το threshold."""

    metrics: dict[str, Any]
    bundle: ModelBundle
    passed: bool
    summary: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Δεδομένα
# --------------------------------------------------------------------------------------


def load_dataset(db_url: str | None = None) -> tuple[pd.DataFrame, DataInfo]:
    """Φορτώνει ιστορικό και αγώνες από τη βάση και υπολογίζει τα features όλων των γραμμών."""
    from elfantasy.db.session import get_engine
    from elfantasy.features.load import load_history, load_played_games

    engine = get_engine(db_url)
    try:
        history = load_history(engine)
        played = load_played_games(engine)
    finally:
        engine.dispose()
    if history.empty:
        raise TrainingError("the database has no player rows: run the ingestion pipeline first")
    frame = build_features(history, games=played)
    return frame, describe_dataset(frame, history, played)


def describe_dataset(frame: pd.DataFrame, history: pd.DataFrame, played: pd.DataFrame) -> DataInfo:
    appearances = frame[frame["is_appearance"]]
    key = appearances[["player_id", "season", "gamecode", "pir", "fantasy_score"]]
    digest = hashlib.sha256(pd.util.hash_pandas_object(key, index=False).to_numpy().tobytes())
    last = history["game_date"].max()
    return DataInfo(
        last_game_date=None if pd.isna(last) else pd.Timestamp(last).date().isoformat(),
        last_played_season=int(history["season"].max()),
        n_games_played=int(len(played)),
        n_player_rows=int(len(history)),
        n_appearances=int(len(appearances)),
        fingerprint=digest.hexdigest(),
    )


def make_splits(frame: pd.DataFrame, config: TrainConfig) -> Splits:
    """Χωρίζει τις συμμετοχές ανά σεζόν. Οι σεζόν > test_season δεν χρησιμοποιούνται ποτέ."""
    if config.test_season <= config.val_season:
        raise TrainingError("--test-season must be later than --val-season")
    rows = frame[frame["is_appearance"] & frame["fantasy_score"].notna() & frame["pir"].notna()]
    train = rows[rows["season"] < config.val_season]
    val = rows[rows["season"] == config.val_season]
    test = rows[rows["season"] == config.test_season]
    final_fit = rows[rows["season"] <= config.val_season]
    for name, part in (("train", train), ("validation", val), ("test", test)):
        if part.empty:
            raise TrainingError(f"the {name} split is empty (check --val-season / --test-season)")
    return Splits(
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
        final_fit.reset_index(drop=True),
    )


def _xy(frame: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    return frame[FEATURE_COLUMNS].to_numpy(dtype=float), frame[TARGET_COLUMN[target]].to_numpy(
        dtype=float
    )


# --------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------


def baseline_predictions(
    frame: pd.DataFrame, target: str, fit_mean: float, fit_median: float
) -> dict[str, np.ndarray]:
    """Προβλέψεις των naive baselines για κάθε γραμμή του `frame`.

    * `global_mean` / `global_median`: ο μέσος όρος / διάμεσος του στόχου στο σύνολο εκπαίδευσης.
    * `naive_rolling5`: μέσος όρος των 5 τελευταίων συμμετοχών, αλλιώς ο μέσος όρος της σεζόν,
      αλλιώς ο καθολικός μέσος όρος.
    * `naive_season_mean`: μέσος όρος της σεζόν μέχρι τώρα, αλλιώς ο μέσος των 5 τελευταίων
      συμμετοχών, αλλιώς ο καθολικός μέσος όρος.
    """
    rolling_column, season_column = BASELINE_COLUMNS[target]
    rolling = frame[rolling_column].to_numpy(dtype=float)
    season = frame[season_column].to_numpy(dtype=float)
    size = len(frame)
    return {
        "global_mean": np.full(size, fit_mean),
        "global_median": np.full(size, fit_median),
        "naive_rolling5": np.where(
            ~np.isnan(rolling), rolling, np.where(~np.isnan(season), season, fit_mean)
        ),
        "naive_season_mean": np.where(
            ~np.isnan(season), season, np.where(~np.isnan(rolling), rolling, fit_mean)
        ),
    }


# --------------------------------------------------------------------------------------
# Μοντέλα
# --------------------------------------------------------------------------------------


def fit_ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> RidgeModel:
    """Ridge με διάμεσο για τα NaN, δείκτες NaN και τυποποίηση. Επιστρέφει παραμέτρους numpy."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    imputed = imputer.fit_transform(features)
    scaler = StandardScaler().fit(imputed)
    ridge = Ridge(alpha=alpha).fit(scaler.transform(imputed), target)
    indicator_columns = (
        np.asarray(imputer.indicator_.features_, dtype=np.int64)
        if imputer.indicator_ is not None
        else np.empty(0, dtype=np.int64)
    )
    return RidgeModel(
        medians=np.asarray(imputer.statistics_, dtype=float),
        indicator_columns=indicator_columns,
        mean=np.asarray(scaler.mean_, dtype=float),
        scale=np.asarray(scaler.scale_, dtype=float),
        coef=np.asarray(ridge.coef_, dtype=float),
        intercept=float(ridge.intercept_),
        alpha=float(alpha),
    )


def _xgb_parameters(spec: ModelSpec, seed: int) -> dict[str, Any]:
    params = {
        "objective": XGB_OBJECTIVES[spec.family.removeprefix("xgb_")],
        "eta": spec.params["eta"],
        "max_depth": spec.params["max_depth"],
        "min_child_weight": spec.params["min_child_weight"],
        "subsample": spec.params["subsample"],
        "colsample_bytree": spec.params["colsample_bytree"],
        "lambda": spec.params["reg_lambda"],
        "eval_metric": "mae",
        "tree_method": "hist",
        "seed": seed,
        "nthread": NTHREAD,
        "verbosity": 0,
    }
    if "huber_slope" in spec.params:
        params["huber_slope"] = spec.params["huber_slope"]
    return params


def _dmatrix(features: np.ndarray, target: np.ndarray | None = None):
    import xgboost as xgb

    return xgb.DMatrix(
        np.asarray(features, dtype=np.float32),
        label=target,
        feature_names=FEATURE_COLUMNS,
        missing=np.nan,
    )


def fit_xgboost_validated(
    spec: ModelSpec,
    train_xy: tuple[np.ndarray, np.ndarray],
    val_xy: tuple[np.ndarray, np.ndarray],
    seed: int,
    limits: dict[str, int],
) -> tuple[np.ndarray, int]:
    """Εκπαιδεύει στο train με early stopping στο MAE του validation.

    Επιστρέφει τις προβλέψεις του validation (με τα δέντρα του καλύτερου γύρου) και τον αριθμό των
    δέντρων του καλύτερου γύρου.
    """
    import xgboost as xgb

    dtrain = _dmatrix(*train_xy)
    dval = _dmatrix(*val_xy)
    booster = xgb.train(
        _xgb_parameters(spec, seed),
        dtrain,
        num_boost_round=limits["max_rounds"],
        evals=[(dval, "val")],
        early_stopping_rounds=limits["early_stopping"],
        verbose_eval=False,
    )
    rounds = int(booster.best_iteration) + 1
    return booster.predict(dval, iteration_range=(0, rounds)), rounds


def fit_xgboost_final(
    spec: ModelSpec, xy: tuple[np.ndarray, np.ndarray], rounds: int, seed: int
) -> XGBoostModel:
    """Εκπαιδεύει με σταθερό αριθμό δέντρων (χωρίς early stopping) και αποθηκεύει σε μορφή UBJ."""
    import xgboost as xgb

    booster = xgb.train(_xgb_parameters(spec, seed), _dmatrix(*xy), num_boost_round=rounds)
    return XGBoostModel(
        raw=bytes(booster.save_raw(raw_format="ubj")),
        feature_columns=list(FEATURE_COLUMNS),
        params={**spec.params, "objective": XGB_OBJECTIVES[spec.family.removeprefix("xgb_")]},
        n_estimators=rounds,
    )


def candidate_specs(config: TrainConfig) -> list[ModelSpec]:
    """Οι διαμορφώσεις προς σύγκριση: με `--no-tune` μία ανά οικογένεια, αλλιώς random search."""
    if not config.tune:
        return [
            ModelSpec("ridge_alpha_10", "ridge", {"alpha": 10.0}),
            ModelSpec("xgb_absoluteerror_default", "xgb_absoluteerror", dict(DEFAULT_XGB)),
        ]
    specs = [
        ModelSpec(f"ridge_alpha_{alpha:g}", "ridge", {"alpha": alpha}) for alpha in RIDGE_ALPHAS
    ]
    rng = np.random.default_rng(config.seed)
    configs: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    while len(configs) < XGB_CONFIGS_PER_OBJECTIVE:
        candidate = {
            name: values[int(rng.integers(len(values)))] for name, values in XGB_GRID.items()
        }
        key = tuple(sorted(candidate.items()))
        if key not in seen:
            seen.add(key)
            configs.append(candidate)
    for short in XGB_OBJECTIVES:
        for index, params in enumerate(configs):
            params = dict(params)
            if short == "pseudohuber":
                params["huber_slope"] = HUBER_SLOPES[int(rng.integers(len(HUBER_SLOPES)))]
            specs.append(ModelSpec(f"xgb_{short}_{index:02d}", f"xgb_{short}", params))
    return specs


# --------------------------------------------------------------------------------------
# Κύρια ροή
# --------------------------------------------------------------------------------------


@dataclass
class _Candidate:
    """Αποτέλεσμα μιας διαμόρφωσης στο validation (εκπαίδευση μόνο στο train)."""

    spec: ModelSpec
    val_mae: float
    rounds: int | None
    val_pred: np.ndarray


@dataclass
class _Trained:
    """Ένα μοντέλο οικογένειας: προβλέψεις validation και τελικά μοντέλα ανά στόχο."""

    spec: ModelSpec
    rounds: dict[str, int | None]
    val_pred: dict[str, np.ndarray]
    final: dict[str, RidgeModel | XGBoostModel] = field(default_factory=dict)
    test_pred: dict[str, np.ndarray] = field(default_factory=dict)


def _validated_fit(
    spec: ModelSpec, target: str, splits: Splits, seed: int, limits: dict[str, int]
) -> tuple[np.ndarray, int | None]:
    """Προβλέψεις validation ενός μοντέλου που εκπαιδεύτηκε στο train (και αριθμός δέντρων)."""
    train_xy = _xy(splits.train, target)
    val_xy = _xy(splits.val, target)
    if spec.kind == "ridge":
        model = fit_ridge(*train_xy, alpha=spec.params["alpha"])
        return model.predict(val_xy[0]), None
    return fit_xgboost_validated(spec, train_xy, val_xy, seed, limits)


def _final_fit(
    spec: ModelSpec, target: str, data: pd.DataFrame, rounds: int | None, seed: int
) -> RidgeModel | XGBoostModel:
    xy = _xy(data, target)
    if spec.kind == "ridge":
        return fit_ridge(*xy, alpha=spec.params["alpha"])
    assert rounds is not None
    return fit_xgboost_final(spec, xy, rounds, seed)


def _round_floats(value: Any, digits: int = 6) -> Any:
    """Αντικαθιστά τα NaN/inf με None και στρογγυλοποιεί τους αριθμούς για το JSON."""
    if isinstance(value, dict):
        return {str(key): _round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(item, digits) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else round(number, digits)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _version_string(info: DataInfo, config: TrainConfig, selected: ModelSpec, now: datetime) -> str:
    """Έκδοση μοντέλου: χρονοσφραγίδα UTC και σύντομο hash δεδομένων και παραμέτρων (χωρίς git)."""
    payload = json.dumps(
        {
            "data": info.fingerprint,
            "features": FEATURE_COLUMNS,
            "val": config.val_season,
            "test": config.test_season,
            "seed": config.seed,
            "tune": config.tune,
            "refit_through": config.final_refit_through,
            "selected": [selected.name, selected.params],
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return f"{now:%Y%m%dT%H%M%SZ}-{digest}"


def _split_description(frame: pd.DataFrame) -> dict[str, Any]:
    seasons = sorted({int(season) for season in frame["season"].unique()})
    return {"seasons": seasons, "rows": int(len(frame))}


def run_training(
    frame: pd.DataFrame,
    info: DataInfo,
    config: TrainConfig,
    threshold: float,
    progress: Callable[[str], None] = logger.info,
) -> TrainOutcome:
    """Εκτελεί ολόκληρο το backtest και επιστρέφει μετρικές και υποψήφιο artifact (χωρίς εγγραφή).

    Το test αγγίζεται μόνο αφού έχει κλειδώσει η επιλογή μοντέλου από το validation.
    """
    started = time.time()
    splits = make_splits(frame, config)
    limits = SEARCH if config.tune else NO_TUNE
    specs = candidate_specs(config)
    progress(
        f"rows: train={len(splits.train)} validation={len(splits.val)} test={len(splits.test)} "
        f"final_fit={len(splits.final_fit)}; candidates={len(specs)}"
    )

    # --- Φάση Α: επιλογή στο validation (το test δεν χρησιμοποιείται καθόλου) ---
    results: list[_Candidate] = []
    for position, spec in enumerate(specs, start=1):
        val_pred, rounds = _validated_fit(spec, "fantasy", splits, config.seed, limits)
        val_mae = M.mae(splits.val["fantasy_score"].to_numpy(), val_pred)
        results.append(_Candidate(spec, val_mae, rounds, val_pred))
        progress(f"  [{position}/{len(specs)}] {spec.name}: validation MAE {val_mae:.4f}")
    selected_spec = min(results, key=lambda item: item.val_mae).spec

    family_best: dict[str, _Candidate] = {}
    for candidate in results:
        best = family_best.get(candidate.spec.family)
        if best is None or candidate.val_mae < best.val_mae:
            family_best[candidate.spec.family] = candidate
    selected_val_mae = family_best[selected_spec.family].val_mae
    progress(f"selected: {selected_spec.name} (validation MAE {selected_val_mae:.4f})")

    # --- Φάση Β: το PIR με τις ίδιες υπερπαραμέτρους και τελικά μοντέλα (train + validation) ---
    trained: dict[str, _Trained] = {}
    for family, candidate in family_best.items():
        pir_val, pir_rounds = _validated_fit(candidate.spec, "pir", splits, config.seed, limits)
        trained[family] = _Trained(
            spec=candidate.spec,
            rounds={"fantasy": candidate.rounds, "pir": pir_rounds},
            val_pred={"fantasy": candidate.val_pred, "pir": pir_val},
        )
    # Το test αγγίζεται εδώ και μόνο εδώ, αφού έχει κλειδώσει η επιλογή.
    for family, item in trained.items():
        for target in ("fantasy", "pir"):
            model = _final_fit(
                item.spec, target, splits.final_fit, item.rounds[target], config.seed
            )
            item.final[target] = model
            item.test_pred[target] = model.predict(_xy(splits.test, target)[0])
        progress(f"  refit through season {config.val_season}: {family}")
    selected_family = selected_spec.family
    selected = trained[selected_family]

    # --- Μετρικές ---
    now = datetime.now(UTC)
    metrics = _assemble_metrics(
        splits, config, info, threshold, results, trained, selected_family, now, started, progress
    )

    bundle = ModelBundle(
        model_version=metrics["model_version"],
        feature_columns=list(FEATURE_COLUMNS),
        models={"fantasy": selected.final["fantasy"], "pir": selected.final["pir"]},
        library_versions=library_versions(),
        trained_at=metrics["created_at_utc"],
        metrics={},
    )
    check = metrics["threshold"]
    passed = bool(check["passed"])
    outcome = TrainOutcome(metrics=metrics, bundle=bundle, passed=passed)
    outcome.extra = {"splits": splits, "trained": trained, "selected_family": selected_family}
    return outcome


def _model_metrics(splits: Splits, trained: _Trained, target: str) -> dict[str, dict[str, float]]:
    column = TARGET_COLUMN[target]
    return {
        "validation": M.regression_metrics(splits.val[column].to_numpy(), trained.val_pred[target]),
        "test": M.regression_metrics(splits.test[column].to_numpy(), trained.test_pred[target]),
    }


def _assemble_metrics(
    splits: Splits,
    config: TrainConfig,
    info: DataInfo,
    threshold: float,
    results: list[_Candidate],
    trained: dict[str, _Trained],
    selected_family: str,
    now: datetime,
    started: float,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    selected = trained[selected_family]
    spec = selected.spec

    # Baselines: στο validation με στατιστικά του train, στο test με στατιστικά του final fit.
    baselines: dict[str, dict[str, dict[str, np.ndarray]]] = {"fantasy": {}, "pir": {}}
    for target in ("fantasy", "pir"):
        column = TARGET_COLUMN[target]
        baselines[target]["validation"] = baseline_predictions(
            splits.val,
            target,
            float(splits.train[column].mean()),
            float(splits.train[column].median()),
        )
        baselines[target]["test"] = baseline_predictions(
            splits.test,
            target,
            float(splits.final_fit[column].mean()),
            float(splits.final_fit[column].median()),
        )

    comparison: dict[str, dict[str, dict[str, float]]] = {"fantasy": {}, "pir": {}}
    for target in ("fantasy", "pir"):
        column = TARGET_COLUMN[target]
        for name in BASELINE_NAMES:
            comparison[target][name] = {
                "validation": M.mae(
                    splits.val[column].to_numpy(), baselines[target]["validation"][name]
                ),
                "test": M.mae(splits.test[column].to_numpy(), baselines[target]["test"][name]),
            }
        for family, item in trained.items():
            comparison[target][family] = {
                "validation": M.mae(splits.val[column].to_numpy(), item.val_pred[target]),
                "test": M.mae(splits.test[column].to_numpy(), item.test_pred[target]),
            }

    def best_baseline(target: str, split: str) -> str:
        return min(BASELINE_NAMES, key=lambda name: comparison[target][name][split])

    fantasy_metrics = _model_metrics(splits, selected, "fantasy")
    pir_metrics = _model_metrics(splits, selected, "pir")
    test_mae = fantasy_metrics["test"]["mae"]
    baseline_test = best_baseline("fantasy", "test")
    baseline_val = best_baseline("fantasy", "validation")

    # Τμήματα και bootstrap για το επιλεγμένο μοντέλο (fantasy), έναντι του καλύτερου baseline.
    segments: dict[str, Any] = {}
    bootstrap: dict[str, Any] = {}
    for split_name, part, model_pred, base_name in (
        ("validation", splits.val, selected.val_pred["fantasy"], baseline_val),
        ("test", splits.test, selected.test_pred["fantasy"], baseline_test),
    ):
        y = part["fantasy_score"].to_numpy()
        base_pred = baselines["fantasy"][split_name][base_name]
        labels = {
            "minutes_mean5": M.minutes_bucket(part["min_mean_5"]),
            "prior_appearances": M.history_bucket(part["games_played_total"]),
            "phase": M.phase_bucket(part["phase"]),
        }
        segments[split_name] = M.segment_report(
            labels, y, {"model": model_pred, "best_baseline": base_pred}
        )
        games = part["season"].astype(str) + "/" + part["gamecode"].astype(str)
        bootstrap[split_name] = {
            "baseline": base_name,
            "model_minus_baseline": M.paired_bootstrap_mae_difference(
                y,
                model_pred,
                base_pred,
                games.to_numpy(),
                iterations=config.bootstrap_iterations,
                seed=config.seed,
            ),
            "model_mae_interval": M.bootstrap_mae_interval(
                y,
                model_pred,
                games.to_numpy(),
                iterations=config.bootstrap_iterations,
                seed=config.seed,
            ),
        }

    importance = selected.final["fantasy"].importance(list(FEATURE_COLUMNS))
    top_features = sorted(importance.items(), key=lambda item: -item[1])[:15]

    best_baseline_mae = comparison["fantasy"][baseline_test]["test"]
    threshold_check = {
        "value": threshold,
        "test_mae": test_mae,
        "best_naive_baseline": baseline_test,
        "best_naive_baseline_test_mae": best_baseline_mae,
        "passed": bool(test_mae < threshold),
        "beats_best_naive_baseline": bool(test_mae < best_baseline_mae),
        "rule": "the artifact is written only if the honest test MAE (fantasy) < threshold",
    }
    trained_seasons = sorted({int(s) for s in splits.final_fit["season"].unique()})
    refit = config.final_refit_through
    metrics: dict[str, Any] = {
        "model_version": _version_string(info, config, spec, now),
        "created_at_utc": now.isoformat(timespec="seconds"),
        "seed": config.seed,
        "tuned": config.tune,
        "protocol": {
            "description": (
                "chronological backtest by season: model selection on validation (trained on "
                "earlier seasons), then refit on train+validation and one evaluation on the "
                "untouched test season"
            ),
            "train": _split_description(splits.train),
            "validation": {"season": config.val_season, "rows": int(len(splits.val))},
            "test": {"season": config.test_season, "rows": int(len(splits.test))},
            "final_fit": _split_description(splits.final_fit),
            "final_fit_through_season": max(trained_seasons),
            "final_refit_through": refit,
            "test_mae_is_out_of_sample": refit is None,
        },
        "data": {
            "last_game_date": info.last_game_date,
            "last_played_season": info.last_played_season,
            "n_games_played": info.n_games_played,
            "n_player_rows": info.n_player_rows,
            "n_appearances": info.n_appearances,
            "fingerprint": info.fingerprint,
        },
        "selected_model": {
            "name": spec.name,
            "family": spec.family,
            "params": spec.params,
            "n_estimators": selected.rounds["fantasy"],
            "n_estimators_pir": selected.rounds["pir"],
            "validation_mae": fantasy_metrics["validation"]["mae"],
        },
        "fantasy": fantasy_metrics,
        "pir": pir_metrics,
        "comparison_mae": comparison,
        "family_best_params": {
            family: {"name": item.spec.name, "params": item.spec.params, "rounds": item.rounds}
            for family, item in trained.items()
        },
        "segments": segments,
        "bootstrap": bootstrap,
        "feature_importance_top15": [
            {"feature": name, "importance": value} for name, value in top_features
        ],
        "feature_columns": list(FEATURE_COLUMNS),
        "threshold": threshold_check,
        "search": {
            "n_candidates": len(results),
            "candidates": [
                {
                    "name": c.spec.name,
                    "family": c.spec.family,
                    "params": c.spec.params,
                    "rounds": c.rounds,
                    "validation_mae": c.val_mae,
                }
                for c in results
            ],
        },
        "library_versions": library_versions(),
        "training_seconds": round(time.time() - started, 1),
    }
    return _round_floats(metrics)


def apply_final_refit(outcome: TrainOutcome, frame: pd.DataFrame, config: TrainConfig) -> None:
    """Προαιρετικό refit του επιλεγμένου μοντέλου σε νεότερα δεδομένα (`--final-refit-through`).

    Εκτελείται ΜΟΝΟ αφού έχει περάσει ο έλεγχος του threshold. Το test MAE του metrics.json
    παραμένει αυτό του ειλικρινούς μοντέλου, αλλά το artifact έχει πια δει τη σεζόν test, άρα η
    τιμή δεν ισχύει out-of-sample για αυτό (το `protocol.test_mae_is_out_of_sample` είναι false).
    """
    through = config.final_refit_through
    assert through is not None
    selected = outcome.extra["trained"][outcome.extra["selected_family"]]
    rows = frame[
        frame["is_appearance"]
        & frame["fantasy_score"].notna()
        & frame["pir"].notna()
        & (frame["season"] <= through)
    ].reset_index(drop=True)
    if rows.empty:
        raise TrainingError(f"no rows for --final-refit-through {through}")
    models = {
        target: _final_fit(selected.spec, target, rows, selected.rounds[target], config.seed)
        for target in ("fantasy", "pir")
    }
    outcome.bundle.models = models
    protocol = outcome.metrics["protocol"]
    protocol["final_refit_through"] = through
    protocol["final_fit_through_season"] = int(rows["season"].max())
    protocol["final_refit_rows"] = int(len(rows))
    protocol["test_mae_is_out_of_sample"] = False
    protocol["note"] = (
        "the saved artifact was refit on seasons up to the final_refit_through season, so the test "
        "metrics (computed with the model fit through the validation season) no longer describe "
        "the "
        "saved artifact out-of-sample"
    )


def write_artifacts(outcome: TrainOutcome, out_dir: Path) -> dict[str, Any]:
    """Γράφει ατομικά το `model.joblib` και το `metrics.json` στον φάκελο `out_dir`."""
    outcome.bundle.metrics = outcome.metrics
    model_bytes = serialize_bundle(outcome.bundle)
    metrics_bytes = (json.dumps(outcome.metrics, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    model_path = out_dir / "model.joblib"
    metrics_path = out_dir / "metrics.json"
    atomic_write_many({model_path: model_bytes, metrics_path: metrics_bytes})
    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "model_bytes": len(model_bytes),
    }


# --------------------------------------------------------------------------------------
# Εμφάνιση αποτελεσμάτων και CLI
# --------------------------------------------------------------------------------------


def format_summary(metrics: dict[str, Any]) -> str:
    """Συνοπτικός πίνακας MAE για το log (fantasy και PIR, validation και test)."""
    lines = []
    for target in ("fantasy", "pir"):
        lines.append(f"MAE {target}:")
        lines.append(f"  {'model':<26}{'validation':>12}{'test':>12}")
        for name, values in metrics["comparison_mae"][target].items():
            lines.append(f"  {name:<26}{values['validation']:>12.4f}{values['test']:>12.4f}")
    selected = metrics["selected_model"]
    fantasy = metrics["fantasy"]["test"]
    check = metrics["threshold"]
    lines.append(
        f"selected {selected['name']}: test MAE {fantasy['mae']:.4f} RMSE {fantasy['rmse']:.4f} "
        f"R2 {fantasy['r2']:.4f} bias {fantasy['bias']:.4f}; "
        f"PIR test MAE {metrics['pir']['test']['mae']:.4f}"
    )
    lines.append(
        f"threshold {check['value']}: test MAE {check['test_mae']:.4f}, best naive baseline "
        f"{check['best_naive_baseline']} {check['best_naive_baseline_test_mae']:.4f} -> "
        f"{'PASSED' if check['passed'] else 'FAILED'}"
    )
    return "\n".join(lines)


def resolve_threshold(cli_value: float | None) -> float:
    """Το threshold από το `--threshold` ή τη ρύθμιση MAE_THRESHOLD. Τιμή ≤ 0 = δεν έχει οριστεί."""
    value = cli_value if cli_value is not None else get_settings().mae_threshold
    if value is None or not math.isfinite(value) or value <= 0:
        raise TrainingError(
            "no MAE threshold is set: pass --threshold X or set MAE_THRESHOLD > 0 "
            "(a value <= 0 means 'not set')"
        )
    return float(value)


def train_and_save(
    config: TrainConfig,
    frame: pd.DataFrame,
    info: DataInfo,
    progress: Callable[[str], None] = logger.info,
) -> int:
    """Ολόκληρη η ροή πάνω σε έτοιμο frame features. Επιστρέφει τον κωδικό εξόδου."""
    threshold = resolve_threshold(config.threshold)
    if config.final_refit_through is not None and config.final_refit_through < config.test_season:
        raise TrainingError("--final-refit-through must be >= --test-season")
    outcome = run_training(frame, info, config, threshold, progress)
    progress(format_summary(outcome.metrics))
    if not outcome.metrics["threshold"]["beats_best_naive_baseline"]:
        progress("WARNING: the model does not beat the best naive baseline on the test season")
    if not outcome.passed:
        progress(
            f"threshold NOT met: test MAE {outcome.metrics['threshold']['test_mae']:.4f} "
            f">= {threshold}; no artifact was written, existing files are untouched"
        )
        return EXIT_BELOW_THRESHOLD
    if config.final_refit_through is not None:
        apply_final_refit(outcome, frame, config)
        progress(
            f"refit on seasons <= {config.final_refit_through} (test MAE no longer out-of-sample)"
        )
    written = write_artifacts(outcome, config.out_dir)
    progress(
        f"wrote {written['model_path']} ({written['model_bytes'] / 1024:.0f} KiB) and "
        f"{written['metrics_path']}; "
        f"model_version {outcome.metrics['model_version']}"
    )
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m elfantasy.model.train",
        description="Train the Euroleague fantasy score model (backtest and quality gate).",
    )
    parser.add_argument("--db", default=None, help="database URL (default: DATABASE_URL setting)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="maximum honest test MAE (fantasy) for the artifact to be written "
        "(default: MAE_THRESHOLD setting; a value <= 0 means 'not set')",
    )
    parser.add_argument(
        "--val-season", type=int, default=2024, help="validation season (default 2024)"
    )
    parser.add_argument("--test-season", type=int, default=2025, help="test season (default 2025)")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("models"), help="output folder (default models)"
    )
    parser.add_argument(
        "--no-tune", action="store_true", help="skip the hyperparameter search (fast, for tests)"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument(
        "--final-refit-through",
        type=int,
        default=None,
        metavar="SEASON",
        help="after the threshold check, also refit on seasons up to SEASON (>= test season); "
        "the test MAE then no longer holds out-of-sample for the saved artifact",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=M.BOOTSTRAP_ITERATIONS, help="bootstrap iterations"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
    config = TrainConfig(
        val_season=args.val_season,
        test_season=args.test_season,
        threshold=args.threshold,
        out_dir=args.out_dir,
        tune=not args.no_tune,
        seed=args.seed,
        final_refit_through=args.final_refit_through,
        db_url=args.db,
        bootstrap_iterations=args.bootstrap,
    )
    try:
        resolve_threshold(config.threshold)  # γρήγορος έλεγχος πριν από την ακριβή εκπαίδευση
        frame, info = load_dataset(config.db_url)
        return train_and_save(config, frame, info)
    except TrainingError as error:
        logger.error("ERROR: %s", error)
        return EXIT_ERROR
    except Exception:
        logger.exception("unexpected error during training")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
