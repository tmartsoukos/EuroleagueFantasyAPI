"""Μετρικές αξιολόγησης: MAE, RMSE, R², bias, MAE ανά τμήμα και paired bootstrap ανά αγώνα.

Όλες οι συναρτήσεις δουλεύουν σε numpy/pandas και δεν εξαρτώνται από το μοντέλο, ώστε να
χρησιμοποιούνται το ίδιο για baselines, Ridge και XGBoost.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE = 0.95


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, R² και bias (μέσο σφάλμα πρόβλεψης − πραγματικό) σε μονάδες του στόχου.

    Το R² υπολογίζεται ως προς τον μέσο όρο του ίδιου συνόλου αξιολόγησης και μπορεί να είναι
    αρνητικό για κακό μοντέλο. Αν το σύνολο έχει μηδενική διακύμανση, το R² είναι NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    total = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "n": int(len(y_true)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(math.sqrt((error**2).mean())),
        "r2": float(1.0 - (error**2).sum() / total) if total > 0 else float("nan"),
        "bias": float(error.mean()),
    }


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Μέσο απόλυτο σφάλμα."""
    return float(np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)).mean())


def minutes_bucket(min_mean_5: pd.Series) -> pd.Series:
    """Τμήμα ανά μέσο όρο λεπτών των τελευταίων 5 συμμετοχών: <10, 10-20, >20, χωρίς ιστορικό."""
    labels = pd.Series("no_history", index=min_mean_5.index, dtype=object)
    labels[min_mean_5 < 10] = "<10 min"
    labels[(min_mean_5 >= 10) & (min_mean_5 <= 20)] = "10-20 min"
    labels[min_mean_5 > 20] = ">20 min"
    return labels


def history_bucket(games_played_total: pd.Series) -> pd.Series:
    """Τμήμα ανά πλήθος προηγούμενων συμμετοχών: <5 ή >=5."""
    return pd.Series(
        np.where(games_played_total < 5, "<5 prior", ">=5 prior"), index=games_played_total.index
    )


def phase_bucket(phase: pd.Series) -> pd.Series:
    """Κανονική περίοδος (RS) έναντι υπόλοιπων φάσεων (PO, PI, FF). Το άγνωστο μένει «unknown»."""
    values = phase.astype(object)
    return pd.Series(
        np.where(values.isna(), "unknown", np.where(values == "RS", "RS", "playoffs/other")),
        index=phase.index,
    )


def segment_report(
    segments: Mapping[str, pd.Series],
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> dict[str, dict[str, dict[str, float]]]:
    """MAE ανά τμήμα και ανά πρόβλεψη.

    `segments`: όνομα διαχωρισμού → Series ετικετών τμήματος. `predictions`: όνομα μοντέλου →
    προβλέψεις. Επιστρέφει `{διαχωρισμός: {τμήμα: {"n": ..., <μοντέλο>: mae, ...}}}`.
    """
    y_true = np.asarray(y_true, dtype=float)
    report: dict[str, dict[str, dict[str, float]]] = {}
    for split_name, labels in segments.items():
        labels = np.asarray(labels, dtype=object)
        parts: dict[str, dict[str, float]] = {}
        for label in sorted({str(value) for value in labels}):
            mask = labels == label
            entry: dict[str, float] = {"n": int(mask.sum())}
            for model_name, predicted in predictions.items():
                entry[model_name] = mae(y_true[mask], np.asarray(predicted)[mask])
            parts[label] = entry
        report[split_name] = parts
    return report


def paired_bootstrap_mae_difference(
    y_true: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap του `MAE(prediction) − MAE(reference)` με επαναδειγματοληψία ανά αγώνα.

    Ο αγώνας (`groups`) είναι η μονάδα επαναδειγματοληψίας, γιατί οι παίκτες του ίδιου αγώνα δεν
    είναι ανεξάρτητοι (ίδιο αποτέλεσμα, ίδιο τέμπο). Η διαφορά είναι αρνητική όταν η
    `prediction` έχει μικρότερο MAE. Επιστρέφει το σημειακό εκτιμητή, το διάστημα εμπιστοσύνης
    95% (εκατοστημόρια 2,5 και 97,5) και το ποσοστό των επαναδειγματοληψιών όπου η
    `prediction` είναι καλύτερη.
    """
    y_true = np.asarray(y_true, dtype=float)
    diff = np.abs(np.asarray(prediction, dtype=float) - y_true) - np.abs(
        np.asarray(reference, dtype=float) - y_true
    )
    codes, _ = pd.factorize(pd.Series(np.asarray(groups)))
    n_groups = int(codes.max()) + 1
    diff_sum = np.bincount(codes, weights=diff, minlength=n_groups)
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    rng = np.random.default_rng(seed)
    low, high = (1 - CONFIDENCE) / 2 * 100, (1 + CONFIDENCE) / 2 * 100
    draws = np.empty(iterations)
    # Σ ανά αγώνα επί πλήθος επαναλήψεων: ένας πίνακας δεικτών ανά δέσμη, για περιορισμένη μνήμη.
    batch = 200
    done = 0
    while done < iterations:
        size = min(batch, iterations - done)
        picks = rng.integers(0, n_groups, size=(size, n_groups))
        draws[done : done + size] = diff_sum[picks].sum(axis=1) / counts[picks].sum(axis=1)
        done += size
    return {
        "difference": float(diff.mean()),
        "ci_low": float(np.percentile(draws, low)),
        "ci_high": float(np.percentile(draws, high)),
        "prob_better": float((draws < 0).mean()),
        "n_games": n_groups,
        "iterations": int(iterations),
    }


def bootstrap_mae_interval(
    y_true: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = 42,
) -> dict[str, float]:
    """Διάστημα εμπιστοσύνης 95% για το MAE μιας πρόβλεψης (επαναδειγματοληψία ανά αγώνα)."""
    error = np.abs(np.asarray(prediction, dtype=float) - np.asarray(y_true, dtype=float))
    codes, _ = pd.factorize(pd.Series(np.asarray(groups)))
    n_groups = int(codes.max()) + 1
    error_sum = np.bincount(codes, weights=error, minlength=n_groups)
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_groups, size=(iterations, n_groups))
    draws = error_sum[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = (1 - CONFIDENCE) / 2 * 100, (1 + CONFIDENCE) / 2 * 100
    return {
        "mae": float(error.mean()),
        "ci_low": float(np.percentile(draws, low)),
        "ci_high": float(np.percentile(draws, high)),
    }
