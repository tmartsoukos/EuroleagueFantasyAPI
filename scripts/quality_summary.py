#!/usr/bin/env python3
"""Τυπώνει σε Markdown τα νούμερα του quality gate του μοντέλου (για το `$GITHUB_STEP_SUMMARY`).

Το job `quality-gate` του CI τρέχει πρώτα το `pytest tests/quality` (που αποφασίζει αν το build
περνά) και μετά αυτό το script, ώστε το MAE, το threshold και τα baselines να φαίνονται στη σελίδα
του run χωρίς να ψάχνει κανείς στα logs:

    python scripts/quality_summary.py >> "$GITHUB_STEP_SUMMARY"

Διαβάζει τα ίδια πράγματα με το test: το committed μοντέλο (`MODEL_PATH`, προεπιλογή
`models/model.joblib`), το held-out fixture `tests/fixtures/holdout_2025.parquet` και το
`MAE_THRESHOLD` (προεπιλογή 6,00). Τρέχει από τη ρίζα του repo και δεν χρειάζεται `.env`, βάση ή
δίκτυο. Τελειώνει πάντα με κωδικό 0 αν διαβαστούν τα αρχεία (το ΑΠΟΤΕΛΕΣΜΑ του gate το δίνει το
pytest)· σφάλμα ανάγνωσης δίνει κωδικό 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "holdout_2025.parquet"
BASELINES = {
    "baseline_naive_season_mean": "naive season mean",
    "baseline_naive_rolling5": "naive rolling 5",
    "baseline_global_median": "global median",
    "baseline_global_mean": "global mean",
}


def build_summary() -> str:
    """Το Markdown της περίληψης (το ίδιο MAE με το tests/quality, στο ίδιο fixture)."""
    from elfantasy.config import get_settings
    from elfantasy.model.artifact import load_bundle
    from elfantasy.model.holdout import read_holdout
    from elfantasy.model.metrics import mae

    settings = get_settings()
    model_path = Path(settings.model_path)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    threshold = settings.mae_threshold
    bundle = load_bundle(model_path)
    frame, meta = read_holdout(FIXTURE)
    predictions = bundle.predict(frame)
    target = frame["fantasy_score"]
    model_mae = mae(target, predictions["fantasy"])
    baselines = {
        label: mae(target, frame[column])
        for column, label in BASELINES.items()
        if column in frame.columns
    }
    best_label, best_mae = min(baselines.items(), key=lambda item: item[1])
    recorded = bundle.metrics.get("fantasy", {}).get("test", {}).get("mae")
    selected = bundle.metrics.get("selected_model", {}).get("name", "?")
    passed = threshold > 0 and model_mae < threshold
    versions = bundle.library_versions

    lines = [
        "### Model quality gate",
        "",
        f"**{'PASS' if passed else 'FAIL'}**: fantasy MAE {model_mae:.4f} on the held-out "
        f"season {meta.get('test_season')} ({len(frame)} appearances), threshold "
        f"{threshold:.2f}" + (f" (margin {threshold - model_mae:.4f})." if passed else "."),
        "",
        "| | fantasy MAE |",
        "|---|---:|",
        f"| model `{bundle.model_version}` ({selected}) | **{model_mae:.4f}** |",
        f"| threshold (`MAE_THRESHOLD`) | {threshold:.2f} |",
    ]
    lines += [f"| baseline: {label} | {value:.4f} |" for label, value in baselines.items()]
    lines += [
        "",
        f"Improvement over the best baseline ({best_label}): {best_mae - model_mae:.4f} "
        f"({(best_mae - model_mae) / best_mae:.1%}).",
    ]
    if isinstance(recorded, (int, float)):
        lines.append(f"MAE recorded at training time (metrics.json): {recorded:.4f}.")
    lines.append(
        "Libraries used to train the model: "
        + ", ".join(f"{name} {versions[name]}" for name in sorted(versions) if name != "python")
        + "."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    try:
        sys.stdout.write(build_summary())
    except Exception as error:  # ένα αρχείο που λείπει ή είναι κατεστραμμένο: καθαρό μήνυμα
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
