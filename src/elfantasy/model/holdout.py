"""Εξαγωγή του holdout fixture για το quality gate (Φάση 6).

Εκτέλεση (από τη ρίζα του repo)::

    python -m elfantasy.model.holdout --out tests/fixtures/holdout_2025.parquet

Το fixture περιέχει ΟΛΕΣ τις συμμετοχές της σεζόν test (προεπιλογή 2025) με:

* τα features του μοντέλου (float32, ακριβώς όπως τα βλέπει το XGBoost),
* τους στόχους `fantasy_score` και `pir`,
* τις προβλέψεις των naive baselines (στήλες `baseline_*`), υπολογισμένες με τα στατιστικά του
  συνόλου εκπαίδευσης του τελικού μοντέλου (σεζόν ≤ `val_season`).

Έτσι το test `tests/quality/test_model_quality.py` ελέγχει το committed μοντέλο χωρίς βάση και
χωρίς κώδικα feature engineering: το fixture είναι «παγωμένο». Το committed μοντέλο έχει
εκπαιδευτεί σε σεζόν ≤ `val_season`, άρα δεν έχει δει ποτέ τη σεζόν του fixture (εκτός αν
εκπαιδεύτηκε με `--final-refit-through`).

Η εξαγωγή είναι ντετερμινιστική (ταξινόμηση κατά σεζόν, αγώνα, παίκτη· σταθερή συμπίεση).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from elfantasy.features.build import FEATURE_COLUMNS, build_features
from elfantasy.model.train import BASELINE_NAMES, TARGET_COLUMN, baseline_predictions

SCHEMA_KEY = b"elfantasy"
ID_COLUMNS = ["player_id", "season", "gamecode"]
TARGETS = ["fantasy_score", "pir"]


def build_holdout(
    frame: pd.DataFrame, val_season: int = 2024, test_season: int = 2025
) -> tuple[pd.DataFrame, dict]:
    """Επιστρέφει το fixture (DataFrame) και τα μεταδεδομένα του από ένα frame features."""
    rows = frame[frame["is_appearance"] & frame["fantasy_score"].notna() & frame["pir"].notna()]
    fit = rows[rows["season"] <= val_season]
    test = rows[rows["season"] == test_season]
    if fit.empty or test.empty:
        raise ValueError("the fit or test season has no rows (check --val-season / --test-season)")
    holdout = test[[*ID_COLUMNS, *TARGETS]].copy()
    holdout[FEATURE_COLUMNS] = test[FEATURE_COLUMNS].astype("float32")
    stats = {}
    for target in ("fantasy", "pir"):
        column = TARGET_COLUMN[target]
        stats[target] = {"mean": float(fit[column].mean()), "median": float(fit[column].median())}
    predictions = baseline_predictions(
        test, "fantasy", stats["fantasy"]["mean"], stats["fantasy"]["median"]
    )
    for name in BASELINE_NAMES:
        holdout[f"baseline_{name}"] = predictions[name]
    holdout = holdout.sort_values(["season", "gamecode", "player_id"], kind="stable")
    holdout = holdout.reset_index(drop=True)
    meta = {
        "description": "held-out test season for the model quality gate (all appearances)",
        "test_season": int(test_season),
        "val_season": int(val_season),
        "fit_seasons_through": int(val_season),
        "rows": int(len(holdout)),
        "n_games": int(holdout[["season", "gamecode"]].drop_duplicates().shape[0]),
        "feature_columns": list(FEATURE_COLUMNS),
        "baseline_fit_stats": stats,
    }
    return holdout, meta


def write_holdout(holdout: pd.DataFrame, meta: dict, path: Path) -> int:
    """Γράφει το parquet (zstd) με τα μεταδεδομένα στο schema. Επιστρέφει το μέγεθος σε bytes."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(holdout, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[SCHEMA_KEY] = json.dumps(meta, sort_keys=True).encode("utf-8")
    table = table.replace_schema_metadata(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", compression_level=19)
    return path.stat().st_size


def read_holdout(path: Path) -> tuple[pd.DataFrame, dict]:
    """Διαβάζει το fixture και τα μεταδεδομένα του."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    raw = (table.schema.metadata or {}).get(SCHEMA_KEY)
    meta = json.loads(raw.decode("utf-8")) if raw else {}
    return table.to_pandas(), meta


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m elfantasy.model.holdout",
        description="Export the held-out test season (features, targets, baselines) to parquet.",
    )
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures/holdout_2025.parquet"))
    parser.add_argument("--db", default=None, help="database URL (default: DATABASE_URL setting)")
    parser.add_argument("--val-season", type=int, default=2024)
    parser.add_argument("--test-season", type=int, default=2025)
    args = parser.parse_args(argv)

    from elfantasy.db.session import get_engine
    from elfantasy.features.load import load_history, load_played_games

    engine = get_engine(args.db)
    try:
        history = load_history(engine)
        played = load_played_games(engine)
    finally:
        engine.dispose()
    if history.empty:
        print("ERROR: the database has no player rows: run the ingestion pipeline first")
        return 1
    frame = build_features(history, games=played)
    try:
        holdout, meta = build_holdout(frame, args.val_season, args.test_season)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1
    size = write_holdout(holdout, meta, args.out)
    print(
        f"wrote {args.out} ({size / 1024:.0f} KiB): {meta['rows']} rows, {meta['n_games']} games, "
        f"{len(FEATURE_COLUMNS)} features, test season {meta['test_season']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
