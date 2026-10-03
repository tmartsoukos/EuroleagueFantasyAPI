"""Καταγραφή των προβλέψεων του μοντέλου στον πίνακα `predictions` (Φάση 5).

Χρήση (από τη ρίζα του repo, με το `DATABASE_URL` στο περιβάλλον ή με `--db`)::

    # ενεργοί παίκτες, σημερινή ημέρα (UTC)
    python -m elfantasy.model.record_predictions
    # όλοι οι γνωστοί παίκτες
    python -m elfantasy.model.record_predictions --all
    python -m elfantasy.model.record_predictions --as-of 2026-10-06

Για κάθε παίκτη με προγραμματισμένο επόμενο αγώνα γράφεται μία γραμμή με το `predicted_fantasy` και
το `predicted_pir` του μοντέλου, ώστε αργότερα (`python -m elfantasy.model.evaluate_recorded`) να
συγκριθούν με τα πραγματικά αποτελέσματα. Το κλειδί της γραμμής είναι (παίκτης, σεζόν, αγώνας,
έκδοση μοντέλου, `as_of`): η εγγραφή είναι upsert, άρα το ξανατρέξιμο της ίδιας ημέρας ΔΕΝ
διπλασιάζει γραμμές (ενημερώνει τις τιμές και το `created_at`). Γραμμές δεν διαγράφονται ποτέ.

Παίκτες χωρίς επόμενο αγώνα στο πρόγραμμα (offseason, ή η ομάδα δεν έχει άλλους αγώνες) ΔΕΝ
καταγράφονται: δεν υπάρχει αγώνας με τον οποίο να συγκριθεί η πρόβλεψη.

Το `as_of` είναι η ημερομηνία από την οποία ο `Predictor` ψάχνει τον επόμενο αγώνα κάθε ομάδας
(προεπιλογή: σήμερα, UTC). Δεν είναι «ταξίδι στο παρελθόν»: το ιστορικό που χρησιμοποιεί το μοντέλο
είναι πάντα ό,τι υπάρχει τώρα στη βάση. Το API ΔΕΝ γράφει ποτέ στον πίνακα (τα GET αιτήματα δεν
έχουν παρενέργειες): η καταγραφή γίνεται μόνο από αυτή την εντολή.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import Engine

from elfantasy.config import get_settings
from elfantasy.db import models
from elfantasy.db.cli import console_logging, report_failure, use_utf8_output
from elfantasy.db.session import ensure_schema, get_engine, upsert
from elfantasy.db.urls import DatabaseUrlError, safe_url
from elfantasy.model.artifact import ModelLoadError
from elfantasy.model.predict import PlayerPrediction, Predictor, utc_today

# Ρητό όνομα: με `python -m` το `__name__` είναι `__main__` και τα μηνύματα δεν θα έφταναν
# στον handler του πακέτου `elfantasy` (db/cli.py).
logger = logging.getLogger("elfantasy.model.record_predictions")

# Το φυσικό κλειδί μιας καταγεγραμμένης πρόβλεψης (ίδιο με το unique constraint του πίνακα).
KEY_COLUMNS = ["player_id", "season", "gamecode", "model_version", "as_of"]


@dataclass(frozen=True)
class RecordResult:
    """Η περίληψη μιας καταγραφής."""

    as_of: date
    model_version: str
    predicted: int  # προβλέψεις που επέστρεψε ο Predictor
    recorded: int  # γραμμές που στάλθηκαν στη βάση (νέες ή ενημερωμένες)
    skipped_without_game: int  # παίκτες χωρίς επόμενο αγώνα στο πρόγραμμα
    skipped_invalid: int  # προβλέψεις με μη πεπερασμένη τιμή (NaN, άπειρο)


def prediction_rows(
    predictions: Sequence[PlayerPrediction], as_of: date, created_at: datetime
) -> tuple[list[dict[str, Any]], int, int]:
    """Μετατρέπει τις προβλέψεις σε γραμμές του πίνακα. Επιστρέφει (γραμμές, παίκτες χωρίς επόμενο
    αγώνα, προβλέψεις με μη πεπερασμένη τιμή): οι δύο τελευταίες κατηγορίες δεν καταγράφονται."""
    rows: list[dict[str, Any]] = []
    without_game = invalid = 0
    for prediction in predictions:
        game = prediction.next_game
        if game is None:
            without_game += 1
            continue
        if not math.isfinite(prediction.predicted_fantasy) or not math.isfinite(
            prediction.predicted_pir
        ):
            invalid += 1
            continue
        rows.append(
            {
                "player_id": prediction.player_id,
                "season": game.season,
                "gamecode": game.gamecode,
                "predicted_fantasy": float(prediction.predicted_fantasy),
                "predicted_pir": float(prediction.predicted_pir),
                "model_version": prediction.model_version,
                "as_of": as_of,
                "created_at": created_at,
            }
        )
    return rows, without_game, invalid


def record_predictions(
    engine: Engine,
    predictor: Predictor,
    *,
    as_of: date | None = None,
    active_only: bool = True,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> RecordResult:
    """Γράφει τις προβλέψεις του `predictor` στον πίνακα `predictions` (idempotent upsert).

    Ο πίνακας πρέπει να υπάρχει (στο Postgres τον δημιουργούν τα migrations· στην SQLite
    δημιουργείται αν λείπει). Όλες οι γραμμές γράφονται σε μία συναλλαγή.
    """
    day = as_of if as_of is not None else utc_today()
    ensure_schema(engine, [models.predictions])  # πριν από τον υπολογισμό: αποτυχία νωρίς
    predictions = predictor.predict_all(as_of=day, active_only=active_only)
    rows, without_game, invalid = prediction_rows(predictions, day, now())
    if rows:
        with engine.begin() as connection:
            upsert(
                connection,
                models.predictions,
                rows,
                KEY_COLUMNS,
                set_=lambda excluded: {
                    "predicted_fantasy": excluded.predicted_fantasy,
                    "predicted_pir": excluded.predicted_pir,
                    "created_at": excluded.created_at,
                },
            )
    return RecordResult(
        as_of=day,
        model_version=predictor.model_version,
        predicted=len(predictions),
        recorded=len(rows),
        skipped_without_game=without_game,
        skipped_invalid=invalid,
    )


# ----------------------------------------------------------------------------------------------
# Γραμμή εντολών
# ----------------------------------------------------------------------------------------------


def _iso_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid date {text!r}: expected YYYY-MM-DD") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m elfantasy.model.record_predictions",
        description="Record the model's predictions for the next game of every player in the "
        "predictions table (idempotent: re-running on the same day does not duplicate rows).",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="database URL (default: the DATABASE_URL setting). Prefer the environment variable: "
        "a URL on the command line ends up in the shell history",
    )
    parser.add_argument(
        "--as-of",
        type=_iso_date,
        default=None,
        metavar="YYYY-MM-DD",
        help="the day from which the next game of each team is looked up (default: today, UTC)",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--active-only",
        dest="active_only",
        action="store_true",
        default=True,
        help="record only the active players (default)",
    )
    scope.add_argument(
        "--all", dest="active_only", action="store_false", help="record every known player"
    )
    parser.add_argument(
        "--model", default=None, help="model artifact (default: the MODEL_PATH setting)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Κωδικοί εξόδου: 0 επιτυχία, 1 αποτυχία (βάση, μοντέλο), 2 άκυρο URL."""
    args = build_parser().parse_args(argv)
    use_utf8_output()
    url = args.db or get_settings().database_url
    with console_logging():
        try:
            engine = get_engine(url)
        except DatabaseUrlError as exc:
            logger.error("%s", exc)
            return 2
        logger.info("Database: %s", safe_url(url))
        predictor = None
        try:
            predictor = Predictor.load(args.model, engine=engine)
            result = record_predictions(
                engine, predictor, as_of=args.as_of, active_only=args.active_only
            )
        except ModelLoadError as exc:
            report_failure("the model could not be loaded", exc, url)
            return 1
        except Exception as exc:
            report_failure("recording the predictions failed", exc, url)
            return 1
        finally:
            if predictor is not None:
                predictor.close()
            engine.dispose()
    print(
        f"Recorded {result.recorded} predictions for as_of={result.as_of.isoformat()} "
        f"(model {result.model_version}); {result.skipped_without_game} players without an "
        f"upcoming game and {result.skipped_invalid} invalid predictions were skipped."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
