"""Σύγκριση των καταγεγραμμένων προβλέψεων με τα πραγματικά αποτελέσματα (Φάση 5).

Χρήση (από τη ρίζα του repo, με το `DATABASE_URL` στο περιβάλλον ή με `--db`)::

    python -m elfantasy.model.evaluate_recorded
    python -m elfantasy.model.evaluate_recorded --model-version MODEL_VERSION --since 2026-10-01

Για κάθε γραμμή του πίνακα `predictions` (που γράφει η εντολή
`python -m elfantasy.model.record_predictions`) ψάχνει το αποτέλεσμα του ίδιου παίκτη στον ίδιο
αγώνα (`player_games`) και υπολογίζει το σφάλμα `predicted_fantasy − fantasy_score`. Κάθε
πρόβλεψη ανήκει σε ΜΙΑ από τις κατηγορίες:

* `pending`: ο αγώνας δεν έχει παιχτεί ακόμη (δεν μετρά στις μετρικές).
* `appeared`: ο παίκτης αγωνίστηκε (γραμμή boxscore με `dnp = false` και λεπτά > 0). ΜΟΝΟ αυτές
  μετρούν στο `mae` και στο `bias`: το μοντέλο προβλέπει υπό την προϋπόθεση ότι ο παίκτης
  αγωνίζεται (docs/MODEL.md), άρα ένας παίκτης που δεν έπαιξε δεν «αστόχησε».
* `dnp`: ο αγώνας παίχτηκε και ο παίκτης έχει γραμμή, αλλά δεν αγωνίστηκε (πραγματική τιμή 0).
* `not_in_boxscore`: ο αγώνας παίχτηκε αλλά ο παίκτης δεν υπάρχει στο boxscore (π.χ. έφυγε από την
  ομάδα, ή ο αγώνας δεν έχει boxscore στην πηγή).

Οι κατηγορίες `dnp` και `not_in_boxscore` αναφέρονται πάντα ρητά, ώστε να φαίνεται πόσο συχνά
η υπόθεση «αγωνίζεται» δεν ισχύει. Το `mae_incl_dnp` είναι ενημερωτική μετρική: το MAE αν οι
παίκτες με γραμμή DNP μετρούσαν και αυτοί με πραγματική τιμή 0. Το `bias` είναι ο μέσος όρος του
`predicted − actual` (θετικό = το μοντέλο υπερεκτιμά). Τα αποτελέσματα ομαδοποιούνται κατά έκδοση
μοντέλου (δεν αναμειγνύονται μοντέλα) και ανά ημέρα του αγώνα.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import Engine, and_, inspect, select

from elfantasy.config import get_settings
from elfantasy.db import models
from elfantasy.db.cli import console_logging, report_failure, use_utf8_output
from elfantasy.db.session import get_engine
from elfantasy.db.urls import DatabaseUrlError, safe_url

# Ρητό όνομα: με `python -m` το `__name__` είναι `__main__` και τα μηνύματα δεν θα έφταναν
# στον handler του πακέτου `elfantasy` (db/cli.py).
logger = logging.getLogger("elfantasy.model.evaluate_recorded")

PENDING = "pending"
APPEARED = "appeared"
DNP = "dnp"
NOT_IN_BOXSCORE = "not_in_boxscore"

SUMMARY_COLUMNS = [
    "predictions",
    "appeared",
    "dnp",
    "not_in_boxscore",
    "mae",
    "bias",
    "mae_incl_dnp",
]

_FRAME_COLUMNS = [
    "player_id",
    "season",
    "gamecode",
    "model_version",
    "as_of",
    "predicted_fantasy",
    "game_date",
    "played",
    "actual_fantasy",
    "dnp",
    "minutes",
]


class EvaluationError(Exception):
    """Ο πίνακας των προβλέψεων λείπει ή δεν μπορεί να διαβαστεί."""


@dataclass(frozen=True)
class Evaluation:
    """Το αποτέλεσμα της αξιολόγησης.

    `daily` έχει δείκτη (έκδοση μοντέλου, ημέρα αγώνα) και `overall` δείκτη την έκδοση μοντέλου·
    και τα δύο έχουν τις στήλες του `SUMMARY_COLUMNS`. Το `recorded` είναι το πλήθος των
    καταγεγραμμένων προβλέψεων που εξετάστηκαν και το `pending` όσες περιμένουν ακόμη αποτέλεσμα.
    """

    daily: pd.DataFrame
    overall: pd.DataFrame
    recorded: int
    pending: int


def load_recorded(
    engine: Engine,
    *,
    model_version: str | None = None,
    since: date | None = None,
    until: date | None = None,
) -> pd.DataFrame:
    """Οι καταγεγραμμένες προβλέψεις μαζί με τον αγώνα και (αν υπάρχει) την πραγματική γραμμή του
    παίκτη. Φίλτρα: έκδοση μοντέλου και ημερομηνίες του αγώνα (`since` ≤ ημέρα ≤ `until`).

    Σηκώνει `EvaluationError` αν ο πίνακας `predictions` δεν υπάρχει. Ένας κενός πίνακας παλιάς
    διάταξης (χωρίς `as_of`) δεν έχει καταγραφές και δίνει κενό αποτέλεσμα.
    """
    inspector = inspect(engine)
    if not inspector.has_table(models.predictions.name):
        raise EvaluationError(
            "the predictions table does not exist: create the schema (python -m "
            "elfantasy.db.migrate for PostgreSQL) and record predictions first"
        )
    columns = {column["name"] for column in inspector.get_columns(models.predictions.name)}
    if "as_of" not in columns:
        return pd.DataFrame(columns=_FRAME_COLUMNS)
    p, g, pg = models.predictions, models.games, models.player_games
    statement = (
        select(
            p.c.player_id,
            p.c.season,
            p.c.gamecode,
            p.c.model_version,
            p.c.as_of,
            p.c.predicted_fantasy,
            g.c.game_date,
            g.c.played,
            pg.c.fantasy_score.label("actual_fantasy"),
            pg.c.dnp,
            pg.c.minutes,
        )
        .select_from(
            p.join(g, and_(p.c.season == g.c.season, p.c.gamecode == g.c.gamecode)).outerjoin(
                pg,
                and_(
                    p.c.season == pg.c.season,
                    p.c.gamecode == pg.c.gamecode,
                    p.c.player_id == pg.c.player_id,
                ),
            )
        )
        .order_by(g.c.game_date, p.c.player_id)
    )
    if model_version is not None:
        statement = statement.where(p.c.model_version == model_version)
    if since is not None:
        statement = statement.where(g.c.game_date >= since)
    if until is not None:
        statement = statement.where(g.c.game_date <= until)
    with engine.connect() as connection:
        return pd.read_sql(statement, connection)


def classify(frame: pd.DataFrame) -> pd.Series:
    """Η κατηγορία κάθε πρόβλεψης: `pending`, `appeared`, `dnp` ή `not_in_boxscore`."""
    played = frame["played"].astype(bool)
    has_row = played & frame["actual_fantasy"].notna()
    dnp = frame["dnp"].astype("boolean").fillna(True).astype(bool)
    minutes = frame["minutes"].astype(float).fillna(0.0)
    appeared = has_row & ~dnp & (minutes > 0)
    return pd.Series(
        np.select(
            [~played, appeared, has_row],
            [PENDING, APPEARED, DNP],
            default=NOT_IN_BOXSCORE,
        ),
        index=frame.index,
        dtype=object,
    )


def _summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Πλήθη ανά κατηγορία και μετρικές σφάλματος, ομαδοποιημένα κατά `keys`."""
    status = frame["status"]
    error = frame["predicted_fantasy"] - frame["actual_fantasy"]
    working = frame.assign(
        is_appeared=status == APPEARED,
        is_dnp=status == DNP,
        is_missing=status == NOT_IN_BOXSCORE,
        appeared_abs=error.abs().where(status == APPEARED),
        appeared_error=error.where(status == APPEARED),
        including_dnp_abs=error.abs().where(status.isin([APPEARED, DNP])),
    )
    grouped = working.groupby(keys, sort=True)
    return pd.DataFrame(
        {
            "predictions": grouped.size(),
            "appeared": grouped["is_appeared"].sum(),
            "dnp": grouped["is_dnp"].sum(),
            "not_in_boxscore": grouped["is_missing"].sum(),
            "mae": grouped["appeared_abs"].mean(),
            "bias": grouped["appeared_error"].mean(),
            "mae_incl_dnp": grouped["including_dnp_abs"].mean(),
        }
    )[SUMMARY_COLUMNS]


def evaluate(frame: pd.DataFrame) -> Evaluation:
    """Υπολογίζει τα συγκεντρωτικά ανά ημέρα και συνολικά από τις προβλέψεις του `load_recorded`."""
    if frame.empty:
        empty = pd.DataFrame(columns=SUMMARY_COLUMNS)
        return Evaluation(daily=empty, overall=empty.copy(), recorded=0, pending=0)
    status = classify(frame)
    labelled = frame.assign(status=status)
    finished = labelled[labelled["status"] != PENDING]
    if finished.empty:
        empty = pd.DataFrame(columns=SUMMARY_COLUMNS)
        pending = len(labelled)
        return Evaluation(daily=empty, overall=empty.copy(), recorded=len(frame), pending=pending)
    return Evaluation(
        daily=_summary(finished, ["model_version", "game_date"]),
        overall=_summary(finished, ["model_version"]),
        recorded=len(frame),
        pending=int((labelled["status"] == PENDING).sum()),
    )


def evaluate_recorded(
    engine: Engine,
    *,
    model_version: str | None = None,
    since: date | None = None,
    until: date | None = None,
) -> Evaluation:
    """Φορτώνει τις καταγεγραμμένες προβλέψεις και τις αξιολογεί."""
    return evaluate(load_recorded(engine, model_version=model_version, since=since, until=until))


def format_evaluation(evaluation: Evaluation) -> str:
    """Η αναφορά για την κονσόλα."""
    lines = [
        f"Recorded predictions: {evaluation.recorded} "
        f"({evaluation.recorded - evaluation.pending} with a played game, "
        f"{evaluation.pending} waiting for the game to be played)"
    ]
    if evaluation.overall.empty:
        lines.append("Nothing to evaluate yet: none of the recorded games has been played.")
        return "\n".join(lines)
    lines += [
        "",
        "Overall (MAE and bias only over players who appeared; dnp and not_in_boxscore are "
        "listed separately):",
        evaluation.overall.to_string(float_format=lambda value: f"{value:.3f}"),
        "",
        "Per game day:",
        evaluation.daily.to_string(float_format=lambda value: f"{value:.3f}"),
    ]
    return "\n".join(lines)


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
        prog="python -m elfantasy.model.evaluate_recorded",
        description="Compare the recorded predictions with the actual results of the games that "
        "have been played: MAE per game day and overall.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="database URL (default: the DATABASE_URL setting). Prefer the environment variable: "
        "a URL on the command line ends up in the shell history",
    )
    parser.add_argument("--model-version", default=None, help="only this model version")
    parser.add_argument(
        "--since", type=_iso_date, default=None, metavar="YYYY-MM-DD", help="first game day"
    )
    parser.add_argument(
        "--until", type=_iso_date, default=None, metavar="YYYY-MM-DD", help="last game day"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Κωδικοί εξόδου: 0 επιτυχία (και όταν δεν υπάρχει ακόμη τίποτα για αξιολόγηση), 1 αποτυχία
    (π.χ. ο πίνακας λείπει), 2 άκυρο URL."""
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
        try:
            evaluation = evaluate_recorded(
                engine, model_version=args.model_version, since=args.since, until=args.until
            )
        except EvaluationError as exc:
            logger.error("%s", exc)
            return 1
        except Exception as exc:
            report_failure("the evaluation failed", exc, url)
            return 1
        finally:
            engine.dispose()
    print(format_evaluation(evaluation))
    return 0


if __name__ == "__main__":
    sys.exit(main())
