"""Έλεγχος υγείας της υπηρεσίας (`GET /health`).

Ο έλεγχος είναι ελαφρύς (το Render θα τον καλεί συχνά): τρία μικρά queries στη βάση και ανάγνωση
των ήδη φορτωμένων μετρικών του μοντέλου. ΔΕΝ υπολογίζει προβλέψεις. Η υπηρεσία είναι `ok` μόνο
αν η βάση φτάνει και περιέχει παίκτες ΚΑΙ το μοντέλο φορτώθηκε και υπολόγισε τις πρώτες
προβλέψεις· αλλιώς είναι `degraded` και τα προβλήματα αναφέρονται με σταθερά, σύντομα μηνύματα
(χωρίς διαδρομές αρχείων, URL βάσης ή stack traces: αυτά γράφονται μόνο στο log του server).
"""

from __future__ import annotations

import logging

from sqlalchemy.exc import SQLAlchemyError

from elfantasy.api.schemas import DatabaseInfo, HealthResponse, ModelInfo
from elfantasy.api.services import clock_today, database_summary
from elfantasy.api.state import PROBLEM_DATABASE, AppState
from elfantasy.model.predict import Predictor

logger = logging.getLogger(__name__)

PROBLEM_NOT_STARTED = "application startup has not completed"
PROBLEM_NO_PLAYERS = "database contains no players (run the ingestion pipeline)"
PROBLEM_NO_MODEL = "model is not loaded"


def _number(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _section(source, key: str) -> dict:
    """Το υποσύνολο `key` ενός λεξικού μετρικών, ή κενό λεξικό αν λείπει ή δεν είναι λεξικό."""
    value = source.get(key) if isinstance(source, dict) else None
    return value if isinstance(value, dict) else {}


def model_info(predictor: Predictor) -> ModelInfo:
    """Στοιχεία του μοντέλου από τις μετρικές εκπαίδευσης (`metrics.json`). Όσα λείπουν από τις
    μετρικές (ή έχουν απροσδόκητη μορφή) γίνονται `null`: το /health δεν αποτυγχάνει ποτέ λόγω
    τους."""
    metrics = predictor.metrics
    threshold = _section(metrics, "threshold")
    protocol = _section(metrics, "protocol")
    test = _section(_section(metrics, "fantasy"), "test")
    trained_through = protocol.get("final_refit_through") or protocol.get(
        "final_fit_through_season"
    )
    selected = _section(metrics, "selected_model").get("name")
    return ModelInfo(
        version=predictor.model_version,
        selected_model=selected if isinstance(selected, str) else None,
        test_mae=_number(threshold.get("test_mae", test.get("mae"))),
        threshold=_number(threshold.get("value")),
        trained_through_season=trained_through if isinstance(trained_through, int) else None,
    )


def build_health(state: AppState) -> HealthResponse:
    """Συνθέτει την απάντηση του /health από την κατάσταση της εφαρμογής και ένα ελαφρύ query."""
    problems = dict(state.problems)
    if not state.started:
        problems["application"] = PROBLEM_NOT_STARTED

    model = None
    if state.predictor is not None:
        model = model_info(state.predictor)
    elif "model" not in problems and "database" not in problems:
        problems["model"] = PROBLEM_NO_MODEL

    database = DatabaseInfo(
        ok=False, players=None, latest_played_game_date=None, next_scheduled_game_date=None
    )
    data_age_days = None
    if state.engine is not None:
        today = clock_today(state.clock)
        try:
            summary = database_summary(state.engine, today)
        except SQLAlchemyError:
            logger.warning("health check: the database query failed", exc_info=True)
            problems["database"] = PROBLEM_DATABASE
        else:
            database = DatabaseInfo(
                ok="database" not in problems and summary.players > 0,
                players=summary.players,
                latest_played_game_date=summary.latest_played_game_date,
                next_scheduled_game_date=summary.next_scheduled_game_date,
            )
            if summary.players == 0 and "database" not in problems:
                problems["database"] = PROBLEM_NO_PLAYERS
            if summary.latest_played_game_date is not None:
                data_age_days = (today - summary.latest_played_game_date).days
    else:
        problems.setdefault("database", PROBLEM_DATABASE)

    return HealthResponse(
        status="degraded" if problems else "ok",
        model=model,
        database=database,
        data_age_days=data_age_days,
        data_loaded_through=None if state.service is None else state.service.loaded_through,
        problems=[f"{name}: {message}" for name, message in sorted(problems.items())],
    )
