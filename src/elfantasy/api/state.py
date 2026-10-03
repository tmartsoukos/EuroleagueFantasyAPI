"""Κατάσταση της εφαρμογής και φόρτωση (startup/shutdown) της βάσης, του μοντέλου και της υπηρεσίας.

Η φόρτωση ΔΕΝ γίνεται στο import του module της εφαρμογής: γίνεται στο lifespan (`start`). Κάθε
αποτυχία (λείπει ή είναι ασύμβατο το μοντέλο, δεν φτάνει η βάση, αποτυγχάνει ο πρώτος
υπολογισμός) καταγράφεται στο log του server και μεταφράζεται σε «degraded» κατάσταση: το
`/health` απαντά 503, τα `/predict` και `/rankings` απαντούν 503 με καθαρό μήνυμα και η
εφαρμογή συνεχίζει να τρέχει. Τα μηνύματα προς τους clients είναι σταθερά και σύντομα: δεν
περιέχουν διαδρομές αρχείων, URL βάσης, credentials ή stack traces.

Μία κοινή engine (από το `DATABASE_URL`) χρησιμοποιείται από την υπηρεσία, τη διαθεσιμότητα και
τον `Predictor`. Engine ή `Predictor` που δίνονται απ' έξω (tests) δεν κλείνουν από την εφαρμογή.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from elfantasy.api.availability import ensure_table
from elfantasy.api.services import Clock, PredictionService, utc_now
from elfantasy.config import Settings, get_settings
from elfantasy.db.session import get_engine
from elfantasy.db.urls import sqlite_file_is_missing
from elfantasy.model.artifact import ModelLoadError
from elfantasy.model.predict import Predictor

logger = logging.getLogger(__name__)

# Σταθερά μηνύματα για τους clients (αγγλικά, χωρίς εσωτερικές λεπτομέρειες).
PROBLEM_DATABASE = "database is not available or not initialised"
PROBLEM_MODEL_MISSING = "model artifact is missing or incompatible"
PROBLEM_MODEL_FAILED = "predictions could not be computed"


@dataclass
class AppState:
    """Ό,τι μοιράζονται τα αιτήματα: ρυθμίσεις, ρολόι, engine, μοντέλο και υπηρεσία.

    `problems` αντιστοιχίζει το όνομα ενός ελέγχου (`database`, `model`) σε ένα σταθερό μήνυμα
    όταν ο έλεγχος απέτυχε στο startup. Τα `owns_*` δείχνουν τι δημιούργησε η ίδια η εφαρμογή (και
    άρα τι πρέπει να κλείσει στο shutdown).
    """

    clock: Clock = utc_now
    settings: Settings | None = None
    engine: Engine | None = None
    predictor: Predictor | None = None
    service: PredictionService | None = None
    problems: dict[str, str] = field(default_factory=dict)
    started: bool = False
    owns_engine: bool = False
    owns_predictor: bool = False


def start(state: AppState) -> None:
    """Φορτώνει βάση, μοντέλο και υπηρεσία. Δεν σηκώνει εξαιρέσεις: τα προβλήματα καταγράφονται
    στο `state.problems` (και στο log) και η εφαρμογή ξεκινά σε degraded κατάσταση."""
    settings = state.settings if state.settings is not None else get_settings()
    state.settings = settings
    state.problems.clear()
    _start_database(state, settings)
    if state.engine is not None and "database" not in state.problems:
        _start_service(state, settings)
    state.started = True


def _start_database(state: AppState, settings: Settings) -> None:
    if state.engine is None:
        url = settings.database_url
        if sqlite_file_is_missing(url):
            logger.error("the SQLite database file does not exist")
            state.problems["database"] = PROBLEM_DATABASE
            return
        try:
            state.engine = get_engine(url)
        except Exception:
            # Άκυρο URL ή λείπει ο driver: το μήνυμα της εξαίρεσης μπορεί να περιέχει στοιχεία
            # σύνδεσης και γράφεται μόνο στο log.
            logger.exception("could not create the database engine")
            state.problems["database"] = PROBLEM_DATABASE
            return
        state.owns_engine = True
    try:
        ensure_table(state.engine)
    except SQLAlchemyError:
        logger.exception("could not reach the database or find the player_availability table")
        state.problems["database"] = PROBLEM_DATABASE


def _start_service(state: AppState, settings: Settings) -> None:
    engine = state.engine
    try:
        if state.predictor is None:
            state.predictor = Predictor.load(settings.model_path, engine=engine)
            state.owns_predictor = True
        service = PredictionService(state.predictor, engine, state.clock)
        service.warm_up()
        state.service = service
    except ModelLoadError:
        logger.exception("the model could not be loaded")
        state.problems["model"] = PROBLEM_MODEL_MISSING
    except SQLAlchemyError:
        logger.exception("the database could not be read while starting the predictor")
        state.problems["database"] = PROBLEM_DATABASE
    except Exception:
        logger.exception("the first predictions could not be computed")
        state.problems["model"] = PROBLEM_MODEL_FAILED


def stop(state: AppState) -> None:
    """Κλείνει ό,τι δημιούργησε η εφαρμογή (όχι ό,τι δόθηκε απ' έξω) και μηδενίζει την κατάσταση,
    ώστε ένα νέο `start` να ξεκινά από την αρχή."""
    if state.owns_predictor and state.predictor is not None:
        state.predictor.close()
        state.predictor = None
    if state.owns_engine and state.engine is not None:
        state.engine.dispose()
        state.engine = None
    state.owns_predictor = False
    state.owns_engine = False
    state.service = None
    state.started = False
