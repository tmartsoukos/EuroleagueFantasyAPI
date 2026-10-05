"""Η εφαρμογή FastAPI: `create_app()` και το module-level `app` για το `uvicorn`.

Εκκίνηση::

    uvicorn elfantasy.api.main:app --host 0.0.0.0 --port $PORT

Το import αυτού του module ΔΕΝ διαβάζει βάση ή μοντέλο και δεν αποτυγχάνει αν λείπουν: η φόρτωση
γίνεται στο lifespan (`elfantasy.api.state.start`) και κάθε αποτυχία γίνεται «degraded» κατάσταση
(το `/health` απαντά 503, τα `/predict` και `/rankings` απαντούν 503 με καθαρό μήνυμα).
"""

import logging
import mimetypes
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from starlette.responses import Response

from elfantasy.api import state as app_state
from elfantasy.api.limits import MAX_REQUEST_BODY_BYTES, BodySizeLimitMiddleware
from elfantasy.api.routers import admin, health, players, predict
from elfantasy.api.routers import availability as availability_router
from elfantasy.api.services import NotFoundError, utc_now
from elfantasy.api.state import AppState
from elfantasy.config import Settings
from elfantasy.model.predict import Predictor

logger = logging.getLogger(__name__)

try:
    API_VERSION = metadata.version("elfantasy")
except metadata.PackageNotFoundError:  # π.χ. τρέξιμο από τον πηγαίο κώδικα χωρίς εγκατάσταση
    API_VERSION = "0.1.0"

API_TITLE = "Euroleague Fantasy Points Predictor API"

# Η web εφαρμογή (στατικά αρχεία χωρίς build) ανήκει στο πακέτο και σερβίρεται από το ίδιο origin.
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
WEB_MOUNT_PATH = "/app"

# Ο τύπος αρχείου προέρχεται από το σύστημα (στα Windows από το μητρώο, που μπορεί να δίνει στο
# .js τον τύπο text/plain, τον οποίο ο browser απορρίπτει για ES modules). Ορίζονται ρητά.
for _suffix, _media_type in {
    ".js": "text/javascript",
    ".css": "text/css",
    ".svg": "image/svg+xml",
    ".webmanifest": "application/manifest+json",
}.items():
    mimetypes.add_type(_media_type, _suffix)


class WebFiles(StaticFiles):
    """Στατικά αρχεία της εφαρμογής με `Cache-Control: no-cache`: ο browser ρωτά (ETag) σε κάθε
    φόρτωση, ώστε μετά από deploy να μη μένουν παλιά modules από την ευρετική cache."""

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


API_DESCRIPTION = """\
Υπηρεσία που προβλέπει το **fantasy score** ενός παίκτη Euroleague για τον **επόμενο αγώνα** της
ομάδας του, με μοντέλο μηχανικής μάθησης (XGBoost) πάνω σε ιστορικά boxscores των σεζόν
2016 έως 2026.

### Πώς να διαβάζεις τις προβλέψεις

* Το fantasy score είναι `PIR + 0,1 × |PIR|` σε νίκη της ομάδας (×1,1 για θετικό PIR, ×0,9 για
  αρνητικό), αλλιώς `PIR` (βλ. `FANTASY_RULES.md`).
* Το `predicted_fantasy` είναι μια **«τυπική» τιμή** (προσεγγίζει τη διάμεσο, όχι τον μέσο όρο): το
  μοντέλο έχει αρνητικό bias περίπου −0,8 πόντους. Ισχύει **υπό την προϋπόθεση ότι ο παίκτης
  αγωνίζεται**.
* **Δεν** περιλαμβάνει captain ×2 ή πάγκο ×0,5: αυτά είναι επιλογές ρόστερ και εφαρμόζονται εκ των
  υστέρων.
* Το μοντέλο **δεν προβλέπει τραυματισμούς** ούτε αλλαγές ρόστερ. Η διαθεσιμότητα (`out`,
  `doubtful`) είναι χειροκίνητη πληροφορία διαχειριστή που εφαρμόζεται μετά την πρόβλεψη και δεν
  περνά ως feature στο μοντέλο.
* Το `predicted_pir` προέρχεται από ανεξάρτητο μοντέλο και δεν υπολογίζεται από το fantasy score.
* Η πρόβλεψη ενός μεμονωμένου αγώνα έχει μεγάλο εγγενή θόρυβο (honest MAE περίπου 5,9 πόντοι
  fantasy): η αξία της είναι κυρίως στη σύγκριση παικτών.

### Ασφάλεια

Τα endpoints ανάγνωσης είναι δημόσια. Τα endpoints διαχείρισης (`POST /availability`,
`DELETE /availability/{player_id}`, `POST /admin/refresh`) απαιτούν το header `X-API-Key` (κουμπί
**Authorize**) και είναι κλειστά (503) όταν δεν έχει οριστεί `ADMIN_API_KEY`.

### Δεδομένα

Η υπηρεσία δεν βλέπει νέα δεδομένα μετά από νέο ingestion μέχρι να κληθεί το `POST /admin/refresh`
(ή να γίνει επανεκκίνηση). Το `/health` δείχνει το data cutoff.
"""

API_TAGS = [
    {"name": "health", "description": "Κατάσταση της υπηρεσίας (βάση, μοντέλο)."},
    {
        "name": "predictions",
        "description": "Προβλέψεις fantasy score: ανά παίκτη και rankings.",
    },
    {"name": "players", "description": "Αναζήτηση παικτών και εύρεση του `player_id`."},
    {
        "name": "availability",
        "description": "Διαθεσιμότητα παικτών (τραυματισμοί, απουσίες): χειροκίνητο override.",
    },
    {"name": "admin", "description": "Διαχείριση (απαιτεί `X-API-Key`)."},
]


def create_app(
    settings: Settings | None = None,
    engine: Engine | None = None,
    predictor: Predictor | None = None,
    clock: Callable[[], datetime] | None = None,
) -> FastAPI:
    """Δημιουργεί την εφαρμογή.

    * `settings`: ρυθμίσεις (προεπιλογή: από το περιβάλλον και το `.env`, στο startup).
    * `engine`: κοινή engine της βάσης (προεπιλογή: από το `DATABASE_URL`). Αν δοθεί απ' έξω, δεν
      κλείνει στο shutdown.
    * `predictor`: έτοιμος `Predictor` (προεπιλογή: `Predictor.load` με τη μία κοινή engine).
    * `clock`: συνάρτηση που επιστρέφει την τρέχουσα ώρα (datetime UTC). Από αυτήν προκύπτει το
      «σήμερα» του `as_of` και το `updated_at` της διαθεσιμότητας· τα tests την αντικαθιστούν.

    Η φόρτωση βάσης και μοντέλου γίνεται στο lifespan, όχι εδώ.
    """
    state = AppState(
        clock=clock if clock is not None else utc_now,
        settings=settings,
        engine=engine,
        predictor=predictor,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await run_in_threadpool(app_state.start, state)
        try:
            yield
        finally:
            await run_in_threadpool(app_state.stop, state)

    app = FastAPI(
        title=API_TITLE,
        summary="Πρόβλεψη fantasy score παικτών Euroleague",
        description=API_DESCRIPTION,
        version=API_VERSION,
        openapi_tags=API_TAGS,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
        redoc_url="/redoc",
    )
    app.state.api = state

    # Όριο μεγέθους σώματος ΠΡΙΝ από οποιοδήποτε routing ή έλεγχο κλειδιού (api/limits.py).
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_REQUEST_BODY_BYTES)

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(players.router)
    app.include_router(availability_router.router)
    app.include_router(admin.router)
    _register_handlers(app)
    # Τελευταίο, ώστε να μη σκιάζει κανένα endpoint του API.
    app.mount(WEB_MOUNT_PATH, WebFiles(directory=WEB_DIR, html=True), name="web")
    return app


def _printable(value: Any) -> Any:
    """Αντικαθιστά τα μεμονωμένα surrogates κάθε κειμένου με την κατάληξη `\\udXXX` (αναδρομικά),
    ώστε το αποτέλεσμα να κωδικοποιείται πάντα σε UTF-8."""
    if isinstance(value, str):
        return value.encode("utf-8", "backslashreplace").decode("utf-8")
    if isinstance(value, list):
        return [_printable(item) for item in value]
    if isinstance(value, dict):
        return {key: _printable(item) for key, item in value.items()}
    return value


def _register_handlers(app: FastAPI) -> None:
    """Καθαρά σφάλματα JSON (`{"detail": ...}`) χωρίς εσωτερικές λεπτομέρειες προς τον client."""

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url=f"{WEB_MOUNT_PATH}/")

    @app.exception_handler(NotFoundError)
    async def not_found_handler(_request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": exc.message})

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Όπως ο προεπιλεγμένος χειριστής του FastAPI (`{"detail": [...]}`, κωδικός 422), αλλά το
        # `input` κάθε σφάλματος καθαρίζεται από μεμονωμένα surrogates (π.χ. "\ud800" σε JSON),
        # που δεν κωδικοποιούνται σε UTF-8 και θα έριχναν την ίδια την απάντηση σφάλματος σε 500.
        return JSONResponse(
            status_code=422, content={"detail": _printable(jsonable_encoder(exc.errors()))}
        )

    @app.exception_handler(SQLAlchemyError)
    async def database_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.error(
            "database error while handling %s %s", request.method, request.url.path, exc_info=exc
        )
        return JSONResponse(status_code=503, content={"detail": "database unavailable"})

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "unhandled error while handling %s %s", request.method, request.url.path, exc_info=exc
        )
        return JSONResponse(status_code=500, content={"detail": "internal server error"})


# Για το `uvicorn elfantasy.api.main:app`. Δεν αγγίζει βάση, μοντέλο ή περιβάλλον στο import.
app = create_app()
