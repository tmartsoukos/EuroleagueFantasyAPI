"""Εξαρτήσεις (dependencies) του FastAPI: κατάσταση εφαρμογής, υπηρεσία, έλεγχος `X-API-Key` και
κανονικοποίηση του `player_id`.

Έλεγχος ταυτότητας διαχειριστή: το header `X-API-Key` συγκρίνεται με το `ADMIN_API_KEY` των
ρυθμίσεων με `secrets.compare_digest` (σταθερός χρόνος). Αν το κλειδί δεν έχει οριστεί (κενό), τα
προστατευμένα endpoints είναι ΚΛΕΙΣΤΑ (HTTP 503), ποτέ ανοιχτά. Το κλειδί δεν γράφεται σε logs,
μηνύματα σφάλματος ή παραδείγματα του OpenAPI.
"""

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Path, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader
from sqlalchemy import Engine

from elfantasy.api.schemas import PLAYER_ID_PATTERN, PLAYER_ID_RE, normalise_player_id
from elfantasy.api.security import admin_key_is_usable, header_bytes
from elfantasy.api.services import PredictionService
from elfantasy.api.state import AppState

API_KEY_HEADER = "X-API-Key"

api_key_scheme = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,
    description=(
        "Κλειδί διαχειριστή για τα προστατευμένα endpoints (`POST /availability`, "
        "`DELETE /availability/{player_id}`, `POST /admin/refresh`). Ορίζεται από το "
        "περιβάλλον με `ADMIN_API_KEY` (τουλάχιστον 24 χαρακτήρες)· αν δεν έχει οριστεί ή είναι "
        "πιο σύντομο, τα endpoints αυτά είναι κλειστά."
    ),
)


def get_state(request: Request) -> AppState:
    """Η κατάσταση της εφαρμογής (δημιουργείται στο `create_app`)."""
    return request.app.state.api


def _unavailable_detail(state: AppState) -> str:
    if "model" in state.problems:
        return "prediction model is not available"
    if "database" in state.problems:
        return "database is not available"
    return "service is not ready"


def get_service(state: Annotated[AppState, Depends(get_state)]) -> PredictionService:
    """Η υπηρεσία πρόβλεψης, ή HTTP 503 αν το μοντέλο ή η βάση δεν φορτώθηκαν."""
    if state.service is None:
        raise HTTPException(status_code=503, detail=_unavailable_detail(state))
    return state.service


def get_engine(state: Annotated[AppState, Depends(get_state)]) -> Engine:
    """Η κοινή engine της βάσης (για τη διαθεσιμότητα), ή HTTP 503 αν δεν υπάρχει."""
    if state.engine is None or "database" in state.problems:
        raise HTTPException(status_code=503, detail="database is not available")
    return state.engine


def require_admin(
    state: Annotated[AppState, Depends(get_state)],
    api_key: Annotated[str | None, Security(api_key_scheme)] = None,
) -> None:
    """Επιτρέπει το αίτημα μόνο με σωστό `X-API-Key`.

    Κενό ή πολύ σύντομο `ADMIN_API_KEY` (< 24 χαρακτήρες) → 503 «admin API is disabled» (πριν από
    οποιονδήποτε έλεγχο κλειδιού). Κλειδί που λείπει ή είναι λάθος → 401.
    """
    configured = state.settings.admin_api_key if state.settings is not None else ""
    if not admin_key_is_usable(configured):
        raise HTTPException(status_code=503, detail="admin API is disabled")
    if api_key is None or not secrets.compare_digest(
        header_bytes(api_key), configured.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def valid_player_id(
    player_id: Annotated[
        str,
        Path(
            description=(
                "Κωδικός παίκτη: `P` + 6 ψηφία (π.χ. `P007200`) ή παλαιά μορφή `P` + 3 γράμματα "
                "(π.χ. `PADF`). Τα πεζά γράμματα και τα κενά στα άκρα γίνονται δεκτά και "
                f"κανονικοποιούνται. Μορφή μετά την κανονικοποίηση: `{PLAYER_ID_PATTERN}`. Για να "
                "βρεις το ID ενός παίκτη χρησιμοποίησε το `GET /players?search=`."
            ),
            examples=["P007200"],
        ),
    ],
) -> str:
    """Κανονικοποιεί (strip, κεφαλαία) και ελέγχει το `player_id` της διαδρομής. Άκυρη μορφή
    δίνει HTTP 422 στην ίδια μορφή με τα σφάλματα επικύρωσης του FastAPI."""
    value = normalise_player_id(player_id)
    if not PLAYER_ID_RE.fullmatch(value):
        raise RequestValidationError(
            [
                {
                    "type": "string_pattern_mismatch",
                    "loc": ("path", "player_id"),
                    "msg": f"String should match pattern '{PLAYER_ID_PATTERN}'",
                    "input": player_id,
                    "ctx": {"pattern": PLAYER_ID_PATTERN},
                }
            ]
        )
    return value
