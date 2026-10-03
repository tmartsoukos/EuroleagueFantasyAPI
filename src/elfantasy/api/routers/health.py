"""Endpoint `GET /health` (και `HEAD /health`, για monitors που χρησιμοποιούν HEAD)."""

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from elfantasy.api.deps import get_state
from elfantasy.api.health import build_health
from elfantasy.api.schemas import HealthResponse
from elfantasy.api.state import AppState

router = APIRouter(tags=["health"])


# Το HEAD δεν εμφανίζεται στο OpenAPI: δίνει τον ίδιο κωδικό (200 ή 503) για monitors που δεν
# υποστηρίζουν GET. Το σώμα της απάντησης δεν στέλνεται από τον server σε αίτημα HEAD.
@router.head("/health", include_in_schema=False)
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Κατάσταση της υπηρεσίας",
    response_description="Η υπηρεσία είναι υγιής (`status: ok`).",
    description=(
        "Ελαφρύς έλεγχος για το Render και την παρακολούθηση: HTTP 200 με `status: ok` μόνο αν "
        "η βάση απαντά και περιέχει παίκτες ΚΑΙ το μοντέλο φορτώθηκε. Αλλιώς HTTP 503 με "
        "`status: degraded` και λίστα `problems` (σύντομα μηνύματα, χωρίς εσωτερικές "
        "λεπτομέρειες). Δεν υπολογίζει προβλέψεις."
    ),
    responses={
        503: {
            "model": HealthResponse,
            "description": "Η υπηρεσία είναι degraded: λείπει το μοντέλο ή δεν φτάνει η βάση.",
        }
    },
)
def health(response: Response, state: Annotated[AppState, Depends(get_state)]) -> HealthResponse:
    report = build_health(state)
    if report.status != "ok":
        response.status_code = 503
    return report
