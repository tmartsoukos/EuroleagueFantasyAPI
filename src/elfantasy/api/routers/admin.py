"""Endpoint διαχείρισης `POST /admin/refresh`."""

from typing import Annotated

from fastapi import APIRouter, Depends

from elfantasy.api.deps import get_service, require_admin
from elfantasy.api.responses import ADMIN_RESPONSES
from elfantasy.api.schemas import RefreshOut
from elfantasy.api.services import PredictionService

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post(
    "/refresh",
    response_model=RefreshOut,
    dependencies=[Depends(require_admin)],
    summary="Ανανέωση δεδομένων από τη βάση",
    response_description="Η ανανέωση ολοκληρώθηκε, με το νέο data cutoff.",
    description=(
        "Ξαναδιαβάζει ιστορικό, αγώνες, πρόγραμμα και ονόματα από τη βάση και ξαναϋπολογίζει τις "
        "προβλέψεις. Χρησιμοποιείται μετά από νέο ingestion: αλλιώς η υπηρεσία δεν βλέπει τα "
        "νέα δεδομένα μέχρι να γίνει επανεκκίνηση. Η απάντηση αναφέρει το νέο data cutoff "
        "(`latest_played_game_date`). Διαρκεί περίπου ένα δευτερόλεπτο. Απαιτεί header "
        "`X-API-Key`."
    ),
    responses=ADMIN_RESPONSES,
)
def refresh(service: Annotated[PredictionService, Depends(get_service)]) -> RefreshOut:
    return service.refresh()
