"""Endpoints διαθεσιμότητας παικτών: `GET`, `POST` και `DELETE /availability`.

Το `GET` είναι δημόσιο (μόνο ανάγνωση). Το `POST` και το `DELETE` είναι προστατευμένα με
`X-API-Key`. Η λογική και η αποθήκευση βρίσκονται στο `elfantasy.api.availability`.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import Engine

from elfantasy.api import availability as store
from elfantasy.api.availability import Availability, AvailabilityStatus
from elfantasy.api.deps import get_engine, get_state, require_admin, valid_player_id
from elfantasy.api.responses import ADMIN_RESPONSES, ERROR_503, error_responses
from elfantasy.api.schemas import AvailabilityEntry, AvailabilityIn, AvailabilityListOut
from elfantasy.api.services import NotFoundError, PlayerNotFound
from elfantasy.api.state import AppState

router = APIRouter(tags=["availability"])


def _entry(record: Availability) -> AvailabilityEntry:
    return AvailabilityEntry(
        player_id=record.player_id,
        name=record.name,
        status=record.status,
        source=record.source,
        note=record.note,
        expected_return=record.expected_return,
        updated_at=record.updated_at,
    )


@router.get(
    "/availability",
    response_model=AvailabilityListOut,
    summary="Λίστα διαθεσιμότητας παικτών",
    response_description="Οι εγγραφές διαθεσιμότητας.",
    description=(
        "Δημόσια λίστα (μόνο ανάγνωση) με τις χειροκίνητες εγγραφές διαθεσιμότητας: πρώτα οι "
        "`out`, μετά οι `doubtful` και μετά οι `available`. Παίκτης χωρίς εγγραφή θεωρείται "
        "διαθέσιμος."
    ),
    responses=ERROR_503,
)
def list_entries(
    engine: Annotated[Engine, Depends(get_engine)],
    status: Annotated[
        AvailabilityStatus | None,
        Query(description="Φίλτρο κατάστασης: `out`, `doubtful` ή `available`."),
    ] = None,
) -> AvailabilityListOut:
    records = store.list_availability(engine, status)
    return AvailabilityListOut(total=len(records), items=[_entry(record) for record in records])


@router.post(
    "/availability",
    response_model=AvailabilityEntry,
    dependencies=[Depends(require_admin)],
    summary="Καταχώρηση διαθεσιμότητας παίκτη (upsert)",
    response_description="Η εγγραφή όπως αποθηκεύτηκε.",
    description=(
        "Δημιουργεί ή **αντικαθιστά ολόκληρη** την εγγραφή διαθεσιμότητας του παίκτη (τα πεδία "
        "που δεν δίνονται γίνονται `null`). Απαιτεί header `X-API-Key`. Το `updated_at` ορίζεται "
        "από τον server (UTC). Η διαθεσιμότητα εφαρμόζεται μετά την πρόβλεψη και δεν περνά στο "
        "μοντέλο: `out` δίνει `predicted_fantasy` 0 και αποκλείει τον παίκτη από τα `/rankings`, "
        "`doubtful` προσθέτει προειδοποίηση. Το `predicted_pir` δεν αλλάζει."
    ),
    responses={
        **ADMIN_RESPONSES,
        **error_responses(404, "Άγνωστος παίκτης."),
    },
)
def upsert_entry(
    body: AvailabilityIn,
    engine: Annotated[Engine, Depends(get_engine)],
    state: Annotated[AppState, Depends(get_state)],
) -> AvailabilityEntry:
    if store.player_name(engine, body.player_id) is None:
        raise PlayerNotFound()
    record = store.save_availability(
        engine,
        player_id=body.player_id,
        status=body.status,
        source=body.source,
        note=body.note,
        expected_return=body.expected_return,
        updated_at=store.to_utc(state.clock()),
    )
    return _entry(record)


@router.delete(
    "/availability/{player_id}",
    status_code=204,
    response_class=Response,
    dependencies=[Depends(require_admin)],
    summary="Αφαίρεση του override διαθεσιμότητας",
    response_description="Η εγγραφή αφαιρέθηκε (χωρίς σώμα απάντησης).",
    description=(
        "Σβήνει την εγγραφή διαθεσιμότητας του παίκτη, ο οποίος θεωρείται πάλι διαθέσιμος. "
        "Απαιτεί header `X-API-Key`. Απαντά 404 αν ο παίκτης είναι άγνωστος ή δεν έχει εγγραφή."
    ),
    responses={
        **ADMIN_RESPONSES,
        **error_responses(404, "Άγνωστος παίκτης ή ο παίκτης δεν έχει εγγραφή διαθεσιμότητας."),
    },
)
def delete_entry(
    player_id: Annotated[str, Depends(valid_player_id)],
    engine: Annotated[Engine, Depends(get_engine)],
) -> Response:
    if store.player_name(engine, player_id) is None:
        raise PlayerNotFound()
    if not store.remove_availability(engine, player_id):
        raise NotFoundError("no availability record for this player")
    return Response(status_code=204)
