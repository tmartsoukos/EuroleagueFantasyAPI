"""Endpoint `GET /players` (αναζήτηση παικτών για την εύρεση του `player_id`)."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from elfantasy.api.deps import get_service
from elfantasy.api.responses import ERROR_503, error_responses
from elfantasy.api.schemas import TEAM_CODE_PATTERN, PlayersOut
from elfantasy.api.services import PredictionService

router = APIRouter(tags=["players"])


@router.get(
    "/players",
    response_model=PlayersOut,
    summary="Αναζήτηση παικτών",
    response_description="Οι παίκτες που ταιριάζουν στα κριτήρια.",
    description=(
        "Βρίσκει το `player_id` ενός παίκτη. Η αναζήτηση στο όνομα δεν κάνει διάκριση "
        "πεζών/κεφαλαίων και τόνων· κάθε λέξη του `search` πρέπει να εμφανίζεται (ως "
        "υποσυμβολοσειρά) στο όνομα, με οποιαδήποτε σειρά, άρα το `sasha vezenkov` βρίσκει τον "
        "`VEZENKOV, SASHA`. Επιστρέφονται και οι ανενεργοί παίκτες (`is_active: false`): πρώτα "
        "οι ενεργοί και μετά αλφαβητικά. Η ομάδα είναι αυτή της τελευταίας γραμμής του παίκτη."
    ),
    responses={
        **error_responses(404, "Άγνωστος κωδικός ομάδας (το μήνυμα περιέχει τους έγκυρους)."),
        **ERROR_503,
    },
)
def players(
    service: Annotated[PredictionService, Depends(get_service)],
    search: Annotated[
        str | None,
        Query(max_length=100, description="Κείμενο αναζήτησης στο όνομα του παίκτη (προαιρετικό)."),
    ] = None,
    team: Annotated[
        str | None,
        Query(
            pattern=TEAM_CODE_PATTERN,
            description="Κωδικός ομάδας (3 χαρακτήρες, πεζά ή κεφαλαία). Άγνωστος κωδικός: 404.",
        ),
    ] = None,
    limit: Annotated[
        int, Query(ge=1, le=100, description="Μέγιστο πλήθος αποτελεσμάτων (1 έως 100).")
    ] = 20,
) -> PlayersOut:
    return service.search_players(search=search, team=team, limit=limit)
