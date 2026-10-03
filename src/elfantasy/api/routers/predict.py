"""Endpoints `GET /predict/{player_id}` και `GET /rankings`."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from elfantasy.api.deps import get_service, valid_player_id
from elfantasy.api.responses import ERROR_503, error_responses
from elfantasy.api.schemas import TEAM_CODE_PATTERN, PredictionOut, RankingsOut
from elfantasy.api.services import PredictionService

router = APIRouter(tags=["predictions"])

PREDICT_DESCRIPTION = """\
Προβλέπει το fantasy score του παίκτη στον **επόμενο προγραμματισμένο αγώνα** της ομάδας του.

* `predicted_fantasy`: η αποτελεσματική πρόβλεψη μετά το override διαθεσιμότητας. Είναι «τυπική»
  (διάμεσος-like) τιμή υπό την προϋπόθεση ότι ο παίκτης αγωνίζεται, χωρίς captain ×2 και πάγκο
  ×0,5. Για παίκτη `out` είναι 0.
* `model_predicted_fantasy`: η ακατέργαστη έξοδος του μοντέλου (χωρίς το override).
* `predicted_pir`: πρόβλεψη PIR από ανεξάρτητο μοντέλο· δεν επηρεάζεται από τη διαθεσιμότητα.
* `notes`: προειδοποιήσεις (π.χ. δεν υπάρχει προγραμματισμένος αγώνας, ο παίκτης είναι `out` ή
  `doubtful`, ανενεργός παίκτης, λίγες προηγούμενες συμμετοχές).

Το μοντέλο δεν γνωρίζει τραυματισμούς: χρησιμοποίησε το `availability` (χειροκίνητη καταχώρηση)."""

RANKINGS_DESCRIPTION = """\
Παίκτες ταξινομημένοι φθίνουσα κατά το **αποτελεσματικό** `predicted_fantasy` (μετά το override
διαθεσιμότητας). Οι παίκτες `out` εξαιρούνται, εκτός αν `include_unavailable=true`: τότε
εμφανίζονται πάντα στο τέλος, με αποτελεσματική τιμή 0 (η ακατέργαστη τιμή του μοντέλου είναι στο
`model_predicted_fantasy`). Οι `doubtful` εμφανίζονται κανονικά, με `availability_status:
doubtful`. Το `rank` είναι η θέση στη φιλτραρισμένη λίστα και δεν εξαρτάται από το `offset`.

Η πρώτη κλήση της ημέρας υπολογίζει τις προβλέψεις όλων των παικτών (~1 s)· οι επόμενες
εξυπηρετούνται από cache."""


@router.get(
    "/predict/{player_id}",
    response_model=PredictionOut,
    response_model_exclude_unset=True,
    summary="Πρόβλεψη fantasy score ενός παίκτη",
    response_description="Η πρόβλεψη για τον επόμενο αγώνα του παίκτη.",
    description=PREDICT_DESCRIPTION,
    responses={
        **error_responses(404, "Άγνωστος παίκτης."),
        **ERROR_503,
    },
)
def predict(
    player_id: Annotated[str, Depends(valid_player_id)],
    service: Annotated[PredictionService, Depends(get_service)],
    include_features: Annotated[
        bool,
        Query(description="Αν `true`, η απάντηση περιλαμβάνει και τα 42 features του μοντέλου."),
    ] = False,
) -> PredictionOut:
    return service.predict(player_id, include_features=include_features)


@router.get(
    "/rankings",
    response_model=RankingsOut,
    summary="Rankings παικτών κατά προβλεπόμενο fantasy score",
    response_description="Μία σελίδα των rankings, με μεταδεδομένα.",
    description=RANKINGS_DESCRIPTION,
    responses={
        **error_responses(404, "Άγνωστος κωδικός ομάδας (το μήνυμα περιέχει τους έγκυρους)."),
        **ERROR_503,
    },
)
def rankings(
    service: Annotated[PredictionService, Depends(get_service)],
    limit: Annotated[
        int, Query(ge=1, le=500, description="Μέγιστο πλήθος παικτών στην απάντηση (1 έως 500).")
    ] = 50,
    offset: Annotated[
        int, Query(ge=0, description="Πόσοι παίκτες παραλείπονται από την αρχή.")
    ] = 0,
    team: Annotated[
        str | None,
        Query(
            pattern=TEAM_CODE_PATTERN,
            description=(
                "Κωδικός ομάδας (3 χαρακτήρες, π.χ. `OLY`· πεζά ή κεφαλαία). "
                "Άγνωστος κωδικός δίνει 404."
            ),
        ),
    ] = None,
    include_unavailable: Annotated[
        bool, Query(description="Αν `true`, περιλαμβάνονται και οι παίκτες `out` (στο τέλος).")
    ] = False,
    active_only: Annotated[
        bool,
        Query(
            description=(
                "Αν `true` (προεπιλογή), μόνο οι ενεργοί παίκτες της τρέχουσας σεζόν. Αν `false`, "
                "όλοι οι γνωστοί παίκτες της βάσης (και όσοι έχουν φύγει από τη λίγκα)."
            )
        ),
    ] = True,
) -> RankingsOut:
    return service.rankings(
        limit=limit,
        offset=offset,
        team=team,
        include_unavailable=include_unavailable,
        active_only=active_only,
    )
