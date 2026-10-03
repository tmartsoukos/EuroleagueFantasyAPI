"""Διαθεσιμότητα παικτών (τραυματισμοί, απουσίες): λογική του override και αποθήκευση.

Το μοντέλο προβλέπει το fantasy score υπό την προϋπόθεση ότι ο παίκτης αγωνίζεται και αγνοεί
τραυματισμούς, απουσίες και αλλαγές ρόστερ που δεν φαίνονται στο ιστορικό (docs/MODEL.md,
ενότητα 14). Η διαθεσιμότητα είναι χειροκίνητη πληροφορία που καταχωρεί διαχειριστής στον πίνακα
`player_availability` και εφαρμόζεται ΜΕΤΑ την πρόβλεψη: δεν περνά ποτέ ως feature στο μοντέλο,
γιατί δεν υπάρχει στο ιστορικό εκπαίδευσης.

Σημασιολογία
------------
* `out`: το αποτελεσματικό `predicted_fantasy` είναι 0,0 (ο παίκτης δεν αγωνίζεται). Η ακατέργαστη
  έξοδος του μοντέλου παραμένει στο `model_predicted_fantasy`. Το `predicted_pir` ΔΕΝ αλλάζει: είναι
  πάντα η ακατέργαστη τιμή του μοντέλου, ακόμη και για παίκτη `out`.
* `doubtful`: η πρόβλεψη μένει ίδια και προστίθεται προειδοποίηση.
* `available` ή καμία εγγραφή: καμία αλλαγή.

Το module δεν εξαρτάται από το FastAPI, ώστε η λογική και η αποθήκευση να ελέγχονται χωρίς HTTP.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import UTC, date, datetime

from sqlalchemy import Engine, delete, select

from elfantasy.db.models import player_availability, players
from elfantasy.db.session import upsert


class AvailabilityStatus(enum.StrEnum):
    """Κατάσταση διαθεσιμότητας ενός παίκτη."""

    OUT = "out"
    DOUBTFUL = "doubtful"
    AVAILABLE = "available"


# Σειρά σοβαρότητας για την ταξινόμηση λιστών: πρώτοι όσοι απουσιάζουν.
_SEVERITY = {
    AvailabilityStatus.OUT: 0,
    AvailabilityStatus.DOUBTFUL: 1,
    AvailabilityStatus.AVAILABLE: 2,
}

# Μηνύματα προς τους clients (αγγλικά, στο πεδίο `notes` της απάντησης).
NOTE_OUT = "player is marked out: effective prediction is 0"
NOTE_DOUBTFUL = "player is marked doubtful: prediction assumes the player plays"
NOTE_STALE = "expected return date has passed: the availability record may be outdated"


@dataclass(frozen=True)
class Availability:
    """Εγγραφή διαθεσιμότητας. Το `updated_at` είναι πάντα datetime UTC με ζώνη ώρας."""

    player_id: str
    status: AvailabilityStatus
    source: str | None = None
    note: str | None = None
    expected_return: date | None = None
    updated_at: datetime | None = None
    name: str | None = None  # όνομα παίκτη, όταν η εγγραφή διαβάζεται από τη βάση


@dataclass(frozen=True)
class EffectivePrediction:
    """Το αποτέλεσμα του override: αποτελεσματικό fantasy και προειδοποιήσεις διαθεσιμότητας."""

    fantasy: float
    notes: tuple[str, ...]


def apply_availability(
    model_fantasy: float,
    availability: Availability | None,
    *,
    today: date | None = None,
) -> EffectivePrediction:
    """Εφαρμόζει τη διαθεσιμότητα στην ακατέργαστη πρόβλεψη fantasy του μοντέλου.

    `out` δίνει 0,0, `doubtful` αφήνει την τιμή ως έχει με προειδοποίηση, και `available` ή
    απουσία εγγραφής δεν αλλάζουν τίποτα. Αν δοθεί `today` και η `expected_return` έχει περάσει
    ενώ ο παίκτης εξακολουθεί να είναι `out` ή `doubtful`, προστίθεται προειδοποίηση ότι η εγγραφή
    μπορεί να είναι παλιά (οι εγγραφές είναι χειροκίνητες και δεν λήγουν μόνες τους).
    """
    status = availability.status if availability is not None else AvailabilityStatus.AVAILABLE
    notes: list[str] = []
    fantasy = model_fantasy
    if status is AvailabilityStatus.OUT:
        fantasy = 0.0
        notes.append(NOTE_OUT)
    elif status is AvailabilityStatus.DOUBTFUL:
        notes.append(NOTE_DOUBTFUL)
    if (
        availability is not None
        and status is not AvailabilityStatus.AVAILABLE
        and today is not None
        and availability.expected_return is not None
        and availability.expected_return < today
    ):
        notes.append(NOTE_STALE)
    return EffectivePrediction(fantasy=fantasy, notes=tuple(notes))


def to_utc(moment: datetime) -> datetime:
    """Το `moment` ως datetime UTC με ζώνη ώρας.

    Naive datetime θεωρείται ήδη UTC: έτσι επιστρέφει η SQLite τα `DateTime(timezone=True)`,
    ενώ το Postgres επιστρέφει datetime με ζώνη.
    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


# --------------------------------------------------------------------------------------
# Αποθήκευση (πίνακας player_availability)
# --------------------------------------------------------------------------------------


def ensure_table(engine: Engine) -> None:
    """Δημιουργεί τον πίνακα `player_availability` αν λείπει (checkfirst). Ασφαλές να ξανατρέξει."""
    player_availability.create(engine, checkfirst=True)


def _query():
    """Η βασική επιλογή: εγγραφές διαθεσιμότητας μαζί με το όνομα του παίκτη."""
    return select(
        player_availability.c.player_id,
        player_availability.c.status,
        player_availability.c.source,
        player_availability.c.note,
        player_availability.c.expected_return,
        player_availability.c.updated_at,
        players.c.name,
    ).select_from(
        player_availability.join(players, players.c.player_id == player_availability.c.player_id)
    )


def _record(row) -> Availability:
    return Availability(
        player_id=row["player_id"],
        status=AvailabilityStatus(row["status"]),
        source=row["source"],
        note=row["note"],
        expected_return=row["expected_return"],
        updated_at=to_utc(row["updated_at"]),
        name=row["name"],
    )


def get_availability(engine: Engine, player_id: str) -> Availability | None:
    """Η εγγραφή διαθεσιμότητας ενός παίκτη, ή None αν δεν υπάρχει."""
    statement = _query().where(player_availability.c.player_id == player_id)
    with engine.connect() as connection:
        row = connection.execute(statement).mappings().first()
    return None if row is None else _record(row)


def list_availability(
    engine: Engine, status: AvailabilityStatus | None = None
) -> list[Availability]:
    """Όλες οι εγγραφές (προαιρετικά με φίλτρο κατάστασης): πρώτα `out`, μετά `doubtful`,
    μετά `available`, και μέσα σε κάθε κατάσταση κατά όνομα."""
    statement = _query()
    if status is not None:
        statement = statement.where(player_availability.c.status == status.value)
    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()
    records = [_record(row) for row in rows]
    records.sort(key=lambda record: (_SEVERITY[record.status], record.name or "", record.player_id))
    return records


def availability_by_player(engine: Engine) -> dict[str, Availability]:
    """Όλες οι εγγραφές ανά `player_id` (ένα μικρό query ανά αίτημα στα rankings)."""
    return {record.player_id: record for record in list_availability(engine)}


def save_availability(
    engine: Engine,
    *,
    player_id: str,
    status: AvailabilityStatus,
    source: str | None,
    note: str | None,
    expected_return: date | None,
    updated_at: datetime,
) -> Availability:
    """Εισάγει ή αντικαθιστά ολόκληρη την εγγραφή του παίκτη (idempotent upsert) και την επιστρέφει.

    Ο παίκτης πρέπει να υπάρχει στον πίνακα `players` (αλλιώς το foreign key απορρίπτει την
    εγγραφή): ο έλεγχος γίνεται από τον καλούντα. Τα πεδία που δεν δίνονται γίνονται NULL.
    """
    row = {
        "player_id": player_id,
        "status": AvailabilityStatus(status).value,
        "source": source,
        "note": note,
        "expected_return": expected_return,
        "updated_at": to_utc(updated_at),
    }
    with engine.begin() as connection:
        upsert(connection, player_availability, [row], ["player_id"])
        stored = (
            connection.execute(_query().where(player_availability.c.player_id == player_id))
            .mappings()
            .one()
        )
    return _record(stored)


def remove_availability(engine: Engine, player_id: str) -> bool:
    """Διαγράφει την εγγραφή του παίκτη. Επιστρέφει False αν δεν υπήρχε."""
    with engine.begin() as connection:
        result = connection.execute(
            delete(player_availability).where(player_availability.c.player_id == player_id)
        )
    return result.rowcount > 0


def player_name(engine: Engine, player_id: str) -> str | None:
    """Το όνομα του παίκτη από τον πίνακα `players`, ή None αν ο παίκτης δεν υπάρχει."""
    with engine.connect() as connection:
        return connection.execute(
            select(players.c.name).where(players.c.player_id == player_id)
        ).scalar_one_or_none()
