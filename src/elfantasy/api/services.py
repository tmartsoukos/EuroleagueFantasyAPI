"""Υπηρεσία πρόβλεψης: συνδυάζει τον `Predictor`, τη διαθεσιμότητα και τα δεδομένα αναφοράς
(ονόματα ομάδων και παικτών) σε απαντήσεις του API.

Το module δεν εξαρτάται από το FastAPI: οι «δεν βρέθηκε» καταστάσεις σηκώνουν εξαιρέσεις του
domain (`NotFoundError`) που ο ορισμός της εφαρμογής μετατρέπει σε HTTP 404. Έτσι η λογική
(ταξινόμηση, φίλτρα, στρογγυλοποίηση, σημειώσεις) ελέγχεται χωρίς HTTP.

Απόδοση: ο `Predictor` υπολογίζει την πρόβλεψη όλων των παικτών μία φορά ανά ημερομηνία `as_of`
και την κρατά σε cache, άρα τα rankings και οι αναζητήσεις δεν ξαναϋπολογίζουν ιστορικό ανά
αίτημα. Ανά αίτημα διαβάζεται μόνο ο μικρός πίνακας διαθεσιμότητας.
"""

from __future__ import annotations

import logging
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime

from sqlalchemy import Engine, func, select

from elfantasy.api.availability import (
    Availability,
    AvailabilityStatus,
    apply_availability,
    availability_by_player,
    get_availability,
    to_utc,
)
from elfantasy.api.schemas import (
    AvailabilityOut,
    NextGameOut,
    PlayersOut,
    PlayerSummary,
    PredictionOut,
    RankingItem,
    RankingsMeta,
    RankingsOut,
    RefreshOut,
)
from elfantasy.db.models import games, players, teams
from elfantasy.model.predict import (
    DEFAULT_ACTIVE_WINDOW_DAYS,
    NextGame,
    PlayerPrediction,
    Predictor,
)

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]

# Λιγότερες από τόσες προηγούμενες συμμετοχές: η πρόβλεψη είναι λιγότερο αξιόπιστη
# (docs/MODEL.md, ενότητα 7: τμήμα «λιγότερες από 5 προηγούμενες συμμετοχές»).
FEW_APPEARANCES = 5

NOTE_NO_GAME = "no scheduled game: neutral context"
NOTE_INACTIVE = (
    "player is not active (no games in the latest season or in the last "
    f"{DEFAULT_ACTIVE_WINDOW_DAYS} days): team and next game may be outdated"
)
NOTE_NO_HISTORY = "player has no previous appearances: prediction is based on the game context only"


class NotFoundError(LookupError):
    """Ό,τι ζητήθηκε δεν υπάρχει (παίκτης, ομάδα). Το `message` πηγαίνει στον client ως `detail`."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class PlayerNotFound(NotFoundError):
    """Άγνωστος παίκτης."""

    def __init__(self):
        super().__init__("player not found; use /players?search=<name> to look up a player_id")


class UnknownTeam(NotFoundError):
    """Άγνωστος κωδικός ομάδας: το μήνυμα περιέχει τους έγκυρους κωδικούς."""

    def __init__(self, code: str, valid_codes: list[str]):
        super().__init__(f"unknown team code {code!r}; valid codes: {', '.join(valid_codes)}")


# --------------------------------------------------------------------------------------
# Βοηθητικά
# --------------------------------------------------------------------------------------


def utc_now() -> datetime:
    """Η τρέχουσα ώρα σε UTC (προεπιλεγμένο ρολόι της εφαρμογής)."""
    return datetime.now(UTC)


def clock_today(clock: Clock) -> date:
    """Η ημερομηνία (UTC) που δίνει το ρολόι. Naive datetime θεωρείται UTC."""
    return to_utc(clock()).date()


def round2(value: float) -> float:
    """Στρογγυλοποίηση σε 2 δεκαδικά (χωρίς `-0.0`)."""
    return round(float(value), 2) + 0.0


# Γράμματα που δεν αναλύονται σε βασικό γράμμα + τόνο με το NFKD, και σημεία στίξης των ονομάτων.
_SEARCH_TRANSLATION = str.maketrans(
    {
        "đ": "d",
        "ł": "l",
        "ø": "o",
        "ı": "i",
        "æ": "ae",
        "œ": "oe",
        "'": "",
        "’": "",
        ".": "",
        "-": " ",
        ",": " ",
    }
)


def normalise_text(value: str) -> str:
    """Κανονικοποίηση για αναζήτηση: πεζά, χωρίς τόνους και σημεία στίξης, με ενιαία κενά.

    Π.χ. `"VEZENKOV, SASHA"` και `"Vezenkóv sasha"` δίνουν και τα δύο `"vezenkov sasha"`.
    """
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(stripped.translate(_SEARCH_TRANSLATION).split())


@dataclass(frozen=True)
class DatabaseSummary:
    """Σύνοψη των δεδομένων της βάσης (για το /health και το /admin/refresh)."""

    players: int
    latest_played_game_date: date | None
    next_scheduled_game_date: date | None


def database_summary(engine: Engine, today: date) -> DatabaseSummary:
    """Ελαφρύ query: πλήθος παικτών, τελευταίος παιγμένος αγώνας και επόμενος προγραμματισμένος.

    Χρησιμοποιεί τα indexes `games(played, game_date)`, άρα είναι ανεξάρτητο από το μέγεθος του
    ιστορικού. Δεν υπολογίζει πρόβλεψη.
    """
    with engine.connect() as connection:
        player_count = connection.execute(select(func.count()).select_from(players)).scalar_one()
        latest = connection.execute(
            select(func.max(games.c.game_date)).where(games.c.played.is_(True))
        ).scalar_one_or_none()
        upcoming = connection.execute(
            select(func.min(games.c.game_date)).where(
                games.c.played.is_(False), games.c.game_date >= today
            )
        ).scalar_one_or_none()
    return DatabaseSummary(int(player_count), latest, upcoming)


@dataclass(frozen=True)
class _PlayerRef:
    name: str
    last_season: int
    search_key: str


# --------------------------------------------------------------------------------------
# Η υπηρεσία
# --------------------------------------------------------------------------------------


class PredictionService:
    """Εφαρμόζει τους κανόνες του API πάνω στον `Predictor` και τη βάση."""

    def __init__(self, predictor: Predictor, engine: Engine, clock: Clock = utc_now):
        self._predictor = predictor
        self._engine = engine
        self._clock = clock
        self._team_names: dict[str, str] = {}
        self._players: dict[str, _PlayerRef] = {}
        self._summary = DatabaseSummary(0, None, None)
        self.reload_reference_data()

    # ----- δεδομένα αναφοράς -----
    def today(self) -> date:
        """Η σημερινή ημερομηνία (UTC) από το ρολόι της εφαρμογής: το `as_of` των προβλέψεων."""
        return clock_today(self._clock)

    def reload_reference_data(self) -> None:
        """Διαβάζει ονόματα ομάδων, παίκτες και data cutoff από τη βάση (στο startup και στο
        refresh). Οι cache αντικαθίστανται ατομικά, ώστε τα αιτήματα που τρέχουν να μη δουν
        μισοτελειωμένη κατάσταση."""
        with self._engine.connect() as connection:
            team_rows = connection.execute(select(teams.c.team_code, teams.c.name)).all()
            player_rows = connection.execute(
                select(players.c.player_id, players.c.name, players.c.last_season)
            ).all()
        self._team_names = {code: name for code, name in team_rows}
        self._players = {
            player_id: _PlayerRef(name, int(last_season), normalise_text(name))
            for player_id, name, last_season in player_rows
        }
        self._summary = database_summary(self._engine, self.today())

    def warm_up(self) -> None:
        """Υπολογίζει εκ των προτέρων τις προβλέψεις της ημέρας, ώστε το πρώτο αίτημα να είναι
        γρήγορο και ένα πρόβλημα στον υπολογισμό να φανεί στο startup."""
        self._predictor.predict_all(as_of=self.today(), active_only=False)

    @property
    def model_version(self) -> str:
        return self._predictor.model_version

    @property
    def loaded_through(self) -> date | None:
        """Ο τελευταίος παιγμένος αγώνας που έχουν φορτώσει οι προβλέψεις (data cutoff)."""
        return self._summary.latest_played_game_date

    @property
    def team_codes(self) -> list[str]:
        """Όλοι οι γνωστοί κωδικοί ομάδων (πίνακας `teams`), αλφαβητικά."""
        return sorted(self._team_names)

    def team_name(self, code: str) -> str:
        return self._team_names.get(code, code)

    def require_team(self, code: str) -> str:
        """Κανονικοποιεί έναν κωδικό ομάδας (strip, κεφαλαία) ή σηκώνει `UnknownTeam`."""
        normalised = code.strip().upper()
        if normalised not in self._team_names:
            raise UnknownTeam(code, self.team_codes)
        return normalised

    # ----- πρόβλεψη παίκτη -----
    def predict(self, player_id: str, *, include_features: bool = False) -> PredictionOut:
        """Η πλήρης απάντηση του `/predict/{player_id}`. Σηκώνει `PlayerNotFound`."""
        today = self.today()
        prediction = self._predictor.predict_player(player_id, as_of=today)
        if prediction is None:
            raise PlayerNotFound()
        availability = get_availability(self._engine, prediction.player_id)
        effective = apply_availability(prediction.predicted_fantasy, availability, today=today)
        notes = [*effective.notes, *_context_notes(prediction)]
        fields: dict = {}
        if include_features:
            fields["features"] = {
                name: None if value is None else round(value, 4)
                for name, value in prediction.features.items()
            }
        return PredictionOut(
            player_id=prediction.player_id,
            name=prediction.name,
            team_code=prediction.team_code,
            team_name=self.team_name(prediction.team_code),
            is_active=prediction.is_active,
            next_game=self._next_game(prediction.next_game),
            predicted_fantasy=round2(effective.fantasy),
            model_predicted_fantasy=round2(prediction.predicted_fantasy),
            predicted_pir=round2(prediction.predicted_pir),
            availability=_availability_out(availability),
            n_prior_appearances=prediction.n_prior_appearances,
            last_appearance_date=prediction.last_appearance_date,
            model_version=prediction.model_version,
            notes=notes,
            **fields,
        )

    def _next_game(self, game: NextGame | None) -> NextGameOut | None:
        if game is None:
            return None
        return NextGameOut(
            season=game.season,
            gamecode=game.gamecode,
            game_date=game.game_date,
            tipoff_utc=None if game.tipoff_utc is None else to_utc(game.tipoff_utc),
            opponent_code=game.opp_code,
            opponent_name=self.team_name(game.opp_code),
            home=game.home,
        )

    # ----- rankings -----
    def rankings(
        self,
        *,
        limit: int,
        offset: int,
        team: str | None,
        include_unavailable: bool,
        active_only: bool,
    ) -> RankingsOut:
        """Παίκτες ταξινομημένοι φθίνουσα κατά το αποτελεσματικό `predicted_fantasy`.

        Οι `out` εξαιρούνται, εκτός αν `include_unavailable`: τότε εμφανίζονται πάντα στο τέλος
        (με αποτελεσματική τιμή 0), ταξινομημένοι κατά την πρόβλεψη του μοντέλου. Το `rank` είναι
        η θέση στη φιλτραρισμένη λίστα, ανεξάρτητη από το `offset`.
        """
        today = self.today()
        team_code = None if team is None else self.require_team(team)
        predictions = self._predictor.predict_all(
            as_of=today, team_code=team_code, active_only=active_only
        )
        records = availability_by_player(self._engine)
        ranked: list[tuple[PlayerPrediction, float, AvailabilityStatus]] = []
        for prediction in predictions:
            record = records.get(prediction.player_id)
            status = record.status if record is not None else AvailabilityStatus.AVAILABLE
            if status is AvailabilityStatus.OUT and not include_unavailable:
                continue
            effective = apply_availability(prediction.predicted_fantasy, record)
            ranked.append((prediction, effective.fantasy, status))
        ranked.sort(
            key=lambda entry: (
                entry[2] is AvailabilityStatus.OUT,
                -entry[1],
                -entry[0].predicted_fantasy,
                entry[0].player_id,
            )
        )
        items = [
            _ranking_item(position, prediction, effective, status)
            for position, (prediction, effective, status) in enumerate(
                ranked[offset : offset + limit], start=offset + 1
            )
        ]
        meta = RankingsMeta(
            as_of=today,
            model_version=self.model_version,
            total=len(ranked),
            limit=limit,
            offset=offset,
        )
        return RankingsOut(meta=meta, items=items)

    # ----- αναζήτηση παικτών -----
    def search_players(self, *, search: str | None, team: str | None, limit: int) -> PlayersOut:
        """Αναζήτηση παικτών κατά όνομα (χωρίς διάκριση πεζών/κεφαλαίων και τόνων) και ομάδα.

        Κάθε λέξη του `search` πρέπει να εμφανίζεται ως υποσυμβολοσειρά στο όνομα, με οποιαδήποτε
        σειρά. Πρώτοι οι ενεργοί παίκτες και μετά αλφαβητικά.
        """
        today = self.today()
        team_code = None if team is None else self.require_team(team)
        tokens = normalise_text(search).split() if search else []
        predictions = self._predictor.predict_all(
            as_of=today, team_code=team_code, active_only=False
        )
        matches: list[tuple[PlayerPrediction, _PlayerRef]] = []
        for prediction in predictions:
            ref = self._players.get(prediction.player_id)
            if ref is None or not all(token in ref.search_key for token in tokens):
                continue
            matches.append((prediction, ref))
        matches.sort(
            key=lambda match: (not match[0].is_active, match[1].search_key, match[0].player_id)
        )
        items = [
            PlayerSummary(
                player_id=prediction.player_id,
                name=ref.name,
                team_code=prediction.team_code,
                team_name=self.team_name(prediction.team_code),
                last_season=ref.last_season,
                is_active=prediction.is_active,
            )
            for prediction, ref in matches[:limit]
        ]
        return PlayersOut(total=len(matches), limit=limit, items=items)

    # ----- ανανέωση δεδομένων -----
    def refresh(self) -> RefreshOut:
        """Ξαναδιαβάζει τη βάση (μετά από νέο ingestion) και ξαναϋπολογίζει τις προβλέψεις."""
        started = time.perf_counter()
        previous = self._summary.latest_played_game_date
        self._predictor.refresh()
        self.reload_reference_data()
        self.warm_up()
        summary = self._summary
        logger.info(
            "data refreshed: latest played game %s (previously %s)",
            summary.latest_played_game_date,
            previous,
        )
        return RefreshOut(
            status="refreshed",
            model_version=self.model_version,
            players=summary.players,
            previous_latest_played_game_date=previous,
            latest_played_game_date=summary.latest_played_game_date,
            next_scheduled_game_date=summary.next_scheduled_game_date,
            refreshed_at=to_utc(self._clock()),
            duration_seconds=round(time.perf_counter() - started, 3),
        )


# --------------------------------------------------------------------------------------
# Μορφοποίηση
# --------------------------------------------------------------------------------------


def _availability_out(availability: Availability | None) -> AvailabilityOut:
    """Η διαθεσιμότητα στην απάντηση: χωρίς εγγραφή, `available` και όλα τα άλλα `null`."""
    if availability is None:
        return AvailabilityOut(
            status=AvailabilityStatus.AVAILABLE,
            source=None,
            note=None,
            expected_return=None,
            updated_at=None,
        )
    return AvailabilityOut(
        status=availability.status,
        source=availability.source,
        note=availability.note,
        expected_return=availability.expected_return,
        updated_at=availability.updated_at,
    )


def _context_notes(prediction: PlayerPrediction) -> list[str]:
    """Προειδοποιήσεις για το πλαίσιο της πρόβλεψης (όχι για τη διαθεσιμότητα)."""
    notes = []
    if prediction.next_game is None:
        notes.append(NOTE_NO_GAME)
    if not prediction.is_active:
        notes.append(NOTE_INACTIVE)
    appearances = prediction.n_prior_appearances
    if appearances == 0:
        notes.append(NOTE_NO_HISTORY)
    elif appearances < FEW_APPEARANCES:
        notes.append(
            f"player has only {appearances} previous appearance{'s' if appearances > 1 else ''}: "
            "prediction is less reliable"
        )
    return notes


def _ranking_item(
    rank: int,
    prediction: PlayerPrediction,
    effective_fantasy: float,
    status: AvailabilityStatus,
) -> RankingItem:
    game = prediction.next_game
    return RankingItem(
        rank=rank,
        player_id=prediction.player_id,
        name=prediction.name,
        team_code=prediction.team_code,
        predicted_fantasy=round2(effective_fantasy),
        model_predicted_fantasy=round2(prediction.predicted_fantasy),
        predicted_pir=round2(prediction.predicted_pir),
        next_opponent_code=None if game is None else game.opp_code,
        next_home=None if game is None else game.home,
        next_game_date=None if game is None else game.game_date,
        availability_status=status,
        is_active=prediction.is_active,
    )
