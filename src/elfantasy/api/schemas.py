"""Μοντέλα Pydantic (αιτήματα και απαντήσεις) του API.

Οι περιγραφές και τα παραδείγματα εμφανίζονται στο Swagger (`/docs`). Τα ονόματα των πεδίων JSON
είναι στα αγγλικά. Το `predicted_fantasy` είναι «τυπική» (διάμεσος-like) τιμή υπό την προϋπόθεση
ότι ο παίκτης αγωνίζεται, χωρίς captain ×2 ή πάγκο ×0,5 (docs/MODEL.md, ενότητες 9 και 14).
"""

import re
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from elfantasy.api.availability import AvailabilityStatus

# Τα IDs των παικτών έχουν δύο μορφές: `P` + 6 ψηφία (π.χ. P007200) ή παλαιά `P` + 3 γράμματα
# (π.χ. PADF). Το API κάνει strip και κεφαλαία πριν από τον έλεγχο (docs/DATA_SOURCES.md, §5.10).
PLAYER_ID_PATTERN = r"^P[A-Z0-9]{3,6}$"
PLAYER_ID_RE = re.compile(PLAYER_ID_PATTERN)
TEAM_CODE_PATTERN = r"^[A-Za-z0-9]{3}$"

NOTE_MAX_LENGTH = 500
SOURCE_MAX_LENGTH = 100


def normalise_player_id(value: str) -> str:
    """Κανονικοποίηση του ID παίκτη: αφαίρεση κενών στα άκρα και κεφαλαία γράμματα."""
    return value.strip().upper()


class ApiModel(BaseModel):
    """Βάση των μοντέλων του API.

    Το `protected_namespaces=()` επιτρέπει πεδία που ξεκινούν από `model_` (π.χ. `model_version`),
    ανεξάρτητα από την έκδοση του Pydantic.
    """

    model_config = ConfigDict(protected_namespaces=())


# --------------------------------------------------------------------------------------
# Πρόβλεψη παίκτη
# --------------------------------------------------------------------------------------


class NextGameOut(ApiModel):
    """Ο επόμενος προγραμματισμένος αγώνας της ομάδας του παίκτη."""

    season: int = Field(description="Έτος έναρξης της σεζόν (π.χ. 2026 για τη σεζόν 2026-27).")
    gamecode: int = Field(description="Κωδικός αγώνα μέσα στη σεζόν.")
    game_date: date = Field(description="Ημερομηνία του αγώνα, όπως στο πρόγραμμα της Euroleague.")
    tipoff_utc: datetime | None = Field(
        description="Ώρα έναρξης σε UTC (ISO-8601 με το επίθημα `Z`), ή `null` αν δεν είναι γνωστή."
    )
    opponent_code: str = Field(description="Κωδικός ομάδας του αντιπάλου (3 γράμματα).")
    opponent_name: str = Field(description="Όνομα της ομάδας του αντιπάλου.")
    home: bool = Field(description="`true` αν η ομάδα του παίκτη παίζει εντός έδρας.")


class AvailabilityOut(ApiModel):
    """Διαθεσιμότητα παίκτη (χειροκίνητη πληροφορία διαχειριστή)."""

    status: AvailabilityStatus = Field(
        default=AvailabilityStatus.AVAILABLE,
        description=(
            "`out`: δεν αγωνίζεται, το `predicted_fantasy` είναι 0. `doubtful`: αμφίβολος, η "
            "πρόβλεψη μένει ίδια με προειδοποίηση. `available`: διαθέσιμος (και η τιμή όταν δεν "
            "υπάρχει καμία εγγραφή)."
        ),
    )
    source: str | None = Field(
        default=None, description="Προέλευση της πληροφορίας (ελεύθερο κείμενο)."
    )
    note: str | None = Field(
        default=None, description="Σημείωση του διαχειριστή (έως 500 χαρακτήρες)."
    )
    expected_return: date | None = Field(
        default=None, description="Αναμενόμενη επιστροφή του παίκτη (ημερομηνία), αν είναι γνωστή."
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Πότε καταχωρήθηκε η εγγραφή (UTC, από τον server). `null` χωρίς εγγραφή.",
    )


class PredictionOut(ApiModel):
    """Πρόβλεψη fantasy score για τον επόμενο αγώνα ενός παίκτη."""

    player_id: str = Field(description="Κωδικός παίκτη (κανονικοποιημένος: κεφαλαία, χωρίς κενά).")
    name: str = Field(
        description="Όνομα του παίκτη (το πιο πρόσφατο της βάσης, μορφή `ΕΠΩΝΥΜΟ, ΟΝΟΜΑ`)."
    )
    team_code: str = Field(
        description="Κωδικός της ομάδας του (η ομάδα της τελευταίας γραμμής του)."
    )
    team_name: str = Field(description="Όνομα της ομάδας του.")
    is_active: bool = Field(
        description=(
            "`true` αν ο παίκτης έχει γραμμή αγώνα στη νεότερη σεζόν που έχει ξεκινήσει ή μέσα "
            "στις τελευταίες 45 ημέρες. Αλλιώς (π.χ. έφυγε από τη λίγκα) η ομάδα και ο επόμενος "
            "αγώνας μπορεί να είναι παλιά."
        )
    )
    next_game: NextGameOut | None = Field(
        description="Ο επόμενος αγώνας της ομάδας του, ή `null` αν δεν υπάρχει προγραμματισμένος."
    )
    predicted_fantasy: float = Field(
        description=(
            "Αποτελεσματική πρόβλεψη fantasy score μετά το override διαθεσιμότητας: ίση με το "
            "`model_predicted_fantasy`, εκτός από τους παίκτες `out` όπου είναι 0. «Τυπική» "
            "(διάμεσος-like) τιμή υπό την προϋπόθεση ότι ο παίκτης αγωνίζεται, χωρίς captain ×2 "
            "και πάγκο ×0,5. Στρογγυλεμένη σε 2 δεκαδικά."
        )
    )
    model_predicted_fantasy: float = Field(
        description="Ακατέργαστη έξοδος του μοντέλου για το fantasy score (2 δεκαδικά)."
    )
    predicted_pir: float = Field(
        description=(
            "Πρόβλεψη PIR από ανεξάρτητο μοντέλο. Δεν επηρεάζεται από τη διαθεσιμότητα: είναι "
            "πάντα η ακατέργαστη τιμή του μοντέλου, ακόμη και για παίκτη `out` (2 δεκαδικά)."
        )
    )
    availability: AvailabilityOut = Field(description="Διαθεσιμότητα του παίκτη.")
    n_prior_appearances: int = Field(
        description="Πόσες συμμετοχές (με λεπτά > 0) έχει στο ιστορικό. 0 = χωρίς ιστορικό."
    )
    last_appearance_date: date | None = Field(
        description="Ημερομηνία της τελευταίας συμμετοχής του, ή `null`."
    )
    model_version: str = Field(description="Έκδοση του μοντέλου που έδωσε την πρόβλεψη.")
    notes: list[str] = Field(
        description="Προειδοποιήσεις για την ερμηνεία της πρόβλεψης (κενή λίστα αν δεν υπάρχουν)."
    )
    features: dict[str, float | None] | None = Field(
        default=None,
        description=(
            "Οι τιμές των 42 features του μοντέλου (4 δεκαδικά, `null` όπου λείπουν). Υπάρχει "
            "μόνο με `include_features=true`."
        ),
    )

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {
                    "player_id": "P003469",
                    "name": "VEZENKOV, SASHA",
                    "team_code": "OLY",
                    "team_name": "OLYMPIACOS PIRAEUS",
                    "is_active": True,
                    "next_game": {
                        "season": 2026,
                        "gamecode": 40,
                        "game_date": "2026-10-09",
                        "tipoff_utc": "2026-10-09T18:15:00Z",
                        "opponent_code": "IST",
                        "opponent_name": "ANADOLU EFES ISTANBUL",
                        "home": True,
                    },
                    "predicted_fantasy": 20.13,
                    "model_predicted_fantasy": 20.13,
                    "predicted_pir": 18.64,
                    "availability": {
                        "status": "available",
                        "source": None,
                        "note": None,
                        "expected_return": None,
                        "updated_at": None,
                    },
                    "n_prior_appearances": 286,
                    "last_appearance_date": "2026-10-01",
                    "model_version": "20261003T140043Z-7ae47948",
                    "notes": [],
                }
            ]
        },
    )


# --------------------------------------------------------------------------------------
# Rankings
# --------------------------------------------------------------------------------------


class RankingsMeta(ApiModel):
    """Μεταδεδομένα της λίστας rankings."""

    as_of: date = Field(
        description="Η ημερομηνία (UTC) από την οποία αναζητήθηκαν οι επόμενοι αγώνες."
    )
    model_version: str = Field(description="Έκδοση του μοντέλου.")
    total: int = Field(
        description="Πόσοι παίκτες περνούν τα φίλτρα (πριν από το `limit`/`offset`)."
    )
    limit: int = Field(description="Μέγιστο πλήθος στοιχείων της σελίδας.")
    offset: int = Field(description="Πόσα στοιχεία παραλείφθηκαν από την αρχή της λίστας.")


class RankingItem(ApiModel):
    """Μία γραμμή των rankings."""

    rank: int = Field(
        description=(
            "Θέση στη φιλτραρισμένη λίστα (1 = καλύτερος). Συνεχόμενη, ανεξάρτητη από το `offset`."
        )
    )
    player_id: str
    name: str
    team_code: str
    predicted_fantasy: float = Field(
        description="Αποτελεσματική πρόβλεψη μετά το override διαθεσιμότητας (0 για `out`)."
    )
    model_predicted_fantasy: float = Field(description="Ακατέργαστη έξοδος του μοντέλου.")
    predicted_pir: float = Field(description="Πρόβλεψη PIR (ακατέργαστη τιμή του μοντέλου).")
    next_opponent_code: str | None = Field(description="Αντίπαλος στον επόμενο αγώνα, ή `null`.")
    next_home: bool | None = Field(
        description="`true` αν ο επόμενος αγώνας είναι εντός έδρας, ή `null`."
    )
    next_game_date: date | None = Field(description="Ημερομηνία του επόμενου αγώνα, ή `null`.")
    availability_status: AvailabilityStatus = Field(description="`out`, `doubtful` ή `available`.")
    is_active: bool


class RankingsOut(ApiModel):
    """Παίκτες ταξινομημένοι φθίνουσα κατά το αποτελεσματικό `predicted_fantasy`."""

    meta: RankingsMeta
    items: list[RankingItem]

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {
                    "meta": {
                        "as_of": "2026-10-03",
                        "model_version": "20261003T140043Z-7ae47948",
                        "total": 262,
                        "limit": 2,
                        "offset": 0,
                    },
                    "items": [
                        {
                            "rank": 1,
                            "player_id": "P009846",
                            "name": "BRYANT, ELIJAH",
                            "team_code": "HTA",
                            "predicted_fantasy": 20.9,
                            "model_predicted_fantasy": 20.9,
                            "predicted_pir": 20.04,
                            "next_opponent_code": "PAM",
                            "next_home": False,
                            "next_game_date": "2026-10-08",
                            "availability_status": "available",
                            "is_active": True,
                        },
                        {
                            "rank": 2,
                            "player_id": "P003469",
                            "name": "VEZENKOV, SASHA",
                            "team_code": "OLY",
                            "predicted_fantasy": 20.13,
                            "model_predicted_fantasy": 20.13,
                            "predicted_pir": 18.64,
                            "next_opponent_code": "IST",
                            "next_home": True,
                            "next_game_date": "2026-10-09",
                            "availability_status": "available",
                            "is_active": True,
                        },
                    ],
                }
            ]
        },
    )


# --------------------------------------------------------------------------------------
# Αναζήτηση παικτών
# --------------------------------------------------------------------------------------


class PlayerSummary(ApiModel):
    """Βασικά στοιχεία παίκτη για την εύρεση του `player_id`."""

    player_id: str
    name: str
    team_code: str = Field(description="Η ομάδα της τελευταίας γραμμής του παίκτη στη βάση.")
    team_name: str
    last_season: int = Field(description="Η τελευταία σεζόν (έτος έναρξης) στην οποία εμφανίζεται.")
    is_active: bool


class PlayersOut(ApiModel):
    """Αποτελέσματα αναζήτησης παικτών: πρώτα οι ενεργοί, μετά κατά αλφαβητική σειρά."""

    total: int = Field(description="Πόσοι παίκτες ταιριάζουν στα κριτήρια (πριν από το `limit`).")
    limit: int
    items: list[PlayerSummary]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "total": 1,
                    "limit": 20,
                    "items": [
                        {
                            "player_id": "P003469",
                            "name": "VEZENKOV, SASHA",
                            "team_code": "OLY",
                            "team_name": "OLYMPIACOS PIRAEUS",
                            "last_season": 2026,
                            "is_active": True,
                        }
                    ],
                }
            ]
        }
    )


# --------------------------------------------------------------------------------------
# Διαθεσιμότητα (αιτήματα και απαντήσεις)
# --------------------------------------------------------------------------------------


class AvailabilityIn(ApiModel):
    """Σώμα του `POST /availability`: αντικαθιστά ολόκληρη την εγγραφή του παίκτη."""

    player_id: str = Field(
        description=(
            "Κωδικός παίκτη (`P` + 6 ψηφία ή παλαιά μορφή `P` + 3 γράμματα). Κενά και πεζά "
            "γράμματα γίνονται δεκτά και κανονικοποιούνται."
        ),
        examples=["P007200"],
    )
    status: AvailabilityStatus = Field(description="`out`, `doubtful` ή `available`.")
    source: str | None = Field(
        default=None,
        max_length=SOURCE_MAX_LENGTH,
        description="Προέλευση της πληροφορίας, π.χ. `club statement` (έως 100 χαρακτήρες).",
    )
    note: str | None = Field(
        default=None,
        max_length=NOTE_MAX_LENGTH,
        description="Σημείωση (έως 500 χαρακτήρες).",
    )
    expected_return: date | None = Field(
        default=None,
        description="Αναμενόμενη επιστροφή, ημερομηνία `YYYY-MM-DD`.",
        examples=["2026-10-20"],
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "player_id": "P007200",
                    "status": "out",
                    "source": "club statement",
                    "note": "ankle sprain",
                    "expected_return": "2026-10-20",
                }
            ]
        },
    )

    @field_validator("player_id")
    @classmethod
    def _check_player_id(cls, value: str) -> str:
        value = normalise_player_id(value)
        if not PLAYER_ID_RE.fullmatch(value):
            raise ValueError(
                "player_id must match ^P[A-Z0-9]{3,6}$ after trimming and upper-casing"
            )
        return value

    @field_validator("source", "note", mode="before")
    @classmethod
    def _strip_text(cls, value):
        # Κενό ή μόνο κενά = δεν δόθηκε τιμή· το όριο μήκους ελέγχεται μετά το strip.
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("expected_return", mode="before")
    @classmethod
    def _iso_date_only(cls, value):
        # Μόνο ημερομηνία ISO (YYYY-MM-DD): όχι αριθμοί (timestamps) που η Pydantic δέχεται ως date.
        if value is None or isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value.strip())
            except ValueError:
                pass
        raise ValueError("expected_return must be an ISO date (YYYY-MM-DD)")


class AvailabilityEntry(ApiModel):
    """Εγγραφή διαθεσιμότητας μαζί με τον παίκτη στον οποίο αναφέρεται."""

    player_id: str
    name: str | None = Field(default=None, description="Όνομα του παίκτη.")
    status: AvailabilityStatus = Field(description="`out`, `doubtful` ή `available`.")
    source: str | None = Field(default=None, description="Προέλευση της πληροφορίας.")
    note: str | None = Field(default=None, description="Σημείωση του διαχειριστή.")
    expected_return: date | None = Field(default=None, description="Αναμενόμενη επιστροφή.")
    updated_at: datetime | None = Field(
        default=None, description="Πότε καταχωρήθηκε η εγγραφή (UTC, από τον server)."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "player_id": "P007200",
                    "name": "LARKIN, SHANE",
                    "status": "out",
                    "source": "club statement",
                    "note": "ankle sprain",
                    "expected_return": "2026-10-20",
                    "updated_at": "2026-10-03T12:00:00Z",
                }
            ]
        }
    )


class AvailabilityListOut(ApiModel):
    """Λίστα εγγραφών διαθεσιμότητας: πρώτα `out`, μετά `doubtful`, μετά `available`."""

    total: int
    items: list[AvailabilityEntry]


# --------------------------------------------------------------------------------------
# Health και διαχείριση
# --------------------------------------------------------------------------------------


class ModelInfo(ApiModel):
    """Στοιχεία του φορτωμένου μοντέλου (από το `metrics.json` της εκπαίδευσης)."""

    version: str = Field(description="Έκδοση του μοντέλου.")
    selected_model: str | None = Field(description="Όνομα της διαμόρφωσης που επιλέχθηκε.")
    test_mae: float | None = Field(
        description="Honest MAE του fantasy score στο test της εκπαίδευσης (σεζόν 2025)."
    )
    threshold: float | None = Field(description="Όριο MAE του quality gate.")
    trained_through_season: int | None = Field(
        description="Τελευταία σεζόν (έτος έναρξης) των δεδομένων εκπαίδευσης."
    )


class DatabaseInfo(ApiModel):
    """Κατάσταση της βάσης και των δεδομένων."""

    ok: bool = Field(description="`true` αν η βάση απαντά και περιέχει παίκτες.")
    players: int | None = Field(description="Πλήθος παικτών στη βάση.")
    latest_played_game_date: date | None = Field(
        description="Ημερομηνία του τελευταίου παιγμένου αγώνα (data cutoff)."
    )
    next_scheduled_game_date: date | None = Field(
        description="Ημερομηνία του πρώτου προγραμματισμένου αγώνα από σήμερα και μετά."
    )


class HealthResponse(ApiModel):
    """Κατάσταση της υπηρεσίας. HTTP 200 όταν είναι `ok`, HTTP 503 όταν είναι `degraded`."""

    status: Literal["ok", "degraded"]
    model: ModelInfo | None = Field(description="`null` αν το μοντέλο δεν φορτώθηκε.")
    database: DatabaseInfo
    data_age_days: int | None = Field(
        default=None,
        description="Ημέρες από τον τελευταίο παιγμένο αγώνα της βάσης μέχρι σήμερα (UTC).",
    )
    data_loaded_through: date | None = Field(
        default=None,
        description=(
            "Τελευταίος παιγμένος αγώνας που έχουν φορτώσει οι προβλέψεις. Αν διαφέρει από το "
            "`database.latest_played_game_date`, η βάση έχει νεότερα δεδομένα και χρειάζεται "
            "`POST /admin/refresh`."
        ),
    )
    commit: str | None = Field(
        default=None,
        description=(
            "Το commit (SHA) του κώδικα που τρέχει, όταν η πλατφόρμα το δηλώνει (στο Render, από "
            "τη μεταβλητή `RENDER_GIT_COMMIT`)· `null` τοπικά. Το CI το χρησιμοποιεί μετά από ένα "
            "deploy για να βεβαιωθεί ότι απαντά η νέα έκδοση και όχι η παλιά."
        ),
    )
    problems: list[str] = Field(
        description="Ποιοι έλεγχοι απέτυχαν (σύντομα μηνύματα, χωρίς εσωτερικές λεπτομέρειες)."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "model": {
                        "version": "20261003T140043Z-7ae47948",
                        "selected_model": "xgb_pseudohuber_09",
                        "test_mae": 5.908791,
                        "threshold": 6.0,
                        "trained_through_season": 2024,
                    },
                    "database": {
                        "ok": True,
                        "players": 1212,
                        "latest_played_game_date": "2026-10-02",
                        "next_scheduled_game_date": "2026-10-07",
                    },
                    "data_age_days": 1,
                    "data_loaded_through": "2026-10-02",
                    "commit": None,
                    "problems": [],
                }
            ]
        }
    )


class RefreshOut(ApiModel):
    """Αποτέλεσμα του `POST /admin/refresh`."""

    status: Literal["refreshed"]
    model_version: str
    players: int = Field(description="Πλήθος παικτών στη βάση μετά την ανανέωση.")
    previous_latest_played_game_date: date | None = Field(
        description="Data cutoff πριν από την ανανέωση (τελευταίος αγώνας που έβλεπε η υπηρεσία)."
    )
    latest_played_game_date: date | None = Field(
        description="Νέο data cutoff: ημερομηνία του τελευταίου παιγμένου αγώνα της βάσης."
    )
    next_scheduled_game_date: date | None = Field(
        description="Ημερομηνία του πρώτου προγραμματισμένου αγώνα από σήμερα και μετά."
    )
    refreshed_at: datetime = Field(description="Πότε ολοκληρώθηκε η ανανέωση (UTC).")
    duration_seconds: float


class ErrorOut(BaseModel):
    """Μορφή σφάλματος: σύντομο μήνυμα στο πεδίο `detail` (στα αγγλικά)."""

    detail: str

    model_config = ConfigDict(json_schema_extra={"examples": [{"detail": "player not found"}]})
