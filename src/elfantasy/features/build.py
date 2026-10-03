"""Feature engineering χωρίς διαρροή (leakage) για την πρόβλεψη του fantasy score (Φάση 3).

Υπάρχει ΜΙΑ ΚΑΙ ΜΟΝΗ διαδρομή υπολογισμού, η `build_features(history, upcoming, games)`, που
εξυπηρετεί και την εκπαίδευση και την online πρόβλεψη:

* `history`: οι παιγμένοι αγώνες, μία γραμμή ανά (παίκτης, αγώνας), με τα στατιστικά τους
  (`elfantasy.features.load.load_history`).
* `upcoming`: γραμμές (παίκτης, αγώνας) για αγώνες που ΔΕΝ έχουν παιχτεί, χωρίς στατιστικά. Οι
  γραμμές αυτές προστίθενται στο ίδιο frame με τις ιστορικές (στατιστικά NaN) και περνούν από
  ακριβώς τον ίδιο κώδικα, άρα τα features μιας πρόβλεψης είναι ίδια με όσα θα είχε η γραμμή αυτή
  στην εκπαίδευση.
* `games` (προαιρετικό): όλοι οι παιγμένοι αγώνες του πίνακα `games`, ώστε η ισχύς των ομάδων
  (σκορ) να υπολογίζεται και για αγώνες χωρίς boxscore (π.χ. 2018/21). Αν λείπει, οι αγώνες
  προκύπτουν από τις γραμμές του `history`.

Κανόνας αποφυγής διαρροής: τα features μιας γραμμής (παίκτης, αγώνας) χρησιμοποιούν ΜΟΝΟ
πληροφορία από αγώνες με χρόνο έναρξης ΑΥΣΤΗΡΑ μικρότερο από τον δικό της, στην ίδια χρονική
ταξινόμηση (`tipoff_utc`, με δεύτερο κλειδί το `gamecode`, ποτέ το `Round`). Ποτέ δεν
χρησιμοποιούνται το σκορ, το `won` ή τα στατιστικά του ίδιου αγώνα. Αγώνες με ίδιο ακριβώς χρόνο
έναρξης στην ίδια ομάδα ή στον ίδιο παίκτη (δεν υπάρχουν στην πράξη) θεωρούνται ταυτόχρονοι και
δεν βλέπουν ο ένας τον άλλον.

Αρχή υλοποίησης («as-of» αναζήτηση): για κάθε ακολουθία (παίκτη ή ομάδας) υπολογίζονται οι
στατιστικές «μετά τον αγώνα» πάνω στους αγώνες-γεγονότα (συμμετοχές, παιγμένοι αγώνες) και κάθε
γραμμή παίρνει τη στατιστική του τελευταίου γεγονότος που έγινε αυστηρά πριν από αυτήν
(`_prior_event_index`). Έτσι το ίδιο πέρασμα δουλεύει για ιστορικές γραμμές, γραμμές DNP και
μελλοντικούς αγώνες. Τα κυλιόμενα παράθυρα υπολογίζονται με τοπικά αθροίσματα (όχι με
διαφορές καθολικών αθροιστικών), ώστε να μην υπάρχει αριθμητικός θόρυβος που εξαρτάται από το
πόσο ιστορικό περιέχει το frame.

Παίκτες χωρίς ιστορικό (πρώτος αγώνας) έχουν NaN στα features φόρμας και λεπτών. Τα μοντέλα τα
χειρίζονται όπως περιγράφεται στο docs/MODEL.md.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Στήλες εισόδου και εξόδου
# --------------------------------------------------------------------------------------

#: Στήλες του `history` (player_games ⨝ games, μόνο παιγμένοι αγώνες).
HISTORY_COLUMNS: tuple[str, ...] = (
    "season",
    "gamecode",
    "player_id",
    "team_code",
    "opp_code",
    "home",
    "game_date",
    "tipoff_utc",
    "is_starter",
    "minutes",
    "dnp",
    "pir",
    "fantasy_score",
    "team_score",
    "opp_score",
)

#: Στήλες του `upcoming` (αγώνες που δεν έχουν παιχτεί, χωρίς στατιστικά).
UPCOMING_COLUMNS: tuple[str, ...] = (
    "player_id",
    "season",
    "gamecode",
    "team_code",
    "opp_code",
    "home",
    "game_date",
    "tipoff_utc",
)

#: Στήλες του προαιρετικού `games` (παιγμένοι αγώνες με σκορ).
GAMES_COLUMNS: tuple[str, ...] = (
    "season",
    "gamecode",
    "game_date",
    "tipoff_utc",
    "home_code",
    "away_code",
    "home_score",
    "away_score",
)

#: Στήλες ταυτότητας της γραμμής.
ID_COLUMNS: list[str] = ["player_id", "season", "gamecode"]

#: Στήλες πλαισίου (meta). ΔΕΝ είναι features. Το `phase` υπάρχει μόνο αν δόθηκε στην είσοδο.
META_COLUMNS: list[str] = [
    "team_code",
    "opp_code",
    "game_date",
    "tipoff_utc",
    "phase",
    "is_upcoming",
    "is_appearance",
]

#: Στόχοι. Έχουν τιμή ΜΟΝΟ για συμμετοχές (dnp == false και λεπτά > 0), αλλιώς NaN.
TARGET_COLUMNS: list[str] = ["fantasy_score", "pir"]

#: Τα features, ανά ομάδα. Οι ορισμοί υπάρχουν στο docs/MODEL.md.
FEATURE_GROUPS: dict[str, list[str]] = {
    "form": [
        "fantasy_mean_3",
        "fantasy_mean_5",
        "fantasy_mean_10",
        "fantasy_mean_20",
        "fantasy_season_mean",
        "fantasy_ewm",
        "pir_mean_3",
        "pir_mean_5",
        "pir_mean_10",
        "pir_mean_20",
        "pir_season_mean",
        "pir_ewm",
        "pir_std_5",
        "pir_per_min_5",
    ],
    "minutes": [
        "min_mean_3",
        "min_mean_5",
        "min_season_mean",
        "min_last",
        "min_trend_5",
        "starter_rate_5",
    ],
    "availability": [
        "games_played_season",
        "games_played_total",
        "team_games_season",
        "dnp_streak",
        "days_since_last_appearance",
        "missed_last10",
    ],
    "context": [
        "home",
        "team_rest_days",
        "opp_rest_days",
        "team_short_rest",
        "team_games_last7",
        "opp_games_last7",
    ],
    "team_strength": [
        "team_win_pct_5",
        "team_pd_5",
        "team_win_pct_season",
        "team_pd_season",
        "opp_win_pct_5",
        "opp_pd_5",
        "opp_win_pct_season",
        "opp_pd_season",
    ],
    "opp_defense": [
        "opp_def_pir_5",
        "opp_def_pir_10",
    ],
}

#: Η σειρά των features που βλέπει το μοντέλο.
FEATURE_COLUMNS: list[str] = [name for group in FEATURE_GROUPS.values() for name in group]

# --------------------------------------------------------------------------------------
# Παράμετροι
# --------------------------------------------------------------------------------------

FORM_WINDOWS = (3, 5, 10, 20)  # κυλιόμενοι μέσοι φόρμας (πάνω στις τελευταίες συμμετοχές)
EWM_HALFLIFE = 3.0  # ημιζωή (σε συμμετοχές) του εκθετικά σταθμισμένου μέσου
EWM_LENGTH = 12  # πόσες τελευταίες συμμετοχές μετρούν στον εκθετικό μέσο (βάρος ουράς < 7%)
STD_WINDOW = 5  # παράθυρο της τυπικής απόκλισης (αστάθεια)
TREND_WINDOW = 5  # παράθυρο της κλίσης των λεπτών
TREND_MIN_POINTS = 3  # ελάχιστες συμμετοχές για να οριστεί κλίση
MINUTES_WINDOWS = (3, 5)
STARTER_WINDOW = 5
PER_MINUTE_MIN_MINUTES = 10.0  # ελάχιστα συνολικά λεπτά στο παράθυρο για να οριστεί PIR ανά λεπτό
TEAM_WINDOW = 5  # κυλιόμενο παράθυρο ισχύος ομάδας (αγώνες)
DEF_WINDOWS = (5, 10)  # κυλιόμενα παράθυρα του «PIR που δέχεται η ομάδα»
REST_CAP_DAYS = 14  # άνω όριο των ημερών ξεκούρασης (διακοπές, μεσοσεζόν)
SHORT_REST_DAYS = 2  # «σύντομη ξεκούραση»: ≤ 2 ημέρες από τον προηγούμενο αγώνα
RECENT_DAYS = 7  # παράθυρο «αγώνες ομάδας στις τελευταίες 7 ημέρες»
DAYS_SINCE_CAP = 365  # άνω όριο των ημερών από την τελευταία συμμετοχή
MISSED_WINDOW = 10  # «αγώνες που έχασε ο παίκτης» στους τελευταίους 10 αγώνες της ομάδας
LEAGUE_MIN_GAMES = 20  # ελάχιστοι αγώνες σεζόν για μέσο όρο λίγκας (αλλιώς η προηγούμενη σεζόν)
BOX_TOLERANCE_MINUTES = 1.5  # ανοχή στα συνολικά λεπτά ομάδας (200 + 25 ανά παράταση)

_ONE_DAY = pd.Timedelta(days=1)
_ONE_SECOND = pd.Timedelta(seconds=1)


# --------------------------------------------------------------------------------------
# Βοηθητικές συναρτήσεις numpy
# --------------------------------------------------------------------------------------


def _first_index_in_group(codes: np.ndarray) -> np.ndarray:
    """Για κάθε θέση, ο δείκτης της πρώτης θέσης της ομάδας της (οι ομάδες είναι συνεχόμενες)."""
    n = len(codes)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = codes[1:] != codes[:-1]
    return np.maximum.accumulate(np.where(starts, np.arange(n), 0))


def _position_in_group(codes: np.ndarray) -> np.ndarray:
    """Θέση (0, 1, 2, ...) μέσα στη συνεχόμενη ομάδα της."""
    return np.arange(len(codes)) - _first_index_in_group(codes)


def _shift(values: np.ndarray, steps: int) -> np.ndarray:
    """Μετατοπίζει τις τιμές `steps` θέσεις προς τα κάτω (float, NaN στις πρώτες θέσεις)."""
    out = np.full(len(values), np.nan)
    if steps == 0:
        out[:] = values
    elif steps < len(values):
        out[steps:] = values[:-steps]
    return out


def _prior_event_index(codes: np.ndarray, times: np.ndarray, is_event: np.ndarray) -> np.ndarray:
    """Για κάθε γραμμή, ο δείκτης (μέσα στον πίνακα γεγονότων) του τελευταίου γεγονότος που έγινε
    αυστηρά πριν από αυτήν στην ίδια ομάδα, ή -1 αν δεν υπάρχει.

    Οι γραμμές είναι ταξινομημένες κατά (ομάδα, χρόνος). «Πίνακας γεγονότων» είναι οι γραμμές με
    `is_event` στην ίδια σειρά. Γραμμές με ίδιο χρόνο στην ίδια ομάδα θεωρούνται ταυτόχρονες:
    το γεγονός μιας γραμμής δεν είναι «πριν» από άλλη γραμμή του ίδιου χρόνου.
    """
    n = len(codes)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    idx = np.arange(n)
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = codes[1:] != codes[:-1]
    new_block = new_group.copy()
    new_block[1:] |= times[1:] != times[:-1]
    group_first = np.maximum.accumulate(np.where(new_group, idx, 0))
    block_first = np.maximum.accumulate(np.where(new_block, idx, 0))
    events = is_event.astype(np.int64)
    before = np.cumsum(events) - events  # γεγονότα αυστηρά πριν από τη θέση, καθολικά
    in_group = before[block_first] - before[group_first]
    return np.where(in_group >= 1, before[block_first] - 1, -1)


def _gather(post: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Παίρνει από τον πίνακα `post` (ανά γεγονός) την τιμή του τελευταίου γεγονότος (NaN αν δεν
    υπάρχει)."""
    out = np.full(len(prior), np.nan)
    found = prior >= 0
    if found.any():
        out[found] = post[prior[found]]
    return out


def _window_sums(
    values: np.ndarray, position: np.ndarray, windows: Sequence[int]
) -> dict[int, np.ndarray]:
    """Αθροίσματα των τελευταίων `w` τιμών (η τρέχουσα περιλαμβάνεται) μέσα στην ίδια ομάδα.

    `position` είναι η θέση κάθε γραμμής μέσα στην ομάδα της. Το άθροισμα υπολογίζεται με
    μετατοπίσεις, δηλαδή με τοπικά αθροίσματα το πολύ `w` όρων (χωρίς καθολικά αθροιστικά).
    """
    result: dict[int, np.ndarray] = {}
    total = np.zeros(len(values))
    for step in range(max(windows)):
        shifted = _shift(values, step)
        total = total + np.where(position >= step, shifted, 0.0)
        if step + 1 in windows:
            result[step + 1] = total.copy()
    return result


def _window_means(
    values: np.ndarray, position: np.ndarray, windows: Sequence[int]
) -> dict[int, np.ndarray]:
    """Κυλιόμενοι μέσοι των τελευταίων `w` τιμών (όσες υπάρχουν, τουλάχιστον μία)."""
    sums = _window_sums(values, position, windows)
    return {w: sums[w] / np.minimum(position + 1, w) for w in windows}


def _window_std(values: np.ndarray, position: np.ndarray, window: int) -> np.ndarray:
    """Δειγματική τυπική απόκλιση (ddof = 1) των τελευταίων `window` τιμών, NaN αν είναι < 2."""
    mean = _window_means(values, position, [window])[window]
    squares = np.zeros(len(values))
    for step in range(window):
        deviation = _shift(values, step) - mean
        squares = squares + np.where(position >= step, deviation**2, 0.0)
    count = np.minimum(position + 1, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = squares / (count - 1)
    return np.where(count >= 2, np.sqrt(variance), np.nan)


def _window_slope(
    values: np.ndarray, position: np.ndarray, window: int, min_points: int
) -> np.ndarray:
    """Κλίση της ευθείας ελαχίστων τετραγώνων των τελευταίων `window` τιμών ως προς τη σειρά τους
    (θετική όταν οι πρόσφατες τιμές είναι μεγαλύτερες από τις παλαιότερες), NaN αν είναι λιγότερες
    από `min_points`.

    Με `x = -j` για τη j-οστή πιο πρόσφατη τιμή: `slope = Σ (x - x̄)·y / Σ (x - x̄)²`.
    """
    count = np.minimum(position + 1, window)
    sum_y = np.zeros(len(values))
    sum_xy = np.zeros(len(values))
    for step in range(window):
        valid = position >= step
        shifted = np.where(valid, _shift(values, step), 0.0)
        sum_y += shifted
        sum_xy += -step * shifted
    mean_x = -(count - 1) / 2.0
    numerator = sum_xy - mean_x * sum_y
    denominator = count * (count**2 - 1) / 12.0
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = numerator / denominator
    return np.where(count >= min_points, slope, np.nan)


def _ewm(values: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Εκθετικά σταθμισμένος μέσος των τελευταίων `EWM_LENGTH` τιμών (η τρέχουσα περιλαμβάνεται).

    Βάρος της j-οστής πιο πρόσφατης τιμής: `0,5 ** (j / EWM_HALFLIFE)`. Τα βάρη κανονικοποιούνται
    ώστε να αθροίζουν στη μονάδα για τις τιμές που υπάρχουν.
    """
    numerator = np.zeros(len(values))
    denominator = np.zeros(len(values))
    for step in range(EWM_LENGTH):
        weight = 0.5 ** (step / EWM_HALFLIFE)
        valid = position >= step
        numerator += np.where(valid, weight * _shift(values, step), 0.0)
        denominator += np.where(valid, weight, 0.0)
    return numerator / denominator


def _season_cumulative_mean(values: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """Αθροιστικός μέσος μέσα στην ομάδα (παίκτης, σεζόν): τιμή «μετά» την τρέχουσα γραμμή."""
    series = pd.Series(values)
    cumulative = series.groupby(group_ids).cumsum().to_numpy()
    count = series.groupby(group_ids).cumcount().to_numpy() + 1
    return cumulative / count


def _run_ids(*keys: np.ndarray) -> np.ndarray:
    """Αριθμεί τα συνεχόμενα τμήματα στα οποία το συνδυαστικό κλειδί `keys` μένει σταθερό."""
    n = len(keys[0])
    if n == 0:
        return np.empty(0, dtype=np.int64)
    change = np.zeros(n, dtype=bool)
    change[0] = True
    for key in keys:
        change[1:] |= key[1:] != key[:-1]
    return np.cumsum(change) - 1


# --------------------------------------------------------------------------------------
# Κανονικοποίηση και συναρμολόγηση εισόδων
# --------------------------------------------------------------------------------------


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _datetime_ns(series: pd.Series) -> pd.Series:
    """Μετατρέπει σε `datetime64[ns]` χωρίς ζώνη ώρας (τα tz-aware μετατρέπονται σε UTC)."""
    converted = pd.to_datetime(series)
    if getattr(converted.dt, "tz", None) is not None:
        converted = converted.dt.tz_convert("UTC").dt.tz_localize(None)
    return converted.astype("datetime64[ns]")


def _normalize_common(frame: pd.DataFrame) -> pd.DataFrame:
    """Κοινή κανονικοποίηση τύπων για history και upcoming."""
    out = pd.DataFrame(index=frame.index)
    out["player_id"] = frame["player_id"].astype(str).str.strip()
    out["season"] = frame["season"].astype(np.int64)
    out["gamecode"] = frame["gamecode"].astype(np.int64)
    out["team_code"] = frame["team_code"].astype(str).str.strip()
    opp = frame["opp_code"]
    out["opp_code"] = opp.where(opp.isna(), opp.astype(str).str.strip())
    out["home"] = pd.to_numeric(frame["home"], errors="coerce").astype(float)
    out["game_date"] = _datetime_ns(frame["game_date"]).dt.normalize()
    out["tipoff_utc"] = _datetime_ns(frame["tipoff_utc"])
    out["phase"] = frame["phase"] if "phase" in frame.columns else None
    return out.reset_index(drop=True)


def _normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    _require_columns(history, HISTORY_COLUMNS, "history")
    out = _normalize_common(history)
    source = history.reset_index(drop=True)
    out["is_starter"] = source["is_starter"].astype(bool)
    out["minutes"] = source["minutes"].astype(float)
    out["dnp"] = source["dnp"].astype(bool)
    out["pir"] = source["pir"].astype(float)
    out["fantasy_score"] = source["fantasy_score"].astype(float)
    out["team_score"] = source["team_score"].astype(float)
    out["opp_score"] = source["opp_score"].astype(float)
    if out.duplicated(ID_COLUMNS).any():
        raise ValueError("history contains duplicate (player_id, season, gamecode) rows")
    return out


def _normalize_upcoming(upcoming: pd.DataFrame | None) -> pd.DataFrame:
    if upcoming is None or len(upcoming) == 0:
        empty = pd.DataFrame(
            {
                "player_id": pd.Series(dtype=str),
                "season": pd.Series(dtype=np.int64),
                "gamecode": pd.Series(dtype=np.int64),
                "team_code": pd.Series(dtype=str),
                "opp_code": pd.Series(dtype=str),
                "home": pd.Series(dtype=float),
                "game_date": pd.Series(dtype="datetime64[ns]"),
                "tipoff_utc": pd.Series(dtype="datetime64[ns]"),
            }
        )
        return _normalize_common(empty)
    _require_columns(upcoming, UPCOMING_COLUMNS, "upcoming")
    out = _normalize_common(upcoming)
    if out.duplicated(ID_COLUMNS).any():
        raise ValueError("upcoming contains duplicate (player_id, season, gamecode) rows")
    return out


def _assemble_player_frame(history: pd.DataFrame, upcoming: pd.DataFrame) -> pd.DataFrame:
    """Ένα frame με ιστορικές και μελλοντικές γραμμές, ταξινομημένο κατά (παίκτης, χρόνος)."""
    keys = set(zip(history["player_id"], history["season"], history["gamecode"], strict=True))
    overlap = [
        key
        for key in zip(upcoming["player_id"], upcoming["season"], upcoming["gamecode"], strict=True)
        if key in keys
    ]
    if overlap:
        raise ValueError(f"upcoming rows overlap with history rows, e.g. {overlap[0]}")

    base = [*ID_COLUMNS, "team_code", "opp_code", "home", "game_date", "tipoff_utc", "phase"]
    hist = history[[*base, "is_starter", "minutes", "dnp", "pir", "fantasy_score"]].copy()
    hist["is_upcoming"] = False
    upc = upcoming[base].copy()
    upc["is_starter"] = False
    upc["minutes"] = np.nan
    upc["dnp"] = False
    upc["pir"] = np.nan
    upc["fantasy_score"] = np.nan
    upc["is_upcoming"] = True
    frame = pd.concat([hist, upc], ignore_index=True)

    # Χρόνος ταξινόμησης: η ώρα έναρξης (UTC). Αν λείπει, το τέλος της ημέρας του αγώνα (έτσι μια
    # γραμμή με άγνωστη ώρα δεν «χάνει» αγώνες της ίδιας ημέρας που έχουν ήδη παιχτεί).
    fallback = frame["game_date"] + _ONE_DAY - _ONE_SECOND
    frame["sort_time"] = frame["tipoff_utc"].fillna(fallback)
    frame = frame.sort_values(
        ["player_id", "sort_time", "season", "gamecode"], kind="stable"
    ).reset_index(drop=True)
    frame["is_appearance"] = (~frame["is_upcoming"]) & (~frame["dnp"]) & (frame["minutes"] > 0)
    frame["_is_row"] = ~frame["is_upcoming"]
    frame["_time"] = frame["sort_time"].to_numpy("datetime64[ns]").astype(np.int64)
    frame["_day"] = (
        frame["game_date"].to_numpy("datetime64[ns]").astype("datetime64[D]").astype(np.int64)
    )
    frame["_pcode"] = pd.factorize(frame["player_id"])[0]
    return frame


def _assemble_games(
    history: pd.DataFrame, upcoming: pd.DataFrame, games: pd.DataFrame | None
) -> pd.DataFrame:
    """Πίνακας αγώνων (ένας ανά (season, gamecode)) από `games`, `history` και `upcoming`.

    Προτεραιότητα στις διπλές εγγραφές: `games` (αν δόθηκε), μετά το `history`, μετά το
    `upcoming`. Οι μελλοντικοί αγώνες έχουν σκορ NaN.
    """
    parts = []
    if games is not None and len(games) > 0:
        _require_columns(games, GAMES_COLUMNS, "games")
        given = pd.DataFrame(
            {
                "season": games["season"].astype(np.int64),
                "gamecode": games["gamecode"].astype(np.int64),
                "game_date": _datetime_ns(games["game_date"]).dt.normalize(),
                "tipoff_utc": _datetime_ns(games["tipoff_utc"]),
                "home_code": games["home_code"].astype(str).str.strip(),
                "away_code": games["away_code"].astype(str).str.strip(),
                "home_score": games["home_score"].astype(float),
                "away_score": games["away_score"].astype(float),
            }
        )
        # Μόνο παιγμένοι αγώνες (με σκορ): το `games` είναι ιστορικό, όχι πρόγραμμα.
        parts.append(given[given["home_score"].notna() & given["away_score"].notna()])

    if len(history) > 0:
        first = history.drop_duplicates(["season", "gamecode"])
        is_home = first["home"] == 1.0
        parts.append(
            pd.DataFrame(
                {
                    "season": first["season"],
                    "gamecode": first["gamecode"],
                    "game_date": first["game_date"],
                    "tipoff_utc": first["tipoff_utc"],
                    "home_code": first["team_code"].where(is_home, first["opp_code"]),
                    "away_code": first["opp_code"].where(is_home, first["team_code"]),
                    "home_score": first["team_score"].where(is_home, first["opp_score"]),
                    "away_score": first["opp_score"].where(is_home, first["team_score"]),
                }
            )
        )

    known = upcoming[upcoming["opp_code"].notna() & upcoming["home"].notna()]
    if len(known) > 0:
        first = known.drop_duplicates(["season", "gamecode"])
        is_home = first["home"] == 1.0
        parts.append(
            pd.DataFrame(
                {
                    "season": first["season"],
                    "gamecode": first["gamecode"],
                    "game_date": first["game_date"],
                    "tipoff_utc": first["tipoff_utc"],
                    "home_code": first["team_code"].where(is_home, first["opp_code"]),
                    "away_code": first["opp_code"].where(is_home, first["team_code"]),
                    "home_score": np.nan,
                    "away_score": np.nan,
                }
            )
        )

    columns = list(GAMES_COLUMNS)
    if not parts:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})
    combined = pd.concat(parts, ignore_index=True)[columns]
    combined = combined.drop_duplicates(["season", "gamecode"], keep="first")
    return combined.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Χαρακτηριστικά ομάδων (ανά αγώνα ομάδας)
# --------------------------------------------------------------------------------------


def _box_summary(history: pd.DataFrame) -> pd.DataFrame:
    """PIR και πληρότητα boxscore ανά (αγώνας, ομάδα).

    Ένα boxscore θεωρείται πλήρες όταν τα συνολικά λεπτά των παικτών της ομάδας είναι
    200 + 25·(παρατάσεις) εντός ανοχής. Ελλιπή boxscores (docs/INGESTION.md, ενότητα 6)
    δίνουν NaN στο «PIR που δέχεται ο αντίπαλος», ώστε να μην μπαίνει μερικό άθροισμα.
    """
    if len(history) == 0:
        return pd.DataFrame(
            {
                "season": pd.Series(dtype=np.int64),
                "gamecode": pd.Series(dtype=np.int64),
                "team_code": pd.Series(dtype=str),
                "pir_sum": pd.Series(dtype=float),
                "box_ok": pd.Series(dtype=bool),
            }
        )
    box = (
        history.groupby(["season", "gamecode", "team_code"], sort=False)
        .agg(pir_sum=("pir", "sum"), minutes_sum=("minutes", "sum"))
        .reset_index()
    )
    overtimes = np.clip(np.rint((box["minutes_sum"] - 200.0) / 25.0), 0, None)
    expected = 200.0 + 25.0 * overtimes
    box["box_ok"] = (box["minutes_sum"] - expected).abs() <= BOX_TOLERANCE_MINUTES
    return box.drop(columns=["minutes_sum"])


def _league_mean_before(
    season: np.ndarray,
    time: np.ndarray,
    event_season: np.ndarray,
    event_time: np.ndarray,
    event_value: np.ndarray,
) -> np.ndarray:
    """Μέσος όρος των τιμών-γεγονότων της ίδιας σεζόν με χρόνο αυστηρά πριν από κάθε γραμμή.

    Για τις πρώτες `LEAGUE_MIN_GAMES` τιμές της σεζόν χρησιμοποιείται ο μέσος όρος ολόκληρης της
    προηγούμενης σεζόν (αν υπάρχει), που είναι ολόκληρος στο παρελθόν. Αν δεν υπάρχει ούτε αυτός,
    ο μερικός μέσος όρος της σεζόν, ή NaN αν δεν έχει γίνει ακόμη κανένα γεγονός.
    """
    out = np.full(len(season), np.nan)
    if len(event_value) == 0:
        return out
    event_seasons = np.unique(event_season)
    full_mean = {int(s): float(event_value[event_season == s].mean()) for s in event_seasons}
    for s in np.unique(season):
        rows = season == s
        in_season = event_season == s
        if in_season.any():
            order = np.argsort(event_time[in_season], kind="stable")
            times = event_time[in_season][order]
            cumulative = np.concatenate(([0.0], np.cumsum(event_value[in_season][order])))
            count = np.searchsorted(times, time[rows], side="left")
            partial = np.where(count > 0, cumulative[count] / np.maximum(count, 1), np.nan)
        else:
            count = np.zeros(int(rows.sum()), dtype=np.int64)
            partial = np.full(int(rows.sum()), np.nan)
        earlier = [int(p) for p in event_seasons if p < s]
        fallback = full_mean[earlier[-1]] if earlier else np.nan
        use_partial = (count >= LEAGUE_MIN_GAMES) | np.isnan(fallback)
        out[rows] = np.where(use_partial, partial, fallback)
    return out


def _team_game_features(games: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """Features ανά (αγώνας, ομάδα): ξεκούραση, φόρτος, ισχύς και αμυντική αξία των αντιπάλων.

    Κάθε αγώνας του `games` δίνει δύο γραμμές (γηπεδούχος και φιλοξενούμενος). Τα features της
    γραμμής (αγώνας g, ομάδα T) χρησιμοποιούν μόνο προηγούμενους αγώνες της T. Οι αγώνες χωρίς
    σκορ (μελλοντικοί) μετρούν στο πρόγραμμα (ξεκούραση, φόρτος) αλλά όχι στην ισχύ.
    """
    home = pd.DataFrame(
        {
            "season": games["season"],
            "gamecode": games["gamecode"],
            "game_date": games["game_date"],
            "tipoff_utc": games["tipoff_utc"],
            "team": games["home_code"],
            "opp": games["away_code"],
            "team_score": games["home_score"],
            "opp_score": games["away_score"],
        }
    )
    away = home.copy()
    away["team"] = games["away_code"]
    away["opp"] = games["home_code"]
    away["team_score"] = games["away_score"]
    away["opp_score"] = games["home_score"]
    tg = pd.concat([home, away], ignore_index=True)

    fallback = tg["game_date"] + _ONE_DAY - _ONE_SECOND
    tg["sort_time"] = tg["tipoff_utc"].fillna(fallback)
    tg = tg.sort_values(["team", "sort_time", "season", "gamecode"], kind="stable")
    tg = tg.reset_index(drop=True)

    # PIR που δέχεται η ομάδα = PIR του αντιπάλου στον ίδιο αγώνα (μόνο από πλήρες boxscore).
    opp_box = box.rename(
        columns={"team_code": "opp", "pir_sum": "opp_pir_sum", "box_ok": "opp_box_ok"}
    )
    tg = tg.merge(opp_box, on=["season", "gamecode", "opp"], how="left")
    opp_ok = tg["opp_box_ok"].eq(True).to_numpy()
    tg["pir_allowed"] = np.where(opp_ok, tg["opp_pir_sum"].to_numpy(float), np.nan)
    # Αγώνες στους οποίους η ομάδα δεν έχει καμία γραμμή παίκτη (ελλιπές ή μελλοντικό boxscore).
    own_box = box[["season", "gamecode", "team_code"]].rename(columns={"team_code": "team"})
    own_box = own_box.assign(has_rows=True)
    tg = tg.merge(own_box, on=["season", "gamecode", "team"], how="left")
    no_rows = tg["has_rows"].ne(True).to_numpy()

    n = len(tg)
    if n == 0:
        return tg.assign(
            tg_no=0,
            tg_row=0,
            no_rows_before=0,
            games_season=0,
            rest_days=np.nan,
            short_rest=np.nan,
            games_last7=0.0,
            win_pct_5=np.nan,
            pd_5=np.nan,
            win_pct_season=np.nan,
            pd_season=np.nan,
            def_pir_5=np.nan,
            def_pir_10=np.nan,
        )
    codes = pd.factorize(tg["team"])[0]
    time = tg["sort_time"].to_numpy("datetime64[ns]").astype(np.int64)
    day = tg["game_date"].to_numpy("datetime64[ns]").astype("datetime64[D]").astype(np.int64)
    season = tg["season"].to_numpy()
    position = _position_in_group(codes)

    # Πρόγραμμα: αριθμός αγώνα της ομάδας, αγώνες σεζόν, ξεκούραση, φόρτος 7 ημερών.
    tg["tg_no"] = position
    tg["tg_row"] = np.arange(n)
    no_rows_int = no_rows.astype(np.int64)
    tg["no_rows_before"] = pd.Series(no_rows_int).groupby(codes).cumsum().to_numpy() - no_rows_int
    tg["games_season"] = pd.Series(np.arange(n)).groupby([codes, season]).cumcount().to_numpy()
    previous_day = _shift(day.astype(float), 1)
    rest = np.where(position >= 1, day - previous_day, np.nan)
    tg["rest_days"] = np.minimum(rest, REST_CAP_DAYS)
    tg["short_rest"] = np.where(np.isnan(rest), np.nan, (rest <= SHORT_REST_DAYS).astype(float))
    recent = np.zeros(n)
    for step in range(1, RECENT_DAYS + 1):
        gap = day - _shift(day.astype(float), step)
        recent += (position >= step) & (gap < RECENT_DAYS)
    tg["games_last7"] = recent

    # Ισχύς ομάδας: ποσοστό νικών και διαφορά πόντων, μόνο από παιγμένους αγώνες (με σκορ).
    played = tg["team_score"].notna().to_numpy() & tg["opp_score"].notna().to_numpy()
    prior_played = _prior_event_index(codes, time, played)
    played_codes = codes[played]
    played_position = _position_in_group(played_codes)
    played_season = season[played]
    margin = (tg["team_score"] - tg["opp_score"]).to_numpy(float)[played]
    won = (margin > 0).astype(float)
    season_groups = _run_ids(played_codes, played_season)
    same_season = _gather(played_season.astype(float), prior_played) == season
    for name, values in (("win_pct", won), ("pd", margin)):
        rolling = _window_means(values, played_position, [TEAM_WINDOW])[TEAM_WINDOW]
        tg[f"{name}_5"] = _gather(rolling, prior_played)
        to_date = _gather(_season_cumulative_mean(values, season_groups), prior_played)
        tg[f"{name}_season"] = np.where(same_season, to_date, np.nan)

    # Αμυντική αξία: κυλιόμενος μέσος του PIR που δέχεται η ομάδα, σε σχέση με τον μέσο όρο της
    # λίγκας (της σεζόν μέχρι εκείνη τη στιγμή). Τιμή > 1 σημαίνει ότι η ομάδα δέχεται περισσότερο
    # PIR από τον μέσο όρο, δηλαδή αδύναμη άμυνα.
    allowed = tg["pir_allowed"].to_numpy()
    has_box = ~np.isnan(allowed)
    prior_box = _prior_event_index(codes, time, has_box)
    box_position = _position_in_group(codes[has_box])
    rolling_allowed = _window_means(allowed[has_box], box_position, DEF_WINDOWS)
    league = _league_mean_before(season, time, season[has_box], time[has_box], allowed[has_box])
    for window in DEF_WINDOWS:
        raw = _gather(rolling_allowed[window], prior_box)
        with np.errstate(divide="ignore", invalid="ignore"):
            tg[f"def_pir_{window}"] = np.where(league > 0, raw / league, np.nan)
    return tg


# --------------------------------------------------------------------------------------
# Χαρακτηριστικά παικτών
# --------------------------------------------------------------------------------------


def _player_features(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Features φόρμας, λεπτών και διαθεσιμότητας ανά γραμμή του `frame` (as-of αναζήτηση)."""
    pcodes = frame["_pcode"].to_numpy()
    time = frame["_time"].to_numpy()
    day = frame["_day"].to_numpy()
    row_season = frame["season"].to_numpy()
    is_app = frame["is_appearance"].to_numpy()
    is_row = frame["_is_row"].to_numpy()
    prior = _prior_event_index(pcodes, time, is_app)
    found = prior >= 0

    appearances = frame.loc[is_app]
    acodes = pcodes[is_app]
    apos = _position_in_group(acodes)
    aseason = appearances["season"].to_numpy()
    aday = appearances["_day"].to_numpy()
    pir = appearances["pir"].to_numpy(float)
    fantasy = appearances["fantasy_score"].to_numpy(float)
    minutes = appearances["minutes"].to_numpy(float)
    starter = appearances["is_starter"].to_numpy(float)

    out: dict[str, np.ndarray] = {}

    # Φόρμα: κυλιόμενοι μέσοι, εκθετικός μέσος, αστάθεια, PIR ανά λεπτό.
    for name, values in (("fantasy_mean", fantasy), ("pir_mean", pir)):
        means = _window_means(values, apos, FORM_WINDOWS)
        for window in FORM_WINDOWS:
            out[f"{name}_{window}"] = _gather(means[window], prior)
    out["fantasy_ewm"] = _gather(_ewm(fantasy, apos), prior)
    out["pir_ewm"] = _gather(_ewm(pir, apos), prior)
    out["pir_std_5"] = _gather(_window_std(pir, apos, STD_WINDOW), prior)
    pir_sum = _window_sums(pir, apos, [5])[5]
    minutes_sum = _window_sums(minutes, apos, [5])[5]
    with np.errstate(divide="ignore", invalid="ignore"):
        per_minute = np.where(minutes_sum >= PER_MINUTE_MIN_MINUTES, pir_sum / minutes_sum, np.nan)
    out["pir_per_min_5"] = _gather(per_minute, prior)

    # Λεπτά και ρόλος.
    minute_means = _window_means(minutes, apos, MINUTES_WINDOWS)
    for window in MINUTES_WINDOWS:
        out[f"min_mean_{window}"] = _gather(minute_means[window], prior)
    out["min_last"] = _gather(minutes, prior)
    out["min_trend_5"] = _gather(
        _window_slope(minutes, apos, TREND_WINDOW, TREND_MIN_POINTS), prior
    )
    out["starter_rate_5"] = _gather(
        _window_means(starter, apos, [STARTER_WINDOW])[STARTER_WINDOW], prior
    )

    # Μέσοι σεζόν μέχρι τώρα: μόνο συμμετοχές της ΙΔΙΑΣ σεζόν με τη γραμμή (αλλιώς NaN/0).
    season_groups = _run_ids(acodes, aseason)
    same_season = _gather(aseason.astype(float), prior) == row_season
    for name, values in (
        ("fantasy_season_mean", fantasy),
        ("pir_season_mean", pir),
        ("min_season_mean", minutes),
    ):
        to_date = _gather(_season_cumulative_mean(values, season_groups), prior)
        out[name] = np.where(same_season, to_date, np.nan)
    in_season_count = _gather(_position_in_group(season_groups) + 1.0, prior)
    out["games_played_season"] = np.where(same_season, in_season_count, 0.0)
    out["games_played_total"] = np.where(found, _gather(apos + 1.0, prior), 0.0)
    since = day - _gather(aday.astype(float), prior)
    out["days_since_last_appearance"] = np.minimum(since, DAYS_SINCE_CAP)

    # Συνεχόμενες γραμμές DNP ακριβώς πριν (πάνω στις γραμμές ρόστερ του παίκτη).
    rows_codes = pcodes[is_row]
    rows_app = is_app[is_row]
    new_player = np.ones(len(rows_codes), dtype=bool)
    new_player[1:] = rows_codes[1:] != rows_codes[:-1]
    segment = np.cumsum(rows_app | new_player)
    not_played = (~rows_app).astype(np.int64)
    streak_after = pd.Series(not_played).groupby(segment).cumsum().to_numpy()
    prior_row = _prior_event_index(pcodes, time, is_row)
    streak = _gather(streak_after.astype(float), prior_row)
    out["dnp_streak"] = np.where(np.isnan(streak), 0.0, streak)
    return out


def _missed_last_games(frame: pd.DataFrame, no_rows_before: np.ndarray) -> np.ndarray:
    """Πόσοι από τους τελευταίους `MISSED_WINDOW` αγώνες της ομάδας (πριν τον τρέχοντα) δεν
    είχαν συμμετοχή του παίκτη.

    Μετρούν μόνο οι αγώνες της ομάδας από την πρώτη γραμμή ρόστερ του παίκτη στην ομάδα και μετά,
    ώστε οι νεοφερμένοι να μην χρεώνονται αγώνες που έγιναν πριν έρθουν. Και οι γραμμές DNP και οι
    αγώνες χωρίς καμία γραμμή του παίκτη (π.χ. τραυματισμός) μετρούν ως «έχασε». Δεν μετρούν οι
    αγώνες στους οποίους ολόκληρη η ομάδα δεν έχει γραμμές παικτών (ελλιπές boxscore), γιατί η
    συμμετοχή δεν είναι γνωστή. Το `no_rows_before` είναι ο αθροιστικός αριθμός τέτοιων αγώνων
    ανά γραμμή του πίνακα αγώνων-ομάδων (`_team_game_features`).
    """
    tg_no = frame["_tg_no"].to_numpy(float)
    has = ~np.isnan(tg_no)
    out = np.full(len(frame), np.nan)
    if not has.any():
        return out
    rows = np.flatnonzero(has)
    player = frame["_pcode"].to_numpy()[rows].astype(np.int64)
    team = pd.factorize(frame["team_code"])[0][rows].astype(np.int64)
    game_no = tg_no[rows].astype(np.int64)
    appeared = frame["is_appearance"].to_numpy()[rows].astype(np.int64)
    group = player * (team.max() + 1) + team
    first = pd.Series(game_no).groupby(group).transform("min").to_numpy()
    scale = 10**7
    key = group * scale + game_no
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    cumulative = np.concatenate(([0], np.cumsum(appeared[order])))
    window_start = np.maximum(game_no - MISSED_WINDOW, first)
    low = np.searchsorted(sorted_key, group * scale + window_start, side="left")
    high = np.searchsorted(sorted_key, group * scale + game_no, side="left")
    # Οι γραμμές του πίνακα αγώνων-ομάδων μιας ομάδας είναι συνεχόμενες και χρονολογικές, άρα ο
    # αγώνας με αριθμό k έχει δείκτη (δείκτης τρέχοντος αγώνα) - (τρέχων αριθμός - k).
    current_row = frame["_tg_row"].to_numpy(float)[rows].astype(np.int64)
    start_row = current_row - (game_no - window_start)
    unknown = no_rows_before[current_row] - no_rows_before[start_row]
    eligible = (game_no - window_start) - unknown
    out[rows] = eligible - (cumulative[high] - cumulative[low])
    return out


# --------------------------------------------------------------------------------------
# Κύρια συνάρτηση
# --------------------------------------------------------------------------------------


def build_features(
    history: pd.DataFrame,
    upcoming: pd.DataFrame | None = None,
    games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Υπολογίζει τα features για κάθε γραμμή του `history` και του `upcoming`.

    Παράμετροι
    ----------
    history
        Παιγμένοι αγώνες, μία γραμμή ανά (παίκτης, αγώνας), στήλες `HISTORY_COLUMNS` (και
        προαιρετικά `phase`). Οι γραμμές DNP περιλαμβάνονται: μετρούν στα features
        διαθεσιμότητας αλλά όχι στα στατιστικά απόδοσης και δεν έχουν στόχο.
    upcoming
        Γραμμές για αγώνες που δεν έχουν παιχτεί, στήλες `UPCOMING_COLUMNS` (και προαιρετικά
        `phase`). Αν το `opp_code` ή το `home` λείπει (NaN/None), η γραμμή έχει «ουδέτερο
        πλαίσιο»: δεν υπάρχει αγώνας, άρα όλα τα features πλαισίου και ομάδων (`home`,
        ξεκούραση, ισχύς ομάδας και αντιπάλου, αμυντική αξία) είναι NaN, ενώ τα features του ίδιου
        του παίκτη υπολογίζονται κανονικά ως την ημερομηνία `game_date` της γραμμής (για ουδέτερη
        γραμμή, η ημερομηνία «ως την οποία» ισχύει το ιστορικό, π.χ. η σημερινή).
    games
        Προαιρετικά, όλοι οι παιγμένοι αγώνες (`GAMES_COLUMNS`), ώστε η ισχύς των ομάδων να
        υπολογίζεται και για αγώνες χωρίς γραμμές παικτών.

    Επιστρέφει
    ----------
    DataFrame με μία γραμμή ανά γραμμή εισόδου, ταξινομημένο κατά (χρόνο έναρξης, season,
    gamecode, player_id), και στήλες `ID_COLUMNS + META_COLUMNS + TARGET_COLUMNS +
    FEATURE_COLUMNS`. Οι στόχοι έχουν τιμή μόνο για συμμετοχές (`is_appearance`).

    Ο υπολογισμός δεν χρησιμοποιεί ποτέ τον τρέχοντα αγώνα ή μεταγενέστερους (βλ. την
    περιγραφή της μονάδας).
    """
    hist = _normalize_history(history)
    upc = _normalize_upcoming(upcoming)
    frame = _assemble_player_frame(hist, upc)
    game_table = _assemble_games(hist, upc, games)
    box = _box_summary(hist)
    team_table = _team_game_features(game_table, box)

    columns = _player_features(frame)

    # Features ομάδας και αντιπάλου, με αριστερή σύζευξη στο (σεζόν, αγώνας, ομάδα).
    own_columns = {
        "rest_days": "team_rest_days",
        "short_rest": "team_short_rest",
        "games_last7": "team_games_last7",
        "games_season": "team_games_season",
        "win_pct_5": "team_win_pct_5",
        "pd_5": "team_pd_5",
        "win_pct_season": "team_win_pct_season",
        "pd_season": "team_pd_season",
        "tg_no": "_tg_no",
        "tg_row": "_tg_row",
    }
    opp_columns = {
        "rest_days": "opp_rest_days",
        "games_last7": "opp_games_last7",
        "win_pct_5": "opp_win_pct_5",
        "pd_5": "opp_pd_5",
        "win_pct_season": "opp_win_pct_season",
        "pd_season": "opp_pd_season",
        "def_pir_5": "opp_def_pir_5",
        "def_pir_10": "opp_def_pir_10",
    }
    keys = ["season", "gamecode", "team_code"]
    own = team_table[["season", "gamecode", "team", *own_columns]].rename(
        columns={"team": "team_code", **own_columns}
    )
    opp = team_table[["season", "gamecode", "team", *opp_columns]].rename(
        columns={"team": "opp_code", **opp_columns}
    )
    n_rows = len(frame)
    frame = frame.merge(own, on=keys, how="left")
    frame = frame.merge(opp, on=["season", "gamecode", "opp_code"], how="left")
    assert len(frame) == n_rows, "team merge must not change the number of rows"

    # Ουδέτερο πλαίσιο: χωρίς αγώνα (άγνωστος αντίπαλος ή γήπεδο) τα features του αγώνα είναι NaN.
    has_game = (frame["opp_code"].notna() & frame["home"].notna()).to_numpy()
    for name in [*own_columns.values(), *opp_columns.values()]:
        frame[name] = frame[name].where(has_game)
    frame["missed_last10"] = _missed_last_games(frame, team_table["no_rows_before"].to_numpy())

    for name, values in columns.items():
        frame[name] = values

    # Στόχοι μόνο για συμμετοχές.
    appearance = frame["is_appearance"]
    frame["fantasy_score"] = frame["fantasy_score"].where(appearance)
    frame["pir"] = frame["pir"].where(appearance)

    frame = frame.sort_values(
        ["sort_time", "season", "gamecode", "player_id"], kind="stable"
    ).reset_index(drop=True)
    ordered = [*ID_COLUMNS, *META_COLUMNS, *TARGET_COLUMNS, *FEATURE_COLUMNS]
    result = frame[ordered].copy()
    result[FEATURE_COLUMNS] = result[FEATURE_COLUMNS].astype(np.float64)
    return result
