"""Καθαρισμός των ακατέργαστων δεδομένων του euroleague_api (καθαρές συναρτήσεις πάνω σε DataFrame).

Είσοδος: το ακατέργαστο boxscore (όλοι οι αγώνες μαζί, με τις στήλες του πακέτου) και, ανά
σεζόν, τα results (`get_gamecodes_season`) και το schedule (`Schedule.get_schedule`).
Έξοδος (`clean_all`): πίνακες `teams`, `players`, `games`, `player_games` με τα ονόματα στηλών της
βάσης, αναφορά ανωμαλιών ονομάτων και αναφορά ποιότητας.

Κανόνες (docs/DATA_SOURCES.md, ενότητες 5 και 12):
- Κλειδί παίκτη είναι το `Player_ID` μετά από strip, κλειδί ομάδας ο κωδικός. Τα ονόματα είναι
  το πιο πρόσφατο (κατά ημερομηνία) όνομα του ίδιου ID και δεν συγχωνεύονται ποτέ διαφορετικά IDs.
- Οι γραμμές `Team` και `Total` αφαιρούνται από τα στατιστικά παικτών, αλλά τα `Total.Points`
  κρατιούνται για έλεγχο του σκορ. Το `IsPlaying` αγνοείται (δεν είναι αξιόπιστο).
- Ημερομηνίες χωρίς εξάρτηση από locale (ρητός πίνακας μηνών), ώρες CET/CEST σε UTC, ταξινόμηση
  πάντα κατά ημερομηνία και ώρα (όχι κατά Round ή Gamecode: υπάρχουν μετατεθειμένοι αγώνες).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

# Οι ώρες των v1 endpoints (results, schedules) είναι σε ώρα Κεντρικής Ευρώπης (CET/CEST) και όχι
# σε τοπική ώρα γηπέδου. Επαληθεύτηκε σε 50 αγώνες του 2025-26 (Νοέμβριος έως Μάρτιος, με αλλαγή
# θερινής ώρας) συγκρίνοντας με το `utcDate` του v2, βλ. docs/INGESTION.md. Το Europe/Madrid έχει
# τους ίδιους κανόνες θερινής ώρας με το Europe/Berlin.
LOCAL_TIMEZONE = ZoneInfo("Europe/Madrid")

# Αγγλικά ονόματα μηνών: ρητός πίνακας αντί για strptime("%b"), που εξαρτάται από το locale του
# υπολογιστή (ελληνικά Windows).
MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Ανοχή (σε λεπτά) για το άθροισμα λεπτών ομάδας: 200 + 25 ανά παράταση. Αποκλίσεις γύρω στο ένα
# λεπτό (π.χ. «199:01», «201:00») υπάρχουν στην πηγή και δεν σημαίνουν ελλιπές boxscore.
TEAM_MINUTES_TOLERANCE = 1.5

TEAM_ROW_ID = "Team"
TOTAL_ROW_ID = "Total"

# Ακατέργαστες στήλες του boxscore -> στήλες του project.
RAW_TO_CLEAN = {
    "Season": "season",
    "Gamecode": "gamecode",
    "Home": "home_flag",
    "Player_ID": "player_id",
    "IsStarter": "is_starter_flag",
    "Team": "team_code",
    "Player": "player_name",
    "Minutes": "minutes_text",
    "Points": "points",
    "FieldGoalsMade2": "fg2_made",
    "FieldGoalsAttempted2": "fg2_attempted",
    "FieldGoalsMade3": "fg3_made",
    "FieldGoalsAttempted3": "fg3_attempted",
    "FreeThrowsMade": "ft_made",
    "FreeThrowsAttempted": "ft_attempted",
    "OffensiveRebounds": "off_reb",
    "DefensiveRebounds": "def_reb",
    "TotalRebounds": "total_reb",
    "Assistances": "assists",
    "Steals": "steals",
    "Turnovers": "turnovers",
    "BlocksFavour": "blocks_favour",
    "BlocksAgainst": "blocks_against",
    "FoulsCommited": "fouls_committed",
    "FoulsReceived": "fouls_received",
    "Valuation": "valuation",
    "Plusminus": "plus_minus",
}

# Ακέραια στατιστικά κάθε γραμμής παίκτη (όλα πρέπει να υπάρχουν, χωρίς NaN).
INTEGER_STAT_COLUMNS = [
    "points",
    "fg2_made",
    "fg2_attempted",
    "fg3_made",
    "fg3_attempted",
    "ft_made",
    "ft_attempted",
    "off_reb",
    "def_reb",
    "total_reb",
    "assists",
    "steals",
    "turnovers",
    "blocks_favour",
    "blocks_against",
    "fouls_committed",
    "fouls_received",
    "valuation",
]

# Οι 18 αριθμητικές στήλες για τον έλεγχο Total = Σ παικτών + Team (ακατέργαστα ονόματα).
RAW_TOTAL_COLUMNS = [
    "Points",
    "FieldGoalsMade2",
    "FieldGoalsAttempted2",
    "FieldGoalsMade3",
    "FieldGoalsAttempted3",
    "FreeThrowsMade",
    "FreeThrowsAttempted",
    "OffensiveRebounds",
    "DefensiveRebounds",
    "TotalRebounds",
    "Assistances",
    "Steals",
    "Turnovers",
    "BlocksFavour",
    "BlocksAgainst",
    "FoulsCommited",
    "FoulsReceived",
    "Valuation",
]

# Στήλες του πίνακα αγώνων (η σειρά ακολουθεί το db/models.py).
GAME_COLUMNS = [
    "season",
    "gamecode",
    "phase",
    "round",
    "game_date",
    "tipoff_utc",
    "home_code",
    "away_code",
    "home_score",
    "away_score",
    "played",
    "winner_code",
]

ISSUE_COLUMNS = ["check", "severity", "season", "gamecode", "key", "detail"]


class DataQualityError(ValueError):
    """Σηκώνεται όταν αποτυγχάνει έλεγχος ποιότητας που δεν επιτρέπεται να προχωρήσει."""

    def __init__(self, message: str, issues: pd.DataFrame | None = None):
        super().__init__(message)
        self.issues = issues if issues is not None else _empty_issues()


# ----------------------------------------------------------------------------------------------
# Απλοί parsers (καθαρές συναρτήσεις πάνω σε κείμενο)
# ----------------------------------------------------------------------------------------------

_DATE_PATTERN = re.compile(r"^\s*([A-Za-z]{3,9})\.?\s+(\d{1,2}),\s*(\d{4})\s*$")
_TIME_PATTERN = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")
_MINUTES_PATTERN = re.compile(r"^(\d{1,3}):([0-5]\d)$")


def parse_date(text: object) -> date:
    """Μετατρέπει ημερομηνία της μορφής «Sep 30, 2025» σε `date`, ανεξάρτητα από το locale.

    Δεν χρησιμοποιεί `strptime`, του οποίου το `%b` εξαρτάται από τη γλώσσα του λειτουργικού.
    Σηκώνει `ValueError` για κάθε άλλη μορφή.
    """
    match = _DATE_PATTERN.match(str(text))
    month = MONTHS.get(match.group(1)[:3].lower()) if match else None
    if match is None or month is None:
        raise ValueError(f"Invalid date: {text!r}")
    try:
        return date(int(match.group(3)), month, int(match.group(2)))
    except ValueError as exc:  # π.χ. 31 Σεπτεμβρίου
        raise ValueError(f"Invalid date: {text!r}") from exc


def tipoff_to_utc(day: date, time_text: object) -> datetime | None:
    """Μετατρέπει ημερομηνία και ώρα CET/CEST («20:45») σε ώρα UTC (naive datetime).

    Επιστρέφει `None` αν η ώρα λείπει ή δεν έχει τη μορφή «ΩΩ:ΛΛ».
    """
    if time_text is None or (not isinstance(time_text, str) and pd.isna(time_text)):
        return None
    match = _TIME_PATTERN.match(str(time_text))
    if match is None or int(match.group(1)) > 23 or int(match.group(2)) > 59:
        if str(time_text).strip():
            logger.warning("Unparseable start time %r on %s", time_text, day)
        return None
    local = datetime(
        day.year,
        day.month,
        day.day,
        int(match.group(1)),
        int(match.group(2)),
        tzinfo=LOCAL_TIMEZONE,
    )
    return local.astimezone(UTC).replace(tzinfo=None)


def parse_minutes(value: object) -> tuple[float, bool]:
    """Μετατρέπει το `Minutes` του boxscore σε (δεκαδικά λεπτά, dnp).

    - «MM:SS» (και «200:00» της γραμμής Total) -> (MM + SS/60, False)
    - «DNP» -> (0.0, True)
    - κενό κείμενο -> (0.0, False): παίκτης που καταγράφηκε στο boxscore χωρίς χρόνο συμμετοχής
      αλλά και χωρίς ένδειξη DNP (π.χ. 2022/313: τρεις παίκτες με από ένα φάουλ και «Minutes» κενό)
    Σηκώνει `ValueError` για οτιδήποτε άλλο (π.χ. NaN ή «07:75»).
    """
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0.0, False
        if text.upper() == "DNP":
            return 0.0, True
        match = _MINUTES_PATTERN.match(text)
        if match:
            return (int(match.group(1)) * 60 + int(match.group(2))) / 60, False
    raise ValueError(f"Invalid Minutes value: {value!r}")


def parse_minutes_column(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Vectorized `parse_minutes`: επιστρέφει (minutes float, dnp bool) με το index της στήλης."""
    # Τιμές που δεν είναι κείμενο (None, NaN) γίνονται NaN και θεωρούνται άκυρες (όχι κενό κείμενο).
    text = values.astype("object").map(lambda v: v.strip() if isinstance(v, str) else float("nan"))
    is_text = text.map(lambda v: isinstance(v, str))
    is_blank = is_text & text.map(lambda v: v == "")
    is_dnp = is_text & text.map(lambda v: isinstance(v, str) and v.upper() == "DNP")
    parts = text.map(lambda v: v if isinstance(v, str) else "").str.extract(_MINUTES_PATTERN)
    invalid = ~is_dnp & ~is_blank & parts[0].isna()
    if invalid.any():
        examples = ", ".join(sorted({repr(v) for v in values[invalid].head(10)}))
        raise DataQualityError(f"{int(invalid.sum())} rows with invalid Minutes, e.g. {examples}")
    seconds = parts[0].fillna("0").astype(int) * 60 + parts[1].fillna("0").astype(int)
    return (seconds / 60).astype(float), is_dnp.astype(bool)


def name_key(name: object) -> str:
    """Κανονικοποιημένο όνομα για σύγκριση: κεφαλαία, χωρίς τόνους, μονά κενά."""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.upper().split())


# ----------------------------------------------------------------------------------------------
# Αναφορά ποιότητας
# ----------------------------------------------------------------------------------------------


def _empty_issues() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in ISSUE_COLUMNS})


def _issues(check: str, severity: str, frame: pd.DataFrame, key: object, detail: object):
    """Γραμμές αναφοράς για τις γραμμές του `frame` (με στήλες season, gamecode).

    Τα `key` και `detail` είναι κείμενο (ίδιο για όλες τις γραμμές) ή λίστα/πίνακας ίδιου μήκους.
    """
    if frame.empty:
        return _empty_issues()
    return pd.DataFrame(
        {
            "check": check,
            "severity": severity,
            "season": frame["season"].to_numpy(),
            "gamecode": frame["gamecode"].to_numpy(),
            "key": key,
            "detail": detail,
        }
    )


def _combine_issues(parts: Iterable[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [part for part in parts if not part.empty]
    if not non_empty:
        return _empty_issues()
    return pd.concat(non_empty, ignore_index=True)[ISSUE_COLUMNS]


@dataclass
class QualityReport:
    """Συγκεντρωτική αναφορά ελέγχων ποιότητας (μία γραμμή ανά πρόβλημα)."""

    issues: pd.DataFrame

    @property
    def errors(self) -> pd.DataFrame:
        return self.issues[self.issues["severity"] == "error"]

    @property
    def warnings(self) -> pd.DataFrame:
        return self.issues[self.issues["severity"] == "warning"]

    def counts(self) -> dict[str, int]:
        """Πλήθος προβλημάτων ανά έλεγχο."""
        return {str(k): int(v) for k, v in self.issues["check"].value_counts().items()}

    def raise_if_errors(self, allow: Iterable[str] = ()) -> None:
        """Σηκώνει `DataQualityError` για προβλήματα σοβαρότητας error.

        Οι έλεγχοι που αναφέρονται στο `allow` δεν σταματούν το pipeline (μένουν στην αναφορά).
        """
        blocking = self.errors[~self.errors["check"].isin(set(allow))]
        if blocking.empty:
            return
        counts = blocking["check"].value_counts().to_dict()
        summary = ", ".join(f"{check}: {count}" for check, count in counts.items())
        raise DataQualityError(f"Data quality checks failed ({summary})", blocking)


# ----------------------------------------------------------------------------------------------
# Boxscore
# ----------------------------------------------------------------------------------------------


def split_boxscore(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Χωρίζει το ακατέργαστο boxscore σε (γραμμές παικτών, γραμμές Team, γραμμές Total).

    Το `Player_ID` έχει trailing spaces και καθαρίζεται με strip. Οι τρεις πίνακες κρατούν τα
    ακατέργαστα ονόματα στηλών.
    """
    df = raw.copy()
    df["Player_ID"] = df["Player_ID"].str.strip()
    is_team = df["Player_ID"] == TEAM_ROW_ID
    is_total = df["Player_ID"] == TOTAL_ROW_ID
    return df[~(is_team | is_total)].copy(), df[is_team].copy(), df[is_total].copy()


def clean_players(players_raw: pd.DataFrame) -> pd.DataFrame:
    """Καθαρίζει τις γραμμές παικτών: μετονομασία στηλών, λεπτά, DNP και αρχική πεντάδα.

    Δεν κάνει ελέγχους τιμών (αυτοί γίνονται στο `check_player_rows`). Το `IsPlaying` και τα
    υπόλοιπα άχρηστα πεδία (π.χ. `Dorsal`) δεν περνούν στην έξοδο.
    """
    required = [column for column in RAW_TO_CLEAN if column != "Plusminus"]
    missing = [column for column in required if column not in players_raw.columns]
    if missing:
        raise DataQualityError(f"Boxscore columns are missing: {missing}")
    df = players_raw.copy()
    if "Plusminus" not in df.columns:
        df["Plusminus"] = float("nan")
    df = df[list(RAW_TO_CLEAN)].rename(columns=RAW_TO_CLEAN)

    df["minutes"], df["dnp"] = parse_minutes_column(df["minutes_text"])
    df["is_starter"] = df["is_starter_flag"] == 1
    df["home_flag"] = df["home_flag"] == 1
    return df.drop(columns=["minutes_text", "is_starter_flag"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# Results και schedule -> αγώνες
# ----------------------------------------------------------------------------------------------

_GAME_FRAME_DTYPES = {
    "season": "int64",
    "gamecode": "int64",
    "phase": "object",
    "round": "Int64",
    "game_date": "object",
    "tipoff_utc": "datetime64[ns]",
    "home_code": "object",
    "away_code": "object",
    "home_name": "object",
    "away_name": "object",
}


def _empty_games(extra_dtypes: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Κενός πίνακας αγώνων με σωστούς τύπους (ώστε να δουλεύουν τα merge)."""
    dtypes = {**_GAME_FRAME_DTYPES, **(extra_dtypes or {})}
    return pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in dtypes.items()})


def _tipoffs(days: Iterable[date], times: Iterable[object]) -> pd.Series:
    values = [tipoff_to_utc(day, time_text) for day, time_text in zip(days, times, strict=True)]
    return pd.Series(pd.to_datetime(values), dtype="datetime64[ns]")


def clean_results(results: pd.DataFrame | None, season: int) -> pd.DataFrame:
    """Μετατρέπει το `get_gamecodes_season` σε πίνακα αγώνων με σκορ.

    Το `played` του πακέτου αγνοείται (bug, DATA_SOURCES §5.5): ως παιγμένος θεωρείται ο αγώνας
    με θετικό σκορ και στις δύο ομάδες (βλ. `build_games`).
    """
    if results is None or results.empty:
        return _empty_games({"home_score": "Int64", "away_score": "Int64"})
    days = [parse_date(text) for text in results["date"]]
    frame = pd.DataFrame(
        {
            "season": season,
            "gamecode": pd.to_numeric(results["gameCode"]).astype("int64"),
            "phase": results["Phase"].astype(str).str.strip(),
            "round": pd.to_numeric(results["Round"]).astype("Int64"),
            "game_date": pd.Series(days, dtype="object"),
            "tipoff_utc": _tipoffs(days, results["time"]),
            "home_code": results["homecode"].astype(str).str.strip(),
            "away_code": results["awaycode"].astype(str).str.strip(),
            "home_name": results["hometeam"].astype(str).str.strip(),
            "away_name": results["awayteam"].astype(str).str.strip(),
            "home_score": pd.to_numeric(results["homescore"]).astype("Int64"),
            "away_score": pd.to_numeric(results["awayscore"]).astype("Int64"),
        }
    )
    return frame.reset_index(drop=True)


def clean_schedule(schedule: pd.DataFrame | None, season: int) -> pd.DataFrame:
    """Μετατρέπει το `Schedule.get_schedule` σε πίνακα αγώνων (παιγμένοι και μελλοντικοί).

    Το `played` είναι κείμενο «true»/«false» (χωρίς το bug του `get_gamecodes_season`).
    """
    if schedule is None or schedule.empty:
        return _empty_games({"played": "bool"})
    days = [parse_date(text) for text in schedule["date"]]
    frame = pd.DataFrame(
        {
            "season": season,
            "gamecode": pd.to_numeric(schedule["game"]).astype("int64"),
            "phase": schedule["round"].astype(str).str.strip(),
            "round": pd.to_numeric(schedule["gameday"]).astype("Int64"),
            "game_date": pd.Series(days, dtype="object"),
            "tipoff_utc": _tipoffs(days, schedule["startime"]),
            "home_code": schedule["homecode"].astype(str).str.strip(),
            "away_code": schedule["awaycode"].astype(str).str.strip(),
            "home_name": schedule["hometeam"].astype(str).str.strip(),
            "away_name": schedule["awayteam"].astype(str).str.strip(),
            "played": schedule["played"].astype(str).str.strip().str.lower() == "true",
        }
    )
    return frame.reset_index(drop=True)


def build_games(results: pd.DataFrame, schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Συνδυάζει results και schedule σε έναν πίνακα αγώνων (ένας αγώνας ανά season, gamecode).

    - Παιγμένος είναι ο αγώνας με θετικό σκορ στα results ή `played == "true"` στο schedule.
    - Για τους παιγμένους αγώνες υπερισχύουν τα στοιχεία των results (ημερομηνία, ώρα, ομάδες),
      για τους μελλοντικούς το schedule.
    - Επιστρέφει και τις διαφορές results/schedule για τον ίδιο αγώνα (για την αναφορά ποιότητας).
    """
    keys = ["season", "gamecode"]
    if "played" not in schedule.columns:
        schedule = _empty_games({"played": "bool"})
    if "home_score" not in results.columns:
        results = _empty_games({"home_score": "Int64", "away_score": "Int64"})
    played_results = results[(results["home_score"] > 0) & (results["away_score"] > 0)]
    zero_score_rows = len(results) - len(played_results)
    if zero_score_rows:
        logger.warning("%d results rows have a zero score: treated as not played", zero_score_rows)

    merged = played_results.merge(
        schedule, on=keys, how="outer", suffixes=("_r", "_s"), indicator=True
    )
    out = merged[keys].copy()
    for column in ["phase", "round", "game_date", "tipoff_utc", "home_code", "away_code"]:
        from_results = merged[f"{column}_r"]
        out[column] = from_results.where(from_results.notna(), merged[f"{column}_s"])
    out["round"] = out["round"].astype("Int64")
    out["tipoff_utc"] = pd.to_datetime(out["tipoff_utc"])
    out["home_score"] = merged["home_score"].astype("Int64")
    out["away_score"] = merged["away_score"].astype("Int64")
    out["played"] = merged["played"].eq(True) | merged["_merge"].isin(["both", "left_only"])

    both = merged[merged["_merge"] == "both"]
    differs = pd.Series(False, index=both.index)
    for column in ["game_date", "home_code", "away_code"]:
        differs |= both[f"{column}_r"] != both[f"{column}_s"]
    both_have_tipoff = both["tipoff_utc_r"].notna() & both["tipoff_utc_s"].notna()
    differs |= both_have_tipoff & (both["tipoff_utc_r"] != both["tipoff_utc_s"])
    return out.reset_index(drop=True), both[differs][keys].reset_index(drop=True)


def build_teams(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Πίνακας ομάδων (team_code, name): canonical όνομα το πιο πρόσφατο ανά κωδικό."""
    parts = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for side in ("home", "away"):
            parts.append(
                pd.DataFrame(
                    {
                        "team_code": frame[f"{side}_code"].to_numpy(),
                        "name": frame[f"{side}_name"].to_numpy(),
                        "game_date": frame["game_date"].to_numpy(),
                        "gamecode": frame["gamecode"].to_numpy(),
                    }
                )
            )
    if not parts:
        return pd.DataFrame(
            {"team_code": pd.Series(dtype="object"), "name": pd.Series(dtype="object")}
        )
    names = pd.concat(parts, ignore_index=True)
    names = names.sort_values(["game_date", "gamecode"], kind="stable")
    latest = names.groupby("team_code", sort=True)["name"].last()
    return latest.rename("name").reset_index()


def finalize_games(games: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """Συμπληρώνει σκορ που λείπουν από τα `Total.Points`, υπολογίζει νικητή και ταξινομεί.

    `totals`: οι γραμμές `Total` του ακατέργαστου boxscore (στήλες Season, Gamecode, Team, Points).
    Η ταξινόμηση γίνεται κατά ημερομηνία και ώρα έναρξης, με τον κωδικό αγώνα μόνο ως ισοβαθμία.
    """
    games = games.copy()
    if not totals.empty:
        points = {
            (int(row.Season), int(row.Gamecode), row.Team): int(row.Points)
            for row in totals.itertuples(index=False)
        }
        needs_score = games[games["played"] & games["home_score"].isna()]
        for index, game in needs_score.iterrows():
            key = (int(game["season"]), int(game["gamecode"]))
            home = points.get((*key, game["home_code"]))
            away = points.get((*key, game["away_code"]))
            if home is not None and away is not None:
                games.loc[index, "home_score"] = home
                games.loc[index, "away_score"] = away
                logger.warning(
                    "Game %s/%s has no score in results: using boxscore totals %s-%s",
                    key[0],
                    key[1],
                    home,
                    away,
                )
    games["home_score"] = games["home_score"].astype("Int64")
    games["away_score"] = games["away_score"].astype("Int64")

    home_won = (games["home_score"] > games["away_score"]).fillna(False).to_numpy(dtype=bool)
    away_won = (games["away_score"] > games["home_score"]).fillna(False).to_numpy(dtype=bool)
    games["winner_code"] = pd.Series(
        [
            home if home_win else (away if away_win else None)
            for home, away, home_win, away_win in zip(
                games["home_code"], games["away_code"], home_won, away_won, strict=True
            )
        ],
        index=games.index,
        dtype="object",
    )
    games = games.sort_values(
        ["game_date", "tipoff_utc", "gamecode"], kind="stable", na_position="last"
    )
    return games[GAME_COLUMNS].reset_index(drop=True)


def attach_game_context(
    players: pd.DataFrame, games: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Προσθέτει σε κάθε γραμμή παίκτη ημερομηνία, ώρα, αντίπαλο, `home` και `won`.

    - `home` προέρχεται από το `Home` του boxscore και διασταυρώνεται με το `home_code` του αγώνα.
    - `won` προκύπτει από το σκορ των results (ή από τα `Total.Points` όπου αυτά λείπουν).
    Επιστρέφει (γραμμές παικτών, ευρήματα ελέγχου).
    """
    context_columns = [
        "season",
        "gamecode",
        "game_date",
        "tipoff_utc",
        "home_code",
        "away_code",
        "home_score",
        "away_score",
    ]
    df = players.merge(
        games[context_columns], on=["season", "gamecode"], how="left", validate="many_to_one"
    )
    ids = df["player_id"].to_numpy()
    problems = []

    missing_game = df["game_date"].isna()
    problems.append(
        _issues(
            "missing_game",
            "error",
            df[missing_game],
            ids[missing_game.to_numpy()],
            "no results/schedule row for this game",
        )
    )

    is_home_team = df["team_code"] == df["home_code"]
    is_away_team = df["team_code"] == df["away_code"]
    not_in_game = ~(is_home_team | is_away_team) & ~missing_game
    problems.append(
        _issues(
            "team_not_in_game",
            "error",
            df[not_in_game],
            ids[not_in_game.to_numpy()],
            "team code is neither home nor away team of the game",
        )
    )

    df["opp_code"] = df["away_code"].where(is_home_team, df["home_code"])
    df["home"] = df["home_flag"]
    flag_mismatch = (df["home_flag"] != is_home_team) & ~missing_game & ~not_in_game
    problems.append(
        _issues(
            "home_flag_mismatch",
            "warning",
            df[flag_mismatch],
            ids[flag_mismatch.to_numpy()],
            "Home flag of the boxscore disagrees with the game metadata",
        )
    )

    own_score = df["home_score"].where(is_home_team, df["away_score"])
    opp_score = df["away_score"].where(is_home_team, df["home_score"])
    df["won"] = (own_score > opp_score).fillna(False).astype(bool)
    return df, _combine_issues(problems)


# ----------------------------------------------------------------------------------------------
# Παίκτες: canonical ονόματα και ανωμαλίες
# ----------------------------------------------------------------------------------------------


def _chronological(players: pd.DataFrame) -> pd.DataFrame:
    """Ταξινόμηση κατά ημερομηνία και ώρα αγώνα (ο κωδικός αγώνα μόνο ως ισοβαθμία)."""
    return players.sort_values(
        ["game_date", "tipoff_utc", "gamecode"], kind="stable", na_position="last"
    )


def canonical_players(players: pd.DataFrame) -> pd.DataFrame:
    """Πίνακας παικτών (player_id, name, first_season, last_season).

    Το canonical όνομα είναι το πιο πρόσφατο όνομα του ID κατά ημερομηνία και ώρα αγώνα.
    Απαιτεί τις στήλες player_id, player_name, season, gamecode, game_date, tipoff_utc.
    """
    grouped = _chronological(players).groupby("player_id", sort=True)
    result = pd.DataFrame(
        {
            "name": grouped["player_name"].last(),
            "first_season": grouped["season"].min(),
            "last_season": grouped["season"].max(),
        }
    )
    return result.reset_index()


_ANOMALY_COLUMNS = [
    "kind",
    "key",
    "player_ids",
    "names",
    "n_rows",
    "first_game_date",
    "last_game_date",
    "canonical_name",
]


def name_anomalies(players: pd.DataFrame) -> pd.DataFrame:
    """Αναφορά ασυνεπειών ονομάτων και IDs. Δεν γίνεται καμία αυτόματη συγχώνευση.

    - `multiple_names_for_id`: ίδιο player_id με περισσότερες από μία γραφές ονόματος.
    - `same_name_multiple_ids`: ίδιο όνομα (χωρίς τόνους και διαφορές κεφαλαίων) με πολλά IDs.
    """
    ordered = _chronological(players)
    canonical = canonical_players(players).set_index("player_id")["name"]
    rows = []

    names_per_id = ordered.groupby("player_id")["player_name"].nunique()
    for player_id in names_per_id[names_per_id > 1].index:
        sub = ordered[ordered["player_id"] == player_id]
        rows.append(
            {
                "kind": "multiple_names_for_id",
                "key": player_id,
                "player_ids": player_id,
                "names": " | ".join(pd.unique(sub["player_name"])),
                "n_rows": len(sub),
                "first_game_date": sub["game_date"].iloc[0],
                "last_game_date": sub["game_date"].iloc[-1],
                "canonical_name": canonical[player_id],
            }
        )

    keyed = ordered.assign(name_key=ordered["player_name"].map(name_key))
    ids_per_name = keyed.groupby("name_key")["player_id"].nunique()
    for key in ids_per_name[ids_per_name > 1].index:
        sub = keyed[keyed["name_key"] == key]
        rows.append(
            {
                "kind": "same_name_multiple_ids",
                "key": key,
                "player_ids": " | ".join(sorted(sub["player_id"].unique())),
                "names": " | ".join(pd.unique(sub["player_name"])),
                "n_rows": len(sub),
                "first_game_date": sub["game_date"].iloc[0],
                "last_game_date": sub["game_date"].iloc[-1],
                "canonical_name": None,
            }
        )
    return pd.DataFrame(rows, columns=_ANOMALY_COLUMNS)


# ----------------------------------------------------------------------------------------------
# Έλεγχοι ποιότητας
# ----------------------------------------------------------------------------------------------


def check_player_rows(players: pd.DataFrame) -> pd.DataFrame:
    """Έλεγχοι ποιότητας στις γραμμές παικτών (μετά το `clean_players`).

    Σοβαρότητα error: διπλές (season, gamecode, player_id), NaN στα αριθμητικά, Points διαφορετικοί
    από 2·FGM2 + 3·FGM3 + FTM, TotalRebounds διαφορετικά από OR + DR. Σοβαρότητα warning:
    εύστοχα περισσότερα από τα συνολικά, αρνητικά στατιστικά, DNP με μη μηδενικά στατιστικά,
    μηδενικά λεπτά χωρίς ένδειξη DNP.
    """
    key_columns = ["season", "gamecode", "player_id"]
    parts = []

    ids = players["player_id"].to_numpy()
    duplicated = players.duplicated(subset=key_columns, keep=False)
    parts.append(
        _issues(
            "duplicate_key",
            "error",
            players[duplicated],
            ids[duplicated.to_numpy()],
            "duplicate (season, gamecode, player_id)",
        )
    )

    required = [*key_columns, "team_code", "player_name", *INTEGER_STAT_COLUMNS]
    nan_mask = players[required].isna()
    has_nan = nan_mask.any(axis=1)
    if has_nan.any():
        detail = [
            "NaN in " + ", ".join(nan_mask.columns[row])
            for row in nan_mask[has_nan].to_numpy(dtype=bool)
        ]
        parts.append(
            _issues("missing_values", "error", players[has_nan], ids[has_nan.to_numpy()], detail)
        )

    usable = players[~has_nan]
    if usable.empty:
        return _combine_issues(parts)
    usable_ids = usable["player_id"].to_numpy()

    expected_points = 2 * usable["fg2_made"] + 3 * usable["fg3_made"] + usable["ft_made"]
    bad = (usable["points"] != expected_points).to_numpy()
    parts.append(
        _issues(
            "points_formula",
            "error",
            usable[bad],
            usable_ids[bad],
            [
                f"points={p}, 2*FGM2+3*FGM3+FTM={e}"
                for p, e in zip(usable["points"][bad], expected_points[bad], strict=True)
            ],
        )
    )

    rebounds_sum = usable["off_reb"] + usable["def_reb"]
    bad = (usable["total_reb"] != rebounds_sum).to_numpy()
    parts.append(
        _issues(
            "rebounds_sum",
            "error",
            usable[bad],
            usable_ids[bad],
            [
                f"total_reb={t}, off_reb+def_reb={s}"
                for t, s in zip(usable["total_reb"][bad], rebounds_sum[bad], strict=True)
            ],
        )
    )

    over = (
        (usable["fg2_made"] > usable["fg2_attempted"])
        | (usable["fg3_made"] > usable["fg3_attempted"])
        | (usable["ft_made"] > usable["ft_attempted"])
    ).to_numpy()
    parts.append(
        _issues(
            "made_exceeds_attempted", "warning", usable[over], usable_ids[over], "made > attempted"
        )
    )

    counts = [column for column in INTEGER_STAT_COLUMNS if column != "valuation"]
    negative = (usable[counts] < 0).any(axis=1).to_numpy()
    parts.append(
        _issues(
            "negative_stat", "warning", usable[negative], usable_ids[negative], "negative count"
        )
    )

    dnp_with_stats = (usable["dnp"] & (usable[INTEGER_STAT_COLUMNS] != 0).any(axis=1)).to_numpy()
    parts.append(
        _issues(
            "dnp_with_stats",
            "warning",
            usable[dnp_with_stats],
            usable_ids[dnp_with_stats],
            "DNP row has non-zero statistics",
        )
    )

    zero_minutes = ((usable["minutes"] == 0) & ~usable["dnp"]).to_numpy()
    parts.append(
        _issues(
            "zero_minutes_not_dnp",
            "warning",
            usable[zero_minutes],
            usable_ids[zero_minutes],
            "0:00 minutes without a DNP mark",
        )
    )
    return _combine_issues(parts)


def check_team_totals(
    players_raw: pd.DataFrame, team_raw: pd.DataFrame, total_raw: pd.DataFrame
) -> pd.DataFrame:
    """Ελέγχει ότι ανά ομάδα και αγώνα ισχύει Total = Σ παικτών + Team (18 αριθμητικές στήλες)."""
    if total_raw.empty:
        return _empty_issues()
    keys = ["Season", "Gamecode", "Team"]
    columns = RAW_TOTAL_COLUMNS
    expected = (
        players_raw.groupby(keys)[columns]
        .sum()
        .add(team_raw.set_index(keys)[columns], fill_value=0)
    )
    difference = total_raw.set_index(keys)[columns].sub(expected, fill_value=0)
    bad = difference[(difference != 0).any(axis=1)]
    rows = []
    for (season, gamecode, team), row in bad.iterrows():
        detail = ", ".join(f"{column}: {value:+.0f}" for column, value in row.items() if value != 0)
        rows.append(
            {
                "check": "team_total_mismatch",
                "severity": "warning",
                "season": season,
                "gamecode": gamecode,
                "key": team,
                "detail": detail,
            }
        )
    return pd.DataFrame(rows, columns=ISSUE_COLUMNS) if rows else _empty_issues()


def check_team_minutes(total_raw: pd.DataFrame) -> pd.DataFrame:
    """Ελέγχει ότι τα λεπτά κάθε ομάδας (γραμμή Total) είναι 200 συν 25 ανά παράταση.

    Μεγάλη απόκλιση σημαίνει ελλιπές boxscore, δηλαδή παίκτης που λείπει από τη λίστα (π.χ.
    2017/14: 176:43 λεπτά και 66 πόντοι για την KHI αντί για 200:00 και 85).
    """
    if total_raw.empty:
        return _empty_issues()
    text = (
        total_raw["Minutes"].astype("object").map(lambda v: v.strip() if isinstance(v, str) else "")
    )
    parts = text.str.extract(_MINUTES_PATTERN)
    minutes = pd.to_numeric(parts[0]) + pd.to_numeric(parts[1]) / 60
    overtimes = ((minutes - 200) / 25).round().clip(lower=0)
    expected = 200 + 25 * overtimes
    bad = ((minutes - expected).abs() > TEAM_MINUTES_TOLERANCE).to_numpy(dtype=bool)
    frame = total_raw.rename(columns={"Season": "season", "Gamecode": "gamecode"})
    return _issues(
        "team_minutes_mismatch",
        "warning",
        frame[bad],
        frame["Team"].to_numpy()[bad],
        [
            f"Total minutes {t}, expected {e:.0f}:00"
            for t, e in zip(text[bad], expected[bad], strict=True)
        ],
    )


def check_total_scores(total_raw: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Συγκρίνει τα `Total.Points` του boxscore με το σκορ των results (ανά ομάδα και αγώνα)."""
    if total_raw.empty or games.empty:
        return _empty_issues()
    merged = total_raw.rename(columns={"Season": "season", "Gamecode": "gamecode"}).merge(
        games[["season", "gamecode", "home_code", "home_score", "away_score"]],
        on=["season", "gamecode"],
        how="inner",
    )
    expected = merged["home_score"].where(
        merged["Team"] == merged["home_code"], merged["away_score"]
    )
    bad = (expected.notna() & (merged["Points"] != expected)).to_numpy(dtype=bool)
    return _issues(
        "score_mismatch",
        "warning",
        merged[bad],
        merged["Team"].to_numpy()[bad],
        [
            f"Total.Points={p}, results score={e}"
            for p, e in zip(merged["Points"][bad], expected[bad], strict=True)
        ],
    )


# ----------------------------------------------------------------------------------------------
# Συνολική ροή
# ----------------------------------------------------------------------------------------------


@dataclass
class CleanData:
    """Αποτέλεσμα του `clean_all`: πίνακες έτοιμοι για υπολογισμό σκορ και εισαγωγή στη βάση."""

    teams: pd.DataFrame
    players: pd.DataFrame
    games: pd.DataFrame
    player_games: pd.DataFrame
    name_anomalies: pd.DataFrame
    quality: QualityReport


def clean_all(
    boxscores: pd.DataFrame,
    results_by_season: Mapping[int, pd.DataFrame | None],
    schedule_by_season: Mapping[int, pd.DataFrame | None],
) -> CleanData:
    """Καθαρίζει ολόκληρο το dataset: boxscores (όλοι οι αγώνες), results και schedule ανά σεζόν.

    Δεν σηκώνει εξαίρεση για τους ελέγχους ποιότητας: επιστρέφει την αναφορά και ο καλών αποφασίζει
    (`CleanData.quality.raise_if_errors()`). Εξαίρεση σηκώνεται μόνο όταν η είσοδος δεν είναι
    επεξεργάσιμη (π.χ. λείπουν στήλες ή το Minutes δεν έχει έγκυρη μορφή).
    """
    players_raw, team_raw, total_raw = split_boxscore(boxscores)
    players = clean_players(players_raw)

    game_frames, name_frames, conflict_frames = [], [], []
    for season in sorted(set(results_by_season) | set(schedule_by_season)):
        results = clean_results(results_by_season.get(season), season)
        schedule = clean_schedule(schedule_by_season.get(season), season)
        name_frames.extend([results, schedule])
        season_games, conflicts = build_games(results, schedule)
        if not season_games.empty:
            game_frames.append(season_games)
        if not conflicts.empty:
            conflict_frames.append(conflicts)
    if not game_frames:
        raise DataQualityError("There are no results or schedule rows for any season")
    games = finalize_games(pd.concat(game_frames, ignore_index=True), total_raw)
    teams = build_teams(name_frames)

    players, context_issues = attach_game_context(players, games)
    players = players.sort_values(
        ["game_date", "tipoff_utc", "gamecode", "team_code", "player_id"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    conflicts = (
        pd.concat(conflict_frames, ignore_index=True)
        if conflict_frames
        else pd.DataFrame({"season": [], "gamecode": []})
    )
    quality = QualityReport(
        _combine_issues(
            [
                check_player_rows(players),
                context_issues,
                check_team_totals(players_raw, team_raw, total_raw),
                check_team_minutes(total_raw),
                check_total_scores(total_raw, games),
                _issues(
                    "results_schedule_conflict",
                    "warning",
                    conflicts,
                    "",
                    "date, start time or teams differ between results and schedule",
                ),
            ]
        )
    )
    return CleanData(
        teams=teams,
        players=canonical_players(players),
        games=games,
        player_games=players,
        name_anomalies=name_anomalies(players),
        quality=quality,
    )
