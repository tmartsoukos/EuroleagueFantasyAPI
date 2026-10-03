"""Άντληση δεδομένων Euroleague με ελεγχόμενο ρυθμό αιτημάτων, retry και cache ανά σεζόν.

Γιατί δικό μας loop και όχι οι μέθοδοι `*_single_season` / `*_round` του euroleague_api:
το πακέτο καλεί το boxscore ανά αγώνα χωρίς καθυστέρηση ή retry και **καταπίνει σιωπηλά** κάθε
λάθος (HTTP 429, κενό σώμα, timeouts), χάνοντας αγώνες χωρίς εξαίρεση ή μετρητή
(docs/DATA_SOURCES.md, ενότητες 5.6, 5.7 και 7). Εδώ:

- Χρησιμοποιούμε τις μεθόδους ανά αγώνα του πακέτου (`BoxScoreData.get_players_boxscore_stats`,
  `get_gamecodes_season`, `Schedule.get_schedule`), ώστε να έχουμε ακριβώς τις στήλες που
  τεκμηριώνονται.
  Το status και η επικεφαλίδα `Retry-After` διαβάζονται από το `HTTPError.response` που σηκώνει το
  πακέτο (`raise_for_status`), και το κενό σώμα HTTP 200 φαίνεται ως `JSONDecodeError`, οπότε δεν
  χρειάζεται δικό μας HTTP layer.
- Ένας καθολικός περιορισμός ρυθμού (`RateLimiter`) ελέγχει τις αρχές των αιτημάτων. Σε HTTP 429
  περιμένουμε `Retry-After` και μειώνουμε τον ρυθμό στο μισό. Σε 5xx, timeouts και σφάλματα σύνδεσης
  γίνεται retry με exponential backoff και πεπερασμένο πλήθος προσπαθειών. Κενό σώμα σημαίνει «δεν
  είναι διαθέσιμο» και ξαναδοκιμάζεται λίγες φορές.
- Το cache είναι ένα parquet ανά σεζόν και είδος στο `data/raw/`, που ενημερώνεται αυξητικά
  (checkpoint κάθε Ν αγώνες) και επιτρέπει συνέχιση μετά από διακοπή.
- Μετά από κάθε σεζόν συγκρίνεται το πλήθος αγώνων που ήρθαν με το πλήθος που αναμένεται και οι
  αγώνες που λείπουν καταγράφονται στο `data/raw/missing.json`.

Όλες οι κλήσεις δικτύου περνούν από ένα αντικείμενο `Source`, ώστε τα tests να τις αντικαθιστούν.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, TypeVar
from xml.parsers.expat import ExpatError

import pandas as pd
import requests

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_RPS = 1.0  # αιτήματα ανά δευτερόλεπτο (έναρξη αιτημάτων)
MAX_RPS = 1.5  # ανώτατο επιτρεπτό όριο: ο διακομιστής έδωσε 429 στα ~2,2 αιτήματα/s
MIN_RPS = 0.05  # η αυτόματη μείωση δεν πέφτει κάτω από 1 αίτημα ανά 20 s


# ----------------------------------------------------------------------------------------------
# Εξαιρέσεις
# ----------------------------------------------------------------------------------------------


class FetchError(Exception):
    """Αποτυχία άντλησης μετά από τις επιτρεπτές προσπάθειες."""


class DataUnavailableError(FetchError):
    """Ο διακομιστής απάντησε, αλλά χωρίς δεδομένα (κενό σώμα, 404 κ.λπ.)."""


class RateLimitedError(FetchError):
    """Συνεχόμενα HTTP 429: η εκτέλεση πρέπει να σταματήσει και να συνεχιστεί αργότερα."""


# ----------------------------------------------------------------------------------------------
# Περιορισμός ρυθμού και retry
# ----------------------------------------------------------------------------------------------


class RateLimiter:
    """Καθολικός περιορισμός ρυθμού: ελάχιστη απόσταση 1/rps ανάμεσα στις αρχές δύο αιτημάτων.

    Είναι thread-safe. Ο ρυθμός μπορεί μόνο να μειωθεί κατά τη διάρκεια μιας εκτέλεσης
    (`slow_down`), ώστε μετά από ένα HTTP 429 να μην ξαναδοκιμάσουμε τον ρυθμό που το προκάλεσε.
    Τα `clock` και `sleep` αντικαθίστανται στα tests με ψεύτικο ρολόι.
    """

    def __init__(
        self,
        rps: float = DEFAULT_RPS,
        *,
        max_rps: float = MAX_RPS,
        min_rps: float = MIN_RPS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not 0 < rps <= max_rps:
            raise ValueError(f"rps must be in (0, {max_rps}], got {rps}")
        self._rps = rps
        self._min_rps = min_rps
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_start = clock()
        self._last_start = clock()

    @property
    def rps(self) -> float:
        """Τρέχων ρυθμός (αιτήματα ανά δευτερόλεπτο)."""
        return self._rps

    def wait(self) -> float:
        """Μπλοκάρει ώσπου να επιτραπεί η έναρξη νέου αιτήματος. Επιστρέφει πόσο περίμενε (s)."""
        with self._lock:
            now = self._clock()
            start = max(now, self._next_start)
            self._last_start = start
            self._next_start = start + 1.0 / self._rps
        delay = start - now
        if delay > 0:
            self._sleep(delay)
        return max(delay, 0.0)

    def slow_down(self, factor: float = 0.5) -> float:
        """Μειώνει τον ρυθμό (προεπιλογή: στο μισό) και επιστρέφει τον νέο ρυθμό."""
        with self._lock:
            self._rps = max(self._min_rps, self._rps * factor)
            self._next_start = max(self._next_start, self._last_start + 1.0 / self._rps)
            return self._rps

    def block_for(self, seconds: float) -> None:
        """Αποκλείει νέα αιτήματα για τουλάχιστον `seconds` από τώρα (π.χ. μετά από HTTP 429)."""
        with self._lock:
            self._next_start = max(self._next_start, self._clock() + seconds)


@dataclass(frozen=True)
class RetryPolicy:
    """Παράμετροι των επαναλήψεων. Οι προεπιλογές είναι συντηρητικές για το Cloudflare του API."""

    max_attempts: int = 5  # συνολικές προσπάθειες για 5xx, timeouts και σφάλματα σύνδεσης
    backoff_base: float = 2.0  # αναμονή (s) πριν τη δεύτερη προσπάθεια
    backoff_factor: float = 2.0
    backoff_max: float = 60.0
    max_rate_limit_retries: int = 6  # διαδοχικά 429 πριν εγκαταλειφθεί η εκτέλεση
    rate_limit_margin: float = 5.0  # περιθώριο (s) που προστίθεται στο Retry-After
    default_retry_after: float = 60.0  # όταν το 429 δεν έχει (έγκυρο) Retry-After
    max_retry_after: float = 900.0  # ανώτατη αναμονή ανά 429 (s)
    empty_retries: int = 2  # ξαναδοκιμές όταν το σώμα είναι κενό (HTTP 200 χωρίς δεδομένα)
    empty_wait: float = 3.0  # αναμονή (s) ανάμεσα στις ξαναδοκιμές κενού σώματος


@dataclass
class RequestStats:
    """Μετρητές αιτημάτων μιας εκτέλεσης (για το log και την αναφορά)."""

    requests: int = 0
    rate_limited: int = 0  # πλήθος HTTP 429
    rate_limit_wait_seconds: float = 0.0
    server_errors: int = 0
    network_errors: int = 0
    empty_responses: int = 0


def parse_retry_after(value: str | None) -> float | None:
    """Διαβάζει την επικεφαλίδα `Retry-After` (δευτερόλεπτα ή ημερομηνία HTTP).

    Επιστρέφει None αν η τιμή λείπει ή δεν έχει έγκυρη μορφή.
    """
    if value is None:
        return None
    text = str(value).strip()
    try:
        return max(float(text), 0.0)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max((when - datetime.now(UTC)).total_seconds(), 0.0)


# Σφάλματα που σημαίνουν «ο διακομιστής απάντησε χωρίς έγκυρα δεδομένα» (κενό σώμα HTTP 200):
# JSON για το boxscore, XML (xmltodict/expat) για τα results και το schedule.
_EMPTY_BODY_ERRORS = (json.JSONDecodeError, requests.exceptions.JSONDecodeError, ExpatError)


class RequestRunner:
    """Εκτελεί κλήσεις δικτύου με τον περιορισμό ρυθμού και την πολιτική επαναλήψεων.

    - HTTP 429: αναμονή `Retry-After` συν περιθώριο, μείωση ρυθμού στο μισό, νέα προσπάθεια. Μετά
      από πολλά συνεχόμενα 429 σηκώνεται `RateLimitedError`.
    - HTTP 5xx, timeouts, σφάλματα σύνδεσης: exponential backoff, μέχρι `max_attempts`.
    - Κενό ή άκυρο σώμα (HTTP 200): λίγες ξαναδοκιμές και μετά `DataUnavailableError`.
    - Άλλο HTTP status (π.χ. 404): άμεσα `DataUnavailableError`.
    - Οποιοδήποτε άλλο σφάλμα (π.χ. απροσδόκητη μορφή απάντησης): `FetchError` χωρίς επανάληψη.
    """

    def __init__(
        self,
        limiter: RateLimiter,
        policy: RetryPolicy | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.limiter = limiter
        self.policy = policy or RetryPolicy()
        self.stats = RequestStats()
        self._sleep = sleep

    def call(self, func: Callable[[], T], *, label: str = "request") -> T:
        """Εκτελεί το `func()` (μία κλήση δικτύου) με retry. Το `label` μπαίνει στα μηνύματα."""
        policy = self.policy
        rate_limited = transient = empty = 0
        while True:
            self.limiter.wait()
            self.stats.requests += 1
            try:
                return func()
            except requests.HTTPError as exc:
                status = getattr(exc.response, "status_code", None)
                if status == 429:
                    rate_limited += 1
                    self._on_rate_limited(exc, rate_limited, label)
                elif status is not None and status >= 500:
                    transient += 1
                    self.stats.server_errors += 1
                    self._backoff(transient, label, f"HTTP {status}", exc)
                elif status is None:
                    transient += 1
                    self.stats.network_errors += 1
                    self._backoff(transient, label, "HTTP error without response", exc)
                else:
                    raise DataUnavailableError(f"{label}: HTTP {status}") from exc
            except _EMPTY_BODY_ERRORS as exc:
                empty += 1
                self.stats.empty_responses += 1
                if empty > policy.empty_retries:
                    raise DataUnavailableError(
                        f"{label}: empty or invalid response body ({empty} attempts)"
                    ) from exc
                logger.warning("%s: empty or invalid body, retrying (%d)", label, empty)
                self._sleep(policy.empty_wait)
            except requests.RequestException as exc:
                transient += 1
                self.stats.network_errors += 1
                self._backoff(transient, label, type(exc).__name__, exc)
            except FetchError:
                raise  # ήδη ταξινομημένο από την πηγή (π.χ. boxscore χωρίς στατιστικά)
            except Exception as exc:
                raise FetchError(f"{label}: unexpected response format ({exc!r})") from exc

    def _on_rate_limited(self, exc: requests.HTTPError, hits: int, label: str) -> None:
        policy = self.policy
        self.stats.rate_limited += 1
        if hits > policy.max_rate_limit_retries:
            raise RateLimitedError(f"{label}: still rate limited after {hits - 1} waits") from exc
        advised = parse_retry_after(exc.response.headers.get("Retry-After"))
        if advised is None:
            advised = policy.default_retry_after * 2 ** (hits - 1)
        wait = min(advised, policy.max_retry_after) + policy.rate_limit_margin
        new_rps = self.limiter.slow_down()
        self.limiter.block_for(wait)
        self.stats.rate_limit_wait_seconds += wait
        logger.warning(
            "%s: HTTP 429 (Retry-After %s): waiting %.0f s, rate lowered to %.2f req/s",
            label,
            exc.response.headers.get("Retry-After"),
            wait,
            new_rps,
        )

    def _backoff(self, attempt: int, label: str, reason: str, exc: Exception) -> None:
        policy = self.policy
        if attempt >= policy.max_attempts:
            raise FetchError(f"{label}: {reason} after {attempt} attempts") from exc
        delay = min(
            policy.backoff_base * policy.backoff_factor ** (attempt - 1), policy.backoff_max
        )
        logger.warning("%s: %s, retry %d in %.0f s", label, reason, attempt, delay)
        self._sleep(delay)


# ----------------------------------------------------------------------------------------------
# Πηγή δεδομένων (euroleague_api)
# ----------------------------------------------------------------------------------------------


class Source(Protocol):
    """Οι τρεις κλήσεις δικτύου του ingestion. Τα tests δίνουν δική τους υλοποίηση."""

    def results(self, season: int) -> pd.DataFrame:
        """Παιγμένοι αγώνες της σεζόν (`get_gamecodes_season`)."""

    def schedule(self, season: int) -> pd.DataFrame:
        """Πρόγραμμα της σεζόν, παιγμένοι και μελλοντικοί αγώνες (`Schedule.get_schedule`)."""

    def boxscore(self, season: int, gamecode: int) -> pd.DataFrame:
        """Boxscore ενός αγώνα (`BoxScoreData.get_players_boxscore_stats`)."""


@contextmanager
def preserved_root_logger() -> Iterator[None]:
    """Επαναφέρει τον root logger μετά το import του euroleague_api.

    Το πακέτο καλεί `logging.basicConfig(level=INFO)` στο import, που προσθέτει handler στον root
    logger. Αυτό δεν πρέπει να επηρεάζει άλλες διεργασίες (π.χ. το API της Φάσης 4).
    """
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    try:
        yield
    finally:
        for handler in list(root.handlers):
            if handler not in handlers:
                root.removeHandler(handler)
        root.setLevel(level)


def load_euroleague_api() -> SimpleNamespace:
    """Εισάγει το euroleague_api (μέσα σε συνάρτηση, όχι στο import της μονάδας).

    Επιστρέφει namespace με τις κλάσεις `BoxScoreData`, `EuroLeagueData`, `Schedule`. Ο root
    logger επαναφέρεται και οι μηνύματα ERROR του πακέτου σιωπούν, γιατί το ingestion καταγράφει
    ήδη κάθε αποτυχία με πιο σαφές μήνυμα (το πακέτο γράφει π.χ. «Didn't find gamecode» για 429).
    """
    with preserved_root_logger():
        from euroleague_api.boxscore_data import BoxScoreData
        from euroleague_api.EuroLeagueData import EuroLeagueData
        from euroleague_api.schedule import Schedule
    logging.getLogger("euroleague_api").setLevel(logging.CRITICAL)
    return SimpleNamespace(
        BoxScoreData=BoxScoreData, EuroLeagueData=EuroLeagueData, Schedule=Schedule
    )


class EuroleagueApiSource:
    """Προσαρμογέας πάνω στο πακέτο euroleague_api: ένα αίτημα ανά μέθοδο, χωρίς retry.

    Τα retry και ο περιορισμός ρυθμού γίνονται από το `RequestRunner`.
    """

    def __init__(self, competition: str = "E"):
        api = load_euroleague_api()
        self._data = api.EuroLeagueData(competition)
        self._schedule = api.Schedule(competition)
        self._boxscore = api.BoxScoreData(competition)

    def results(self, season: int) -> pd.DataFrame:
        return self._data.get_gamecodes_season(season)

    def schedule(self, season: int) -> pd.DataFrame:
        return self._schedule.get_schedule(season)

    def boxscore(self, season: int, gamecode: int) -> pd.DataFrame:
        try:
            return self._boxscore.get_players_boxscore_stats(season, gamecode)
        except NotImplementedError as exc:
            # Για ορισμένους αγώνες το API επιστρέφει «σκελετό» χωρίς στατιστικά (ομάδα «N/D»,
            # κενή λίστα παικτών, `tmr` null) και το `json_normalize` του πακέτου αποτυγχάνει.
            raise DataUnavailableError(
                f"boxscore {season}/{gamecode} has no player statistics "
                "(the API returned an empty skeleton)"
            ) from exc


# ----------------------------------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------------------------------


def replace_with_retry(
    source: Path,
    target: Path,
    *,
    attempts: int = 6,
    delay: float = 0.25,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Ατομική αντικατάσταση αρχείου (`os.replace`) με λίγες επαναλήψεις.

    Στα Windows ένα πρόγραμμα antivirus ή ο indexer μπορεί να κρατά για λίγο ανοιχτό ένα αρχείο
    που μόλις γράφτηκε, οπότε το `os.replace` σηκώνει `PermissionError` χωρίς να υπάρχει πρόβλημα.
    """
    for attempt in range(1, attempts + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            logger.warning("Could not replace %s (attempt %d), retrying", target.name, attempt)
            sleep(delay)


class RawCache:
    """Cache ακατέργαστων δεδομένων: ένα parquet ανά σεζόν και είδος στον φάκελο `root`.

    - `results_{season}.parquet`, `schedule_{season}.parquet`: ξαναγράφονται ολόκληρα.
    - `boxscores_{season}.parquet`: όλες οι γραμμές boxscore των αγώνων που έχουν έρθει, με αυξητική
      ενημέρωση (checkpoint). Η εγγραφή είναι ατομική (προσωρινό αρχείο και `os.replace`).
    - `missing.json`: ανά σεζόν, πόσοι αγώνες αναμένονταν, πόσοι ήρθαν και ποιοι λείπουν.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def path(self, kind: str, season: int) -> Path:
        return self.root / f"{kind}_{season}.parquet"

    @property
    def missing_path(self) -> Path:
        return self.root / "missing.json"

    def _read(self, path: Path) -> pd.DataFrame | None:
        return pd.read_parquet(path) if path.exists() else None

    def _write(self, path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        frame.to_parquet(temporary, index=False)
        replace_with_retry(temporary, path)

    def load_results(self, season: int) -> pd.DataFrame | None:
        return self._read(self.path("results", season))

    def save_results(self, season: int, frame: pd.DataFrame) -> None:
        self._write(self.path("results", season), frame)

    def load_schedule(self, season: int) -> pd.DataFrame | None:
        return self._read(self.path("schedule", season))

    def save_schedule(self, season: int, frame: pd.DataFrame) -> None:
        self._write(self.path("schedule", season), frame)

    def load_boxscores(self, season: int) -> pd.DataFrame | None:
        return self._read(self.path("boxscores", season))

    def boxscore_gamecodes(self, season: int) -> set[int]:
        """Οι κωδικοί αγώνων με boxscore στο cache."""
        frame = self.load_boxscores(season)
        if frame is None or frame.empty:
            return set()
        return {int(code) for code in frame["Gamecode"].unique()}

    def append_boxscores(self, season: int, frames: list[pd.DataFrame]) -> None:
        """Προσθέτει αγώνες στο cache της σεζόν (αντικαθιστά παλιές γραμμές των ίδιων αγώνων)."""
        if not frames:
            return
        new = pd.concat(frames, ignore_index=True)
        existing = self.load_boxscores(season)
        if existing is not None and not existing.empty:
            existing = existing[~existing["Gamecode"].isin(new["Gamecode"].unique())]
            new = pd.concat([existing, new], ignore_index=True)
        new = new.sort_values("Gamecode", kind="stable").reset_index(drop=True)
        self._write(self.path("boxscores", season), new)

    def load_missing(self) -> dict[str, dict]:
        if not self.missing_path.exists():
            return {}
        return json.loads(self.missing_path.read_text(encoding="utf-8"))

    def record_season(self, report: SeasonReport) -> None:
        """Ενημερώνει το `missing.json` με το αποτέλεσμα της σεζόν."""
        data = self.load_missing()
        data[str(report.season)] = {
            "expected": report.expected,
            "fetched": report.fetched,
            "missing": report.missing,
            "reasons": {str(code): reason for code, reason in report.reasons.items()},
            "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.missing_path.with_name("missing.json.tmp")
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        replace_with_retry(temporary, self.missing_path)


# ----------------------------------------------------------------------------------------------
# Άντληση σεζόν
# ----------------------------------------------------------------------------------------------


@dataclass
class SeasonReport:
    """Αποτέλεσμα της άντλησης μιας σεζόν."""

    season: int
    expected: int = 0  # αγώνες που αναμένονται (results και παιγμένοι του schedule)
    fetched: int = 0  # από αυτούς, όσοι έχουν boxscore στο cache
    new: int = 0  # αγώνες που ήρθαν σε αυτή την εκτέλεση
    missing: list[int] = field(default_factory=list)
    reasons: dict[int, str] = field(default_factory=dict)
    metadata_refreshed: bool = False
    error: str | None = None  # η σεζόν δεν ολοκληρώθηκε (π.χ. αποτυχία results και schedule)


def expected_gamecodes(results: pd.DataFrame | None, schedule: pd.DataFrame | None) -> list[int]:
    """Οι κωδικοί των αγώνων που έχουν παιχτεί: results με θετικό σκορ και schedule με played=true.

    Το `played` του `get_gamecodes_season` δεν χρησιμοποιείται (bug, DATA_SOURCES §5.5).
    """
    codes: set[int] = set()
    if results is not None and not results.empty:
        played = (pd.to_numeric(results["homescore"]) > 0) & (
            pd.to_numeric(results["awayscore"]) > 0
        )
        codes |= {int(code) for code in pd.to_numeric(results.loc[played, "gameCode"])}
    if schedule is not None and not schedule.empty:
        played = schedule["played"].astype(str).str.strip().str.lower() == "true"
        codes |= {int(code) for code in pd.to_numeric(schedule.loc[played, "game"])}
    return sorted(codes)


def season_in_progress(schedule: pd.DataFrame | None) -> bool:
    """True αν το schedule έχει αγώνες που δεν έχουν παιχτεί (σεζόν σε εξέλιξη ή ακυρώσεις)."""
    if schedule is None or schedule.empty:
        return True
    return bool((schedule["played"].astype(str).str.strip().str.lower() != "true").any())


def _refresh_metadata(
    season: int, source: Source, runner: RequestRunner, cache: RawCache
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Κατεβάζει results και schedule της σεζόν. Αν το ένα αποτύχει, συνεχίζουμε με το άλλο."""
    results = cache.load_results(season)
    schedule = cache.load_schedule(season)
    failures = []
    try:
        results = runner.call(lambda: source.results(season), label=f"results {season}")
        cache.save_results(season, results)
    except RateLimitedError:
        raise
    except FetchError as exc:
        failures.append(f"results: {exc}")
        logger.warning("Season %s: results not refreshed (%s)", season, exc)
    try:
        schedule = runner.call(lambda: source.schedule(season), label=f"schedule {season}")
        cache.save_schedule(season, schedule)
    except RateLimitedError:
        raise
    except FetchError as exc:
        failures.append(f"schedule: {exc}")
        logger.warning("Season %s: schedule not refreshed (%s)", season, exc)
    if results is None and schedule is None:
        raise FetchError(f"Season {season}: neither results nor schedule available ({failures})")
    return results, schedule


def fetch_season(
    season: int,
    source: Source,
    runner: RequestRunner,
    cache: RawCache,
    *,
    refresh_metadata: bool = False,
    checkpoint_every: int = 20,
    progress_every: int = 25,
    retry_passes: int = 1,
    retry_pause: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
) -> SeasonReport:
    """Κατεβάζει τα boxscores που λείπουν για μία σεζόν και ενημερώνει το cache.

    Το results και το schedule ανανεώνονται όταν ζητηθεί (`refresh_metadata`), όταν λείπουν από
    το cache ή όταν η σεζόν δεν έχει ολοκληρωθεί. Η συνέχιση μετά από διακοπή γίνεται αυτόματα: οι
    αγώνες που υπάρχουν ήδη στο cache δεν ζητούνται ξανά. Αγώνες που αποτυγχάνουν ξαναδοκιμάζονται
    στο τέλος (`retry_passes` φορές) και όσοι παραμένουν καταγράφονται στο `missing.json`.
    """
    results = cache.load_results(season)
    schedule = cache.load_schedule(season)
    refresh = (
        refresh_metadata or results is None or schedule is None or season_in_progress(schedule)
    )
    if refresh:
        results, schedule = _refresh_metadata(season, source, runner, cache)

    expected = expected_gamecodes(results, schedule)
    cached = cache.boxscore_gamecodes(season)
    todo = [code for code in expected if code not in cached]
    logger.info(
        "Season %s: %d games expected, %d cached, %d to fetch",
        season,
        len(expected),
        len(expected) - len(todo),
        len(todo),
    )

    reasons: dict[int, str] = {}
    buffer: list[pd.DataFrame] = []
    fetched_now = 0
    started = time.monotonic()

    def flush() -> None:
        nonlocal buffer
        if buffer:
            cache.append_boxscores(season, buffer)
            buffer = []

    def try_game(gamecode: int) -> bool:
        nonlocal fetched_now
        try:
            frame = runner.call(
                lambda: source.boxscore(season, gamecode), label=f"boxscore {season}/{gamecode}"
            )
        except FetchError as exc:
            if isinstance(exc, RateLimitedError):
                raise
            reasons[gamecode] = str(exc)
            logger.warning("Game %s/%s not fetched: %s", season, gamecode, exc)
            return False
        if frame is None or frame.empty:
            reasons[gamecode] = "empty boxscore table"
            logger.warning("Game %s/%s returned an empty table", season, gamecode)
            return False
        buffer.append(frame)
        reasons.pop(gamecode, None)
        fetched_now += 1
        return True

    try:
        for position, gamecode in enumerate(todo, start=1):
            try_game(gamecode)
            if len(buffer) >= checkpoint_every:
                flush()
            if position % progress_every == 0:
                elapsed = time.monotonic() - started
                remaining = elapsed / position * (len(todo) - position)
                logger.info(
                    "Season %s: %d/%d games done (%d new, %d failed), elapsed %.0f s, "
                    "ETA %.0f s, rate %.2f req/s, 429s so far: %d",
                    season,
                    position,
                    len(todo),
                    fetched_now,
                    len(reasons),
                    elapsed,
                    remaining,
                    runner.limiter.rps,
                    runner.stats.rate_limited,
                )
        for retry_pass in range(1, retry_passes + 1):
            if not reasons:
                break
            logger.info(
                "Season %s: retrying %d missing games (pass %d of %d)",
                season,
                len(reasons),
                retry_pass,
                retry_passes,
            )
            sleep(retry_pause)
            for gamecode in sorted(reasons):
                try_game(gamecode)
    finally:
        flush()

    cached = cache.boxscore_gamecodes(season)
    missing = [code for code in expected if code not in cached]
    report = SeasonReport(
        season=season,
        expected=len(expected),
        fetched=len(expected) - len(missing),
        new=fetched_now,
        missing=missing,
        reasons={code: reasons.get(code, "not fetched") for code in missing},
        metadata_refreshed=refresh,
    )
    cache.record_season(report)
    if missing:
        logger.error(
            "Season %s: %d of %d games are still missing: %s",
            season,
            len(missing),
            len(expected),
            missing,
        )
    else:
        logger.info(
            "Season %s: complete, %d of %d games in cache (%d new)",
            season,
            report.fetched,
            report.expected,
            fetched_now,
        )
    return report


def fetch_seasons(
    seasons: Iterable[int],
    source: Source,
    runner: RequestRunner,
    cache: RawCache,
    *,
    refresh_seasons: Iterable[int] = (),
    **options,
) -> list[SeasonReport]:
    """Κατεβάζει όλες τις σεζόν με τη σειρά. Μια σεζόν που αποτυγχάνει δεν σταματά τις επόμενες.

    Εξαίρεση: το `RateLimitedError` (συνεχόμενα 429) σταματά ολόκληρη την εκτέλεση. Το cache
    είναι ασφαλές και η εκτέλεση συνεχίζεται από εκεί που έμεινε με νέο τρέξιμο αργότερα.
    """
    refresh = set(refresh_seasons)
    reports = []
    for season in seasons:
        try:
            reports.append(
                fetch_season(
                    season, source, runner, cache, refresh_metadata=season in refresh, **options
                )
            )
        except RateLimitedError:
            logger.error("Stopping: the server keeps rate limiting us. Re-run later to resume.")
            raise
        except FetchError as exc:
            logger.error("Season %s failed: %s", season, exc)
            reports.append(SeasonReport(season=season, error=str(exc)))
    stats = runner.stats
    logger.info(
        "Fetch finished: %d requests, %d HTTP 429 (%.0f s waited), %d 5xx, %d network errors, "
        "%d empty bodies, final rate %.2f req/s",
        stats.requests,
        stats.rate_limited,
        stats.rate_limit_wait_seconds,
        stats.server_errors,
        stats.network_errors,
        stats.empty_responses,
        runner.limiter.rps,
    )
    return reports
