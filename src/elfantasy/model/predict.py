"""Πρόβλεψη fantasy score για τον επόμενο αγώνα κάθε παίκτη (API για τη Φάση 4).

Βασική χρήση::

    predictor = Predictor.load()           # μοντέλο από MODEL_PATH, βάση από DATABASE_URL
    predictor.predict_player("P007200")    # PlayerPrediction, ή None αν ο παίκτης είναι άγνωστος
    predictor.predict_all(team_code="IST")  # PlayerPrediction ανά παίκτη, φθίνουσα κατά fantasy
    predictor.refresh()                    # ξαναδιαβάζει τη βάση (μετά από νέο ingestion)

Σχεδιασμός
----------
* Η πρόβλεψη χρησιμοποιεί ΤΗΝ ΙΔΙΑ διαδρομή features με την εκπαίδευση: καλείται το
  `build_features(history, upcoming, games)` με το ιστορικό της βάσης και μία γραμμή `upcoming` ανά
  παίκτη (ο επόμενος αγώνας της ομάδας του). Το πέρασμα είναι vectorized για ΟΛΟΥΣ τους παίκτες
  (~1 s για ολόκληρο το ιστορικό του project) και το αποτέλεσμα κρατιέται σε cache ανά ημερομηνία
  `as_of`: οι επόμενες κλήσεις `predict_player`/`predict_all` για την ίδια ημερομηνία είναι O(1)/
  O(παίκτες) χωρίς νέο υπολογισμό. Το `refresh()` αδειάζει την cache.
* Το `predict_player` και το αντίστοιχο στοιχείο του `predict_all` είναι πάντα ίδια, γιατί
  προέρχονται από το ίδιο αποτέλεσμα.
* Το override διαθεσιμότητας (τραυματισμοί) ΔΕΝ ανήκει εδώ: εφαρμόζεται από το API μετά την
  πρόβλεψη. Το μοντέλο αγνοεί τραυματισμούς, απουσίες και αλλαγές ρόστερ που δεν φαίνονται στο
  ιστορικό.

Ορισμοί
-------
**Ομάδα παίκτη:** η ομάδα της τελευταίας γραμμής του στη βάση (συμπεριλαμβανομένων των γραμμών
DNP).

**Επόμενος αγώνας:** ο πρώτος αγώνας του πίνακα `games` με `played = false` και `game_date ≥
as_of` (προεπιλογή: σήμερα, UTC) όπου η ομάδα του είναι γηπεδούχος ή φιλοξενούμενος, με σειρά
ώρας έναρξης (`tipoff_utc`, αλλιώς ημερομηνία και `gamecode`). Αγώνες του παρελθόντος που δεν
διεξήχθησαν ποτέ (π.χ. ακυρώσεις) έχουν `played = false` αλλά ημερομηνία πριν από το `as_of`,
οπότε δεν μετρούν.

**Χωρίς προγραμματισμένο αγώνα** (π.χ. offseason, ή η ομάδα δεν έχει άλλους αγώνες στο
πρόγραμμα): `next_game = None` και η πρόβλεψη γίνεται με «ουδέτερο πλαίσιο»: τα features
φόρμας, λεπτών και διαθεσιμότητας υπολογίζονται κανονικά μέχρι το `as_of`, ενώ τα features
πλαισίου και ομάδων (γηπεδούχος, ξεκούραση, ισχύς ομάδας και αντιπάλου, αμυντική αξία) είναι
NaN και το μοντέλο τα χειρίζεται όπως χειρίζεται κάθε άγνωστη τιμή.

**Ενεργός παίκτης:** (α) έχει γραμμή (παίκτη ή DNP) στη νεότερη σεζόν που έχει παιγμένους
αγώνες με ημερομηνία ≤ `as_of`, ή (β) έχει γραμμή μέσα στις τελευταίες
`active_window_days` ημέρες πριν από το `as_of` (προεπιλογή 45). Αν το σύνολο προκύψει κενό
(π.χ. `as_of` πριν από κάθε αγώνα της βάσης), χρησιμοποιείται η νεότερη σεζόν της βάσης και, αν
και αυτή είναι κενή, όλοι οι παίκτες: το `predict_all(active_only=True)` δεν επιστρέφει ποτέ κενή
λίστα όσο υπάρχουν παίκτες στη βάση. Στο offseason «ενεργοί» είναι οι παίκτες της τελευταίας
σεζόν που παίχτηκε.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from elfantasy.config import get_settings
from elfantasy.db.session import get_engine
from elfantasy.features.build import build_features
from elfantasy.features.load import load_history, load_played_games, load_players, load_schedule
from elfantasy.model.artifact import ModelBundle, ModelLoadError, load_bundle

DEFAULT_ACTIVE_WINDOW_DAYS = 45
_CACHE_SIZE = 4  # πόσες ημερομηνίες as_of κρατιούνται στην cache
_NEUTRAL_GAMECODE = -1  # τεχνικός κωδικός αγώνα για γραμμές χωρίς προγραμματισμένο αγώνα


def utc_today() -> date:
    """Η σημερινή ημερομηνία σε UTC (προεπιλογή του `as_of`)."""
    return datetime.now(UTC).date()


@dataclass(frozen=True)
class NextGame:
    """Ο επόμενος αγώνας μιας ομάδας.

    `tipoff_utc` είναι naive datetime σε UTC (όπως στη βάση) ή None αν η ώρα δεν είναι γνωστή.
    `team_code` είναι η ομάδα του παίκτη και `opp_code` ο αντίπαλος· `home` είναι True όταν η
    ομάδα του παίκτη είναι γηπεδούχος.
    """

    season: int
    gamecode: int
    game_date: date
    tipoff_utc: datetime | None
    team_code: str
    opp_code: str
    home: bool


@dataclass(frozen=True)
class PlayerPrediction:
    """Η πρόβλεψη για έναν παίκτη.

    * `predicted_fantasy`: προβλεπόμενο fantasy score (PIR + |PIR|/10 σε νίκη), ακατέργαστο,
      χωρίς captain/πάγκο, υπό την προϋπόθεση ότι ο παίκτης αγωνίζεται.
    * `predicted_pir`: προβλεπόμενο PIR.
    * `n_prior_appearances`: πόσες συμμετοχές (με λεπτά > 0) έχει στο ιστορικό πριν τον αγώνα·
      όσο μικρότερο, τόσο λιγότερο αξιόπιστη η πρόβλεψη (0 = χωρίς ιστορικό).
    * `last_appearance_date`: ημερομηνία της τελευταίας συμμετοχής ή None.
    * `features`: οι τιμές των features του μοντέλου (None όπου λείπουν).
    * `is_active`: αν ο παίκτης θεωρείται ενεργός (βλ. τον ορισμό στη μονάδα).
    """

    player_id: str
    name: str
    team_code: str
    next_game: NextGame | None
    predicted_fantasy: float
    predicted_pir: float
    n_prior_appearances: int
    last_appearance_date: date | None
    features: dict[str, float | None]
    model_version: str
    is_active: bool = True


def _to_date(value: date | datetime | None) -> date:
    if value is None:
        return utc_today()
    if isinstance(value, datetime):
        return value.date()
    return value


class Predictor:
    """Φορτώνει το μοντέλο και τη βάση και προβλέπει τον επόμενο αγώνα των παικτών."""

    def __init__(
        self,
        bundle: ModelBundle,
        engine: Engine,
        *,
        active_window_days: int = DEFAULT_ACTIVE_WINDOW_DAYS,
        today: Callable[[], date] = utc_today,
        metrics: dict | None = None,
        owns_engine: bool = False,
    ):
        self._bundle = bundle
        self._engine = engine
        self._owns_engine = owns_engine
        self._active_window_days = active_window_days
        self._today = today
        self._metrics = metrics if metrics is not None else dict(bundle.metrics)
        self._lock = threading.RLock()
        self._cache: OrderedDict[date, dict[str, PlayerPrediction]] = OrderedDict()
        self._history = pd.DataFrame()
        self.refresh()

    # ----- φόρτωση -----
    @classmethod
    def load(
        cls,
        model_path: Path | str | None = None,
        engine: Engine | None = None,
        *,
        active_window_days: int = DEFAULT_ACTIVE_WINDOW_DAYS,
    ) -> Predictor:
        """Φορτώνει το μοντέλο (προεπιλογή: `MODEL_PATH` των ρυθμίσεων) και τη βάση
        (προεπιλογή: `DATABASE_URL`). Σηκώνει `ModelLoadError` αν το μοντέλο λείπει, είναι
        κατεστραμμένο ή ασύμβατο."""
        path = Path(model_path) if model_path is not None else Path(get_settings().model_path)
        bundle = load_bundle(path)
        metrics = dict(bundle.metrics)
        if not metrics:
            metrics = _read_metrics_file(path.parent / "metrics.json")
        return cls(
            bundle,
            engine if engine is not None else get_engine(),
            active_window_days=active_window_days,
            metrics=metrics,
            owns_engine=engine is None,
        )

    def close(self) -> None:
        """Κλείνει τις συνδέσεις της βάσης, αν ο Predictor δημιούργησε μόνος του τον engine
        (δηλαδή το `load()` κλήθηκε χωρίς `engine`). Ένας engine που δόθηκε απ' έξω δεν κλείνει."""
        if self._owns_engine:
            self._engine.dispose()

    @property
    def model_version(self) -> str:
        """Η έκδοση του φορτωμένου μοντέλου."""
        return self._bundle.model_version

    @property
    def metrics(self) -> dict:
        """Οι μετρικές εκπαίδευσης (περιεχόμενο του metrics.json): MAE, threshold, κ.λπ."""
        return self._metrics

    def refresh(self) -> None:
        """Ξαναδιαβάζει ιστορικό, αγώνες και πρόγραμμα από τη βάση και αδειάζει την cache."""
        history = load_history(self._engine)
        games = load_played_games(self._engine)
        schedule = load_schedule(self._engine)
        players = load_players(self._engine)
        with self._lock:
            self._history = history
            self._games = games
            self._schedule = schedule
            self._names = dict(zip(players["player_id"], players["name"], strict=True))
            self._index_history()
            self._cache.clear()

    def _index_history(self) -> None:
        """Προϋπολογίζει ό,τι χρειάζονται οι ερωτήσεις (ομάδα, τελευταία συμμετοχή, σεζόν)."""
        history = self._history
        if history.empty:
            self._latest = pd.DataFrame(columns=["team_code"])
            self._last_appearance = pd.Series(dtype="datetime64[ns]")
            self._season_starts = pd.Series(dtype="datetime64[ns]")
            return
        fallback = history["game_date"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        order = history.assign(_t=history["tipoff_utc"].fillna(fallback)).sort_values(
            ["_t", "season", "gamecode"], kind="stable"
        )
        self._latest = order.groupby("player_id", sort=False).tail(1).set_index("player_id")
        appeared = history[(~history["dnp"]) & (history["minutes"] > 0)]
        self._last_appearance = appeared.groupby("player_id")["game_date"].max()
        self._season_starts = history.groupby("season")["game_date"].min()

    # ----- ορισμοί -----
    def _reference_season(self, as_of: date) -> int | None:
        """Η νεότερη σεζόν με παιγμένους αγώνες ≤ as_of, αλλιώς η νεότερη σεζόν της βάσης."""
        if self._season_starts.empty:
            return None
        started = self._season_starts[self._season_starts <= pd.Timestamp(as_of)]
        return int(started.index.max()) if len(started) else int(self._season_starts.index.max())

    def _active_ids(self, as_of: date) -> set[str]:
        history = self._history
        if history.empty:
            return set()
        season = self._reference_season(as_of)
        active = set(history.loc[history["season"] == season, "player_id"])
        window_start = pd.Timestamp(as_of) - pd.Timedelta(days=self._active_window_days)
        recent = history[
            (history["game_date"] >= window_start) & (history["game_date"] <= pd.Timestamp(as_of))
        ]
        active |= set(recent["player_id"])
        if not active:
            newest = int(history["season"].max())
            active = set(history.loc[history["season"] == newest, "player_id"])
        if not active:
            active = set(history["player_id"])
        return active

    def _next_games(self, as_of: date) -> pd.DataFrame:
        """Ο επόμενος αγώνας κάθε ομάδας (δείκτης: κωδικός ομάδας)."""
        schedule = self._schedule
        columns = ["season", "gamecode", "game_date", "tipoff_utc", "phase", "opp", "home"]
        if schedule.empty:
            return pd.DataFrame(columns=columns)
        upcoming = schedule[schedule["game_date"] >= pd.Timestamp(as_of)]
        if upcoming.empty:
            return pd.DataFrame(columns=columns)
        home_side = upcoming.assign(
            team=upcoming["home_code"], opp=upcoming["away_code"], home=True
        )
        away_side = upcoming.assign(
            team=upcoming["away_code"], opp=upcoming["home_code"], home=False
        )
        sides = pd.concat([home_side, away_side], ignore_index=True)
        fallback = sides["game_date"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        sides["_t"] = sides["tipoff_utc"].fillna(fallback)
        sides = sides.sort_values(["_t", "game_date", "season", "gamecode"], kind="stable")
        first = sides.drop_duplicates("team", keep="first").set_index("team")
        return first[columns]

    # ----- υπολογισμός -----
    def _upcoming_rows(self, as_of: date) -> pd.DataFrame:
        """Μία γραμμή `upcoming` ανά παίκτη: ο επόμενος αγώνας της ομάδας του ή ουδέτερο πλαίσιο."""
        latest = self._latest
        next_games = self._next_games(as_of)
        season = self._reference_season(as_of)
        teams = latest["team_code"]
        game = next_games.reindex(teams.to_numpy())
        has_game = game["gamecode"].notna().to_numpy()
        neutral_time = pd.Timestamp(as_of) + pd.Timedelta(days=1)
        rows = pd.DataFrame(
            {
                "player_id": teams.index.to_numpy(),
                "season": np.where(has_game, game["season"].to_numpy(), season).astype(np.int64),
                "gamecode": np.where(
                    has_game, game["gamecode"].to_numpy(), _NEUTRAL_GAMECODE
                ).astype(np.int64),
                "team_code": teams.to_numpy(),
                "opp_code": np.where(has_game, game["opp"].to_numpy(), None),
                "home": np.where(has_game, game["home"].astype(float).to_numpy(), np.nan),
                "game_date": np.where(has_game, game["game_date"].to_numpy(), pd.Timestamp(as_of)),
                "tipoff_utc": np.where(has_game, game["tipoff_utc"].to_numpy(), neutral_time),
                "phase": np.where(has_game, game["phase"].to_numpy(), None),
            }
        )
        rows["game_date"] = pd.to_datetime(rows["game_date"]).astype("datetime64[ns]")
        rows["tipoff_utc"] = pd.to_datetime(rows["tipoff_utc"]).astype("datetime64[ns]")
        rows["opp_code"] = rows["opp_code"].astype(object)
        return rows

    def _compute(self, as_of: date) -> dict[str, PlayerPrediction]:
        if self._history.empty:
            return {}
        upcoming = self._upcoming_rows(as_of)
        frame = build_features(self._history, upcoming, games=self._games)
        rows = frame[frame["is_upcoming"]].reset_index(drop=True)
        # Μνήμη: μόνο οι γραμμές των επόμενων αγώνων χρειάζονται από εδώ και πέρα.
        del frame, upcoming
        predictions = self._bundle.predict(rows)
        active = self._active_ids(as_of)
        columns = self._bundle.feature_columns

        # Μετατροπή σε λίστες Python μία φορά (πολύ ταχύτερη από πρόσβαση ανά γραμμή του frame).
        player_ids = rows["player_id"].tolist()
        teams = rows["team_code"].tolist()
        opponents = rows["opp_code"].tolist()
        homes = rows["home"].tolist()
        seasons = rows["season"].tolist()
        gamecodes = rows["gamecode"].tolist()
        dates = rows["game_date"].dt.date.tolist()
        tipoffs = rows["tipoff_utc"].tolist()
        prior = rows["games_played_total"].astype(int).tolist()
        feature_rows = rows[columns].to_numpy(dtype=float).tolist()
        fantasy = predictions["fantasy"].tolist()
        pir = predictions["pir"].tolist()

        result: dict[str, PlayerPrediction] = {}
        for index, player_id in enumerate(player_ids):
            next_game = None
            if (
                opponents[index] is not None
                and not pd.isna(opponents[index])
                and homes[index] == homes[index]
            ):
                tipoff = tipoffs[index]
                next_game = NextGame(
                    season=int(seasons[index]),
                    gamecode=int(gamecodes[index]),
                    game_date=dates[index],
                    tipoff_utc=None if pd.isna(tipoff) else tipoff.to_pydatetime(),
                    team_code=str(teams[index]),
                    opp_code=str(opponents[index]),
                    home=bool(homes[index] == 1.0),
                )
            last = self._last_appearance.get(player_id)
            result[str(player_id)] = PlayerPrediction(
                player_id=str(player_id),
                name=str(self._names.get(player_id, player_id)),
                team_code=str(teams[index]),
                next_game=next_game,
                predicted_fantasy=float(fantasy[index]),
                predicted_pir=float(pir[index]),
                n_prior_appearances=int(prior[index]),
                last_appearance_date=None if last is None or pd.isna(last) else last.date(),
                features={
                    name: (value if value == value else None)
                    for name, value in zip(columns, feature_rows[index], strict=True)
                },
                model_version=self._bundle.model_version,
                is_active=str(player_id) in active,
            )
        return result

    def _predictions(self, as_of: date) -> dict[str, PlayerPrediction]:
        with self._lock:
            cached = self._cache.get(as_of)
            if cached is not None:
                self._cache.move_to_end(as_of)
                return cached
            computed = self._compute(as_of)
            self._cache[as_of] = computed
            while len(self._cache) > _CACHE_SIZE:
                self._cache.popitem(last=False)
            return computed

    # ----- δημόσιο API -----
    def predict_player(self, player_id: str, as_of: date | None = None) -> PlayerPrediction | None:
        """Πρόβλεψη για έναν παίκτη, ή None αν ο παίκτης είναι άγνωστος.

        Το `player_id` δέχεται κενά και πεζά γράμματα (strip και κεφαλαία). `as_of` είναι η
        ημερομηνία «από την οποία» ψάχνουμε τον επόμενο αγώνα (προεπιλογή σήμερα, UTC).
        """
        key = str(player_id).strip().upper()
        if not key:
            return None
        return self._predictions(_to_date(as_of) if as_of is not None else self._today()).get(key)

    def predict_all(
        self,
        as_of: date | None = None,
        team_code: str | None = None,
        active_only: bool = True,
    ) -> list[PlayerPrediction]:
        """Προβλέψεις για πολλούς παίκτες, φθίνουσα κατά `predicted_fantasy`.

        * `team_code`: μόνο οι παίκτες της ομάδας αυτής (strip και κεφαλαία).
        * `active_only`: μόνο οι ενεργοί παίκτες (βλ. τον ορισμό στη μονάδα). Με `False`
          επιστρέφονται όλοι οι γνωστοί παίκτες της βάσης.

        Σε ισοβαθμία το αποτέλεσμα ταξινομείται κατά `player_id`, ώστε να είναι σταθερό.
        """
        day = _to_date(as_of) if as_of is not None else self._today()
        predictions = self._predictions(day).values()
        team = None if team_code is None else str(team_code).strip().upper()
        selected = [
            prediction
            for prediction in predictions
            if (not active_only or prediction.is_active)
            and (team is None or prediction.team_code == team)
        ]
        selected.sort(key=lambda prediction: (-prediction.predicted_fantasy, prediction.player_id))
        return selected

    # ----- βοηθητικά -----
    def next_game_for_team(self, team_code: str, as_of: date | None = None) -> NextGame | None:
        """Ο επόμενος αγώνας μιας ομάδας (ή None). Δεν υπολογίζει μοντέλο."""
        day = _to_date(as_of) if as_of is not None else self._today()
        team = str(team_code).strip().upper()
        games = self._next_games(day)
        if team not in games.index:
            return None
        row = games.loc[team]
        tipoff = row["tipoff_utc"]
        return NextGame(
            season=int(row["season"]),
            gamecode=int(row["gamecode"]),
            game_date=row["game_date"].date(),
            tipoff_utc=None if pd.isna(tipoff) else tipoff.to_pydatetime(),
            team_code=team,
            opp_code=str(row["opp"]),
            home=bool(row["home"]),
        )


def _read_metrics_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


__all__ = [
    "DEFAULT_ACTIVE_WINDOW_DAYS",
    "ModelLoadError",
    "NextGame",
    "PlayerPrediction",
    "Predictor",
    "utc_today",
]
