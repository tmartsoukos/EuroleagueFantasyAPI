"""Βοηθητικά των tests του API (Φάση 4): ρολόι, ρυθμίσεις, ψεύτικος Predictor και μικρή βάση.

Το `FakePredictor` υλοποιεί ακριβώς όσα καλεί η υπηρεσία του API (`predict_player`, `predict_all`,
`refresh`, `model_version`, `metrics`), ώστε οι κανόνες των rankings (ταξινόμηση, ισοβαθμίες,
αρνητικές τιμές, `out`) να ελέγχονται με ακριβείς, χειροποίητες τιμές και όχι με τις τυχαίες
προβλέψεις ενός μοντέλου.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from sqlalchemy import Engine

from elfantasy.config import Settings
from elfantasy.db import models
from elfantasy.db.session import create_all, get_engine
from elfantasy.model.predict import NextGame, PlayerPrediction

# Η ημέρα του πλήρους τρεξίματος: τα 2026/31-33 των πραγματικών fixtures είναι μελλοντικά.
REAL_TODAY = date(2026, 10, 3)

ADMIN_KEY = "test-admin-key-0123456789"
ADMIN_HEADERS = {"X-API-Key": ADMIN_KEY}


class FakeClock:
    """Ρολόι που ορίζεται από το test (datetime UTC) και μπορεί να προχωρήσει."""

    def __init__(self, moment: datetime):
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment

    def advance(self, **kwargs) -> None:
        self.moment += timedelta(**kwargs)


def noon_utc(day: date) -> datetime:
    """Η 12:00 UTC της ημέρας `day`."""
    return datetime(day.year, day.month, day.day, 12, tzinfo=UTC)


def make_settings(**overrides) -> Settings:
    """Ρυθμίσεις για tests: δεν διαβάζεται το `.env`, κλειδί διαχειριστή το `ADMIN_KEY`."""
    values = {
        "database_url": "sqlite:///unused-by-tests.db",
        "model_path": "unused-by-tests.joblib",
        "admin_api_key": ADMIN_KEY,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class FakePredictor:
    """Ψεύτικος `Predictor` με έτοιμες προβλέψεις."""

    model_version = "fake-v1"

    def __init__(self, predictions: list[PlayerPrediction], metrics: dict | None = None):
        self._predictions = {p.player_id: p for p in predictions}
        self.metrics = metrics if metrics is not None else {}
        self.refresh_calls = 0
        self.predict_all_calls: list[dict] = []

    def predict_player(self, player_id: str, as_of: date | None = None):
        return self._predictions.get(str(player_id).strip().upper())

    def predict_all(self, as_of=None, team_code=None, active_only=True):
        self.predict_all_calls.append(
            {"as_of": as_of, "team_code": team_code, "active_only": active_only}
        )
        team = None if team_code is None else team_code.strip().upper()
        selected = [
            p
            for p in self._predictions.values()
            if (not active_only or p.is_active) and (team is None or p.team_code == team)
        ]
        selected.sort(key=lambda p: (-p.predicted_fantasy, p.player_id))
        return selected

    def refresh(self) -> None:
        self.refresh_calls += 1

    def close(self) -> None:
        pass


def make_prediction(
    player_id: str,
    fantasy: float,
    *,
    name: str | None = None,
    team: str = "AAA",
    pir: float | None = None,
    active: bool = True,
    next_game: NextGame | None = None,
    appearances: int = 30,
    last_appearance: date | None = date(2026, 10, 1),
) -> PlayerPrediction:
    """Χειροποίητη πρόβλεψη. Χωρίς `next_game` δεν υπάρχει προγραμματισμένος αγώνας."""
    return PlayerPrediction(
        player_id=player_id,
        name=name if name is not None else f"PLAYER {player_id}, TEST",
        team_code=team,
        next_game=next_game,
        predicted_fantasy=fantasy,
        predicted_pir=fantasy - 0.5 if pir is None else pir,
        n_prior_appearances=appearances,
        last_appearance_date=last_appearance,
        features={"pir_mean_5": 7.123456789, "home": None},
        model_version="fake-v1",
        is_active=active,
    )


def make_next_game(
    team: str = "AAA", opponent: str = "BBB", home: bool = True, day: date = date(2026, 10, 7)
) -> NextGame:
    return NextGame(
        season=2026,
        gamecode=31,
        game_date=day,
        tipoff_utc=datetime(day.year, day.month, day.day, 18, 45),
        team_code=team,
        opp_code=opponent,
        home=home,
    )


def make_small_database(path: Path, players: list[tuple[str, str]], teams=("AAA", "BBB")) -> Engine:
    """Μικρή βάση SQLite με πίνακες, ομάδες και παίκτες `(player_id, όνομα)` (χωρίς αγώνες)."""
    engine = get_engine(f"sqlite:///{path.as_posix()}")
    create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            models.teams.insert(),
            [{"team_code": code, "name": f"TEAM {code}"} for code in teams],
        )
        if players:
            connection.execute(
                models.players.insert(),
                [
                    {
                        "player_id": player_id,
                        "name": name,
                        "first_season": 2026,
                        "last_season": 2026,
                    }
                    for player_id, name in players
                ],
            )
    return engine
