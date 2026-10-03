"""Ανάγνωση δεδομένων από τη βάση για το feature engineering, την εκπαίδευση και το API.

Οι συναρτήσεις επιστρέφουν DataFrame με ακριβώς τις στήλες που περιμένει το
`elfantasy.features.build.build_features`. Ο ίδιος κώδικας φόρτωσης εξυπηρετεί το training και
τον `Predictor`, ώστε τα δεδομένα εισόδου να είναι πάντα ίδιας μορφής.

Ώρες: το `tipoff_utc` της βάσης είναι naive datetime σε UTC και το `game_date` ημερομηνία.
Όλες οι στήλες χρόνου επιστρέφονται ως `datetime64[ns]`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import Engine, select

from elfantasy.db.models import games, player_games, players


def _to_datetime_ns(series: pd.Series) -> pd.Series:
    """Μετατρέπει στήλη ημερομηνιών/ωρών σε `datetime64[ns]` (το pandas 3 δίνει s ή us)."""
    return pd.to_datetime(series).astype("datetime64[ns]")


def load_history(engine: Engine) -> pd.DataFrame:
    """Επιστρέφει το ιστορικό: `player_games` ⨝ `games` για τους παιγμένους αγώνες.

    Μία γραμμή ανά (παίκτης, αγώνας), συμπεριλαμβανομένων των γραμμών DNP. Στήλες:
    `season, gamecode, player_id, team_code, opp_code, home, is_starter, minutes, dnp, pir,
    fantasy_score, won, game_date, tipoff_utc, phase, team_score, opp_score`. Τα `team_score` και
    `opp_score` είναι το σκορ της ομάδας του παίκτη και του αντιπάλου (από το `games`).
    """
    on_game = (player_games.c.season == games.c.season) & (
        player_games.c.gamecode == games.c.gamecode
    )
    statement = (
        select(
            player_games.c.season,
            player_games.c.gamecode,
            player_games.c.player_id,
            player_games.c.team_code,
            player_games.c.opp_code,
            player_games.c.home,
            player_games.c.is_starter,
            player_games.c.minutes,
            player_games.c.dnp,
            player_games.c.pir,
            player_games.c.fantasy_score,
            player_games.c.won,
            games.c.game_date,
            games.c.tipoff_utc,
            games.c.phase,
            games.c.home_score,
            games.c.away_score,
        )
        .select_from(player_games.join(games, on_game))
        .where(games.c.played.is_(True))
    )
    with engine.connect() as connection:
        frame = pd.read_sql(statement, connection)
    frame["game_date"] = _to_datetime_ns(frame["game_date"])
    frame["tipoff_utc"] = _to_datetime_ns(frame["tipoff_utc"])
    home = frame["home"].astype(bool)
    frame["team_score"] = np.where(home, frame["home_score"], frame["away_score"]).astype(float)
    frame["opp_score"] = np.where(home, frame["away_score"], frame["home_score"]).astype(float)
    frame = frame.drop(columns=["home_score", "away_score"])
    for column in ("home", "is_starter", "dnp", "won"):
        frame[column] = frame[column].astype(bool)
    return frame


def load_played_games(engine: Engine) -> pd.DataFrame:
    """Επιστρέφει όλους τους παιγμένους αγώνες του `games` (και όσους δεν έχουν boxscore).

    Στήλες: `season, gamecode, game_date, tipoff_utc, phase, home_code, away_code, home_score,
    away_score`. Χρησιμοποιείται για τα χαρακτηριστικά ισχύος των ομάδων, που υπολογίζονται από
    τα σκορ και άρα δεν απαιτούν boxscore (π.χ. ο αγώνας 2018/21 δεν έχει γραμμές παικτών).
    """
    statement = select(
        games.c.season,
        games.c.gamecode,
        games.c.game_date,
        games.c.tipoff_utc,
        games.c.phase,
        games.c.home_code,
        games.c.away_code,
        games.c.home_score,
        games.c.away_score,
    ).where(games.c.played.is_(True))
    with engine.connect() as connection:
        frame = pd.read_sql(statement, connection)
    frame["game_date"] = _to_datetime_ns(frame["game_date"])
    frame["tipoff_utc"] = _to_datetime_ns(frame["tipoff_utc"])
    frame["home_score"] = frame["home_score"].astype(float)
    frame["away_score"] = frame["away_score"].astype(float)
    return frame


def load_schedule(engine: Engine) -> pd.DataFrame:
    """Επιστρέφει τους μη παιγμένους αγώνες (`played = false`): μελλοντικούς ΚΑΙ ακυρωμένους.

    Στήλες: `season, gamecode, game_date, tipoff_utc, phase, home_code, away_code`. Ο καλών
    φιλτράρει κατά ημερομηνία: αγώνες του παρελθόντος με `played = false` δεν διεξήχθησαν ποτέ
    (docs/INGESTION.md, ενότητα 5).
    """
    statement = select(
        games.c.season,
        games.c.gamecode,
        games.c.game_date,
        games.c.tipoff_utc,
        games.c.phase,
        games.c.home_code,
        games.c.away_code,
    ).where(games.c.played.is_(False))
    with engine.connect() as connection:
        frame = pd.read_sql(statement, connection)
    frame["game_date"] = _to_datetime_ns(frame["game_date"])
    frame["tipoff_utc"] = _to_datetime_ns(frame["tipoff_utc"])
    return frame


def load_players(engine: Engine) -> pd.DataFrame:
    """Επιστρέφει τους παίκτες: `player_id, name, first_season, last_season`."""
    statement = select(
        players.c.player_id, players.c.name, players.c.first_season, players.c.last_season
    )
    with engine.connect() as connection:
        return pd.read_sql(statement, connection)
