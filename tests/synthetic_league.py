"""Συνθετικό πρωτάθλημα για tests: ντετερμινιστικό (με σπόρο), μικρό και με ρεαλιστικά στατιστικά.

Η γεννήτρια παράγει ό,τι χρειάζονται τα tests του feature engineering, της εκπαίδευσης και του
Predictor χωρίς την πραγματική βάση:

* ιστορικό σε μορφή `load_history` (παίκτης × αγώνας, με γραμμές DNP, τραυματισμούς, νέους
  παίκτες, μεταγραφές και μετατεθειμένους αγώνες),
* πίνακας παιγμένων αγώνων σε μορφή `load_played_games`,
* πρόγραμμα μελλοντικών αγώνων (`played = false`) στη νεότερη σεζόν,
* εγγραφή όλων στη βάση SQLite με το σχήμα του project (`write_to_database`).

Τα στατιστικά ακολουθούν ρεαλιστικές τάξεις μεγέθους (λεπτά: αρχική πεντάδα ~27, ρελέ ~17, πάγκος
~6, συνολικά 200 ανά ομάδα· PIR με μέσο όρο γύρω στο 8 και μεγάλο θόρυβο ανά αγώνα, +1 σε νίκη).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from elfantasy.db import models
from elfantasy.db.session import create_all
from elfantasy.scoring import fantasy_score

BOX_COLUMNS = (
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
)


@dataclass
class League:
    """Το αποτέλεσμα της γεννήτριας."""

    history: pd.DataFrame  # μορφή load_history
    games: pd.DataFrame  # μορφή load_played_games
    schedule: pd.DataFrame  # μορφή load_schedule (μόνο μελλοντικοί αγώνες)
    players: pd.DataFrame  # player_id, name, first_season, last_season
    teams: list[str]
    seasons: tuple[int, ...]

    @property
    def last_played_date(self) -> date:
        return self.games["game_date"].max().date()

    @property
    def first_future_date(self) -> date:
        return self.schedule["game_date"].min().date()


def _tipoff(day: date, rng: np.random.Generator) -> datetime:
    """Ώρα έναρξης (naive UTC) σε βήματα του μισάωρου από τις 17:00 έως τις 20:30."""
    return datetime(day.year, day.month, day.day, 17) + timedelta(
        minutes=30 * int(rng.integers(0, 8))
    )


def make_league(
    seed: int = 7,
    seasons: tuple[int, ...] = (2020, 2021, 2022, 2023),
    n_teams: int = 8,
    rounds: int = 16,
    future_rounds: int = 3,
    pool: int = 11,
) -> League:
    """Φτιάχνει ένα συνθετικό πρωτάθλημα.

    `rounds` είναι οι αγωνιστικές κάθε σεζόν και `future_rounds` οι επιπλέον αγωνιστικές της
    τελευταίας σεζόν που δεν έχουν παιχτεί ακόμη. Κάθε ομάδα έχει `pool` παίκτες.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{index:02d}" for index in range(n_teams)]
    attributes: dict[str, dict] = {}
    roster: dict[str, list[str]] = {team: [] for team in teams}
    counter = 0

    def new_player(team: str) -> str:
        nonlocal counter
        counter += 1
        player_id = f"P{counter:06d}"
        attributes[player_id] = {
            "name": f"PLAYER {counter}, TEST",
            "skill": float(rng.uniform(0.22, 0.62)),  # PIR ανά λεπτό
            "role": float(rng.choice([27.0, 27.0, 17.0, 17.0, 10.0, 5.0])),
            "injured_until": -1,
            "never_plays": False,
            "first_season": None,
            "last_season": None,
        }
        roster[team].append(player_id)
        return player_id

    for team in teams:
        for _ in range(pool):
            new_player(team)
    # Ένας παίκτης που βρίσκεται στη λίστα αγώνα αλλά δεν παίζει ποτέ (μόνο γραμμές DNP).
    attributes[roster[teams[0]][0]]["never_plays"] = True

    history_rows: list[dict] = []
    game_rows: list[dict] = []
    future_rows: list[dict] = []
    team_games: dict[str, int] = dict.fromkeys(teams, 0)  # αγώνες που έχει παίξει κάθε ομάδα
    gamecode = 0

    for season_index, season in enumerate(seasons):
        last_season = season_index == len(seasons) - 1
        if season_index > 0:
            # Μεσοσεζόν: αποχωρούν 2 παίκτες ανά ομάδα, έρχονται 2 νέοι και γίνεται μία μεταγραφή.
            for team in teams:
                for player_id in list(rng.choice(roster[team], size=2, replace=False)):
                    roster[team].remove(player_id)
                for _ in range(2):
                    new_player(team)
            source, target = rng.choice(teams, size=2, replace=False)
            moved = str(rng.choice(roster[source]))
            roster[source].remove(moved)
            roster[target].append(moved)
        strength = {team: float(rng.normal(0.0, 3.0)) for team in teams}
        gamecode = 0
        start = date(season, 10, 2)
        total_rounds = rounds + (future_rounds if last_season else 0)
        round_date = start
        used_dates: dict[str, set[date]] = {team: set() for team in teams}
        season_games: list[dict] = []
        for round_number in range(1, total_rounds + 1):
            order = [str(team) for team in rng.permutation(teams)]
            for pair in range(0, n_teams, 2):
                home_code, away_code = order[pair], order[pair + 1]
                gamecode += 1
                game_day = round_date
                # Μερικοί αγώνες μετατίθενται αργότερα: το gamecode δεν είναι χρονολογικό. Στη
                # τελευταία σεζόν δεν υπάρχουν αναβολές, ώστε κανένας παιγμένος αγώνας να μην έχει
                # ημερομηνία μεταγενέστερη από τους μελλοντικούς.
                if not last_season and round_number <= rounds and rng.random() < 0.08:
                    game_day = round_date + timedelta(days=int(rng.integers(12, 26)))
                while game_day in used_dates[home_code] or game_day in used_dates[away_code]:
                    game_day += timedelta(days=1)
                used_dates[home_code].add(game_day)
                used_dates[away_code].add(game_day)
                season_games.append(
                    {
                        "season": season,
                        "gamecode": gamecode,
                        "round": round_number,
                        "game_date": game_day,
                        "tipoff_utc": _tipoff(game_day, rng),
                        "home_code": home_code,
                        "away_code": away_code,
                        "future": last_season and round_number > rounds,
                    }
                )
            round_date += timedelta(days=int(rng.integers(3, 9)))

        season_games.sort(key=lambda g: (g["tipoff_utc"], g["gamecode"]))
        for game in season_games:
            home_code, away_code = game["home_code"], game["away_code"]
            if game["future"]:
                future_rows.append(
                    {
                        "season": season,
                        "gamecode": game["gamecode"],
                        "game_date": pd.Timestamp(game["game_date"]),
                        "tipoff_utc": pd.Timestamp(game["tipoff_utc"]),
                        "phase": "RS",
                        "home_code": home_code,
                        "away_code": away_code,
                    }
                )
                continue
            home_score = int(
                round(82 + 0.5 * (strength[home_code] - strength[away_code]) + rng.normal(0, 8))
            )
            away_score = int(
                round(80 + 0.5 * (strength[away_code] - strength[home_code]) + rng.normal(0, 8))
            )
            if home_score == away_score:
                home_score += 1
            game_rows.append(
                {
                    "season": season,
                    "gamecode": game["gamecode"],
                    "game_date": pd.Timestamp(game["game_date"]),
                    "tipoff_utc": pd.Timestamp(game["tipoff_utc"]),
                    "phase": "RS",
                    "home_code": home_code,
                    "away_code": away_code,
                    "home_score": float(home_score),
                    "away_score": float(away_score),
                }
            )
            for team, opp, home, score, opp_score in (
                (home_code, away_code, True, home_score, away_score),
                (away_code, home_code, False, away_score, home_score),
            ):
                won = score > opp_score
                team_games[team] += 1
                game_index = team_games[team]
                available: list[str] = []
                injured_rows: list[str] = []
                for player_id in roster[team]:
                    info = attributes[player_id]
                    if info["injured_until"] >= game_index:
                        if rng.random() < 0.4:
                            injured_rows.append(player_id)  # γραμμή DNP κατά τον τραυματισμό
                        continue
                    if rng.random() < 0.025:
                        info["injured_until"] = game_index + int(rng.integers(1, 8))
                        continue
                    available.append(player_id)
                # DNP «επιλογής προπονητή» για τους παίκτες του βάθους.
                playing = [
                    p
                    for p in available
                    if not attributes[p]["never_plays"]
                    and not (attributes[p]["role"] <= 10 and rng.random() < 0.3)
                ][:12]
                benched = [p for p in available if p not in playing][: max(0, 12 - len(playing))]
                ranked = sorted(playing, key=lambda p: -attributes[p]["role"])
                starters = set(ranked[:5])
                raw = np.array(
                    [attributes[p]["role"] * rng.uniform(0.8, 1.2) for p in playing], dtype=float
                )
                total_minutes = 225.0 if rng.random() < 0.08 else 200.0
                minutes = raw / raw.sum() * total_minutes if len(raw) else raw
                for player_id, played_minutes in zip(playing, minutes, strict=True):
                    info = attributes[player_id]
                    mean_pir = info["skill"] * played_minutes + (1.0 if won else -1.0)
                    pir_value = int(np.clip(round(mean_pir + rng.normal(0, 4.0)), -10, 50))
                    history_rows.append(
                        _history_row(
                            season,
                            game,
                            team,
                            opp,
                            home,
                            player_id,
                            player_id in starters,
                            float(played_minutes),
                            False,
                            pir_value,
                            won,
                            score,
                            opp_score,
                        )
                    )
                    _touch(info, season)
                for player_id in [*benched, *injured_rows]:
                    history_rows.append(
                        _history_row(
                            season,
                            game,
                            team,
                            opp,
                            home,
                            player_id,
                            False,
                            0.0,
                            True,
                            0,
                            won,
                            score,
                            opp_score,
                        )
                    )
                    _touch(attributes[player_id], season)

    history = pd.DataFrame(history_rows)
    history["game_date"] = pd.to_datetime(history["game_date"]).astype("datetime64[ns]")
    history["tipoff_utc"] = pd.to_datetime(history["tipoff_utc"]).astype("datetime64[ns]")
    games = pd.DataFrame(game_rows)
    schedule = pd.DataFrame(future_rows)
    for frame in (games, schedule):
        frame["game_date"] = frame["game_date"].astype("datetime64[ns]")
        frame["tipoff_utc"] = frame["tipoff_utc"].astype("datetime64[ns]")
    players = pd.DataFrame(
        [
            {
                "player_id": player_id,
                "name": info["name"],
                "first_season": info["first_season"],
                "last_season": info["last_season"],
            }
            for player_id, info in attributes.items()
            if info["first_season"] is not None
        ]
    )
    return League(history, games, schedule, players, teams, tuple(seasons))


def _touch(info: dict, season: int) -> None:
    if info["first_season"] is None:
        info["first_season"] = season
    info["last_season"] = season


def _history_row(
    season: int,
    game: dict,
    team: str,
    opp: str,
    home: bool,
    player_id: str,
    starter: bool,
    minutes: float,
    dnp: bool,
    pir_value: int,
    won: bool,
    team_score: int,
    opp_score: int,
) -> dict:
    return {
        "season": season,
        "gamecode": game["gamecode"],
        "player_id": player_id,
        "team_code": team,
        "opp_code": opp,
        "home": home,
        "is_starter": bool(starter),
        "minutes": minutes,
        "dnp": dnp,
        "pir": pir_value,
        "fantasy_score": fantasy_score(pir_value, won),
        "won": won,
        "game_date": pd.Timestamp(game["game_date"]),
        "tipoff_utc": pd.Timestamp(game["tipoff_utc"]),
        "phase": "RS",
        "team_score": float(team_score),
        "opp_score": float(opp_score),
    }


def write_to_database(engine: Engine, league: League) -> None:
    """Γράφει το πρωτάθλημα στη βάση (σχήμα του project): teams, players, games, player_games."""
    create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            models.teams.insert(),
            [{"team_code": team, "name": f"TEAM {team}"} for team in league.teams],
        )
        connection.execute(
            models.players.insert(),
            [
                {
                    "player_id": row.player_id,
                    "name": row.name,
                    "first_season": int(row.first_season),
                    "last_season": int(row.last_season),
                }
                for row in league.players.itertuples()
            ],
        )
        game_records = []
        for row in league.games.itertuples():
            home_wins = row.home_score > row.away_score
            game_records.append(
                {
                    "season": int(row.season),
                    "gamecode": int(row.gamecode),
                    "phase": row.phase,
                    "round": 1,
                    "game_date": row.game_date.date(),
                    "tipoff_utc": row.tipoff_utc.to_pydatetime(),
                    "home_code": row.home_code,
                    "away_code": row.away_code,
                    "home_score": int(row.home_score),
                    "away_score": int(row.away_score),
                    "played": True,
                    "winner_code": row.home_code if home_wins else row.away_code,
                }
            )
        for row in league.schedule.itertuples():
            game_records.append(
                {
                    "season": int(row.season),
                    "gamecode": int(row.gamecode),
                    "phase": row.phase,
                    "round": 1,
                    "game_date": row.game_date.date(),
                    "tipoff_utc": row.tipoff_utc.to_pydatetime(),
                    "home_code": row.home_code,
                    "away_code": row.away_code,
                    "home_score": None,
                    "away_score": None,
                    "played": False,
                    "winner_code": None,
                }
            )
        connection.execute(models.games.insert(), game_records)
        records = []
        for row in league.history.itertuples():
            record = {
                "season": int(row.season),
                "gamecode": int(row.gamecode),
                "player_id": row.player_id,
                "team_code": row.team_code,
                "opp_code": row.opp_code,
                "home": bool(row.home),
                "is_starter": bool(row.is_starter),
                "minutes": float(row.minutes),
                "dnp": bool(row.dnp),
                "plus_minus": None,
                "valuation": int(row.pir),
                "pir": int(row.pir),
                "won": bool(row.won),
                "fantasy_score": float(row.fantasy_score),
            }
            record.update({column: 0 for column in BOX_COLUMNS})
            records.append(record)
        connection.execute(models.player_games.insert(), records)
