"""Βοηθητικά των tests καταγραφής και αξιολόγησης προβλέψεων (Φάση 5).

Προσομοιώνουν το «μέλλον»: οι προβλέψεις έχουν καταγραφεί για μελλοντικούς αγώνες και κάποια στιγμή
οι αγώνες παίζονται, δηλαδή η βάση παίρνει σκορ και γραμμές boxscore.
"""

from __future__ import annotations

from sqlalchemy import Engine, select, text
from synthetic_league import BOX_COLUMNS

from elfantasy.db import models


def recorded_rows(engine: Engine) -> list[dict]:
    """Όλες οι γραμμές του πίνακα `predictions`, με σειρά `id`."""
    with engine.connect() as connection:
        rows = connection.execute(select(models.predictions).order_by(models.predictions.c.id))
        return [dict(row) for row in rows.mappings()]


def predicted_for_game(engine: Engine, season: int, gamecode: int) -> list[dict]:
    """Οι καταγεγραμμένες προβλέψεις ενός αγώνα."""
    return [
        row
        for row in recorded_rows(engine)
        if (row["season"], row["gamecode"]) == (season, gamecode)
    ]


def busiest_game(engine: Engine) -> tuple[int, int]:
    """Ο αγώνας με τις περισσότερες καταγεγραμμένες προβλέψεις (ντετερμινιστικά)."""
    counts: dict[tuple[int, int], int] = {}
    for row in recorded_rows(engine):
        key = (row["season"], row["gamecode"])
        counts[key] = counts.get(key, 0) + 1
    return max(counts, key=lambda key: (counts[key], key))


def play_game(
    engine: Engine,
    predictor,
    season: int,
    gamecode: int,
    *,
    appeared: dict[str, float],
    dnp: list[str],
    scores: tuple[int, int] = (90, 80),
) -> None:
    """Ο αγώνας παίζεται: η βάση παίρνει σκορ και γραμμές boxscore.

    `appeared`: {player_id: πραγματικό fantasy}· `dnp`: παίκτες με γραμμή DNP. Οι υπόλοιποι παίκτες
    του αγώνα δεν έχουν γραμμή (δεν υπάρχουν στο boxscore). Η ομάδα και ο αντίπαλος κάθε παίκτη
    προέρχονται από τον επόμενο αγώνα που προέβλεψε ο `predictor`.
    """
    games = models.games
    with engine.begin() as connection:
        row = connection.execute(
            select(games.c.home_code, games.c.away_code).where(
                (games.c.season == season) & (games.c.gamecode == gamecode)
            )
        ).one()
        connection.execute(
            games.update()
            .where((games.c.season == season) & (games.c.gamecode == gamecode))
            .values(
                played=True, home_score=scores[0], away_score=scores[1], winner_code=row.home_code
            )
        )
        for player_id in [*appeared, *dnp]:
            game = predictor.predict_player(player_id).next_game
            did_play = player_id in appeared
            fantasy = float(appeared[player_id]) if did_play else 0.0
            values = {column: 0 for column in BOX_COLUMNS}
            values.update(
                season=season,
                gamecode=gamecode,
                player_id=player_id,
                team_code=game.team_code,
                opp_code=game.opp_code,
                home=game.home,
                is_starter=False,
                minutes=21.5 if did_play else 0.0,
                dnp=not did_play,
                plus_minus=None,
                valuation=int(round(fantasy)),
                pir=int(round(fantasy)),
                won=(game.team_code == row.home_code),
                fantasy_score=fantasy,
            )
            connection.execute(models.player_games.insert().values(**values))


# Η διάταξη του πίνακα `predictions` των Φάσεων 2 έως 4 (χωρίς `as_of`), όπως υπάρχει ακόμη στις
# τοπικές βάσεις SQLite που δημιουργήθηκαν τότε.
LEGACY_PREDICTIONS_DDL = (
    "CREATE TABLE predictions (id INTEGER NOT NULL, player_id VARCHAR NOT NULL, season INTEGER, "
    "gamecode INTEGER, predicted_fantasy FLOAT NOT NULL, predicted_pir FLOAT, "
    "model_version VARCHAR NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL, "
    "CONSTRAINT pk_predictions PRIMARY KEY (id), "
    "CONSTRAINT fk_predictions_player_id_players FOREIGN KEY(player_id) "
    "REFERENCES players (player_id))"
)


def make_legacy_predictions(engine: Engine) -> None:
    """Αντικαθιστά τον πίνακα `predictions` της βάσης (SQLite) με την παλιά του διάταξη, κενό."""
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS predictions"))
        connection.execute(text(LEGACY_PREDICTIONS_DDL))
        connection.execute(text("CREATE INDEX ix_predictions_player_id ON predictions (player_id)"))
