"""Σχήμα της βάσης (SQLAlchemy 2.x Core), ίδιο για SQLite (τοπικά) και Postgres (Φάση 5).

Χρησιμοποιούνται μόνο γενικοί τύποι της SQLAlchemy (Integer, String, Float, Boolean, Date,
DateTime), ώστε το `metadata.create_all()` να δουλεύει σε κάθε dialect. Η ώρα έναρξης
`tipoff_utc` αποθηκεύεται σε UTC χωρίς πληροφορία ζώνης (naive datetime).

Το κλειδί των παικτών είναι πάντα το `player_id` (όχι το όνομα) και το κλειδί των ομάδων ο
κωδικός (`team_code`): τα ονόματα αλλάζουν με τα χρόνια (docs/DATA_SOURCES.md, ενότητες 5.11
και 5.16).
"""

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    func,
)

# Συμβάσεις ονομασίας: σταθερά ονόματα constraints/indexes σε κάθε βάση (χρήσιμο για migrations).
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)

# Ομάδες: κλειδί ο κωδικός (π.χ. "IST"), canonical όνομα το πιο πρόσφατο.
teams = Table(
    "teams",
    metadata,
    Column("team_code", String, primary_key=True),
    Column("name", String, nullable=False),
)

# Παίκτες: κλειδί το player_id (μετά από strip, π.χ. "P007200" ή παλαιά μορφή "PADF").
# Το name είναι το πιο πρόσφατο όνομα του ίδιου ID.
players = Table(
    "players",
    metadata,
    Column("player_id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("first_season", Integer, nullable=False),
    Column("last_season", Integer, nullable=False),
)

# Αγώνες: και οι παιγμένοι και οι μελλοντικοί από το schedule (played = false), ώστε το API να
# βρίσκει τον επόμενο αγώνα μιας ομάδας.
games = Table(
    "games",
    metadata,
    Column("season", Integer, primary_key=True, autoincrement=False),
    Column("gamecode", Integer, primary_key=True, autoincrement=False),
    Column("phase", String),  # RS, PO, FF, PI
    Column("round", Integer),
    Column("game_date", Date, nullable=False),
    Column("tipoff_utc", DateTime),
    Column("home_code", String, ForeignKey("teams.team_code"), nullable=False),
    Column("away_code", String, ForeignKey("teams.team_code"), nullable=False),
    Column("home_score", Integer),
    Column("away_score", Integer),
    Column("played", Boolean, nullable=False),
    Column("winner_code", String, ForeignKey("teams.team_code")),
    Index("ix_games_game_date", "game_date"),
    Index("ix_games_played_game_date", "played", "game_date"),
    Index("ix_games_home_code", "home_code"),
    Index("ix_games_away_code", "away_code"),
)

# Στατιστικά παίκτη ανά αγώνα (συμπεριλαμβάνονται και οι γραμμές DNP, με dnp = true και όλα 0).
player_games = Table(
    "player_games",
    metadata,
    Column("season", Integer, primary_key=True, autoincrement=False),
    Column("gamecode", Integer, primary_key=True, autoincrement=False),
    Column("player_id", String, ForeignKey("players.player_id"), primary_key=True),
    Column("team_code", String, ForeignKey("teams.team_code"), nullable=False),
    Column("opp_code", String, ForeignKey("teams.team_code"), nullable=False),
    Column("home", Boolean, nullable=False),
    Column("is_starter", Boolean, nullable=False),
    Column("minutes", Float, nullable=False),
    Column("dnp", Boolean, nullable=False),
    Column("points", Integer, nullable=False),
    Column("fg2_made", Integer, nullable=False),
    Column("fg2_attempted", Integer, nullable=False),
    Column("fg3_made", Integer, nullable=False),
    Column("fg3_attempted", Integer, nullable=False),
    Column("ft_made", Integer, nullable=False),
    Column("ft_attempted", Integer, nullable=False),
    Column("off_reb", Integer, nullable=False),
    Column("def_reb", Integer, nullable=False),
    Column("total_reb", Integer, nullable=False),
    Column("assists", Integer, nullable=False),
    Column("steals", Integer, nullable=False),
    Column("turnovers", Integer, nullable=False),
    Column("blocks_favour", Integer, nullable=False),
    Column("blocks_against", Integer, nullable=False),
    Column("fouls_committed", Integer, nullable=False),
    Column("fouls_received", Integer, nullable=False),
    Column("plus_minus", Float),
    Column("valuation", Integer, nullable=False),  # η στήλη Valuation του API (= PIR)
    Column("pir", Integer, nullable=False),  # υπολογισμένο από τα στατιστικά (scoring.py)
    Column("won", Boolean, nullable=False),
    Column("fantasy_score", Float, nullable=False),
    ForeignKeyConstraint(
        ["season", "gamecode"],
        ["games.season", "games.gamecode"],
        name="fk_player_games_game",
    ),
    Index("ix_player_games_player_id", "player_id"),
    Index("ix_player_games_team_code", "team_code", "season"),
    Index("ix_player_games_opp_code", "opp_code", "season"),
)

# Προβλέψεις του μοντέλου (χρησιμοποιείται στις Φάσεις 4 και 5). Τα season/gamecode μένουν
# nullable και χωρίς FK προς το games, ώστε να επιτρέπονται προβλέψεις και όταν δεν υπάρχει
# προγραμματισμένος αγώνας (neutral features).
predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("player_id", String, ForeignKey("players.player_id"), nullable=False),
    Column("season", Integer),
    Column("gamecode", Integer),
    Column("predicted_fantasy", Float, nullable=False),
    Column("predicted_pir", Float),
    Column("model_version", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Index("ix_predictions_player_id", "player_id"),
)
