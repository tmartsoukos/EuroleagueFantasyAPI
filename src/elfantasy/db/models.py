"""Σχήμα της βάσης (SQLAlchemy 2.x Core), ίδιο για SQLite (τοπικά) και Postgres (Supabase, Φάση 5).

Χρησιμοποιούνται μόνο γενικοί τύποι της SQLAlchemy (Integer, Text, Double, Boolean, Date,
DateTime), ώστε το `metadata.create_all()` να δουλεύει σε κάθε dialect. Στο Postgres
μεταγλωττίζονται σε `integer`, `text`, `double precision`, `boolean`, `date`, `timestamp` και
`timestamptz`. Η ώρα έναρξης `tipoff_utc` αποθηκεύεται σε UTC χωρίς πληροφορία ζώνης (naive
datetime).

Το κλειδί των παικτών είναι πάντα το `player_id` (όχι το όνομα) και το κλειδί των ομάδων ο
κωδικός (`team_code`): τα ονόματα αλλάζουν με τα χρόνια (docs/DATA_SOURCES.md, ενότητες 5.11
και 5.16).

Στο Postgres το σχήμα δημιουργείται ΜΟΝΟ από τα migrations (`db/migrations/*.sql`, εργαλείο
`python -m elfantasy.db.migrate`): το `001_init.sql` παράγεται αυτούσιο από αυτό το αρχείο και ένα
test ελέγχει ότι δεν υπάρχει απόκλιση (docs/DATABASE.md). Όποιος αλλάζει το σχήμα εδώ πρέπει να
προσθέσει και νέο migration. Στο SQLite το σχήμα δημιουργείται με `create_all`.
"""

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Double,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    func,
)

# Συμβάσεις ονομασίας: σταθερά ονόματα constraints/indexes σε κάθε βάση (χρήσιμο για migrations).
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)

# Ομάδες: κλειδί ο κωδικός (π.χ. "IST"), canonical όνομα το πιο πρόσφατο.
teams = Table(
    "teams",
    metadata,
    Column("team_code", Text, primary_key=True),
    Column("name", Text, nullable=False),
)

# Παίκτες: κλειδί το player_id (μετά από strip, π.χ. "P007200" ή παλαιά μορφή "PADF").
# Το name είναι το πιο πρόσφατο όνομα του ίδιου ID.
players = Table(
    "players",
    metadata,
    Column("player_id", Text, primary_key=True),
    Column("name", Text, nullable=False),
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
    Column("phase", Text),  # RS, PO, FF, PI
    Column("round", Integer),
    Column("game_date", Date, nullable=False),
    Column("tipoff_utc", DateTime),
    Column("home_code", Text, ForeignKey("teams.team_code"), nullable=False),
    Column("away_code", Text, ForeignKey("teams.team_code"), nullable=False),
    Column("home_score", Integer),
    Column("away_score", Integer),
    Column("played", Boolean, nullable=False),
    Column("winner_code", Text, ForeignKey("teams.team_code")),
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
    Column("player_id", Text, ForeignKey("players.player_id"), primary_key=True),
    Column("team_code", Text, ForeignKey("teams.team_code"), nullable=False),
    Column("opp_code", Text, ForeignKey("teams.team_code"), nullable=False),
    Column("home", Boolean, nullable=False),
    Column("is_starter", Boolean, nullable=False),
    Column("minutes", Double, nullable=False),
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
    Column("plus_minus", Double),
    Column("valuation", Integer, nullable=False),  # η στήλη Valuation του API (= PIR)
    Column("pir", Integer, nullable=False),  # υπολογισμένο από τα στατιστικά (scoring.py)
    Column("won", Boolean, nullable=False),
    Column("fantasy_score", Double, nullable=False),
    ForeignKeyConstraint(
        ["season", "gamecode"],
        ["games.season", "games.gamecode"],
        name="fk_player_games_game",
    ),
    Index("ix_player_games_player_id", "player_id"),
    Index("ix_player_games_team_code", "team_code", "season"),
    Index("ix_player_games_opp_code", "opp_code", "season"),
)

# Καταγεγραμμένες προβλέψεις του μοντέλου (Φάση 5, `python -m elfantasy.model.record_predictions`).
# Μία γραμμή ανά (παίκτης, επόμενος αγώνας, έκδοση μοντέλου, ημέρα πρόβλεψης `as_of`): το unique
# constraint κάνει το ξανατρέξιμο της ίδιας ημέρας idempotent (upsert, χωρίς διπλές γραμμές). Το
# `as_of` είναι η ημερομηνία από την οποία ψάχνει ο Predictor τον επόμενο αγώνα. Τα `season` και
# `gamecode` είναι ΠΑΝΤΑ συμπληρωμένα και δείχνουν σε υπαρκτό αγώνα του πίνακα `games`: παίκτες
# χωρίς προγραμματισμένο αγώνα (ουδέτερο πλαίσιο) ΔΕΝ καταγράφονται, γιατί δεν υπάρχει αγώνας με
# τον οποίο να συγκριθεί η πρόβλεψη, και γιατί το NULL σε unique constraint δεν θα απέτρεπε
# διπλές γραμμές. Το API δεν γράφει ποτέ εδώ (τα GET αιτήματα δεν έχουν παρενέργειες).
predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("player_id", Text, ForeignKey("players.player_id"), nullable=False),
    Column("season", Integer, nullable=False),
    Column("gamecode", Integer, nullable=False),
    Column("predicted_fantasy", Double, nullable=False),
    Column("predicted_pir", Double),
    Column("model_version", Text, nullable=False),
    Column("as_of", Date, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    ForeignKeyConstraint(["season", "gamecode"], ["games.season", "games.gamecode"]),
    UniqueConstraint("player_id", "season", "gamecode", "model_version", "as_of"),
    Index("ix_predictions_player_id", "player_id"),
)

# Διαθεσιμότητα παικτών (Φάση 4): χειροκίνητο override για τραυματισμούς και απουσίες, μία γραμμή
# ανά παίκτη. Γράφεται ΜΟΝΟ από το API (POST/DELETE /availability, με X-API-Key), ποτέ από το
# ingestion. Τα `out` και `doubtful` εφαρμόζονται από το API μετά την πρόβλεψη του μοντέλου και
# δεν περνούν ως feature (docs/API.md). Παίκτης χωρίς γραμμή θεωρείται διαθέσιμος. Το
# `updated_at` ορίζεται πάντα από τον server, σε UTC. Το CHECK constraint δουλεύει αυτούσιο σε
# SQLite και Postgres. Στο SQLite το ίδιο το API δημιουργεί τον πίνακα στο startup αν λείπει
# (checkfirst)· στο Postgres ο πίνακας δημιουργείται από τα migrations και το API δεν τον
# δημιουργεί ποτέ (db/session.py, `ensure_schema`).
player_availability = Table(
    "player_availability",
    metadata,
    Column("player_id", Text, ForeignKey("players.player_id"), primary_key=True),
    Column("status", Text, nullable=False),  # out | doubtful | available
    Column("source", Text),  # ελεύθερο κείμενο: από πού προέρχεται η πληροφορία
    Column("note", Text),  # ελεύθερο κείμενο (το API επιτρέπει έως 500 χαρακτήρες)
    Column("expected_return", Date),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    CheckConstraint("status IN ('out', 'doubtful', 'available')", name="status"),
)

# Η σειρά με την οποία δημιουργούνται και γεμίζουν οι πίνακες (migrations, μεταφορά δεδομένων):
# κάθε πίνακας έπεται όσων αναφέρονται από τα foreign keys του. Ένα test ελέγχει ότι η σειρά
# είναι έγκυρη και ότι καλύπτει όλο το `metadata`, άρα ένας νέος πίνακας δεν μπορεί να ξεχαστεί.
TABLE_ORDER = (teams, players, games, player_games, predictions, player_availability)
