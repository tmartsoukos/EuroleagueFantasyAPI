-- 001_init.sql: αρχικό σχήμα της βάσης (PostgreSQL / Supabase).
--
-- ΠΑΡΑΓΕΤΑΙ ΑΥΤΟΜΑΤΑ από το src/elfantasy/db/models.py με την εντολή
--     python -m elfantasy.db.migrate --print-sql
-- και ένα test ελέγχει ότι το αρχείο ισούται με την έξοδό της. Μην το επεξεργάζεσαι με το χέρι.
--
-- Μετά την εφαρμογή του σε μια βάση το αρχείο ΔΕΝ αλλάζει ποτέ: το checksum του καταγράφεται στον
-- πίνακα schema_migrations και ο runner αρνείται να συνεχίσει αν διαφέρει. Κάθε επόμενη αλλαγή του
-- σχήματος γίνεται με ΝΕΟ αρχείο (003_..., 004_...).
--
-- Τύποι: text, integer, double precision, boolean, date, timestamp (η ώρα έναρξης tipoff_utc είναι
-- σε UTC, χωρίς ζώνη) και timestamptz (χρονικές στιγμές με ζώνη).

-- Ομάδες: κλειδί ο κωδικός της ομάδας (π.χ. IST), όνομα το πιο πρόσφατο.
CREATE TABLE teams (
    team_code TEXT NOT NULL,
    name TEXT NOT NULL,
    CONSTRAINT pk_teams PRIMARY KEY (team_code)
);

-- Παίκτες: κλειδί το player_id (π.χ. P007200 ή παλαιά μορφή PADF), όνομα το πιο πρόσφατο.
CREATE TABLE players (
    player_id TEXT NOT NULL,
    name TEXT NOT NULL,
    first_season INTEGER NOT NULL,
    last_season INTEGER NOT NULL,
    CONSTRAINT pk_players PRIMARY KEY (player_id)
);

-- Αγώνες: παιγμένοι και μελλοντικοί του προγράμματος (played = false). Η ώρα έναρξης
-- tipoff_utc είναι σε UTC, χωρίς ζώνη ώρας.
CREATE TABLE games (
    season INTEGER NOT NULL,
    gamecode INTEGER NOT NULL,
    phase TEXT,
    round INTEGER,
    game_date DATE NOT NULL,
    tipoff_utc TIMESTAMP WITHOUT TIME ZONE,
    home_code TEXT NOT NULL,
    away_code TEXT NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    played BOOLEAN NOT NULL,
    winner_code TEXT,
    CONSTRAINT pk_games PRIMARY KEY (season, gamecode),
    CONSTRAINT fk_games_home_code_teams FOREIGN KEY(home_code) REFERENCES teams (team_code),
    CONSTRAINT fk_games_away_code_teams FOREIGN KEY(away_code) REFERENCES teams (team_code),
    CONSTRAINT fk_games_winner_code_teams FOREIGN KEY(winner_code) REFERENCES teams (team_code)
);
CREATE INDEX ix_games_away_code ON games (away_code);
CREATE INDEX ix_games_game_date ON games (game_date);
CREATE INDEX ix_games_home_code ON games (home_code);
CREATE INDEX ix_games_played_game_date ON games (played, game_date);

-- Στατιστικά παίκτη ανά αγώνα, μαζί με τις γραμμές DNP (dnp = true, όλα 0). Τα pir και
-- fantasy_score υπολογίζονται από τα στατιστικά (scoring.py)· το valuation είναι η στήλη
-- Valuation του API.
CREATE TABLE player_games (
    season INTEGER NOT NULL,
    gamecode INTEGER NOT NULL,
    player_id TEXT NOT NULL,
    team_code TEXT NOT NULL,
    opp_code TEXT NOT NULL,
    home BOOLEAN NOT NULL,
    is_starter BOOLEAN NOT NULL,
    minutes DOUBLE PRECISION NOT NULL,
    dnp BOOLEAN NOT NULL,
    points INTEGER NOT NULL,
    fg2_made INTEGER NOT NULL,
    fg2_attempted INTEGER NOT NULL,
    fg3_made INTEGER NOT NULL,
    fg3_attempted INTEGER NOT NULL,
    ft_made INTEGER NOT NULL,
    ft_attempted INTEGER NOT NULL,
    off_reb INTEGER NOT NULL,
    def_reb INTEGER NOT NULL,
    total_reb INTEGER NOT NULL,
    assists INTEGER NOT NULL,
    steals INTEGER NOT NULL,
    turnovers INTEGER NOT NULL,
    blocks_favour INTEGER NOT NULL,
    blocks_against INTEGER NOT NULL,
    fouls_committed INTEGER NOT NULL,
    fouls_received INTEGER NOT NULL,
    plus_minus DOUBLE PRECISION,
    valuation INTEGER NOT NULL,
    pir INTEGER NOT NULL,
    won BOOLEAN NOT NULL,
    fantasy_score DOUBLE PRECISION NOT NULL,
    CONSTRAINT pk_player_games PRIMARY KEY (season, gamecode, player_id),
    CONSTRAINT fk_player_games_game FOREIGN KEY(season, gamecode) REFERENCES games (season, gamecode),
    CONSTRAINT fk_player_games_player_id_players FOREIGN KEY(player_id) REFERENCES players (player_id),
    CONSTRAINT fk_player_games_team_code_teams FOREIGN KEY(team_code) REFERENCES teams (team_code),
    CONSTRAINT fk_player_games_opp_code_teams FOREIGN KEY(opp_code) REFERENCES teams (team_code)
);
CREATE INDEX ix_player_games_opp_code ON player_games (opp_code, season);
CREATE INDEX ix_player_games_player_id ON player_games (player_id);
CREATE INDEX ix_player_games_team_code ON player_games (team_code, season);

-- Καταγεγραμμένες προβλέψεις του μοντέλου (python -m elfantasy.model.record_predictions):
-- μία γραμμή ανά (παίκτης, αγώνας, έκδοση μοντέλου, ημέρα πρόβλεψης as_of).
CREATE TABLE predictions (
    id SERIAL NOT NULL,
    player_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    gamecode INTEGER NOT NULL,
    predicted_fantasy DOUBLE PRECISION NOT NULL,
    predicted_pir DOUBLE PRECISION,
    model_version TEXT NOT NULL,
    as_of DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    CONSTRAINT pk_predictions PRIMARY KEY (id),
    CONSTRAINT fk_predictions_season_gamecode_games FOREIGN KEY(season, gamecode) REFERENCES games (season, gamecode),
    CONSTRAINT uq_predictions_player_id_season_gamecode_model_version_as_of UNIQUE (player_id, season, gamecode, model_version, as_of),
    CONSTRAINT fk_predictions_player_id_players FOREIGN KEY(player_id) REFERENCES players (player_id)
);
CREATE INDEX ix_predictions_player_id ON predictions (player_id);

-- Διαθεσιμότητα παικτών (τραυματισμοί, απουσίες): χειροκίνητο override, γράφεται μόνο
-- από το API.
CREATE TABLE player_availability (
    player_id TEXT NOT NULL,
    status TEXT NOT NULL,
    source TEXT,
    note TEXT,
    expected_return DATE,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    CONSTRAINT pk_player_availability PRIMARY KEY (player_id),
    CONSTRAINT ck_player_availability_status CHECK (status IN ('out', 'doubtful', 'available')),
    CONSTRAINT fk_player_availability_player_id_players FOREIGN KEY(player_id) REFERENCES players (player_id)
);
