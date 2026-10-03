"""Tests του db/models.py και db/session.py: σχήμα, engine και idempotent upsert."""

from datetime import date, datetime

import pytest
from sqlalchemy import case, func, inspect, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateIndex, CreateTable

from elfantasy.config import get_settings
from elfantasy.db import models, session
from elfantasy.db.session import create_all, get_engine, normalize_database_url, upsert


@pytest.fixture
def engine(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 'test.db'}")
    create_all(engine)
    yield engine
    engine.dispose()


def count(conn, table) -> int:
    return conn.execute(select(func.count()).select_from(table)).scalar_one()


class TestSchema:
    def test_tables_and_columns(self, engine):
        inspector = inspect(engine)
        assert set(inspector.get_table_names()) == {
            "teams",
            "players",
            "games",
            "player_games",
            "predictions",
        }
        columns = {
            name: [c["name"] for c in inspector.get_columns(name)]
            for name in inspector.get_table_names()
        }
        assert columns["teams"] == ["team_code", "name"]
        assert columns["players"] == ["player_id", "name", "first_season", "last_season"]
        assert columns["games"] == [
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
        assert columns["player_games"] == [
            "season",
            "gamecode",
            "player_id",
            "team_code",
            "opp_code",
            "home",
            "is_starter",
            "minutes",
            "dnp",
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
            "plus_minus",
            "valuation",
            "pir",
            "won",
            "fantasy_score",
        ]
        assert columns["predictions"] == [
            "id",
            "player_id",
            "season",
            "gamecode",
            "predicted_fantasy",
            "predicted_pir",
            "model_version",
            "created_at",
        ]

    def test_primary_keys(self, engine):
        inspector = inspect(engine)
        assert inspector.get_pk_constraint("teams")["constrained_columns"] == ["team_code"]
        assert inspector.get_pk_constraint("players")["constrained_columns"] == ["player_id"]
        assert inspector.get_pk_constraint("games")["constrained_columns"] == ["season", "gamecode"]
        assert inspector.get_pk_constraint("player_games")["constrained_columns"] == [
            "season",
            "gamecode",
            "player_id",
        ]
        assert inspector.get_pk_constraint("predictions")["constrained_columns"] == ["id"]

    def test_indexes(self, engine):
        inspector = inspect(engine)
        games = {i["name"]: i["column_names"] for i in inspector.get_indexes("games")}
        assert games["ix_games_game_date"] == ["game_date"]
        assert games["ix_games_played_game_date"] == ["played", "game_date"]
        player_games = {i["name"]: i["column_names"] for i in inspector.get_indexes("player_games")}
        assert player_games["ix_player_games_player_id"] == ["player_id"]

    def test_foreign_keys(self, engine):
        inspector = inspect(engine)
        referred = {
            (table, fk["referred_table"], tuple(fk["constrained_columns"]))
            for table in ("games", "player_games", "predictions")
            for fk in inspector.get_foreign_keys(table)
        }
        assert ("games", "teams", ("home_code",)) in referred
        assert ("games", "teams", ("away_code",)) in referred
        assert ("player_games", "players", ("player_id",)) in referred
        assert ("player_games", "games", ("season", "gamecode")) in referred
        assert ("predictions", "players", ("player_id",)) in referred

    def test_create_all_is_safe_to_repeat(self, engine):
        create_all(engine)
        create_all(engine)
        assert len(inspect(engine).get_table_names()) == 5

    def test_foreign_keys_are_enforced_in_sqlite(self, engine):
        with pytest.raises(IntegrityError), engine.begin() as conn:
            conn.execute(
                models.games.insert().values(
                    season=2025,
                    gamecode=1,
                    game_date=date(2025, 9, 30),
                    home_code="XXX",
                    away_code="YYY",
                    played=False,
                )
            )

    def test_predictions_get_an_id_and_a_timestamp(self, engine):
        with engine.begin() as conn:
            conn.execute(
                models.teams.insert().values(team_code="IST", name="ANADOLU EFES ISTANBUL")
            )
            conn.execute(
                models.players.insert().values(
                    player_id="P007200", name="LARKIN, SHANE", first_season=2025, last_season=2025
                )
            )
            for _ in range(2):
                conn.execute(
                    models.predictions.insert().values(
                        player_id="P007200", predicted_fantasy=21.5, model_version="test"
                    )
                )
            rows = conn.execute(select(models.predictions)).mappings().all()
        assert [row["id"] for row in rows] == [1, 2]
        assert all(isinstance(row["created_at"], datetime) for row in rows)
        assert rows[0]["season"] is None and rows[0]["gamecode"] is None

    def test_schema_compiles_for_postgres(self):
        """Το ίδιο σχήμα πρέπει να δουλεύει στο Supabase (Φάση 5): έλεγχος του DDL για Postgres."""
        dialect = postgresql.dialect()
        ddl = {
            table.name: str(CreateTable(table).compile(dialect=dialect))
            for table in models.metadata.sorted_tables
        }
        assert "id SERIAL NOT NULL" in ddl["predictions"]
        assert "created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL" in ddl["predictions"]
        assert "tipoff_utc TIMESTAMP WITHOUT TIME ZONE" in ddl["games"]
        assert "game_date DATE NOT NULL" in ddl["games"]
        assert (
            "FOREIGN KEY(season, gamecode) REFERENCES games (season, gamecode)"
            in ddl["player_games"]
        )
        assert "fantasy_score FLOAT NOT NULL" in ddl["player_games"]
        indexes = [
            str(CreateIndex(index).compile(dialect=dialect))
            for table in models.metadata.sorted_tables
            for index in table.indexes
        ]
        assert "CREATE INDEX ix_games_played_game_date ON games (played, game_date)" in indexes
        assert len(indexes) == 8


class TestEngine:
    def test_creates_the_folder_of_a_sqlite_file(self, tmp_path):
        target = tmp_path / "nested" / "folder" / "db.sqlite"
        engine = get_engine(f"sqlite:///{target}")
        create_all(engine)
        engine.dispose()
        assert target.exists()

    def test_in_memory_database(self):
        engine = get_engine("sqlite://")
        create_all(engine)
        assert "teams" in inspect(engine).get_table_names()
        engine.dispose()

    def test_default_url_comes_from_the_settings(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'from_env.db'}")
        get_settings.cache_clear()
        try:
            engine = get_engine()
            assert engine.url.database.endswith("from_env.db")
            engine.dispose()
        finally:
            get_settings.cache_clear()

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("postgresql://u:p@host:6543/postgres", "postgresql+psycopg://u:p@host:6543/postgres"),
            ("postgres://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
            ("postgresql+psycopg://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
            ("sqlite:///data/elfantasy.db", "sqlite:///data/elfantasy.db"),
        ],
    )
    def test_postgres_urls_use_the_psycopg_driver(self, url, expected):
        assert normalize_database_url(url) == expected


class TestUpsert:
    def test_insert_then_update_without_duplicates(self, engine):
        with engine.begin() as conn:
            assert (
                upsert(
                    conn,
                    models.teams,
                    [{"team_code": "IST", "name": "A"}, {"team_code": "TEL", "name": "B"}],
                    ["team_code"],
                )
                == 2
            )
            upsert(conn, models.teams, [{"team_code": "IST", "name": "A2"}], ["team_code"])
            rows = conn.execute(select(models.teams).order_by(models.teams.c.team_code)).all()
        assert rows == [("IST", "A2"), ("TEL", "B")]

    def test_rerunning_the_same_rows_is_idempotent(self, engine):
        rows = [{"team_code": code, "name": code * 2} for code in ("AAA", "BBB", "CCC")]
        with engine.begin() as conn:
            for _ in range(3):
                upsert(conn, models.teams, rows, ["team_code"])
            assert count(conn, models.teams) == 3

    def test_empty_input_does_nothing(self, engine):
        with engine.begin() as conn:
            assert upsert(conn, models.teams, [], ["team_code"]) == 0
            assert count(conn, models.teams) == 0

    def test_composite_key(self, engine):
        teams = [{"team_code": "IST", "name": "A"}, {"team_code": "TEL", "name": "B"}]
        game = {
            "season": 2025,
            "gamecode": 1,
            "phase": "RS",
            "round": 1,
            "game_date": date(2025, 9, 30),
            "tipoff_utc": datetime(2025, 9, 30, 17, 45),
            "home_code": "IST",
            "away_code": "TEL",
            "home_score": None,
            "away_score": None,
            "played": False,
            "winner_code": None,
        }
        with engine.begin() as conn:
            upsert(conn, models.teams, teams, ["team_code"])
            upsert(conn, models.games, [game], ["season", "gamecode"])
            upsert(
                conn,
                models.games,
                [
                    {
                        **game,
                        "played": True,
                        "home_score": 85,
                        "away_score": 78,
                        "winner_code": "IST",
                    }
                ],
                ["season", "gamecode"],
            )
            row = conn.execute(select(models.games)).mappings().one()
        assert (row["played"], row["home_score"], row["winner_code"]) == (True, 85, "IST")
        assert row["tipoff_utc"] == datetime(2025, 9, 30, 17, 45)
        assert row["game_date"] == date(2025, 9, 30)

    def test_empty_set_keeps_existing_rows_and_inserts_new_ones(self, engine):
        with engine.begin() as conn:
            upsert(conn, models.teams, [{"team_code": "IST", "name": "OLD"}], ["team_code"])
            upsert(
                conn,
                models.teams,
                [{"team_code": "IST", "name": "NEW"}, {"team_code": "TEL", "name": "NEW"}],
                ["team_code"],
                set_={},
            )
            rows = conn.execute(select(models.teams).order_by(models.teams.c.team_code)).all()
        assert rows == [("IST", "OLD"), ("TEL", "NEW")]

    def test_explicit_set_expressions_can_keep_the_widest_season_range(self, engine):
        table = models.players.c

        def widest(excluded):
            return {
                "first_season": case(
                    (excluded.first_season < table.first_season, excluded.first_season),
                    else_=table.first_season,
                ),
                "last_season": case(
                    (excluded.last_season > table.last_season, excluded.last_season),
                    else_=table.last_season,
                ),
            }

        def player(first, last, name):
            return {"player_id": "P1", "name": name, "first_season": first, "last_season": last}

        with engine.begin() as conn:
            upsert(conn, models.players, [player(2018, 2022, "OLD")], ["player_id"])
            upsert(conn, models.players, [player(2020, 2025, "NEW")], ["player_id"], set_=widest)
            upsert(conn, models.players, [player(2016, 2019, "OLDER")], ["player_id"], set_=widest)
            row = conn.execute(select(models.players)).mappings().one()
        # Το εύρος μόνο διευρύνεται, ενώ το όνομα δεν ενημερώνεται (δεν υπάρχει στο set_).
        assert (row["first_season"], row["last_season"], row["name"]) == (2016, 2025, "OLD")

    def test_set_may_be_a_plain_mapping(self, engine):
        with engine.begin() as conn:
            upsert(conn, models.teams, [{"team_code": "IST", "name": "OLD"}], ["team_code"])
            upsert(
                conn,
                models.teams,
                [{"team_code": "IST", "name": "NEW"}],
                ["team_code"],
                set_={"name": "FIXED"},
            )
            assert conn.execute(select(models.teams.c.name)).scalar_one() == "FIXED"

    def test_large_inputs_are_sent_in_chunks(self, engine, monkeypatch):
        monkeypatch.setattr(session, "UPSERT_CHUNK_SIZE", 2)
        rows = [{"team_code": f"T{i:02d}", "name": f"Team {i}"} for i in range(5)]
        with engine.begin() as conn:
            assert upsert(conn, models.teams, rows, ["team_code"]) == 5
            assert count(conn, models.teams) == 5

    def test_unknown_dialect_is_rejected(self):
        class Dialect:
            name = "mysql"

        class Connection:
            dialect = Dialect()

        with pytest.raises(NotImplementedError, match="mysql"):
            upsert(Connection(), models.teams, [{"team_code": "A", "name": "B"}], ["team_code"])

    def test_postgres_statement_uses_on_conflict_do_update(self):
        executed = []

        class Connection:
            dialect = postgresql.dialect()

            def execute(self, statement, parameters):
                executed.append((statement, parameters))

        rows = [{"team_code": "IST", "name": "A"}]
        upsert(Connection(), models.teams, rows, ["team_code"])
        statement, parameters = executed[0]
        sql = str(statement.compile(dialect=postgresql.dialect()))
        assert "ON CONFLICT (team_code) DO UPDATE SET name = excluded.name" in sql
        assert parameters == rows

    def test_postgres_statement_without_updates_does_nothing_on_conflict(self):
        executed = []

        class Connection:
            dialect = postgresql.dialect()

            def execute(self, statement, parameters):
                executed.append(statement)

        upsert(
            Connection(), models.teams, [{"team_code": "IST", "name": "A"}], ["team_code"], set_={}
        )
        assert "ON CONFLICT (team_code) DO NOTHING" in str(
            executed[0].compile(dialect=postgresql.dialect())
        )
