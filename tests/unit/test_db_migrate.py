"""Tests του db/migrate.py: SQL splitter, φόρτωση αρχείων, τα πραγματικά migrations (drift έναντι
του db/models.py), ο runner (σειρά, checksum, idempotency, ατομικότητα) και το CLI.

Ο runner δοκιμάζεται εδώ offline σε SQLite με συνθετικά migrations· σε πραγματικό Postgres
δοκιμάζεται στο tests/integration/test_postgres.py (marker `postgres`).
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import inspect, text

from elfantasy.db import migrate, models
from elfantasy.db.migrate import (
    MIGRATIONS_DIR,
    MigrationConsistencyError,
    MigrationError,
    MigrationFileError,
    Migrator,
    NotPostgresError,
    checksum_of,
    generate_init_sql,
    load_migrations,
    require_postgres,
    split_sql,
)
from elfantasy.db.session import get_engine

SECRET = "S3cr3t-Pa55"


def write_migrations(folder: Path, files: dict[str, str]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, sql in files.items():
        (folder / name).write_bytes(sql.encode("utf-8"))
    return folder


def rows(engine, sql: str) -> list[tuple]:
    with engine.connect() as connection:
        return [tuple(row) for row in connection.execute(text(sql)).all()]


# ----------------------------------------------------------------------------------------------
# SQL splitter
# ----------------------------------------------------------------------------------------------


class TestSplitSql:
    @pytest.mark.parametrize(
        ("script", "expected"),
        [
            ("SELECT 1; SELECT 2;", ["SELECT 1", "SELECT 2"]),
            ("SELECT 1;\n\nSELECT 2", ["SELECT 1", "SELECT 2"]),  # χωρίς τελικό ερωτηματικό
            (
                "INSERT INTO t VALUES ('a;b'); SELECT 1",
                ["INSERT INTO t VALUES ('a;b')", "SELECT 1"],
            ),
            ("SELECT 'it''s; fine'; SELECT 2", ["SELECT 'it''s; fine'", "SELECT 2"]),
            ('CREATE TABLE "a;b" (x int); SELECT 1', ['CREATE TABLE "a;b" (x int)', "SELECT 1"]),
            ('SELECT "we""ird;"; SELECT 2', ['SELECT "we""ird;"', "SELECT 2"]),
            ("-- a; b\nSELECT 1;", ["SELECT 1"]),
            ("SELECT 1; -- trailing; comment", ["SELECT 1"]),
            ("/* a; /* nested; */ still; */ SELECT 1;", ["SELECT 1"]),
            ("SELECT 1 /* inline; */ + 2;", ["SELECT 1   + 2"]),
            (
                "DO $$ BEGIN PERFORM 1; PERFORM 2; END $$; SELECT 3",
                ["DO $$ BEGIN PERFORM 1; PERFORM 2; END $$", "SELECT 3"],
            ),
            (
                "DO $body$ BEGIN x; y; END $body$; SELECT 3",
                ["DO $body$ BEGIN x; y; END $body$", "SELECT 3"],
            ),
            ("DO $a$ x $$ ; $$ y $a$; SELECT 1", ["DO $a$ x $$ ; $$ y $a$", "SELECT 1"]),
            ("SELECT $1; SELECT 2", ["SELECT $1", "SELECT 2"]),  # placeholder, όχι dollar quote
            ("SELECT foo$bar$baz; SELECT 2", ["SELECT foo$bar$baz", "SELECT 2"]),  # $ μέσα σε όνομα
            ("INSERT INTO t VALUES ('Γειά; σου');", ["INSERT INTO t VALUES ('Γειά; σου')"]),
            ("", []),
            ("   \n\t ", []),
            ("-- only a comment\n/* and another */", []),
            (";;;", []),
        ],
    )
    def test_cases(self, script, expected):
        assert split_sql(script) == expected

    def test_an_unterminated_string_takes_the_rest_without_crashing(self):
        assert split_sql("SELECT 1; SELECT 'open; still") == ["SELECT 1", "SELECT 'open; still"]

    def test_an_unterminated_dollar_quote_takes_the_rest(self):
        assert split_sql("DO $$ BEGIN; END") == ["DO $$ BEGIN; END"]

    def test_an_unterminated_block_comment_swallows_the_rest(self):
        assert split_sql("SELECT 1; /* never closed; SELECT 2;") == ["SELECT 1"]

    def test_a_comment_inside_a_dollar_quoted_body_is_preserved(self):
        statement = "DO $$ BEGIN -- keep; me\n PERFORM 1; END $$"
        assert split_sql(statement + ";") == [statement]


# ----------------------------------------------------------------------------------------------
# Φόρτωση αρχείων
# ----------------------------------------------------------------------------------------------


class TestLoadMigrations:
    def test_sorted_by_number_not_by_text(self, tmp_path):
        folder = write_migrations(
            tmp_path,
            {
                "010_c.sql": "SELECT 3;",
                "002_b.sql": "SELECT 2;",
                "001_a.sql": "SELECT 1;",
                "1000_d.sql": "SELECT 4;",
            },
        )
        loaded = load_migrations(folder)
        assert [m.version for m in loaded] == ["001_a", "002_b", "010_c", "1000_d"]
        assert [m.number for m in loaded] == [1, 2, 10, 1000]

    def test_other_files_are_ignored(self, tmp_path):
        folder = write_migrations(
            tmp_path, {"001_a.sql": "SELECT 1;", "README.md": "x", "notes.txt": "y"}
        )
        assert [m.version for m in load_migrations(folder)] == ["001_a"]

    @pytest.mark.parametrize(
        "name",
        ["1_a.sql", "01_a.sql", "001-a.sql", "001_A.sql", "001_.sql", "a_001.sql", "001_a b.sql"],
    )
    def test_invalid_names_are_rejected(self, tmp_path, name):
        folder = write_migrations(tmp_path, {name: "SELECT 1;"})
        with pytest.raises(MigrationFileError, match="invalid migration file name"):
            load_migrations(folder)

    def test_a_byte_order_mark_is_rejected(self, tmp_path):
        folder = tmp_path
        (folder / "001_a.sql").write_bytes(b"\xef\xbb\xbfSELECT 1;")
        with pytest.raises(MigrationFileError, match="byte order mark"):
            load_migrations(folder)

    def test_a_file_that_is_not_utf8_is_rejected(self, tmp_path):
        (tmp_path / "001_a.sql").write_bytes(b"SELECT '\xe1\xe2';")  # cp1253, όχι UTF-8
        with pytest.raises(MigrationFileError, match="not valid UTF-8"):
            load_migrations(tmp_path)

    @pytest.mark.parametrize("content", ["", "   \n", "-- only a comment\n", "/* x */;"])
    def test_a_file_without_statements_is_rejected(self, tmp_path, content):
        folder = write_migrations(tmp_path, {"001_a.sql": content})
        with pytest.raises(MigrationFileError, match="no SQL statements"):
            load_migrations(folder)

    def test_duplicate_numbers_are_rejected(self, tmp_path):
        folder = write_migrations(tmp_path, {"001_a.sql": "SELECT 1;", "001_b.sql": "SELECT 2;"})
        with pytest.raises(MigrationFileError, match="same number"):
            load_migrations(folder)

    def test_a_missing_folder_is_rejected(self, tmp_path):
        with pytest.raises(MigrationFileError, match="not found"):
            load_migrations(tmp_path / "nope")

    def test_the_checksum_ignores_line_ending_style(self, tmp_path):
        lf = write_migrations(tmp_path / "lf", {"001_a.sql": "SELECT 1;\nSELECT 2;\n"})
        crlf = write_migrations(tmp_path / "crlf", {"001_a.sql": "SELECT 1;\r\nSELECT 2;\r\n"})
        first, second = load_migrations(lf)[0], load_migrations(crlf)[0]
        assert first.checksum == second.checksum
        assert "\r" not in second.sql
        assert checksum_of("x\r\ny\rz") == checksum_of("x\ny\nz")

    def test_the_checksum_detects_any_other_change(self):
        assert checksum_of("SELECT 1;") != checksum_of("SELECT 2;")
        assert len(checksum_of("")) == 64


# ----------------------------------------------------------------------------------------------
# Τα πραγματικά migrations του project
# ----------------------------------------------------------------------------------------------


class TestProjectMigrations:
    def test_the_two_migrations_exist_in_order(self):
        assert [m.version for m in load_migrations()] == ["001_init", "002_row_level_security"]

    def test_001_equals_the_ddl_generated_from_the_models(self):
        """DRIFT TEST: αν αλλάξει το db/models.py χωρίς νέο migration, αποτυγχάνει εδώ.

        Αν το 001 δεν έχει εφαρμοστεί ποτέ σε καμία βάση: `python -m elfantasy.db.migrate
        --print-sql > src/elfantasy/db/migrations/001_init.sql`. Αν έχει ήδη εφαρμοστεί (π.χ. στο
        Supabase) ΜΗΝ το αλλάξεις: πρόσθεσε νέο migration (003_...) με την αλλαγή του σχήματος και
        ενημέρωσε αυτό το test ώστε να συγκρίνει το 001 με το σχήμα της εποχής του.
        """
        committed = (MIGRATIONS_DIR / "001_init.sql").read_bytes().decode("utf-8")
        assert committed == generate_init_sql(), (
            "src/elfantasy/db/migrations/001_init.sql differs from the DDL generated from "
            "db/models.py: see the docstring of this test"
        )

    def test_the_generated_ddl_is_deterministic_across_hash_seeds(self):
        """Η σειρά του `table.indexes` (set) εξαρτάται από το hash seed: το αποτέλεσμα όχι."""
        code = (
            "import sys; from elfantasy.db.migrate import generate_init_sql; "
            "sys.stdout.buffer.write(generate_init_sql().encode('utf-8'))"
        )
        outputs = []
        for seed in ("1", "2", "12345"):
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                check=True,
                env={**_clean_env(), "PYTHONHASHSEED": seed},
            )
            outputs.append(result.stdout)
        assert outputs[0] == outputs[1] == outputs[2] == generate_init_sql().encode("utf-8")

    def test_every_table_has_a_description_and_the_order_is_valid(self):
        names = [table.name for table in models.TABLE_ORDER]
        assert sorted(names) == sorted(models.metadata.tables)  # κανένας πίνακας δεν ξεχνιέται
        assert set(migrate.TABLE_DESCRIPTIONS) == set(names)
        seen = set()
        for table in models.TABLE_ORDER:
            for foreign_key in table.foreign_keys:
                parent = foreign_key.column.table.name
                assert parent == table.name or parent in seen, f"{table.name} before {parent}"
            seen.add(table.name)

    def test_the_generated_sql_has_the_expected_statements(self):
        statements = split_sql(generate_init_sql())
        creates = [s for s in statements if s.startswith("CREATE TABLE")]
        indexes = [s for s in statements if s.startswith("CREATE INDEX")]
        assert len(creates) == len(models.TABLE_ORDER) == 6
        assert len(indexes) == sum(len(table.indexes) for table in models.TABLE_ORDER) == 8
        assert len(statements) == len(creates) + len(indexes)
        assert [re.match(r"CREATE TABLE (\w+)", s).group(1) for s in creates] == [
            t.name for t in models.TABLE_ORDER
        ]

    def test_postgres_types_and_constraints_in_001(self):
        sql = (MIGRATIONS_DIR / "001_init.sql").read_text(encoding="utf-8")
        for fragment in (
            "minutes DOUBLE PRECISION NOT NULL",
            "fantasy_score DOUBLE PRECISION NOT NULL",
            "played BOOLEAN NOT NULL",
            "game_date DATE NOT NULL",
            "tipoff_utc TIMESTAMP WITHOUT TIME ZONE",
            "updated_at TIMESTAMP WITH TIME ZONE NOT NULL",
            "created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL",
            "id SERIAL NOT NULL",
            "season INTEGER NOT NULL",
            "name TEXT NOT NULL",
            "CHECK (status IN ('out', 'doubtful', 'available'))",
            "UNIQUE (player_id, season, gamecode, model_version, as_of)",
            "CONSTRAINT pk_player_games PRIMARY KEY (season, gamecode, player_id)",
            "CREATE INDEX ix_games_played_game_date ON games (played, game_date)",
        ):
            assert fragment in sql, fragment
        assert "VARCHAR" not in sql and " FLOAT" not in sql and "IF NOT EXISTS" not in sql

    def test_all_identifiers_fit_in_postgres_63_bytes(self):
        """Το Postgres κόβει σιωπηλά ονόματα πάνω από 63 bytes: τα ονόματα του DDL χωράνε."""
        sql = (MIGRATIONS_DIR / "001_init.sql").read_text(encoding="utf-8")
        identifiers = re.findall(r"(?:CONSTRAINT|CREATE INDEX|CREATE TABLE) (\w+)", sql)
        assert len(identifiers) > 25
        assert max(len(name.encode("utf-8")) for name in identifiers) <= 63

    @pytest.mark.parametrize("name", ["001_init.sql", "002_row_level_security.sql"])
    def test_file_hygiene(self, name):
        raw = (MIGRATIONS_DIR / name).read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), "BOM"
        assert b"\r" not in raw, "αλλαγές γραμμής LF μόνο"
        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
        text_ = raw.decode("utf-8")
        assert "%" not in text_, "ο % θα ήταν placeholder για κάποιους drivers"
        assert "\t" not in text_
        assert all(line == line.rstrip() for line in text_.splitlines()), "κενά στο τέλος γραμμής"

    def test_002_enables_row_level_security_on_every_table_without_policies(self):
        sql = (MIGRATIONS_DIR / "002_row_level_security.sql").read_text(encoding="utf-8")
        statements = split_sql(sql)
        enabled = {
            match.group(1)
            for statement in statements
            if (match := re.fullmatch(r"ALTER TABLE (\w+) ENABLE ROW LEVEL SECURITY", statement))
        }
        # ο schema_migrations (δημιουργείται από τον runner) ενεργοποιείται μέσα σε DO block
        assert enabled == {table.name for table in models.TABLE_ORDER}
        assert "ALTER TABLE schema_migrations ENABLE ROW LEVEL SECURITY" in sql
        assert "to_regclass('schema_migrations')" in sql
        code = "\n".join(statements).upper()  # χωρίς τα σχόλια
        assert "POLICY" not in code, "deny-all: καμία policy"
        assert "FORCE ROW LEVEL SECURITY" not in code

    def test_002_revokes_only_when_the_supabase_roles_exist(self):
        sql = (MIGRATIONS_DIR / "002_row_level_security.sql").read_text(encoding="utf-8")
        blocks = [s for s in split_sql(sql) if s.startswith("DO ")]
        revoking = [block for block in blocks if "REVOKE" in block]
        assert len(revoking) == 1
        block = revoking[0]
        for role in ("anon", "authenticated"):
            guard = f"IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}')"
            assert guard in block
            assert block.index(guard) < block.index(f"FROM {role};")
        # τίποτα δεν αφαιρείται εκτός από τους δύο ρόλους και τους πίνακες/ακολουθία του project
        revoked = re.findall(r"REVOKE ALL ON (TABLE|SEQUENCE) ([\w, ]+) FROM (\w+);", block)
        assert {role for _, _, role in revoked} == {"anon", "authenticated"}
        tables = {
            name.strip()
            for kind, names, _ in revoked
            if kind == "TABLE"
            for name in names.split(",")
        }
        assert tables == {t.name for t in models.TABLE_ORDER} | {"schema_migrations"}
        assert {names for kind, names, _ in revoked if kind == "SEQUENCE"} == {"predictions_id_seq"}

    def test_the_sequence_name_matches_what_postgres_creates_for_serial(self):
        # SERIAL δημιουργεί <πίνακας>_<στήλη>_seq: το 002 αναφέρεται σε αυτό το όνομα
        assert "predictions_id_seq" in (MIGRATIONS_DIR / "002_row_level_security.sql").read_text(
            encoding="utf-8"
        )
        assert "id SERIAL" in (MIGRATIONS_DIR / "001_init.sql").read_text(encoding="utf-8")

    def test_the_package_ships_the_sql_files(self):
        config = (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
        assert "db/migrations/*.sql" in config


def _clean_env() -> dict[str, str]:
    import os

    keep = ("SYSTEMROOT", "PATH", "TEMP", "TMP", "PYTHONPATH", "PYTHONUTF8", "PYTHONIOENCODING")
    return {key: os.environ[key] for key in keep if key in os.environ}


# ----------------------------------------------------------------------------------------------
# Runner (SQLite, συνθετικά migrations)
# ----------------------------------------------------------------------------------------------

BASIC = {
    "001_create.sql": "CREATE TABLE items (x INTEGER);\nINSERT INTO items VALUES (1);\n",
    "002_more.sql": "INSERT INTO items VALUES (2);\nINSERT INTO items VALUES (3);\n",
}


@pytest.fixture
def engine(tmp_path):
    engine = get_engine(f"sqlite:///{(tmp_path / 'target.db').as_posix()}")
    yield engine
    engine.dispose()


@pytest.fixture
def folder(tmp_path):
    return write_migrations(tmp_path / "migrations", BASIC)


class TestMigrator:
    def test_applies_in_order_and_records_each_migration(self, engine, folder):
        applied = Migrator(engine, folder).apply()
        assert applied == ["001_create", "002_more"]
        assert rows(engine, "SELECT x FROM items ORDER BY x") == [(1,), (2,), (3,)]
        recorded = rows(engine, "SELECT version, checksum FROM schema_migrations ORDER BY version")
        expected = [(m.version, m.checksum) for m in load_migrations(folder)]
        assert recorded == expected

    def test_a_second_run_does_nothing(self, engine, folder):
        migrator = Migrator(engine, folder)
        migrator.apply()
        assert migrator.apply() == []
        assert rows(engine, "SELECT COUNT(*) FROM items") == [(3,)]
        assert rows(engine, "SELECT COUNT(*) FROM schema_migrations") == [(2,)]

    def test_only_new_migrations_are_applied_later(self, engine, folder):
        migrator = Migrator(engine, folder)
        migrator.apply()
        write_migrations(folder, {"003_third.sql": "INSERT INTO items VALUES (4);"})
        assert migrator.apply() == ["003_third"]
        assert rows(engine, "SELECT COUNT(*) FROM items") == [(4,)]

    def test_the_status_of_each_migration(self, engine, folder):
        migrator = Migrator(engine, folder)
        before = migrator.status()
        assert [(s.version, s.state, s.applied_at) for s in before] == [
            ("001_create", "pending", None),
            ("002_more", "pending", None),
        ]
        migrator.apply()
        after = migrator.status()
        assert [s.state for s in after] == ["applied", "applied"]
        assert all(s.applied_at is not None for s in after)
        assert all(s.file_checksum == s.recorded_checksum for s in after)
        assert migrator.problems(after) == []

    def test_dry_run_reports_without_touching_the_database(self, engine, folder):
        pending = Migrator(engine, folder).apply(dry_run=True)
        assert pending == ["001_create", "002_more"]
        assert inspect(engine).get_table_names() == []  # ούτε ο πίνακας schema_migrations

    def test_dry_run_when_up_to_date(self, engine, folder):
        migrator = Migrator(engine, folder)
        migrator.apply()
        assert migrator.apply(dry_run=True) == []

    def test_an_empty_folder_state(self, engine, tmp_path):
        folder = write_migrations(
            tmp_path / "one", {"001_only.sql": "CREATE TABLE only_one (x INTEGER);"}
        )
        assert Migrator(engine, folder).apply() == ["001_only"]

    # ----- ασυνέπειες: τίποτα δεν εφαρμόζεται -----
    def test_an_edited_applied_migration_stops_everything(self, engine, folder):
        migrator = Migrator(engine, folder)
        migrator.apply(dry_run=False)
        (folder / "001_create.sql").write_text(
            "CREATE TABLE items (x INTEGER); -- edited\n", encoding="utf-8"
        )
        write_migrations(folder, {"003_third.sql": "INSERT INTO items VALUES (99);"})
        statuses = migrator.status()
        assert [s.state for s in statuses] == ["changed", "applied", "pending"]
        with pytest.raises(
            MigrationConsistencyError, match="001_create.*changed after it was applied"
        ):
            migrator.apply()
        assert rows(engine, "SELECT COUNT(*) FROM items WHERE x = 99") == [
            (0,)
        ]  # το 003 δεν εφαρμόστηκε

    def test_an_applied_migration_without_a_file_stops_everything(self, engine, folder):
        migrator = Migrator(engine, folder)
        migrator.apply()
        (folder / "002_more.sql").unlink()
        assert [s.state for s in migrator.status()] == ["applied", "missing"]
        with pytest.raises(MigrationConsistencyError, match="002_more.*does not exist"):
            migrator.apply()

    def test_a_pending_migration_older_than_an_applied_one_is_rejected(self, engine, tmp_path):
        folder = write_migrations(
            tmp_path / "m",
            {"001_a.sql": "CREATE TABLE t (x INTEGER);", "003_c.sql": "INSERT INTO t VALUES (3);"},
        )
        migrator = Migrator(engine, folder)
        migrator.apply()
        write_migrations(folder, {"002_b.sql": "INSERT INTO t VALUES (2);"})
        with pytest.raises(MigrationConsistencyError, match="002_b.*older than the applied 003_c"):
            migrator.apply()
        assert rows(engine, "SELECT x FROM t ORDER BY x") == [(3,)]

    def test_every_problem_is_reported_at_once(self, engine, folder):
        migrator = Migrator(engine, folder)
        migrator.apply()
        (folder / "001_create.sql").write_text(
            "CREATE TABLE items (x INTEGER); -- edited", encoding="utf-8"
        )
        (folder / "002_more.sql").unlink()
        with pytest.raises(MigrationConsistencyError) as excinfo:
            migrator.apply()
        assert "001_create" in str(excinfo.value) and "002_more" in str(excinfo.value)

    # ----- αποτυχία μέσα σε migration -----
    def test_a_failing_migration_is_rolled_back_and_not_recorded(self, engine, tmp_path):
        folder = write_migrations(
            tmp_path / "m",
            {
                "001_ok.sql": "CREATE TABLE items (x INTEGER);\nINSERT INTO items VALUES (1);",
                "002_bad.sql": "INSERT INTO items VALUES (2);\nINSERT INTO items VALUES (3);\n"
                "INSERT INTO no_such_table VALUES (1);\nINSERT INTO items VALUES (4);",
                "003_never.sql": "INSERT INTO items VALUES (5);",
            },
        )
        with pytest.raises(
            MigrationError, match=r"002_bad failed at statement 3 of 4 \(INSERT INTO no_such_table"
        ) as excinfo:
            Migrator(engine, folder).apply()
        assert "no such table" in str(excinfo.value)
        assert rows(engine, "SELECT x FROM items ORDER BY x") == [(1,)]  # τα 2 και 3 ακυρώθηκαν
        assert rows(engine, "SELECT version FROM schema_migrations") == [("001_ok",)]

    def test_a_fixed_migration_applies_on_the_next_run(self, engine, tmp_path):
        folder = write_migrations(
            tmp_path / "m",
            {
                "001_ok.sql": "CREATE TABLE items (x INTEGER);",
                "002_bad.sql": "INSERT INTO nope VALUES (1);",
            },
        )
        migrator = Migrator(engine, folder)
        with pytest.raises(MigrationError):
            migrator.apply()
        (folder / "002_bad.sql").write_text("INSERT INTO items VALUES (2);", encoding="utf-8")
        assert migrator.apply() == ["002_bad"]
        assert rows(engine, "SELECT x FROM items") == [(2,)]

    def test_the_error_message_never_contains_a_password(self, engine, folder, monkeypatch):
        def failing(connection, statement):
            raise RuntimeError(f"connection to postgresql://user:{SECRET}@db.example.com/db lost")

        monkeypatch.setattr(migrate, "_run_statement", failing)
        with pytest.raises(MigrationError) as excinfo:
            Migrator(engine, folder).apply()
        assert SECRET not in str(excinfo.value)
        assert "user:***@db.example.com" in str(excinfo.value)

    def test_a_migration_applied_by_another_process_meanwhile_is_skipped(self, engine, folder):
        migrator = Migrator(engine, folder)
        migrator._bootstrap()
        first = load_migrations(folder)[0]
        with engine.begin() as connection:
            connection.execute(
                migrate.schema_migrations.insert().values(
                    version=first.version, checksum=first.checksum
                )
            )
        assert migrator._apply_one(first) is False
        assert "items" not in inspect(engine).get_table_names()  # το SQL δεν εκτελέστηκε

    def test_statements_run_as_is_with_percent_and_colon_characters(self, engine, tmp_path):
        folder = write_migrations(
            tmp_path / "m",
            {
                "001_x.sql": "CREATE TABLE t (v TEXT);\n"
                "INSERT INTO t VALUES ('100% sure: yes; really');\n"
                "-- a comment with %s and :name\nINSERT INTO t VALUES ('x');"
            },
        )
        Migrator(engine, folder).apply()
        assert rows(engine, "SELECT v FROM t ORDER BY v") == [("100% sure: yes; really",), ("x",)]

    def test_the_bookkeeping_table_is_created_once_and_has_the_documented_columns(
        self, engine, folder
    ):
        migrator = Migrator(engine, folder)
        migrator._bootstrap()
        migrator._bootstrap()  # idempotent
        columns = {c["name"]: c for c in inspect(engine).get_columns("schema_migrations")}
        assert list(columns) == ["version", "applied_at", "checksum"]
        assert all(not column["nullable"] for column in columns.values())
        assert inspect(engine).get_pk_constraint("schema_migrations")["constrained_columns"] == [
            "version"
        ]


class TestFormatStatus:
    def test_table(self, engine, folder):
        migrator = Migrator(engine, folder)
        text_ = migrate.format_status(migrator.status())
        assert text_.splitlines()[0].split() == [
            "version",
            "state",
            "applied",
            "at",
            "(UTC)",
            "checksum",
        ]
        assert "001_create" in text_ and "pending" in text_
        migrator.apply()
        assert "applied" in migrate.format_status(migrator.status())

    def test_empty(self):
        assert migrate.format_status([]) == "(no migrations)"

    def test_times_are_shown_in_utc_whatever_the_session_zone(self):
        from datetime import UTC, datetime, timedelta, timezone

        local = datetime(2026, 10, 4, 0, 44, 8, tzinfo=timezone(timedelta(hours=3)))
        status = migrate.MigrationStatus(
            "001_init", "applied", migrate._as_utc(local), "ab" * 32, "ab" * 32
        )
        assert status.applied_at == datetime(2026, 10, 3, 21, 44, 8, tzinfo=UTC)
        assert "2026-10-03 21:44:08" in migrate.format_status([status])
        # η SQLite επιστρέφει naive ώρες UTC: μένουν ίδιες
        assert migrate._as_utc(datetime(2026, 10, 3, 21, 44, 8)) == status.applied_at


# ----------------------------------------------------------------------------------------------
# Έλεγχος URL
# ----------------------------------------------------------------------------------------------


class TestRequirePostgres:
    def test_postgres_urls_pass(self):
        require_postgres("postgresql://u:p@h:5432/postgres")
        require_postgres("postgres://u:p@h/db")
        require_postgres("postgresql+psycopg://u:p@h/db")

    def test_sqlite_gets_a_clear_message(self):
        with pytest.raises(NotPostgresError, match="PostgreSQL only.*sqlite.*create_all"):
            require_postgres("sqlite:///data/elfantasy.db")

    def test_an_invalid_url_is_reported_without_echoing_it(self):
        with pytest.raises(NotPostgresError) as excinfo:
            require_postgres(f"postgresql://u:{SECRET}@h:badport/db")
        assert SECRET not in str(excinfo.value)


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------


class TestCli:
    def test_print_sql_writes_exactly_the_committed_file(self, capsysbinary, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "this is not a url")  # δεν χρειάζεται βάση
        assert migrate.main(["--print-sql"]) == 0
        assert capsysbinary.readouterr().out == (MIGRATIONS_DIR / "001_init.sql").read_bytes()

    def test_sqlite_is_refused_with_exit_code_2_and_nothing_is_created(self, tmp_path, capsys):
        db = tmp_path / "x.db"
        assert migrate.main(["--db", f"sqlite:///{db.as_posix()}"]) == 2
        out = capsys.readouterr().out
        assert "PostgreSQL only" in out and "create_all" in out
        assert not db.exists()

    def test_an_invalid_url_is_exit_code_2_without_the_secret(self, capsys):
        assert migrate.main(["--db", f"postgresql://u:{SECRET}@h:badport/db"]) == 2
        assert SECRET not in capsys.readouterr().out

    def test_an_invalid_prepare_threshold_is_exit_code_2_without_the_secret(self, capsys):
        url = f"postgresql://u:{SECRET}@127.0.0.1:1/db?prepare_threshold=oops"
        assert migrate.main(["--db", url]) == 2
        out = capsys.readouterr().out
        assert "prepare_threshold" in out and SECRET not in out

    def test_the_default_url_is_the_database_url_setting(self, monkeypatch, capsys):
        monkeypatch.setenv("DATABASE_URL", "sqlite:///from-env.db")
        migrate.get_settings.cache_clear()
        assert migrate.main(["--status"]) == 2
        assert "sqlite" in capsys.readouterr().out

    def test_the_modes_are_mutually_exclusive(self):
        with pytest.raises(SystemExit) as excinfo:
            migrate.main(["--status", "--dry-run"])
        assert excinfo.value.code == 2

    def test_an_unreachable_server_fails_with_exit_code_1_and_hides_the_password(self, capsys):
        # η πόρτα 1 του loopback δεν ακούει (καμία κίνηση δικτύου)· το connect_timeout του URL
        # περιορίζει την αναμονή στα Windows, όπου η απόρριψη δεν είναι πάντα άμεση
        url = f"postgresql://postgres:{SECRET}@127.0.0.1:1/postgres?connect_timeout=1"
        code = migrate.main(["--db", url])
        out = capsys.readouterr().out
        assert code == 1
        assert SECRET not in out and "postgres:***@127.0.0.1:1" in out

    @pytest.fixture
    def sqlite_cli(self, monkeypatch, tmp_path):
        """Το CLI πάνω σε SQLite (ο έλεγχος Postgres παρακάμπτεται, μόνο στα tests)."""
        monkeypatch.setattr(migrate, "require_postgres", lambda url: None)
        folder = write_migrations(tmp_path / "migrations", BASIC)
        db = f"sqlite:///{(tmp_path / 'cli.db').as_posix()}"
        return ["--db", db, "--migrations-dir", str(folder)], folder

    def test_status_dry_run_and_apply(self, sqlite_cli, capsys):
        arguments, _ = sqlite_cli
        assert migrate.main([*arguments, "--status"]) == 0
        assert "pending" in capsys.readouterr().out
        assert migrate.main([*arguments, "--dry-run"]) == 0
        assert "Would apply: 001_create, 002_more" in capsys.readouterr().out
        assert migrate.main(arguments) == 0
        assert "Applied: 001_create, 002_more" in capsys.readouterr().out
        assert migrate.main(arguments) == 0
        assert "nothing (up to date)" in capsys.readouterr().out
        assert migrate.main([*arguments, "--status"]) == 0
        assert capsys.readouterr().out.count("applied") >= 2

    def test_status_exits_1_when_a_migration_changed(self, sqlite_cli, capsys):
        arguments, folder = sqlite_cli
        assert migrate.main(arguments) == 0
        (folder / "001_create.sql").write_text(
            "CREATE TABLE items (x INTEGER); -- edited", encoding="utf-8"
        )
        capsys.readouterr()
        assert migrate.main([*arguments, "--status"]) == 1
        assert "changed" in capsys.readouterr().out
        assert migrate.main(arguments) == 1  # και η εφαρμογή αρνείται
        assert "changed after it was applied" in capsys.readouterr().out

    def test_a_failing_migration_exits_with_1(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(migrate, "require_postgres", lambda url: None)
        folder = write_migrations(tmp_path / "m", {"001_bad.sql": "INSERT INTO nope VALUES (1);"})
        db = f"sqlite:///{(tmp_path / 'f.db').as_posix()}"
        assert migrate.main(["--db", db, "--migrations-dir", str(folder)]) == 1
        assert "001_bad failed at statement 1 of 1" in capsys.readouterr().out

    def test_a_broken_folder_exits_with_1(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(migrate, "require_postgres", lambda url: None)
        db = f"sqlite:///{(tmp_path / 'f.db').as_posix()}"
        assert migrate.main(["--db", db, "--migrations-dir", str(tmp_path / "missing")]) == 1
        assert "not found" in capsys.readouterr().out

    def test_an_unexpected_error_is_reported_without_the_password(
        self, monkeypatch, tmp_path, capsys
    ):
        monkeypatch.setattr(migrate, "require_postgres", lambda url: None)

        def exploding(self, *, dry_run=False):
            raise RuntimeError(f"boom with postgresql://u:{SECRET}@h/db inside")

        monkeypatch.setattr(Migrator, "apply", exploding)
        db = f"sqlite:///{(tmp_path / 'g.db').as_posix()}"
        assert migrate.main(["--db", db]) == 1
        out = capsys.readouterr().out
        assert "boom" in out and SECRET not in out
