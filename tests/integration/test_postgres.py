"""Tests σε ΠΡΑΓΜΑΤΙΚΟ Postgres (τοπικός server, marker `postgres`): ό,τι δεν δείχνει η SQLite.

Ο server είναι το `TEST_DATABASE_URL` (μόνο τοπικός) ή ο ενσωματωμένος `pixeltable-pgserver`
(βλ. tests/pg_support.py). Τα tests παραλείπονται αν δεν υπάρχει· ΠΟΤΕ δεν συνδέονται σε remote
βάση. Καλύπτουν: τα migrations (ατομικότητα, checksum, ταυτόχρονα τρεξίματα, RLS, δικαιώματα),
ότι το σχήμα της βάσης ισούται με το `db/models.py`, τη μεταφορά SQLite → Postgres (και τη
διαχείριση ζωνών ώρας), το ingestion, τον `Predictor`, το API, την καταγραφή προβλέψεων και τον
engine (pre-ping, παράμετροι σύνδεσης).
"""

from __future__ import annotations

import shutil
import threading
from datetime import UTC, date, datetime, timedelta

import pytest
from api_support import ADMIN_HEADERS, make_settings
from pg_support import with_query
from recorded_support import busiest_game, play_game, predicted_for_game
from sqlalchemy import (
    CheckConstraint,
    UniqueConstraint,
    create_engine,
    delete,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError, ProgrammingError
from synthetic_league import add_recorded_rows, write_to_database

from elfantasy.db import models, transfer
from elfantasy.db.migrate import (
    MIGRATIONS_DIR,
    MigrationConsistencyError,
    MigrationError,
    Migrator,
    load_migrations,
)
from elfantasy.db.session import SchemaNotInitialisedError, ensure_schema, get_engine
from elfantasy.ingest import pipeline
from elfantasy.ingest import verify as ingest_verify
from elfantasy.model import evaluate_recorded as evaluator
from elfantasy.model import record_predictions as recorder
from elfantasy.model.predict import Predictor

pytestmark = pytest.mark.postgres

ALL_TABLES = {table.name for table in models.TABLE_ORDER}
SEASONS = [2016, 2023, 2024, 2025, 2026]
TOKYO = {"options": "-c TimeZone=Asia/Tokyo"}  # συνεδρία με ζώνη +09:00, για τα tests χρόνου


def table_names(engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def scalar(engine, sql: str, **params):
    with engine.connect() as connection:
        return connection.execute(text(sql), params).scalar_one()


def count(engine, table) -> int:
    with engine.connect() as connection:
        return connection.execute(select(func.count()).select_from(table)).scalar_one()


def counts(engine) -> dict[str, int]:
    return {table.name: count(engine, table) for table in models.TABLE_ORDER}


def utc_engine(url: str):
    """Engine με συνεδρία UTC (ανεξάρτητα από τη ζώνη του server) για την ανάγνωση χρόνων."""
    return create_engine(url, connect_args={"options": "-c TimeZone=UTC"})


# ----------------------------------------------------------------------------------------------
# Migrations
# ----------------------------------------------------------------------------------------------


class TestMigrations:
    def test_apply_is_idempotent_and_records_every_migration(self, pg_engine):
        migrator = Migrator(pg_engine)
        assert migrator.apply(dry_run=True) == ["001_init", "002_row_level_security"]
        assert table_names(pg_engine) == set()  # το dry run δεν άγγιξε τη βάση
        assert migrator.apply() == ["001_init", "002_row_level_security"]
        assert migrator.apply() == []  # δεύτερη φορά: τίποτα
        assert table_names(pg_engine) == ALL_TABLES | {"schema_migrations"}
        with pg_engine.connect() as connection:
            recorded = connection.execute(
                text("SELECT version, checksum FROM schema_migrations ORDER BY version")
            ).all()
        assert [tuple(row) for row in recorded] == [
            (m.version, m.checksum) for m in load_migrations()
        ]
        statuses = migrator.status()
        assert [s.state for s in statuses] == ["applied", "applied"]
        assert all(s.applied_at is not None and s.applied_at.tzinfo is not None for s in statuses)
        assert migrator.problems(statuses) == []

    def test_the_bookkeeping_table_has_the_documented_definition(self, pg_migrated):
        columns = {c["name"]: c for c in inspect(pg_migrated).get_columns("schema_migrations")}
        assert [str(columns[name]["type"]) for name in ("version", "applied_at", "checksum")] == [
            "TEXT",
            "TIMESTAMP",
            "TEXT",
        ]
        assert str(columns["applied_at"]["type"].timezone) == "True"
        assert "now()" in columns["applied_at"]["default"]
        assert inspect(pg_migrated).get_pk_constraint("schema_migrations")[
            "constrained_columns"
        ] == ["version"]

    def test_row_level_security_is_enabled_on_every_table_without_policies(self, pg_migrated):
        with pg_migrated.connect() as connection:
            flags = {
                row.relname: (row.relrowsecurity, row.relforcerowsecurity)
                for row in connection.execute(
                    text(
                        "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class "
                        "WHERE relnamespace = 'public'::regnamespace AND relkind = 'r'"
                    )
                )
            }
            policies = connection.execute(text("SELECT count(*) FROM pg_policies")).scalar_one()
        assert flags == dict.fromkeys(ALL_TABLES | {"schema_migrations"}, (True, False))
        assert policies == 0  # deny-all: καμία policy

    def test_the_second_migration_runs_on_plain_postgres_without_supabase_roles(self, pg_engine):
        roles = scalar(
            pg_engine, "SELECT count(*) FROM pg_roles WHERE rolname IN ('anon', 'authenticated')"
        )
        if roles:
            pytest.skip("ο server έχει ήδη ρόλους anon/authenticated")
        assert Migrator(pg_engine).apply() == ["001_init", "002_row_level_security"]

    def test_the_schema_matches_the_models(self, pg_migrated):
        """Drift test: το σχήμα που δίνουν τα migrations ισούται με το σχήμα του db/models.py."""
        inspector = inspect(pg_migrated)
        dialect = postgresql.dialect()
        for table in models.TABLE_ORDER:
            columns = inspector.get_columns(table.name)
            assert [c["name"] for c in columns] == [c.name for c in table.columns], table.name
            for actual, expected in zip(columns, table.columns, strict=True):
                where = f"{table.name}.{expected.name}"
                assert actual["type"].compile(dialect=dialect) == expected.type.compile(
                    dialect=dialect
                ), where
                assert actual["nullable"] == expected.nullable, where
            primary_key = inspector.get_pk_constraint(table.name)
            assert primary_key["name"] == table.primary_key.name
            assert primary_key["constrained_columns"] == [c.name for c in table.primary_key.columns]
            foreign_keys = {
                fk["name"]: (
                    fk["constrained_columns"],
                    fk["referred_table"],
                    fk["referred_columns"],
                )
                for fk in inspector.get_foreign_keys(table.name)
            }
            expected_fks = {
                fk.name: (
                    [element.parent.name for element in fk.elements],
                    fk.referred_table.name,
                    [element.column.name for element in fk.elements],
                )
                for fk in table.foreign_key_constraints
            }
            assert foreign_keys == expected_fks, table.name
            uniques = {
                u["name"]: u["column_names"] for u in inspector.get_unique_constraints(table.name)
            }
            expected_uniques = {
                c.name: [column.name for column in c.columns]
                for c in table.constraints
                if isinstance(c, UniqueConstraint)
            }
            assert uniques == expected_uniques, table.name
            checks = {c["name"] for c in inspector.get_check_constraints(table.name)}
            assert checks == {c.name for c in table.constraints if isinstance(c, CheckConstraint)}
            indexes = {
                i["name"]: i["column_names"]
                for i in inspector.get_indexes(table.name)
                if not i.get("duplicates_constraint")
            }
            assert indexes == {i.name: [c.name for c in i.columns] for i in table.indexes}, (
                table.name
            )

    def test_defaults_and_the_serial_column(self, pg_migrated):
        columns = {c["name"]: c for c in inspect(pg_migrated).get_columns("predictions")}
        assert "predictions_id_seq" in columns["id"]["default"]
        assert "now()" in columns["created_at"]["default"]
        with pg_migrated.begin() as connection:
            connection.execute(text("INSERT INTO teams VALUES ('AAA', 'A'), ('BBB', 'B')"))
            connection.execute(text("INSERT INTO players VALUES ('P1', 'X', 2026, 2026)"))
            connection.execute(
                text(
                    "INSERT INTO games (season, gamecode, game_date, home_code, away_code, played) "
                    "VALUES (2026, 1, '2026-10-07', 'AAA', 'BBB', false)"
                )
            )
            ids = (
                connection.execute(
                    text(
                        "INSERT INTO predictions (player_id, season, gamecode, predicted_fantasy, "
                        "model_version, as_of) VALUES ('P1', 2026, 1, 5.0, 'v', '2026-10-03'), "
                        "('P1', 2026, 1, 6.0, 'v', '2026-10-04') RETURNING id"
                    )
                )
                .scalars()
                .all()
            )
        assert ids == [1, 2]
        created = scalar(pg_migrated, "SELECT created_at FROM predictions WHERE id = 1")
        assert created.tzinfo is not None
        assert abs(datetime.now(UTC) - created) < timedelta(minutes=1)

    def test_constraints_are_enforced_by_postgres(self, pg_migrated):
        with pg_migrated.begin() as connection:
            connection.execute(text("INSERT INTO teams VALUES ('AAA', 'A')"))
        for statement in (
            "INSERT INTO players VALUES (NULL, 'X', 2026, 2026)",  # NOT NULL / PK
            "INSERT INTO games (season, gamecode, game_date, home_code, away_code, played) "
            "VALUES (2026, 1, '2026-10-07', 'AAA', 'ZZZ', false)",  # FK προς teams
            "INSERT INTO player_availability VALUES ('NOPE', 'out', NULL, NULL, NULL, now())",  # FK
        ):
            with pytest.raises(IntegrityError), pg_migrated.begin() as connection:
                connection.execute(text(statement))
        with pg_migrated.begin() as connection:
            connection.execute(text("INSERT INTO players VALUES ('P1', 'X', 2026, 2026)"))
        with (
            pytest.raises(IntegrityError, match="ck_player_availability_status"),
            pg_migrated.begin() as connection,
        ):
            connection.execute(
                text(
                    "INSERT INTO player_availability "
                    "VALUES ('P1', 'broken', NULL, NULL, NULL, now())"
                )
            )

    # ----- ατομικότητα και ασφάλεια -----
    def test_a_failing_migration_leaves_nothing_behind(self, pg_engine, tmp_path):
        folder = tmp_path / "m"
        folder.mkdir()
        (folder / "001_ok.sql").write_text("CREATE TABLE a (x integer);", encoding="utf-8")
        (folder / "002_bad.sql").write_text(
            "CREATE TABLE b (x integer);\nINSERT INTO b VALUES (1);\nCREATE TABLE a (x integer);",
            encoding="utf-8",
        )
        with pytest.raises(
            MigrationError, match=r"002_bad failed at statement 3 of 3.*already exists"
        ):
            Migrator(pg_engine, folder).apply()
        assert table_names(pg_engine) == {
            "a",
            "schema_migrations",
        }  # ούτε ο πίνακας b (transactional DDL)
        assert (
            scalar(pg_engine, "SELECT string_agg(version, ',') FROM schema_migrations") == "001_ok"
        )

    def test_an_edited_applied_migration_is_refused_on_postgres(self, pg_engine, tmp_path):
        folder = tmp_path / "copy"
        shutil.copytree(MIGRATIONS_DIR, folder)
        migrator = Migrator(pg_engine, folder)
        migrator.apply()
        with (folder / "001_init.sql").open("a", encoding="utf-8") as handle:
            handle.write("-- μια αλλαγή μετά την εφαρμογή\n")
        assert [s.state for s in migrator.status()] == ["changed", "applied"]
        with pytest.raises(
            MigrationConsistencyError, match="001_init.*changed after it was applied"
        ):
            migrator.apply()

    def test_two_runners_at_the_same_time_apply_each_migration_exactly_once(self, pg_url):
        engines = [get_engine(pg_url), get_engine(pg_url)]
        barrier = threading.Barrier(2)
        applied: list[list[str]] = []
        errors: list[BaseException] = []

        def worker(engine):
            try:
                barrier.wait(timeout=30)
                applied.append(Migrator(engine).apply())
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(engine,)) for engine in engines]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert not errors, errors
        assert sorted(version for result in applied for version in result) == [
            "001_init",
            "002_row_level_security",
        ]
        assert scalar(engines[0], "SELECT count(*) FROM schema_migrations") == 2
        for engine in engines:
            engine.dispose()

    def test_the_supabase_roles_lose_all_access(self, postgres_server, pg_engine):
        """Μιμείται το Supabase: οι ρόλοι anon και authenticated παίρνουν δικαιώματα σε κάθε νέο
        πίνακα. Μετά τα migrations δεν έχουν κανένα, και το RLS δίνει «deny-all» ακόμη κι αν τους
        ξαναδοθεί δικαίωμα. Μόνο στον ενσωματωμένο server: οι ρόλοι ανήκουν στο cluster."""
        if not postgres_server.embedded:
            pytest.skip(
                "δημιουργεί ρόλους σε επίπεδο server: μόνο στον προσωρινό ενσωματωμένο server"
            )
        admin = create_engine(postgres_server.admin_url, isolation_level="AUTOCOMMIT")
        roles = ("anon", "authenticated")
        with admin.connect() as connection:
            for role in roles:
                connection.execute(text(f"CREATE ROLE {role} NOLOGIN"))
        try:
            with pg_engine.begin() as connection:
                for kind in ("TABLES", "SEQUENCES"):  # όπως ορίζει το Supabase στο public
                    connection.execute(
                        text(
                            "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
                            f"GRANT ALL ON {kind} TO anon, authenticated"
                        )
                    )
            Migrator(pg_engine).apply()
            privileges = (
                "SELECT",
                "INSERT",
                "UPDATE",
                "DELETE",
                "TRUNCATE",
                "REFERENCES",
                "TRIGGER",
            )
            with pg_engine.connect() as connection:
                for role in roles:
                    for table in sorted(ALL_TABLES | {"schema_migrations"}):
                        for privilege in privileges:
                            allowed = connection.execute(
                                text("SELECT has_table_privilege(:role, :table, :privilege)"),
                                {"role": role, "table": table, "privilege": privilege},
                            ).scalar_one()
                            assert not allowed, f"{role} {privilege} {table}"
                    usage = connection.execute(
                        text("SELECT has_sequence_privilege(:role, 'predictions_id_seq', 'USAGE')"),
                        {"role": role},
                    ).scalar_one()
                    assert not usage
            with pg_engine.begin() as connection:
                connection.execute(text("INSERT INTO teams VALUES ('AAA', 'Alpha')"))
                connection.execute(
                    text("GRANT SELECT ON teams TO anon")
                )  # κάποιος ξαναδίνει δικαίωμα
            with pg_engine.connect() as connection:
                connection.execute(text("SET ROLE anon"))
                assert (
                    connection.execute(text("SELECT count(*) FROM teams")).scalar_one() == 0
                )  # RLS
                connection.execute(text("RESET ROLE"))
            with pg_engine.begin() as connection:
                connection.execute(text("REVOKE SELECT ON teams FROM anon"))
            with (
                pytest.raises(ProgrammingError, match="permission denied"),
                pg_engine.connect() as connection,
            ):
                connection.execute(text("SET ROLE authenticated"))
                connection.execute(text("SELECT * FROM teams"))
            # ο ρόλος της εφαρμογής (ιδιοκτήτης) παρακάμπτει το RLS και βλέπει τα δεδομένα
            assert count(pg_engine, models.teams) == 1
        finally:
            with pg_engine.begin() as connection:
                connection.execute(text("DROP OWNED BY anon, authenticated"))
            with admin.connect() as connection:
                connection.execute(text("DROP ROLE IF EXISTS anon, authenticated"))
            admin.dispose()


# ----------------------------------------------------------------------------------------------
# Μεταφορά SQLite → Postgres
# ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sqlite_source(tmp_path_factory, synthetic_league):
    path = tmp_path_factory.mktemp("pg_transfer") / "source.db"
    engine = get_engine(f"sqlite:///{path.as_posix()}")
    write_to_database(engine, synthetic_league)
    add_recorded_rows(engine, synthetic_league)
    engine.dispose()
    return path


class TestTransferToPostgres:
    @pytest.fixture
    def source_url(self, sqlite_source):
        return f"sqlite:///{sqlite_source.as_posix()}"

    def test_a_full_transfer_is_verified_and_survives_a_non_utc_session(
        self, source_url, pg_migrated, pg_url, sqlite_source, capsys
    ):
        before = sqlite_source.read_bytes()
        assert transfer.main(["--from", source_url, "--to", with_query(pg_url, **TOKYO)]) == 0
        out = capsys.readouterr().out
        assert "VERIFICATION OK" in out and "MISMATCH" not in out
        assert sqlite_source.read_bytes() == before  # η πηγή δεν άλλαξε
        source = get_engine(source_url)
        assert counts(pg_migrated) == counts(source)
        source.dispose()
        # οι χρονικές στιγμές είναι σωστές όταν διαβαστούν με συνεδρία UTC: η εγγραφή έγινε με
        # συνεδρία +09:00, άρα μια naive ώρα θα είχε μετατοπιστεί κατά 9 ώρες
        reader = utc_engine(pg_url)
        assert scalar(reader, "SELECT created_at FROM predictions ORDER BY id LIMIT 1") == datetime(
            2026, 10, 3, 12, 30, 15, 123456, tzinfo=UTC
        )
        assert scalar(reader, "SELECT updated_at FROM player_availability") == datetime(
            2026, 10, 3, 9, 0, tzinfo=UTC
        )
        reader.dispose()
        assert (
            scalar(pg_migrated, "SELECT name FROM players WHERE player_id = 'P999999'")
            == "ĐORĐEVIĆ, ΓΙΩΡΓΟΣ ñ"
        )
        assert scalar(pg_migrated, "SELECT note FROM player_availability") == "ankle, Γειά σου"

    def test_repeating_is_idempotent_and_the_serial_sequence_stays_usable(
        self, source_url, pg_migrated, pg_url, capsys
    ):
        assert transfer.main(["--from", source_url, "--to", pg_url]) == 0
        first = counts(pg_migrated)
        assert transfer.main(["--from", source_url, "--to", pg_url]) == 0
        assert counts(pg_migrated) == first
        assert transfer.main(["--from", source_url, "--to", pg_url, "--verify-only"]) == 0
        capsys.readouterr()
        # τα ids των predictions δεν αντιγράφηκαν (δίνει δικά του ο προορισμός) και η ακολουθία
        # συνεχίζει χωρίς σύγκρουση. Το `INSERT ... ON CONFLICT` καταναλώνει τιμές της ακολουθίας
        # ακόμη και για γραμμές που ενημερώνονται (σύμπτωμα του Postgres): τα ids έχουν κενά,
        # δεν επαναλαμβάνονται όμως ποτέ.
        with pg_migrated.connect() as connection:
            ids = connection.execute(text("SELECT id FROM predictions ORDER BY id")).scalars().all()
        assert ids == [1, 2, 3]  # οι γραμμές ενημερώθηκαν, δεν αντικαταστάθηκαν
        with pg_migrated.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO predictions (player_id, season, gamecode, predicted_fantasy, "
                    "model_version, as_of) SELECT player_id, season, gamecode, 1.0, 'another', "
                    "as_of FROM predictions WHERE id = 1"
                )
            )
        assert count(pg_migrated, models.predictions) == 4
        assert scalar(pg_migrated, "SELECT max(id) FROM predictions") > 3

    def test_dry_run_writes_nothing(self, source_url, pg_migrated, pg_url, capsys):
        assert transfer.main(["--from", source_url, "--to", pg_url, "--dry-run"]) == 0
        assert sum(counts(pg_migrated).values()) == 0
        assert "Dry run" in capsys.readouterr().out

    def test_tampering_with_the_target_is_detected(self, source_url, pg_migrated, pg_url, capsys):
        assert transfer.main(["--from", source_url, "--to", pg_url]) == 0
        with pg_migrated.begin() as connection:
            connection.execute(
                text(
                    "UPDATE player_games SET minutes = minutes + 0.0000001 "
                    "WHERE ctid = (SELECT min(ctid) FROM player_games)"
                )
            )
        capsys.readouterr()
        assert transfer.main(["--from", source_url, "--to", pg_url, "--verify-only"]) == 1
        assert "VERIFICATION FAILED" in capsys.readouterr().out
        with pg_migrated.begin() as connection:
            connection.execute(text("INSERT INTO teams VALUES ('ZZZ', 'EXTRA')"))
        assert (
            transfer.main(
                ["--from", source_url, "--to", pg_url, "--tables", "teams", "--verify-only"]
            )
            == 1
        )

    def test_children_before_parents_are_rejected_by_postgres(
        self, source_url, pg_migrated, pg_url, capsys
    ):
        assert transfer.main(["--from", source_url, "--to", pg_url, "--tables", "games"]) == 1
        assert "parent tables" in capsys.readouterr().out
        assert count(pg_migrated, models.games) == 0

    def test_an_unmigrated_target_is_refused_with_a_pointer_to_the_migrations(
        self, source_url, pg_url, pg_engine, capsys
    ):
        assert transfer.main(["--from", source_url, "--to", pg_url]) == 1
        out = capsys.readouterr().out
        assert "missing tables" in out and "elfantasy.db.migrate" in out
        assert table_names(pg_engine) == set()  # δεν δημιουργήθηκε τίποτα

    def test_the_same_postgres_database_is_refused(self, pg_url, capsys):
        assert transfer.main(["--from", pg_url, "--to", pg_url]) == 2
        assert "onto itself" in capsys.readouterr().out

    def test_postgres_can_be_the_source_of_a_verification(
        self, source_url, pg_migrated, pg_url, tmp_path
    ):
        assert transfer.main(["--from", source_url, "--to", pg_url]) == 0
        mirror = tmp_path / "mirror.db"
        mirror_url = f"sqlite:///{mirror.as_posix()}"
        mirror_engine = get_engine(mirror_url)
        ensure_schema(mirror_engine)
        mirror_engine.dispose()
        # πίσω από το Postgres σε SQLite (π.χ. αντίγραφο ασφαλείας): ίδια επαλήθευση
        assert transfer.main(["--from", pg_url, "--to", mirror_url, "--allow-non-postgres"]) == 0


# ----------------------------------------------------------------------------------------------
# Ingestion σε Postgres
# ----------------------------------------------------------------------------------------------


class TestPipelineOnPostgres:
    def test_the_pipeline_writes_the_same_data_as_on_sqlite(
        self, fixture_cache, tmp_path, pg_migrated, pg_url
    ):
        data_dir = tmp_path / "data"
        sqlite_url = f"sqlite:///{(tmp_path / 'ref.db').as_posix()}"
        reference = pipeline.run_pipeline(
            SEASONS, db_url=sqlite_url, data_dir=data_dir, fetch=False
        )
        result = pipeline.run_pipeline(SEASONS, db_url=pg_url, data_dir=data_dir, fetch=False)
        assert result.table_counts == reference.table_counts
        assert result.table_counts["player_games"] > 0
        sqlite_engine = get_engine(sqlite_url)
        report = transfer.verify(sqlite_engine, pg_migrated)
        assert report.ok, transfer.format_report(report)  # ίδιο περιεχόμενο, στήλη προς στήλη
        # δεύτερο τρέξιμο (upsert): τίποτα δεν διπλασιάζεται
        again = pipeline.run_pipeline(SEASONS, db_url=pg_url, data_dir=data_dir, fetch=False)
        assert again.table_counts == result.table_counts
        assert transfer.verify(sqlite_engine, pg_migrated).ok
        sqlite_engine.dispose()

    def test_the_verification_tool_gives_the_same_report_on_postgres(
        self, fixture_cache, tmp_path, pg_migrated, pg_url
    ):
        data_dir = tmp_path / "data"
        sqlite_url = f"sqlite:///{(tmp_path / 'ref.db').as_posix()}"
        pipeline.run_pipeline(SEASONS, db_url=sqlite_url, data_dir=data_dir, fetch=False)
        pipeline.run_pipeline(SEASONS, db_url=pg_url, data_dir=data_dir, fetch=False)
        sqlite_engine = get_engine(sqlite_url)
        expected = ingest_verify.verify(sqlite_engine, tmp_path / "r1", expected_games={})
        actual = ingest_verify.verify(pg_migrated, tmp_path / "r2", expected_games={})
        assert actual.failures == expected.failures
        assert (actual.rows, actual.pir_matches, actual.dnp_rows) == (
            expected.rows,
            expected.pir_matches,
            expected.dnp_rows,
        )
        assert actual.season_summary.equals(expected.season_summary)
        sqlite_engine.dispose()

    def test_the_command_line_writes_to_postgres_with_the_db_option(
        self, fixture_cache, tmp_path, pg_migrated, pg_url, restore_logging
    ):
        arguments = [
            "--seasons",
            "2016,2023,2024,2025,2026",
            "--no-fetch",
            "--db",
            pg_url,
            "--data-dir",
            str(tmp_path / "data"),
        ]
        assert pipeline.main(arguments) == 0
        assert count(pg_migrated, models.player_games) > 0

    def test_the_pipeline_never_creates_tables_in_an_unmigrated_postgres(
        self, fixture_cache, tmp_path, pg_engine, pg_url
    ):
        with pytest.raises(SchemaNotInitialisedError, match="elfantasy.db.migrate"):
            pipeline.run_pipeline(SEASONS, db_url=pg_url, data_dir=tmp_path / "data", fetch=False)
        assert table_names(pg_engine) == set()

    def test_the_pipeline_accepts_a_database_without_schema_migrations(
        self, fixture_cache, tmp_path, pg_engine, pg_url
    ):
        """Πίνακες που υπάρχουν (π.χ. από άλλο εργαλείο) αρκούν: ελέγχεται μόνο η ύπαρξή τους."""
        Migrator(pg_engine).apply()
        with pg_engine.begin() as connection:
            connection.execute(text("DROP TABLE schema_migrations"))
        assert ensure_schema(pg_engine) == "existing"
        result = pipeline.run_pipeline(
            SEASONS, db_url=pg_url, data_dir=tmp_path / "data", fetch=False
        )
        assert result.table_counts["games"] > 0


# ----------------------------------------------------------------------------------------------
# Predictor και API σε Postgres
# ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_league_engine(pg_league_url):
    engine = get_engine(pg_league_url)
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def pg_predictor(pg_league_engine, tiny_xgb_bundle, api_today):
    return Predictor(tiny_xgb_bundle, pg_league_engine, today=lambda: api_today)


@pytest.fixture
def pg_client(pg_league_engine, pg_predictor, api_clock):
    from fastapi.testclient import TestClient

    from elfantasy.api.main import create_app

    app = create_app(
        settings=make_settings(), engine=pg_league_engine, predictor=pg_predictor, clock=api_clock
    )
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    with pg_league_engine.begin() as connection:
        connection.execute(delete(models.player_availability))


class TestPredictorAndApiOnPostgres:
    def test_the_predictor_gives_the_same_predictions_as_on_sqlite(
        self, pg_predictor, api_predictor, api_today
    ):
        postgres = pg_predictor.predict_all(as_of=api_today, active_only=False)
        sqlite = api_predictor.predict_all(as_of=api_today, active_only=False)
        assert [p.player_id for p in postgres] == [p.player_id for p in sqlite]
        for first, second in zip(postgres, sqlite, strict=True):
            assert first.predicted_fantasy == pytest.approx(second.predicted_fantasy, abs=1e-9)
            assert first.predicted_pir == pytest.approx(second.predicted_pir, abs=1e-9)
            assert first.next_game == second.next_game
            assert first.n_prior_appearances == second.n_prior_appearances

    def test_health_and_rankings_match_the_sqlite_api(self, pg_client, client):
        health = pg_client.get("/health")
        assert health.status_code == 200 and health.json()["status"] == "ok"
        assert health.json()["database"] == client.get("/health").json()["database"]
        query = "/rankings?limit=500&active_only=false&include_unavailable=true"
        assert pg_client.get(query).json() == client.get(query).json()

    def test_a_prediction_endpoint(self, pg_client, client):
        player = client.get("/rankings?limit=1").json()["items"][0]["player_id"]
        assert pg_client.get(f"/predict/{player}").json() == client.get(f"/predict/{player}").json()
        assert pg_client.get("/predict/P998877").status_code == 404

    def test_availability_roundtrips_with_timestamptz(self, pg_client, api_clock):
        player = pg_client.get("/rankings?limit=1").json()["items"][0]["player_id"]
        body = {
            "player_id": player,
            "status": "out",
            "source": "test",
            "note": "Γειά σου",
            "expected_return": "2026-10-20",
        }
        created = pg_client.post("/availability", json=body, headers=ADMIN_HEADERS)
        assert created.status_code == 200
        assert created.json()["updated_at"].startswith(
            api_clock.moment.strftime("%Y-%m-%dT%H:%M:%S")
        )
        assert created.json()["updated_at"].endswith("Z")
        listed = pg_client.get("/availability").json()
        assert [item["player_id"] for item in listed["items"]] == [player]
        assert pg_client.get(f"/predict/{player}").json()["predicted_fantasy"] == 0.0
        assert player not in [
            item["player_id"] for item in pg_client.get("/rankings?limit=500").json()["items"]
        ]
        # αντικατάσταση (upsert) και διαγραφή
        body["status"] = "doubtful"
        assert pg_client.post("/availability", json=body, headers=ADMIN_HEADERS).status_code == 200
        assert len(pg_client.get("/availability").json()["items"]) == 1
        assert pg_client.delete(f"/availability/{player}", headers=ADMIN_HEADERS).status_code == 204
        assert pg_client.get("/availability").json()["items"] == []

    def test_refresh_reloads_the_data_from_postgres(self, pg_client):
        response = pg_client.post("/admin/refresh", headers=ADMIN_HEADERS)
        assert response.status_code == 200 and response.json()["status"] == "refreshed"

    def test_get_requests_do_not_write_predictions(self, pg_client, pg_league_engine):
        before = count(pg_league_engine, models.predictions)
        for path in ("/health", "/rankings", "/players?search=player", "/availability"):
            assert pg_client.get(path).status_code == 200
        player = pg_client.get("/rankings?limit=1").json()["items"][0]["player_id"]
        assert pg_client.get(f"/predict/{player}").status_code == 200
        assert count(pg_league_engine, models.predictions) == before == 0

    def test_an_unmigrated_database_makes_the_app_degraded_and_creates_nothing(
        self, pg_engine, tiny_xgb_bundle, api_clock, api_today
    ):
        from fastapi.testclient import TestClient

        from elfantasy.api.main import create_app

        app = create_app(settings=make_settings(), engine=pg_engine, clock=api_clock)
        with TestClient(app, raise_server_exceptions=False) as test_client:
            health = test_client.get("/health")
            assert health.status_code == 503
            assert any("database" in problem for problem in health.json()["problems"])
            assert test_client.get("/rankings").status_code == 503
        assert table_names(pg_engine) == set()  # το startup δεν δημιούργησε τον πίνακα availability


# ----------------------------------------------------------------------------------------------
# Καταγραφή και αξιολόγηση προβλέψεων σε Postgres
# ----------------------------------------------------------------------------------------------


class TestRecordedPredictionsOnPostgres:
    @pytest.fixture
    def league(self, pg_migrated, synthetic_league, tiny_xgb_bundle, api_today):
        write_to_database(pg_migrated, synthetic_league)
        predictor = Predictor(tiny_xgb_bundle, pg_migrated, today=lambda: api_today)
        return pg_migrated, predictor

    def test_recording_is_idempotent_and_uses_timestamptz(self, league, api_today):
        engine, predictor = league
        first = recorder.record_predictions(
            engine, predictor, as_of=api_today, now=lambda: datetime(2026, 10, 3, 12, tzinfo=UTC)
        )
        ids = scalar(engine, "SELECT string_agg(id::text, ',' ORDER BY id) FROM predictions")
        second = recorder.record_predictions(
            engine, predictor, as_of=api_today, now=lambda: datetime(2026, 10, 3, 17, tzinfo=UTC)
        )
        assert first.recorded == second.recorded == count(engine, models.predictions) > 0
        assert (
            scalar(engine, "SELECT string_agg(id::text, ',' ORDER BY id) FROM predictions") == ids
        )
        reader = utc_engine(engine.url.render_as_string(hide_password=False))
        assert scalar(reader, "SELECT min(created_at) FROM predictions") == datetime(
            2026, 10, 3, 17, tzinfo=UTC
        )
        reader.dispose()
        # η ίδια πρόβλεψη δεν μπορεί να εισαχθεί δύο φορές (unique constraint του Postgres)
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO predictions (player_id, season, gamecode, predicted_fantasy, "
                    "model_version, as_of) SELECT player_id, season, gamecode, 1.0, "
                    "model_version, as_of FROM predictions LIMIT 1"
                )
            )

    def test_the_command_line_and_the_evaluation(self, league, api_today, monkeypatch, capsys):
        engine, predictor = league
        url = engine.url.render_as_string(hide_password=False)
        monkeypatch.setattr(Predictor, "load", classmethod(lambda cls, *a, **k: predictor))
        assert recorder.main(["--db", url, "--as-of", api_today.isoformat()]) == 0
        assert "Recorded" in capsys.readouterr().out
        recorded = count(engine, models.predictions)
        assert recorded > 0
        # δεν έχει παιχτεί κανένας αγώνας: τίποτα προς αξιολόγηση
        assert evaluator.main(["--db", url]) == 0
        assert "Nothing to evaluate yet" in capsys.readouterr().out
        # ο πρώτος αγώνας παίζεται: ένας παίκτης αγωνίστηκε και ένας έχει γραμμή DNP
        season, gamecode = busiest_game(engine)
        players = sorted(row["player_id"] for row in predicted_for_game(engine, season, gamecode))
        assert len(players) >= 2
        play_game(
            engine, predictor, season, gamecode, appeared={players[0]: 15.0}, dnp=[players[1]]
        )
        result = evaluator.evaluate_recorded(engine)
        day = result.daily.iloc[0]
        assert (day["appeared"], day["dnp"]) == (1, 1)
        assert day["not_in_boxscore"] == len(players) - 2
        assert day["mae"] >= 0 and result.pending == recorded - len(players)
        assert evaluator.main(["--db", url]) == 0
        assert "Per game day" in capsys.readouterr().out


# ----------------------------------------------------------------------------------------------
# Engine και σύνδεση
# ----------------------------------------------------------------------------------------------


class TestEngineOnPostgres:
    def test_a_connection_works_and_the_pool_settings_are_applied(self, pg_engine):
        assert scalar(pg_engine, "SELECT 1") == 1
        assert pg_engine.pool.size() == 5 and pg_engine.pool._max_overflow == 5
        assert pg_engine.pool._recycle == 300 and pg_engine.pool._pre_ping is True

    def test_pre_ping_replaces_a_connection_that_the_server_closed(
        self, postgres_server, pg_engine
    ):
        """Ο pooler ή ο firewall κόβει συνδέσεις σε αδράνεια: το pool_pre_ping τις αντικαθιστά
        αντί να αποτύχει το επόμενο αίτημα."""
        with pg_engine.connect() as connection:
            pid = connection.execute(text("SELECT pg_backend_pid()")).scalar_one()
        admin = create_engine(postgres_server.admin_url, isolation_level="AUTOCOMMIT")
        with admin.connect() as connection:
            assert connection.execute(
                text("SELECT pg_terminate_backend(:pid)"), {"pid": pid}
            ).scalar_one()
        admin.dispose()
        assert scalar(pg_engine, "SELECT 1") == 1  # χωρίς σφάλμα: νέα σύνδεση
        assert scalar(pg_engine, "SELECT pg_backend_pid()") != pid

    @pytest.mark.parametrize(
        "parameter",
        ["prepare_threshold=none", "prepare_threshold=0", "sslmode=disable", "sslmode=prefer"],
    )
    def test_connection_parameters_of_the_url_are_accepted(self, pg_url, parameter):
        engine = get_engine(with_query(pg_url, **dict([parameter.split("=")])))
        assert scalar(engine, "SELECT 'ok'") == "ok"
        engine.dispose()

    def test_prepared_statements_are_disabled_with_none(self, pg_url):
        engine = get_engine(with_query(pg_url, prepare_threshold="none"))
        with engine.connect() as connection:
            for _ in range(8):  # πάνω από το προεπιλεγμένο όριο 5: χωρίς none θα προετοιμαζόταν
                connection.execute(text("SELECT 1 + 1"))
            prepared = connection.execute(
                text("SELECT count(*) FROM pg_prepared_statements")
            ).scalar_one()
        assert prepared == 0
        engine.dispose()

    def test_floats_are_read_back_exactly_even_if_the_server_truncates_them(
        self, pg_migrated, pg_url
    ):
        """Το Supabase έχει extra_float_digits = 0 στον server: ένα απλό engine διαβάζει τα
        `double precision` με 15 ψηφία. Το `get_engine` ζητά 3 σε κάθε σύνδεση."""
        from sqlalchemy.engine import make_url

        from elfantasy.db.urls import normalize_database_url

        value = 19.983333333333334  # 1199 δευτερόλεπτα / 60: χρειάζεται 17 σημαντικά ψηφία
        with pg_migrated.begin() as connection:
            database = connection.execute(text("SELECT current_database()")).scalar_one()
            connection.execute(text(f'ALTER DATABASE "{database}" SET extra_float_digits = 0'))
            connection.execute(text("INSERT INTO teams VALUES ('AAA', 'A'), ('BBB', 'B')"))
            connection.execute(text("INSERT INTO players VALUES ('P1', 'X', 2026, 2026)"))
            connection.execute(
                text(
                    "INSERT INTO games (season, gamecode, game_date, home_code, away_code, played)"
                    " VALUES (2026, 1, '2026-10-01', 'AAA', 'BBB', true)"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO player_games (season, gamecode, player_id, team_code, opp_code,"
                    " home, is_starter, minutes, dnp, points, fg2_made, fg2_attempted, fg3_made,"
                    " fg3_attempted, ft_made, ft_attempted, off_reb, def_reb, total_reb, assists,"
                    " steals, turnovers, blocks_favour, blocks_against, fouls_committed,"
                    " fouls_received, valuation, pir, won, fantasy_score)"
                    " VALUES (2026, 1, 'P1', 'AAA', 'BBB', true, true, :minutes, false,"
                    " 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, true, 0)"
                ),
                {"minutes": value},
            )
        query = "SELECT minutes FROM player_games WHERE player_id = 'P1'"

        control = create_engine(make_url(normalize_database_url(pg_url)))  # χωρίς τον listener
        assert scalar(control, query) == 19.9833333333333  # το πρόβλημα: 15 ψηφία
        control.dispose()

        engine = get_engine(pg_url)
        assert scalar(engine, query) == value  # η λύση: πλήρης ακρίβεια
        assert scalar(engine, "SHOW extra_float_digits") == "3"
        engine.dispose()

    def test_the_session_timezone_does_not_matter_for_the_project_data(self, pg_migrated, pg_url):
        """Το σχήμα και ο κώδικας δεν εξαρτώνται από τη ζώνη ώρας του server."""
        engine = get_engine(with_query(pg_url, **TOKYO))
        assert scalar(engine, "SHOW TimeZone") == "Asia/Tokyo"
        with engine.begin() as connection:
            connection.execute(text("INSERT INTO teams VALUES ('AAA', 'A')"))
            connection.execute(text("INSERT INTO players VALUES ('P1', 'X', 2026, 2026)"))
        from elfantasy.api.availability import (
            AvailabilityStatus,
            get_availability,
            save_availability,
        )

        saved = save_availability(
            engine,
            player_id="P1",
            status=AvailabilityStatus.OUT,
            source=None,
            note=None,
            expected_return=date(2026, 10, 20),
            updated_at=datetime(2026, 10, 3, 12, tzinfo=UTC),
        )
        assert saved.updated_at == datetime(2026, 10, 3, 12, tzinfo=UTC)
        assert get_availability(engine, "P1").updated_at == datetime(2026, 10, 3, 12, tzinfo=UTC)
        engine.dispose()
