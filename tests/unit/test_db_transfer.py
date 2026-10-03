"""Tests του db/transfer.py: μεταφορά με upsert, επαλήθευση, προστασίες και χειρισμός τύπων.

Όλα τρέχουν offline με SQLite → SQLite (`--allow-non-postgres`) πάνω στο συνθετικό πρωτάθλημα·
η μεταφορά προς πραγματικό Postgres δοκιμάζεται στο tests/integration/test_postgres.py.
"""

import hashlib
import logging
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from recorded_support import make_legacy_predictions
from sqlalchemy import func, select, text
from sqlalchemy.exc import OperationalError
from synthetic_league import add_recorded_rows, write_to_database

from elfantasy.db import models, transfer
from elfantasy.db.session import create_all, get_engine
from elfantasy.db.transfer import (
    TransferError,
    coercer_for,
    conflict_columns,
    format_report,
    open_source_engine,
    select_tables,
    table_digest,
    transfer_columns,
    values_match,
    verify,
)

SECRET = "S3cr3t-Pa55"
ALL_TABLES = [table.name for table in models.TABLE_ORDER]


def sqlite_url(path) -> str:
    return f"sqlite:///{path.as_posix()}"


def counts(engine) -> dict[str, int]:
    with engine.connect() as connection:
        return {
            table.name: connection.execute(select(func.count()).select_from(table)).scalar_one()
            for table in models.TABLE_ORDER
        }


def file_hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def source_path(tmp_path_factory, synthetic_league):
    """Βάση SQLite με το συνθετικό πρωτάθλημα ΚΑΙ γραμμές στους `predictions` και
    `player_availability`. Μόνο για ανάγνωση: κανένα test δεν τη γράφει."""
    path = tmp_path_factory.mktemp("transfer_source") / "source.db"
    engine = get_engine(sqlite_url(path))
    write_to_database(engine, synthetic_league)
    add_recorded_rows(engine, synthetic_league)
    engine.dispose()
    return path


@pytest.fixture
def source_url(source_path):
    return sqlite_url(source_path)


@pytest.fixture
def target_path(tmp_path):
    path = tmp_path / "target.db"
    engine = get_engine(sqlite_url(path))
    create_all(engine)
    engine.dispose()
    return path


@pytest.fixture
def target_url(target_path):
    return sqlite_url(target_path)


@pytest.fixture
def engines(source_url, target_url):
    source, target = open_source_engine(source_url), get_engine(target_url)
    yield source, target
    source.dispose()
    target.dispose()


def run(source_url, target_url, *extra) -> int:
    return transfer.main(["--from", source_url, "--to", target_url, "--allow-non-postgres", *extra])


# ----------------------------------------------------------------------------------------------
# Μεταφορά και επαλήθευση
# ----------------------------------------------------------------------------------------------


class TestTransfer:
    def test_a_full_copy_is_identical_and_verified(self, engines):
        source, target = engines
        results = transfer.transfer(source, target)
        assert [r.name for r in results] == ALL_TABLES
        assert counts(target) == counts(source)
        assert all(counts(target)[name] > 0 for name in ALL_TABLES)  # και predictions, availability
        assert [r.written for r in results] == [counts(source)[name] for name in ALL_TABLES]
        report = verify(source, target)
        assert report.ok, format_report(report)

    def test_repeating_the_transfer_is_idempotent(self, engines):
        source, target = engines
        transfer.transfer(source, target)
        first = counts(target)
        transfer.transfer(source, target)
        assert counts(target) == first
        assert verify(source, target).ok

    def test_a_tiny_batch_size_gives_the_same_result(self, engines):
        # 7 γραμμές ανά συναλλαγή: πολλές παρτίδες, με μισογεμάτη την τελευταία
        source, target = engines
        subset = ["teams", "players", "games", "predictions", "player_availability"]
        results = transfer.transfer(source, target, subset, batch_size=7)
        assert {name: counts(target)[name] for name in subset} == {
            name: counts(source)[name] for name in subset
        }
        assert verify(source, target, subset, batch_size=7).ok
        assert sum(r.written for r in results) == sum(counts(source)[name] for name in subset)
        assert counts(source)["games"] % 7 != 0  # η τελευταία παρτίδα είναι μισογεμάτη

    def test_a_batch_size_equal_to_and_larger_than_a_table(self, engines):
        source, target = engines
        size = counts(source)["teams"]
        transfer.transfer(source, target, ["teams"], batch_size=size)  # ακριβώς όσο ο πίνακας
        transfer.transfer(source, target, ["teams", "players"], batch_size=10**6)
        assert verify(source, target, ["teams", "players"]).ok

    def test_rows_changed_in_the_source_are_updated_not_duplicated(
        self, tmp_path, source_path, target_url
    ):
        copy = tmp_path / "copy.db"
        copy.write_bytes(source_path.read_bytes())
        source, target = get_engine(sqlite_url(copy)), get_engine(target_url)
        transfer.transfer(source, target)
        with source.begin() as connection:
            connection.execute(text("UPDATE players SET name = 'RENAMED, PLAYER' WHERE rowid = 1"))
            connection.execute(text("UPDATE predictions SET predicted_fantasy = 99.5 WHERE id = 1"))
        assert not verify(source, target).ok
        transfer.transfer(source, target)
        assert counts(target) == counts(source)
        assert verify(source, target).ok
        with target.connect() as connection:
            assert (
                connection.execute(
                    select(func.count())
                    .select_from(models.players)
                    .where(models.players.c.name == "RENAMED, PLAYER")
                ).scalar_one()
                == 1
            )
        source.dispose()
        target.dispose()

    def test_the_tables_option_copies_a_subset_in_foreign_key_order(self, engines):
        source, target = engines
        results = transfer.transfer(source, target, ["players", "teams"])  # σειρά αλλαγμένη
        assert [r.name for r in results] == ["teams", "players"]
        assert counts(target)["games"] == 0
        assert verify(source, target, ["teams", "players"]).ok
        assert not verify(source, target).ok  # οι υπόλοιποι πίνακες λείπουν από τον προορισμό

    def test_children_before_parents_fail_with_a_clear_message(self, engines):
        source, target = engines
        with pytest.raises(TransferError, match="parent tables"):
            transfer.transfer(source, target, ["games"])
        assert counts(target)["games"] == 0

    def test_other_integrity_errors_get_no_foreign_key_hint_and_no_row_values(
        self, tmp_path, target_url
    ):
        # πηγή χωρίς NOT NULL: μια γραμμή χωρίς όνομα απορρίπτεται από τον προορισμό
        loose = get_engine(sqlite_url(tmp_path / "loose.db"))
        with loose.begin() as connection:
            connection.execute(
                text("CREATE TABLE teams (team_code VARCHAR PRIMARY KEY, name VARCHAR)")
            )
            connection.execute(text("INSERT INTO teams VALUES ('SECRETCODE', NULL)"))
        target = get_engine(target_url)
        with pytest.raises(TransferError) as excinfo:
            transfer.transfer(loose, target, ["teams"])
        message = str(excinfo.value)
        assert "teams: the target rejected a batch (IntegrityError)" in message
        assert "parent tables" not in message and "SECRETCODE" not in message
        loose.dispose()
        target.dispose()

    def test_dry_run_writes_nothing(self, engines, caplog):
        source, target = engines
        with caplog.at_level(logging.INFO, logger="elfantasy.db.transfer"):
            results = transfer.transfer(source, target, dry_run=True)
        assert sum(counts(target).values()) == 0
        assert all(r.written == 0 for r in results)
        assert [r.source_rows for r in results if r.source_rows] == [
            counts(source)[name] for name in ALL_TABLES
        ]
        assert "would copy" in caplog.text

    def test_progress_is_logged_per_table(self, engines, caplog):
        source, target = engines
        with caplog.at_level(logging.INFO, logger="elfantasy.db.transfer"):
            transfer.transfer(source, target, batch_size=500)
        assert "player_games: copying" in caplog.text
        assert "rows (100%)" in caplog.text

    def test_the_target_must_have_the_tables(self, source_url, tmp_path):
        empty = get_engine(sqlite_url(tmp_path / "empty.db"))
        source = open_source_engine(source_url)
        with pytest.raises(TransferError, match="missing tables.*python -m elfantasy.db.migrate"):
            transfer.transfer(source, empty)
        source.dispose()
        empty.dispose()

    def test_invalid_options(self, engines):
        source, target = engines
        with pytest.raises(TransferError, match="batch size"):
            transfer.transfer(source, target, batch_size=0)
        with pytest.raises(TransferError, match="unknown table.*bogus"):
            transfer.transfer(source, target, ["bogus"])

    # ----- πηγές με λείποντες πίνακες ή παλιά διάταξη -----
    def test_a_table_missing_from_the_source_is_skipped(self, tmp_path, source_path, target_url):
        copy = tmp_path / "copy.db"
        copy.write_bytes(source_path.read_bytes())
        source, target = get_engine(sqlite_url(copy)), get_engine(target_url)
        with source.begin() as connection:
            connection.execute(text("DROP TABLE player_availability"))
        results = {r.name: r for r in transfer.transfer(source, target)}
        assert "does not exist in the source" in results["player_availability"].note
        assert counts(target)["player_availability"] == 0
        assert verify(source, target).ok  # ο πίνακας συγκρίνεται ως κενός
        source.dispose()
        target.dispose()

    def test_an_empty_legacy_predictions_table_is_skipped(self, tmp_path, target_url):
        legacy = get_engine(sqlite_url(tmp_path / "legacy.db"))
        create_all(legacy)
        make_legacy_predictions(legacy)
        with legacy.begin() as connection:
            connection.execute(models.teams.insert().values(team_code="AAA", name="A"))
        target = get_engine(target_url)
        results = {r.name: r for r in transfer.transfer(legacy, target)}
        assert (
            "old layout" in results["predictions"].note and "as_of" in results["predictions"].note
        )
        assert results["teams"].written == 1
        assert verify(legacy, target).ok
        legacy.dispose()
        target.dispose()

    def test_a_legacy_predictions_table_with_rows_is_an_error(self, tmp_path, target_url):
        legacy = get_engine(sqlite_url(tmp_path / "legacy.db"))
        create_all(legacy)
        with legacy.begin() as connection:
            connection.execute(text("DROP TABLE predictions"))
            connection.execute(
                text(
                    "CREATE TABLE predictions (id INTEGER PRIMARY KEY, player_id VARCHAR,"
                    " predicted_fantasy FLOAT, model_version VARCHAR)"
                )
            )
            connection.execute(text("INSERT INTO predictions VALUES (1, 'P1', 1.0, 'v')"))
        target = get_engine(target_url)
        with pytest.raises(TransferError, match="predictions has 1 rows but lacks the columns"):
            transfer.transfer(legacy, target)
        legacy.dispose()
        target.dispose()


class TestVerify:
    @pytest.fixture
    def copied(self, engines):
        source, target = engines
        transfer.transfer(source, target)
        return source, target

    @staticmethod
    def failures(report):
        return {(c.table, c.metric) for c in report.failures}

    def test_a_changed_value_is_detected_by_the_aggregates_and_the_digest(self, copied):
        source, target = copied
        with target.begin() as connection:
            connection.execute(text("UPDATE player_games SET pir = pir + 1 WHERE rowid = 5"))
        report = verify(source, target)
        assert not report.ok
        assert ("player_games", "sum(pir)") in self.failures(report)
        assert ("player_games", "content digest") in self.failures(report)

    def test_a_change_too_small_for_the_aggregates_is_caught_by_the_digest(self, copied):
        source, target = copied
        with target.begin() as connection:
            connection.execute(
                text("UPDATE player_games SET minutes = minutes + 0.0000001 WHERE rowid = 5")
            )
        report = verify(source, target)
        assert self.failures(report) == {("player_games", "content digest")}

    def test_a_text_change_is_caught_only_by_the_digest(self, copied):
        source, target = copied
        with target.begin() as connection:
            connection.execute(text("UPDATE players SET name = name || '!' WHERE rowid = 2"))
        assert self.failures(verify(source, target)) == {("players", "content digest")}

    def test_a_missing_row_is_detected(self, copied):
        source, target = copied
        with target.begin() as connection:
            connection.execute(text("DELETE FROM player_games WHERE rowid = 5"))
        failures = self.failures(verify(source, target))
        assert ("player_games", "count") in failures and (
            "player_games",
            "content digest",
        ) in failures

    def test_an_extra_row_in_the_target_is_detected(self, copied):
        source, target = copied
        with target.begin() as connection:
            connection.execute(models.teams.insert().values(team_code="ZZZ", name="EXTRA"))
        assert ("teams", "count") in self.failures(verify(source, target))

    def test_per_season_points_are_compared(self, copied):
        source, target = copied
        with target.begin() as connection:
            connection.execute(text("UPDATE player_games SET points = points + 1 WHERE rowid = 5"))
        failures = self.failures(verify(source, target))
        assert any(metric.startswith("sum(points) season=") for _, metric in failures)

    def test_the_availability_status_counts_are_compared(self, copied):
        source, target = copied
        with target.begin() as connection:
            connection.execute(text("UPDATE player_availability SET status = 'out'"))
        assert ("player_availability", "count status=doubtful") in self.failures(
            verify(source, target)
        )

    def test_the_digest_can_be_skipped(self, copied):
        source, target = copied
        report = verify(source, target, row_digest=False)
        assert report.ok and not [c for c in report.checks if c.metric == "content digest"]

    def test_the_report_table_lists_every_check(self, copied):
        source, target = copied
        with target.begin() as connection:
            connection.execute(text("UPDATE player_games SET pir = pir + 1 WHERE rowid = 5"))
        text_ = format_report(verify(source, target))
        lines = text_.splitlines()
        assert lines[0].split() == ["table", "check", "source", "target", "result"]
        assert "MISMATCH" in text_ and "VERIFICATION FAILED" in lines[-1]
        assert any("player_games" in line and "sum(pir)" in line for line in lines)

    def test_a_clean_report_says_ok(self, copied):
        source, target = copied
        report = verify(source, target)
        assert (
            format_report(report).splitlines()[-1]
            == f"VERIFICATION OK ({len(report.checks)} checks)"
        )

    def test_the_report_has_the_documented_aggregates(self, copied):
        source, target = copied
        metrics = {(c.table, c.metric) for c in verify(source, target).checks}
        for expected in (
            ("teams", "count"),
            ("players", "count"),
            ("games", "min(game_date)"),
            ("games", "max(game_date)"),
            ("games", "played games"),
            ("player_games", "count"),
            ("player_games", "sum(pir)"),
            ("player_games", "sum(fantasy_score)"),
            ("player_games", "content digest"),
            ("predictions", "sum(predicted_fantasy)"),
        ):
            assert expected in metrics
        assert any(m.startswith("sum(points) season=") for _, m in metrics)

    def test_the_counts_of_tables_without_columns_in_the_aggregates_are_real(self, copied):
        # regression: το `SELECT count(*)` χωρίς FROM θα έδινε 1 για τον πίνακα teams
        source, target = copied
        by_name = {(c.table, c.metric): c for c in verify(source, target).checks}
        assert by_name[("teams", "count")].source == counts(source)["teams"] > 1
        assert by_name[("player_availability", "count")].target == 1

    def test_a_table_absent_from_the_source_is_compared_as_empty(self, tmp_path, engines):
        _, target = engines
        bare = get_engine(sqlite_url(tmp_path / "bare.db"))
        report = verify(bare, target, ["player_availability"])
        assert report.ok
        with target.begin() as connection:
            connection.execute(models.teams.insert().values(team_code="AAA", name="A"))
            connection.execute(
                models.players.insert().values(
                    player_id="P1", name="X", first_season=1, last_season=1
                )
            )
            connection.execute(
                models.player_availability.insert().values(
                    player_id="P1", status="out", updated_at=datetime(2026, 1, 1, tzinfo=UTC)
                )
            )
        assert not verify(bare, target, ["player_availability"]).ok
        bare.dispose()


class TestDigest:
    @pytest.fixture
    def make(self, tmp_path):
        """Φτιάχνει βάσεις SQLite και τις κλείνει όλες στο τέλος του test."""
        opened = []

        def build(name, team_rows=None, *, with_tables=True):
            engine = get_engine(sqlite_url(tmp_path / name))
            opened.append(engine)
            if with_tables:
                create_all(engine)
            if team_rows:
                with engine.begin() as connection:
                    connection.execute(models.teams.insert(), team_rows)
            return engine

        yield build
        for engine in opened:
            engine.dispose()

    def test_the_digest_does_not_depend_on_row_order(self, make):
        rows = [{"team_code": c, "name": f"Team {c}"} for c in ("AAA", "BBB", "CCC", "DDD")]
        first = make("a.db", rows)
        second = make("b.db", list(reversed(rows)))
        assert table_digest(first, models.teams) == table_digest(second, models.teams)
        assert table_digest(first, models.teams).startswith("4:")

    def test_any_difference_changes_the_digest(self, make):
        rows = [{"team_code": c, "name": f"Team {c}"} for c in ("AAA", "BBB")]
        base = table_digest(make("a.db", rows), models.teams)
        renamed = [rows[0], {**rows[1], "name": "Team BBb"}]
        assert table_digest(make("b.db", renamed), models.teams) != base
        fewer = make("c.db", rows[:1])
        assert table_digest(fewer, models.teams) != base

    def test_an_empty_table(self, make):
        assert table_digest(make("e.db"), models.teams) == "0:0000000000000000"

    def test_null_and_empty_text_are_different(self, make):
        engine = make("n.db", [{"team_code": "AAA", "name": "x"}])
        with engine.begin() as connection:
            connection.execute(
                models.games.insert().values(
                    season=1,
                    gamecode=1,
                    game_date=date(2026, 1, 1),
                    home_code="AAA",
                    away_code="AAA",
                    played=False,
                    phase=None,
                )
            )
        first = table_digest(engine, models.games)
        with engine.begin() as connection:
            connection.execute(text("UPDATE games SET phase = ''"))
        assert table_digest(engine, models.games) != first


class TestValuesMatch:
    @pytest.mark.parametrize(
        ("first", "second", "expected"),
        [
            (None, None, True),
            (None, 0, False),
            (0, None, False),
            (5, 5, True),
            (5, 6, False),
            (564701.9, 564701.9000000001, True),
            (1.0, 1.0000001, True),  # κάτω από την ανοχή 1e-6
            (1.0, 1.001, False),
            (Decimal("2.5"), 2.5, True),
            (date(2026, 1, 1), date(2026, 1, 1), True),
            (date(2026, 1, 1), date(2026, 1, 2), False),
            ("a", "a", True),
            ("a", "b", False),
        ],
    )
    def test_cases(self, first, second, expected):
        assert values_match(first, second) is expected


# ----------------------------------------------------------------------------------------------
# Τύποι και στήλες
# ----------------------------------------------------------------------------------------------


class TestCoercion:
    def test_booleans_from_sqlite_integers(self):
        coerce = coercer_for(models.player_games.c.home)
        assert (coerce(1), coerce(0), coerce(True), coerce(None)) == (True, False, True, None)

    def test_numbers(self):
        assert coercer_for(models.player_games.c.minutes)(5) == 5.0
        assert isinstance(coercer_for(models.player_games.c.minutes)(5), float)
        assert coercer_for(models.player_games.c.plus_minus)(None) is None
        assert coercer_for(models.players.c.first_season)("2016") == 2016
        assert isinstance(coercer_for(models.players.c.first_season)(2016.0), int)

    def test_text(self):
        assert coercer_for(models.teams.c.name)(123) == "123"
        assert coercer_for(models.teams.c.name)(None) is None

    def test_dates(self):
        coerce = coercer_for(models.games.c.game_date)
        assert coerce(datetime(2026, 10, 3, 12)) == date(2026, 10, 3)
        assert coerce("2026-10-03") == date(2026, 10, 3)
        assert coerce(date(2026, 10, 3)) == date(2026, 10, 3)
        assert coerce(None) is None

    def test_a_timestamp_without_zone_is_kept_as_naive_utc(self):
        coerce = coercer_for(models.games.c.tipoff_utc)
        assert coerce(datetime(2026, 10, 7, 18, 45)) == datetime(2026, 10, 7, 18, 45)
        assert coerce("2026-10-07 18:45:00") == datetime(2026, 10, 7, 18, 45)
        aware = datetime(2026, 10, 7, 21, 45, tzinfo=timezone(timedelta(hours=3)))
        assert coerce(aware) == datetime(2026, 10, 7, 18, 45) and coerce(aware).tzinfo is None
        assert coerce(None) is None

    def test_a_timestamptz_always_gets_an_explicit_utc_zone(self):
        """Μια naive ώρα σε στήλη timestamptz θα ερμηνευόταν στη ζώνη του server: γίνεται UTC."""
        coerce = coercer_for(models.predictions.c.created_at)
        naive = coerce(datetime(2026, 10, 3, 12, 30))
        assert naive == datetime(
            2026, 10, 3, 12, 30, tzinfo=UTC
        ) and naive.utcoffset() == timedelta(0)
        aware = coerce(datetime(2026, 10, 3, 15, 30, tzinfo=timezone(timedelta(hours=3))))
        assert aware == datetime(2026, 10, 3, 12, 30, tzinfo=UTC)
        assert coerce(datetime(2026, 10, 3, 12, 30)) == coerce(
            datetime(2026, 10, 3, 15, 30, tzinfo=timezone(timedelta(hours=3)))
        )
        assert coerce(None) is None
        assert coerce(date(2026, 10, 3)) == datetime(2026, 10, 3, tzinfo=UTC)


class TestColumns:
    def test_the_surrogate_id_is_not_copied(self):
        assert "id" not in [c.name for c in transfer_columns(models.predictions)]
        assert len(transfer_columns(models.predictions)) == len(models.predictions.columns) - 1
        for table in models.TABLE_ORDER:
            if table is not models.predictions:
                assert len(transfer_columns(table)) == len(table.columns)

    def test_conflict_columns(self):
        assert conflict_columns(models.teams) == ["team_code"]
        assert conflict_columns(models.games) == ["season", "gamecode"]
        assert conflict_columns(models.player_games) == ["season", "gamecode", "player_id"]
        assert conflict_columns(models.predictions) == [
            "player_id",
            "season",
            "gamecode",
            "model_version",
            "as_of",
        ]

    def test_table_selection_is_always_in_foreign_key_order(self):
        assert [t.name for t in select_tables(None)] == ALL_TABLES
        assert [t.name for t in select_tables(["games", "teams"])] == ["teams", "games"]
        with pytest.raises(TransferError, match="unknown table"):
            select_tables(["nope"])


# ----------------------------------------------------------------------------------------------
# Η πηγή ανοίγει μόνο για ανάγνωση
# ----------------------------------------------------------------------------------------------


class TestSourceIsReadOnly:
    def test_the_source_cannot_be_written(self, source_url):
        source = open_source_engine(source_url)
        with pytest.raises(OperationalError, match="readonly"), source.begin() as connection:
            connection.execute(models.teams.insert().values(team_code="ZZZ", name="nope"))
        source.dispose()

    def test_a_full_transfer_leaves_the_source_file_byte_identical(self, source_path, target_url):
        before = file_hash(source_path)
        assert run(sqlite_url(source_path), target_url) == 0
        assert file_hash(source_path) == before
        assert not list(source_path.parent.glob("*-journal")) and not list(
            source_path.parent.glob("*-wal")
        )

    def test_a_missing_file_is_not_created(self, tmp_path):
        missing = tmp_path / "absent" / "x.db"
        with pytest.raises(TransferError, match="does not exist"):
            open_source_engine(sqlite_url(missing))
        assert not missing.parent.exists()

    def test_non_file_urls_are_opened_normally(self):
        engine = open_source_engine("sqlite://")
        assert engine.dialect.name == "sqlite"
        engine.dispose()


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------


class TestCli:
    def test_a_successful_run(self, source_url, target_url, capsys):
        assert run(source_url, target_url) == 0
        out = capsys.readouterr().out
        assert "VERIFICATION OK" in out
        assert "Copied" in out and "rows/s" in out
        assert "Total time" in out

    def test_a_second_run_is_safe(self, source_url, target_url, target_path):
        assert run(source_url, target_url) == 0
        assert run(source_url, target_url) == 0
        engine = get_engine(target_url)
        assert counts(engine)["player_games"] > 0
        engine.dispose()

    def test_dry_run_exits_0_and_writes_nothing(self, source_url, target_url, capsys):
        assert run(source_url, target_url, "--dry-run") == 0
        out = capsys.readouterr().out
        assert "Dry run" in out and "would copy" in out and "VERIFICATION" not in out
        engine = get_engine(target_url)
        assert sum(counts(engine).values()) == 0
        engine.dispose()

    def test_verify_only_fails_before_the_copy_and_passes_after(
        self, source_url, target_url, capsys
    ):
        assert run(source_url, target_url, "--verify-only") == 1
        assert "MISMATCH" in capsys.readouterr().out
        assert run(source_url, target_url) == 0
        capsys.readouterr()
        assert run(source_url, target_url, "--verify-only") == 0
        assert "VERIFICATION OK" in capsys.readouterr().out

    def test_a_mismatch_gives_a_nonzero_exit_code(
        self, source_url, target_url, target_path, capsys
    ):
        assert run(source_url, target_url) == 0
        engine = get_engine(target_url)
        with engine.begin() as connection:
            connection.execute(text("UPDATE player_games SET pir = pir + 1 WHERE rowid = 3"))
        engine.dispose()
        capsys.readouterr()
        assert run(source_url, target_url, "--verify-only") == 1
        assert "VERIFICATION FAILED" in capsys.readouterr().out

    def test_no_row_digest(self, source_url, target_url, capsys):
        assert run(source_url, target_url, "--no-row-digest") == 0
        assert "content digest" not in capsys.readouterr().out

    def test_the_tables_option_accepts_commas_and_spaces(self, source_url, target_url):
        assert run(source_url, target_url, "--tables", "teams,players", "games") == 0
        engine = get_engine(target_url)
        assert counts(engine)["teams"] > 0 and counts(engine)["player_games"] == 0
        engine.dispose()

    def test_the_default_target_is_the_database_url_setting(
        self, source_url, target_url, monkeypatch, capsys
    ):
        monkeypatch.setenv("DATABASE_URL", target_url)
        transfer.get_settings.cache_clear()
        assert transfer.main(["--from", source_url, "--allow-non-postgres"]) == 0
        assert "VERIFICATION OK" in capsys.readouterr().out

    # ----- προστασίες -----
    def test_it_refuses_to_copy_a_database_onto_itself(self, source_url, source_path, capsys):
        assert run(source_url, source_url) == 2
        assert "onto itself" in capsys.readouterr().out
        # και με άλλη γραφή της ίδιας διαδρομής
        assert (
            run(source_url, f"sqlite:///{source_path.parent.as_posix()}/./{source_path.name}") == 2
        )

    def test_a_target_that_is_not_postgres_is_refused_without_the_flag(
        self, source_url, target_url, capsys
    ):
        assert transfer.main(["--from", source_url, "--to", target_url]) == 2
        assert "not PostgreSQL" in capsys.readouterr().out
        engine = get_engine(target_url)
        assert sum(counts(engine).values()) == 0
        engine.dispose()

    def test_a_missing_source_is_refused_and_not_created(self, tmp_path, target_url, capsys):
        missing = tmp_path / "nowhere" / "x.db"
        assert run(sqlite_url(missing), target_url) == 2
        assert "does not exist" in capsys.readouterr().out
        assert not missing.parent.exists()

    @pytest.mark.parametrize("extra", [["--tables", "bogus"], ["--batch-size", "0"]])
    def test_invalid_options_exit_with_2(self, source_url, target_url, extra, capsys):
        assert run(source_url, target_url, *extra) == 2

    def test_an_invalid_url_exits_with_2_without_the_secret(self, source_url, capsys):
        assert (
            transfer.main(["--from", source_url, "--to", f"postgresql://u:{SECRET}@h:badport/db"])
            == 2
        )
        assert SECRET not in capsys.readouterr().out

    def test_an_invalid_target_option_closes_the_source_and_exits_with_2(self, source_url, capsys):
        target = f"postgresql://u:{SECRET}@127.0.0.1:1/db?prepare_threshold=oops"
        assert transfer.main(["--from", source_url, "--to", target]) == 2
        out = capsys.readouterr().out
        assert "prepare_threshold" in out and SECRET not in out

    def test_a_target_without_tables_exits_with_1_and_points_to_the_migrations(
        self, source_url, tmp_path, capsys
    ):
        assert run(source_url, sqlite_url(tmp_path / "empty.db")) == 1
        out = capsys.readouterr().out
        assert "missing tables" in out and "elfantasy.db.migrate" in out

    def test_an_unreachable_target_fails_without_leaking_the_password(self, source_url, capsys):
        target = f"postgresql://postgres:{SECRET}@127.0.0.1:1/postgres?connect_timeout=1"
        assert transfer.main(["--from", source_url, "--to", target]) == 1
        out = capsys.readouterr().out
        assert SECRET not in out
        assert "Target: postgresql+psycopg://postgres:***@127.0.0.1:1/postgres" in out

    def test_the_urls_in_the_log_have_hidden_passwords(self, source_url, target_url, caplog):
        with caplog.at_level(logging.INFO, logger="elfantasy.db.transfer"):
            run(source_url, target_url)
        assert "Source:" in caplog.text and "(read-only)" in caplog.text
        assert "Target:" in caplog.text
