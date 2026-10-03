"""Unit tests της λογικής διαθεσιμότητας (`elfantasy.api.availability`) και του πίνακα
`player_availability`, χωρίς HTTP: κανόνες του override, αποθήκευση, ταξινόμηση, constraints,
ταυτόχρονα upserts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta, timezone

import pytest
from api_support import make_small_database
from sqlalchemy import func, insert, inspect, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable

from elfantasy.api import availability as av
from elfantasy.api.availability import (
    NOTE_DOUBTFUL,
    NOTE_OUT,
    NOTE_STALE,
    Availability,
    AvailabilityStatus,
    apply_availability,
    to_utc,
)
from elfantasy.db import models

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)


def record(status, **fields) -> Availability:
    return Availability(player_id="P000001", status=AvailabilityStatus(status), **fields)


class TestApplyAvailability:
    def test_no_record_changes_nothing(self):
        result = apply_availability(12.5, None)
        assert result.fantasy == 12.5 and result.notes == ()

    def test_available_changes_nothing(self):
        result = apply_availability(12.5, record("available", note="cleared"))
        assert result.fantasy == 12.5 and result.notes == ()

    def test_out_gives_zero_with_a_note(self):
        result = apply_availability(12.5, record("out"))
        assert result.fantasy == 0.0
        assert result.notes == (NOTE_OUT,)

    def test_out_is_zero_even_for_a_negative_model_value(self):
        assert apply_availability(-3.2, record("out")).fantasy == 0.0

    def test_doubtful_keeps_the_value_and_warns(self):
        result = apply_availability(12.5, record("doubtful"))
        assert result.fantasy == 12.5
        assert result.notes == (NOTE_DOUBTFUL,)

    @pytest.mark.parametrize("status", ["out", "doubtful"])
    def test_a_past_expected_return_adds_a_stale_warning(self, status):
        result = apply_availability(
            10.0, record(status, expected_return=date(2026, 10, 2)), today=date(2026, 10, 3)
        )
        assert result.notes[-1] == NOTE_STALE
        assert len(result.notes) == 2

    def test_the_stale_warning_needs_a_date_in_the_past(self):
        for expected in (date(2026, 10, 3), date(2026, 10, 4), None):
            result = apply_availability(
                10.0, record("out", expected_return=expected), today=date(2026, 10, 3)
            )
            assert result.notes == (NOTE_OUT,)

    def test_the_stale_warning_is_skipped_without_today_and_for_available(self):
        old = date(2000, 1, 1)
        assert apply_availability(10.0, record("out", expected_return=old)).notes == (NOTE_OUT,)
        assert (
            apply_availability(
                10.0, record("available", expected_return=old), today=date(2026, 10, 3)
            ).notes
            == ()
        )

    def test_the_inputs_are_not_modified(self):
        original = record("out", note="x")
        apply_availability(10.0, original)
        assert original.status is AvailabilityStatus.OUT and original.note == "x"

    def test_the_statuses_are_the_documented_ones(self):
        assert [s.value for s in AvailabilityStatus] == ["out", "doubtful", "available"]
        assert AvailabilityStatus("out") == "out"  # StrEnum


class TestToUtc:
    def test_a_naive_datetime_is_taken_as_utc(self):
        assert to_utc(datetime(2026, 10, 3, 12)) == NOW
        assert to_utc(datetime(2026, 10, 3, 12)).tzinfo is UTC

    def test_an_aware_datetime_is_converted(self):
        moment = datetime(2026, 10, 3, 15, 30, tzinfo=timezone(timedelta(hours=3)))
        assert to_utc(moment) == datetime(2026, 10, 3, 12, 30, tzinfo=UTC)
        assert to_utc(moment).utcoffset() == timedelta(0)

    def test_utc_stays_as_is(self):
        assert to_utc(NOW) == NOW


@pytest.fixture
def engine(tmp_path):
    players = [
        ("P000001", "ALPHA, ANNA"),
        ("P000002", "BRAVO, BEN"),
        ("P000003", "CHARLIE, CARL"),
        ("P000004", "DELTA, DAN"),
    ]
    engine = make_small_database(tmp_path / "availability.db", players)
    yield engine
    engine.dispose()


def save(engine, player_id, status, *, note=None, source=None, expected=None, at=NOW):
    return av.save_availability(
        engine,
        player_id=player_id,
        status=AvailabilityStatus(status),
        source=source,
        note=note,
        expected_return=expected,
        updated_at=at,
    )


class TestRepository:
    def test_save_and_get_roundtrip(self, engine):
        saved = save(
            engine,
            "P000001",
            "out",
            note="ankle",
            source="club",
            expected=date(2026, 10, 20),
        )
        assert saved == Availability(
            player_id="P000001",
            status=AvailabilityStatus.OUT,
            source="club",
            note="ankle",
            expected_return=date(2026, 10, 20),
            updated_at=NOW,
            name="ALPHA, ANNA",
        )
        assert av.get_availability(engine, "P000001") == saved
        assert saved.updated_at.tzinfo is UTC

    def test_a_missing_record_is_none(self, engine):
        assert av.get_availability(engine, "P000001") is None
        assert av.get_availability(engine, "P999999") is None

    def test_a_second_save_replaces_the_whole_record(self, engine):
        save(engine, "P000001", "out", note="ankle", source="club", expected=date(2026, 10, 20))
        later = NOW + timedelta(hours=2)
        second = save(engine, "P000001", "doubtful", at=later)
        assert second.status is AvailabilityStatus.DOUBTFUL
        assert (second.note, second.source, second.expected_return) == (None, None, None)
        assert second.updated_at == later
        assert len(av.list_availability(engine)) == 1

    def test_a_non_utc_updated_at_is_stored_as_utc(self, engine):
        local = datetime(2026, 10, 3, 15, 30, tzinfo=timezone(timedelta(hours=3)))
        assert save(engine, "P000001", "out", at=local).updated_at == (
            datetime(2026, 10, 3, 12, 30, tzinfo=UTC)
        )

    def test_the_list_is_ordered_by_severity_then_name(self, engine):
        save(engine, "P000004", "available")
        save(engine, "P000002", "out")
        save(engine, "P000003", "doubtful")
        save(engine, "P000001", "out")
        assert [(r.player_id, r.status.value) for r in av.list_availability(engine)] == [
            ("P000001", "out"),
            ("P000002", "out"),
            ("P000003", "doubtful"),
            ("P000004", "available"),
        ]

    def test_the_list_can_be_filtered_by_status(self, engine):
        save(engine, "P000001", "out")
        save(engine, "P000002", "doubtful")
        save(engine, "P000003", "out")
        assert [r.player_id for r in av.list_availability(engine, AvailabilityStatus.OUT)] == [
            "P000001",
            "P000003",
        ]
        assert [
            r.player_id for r in av.list_availability(engine, AvailabilityStatus.AVAILABLE)
        ] == []

    def test_availability_by_player(self, engine):
        save(engine, "P000002", "out")
        save(engine, "P000003", "doubtful")
        by_player = av.availability_by_player(engine)
        assert set(by_player) == {"P000002", "P000003"}
        assert by_player["P000002"].status is AvailabilityStatus.OUT

    def test_remove(self, engine):
        save(engine, "P000001", "out")
        assert av.remove_availability(engine, "P000001") is True
        assert av.remove_availability(engine, "P000001") is False
        assert av.get_availability(engine, "P000001") is None

    def test_player_name(self, engine):
        assert av.player_name(engine, "P000002") == "BRAVO, BEN"
        assert av.player_name(engine, "P999999") is None

    def test_ensure_table_creates_a_missing_table_and_is_repeatable(self, tmp_path):
        from elfantasy.db.session import get_engine

        engine = get_engine(f"sqlite:///{(tmp_path / 'bare.db').as_posix()}")
        models.players.create(engine)
        models.teams.create(engine)
        assert "player_availability" not in inspect(engine).get_table_names()
        av.ensure_table(engine)
        av.ensure_table(engine)
        assert "player_availability" in inspect(engine).get_table_names()
        assert av.list_availability(engine) == []
        engine.dispose()


class TestConstraints:
    def test_the_database_rejects_an_unknown_status(self, engine):
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(
                insert(models.player_availability).values(
                    player_id="P000001", status="injured", updated_at=NOW
                )
            )

    def test_the_database_rejects_an_unknown_player(self, engine):
        with pytest.raises(IntegrityError):
            save(engine, "P999999", "out")

    def test_a_missing_status_or_time_is_rejected(self, engine):
        for values in ({"status": "out"}, {"updated_at": NOW}):
            with pytest.raises(IntegrityError), engine.begin() as connection:
                connection.execute(
                    insert(models.player_availability).values(player_id="P000001", **values)
                )

    def test_one_record_per_player(self, engine):
        save(engine, "P000001", "out")
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(
                insert(models.player_availability).values(
                    player_id="P000001", status="out", updated_at=NOW
                )
            )

    def test_the_players_are_not_touched(self, engine):
        save(engine, "P000001", "out")
        with engine.connect() as connection:
            count = connection.execute(
                select(func.count()).select_from(models.players)
            ).scalar_one()
        assert count == 4


class TestSchema:
    def test_columns_primary_key_and_foreign_key(self, engine):
        inspector = inspect(engine)
        columns = {c["name"]: c for c in inspector.get_columns("player_availability")}
        assert list(columns) == [
            "player_id",
            "status",
            "source",
            "note",
            "expected_return",
            "updated_at",
        ]
        assert columns["status"]["nullable"] is False
        assert columns["updated_at"]["nullable"] is False
        assert columns["source"]["nullable"] is True and columns["note"]["nullable"] is True
        assert inspector.get_pk_constraint("player_availability")["constrained_columns"] == [
            "player_id"
        ]
        (foreign_key,) = inspector.get_foreign_keys("player_availability")
        assert foreign_key["referred_table"] == "players"
        assert foreign_key["constrained_columns"] == ["player_id"]

    def test_the_check_constraint_has_a_stable_name(self, engine):
        checks = inspect(engine).get_check_constraints("player_availability")
        assert [c["name"] for c in checks] == ["ck_player_availability_status"]

    def test_the_schema_compiles_for_postgres(self):
        ddl = str(CreateTable(models.player_availability).compile(dialect=postgresql.dialect()))
        assert "player_id VARCHAR NOT NULL" in ddl
        assert "status VARCHAR NOT NULL" in ddl
        assert "expected_return DATE" in ddl
        assert "updated_at TIMESTAMP WITH TIME ZONE NOT NULL" in ddl
        assert "CONSTRAINT pk_player_availability PRIMARY KEY (player_id)" in ddl
        assert "CONSTRAINT ck_player_availability_status CHECK" in ddl
        assert "status IN ('out', 'doubtful', 'available')" in ddl
        assert "FOREIGN KEY(player_id) REFERENCES players (player_id)" in ddl

    def test_the_upsert_works_on_the_postgres_dialect(self):
        from sqlalchemy.dialects.postgresql import insert

        statement = insert(models.player_availability).values(
            player_id="P1", status="out", updated_at=NOW
        )
        statement = statement.on_conflict_do_update(
            index_elements=["player_id"], set_={"status": statement.excluded.status}
        )
        assert "ON CONFLICT (player_id) DO UPDATE" in str(
            statement.compile(dialect=postgresql.dialect())
        )


class TestConcurrency:
    def test_concurrent_upserts_of_the_same_player_leave_one_consistent_record(self, engine):
        statuses = ["out", "doubtful", "available"]

        def worker(index: int):
            for round_number in range(5):
                status = statuses[(index + round_number) % 3]
                save(
                    engine,
                    "P000001",
                    status,
                    note=f"worker {index}",
                    at=NOW + timedelta(seconds=index),
                )

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(8)))
        records = av.list_availability(engine)
        assert len(records) == 1
        final = records[0]
        assert final.status.value in statuses
        assert final.note and final.note.startswith("worker ")

    def test_concurrent_upserts_of_different_players_are_all_stored(self, engine):
        def worker(player_id: str):
            for _ in range(5):
                save(engine, player_id, "out")

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(worker, ["P000001", "P000002", "P000003", "P000004"]))
        assert {r.player_id for r in av.list_availability(engine)} == {
            "P000001",
            "P000002",
            "P000003",
            "P000004",
        }

    def test_repeating_the_same_upsert_is_idempotent(self, engine):
        first = save(engine, "P000001", "out", note="x")
        for _ in range(3):
            assert save(engine, "P000001", "out", note="x") == first
        assert len(av.list_availability(engine)) == 1
