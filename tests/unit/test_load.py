"""Tests του `elfantasy.features.load`: η ανάγνωση σε τμήματα (μνήμη) δίνει ΑΚΡΙΒΩΣ τα ίδια
αποτελέσματα με την ανάγνωση χωρίς τμήματα.

Το `load_history` διαβάζει ~73.000 γραμμές. Το απλό `pd.read_sql` μετατρέπει όλες τις γραμμές σε
αντικείμενα Python πριν φτιάξει τις στήλες (~100 MB προσωρινά στο Render, όπου η μνήμη είναι
512 MB), γι' αυτό διαβάζει σε τμήματα. Εδώ ελέγχεται ότι τίποτα δεν αλλάζει στα δεδομένα: ούτε οι
τιμές, ούτε οι τύποι, ούτε στις οριακές περιπτώσεις (τμήμα που είναι όλο NULL, κενό αποτέλεσμα).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select, text, update

from elfantasy.db import models
from elfantasy.db.session import get_engine
from elfantasy.features import load


def reference_load_history(engine) -> pd.DataFrame:
    """Η υλοποίηση του `load_history` ΠΡΙΝ από την ανάγνωση σε τμήματα (απλό `pd.read_sql`)."""
    games, player_games = models.games, models.player_games
    on_game = (player_games.c.season == games.c.season) & (
        player_games.c.gamecode == games.c.gamecode
    )
    statement = (
        select(
            player_games.c.season,
            player_games.c.gamecode,
            player_games.c.player_id,
            player_games.c.team_code,
            player_games.c.opp_code,
            player_games.c.home,
            player_games.c.is_starter,
            player_games.c.minutes,
            player_games.c.dnp,
            player_games.c.pir,
            player_games.c.fantasy_score,
            player_games.c.won,
            games.c.game_date,
            games.c.tipoff_utc,
            games.c.phase,
            games.c.home_score,
            games.c.away_score,
        )
        .select_from(player_games.join(games, on_game))
        .where(games.c.played.is_(True))
    )
    with engine.connect() as connection:
        frame = pd.read_sql(statement, connection)
    frame["game_date"] = load._to_datetime_ns(frame["game_date"])
    frame["tipoff_utc"] = load._to_datetime_ns(frame["tipoff_utc"])
    home = frame["home"].astype(bool)
    frame["team_score"] = np.where(home, frame["home_score"], frame["away_score"]).astype(float)
    frame["opp_score"] = np.where(home, frame["away_score"], frame["home_score"]).astype(float)
    frame = frame.drop(columns=["home_score", "away_score"])
    for column in ("home", "is_starter", "dnp", "won"):
        frame[column] = frame[column].astype(bool)
    return frame


def assert_same(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_dtype=True)
    assert actual.dtypes.astype(str).to_dict() == expected.dtypes.astype(str).to_dict()


# --------------------------------------------------------------------------------------
# load_history
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_rows", [97, 400, 2600, 100_000])
def test_load_history_is_identical_for_any_chunk_size(api_engine, chunk_rows):
    expected = reference_load_history(api_engine)
    assert len(expected) > 200  # το συνθετικό πρωτάθλημα έχει αρκετές γραμμές για πολλά τμήματα
    assert_same(load.load_history(api_engine, chunk_rows=chunk_rows), expected)


def test_the_default_chunk_size_gives_the_same_result(api_engine):
    assert_same(load.load_history(api_engine), reference_load_history(api_engine))


def test_load_history_on_the_real_fixture_games(fixture_database_url):
    engine = get_engine(fixture_database_url)
    try:
        expected = reference_load_history(engine)
        assert len(expected) > 100
        for chunk_rows in (3, 40):
            assert_same(load.load_history(engine, chunk_rows=chunk_rows), expected)
    finally:
        engine.dispose()


def test_a_chunk_that_is_entirely_null_in_a_column_keeps_the_types(league_db):
    """Το `phase` είναι nullable: αν ένα ολόκληρο τμήμα έχει NULL, το pandas θα έδινε `object`."""
    with league_db.begin() as connection:
        connection.execute(update(models.games).values(phase=None))  # όλα NULL
    expected = reference_load_history(league_db)
    assert_same(load.load_history(league_db, chunk_rows=300), expected)
    with league_db.begin() as connection:
        connection.execute(update(models.games).values(phase="RS"))
        connection.execute(
            update(models.games).where(models.games.c.gamecode % 2 == 0).values(phase=None)
        )
    mixed = reference_load_history(league_db)
    assert mixed["phase"].isna().any() and mixed["phase"].notna().any()
    for chunk_rows in (97, 600):
        assert_same(load.load_history(league_db, chunk_rows=chunk_rows), mixed)


def test_unknown_tipoff_times_are_handled_per_chunk(league_db):
    with league_db.begin() as connection:
        connection.execute(update(models.games).values(tipoff_utc=None))
    assert_same(load.load_history(league_db, chunk_rows=300), reference_load_history(league_db))


def test_an_empty_database_gives_an_empty_frame_with_the_same_columns(tmp_path):
    engine = get_engine(f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")
    models.metadata.create_all(engine)
    try:
        expected = reference_load_history(engine)
        actual = load.load_history(engine)
        assert len(actual) == 0
        assert list(actual.columns) == list(expected.columns)
        assert_same(actual, expected)
    finally:
        engine.dispose()


# --------------------------------------------------------------------------------------
# read_in_chunks
# --------------------------------------------------------------------------------------


@pytest.fixture
def table():
    """Μικρός πίνακας με όλους τους τύπους και μπλοκ από NULL, για να υπάρχουν τμήματα όλα NULL."""
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE t (name TEXT, n INTEGER, x REAL, flag INTEGER, day TEXT)")
        )
        rows = [("a", 1, 1.5, 1, "2026-01-01"), ("b", 2, 2.5, 0, "2026-01-02")]
        rows += [(None, None, None, None, None)] * 5
        rows += [("c", 8, 8.5, 1, "2026-01-08"), (None, 9, None, 0, "2026-01-09")]
        for row in rows:
            connection.execute(
                text("INSERT INTO t VALUES (:a, :b, :c, :d, :e)"),
                dict(zip("abcde", row, strict=True)),
            )
    yield engine
    engine.dispose()


QUERY = text("SELECT name, n, x, flag, day FROM t ORDER BY rowid")


@pytest.mark.parametrize("chunk_rows", [1, 2, 3, 4, 100])
def test_read_in_chunks_equals_read_sql_even_with_null_blocks(table, chunk_rows):
    with table.connect() as connection:
        expected = pd.read_sql(QUERY, connection)
        actual = load.read_in_chunks(QUERY, connection, chunk_rows)
    assert_same(actual, expected)


def test_read_in_chunks_returns_the_chunk_itself_when_there_is_one(table):
    with table.connect() as connection:
        frame = load.read_in_chunks(QUERY, connection, 1000)
    assert len(frame) == 9 and str(frame["name"].dtype) == "str"


def test_read_in_chunks_with_no_rows(table):
    empty = text("SELECT name, n, x FROM t WHERE n > 1000")
    with table.connect() as connection:
        expected = pd.read_sql(empty, connection)
        actual = load.read_in_chunks(empty, connection, 2)
    assert_same(actual, expected)


def test_a_pandas_that_yields_no_chunk_for_an_empty_result_is_handled(table, monkeypatch):
    """Με το pandas 3.0 το αποτέλεσμα χωρίς γραμμές δίνει ένα κενό τμήμα· μια άλλη έκδοση θα
    μπορούσε να μη δώσει κανένα, και τότε διαβάζεται κανονικά χωρίς τμήματα."""
    empty = text("SELECT name, n, x FROM t WHERE n > 1000")
    real = pd.read_sql

    def without_chunks(statement, connection, chunksize=None, **kwargs):
        if chunksize is not None:
            return iter(())
        return real(statement, connection, **kwargs)

    with table.connect() as connection:
        expected = real(empty, connection)
        monkeypatch.setattr(pd, "read_sql", without_chunks)
        actual = load.read_in_chunks(empty, connection, 2)
    assert_same(actual, expected)


@pytest.mark.parametrize(
    ("column", "template", "kind"),
    [
        (["a", None], pd.StringDtype(na_value=np.nan), "str"),
        ([1, None], np.dtype("int64"), "float64"),
        ([1.5, None], np.dtype("float64"), "float64"),
        ([pd.Timestamp("2026-01-01"), None], np.dtype("datetime64[us]"), "datetime64[us]"),
        ([True, None], np.dtype("bool"), "object"),  # οι λογικές τιμές με NULL μένουν ως έχουν
    ],
)
def test_restore_dtype(column, template, kind):
    restored = load._restore_dtype(pd.Series(column, dtype=object), template)
    assert str(restored.dtype) == kind
