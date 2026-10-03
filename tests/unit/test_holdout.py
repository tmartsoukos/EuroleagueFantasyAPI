"""Tests της εξαγωγής του holdout fixture (`elfantasy.model.holdout`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synthetic_league import write_to_database

from elfantasy.db.session import create_all, get_engine
from elfantasy.features.build import FEATURE_COLUMNS
from elfantasy.model import holdout as H
from elfantasy.model.train import BASELINE_NAMES, baseline_predictions

VAL, TEST = 2022, 2023


@pytest.fixture(scope="module")
def built(synthetic_frame):
    return H.build_holdout(synthetic_frame, VAL, TEST)


@pytest.fixture(scope="module")
def db_url(synthetic_league, tmp_path_factory):
    path = tmp_path_factory.mktemp("holdout_db") / "league.db"
    url = f"sqlite:///{path.as_posix()}"
    engine = get_engine(url)
    write_to_database(engine, synthetic_league)
    engine.dispose()
    return url


class TestBuild:
    def test_contains_every_appearance_of_the_test_season(self, built, synthetic_frame):
        holdout, meta = built
        rows = synthetic_frame[
            synthetic_frame["is_appearance"] & (synthetic_frame["season"] == TEST)
        ]
        assert len(holdout) == len(rows) == meta["rows"]
        assert set(holdout["season"]) == {TEST}
        assert not holdout.duplicated(["player_id", "season", "gamecode"]).any()
        order = holdout[["season", "gamecode", "player_id"]]
        assert order.equals(order.sort_values(list(order.columns), kind="stable"))

    def test_columns(self, built):
        holdout, _ = built
        baselines = [f"baseline_{name}" for name in BASELINE_NAMES]
        assert list(holdout.columns) == [
            "player_id",
            "season",
            "gamecode",
            "fantasy_score",
            "pir",
            *FEATURE_COLUMNS,
            *baselines,
        ]
        assert holdout[["fantasy_score", "pir", *baselines]].notna().all().all()

    def test_features_are_stored_as_float32_like_the_model_sees_them(self, built, synthetic_frame):
        holdout, _ = built
        assert all(holdout[name].dtype == np.float32 for name in FEATURE_COLUMNS)
        rows = synthetic_frame[
            synthetic_frame["is_appearance"] & (synthetic_frame["season"] == TEST)
        ]
        rows = rows.sort_values(["season", "gamecode", "player_id"], kind="stable")
        expected = rows[FEATURE_COLUMNS].to_numpy(float).astype(np.float32)
        np.testing.assert_array_equal(holdout[FEATURE_COLUMNS].to_numpy(), expected)

    def test_baselines_use_the_statistics_of_the_fit_seasons_only(self, built, synthetic_frame):
        holdout, meta = built
        appearances = synthetic_frame[synthetic_frame["is_appearance"]]
        fit = appearances[appearances["season"] <= VAL]
        assert meta["baseline_fit_stats"]["fantasy"]["mean"] == pytest.approx(
            fit["fantasy_score"].mean()
        )
        assert holdout["baseline_global_mean"].nunique() == 1
        assert holdout["baseline_global_mean"].iloc[0] == pytest.approx(fit["fantasy_score"].mean())
        expected = baseline_predictions(
            holdout, "fantasy", meta["baseline_fit_stats"]["fantasy"]["mean"], 0.0
        )
        np.testing.assert_allclose(
            holdout["baseline_naive_rolling5"], expected["naive_rolling5"], rtol=1e-6
        )

    def test_metadata(self, built):
        _, meta = built
        assert meta["test_season"] == TEST and meta["val_season"] == VAL
        assert meta["feature_columns"] == FEATURE_COLUMNS
        assert meta["n_games"] > 10

    def test_an_empty_season_is_an_error(self, synthetic_frame):
        with pytest.raises(ValueError, match="no rows"):
            H.build_holdout(synthetic_frame, VAL, 2031)
        with pytest.raises(ValueError, match="no rows"):
            H.build_holdout(synthetic_frame, 2010, TEST)


class TestFile:
    def test_roundtrip_keeps_values_types_and_metadata(self, built, tmp_path):
        holdout, meta = built
        path = tmp_path / "nested" / "holdout.parquet"
        size = H.write_holdout(holdout, meta, path)
        assert size == path.stat().st_size > 0
        again, meta_again = H.read_holdout(path)
        pd.testing.assert_frame_equal(again, holdout)
        assert meta_again == meta

    def test_the_file_is_deterministic(self, built, tmp_path):
        holdout, meta = built
        first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"
        H.write_holdout(holdout, meta, first)
        H.write_holdout(holdout, meta, second)
        assert first.read_bytes() == second.read_bytes()

    def test_a_file_without_metadata_reads_as_empty_metadata(self, built, tmp_path):
        holdout, _ = built
        path = tmp_path / "plain.parquet"
        holdout.to_parquet(path, index=False)
        frame, meta = H.read_holdout(path)
        assert len(frame) == len(holdout) and meta == {}


class TestCommandLine:
    def test_exports_the_fixture(self, db_url, tmp_path, capsys):
        out = tmp_path / "holdout.parquet"
        args = [
            "--db",
            db_url,
            "--out",
            str(out),
            "--val-season",
            str(VAL),
            "--test-season",
            str(TEST),
        ]
        assert H.main(args) == 0
        frame, meta = H.read_holdout(out)
        assert meta["rows"] == len(frame) > 100
        assert "wrote" in capsys.readouterr().out

    def test_bad_seasons_fail(self, db_url, tmp_path, capsys):
        out = tmp_path / "holdout.parquet"
        assert H.main(["--db", db_url, "--out", str(out), "--test-season", "2031"]) == 1
        assert "ERROR" in capsys.readouterr().out
        assert not out.exists()

    def test_an_empty_database_fails(self, tmp_path, capsys):
        url = f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"
        engine = get_engine(url)
        create_all(engine)
        engine.dispose()
        assert H.main(["--db", url, "--out", str(tmp_path / "x.parquet")]) == 1
        assert "no player rows" in capsys.readouterr().out
