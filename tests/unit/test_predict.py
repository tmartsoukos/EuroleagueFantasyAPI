"""Tests του `Predictor` (`elfantasy.model.predict`) πάνω σε προσωρινή βάση SQLite με συνθετικά
δεδομένα και μικρό μοντέλο.

Ελέγχουν τη διεπαφή που θα χρησιμοποιήσει το API (Φάση 4): άγνωστος παίκτης, επόμενος αγώνας,
παίκτης χωρίς προγραμματισμένο αγώνα (ουδέτερο πλαίσιο), ταξινόμηση, φίλτρα, ενεργοί παίκτες,
αλλαγή του `as_of`, συνέπεια `predict_player` και `predict_all`, cache και `refresh`.
"""

from __future__ import annotations

import dataclasses
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import insert, update
from synthetic_league import write_to_database

from elfantasy.db import models
from elfantasy.db.session import get_engine
from elfantasy.features.build import FEATURE_COLUMNS, UPCOMING_COLUMNS, build_features
from elfantasy.model import predict as P
from elfantasy.model.artifact import ModelLoadError, save_bundle
from elfantasy.model.predict import NextGame, PlayerPrediction, Predictor

CONTEXT_FEATURES = [
    "home",
    "team_rest_days",
    "opp_rest_days",
    "team_short_rest",
    "team_games_last7",
    "opp_games_last7",
    "team_win_pct_5",
    "opp_win_pct_5",
    "opp_pd_season",
    "opp_def_pir_5",
    "opp_def_pir_10",
    "team_games_season",
    "missed_last10",
]


@pytest.fixture(scope="module")
def as_of(synthetic_league) -> date:
    """Η ημέρα μετά τον τελευταίο παιγμένο αγώνα (οι επόμενοι αγώνες είναι στο πρόγραμμα)."""
    return synthetic_league.last_played_date + timedelta(days=1)


@pytest.fixture(scope="module")
def database(synthetic_league, tmp_path_factory):
    path = tmp_path_factory.mktemp("predict") / "league.db"
    engine = get_engine(f"sqlite:///{path.as_posix()}")
    write_to_database(engine, synthetic_league)
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def predictor(tiny_xgb_bundle, database, as_of):
    return Predictor(tiny_xgb_bundle, database, today=lambda: as_of)


@pytest.fixture
def fresh_database(synthetic_league, tmp_path):
    engine = get_engine(f"sqlite:///{(tmp_path / 'fresh.db').as_posix()}")
    write_to_database(engine, synthetic_league)
    yield engine
    engine.dispose()


def latest_rows(history: pd.DataFrame) -> pd.DataFrame:
    order = history.sort_values(["tipoff_utc", "season", "gamecode"], kind="stable")
    return order.groupby("player_id").tail(1).set_index("player_id")


def first_games(schedule: pd.DataFrame, as_of: date) -> dict[str, pd.Series]:
    """Ο επόμενος αγώνας κάθε ομάδας, υπολογισμένος ανεξάρτητα από τον Predictor."""
    future = schedule[schedule["game_date"] >= pd.Timestamp(as_of)].sort_values(
        ["tipoff_utc", "gamecode"]
    )
    result: dict[str, pd.Series] = {}
    for game in future.itertuples():
        for team, opp, home in (
            (game.home_code, game.away_code, True),
            (game.away_code, game.home_code, False),
        ):
            result.setdefault(team, pd.Series({**game._asdict(), "opp": opp, "home": home}))
    return result


class TestBasics:
    def test_unknown_players_return_none(self, predictor):
        for player_id in ("P999999", "", "   ", "NOBODY"):
            assert predictor.predict_player(player_id) is None

    def test_the_player_id_is_stripped_and_uppercased(self, predictor, synthetic_league):
        player_id = synthetic_league.players["player_id"].iloc[3]
        reference = predictor.predict_player(player_id)
        assert reference is not None
        assert predictor.predict_player(f"  {player_id.lower()} ") == reference

    def test_the_model_version_and_metrics_are_exposed(self, predictor, tiny_xgb_bundle):
        assert predictor.model_version == "test-xgb-v1"
        assert predictor.metrics["threshold"]["value"] == 9.99
        prediction = predictor.predict_all()[0]
        assert prediction.model_version == "test-xgb-v1"

    def test_the_dataclasses_are_frozen(self, predictor):
        prediction = predictor.predict_all()[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            prediction.predicted_fantasy = 0.0
        assert dataclasses.is_dataclass(NextGame) and dataclasses.is_dataclass(PlayerPrediction)
        game = prediction.next_game
        assert game is not None
        with pytest.raises(dataclasses.FrozenInstanceError):
            game.home = not game.home

    def test_as_of_accepts_a_datetime(self, predictor, as_of, synthetic_league):
        player_id = synthetic_league.players["player_id"].iloc[0]
        same_day = predictor.predict_player(
            player_id, as_of=datetime(as_of.year, as_of.month, as_of.day, 15)
        )
        assert same_day == predictor.predict_player(player_id, as_of=as_of)

    def test_the_default_as_of_is_today_in_utc(self, tiny_xgb_bundle, database):
        assert P.utc_today() == datetime.now(P.UTC).date()
        default = Predictor(tiny_xgb_bundle, database)  # χωρίς today: σημερινή ημερομηνία
        assert default.predict_all(active_only=False)  # δεν σκάει και επιστρέφει προβλέψεις


class TestNextGame:
    def test_the_next_game_is_the_first_scheduled_game_of_the_players_team(
        self, predictor, synthetic_league, as_of
    ):
        latest = latest_rows(synthetic_league.history)
        expected = first_games(synthetic_league.schedule, as_of)
        checked = 0
        for player_id in latest.index[:60]:
            team = latest.loc[player_id, "team_code"]
            prediction = predictor.predict_player(player_id)
            game = expected[team]
            assert prediction.team_code == team
            assert prediction.next_game == NextGame(
                season=int(game["season"]),
                gamecode=int(game["gamecode"]),
                game_date=game["game_date"].date(),
                tipoff_utc=game["tipoff_utc"].to_pydatetime(),
                team_code=team,
                opp_code=game["opp"],
                home=bool(game["home"]),
            )
            assert isinstance(prediction.next_game.tipoff_utc, datetime)
            assert prediction.next_game.tipoff_utc.tzinfo is None  # naive UTC, όπως στη βάση
            checked += 1
        assert checked == 60

    def test_home_and_away_flags_both_occur(self, predictor):
        homes = {p.next_game.home for p in predictor.predict_all(active_only=False)}
        assert homes == {True, False}

    def test_a_later_as_of_selects_the_following_game(self, predictor, synthetic_league, as_of):
        schedule = synthetic_league.schedule.sort_values(["tipoff_utc", "gamecode"])
        team = schedule.iloc[0]["home_code"]
        team_games = schedule[(schedule["home_code"] == team) | (schedule["away_code"] == team)]
        first, second = team_games.iloc[0], team_games.iloc[1]
        player_id = latest_rows(synthetic_league.history).query("team_code == @team").index[0]
        before = predictor.predict_player(player_id, as_of=first["game_date"].date())
        after = predictor.predict_player(
            player_id, as_of=first["game_date"].date() + timedelta(days=1)
        )
        assert before.next_game.gamecode == first["gamecode"]
        assert after.next_game.gamecode == second["gamecode"]
        assert after.next_game.game_date > before.next_game.game_date

    def test_features_change_with_the_next_game_but_the_form_stays(
        self, predictor, synthetic_league
    ):
        schedule = synthetic_league.schedule.sort_values(["tipoff_utc", "gamecode"])
        team = schedule.iloc[0]["home_code"]
        team_games = schedule[(schedule["home_code"] == team) | (schedule["away_code"] == team)]
        first = team_games.iloc[0]
        player_id = latest_rows(synthetic_league.history).query("team_code == @team").index[0]
        a = predictor.predict_player(player_id, as_of=first["game_date"].date())
        b = predictor.predict_player(player_id, as_of=first["game_date"].date() + timedelta(days=1))
        assert a.features["pir_mean_5"] == b.features["pir_mean_5"]
        assert a.features["team_rest_days"] != b.features["team_rest_days"] or (
            a.features["opp_games_last7"] != b.features["opp_games_last7"]
        )

    def test_games_before_as_of_are_skipped_even_if_they_were_never_played(
        self, fresh_database, tiny_xgb_bundle, synthetic_league, as_of
    ):
        # Ακυρωμένος αγώνας του παρελθόντος (played = false) για μια ομάδα: δεν είναι «επόμενος».
        team = synthetic_league.schedule.iloc[0]["home_code"]
        other = synthetic_league.schedule.iloc[0]["away_code"]
        with fresh_database.begin() as connection:
            connection.execute(
                insert(models.games),
                {
                    "season": 2023,
                    "gamecode": 999,
                    "phase": "RS",
                    "round": 1,
                    "game_date": as_of - timedelta(days=30),
                    "tipoff_utc": datetime(as_of.year, as_of.month, 1, 12) - timedelta(days=30),
                    "home_code": team,
                    "away_code": other,
                    "home_score": None,
                    "away_score": None,
                    "played": False,
                    "winner_code": None,
                },
            )
        predictor = Predictor(tiny_xgb_bundle, fresh_database, today=lambda: as_of)
        player_id = latest_rows(synthetic_league.history).query("team_code == @team").index[0]
        assert predictor.predict_player(player_id).next_game.gamecode != 999

    def test_an_unknown_tipoff_time_is_none_and_ordered_by_date(
        self, fresh_database, tiny_xgb_bundle, synthetic_league, as_of
    ):
        game = synthetic_league.schedule.sort_values(["tipoff_utc", "gamecode"]).iloc[0]
        with fresh_database.begin() as connection:
            connection.execute(
                update(models.games)
                .where(models.games.c.season == int(game["season"]))
                .where(models.games.c.gamecode == int(game["gamecode"]))
                .values(tipoff_utc=None)
            )
        predictor = Predictor(tiny_xgb_bundle, fresh_database, today=lambda: as_of)
        latest = latest_rows(synthetic_league.history)
        player_id = latest[latest["team_code"] == game["home_code"]].index[0]
        prediction = predictor.predict_player(player_id)
        assert prediction.next_game.gamecode == int(game["gamecode"])
        assert prediction.next_game.tipoff_utc is None
        assert np.isfinite(prediction.predicted_fantasy)

    def test_next_game_for_team(self, predictor, synthetic_league, as_of):
        expected = first_games(synthetic_league.schedule, as_of)
        team = synthetic_league.teams[2]
        game = predictor.next_game_for_team(f" {team.lower()} ")
        assert (game.gamecode, game.opp_code, game.home) == (
            int(expected[team]["gamecode"]),
            expected[team]["opp"],
            bool(expected[team]["home"]),
        )
        assert predictor.next_game_for_team("XXX") is None


class TestWithoutAScheduledGame:
    def test_neutral_context_when_the_schedule_is_over(self, predictor, synthetic_league):
        after_everything = synthetic_league.schedule["game_date"].max().date() + timedelta(days=1)
        players = predictor.predict_all(as_of=after_everything)
        assert players, "the offseason must not produce an empty ranking"
        for prediction in players:
            assert prediction.next_game is None
            assert np.isfinite(prediction.predicted_fantasy) and np.isfinite(
                prediction.predicted_pir
            )
            for name in CONTEXT_FEATURES:
                assert prediction.features[name] is None, name

    def test_player_features_are_still_computed(self, predictor, synthetic_league):
        after_everything = synthetic_league.schedule["game_date"].max().date() + timedelta(days=1)
        veteran = max(
            predictor.predict_all(as_of=after_everything), key=lambda p: p.n_prior_appearances
        )
        assert veteran.n_prior_appearances > 20
        assert veteran.features["pir_mean_5"] is not None
        assert veteran.features["games_played_total"] == veteran.n_prior_appearances
        assert veteran.features["days_since_last_appearance"] is not None

    def test_a_team_without_games_gets_neutral_context_while_others_have_a_game(
        self, fresh_database, tiny_xgb_bundle, synthetic_league, as_of
    ):
        team = synthetic_league.teams[0]
        with fresh_database.begin() as connection:
            for column in ("home_code", "away_code"):
                connection.execute(
                    models.games.delete()
                    .where(models.games.c.played.is_(False))
                    .where(getattr(models.games.c, column) == team)
                )
        predictor = Predictor(tiny_xgb_bundle, fresh_database, today=lambda: as_of)
        by_team = {}
        for prediction in predictor.predict_all():
            by_team.setdefault(prediction.team_code, []).append(prediction)
        assert all(p.next_game is None for p in by_team[team])
        others = [p for t, group in by_team.items() if t != team for p in group]
        assert others and all(p.next_game is not None for p in others)

    def test_neutral_predictions_are_not_equal_to_those_with_context(
        self, predictor, synthetic_league, as_of
    ):
        player_id = latest_rows(synthetic_league.history).index[5]
        with_game = predictor.predict_player(player_id)
        neutral = predictor.predict_player(player_id, as_of=as_of + timedelta(days=500))
        assert with_game.next_game is not None and neutral.next_game is None
        assert with_game.features["home"] is not None and neutral.features["home"] is None


class TestRankings:
    def test_sorted_by_predicted_fantasy_descending(self, predictor):
        players = predictor.predict_all()
        values = [p.predicted_fantasy for p in players]
        assert values == sorted(values, reverse=True)
        ties = [
            (a, b)
            for a, b in zip(players, players[1:], strict=False)
            if a.predicted_fantasy == b.predicted_fantasy
        ]
        assert all(a.player_id < b.player_id for a, b in ties)

    def test_team_filter(self, predictor, synthetic_league):
        team = synthetic_league.teams[1]
        players = predictor.predict_all(team_code=f"  {team.lower()}  ")
        assert players and all(p.team_code == team for p in players)
        everyone = predictor.predict_all()
        assert len(players) == sum(1 for p in everyone if p.team_code == team)
        assert [p.player_id for p in players] == [
            p.player_id for p in everyone if p.team_code == team
        ]
        assert predictor.predict_all(team_code="NOPE") == []

    def test_active_players_are_those_with_a_row_in_the_newest_season(
        self, predictor, synthetic_league, as_of
    ):
        history = synthetic_league.history
        newest = history["season"].max()
        expected = set(history.loc[history["season"] == newest, "player_id"])
        window_start = pd.Timestamp(as_of) - pd.Timedelta(days=P.DEFAULT_ACTIVE_WINDOW_DAYS)
        expected |= set(history.loc[history["game_date"] >= window_start, "player_id"])
        active = {p.player_id for p in predictor.predict_all()}
        assert active == expected
        everyone = predictor.predict_all(active_only=False)
        assert len(everyone) == history["player_id"].nunique() > len(active)
        assert {p.player_id for p in everyone if p.is_active} == active

    def test_players_who_left_are_inactive_but_can_still_be_predicted(
        self, predictor, synthetic_league
    ):
        history = synthetic_league.history
        gone = set(history["player_id"]) - set(history.loc[history["season"] == 2023, "player_id"])
        assert gone
        prediction = predictor.predict_player(sorted(gone)[0])
        assert prediction is not None and prediction.is_active is False

    def test_the_active_list_is_never_empty(self, predictor, as_of):
        far_future = predictor.predict_all(as_of=as_of + timedelta(days=2000))
        before_everything = predictor.predict_all(as_of=date(1990, 1, 1))
        assert far_future and before_everything
        assert {p.player_id for p in far_future} == {p.player_id for p in before_everything}

    def test_the_active_window_widens_the_set_around_a_season_change(
        self, tiny_xgb_bundle, database, as_of
    ):
        wide = Predictor(tiny_xgb_bundle, database, today=lambda: as_of, active_window_days=3650)
        narrow = Predictor(tiny_xgb_bundle, database, today=lambda: as_of, active_window_days=0)
        assert len(wide.predict_all()) > len(narrow.predict_all())

    def test_predict_player_equals_the_element_of_predict_all(self, predictor):
        for prediction in predictor.predict_all(active_only=False)[::25]:
            assert predictor.predict_player(prediction.player_id) == prediction

    def test_every_prediction_is_finite_including_players_without_history(
        self, predictor, synthetic_league
    ):
        everyone = predictor.predict_all(active_only=False)
        assert any(p.n_prior_appearances == 0 for p in everyone), (
            "needs a player with no appearances"
        )
        for prediction in everyone:
            assert np.isfinite(prediction.predicted_fantasy)
            assert np.isfinite(prediction.predicted_pir)
            assert set(prediction.features) == set(FEATURE_COLUMNS)

    def test_prediction_fields(self, predictor, synthetic_league):
        history = synthetic_league.history
        appeared = history[(~history["dnp"]) & (history["minutes"] > 0)]
        counts = appeared.groupby("player_id").size()
        last = appeared.groupby("player_id")["game_date"].max()
        names = synthetic_league.players.set_index("player_id")["name"]
        for prediction in predictor.predict_all(active_only=False)[::17]:
            assert prediction.name == names[prediction.player_id]
            assert prediction.n_prior_appearances == counts.get(prediction.player_id, 0)
            expected_last = last.get(prediction.player_id)
            assert prediction.last_appearance_date == (
                None if expected_last is None else expected_last.date()
            )
            assert isinstance(prediction.predicted_fantasy, float)
            assert all(v is None or isinstance(v, float) for v in prediction.features.values())

    def test_the_features_are_those_of_the_training_code_path(
        self, predictor, synthetic_league, as_of
    ):
        player_id = latest_rows(synthetic_league.history).index[7]
        prediction = predictor.predict_player(player_id)
        game = prediction.next_game
        upcoming = pd.DataFrame(
            [
                {
                    "player_id": player_id,
                    "season": game.season,
                    "gamecode": game.gamecode,
                    "team_code": game.team_code,
                    "opp_code": game.opp_code,
                    "home": float(game.home),
                    "game_date": pd.Timestamp(game.game_date),
                    "tipoff_utc": pd.Timestamp(game.tipoff_utc),
                }
            ]
        )
        assert set(upcoming.columns) == set(UPCOMING_COLUMNS)
        features = build_features(synthetic_league.history, upcoming, games=synthetic_league.games)
        row = features[features["is_upcoming"]].iloc[0]
        for name in FEATURE_COLUMNS:
            expected = row[name]
            got = prediction.features[name]
            if np.isnan(expected):
                assert got is None
            else:
                assert got == pytest.approx(expected)


class TestCacheAndRefresh:
    def test_one_pass_per_as_of_date(self, tiny_xgb_bundle, database, as_of, monkeypatch):
        calls = []
        original = P.build_features

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(P, "build_features", counting)
        predictor = Predictor(tiny_xgb_bundle, database, today=lambda: as_of)
        predictor.predict_all()
        predictor.predict_all(team_code="T01")
        predictor.predict_all(active_only=False)
        predictor.predict_player("P000001")
        assert len(calls) == 1
        predictor.predict_all(as_of=as_of + timedelta(days=1))
        assert len(calls) == 2
        predictor.refresh()
        predictor.predict_all()
        assert len(calls) == 3

    def test_concurrent_requests_share_one_computation(
        self, tiny_xgb_bundle, database, as_of, monkeypatch
    ):
        calls = []
        original = P.build_features

        def slow(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(P, "build_features", slow)
        predictor = Predictor(tiny_xgb_bundle, database, today=lambda: as_of)
        barrier = threading.Barrier(6)

        def worker(_):
            barrier.wait()
            return [p.player_id for p in predictor.predict_all()[:5]]

        with ThreadPoolExecutor(6) as pool:
            results = list(pool.map(worker, range(6)))
        assert len(calls) == 1
        assert all(result == results[0] for result in results)

    def test_the_cache_is_bounded(self, tiny_xgb_bundle, database, as_of, monkeypatch):
        monkeypatch.setattr(Predictor, "_compute", lambda self, day: {"day": day})
        predictor = Predictor(tiny_xgb_bundle, database, today=lambda: as_of)
        for offset in range(P._CACHE_SIZE + 3):
            predictor._predictions(as_of + timedelta(days=offset))
        assert len(predictor._cache) == P._CACHE_SIZE
        # Η παλαιότερη ημερομηνία αφαιρείται πρώτη (LRU)
        assert as_of not in predictor._cache
        assert as_of + timedelta(days=P._CACHE_SIZE + 2) in predictor._cache

    def test_refresh_picks_up_a_newly_played_game(
        self, fresh_database, tiny_xgb_bundle, synthetic_league, as_of
    ):
        schedule = synthetic_league.schedule.sort_values(["tipoff_utc", "gamecode"])
        game = schedule.iloc[0]
        home, away = game["home_code"], game["away_code"]
        latest = latest_rows(synthetic_league.history)
        player_id = latest[latest["team_code"] == home].index[0]
        predictor = Predictor(tiny_xgb_bundle, fresh_database, today=lambda: as_of)
        before = predictor.predict_player(player_id)
        assert before.next_game.gamecode == int(game["gamecode"])

        with fresh_database.begin() as connection:  # ο αγώνας παίχτηκε και ο παίκτης έπαιξε
            connection.execute(
                update(models.games)
                .where(models.games.c.season == int(game["season"]))
                .where(models.games.c.gamecode == int(game["gamecode"]))
                .values(played=True, home_score=90, away_score=80, winner_code=home)
            )
            row = {column.name: 0 for column in models.player_games.columns}
            row.update(
                season=int(game["season"]),
                gamecode=int(game["gamecode"]),
                player_id=player_id,
                team_code=home,
                opp_code=away,
                home=True,
                is_starter=True,
                minutes=30.0,
                dnp=False,
                plus_minus=None,
                valuation=25,
                pir=25,
                won=True,
                fantasy_score=27.5,
            )
            connection.execute(insert(models.player_games), row)
        assert (
            predictor.predict_player(player_id).n_prior_appearances == before.n_prior_appearances
        )  # cache
        predictor.refresh()
        after = predictor.predict_player(player_id, as_of=game["game_date"].date())
        assert after.n_prior_appearances == before.n_prior_appearances + 1
        assert after.last_appearance_date == game["game_date"].date()
        assert after.next_game.gamecode != int(game["gamecode"])
        assert after.features["min_last"] == 30.0


class TestLoading:
    def test_load_from_a_path_and_an_engine(self, tiny_xgb_bundle, database, tmp_path):
        path = tmp_path / "model.joblib"
        save_bundle(tiny_xgb_bundle, path)
        loaded = Predictor.load(path, database)
        assert loaded.model_version == "test-xgb-v1"
        assert loaded.predict_all(active_only=False)

    def test_defaults_come_from_the_settings(
        self, tiny_ridge_bundle, synthetic_league, tmp_path, monkeypatch
    ):
        from elfantasy.config import get_settings

        model_path = tmp_path / "models" / "m.joblib"
        save_bundle(tiny_ridge_bundle, model_path)
        db_path = tmp_path / "from_settings.db"
        engine = get_engine(f"sqlite:///{db_path.as_posix()}")
        write_to_database(engine, synthetic_league)
        engine.dispose()
        monkeypatch.setenv("MODEL_PATH", str(model_path))
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
        get_settings.cache_clear()
        loaded = Predictor.load()
        assert loaded.model_version == "test-ridge-v1"
        assert loaded.predict_all(active_only=False)
        loaded.close()  # ο Predictor δημιούργησε μόνος του τον engine: το close() τον κλείνει

    def test_the_metrics_fall_back_to_metrics_json(self, tiny_xgb_bundle, database, tmp_path):
        import dataclasses as dc
        import json

        bare = dc.replace(tiny_xgb_bundle, metrics={})
        path = tmp_path / "model.joblib"
        save_bundle(bare, path)
        (tmp_path / "metrics.json").write_text(
            json.dumps({"threshold": {"value": 5.5}}), encoding="utf-8"
        )
        assert Predictor.load(path, database).metrics["threshold"]["value"] == 5.5

    def test_close_disposes_only_an_engine_created_by_the_predictor(
        self, tiny_xgb_bundle, database, tmp_path, monkeypatch
    ):
        path = tmp_path / "model.joblib"
        save_bundle(tiny_xgb_bundle, path)
        disposed = []
        real_dispose = database.dispose
        monkeypatch.setattr(database, "dispose", lambda *a, **k: disposed.append(1))
        external = Predictor.load(path, database)  # engine απ' έξω: δεν κλείνει
        external.close()
        assert disposed == []
        monkeypatch.setattr(P, "get_engine", lambda: database)
        owned = Predictor.load(path)  # ο engine δημιουργείται από τις ρυθμίσεις: κλείνει
        owned.close()
        assert disposed == [1]
        monkeypatch.undo()
        real_dispose()

    def test_a_missing_or_broken_model_is_a_clear_error(self, database, tmp_path):
        with pytest.raises(ModelLoadError, match="not found"):
            Predictor.load(tmp_path / "missing.joblib", database)
        broken = tmp_path / "broken.joblib"
        broken.write_bytes(b"not a model")
        with pytest.raises(ModelLoadError, match="corrupted"):
            Predictor.load(broken, database)

    def test_an_empty_database_gives_empty_results_not_an_error(self, tiny_xgb_bundle, tmp_path):
        from elfantasy.db.session import create_all

        engine = get_engine(f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")
        create_all(engine)
        predictor = Predictor(tiny_xgb_bundle, engine)
        assert predictor.predict_all() == []
        assert predictor.predict_player("P000001") is None
        engine.dispose()

    def test_a_ridge_model_works_too(self, tiny_ridge_bundle, database, as_of):
        predictor = Predictor(tiny_ridge_bundle, database, today=lambda: as_of)
        players = predictor.predict_all()
        assert players and all(np.isfinite(p.predicted_fantasy) for p in players)
