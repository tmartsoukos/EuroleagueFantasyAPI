"""Tests του `POST /admin/refresh`: προστασία, αναφορά του νέου data cutoff και ανανέωση των
προβλέψεων και των ονομάτων ομάδων μετά από νέο ingestion.

Κάθε test δουλεύει σε δική του βάση (αντίγραφο του συνθετικού πρωταθλήματος), γιατί γράφει νέους
αγώνες και ονόματα ομάδων.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest
from api_support import ADMIN_HEADERS, FakeClock, make_settings, noon_utc
from fastapi.testclient import TestClient
from sqlalchemy import update
from sqlalchemy.exc import OperationalError
from synthetic_league import write_to_database

from elfantasy.api.main import create_app
from elfantasy.db import models
from elfantasy.db.session import get_engine
from elfantasy.model.predict import Predictor


@pytest.fixture
def private(tmp_path, synthetic_league, tiny_xgb_bundle, api_today):
    """Εφαρμογή πάνω σε δική της βάση, για tests που αλλάζουν τα δεδομένα."""
    engine = get_engine(f"sqlite:///{(tmp_path / 'private.db').as_posix()}")
    write_to_database(engine, synthetic_league)
    predictor = Predictor(tiny_xgb_bundle, engine, today=lambda: api_today)
    clock = FakeClock(noon_utc(api_today))
    app = create_app(settings=make_settings(), engine=engine, predictor=predictor, clock=clock)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield SimpleNamespace(client=client, engine=engine, predictor=predictor, clock=clock)
    engine.dispose()


def refresh(client, headers=ADMIN_HEADERS):
    return client.post("/admin/refresh", headers=headers)


def add_played_game(engine, day, home, away, gamecode=9001):
    """Ένας νέος παιγμένος αγώνας (χωρίς boxscore), όπως θα τον έφερνε ένα νέο ingestion."""
    with engine.begin() as connection:
        connection.execute(
            models.games.insert(),
            {
                "season": 2023,
                "gamecode": gamecode,
                "phase": "RS",
                "round": 1,
                "game_date": day,
                "tipoff_utc": datetime(day.year, day.month, day.day, 18),
                "home_code": home,
                "away_code": away,
                "home_score": 80,
                "away_score": 70,
                "played": True,
                "winner_code": home,
            },
        )


class TestAccess:
    def test_the_key_is_required(self, client):
        assert client.post("/admin/refresh").status_code == 401
        assert client.post("/admin/refresh", headers={"X-API-Key": "nope"}).status_code == 401

    def test_only_post_is_allowed(self, client):
        assert client.get("/admin/refresh", headers=ADMIN_HEADERS).status_code == 405


class TestRefresh:
    def test_the_response_reports_the_data_cutoff(self, private, synthetic_league, api_clock):
        response = refresh(private.client)
        assert response.status_code == 200
        body = response.json()
        last = synthetic_league.last_played_date.isoformat()
        assert body["status"] == "refreshed"
        assert body["model_version"] == "test-xgb-v1"
        assert body["players"] == len(synthetic_league.players)
        assert body["previous_latest_played_game_date"] == last
        assert body["latest_played_game_date"] == last
        assert body["next_scheduled_game_date"] == synthetic_league.first_future_date.isoformat()
        assert body["refreshed_at"] == private.clock.moment.isoformat().replace("+00:00", "Z")
        assert isinstance(body["duration_seconds"], float) and body["duration_seconds"] >= 0

    def test_new_data_is_visible_only_after_the_refresh(self, private, synthetic_league, api_today):
        client = private.client
        old_cutoff = synthetic_league.last_played_date
        home, away = synthetic_league.teams[0], synthetic_league.teams[1]
        player = next(
            p for p in private.predictor.predict_all(as_of=api_today) if p.team_code == home
        )
        before = client.get(f"/predict/{player.player_id}?include_features=true").json()
        assert before["team_name"] == f"TEAM {home}"

        # νέο ingestion: ένας νέος παιγμένος αγώνας και αλλαγή του ονόματος της ομάδας
        add_played_game(private.engine, api_today, home, away)
        with private.engine.begin() as connection:
            connection.execute(
                update(models.teams)
                .where(models.teams.c.team_code == home)
                .values(name="RENAMED TEAM")
            )

        # η υπηρεσία δεν βλέπει ακόμη τίποτα: ούτε προβλέψεις ούτε ονόματα (cache)
        unchanged = client.get(f"/predict/{player.player_id}?include_features=true").json()
        assert unchanged == before
        health = client.get("/health").json()
        assert health["database"]["latest_played_game_date"] == api_today.isoformat()  # η βάση
        assert health["data_loaded_through"] == old_cutoff.isoformat()  # οι προβλέψεις

        body = refresh(client).json()
        assert body["previous_latest_played_game_date"] == old_cutoff.isoformat()
        assert body["latest_played_game_date"] == api_today.isoformat()  # το νέο data cutoff

        after = client.get(f"/predict/{player.player_id}?include_features=true").json()
        assert after["team_name"] == "RENAMED TEAM"
        assert after["features"] != before["features"]  # ο νέος αγώνας μπήκε στα features
        assert after["features"]["team_games_season"] == before["features"]["team_games_season"] + 1
        health = client.get("/health").json()
        assert health["data_loaded_through"] == api_today.isoformat()

    def test_the_cutoff_before_the_second_refresh_is_the_one_of_the_first(
        self, private, synthetic_league, api_today
    ):
        refresh(private.client)
        add_played_game(
            private.engine, api_today, synthetic_league.teams[0], synthetic_league.teams[1]
        )
        second = refresh(private.client).json()
        assert (
            second["previous_latest_played_game_date"]
            == synthetic_league.last_played_date.isoformat()
        )
        third = refresh(private.client).json()
        assert third["previous_latest_played_game_date"] == api_today.isoformat()
        assert third["latest_played_game_date"] == api_today.isoformat()

    def test_a_new_player_in_the_database_becomes_searchable_after_the_refresh(self, private):
        # Ο παίκτης χρειάζεται γραμμές αγώνα για να έχει πρόβλεψη: εδώ ελέγχεται μόνο ότι η
        # ανανέωση δεν αλλάζει τα ήδη γνωστά και ότι ο κατάλογος διαβάζεται ξανά από τη βάση.
        before = private.client.get("/players?search=player&limit=100").json()["total"]
        with private.engine.begin() as connection:
            connection.execute(
                update(models.players)
                .where(models.players.c.player_id == "P000006")
                .values(name="RENAMED, PLAYER")
            )
        assert private.client.get("/players?search=renamed").json()["total"] == 0
        refresh(private.client)
        assert private.client.get("/players?search=renamed").json()["total"] == 1
        assert private.client.get("/players?search=player&limit=100").json()["total"] == before

    def test_the_refresh_keeps_the_availability_records(self, private, api_today):
        player = private.predictor.predict_all(as_of=api_today)[0]
        private.client.post(
            "/availability",
            json={"player_id": player.player_id, "status": "out"},
            headers=ADMIN_HEADERS,
        )
        refresh(private.client)
        assert private.client.get(f"/predict/{player.player_id}").json()["predicted_fantasy"] == 0.0

    def test_a_database_failure_is_a_503_and_the_old_predictions_keep_working(
        self, private, monkeypatch
    ):
        def fail(self):
            raise OperationalError("SELECT secret", {}, Exception("connection lost"))

        monkeypatch.setattr(Predictor, "refresh", fail)
        response = refresh(private.client)
        assert response.status_code == 503
        assert response.json() == {"detail": "database unavailable"}
        assert "secret" not in response.text and "connection lost" not in response.text
        assert private.client.get("/rankings?limit=1").status_code == 200

    def test_a_degraded_service_cannot_be_refreshed(self, tmp_path, api_engine):
        settings = make_settings(
            database_url=api_engine.url.render_as_string(hide_password=False),
            model_path=str(tmp_path / "missing.joblib"),
        )
        with TestClient(create_app(settings=settings), raise_server_exceptions=False) as client:
            response = refresh(client)
        assert response.status_code == 503
        assert response.json() == {"detail": "prediction model is not available"}
