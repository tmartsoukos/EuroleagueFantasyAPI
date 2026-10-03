"""Tests της διαχείρισης σφαλμάτων: καθαρό JSON `{"detail": ...}` για κάθε σφάλμα και καμία
εσωτερική λεπτομέρεια (stack trace, διαδρομές, SQL) προς τον client. Οι λεπτομέρειες γράφονται
μόνο στο log του server.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from api_support import ADMIN_HEADERS
from sqlalchemy.exc import IntegrityError, OperationalError

from elfantasy.api.availability import save_availability
from elfantasy.api.services import NotFoundError, PredictionService

SECRET = "secret-internal-detail C:/Users/secret/project/file.py"


def assert_clean(response, status: int) -> None:
    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert set(body) == {"detail"}
    for forbidden in ("Traceback", 'File "', SECRET, "secret", "sqlalchemy", "sqlite", ".py"):
        assert forbidden not in response.text, forbidden


class TestUnexpectedErrors:
    def test_an_unexpected_error_is_a_generic_500_and_is_logged(self, client, monkeypatch, caplog):
        def boom(self, **kwargs):
            raise RuntimeError(SECRET)

        monkeypatch.setattr(PredictionService, "rankings", boom)
        with caplog.at_level("ERROR"):
            response = client.get("/rankings")
        assert_clean(response, 500)
        assert response.json() == {"detail": "internal server error"}
        assert "unhandled error while handling GET /rankings" in caplog.text
        assert SECRET in caplog.text  # η λεπτομέρεια μένει στον server

    def test_an_unexpected_error_in_a_write_endpoint(self, client, monkeypatch):
        def boom(*args, **kwargs):
            raise ValueError(SECRET)

        monkeypatch.setattr("elfantasy.api.routers.availability.store.save_availability", boom)
        response = client.post(
            "/availability",
            json={
                "player_id": client.get("/rankings?limit=1").json()["items"][0]["player_id"],
                "status": "out",
            },
            headers=ADMIN_HEADERS,
        )
        assert_clean(response, 500)
        assert response.json() == {"detail": "internal server error"}

    def test_a_database_error_is_a_503_without_details(self, client, monkeypatch, caplog):
        def broken(self, **kwargs):
            raise OperationalError(f"SELECT {SECRET}", {}, Exception(SECRET))

        monkeypatch.setattr(PredictionService, "search_players", broken)
        with caplog.at_level("ERROR"):
            response = client.get("/players?search=x")
        assert_clean(response, 503)
        assert response.json() == {"detail": "database unavailable"}
        assert "database error while handling GET /players" in caplog.text

    def test_an_integrity_error_is_not_leaked(self, client, monkeypatch):
        def broken(*args, **kwargs):
            raise IntegrityError("INSERT INTO player_availability", {}, Exception(SECRET))

        monkeypatch.setattr("elfantasy.api.routers.availability.store.save_availability", broken)
        player_id = client.get("/rankings?limit=1").json()["items"][0]["player_id"]
        response = client.post(
            "/availability", json={"player_id": player_id, "status": "out"}, headers=ADMIN_HEADERS
        )
        assert_clean(response, 503)

    def test_a_domain_not_found_error_is_a_404(self, client, monkeypatch):
        def missing(self, player_id, **kwargs):
            raise NotFoundError("something is missing")

        monkeypatch.setattr(PredictionService, "predict", missing)
        response = client.get("/predict/P000006")
        assert response.status_code == 404
        assert response.json() == {"detail": "something is missing"}


class TestErrorFormat:
    @pytest.mark.parametrize(
        ("method", "path", "kwargs", "status"),
        [
            ("get", "/no/such/path", {}, 404),
            ("get", "/predict/P999999", {}, 404),
            ("get", "/predict/abc", {}, 422),
            ("post", "/predict/P000006", {}, 405),
            ("get", "/availability/P000006", {}, 405),
            ("post", "/availability", {"json": {"player_id": "P000006", "status": "out"}}, 401),
            ("get", "/rankings?limit=0", {}, 422),
            ("get", "/rankings?team=ZZZ", {}, 404),
        ],
    )
    def test_every_error_is_json_with_a_detail(self, client, method, path, kwargs, status):
        response = getattr(client, method)(path, **kwargs)
        assert response.status_code == status
        assert response.headers["content-type"].startswith("application/json")
        assert "detail" in response.json()
        assert "Traceback" not in response.text

    def test_a_validation_error_has_a_loc_a_message_and_a_type(self, client):
        (error,) = client.get("/rankings?limit=0").json()["detail"]
        assert error["loc"] == ["query", "limit"]
        assert error["type"] == "greater_than_equal"
        assert error["msg"]

    def test_the_failure_of_the_repository_is_not_hidden_as_a_success(self, client, api_engine):
        # Ένας παίκτης που δεν υπάρχει παραβιάζει το foreign key· το API ελέγχει πρώτα την ύπαρξη
        # (404), αλλά το repository από μόνο του αρνείται την εγγραφή.
        with pytest.raises(IntegrityError):
            save_availability(
                api_engine,
                player_id="P999999",
                status="out",
                source=None,
                note=None,
                expected_return=None,
                updated_at=datetime(2026, 1, 1),
            )
