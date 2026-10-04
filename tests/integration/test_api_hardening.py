"""Tests σκλήρυνσης του API (ευρήματα του review της Φάσης 7): όριο σώματος αιτήματος πριν από τον
έλεγχο κλειδιού, ελάχιστο μήκος και κωδικοποίηση του `ADMIN_API_KEY`, και απόρριψη χαρακτήρων
που δεν αποθηκεύονται (NUL, μεμονωμένα surrogates) στο `POST /availability`.
"""

from __future__ import annotations

import logging

import pytest
from api_support import ADMIN_HEADERS, make_settings
from fastapi.testclient import TestClient

from elfantasy.api.limits import MAX_REQUEST_BODY_BYTES
from elfantasy.api.main import create_app
from elfantasy.api.security import MIN_ADMIN_KEY_LENGTH

JSON = {"Content-Type": "application/json"}


@pytest.fixture
def player_id(client) -> str:
    return client.get("/rankings", params={"limit": 1}).json()["items"][0]["player_id"]


def availability_body(player_id: str, **extra) -> dict:
    return {"player_id": player_id, "status": "out", **extra}


class TestBodySizeLimit:
    def test_a_huge_body_without_a_key_is_413_not_401(self, client):
        """Το όριο ισχύει πριν από τον έλεγχο κλειδιού: το σώμα δεν διαβάζεται καν."""
        response = client.post(
            "/availability", content=b"x" * (MAX_REQUEST_BODY_BYTES + 1), headers=JSON
        )
        assert response.status_code == 413
        assert response.json() == {"detail": "request body too large"}

    def test_a_huge_body_with_the_right_key_is_413_too(self, client):
        response = client.post(
            "/availability",
            content=b"x" * (MAX_REQUEST_BODY_BYTES + 1),
            headers={**ADMIN_HEADERS, **JSON},
        )
        assert response.status_code == 413

    def test_a_huge_body_with_a_wrong_key_is_413(self, client):
        response = client.post(
            "/availability",
            content=b"x" * (4 * MAX_REQUEST_BODY_BYTES),
            headers={"X-API-Key": "wrong", **JSON},
        )
        assert response.status_code == 413

    def test_a_huge_streamed_body_is_413(self, client):
        def stream():
            for _ in range(40):
                yield b"x" * 4096

        response = client.post("/availability", content=stream(), headers=JSON)
        assert response.status_code == 413
        assert response.json() == {"detail": "request body too large"}

    def test_a_huge_json_value_is_413_before_it_is_parsed(self, client, player_id):
        body = availability_body(player_id, note="y" * (2 * MAX_REQUEST_BODY_BYTES))
        response = client.post("/availability", json=body, headers=ADMIN_HEADERS)
        assert response.status_code == 413

    def test_every_route_has_the_limit(self, client):
        for method, path in [
            ("post", "/admin/refresh"),
            ("delete", "/availability/P000001"),
            ("get", "/health"),
            ("get", "/rankings"),
        ]:
            response = client.request(
                method.upper(),
                path,
                content=b"x" * (MAX_REQUEST_BODY_BYTES + 1),
                headers=ADMIN_HEADERS,
            )
            assert response.status_code == 413, (method, path)

    def test_a_normal_request_still_works(self, client, player_id):
        body = availability_body(player_id, note="ankle", source="club")
        response = client.post("/availability", json=body, headers=ADMIN_HEADERS)
        assert response.status_code == 200 and response.json()["status"] == "out"
        assert client.get("/health").status_code == 200

    def test_a_body_at_the_limit_is_read_normally(self, client):
        """Στο όριο το σώμα διαβάζεται κανονικά: εδώ δεν είναι έγκυρο JSON, άρα 422 (όχι 413)."""
        response = client.post(
            "/availability",
            content=b"x" * MAX_REQUEST_BODY_BYTES,
            headers={**ADMIN_HEADERS, **JSON},
        )
        assert response.status_code == 422

    def test_a_request_just_over_the_limit_never_reaches_the_database(self, client, api_engine):
        from sqlalchemy import func, select

        from elfantasy.db import models

        client.post(
            "/availability", content=b"x" * (MAX_REQUEST_BODY_BYTES + 1), headers=ADMIN_HEADERS
        )
        with api_engine.connect() as connection:
            count = connection.execute(
                select(func.count()).select_from(models.player_availability)
            ).scalar_one()
        assert count == 0


class TestAdminKeyStrength:
    def make_client(self, api_engine, api_predictor, api_clock, key: str):
        app = create_app(
            settings=make_settings(admin_api_key=key),
            engine=api_engine,
            predictor=api_predictor,
            clock=api_clock,
        )
        return TestClient(app, raise_server_exceptions=False)

    @pytest.mark.parametrize("key", ["1234", "short-key", "x" * (MIN_ADMIN_KEY_LENGTH - 1)])
    def test_a_short_key_never_opens_the_admin_endpoints(
        self, api_engine, api_predictor, api_clock, key
    ):
        with self.make_client(api_engine, api_predictor, api_clock, key) as short_client:
            for sent in (key, "wrong", None):
                headers = {} if sent is None else {"X-API-Key": sent}
                response = short_client.post("/admin/refresh", headers=headers)
                assert response.status_code == 503
                assert response.json() == {"detail": "admin API is disabled"}
            assert short_client.get("/health").status_code == 200  # το υπόλοιπο API δουλεύει

    def test_the_minimum_length_works(self, api_engine, api_predictor, api_clock):
        key = "k" * MIN_ADMIN_KEY_LENGTH
        with self.make_client(api_engine, api_predictor, api_clock, key) as strong_client:
            assert (
                strong_client.post("/admin/refresh", headers={"X-API-Key": key}).status_code == 200
            )

    def test_the_startup_warns_about_a_short_key_without_logging_it(
        self, api_engine, api_predictor, api_clock, caplog
    ):
        secret = "tiny-secret"
        with caplog.at_level(logging.ERROR, logger="elfantasy.api.state"):
            with self.make_client(api_engine, api_predictor, api_clock, secret):
                pass
        assert any("ADMIN_API_KEY is shorter than" in r.getMessage() for r in caplog.records)
        assert secret not in caplog.text

    def test_a_good_or_missing_key_does_not_warn(
        self, api_engine, api_predictor, api_clock, caplog
    ):
        with caplog.at_level(logging.ERROR, logger="elfantasy.api.state"):
            for key in ("", "g" * 40):
                with self.make_client(api_engine, api_predictor, api_clock, key):
                    pass
        assert "ADMIN_API_KEY" not in caplog.text

    def test_a_non_ascii_key_sent_as_utf_8_works_over_http(
        self, api_engine, api_predictor, api_clock
    ):
        key = "κλειδί-ασφαλείας-δοκιμής-2026"
        with self.make_client(api_engine, api_predictor, api_clock, key) as greek_client:
            ok = greek_client.post("/admin/refresh", headers={"X-API-Key": key.encode("utf-8")})
            bad = greek_client.post(
                "/admin/refresh", headers={"X-API-Key": (key[:-1] + "x").encode("utf-8")}
            )
        assert ok.status_code == 200
        assert bad.status_code == 401


class TestUnstorableText:
    @pytest.mark.parametrize("field", ["note", "source"])
    def test_a_nul_character_is_422(self, client, player_id, field):
        response = client.post(
            "/availability",
            json=availability_body(player_id, **{field: "a\u0000b"}),
            headers=ADMIN_HEADERS,
        )
        assert response.status_code == 422
        assert "NUL" in response.text

    @pytest.mark.parametrize("field", ["note", "source"])
    def test_a_lone_surrogate_is_422_not_500(self, client, player_id, field):
        raw = f'{{"player_id": "{player_id}", "status": "out", "{field}": "x\\ud800y"}}'
        response = client.post(
            "/availability", content=raw.encode("ascii"), headers={**ADMIN_HEADERS, **JSON}
        )
        assert response.status_code == 422
        assert "ud800" in response.text  # το σφάλμα επιστρέφεται με escaped τιμή, όχι 500

    def test_the_rejection_stores_nothing(self, client, player_id):
        client.post(
            "/availability",
            json=availability_body(player_id, note="a\u0000b"),
            headers=ADMIN_HEADERS,
        )
        assert client.get("/availability").json()["total"] == 0

    @pytest.mark.parametrize("text", ["ankle", "τραυματισμός", "😀 emoji", "tab\tand\nnewline"])
    def test_ordinary_unicode_text_is_stored(self, client, player_id, text):
        response = client.post(
            "/availability", json=availability_body(player_id, note=text), headers=ADMIN_HEADERS
        )
        assert response.status_code == 200
        assert response.json()["note"] == text.strip()
