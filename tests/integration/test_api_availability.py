"""Tests των endpoints διαθεσιμότητας: `GET`, `POST` και `DELETE /availability`.

Ελέγχονται η προστασία με `X-API-Key` (401 και 503 όταν δεν έχει οριστεί κλειδί), το upsert, η
επικύρωση της εισόδου και η άμεση ισχύς στα `/predict` και `/rankings`.
"""

from __future__ import annotations

import secrets
import shutil
from datetime import UTC, datetime, timedelta, timezone

import pytest
from api_support import (
    ADMIN_HEADERS,
    ADMIN_KEY,
    REAL_TODAY,
    FakeClock,
    make_settings,
    noon_utc,
)
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.engine import make_url

from elfantasy.api import availability as store
from elfantasy.api import deps
from elfantasy.api.main import create_app
from elfantasy.db import models
from elfantasy.db.session import get_engine
from elfantasy.model.predict import Predictor

PROTECTED = [
    ("post", "/availability", {"json": {"player_id": "P000006", "status": "out"}}),
    ("delete", "/availability/P000006", {}),
    ("post", "/admin/refresh", {}),
]


@pytest.fixture(scope="module")
def players(api_predictor, api_today):
    """Οι ενεργοί παίκτες του συνθετικού πρωταθλήματος, με το όνομά τους."""
    return api_predictor.predict_all(as_of=api_today)


def post(client, body, headers=ADMIN_HEADERS):
    return client.post("/availability", json=body, headers=headers)


def rows_in_database(engine) -> int:
    with engine.connect() as connection:
        return connection.execute(
            select(func.count()).select_from(models.player_availability)
        ).scalar_one()


class TestAuthentication:
    @pytest.mark.parametrize(("method", "path", "kwargs"), PROTECTED)
    def test_a_missing_key_is_401(self, client, method, path, kwargs):
        response = getattr(client, method)(path, **kwargs)
        assert response.status_code == 401
        assert response.json() == {"detail": "invalid or missing API key"}

    @pytest.mark.parametrize(("method", "path", "kwargs"), PROTECTED)
    @pytest.mark.parametrize(
        "key",
        [
            "wrong",
            "",
            ADMIN_KEY + " ",
            ADMIN_KEY.upper(),
            ADMIN_KEY[:-1],
            "κλειδί".encode(),  # μη-ASCII bytes: δεν πρέπει να προκαλέσουν σφάλμα 500
        ],
    )
    def test_a_wrong_key_is_401(self, client, method, path, kwargs, key):
        response = getattr(client, method)(path, headers={"X-API-Key": key}, **kwargs)
        assert response.status_code == 401
        assert response.json() == {"detail": "invalid or missing API key"}

    def test_the_key_goes_in_the_x_api_key_header_only(self, client, api_engine):
        for variant in (
            {"headers": {"Authorization": f"Bearer {ADMIN_KEY}"}},
            {"params": {"api_key": ADMIN_KEY}},
            {"params": {"X-API-Key": ADMIN_KEY}},
        ):
            response = client.post(
                "/availability", json={"player_id": "P000006", "status": "out"}, **variant
            )
            assert response.status_code == 401, variant
        assert rows_in_database(api_engine) == 0

    def test_a_wrong_key_writes_nothing(self, client, api_engine):
        assert (
            post(client, {"player_id": "P000006", "status": "out"}, {"X-API-Key": "x"}).status_code
            == 401
        )
        assert rows_in_database(api_engine) == 0

    def test_a_correct_key_is_accepted(self, client, players):
        response = post(client, {"player_id": players[0].player_id, "status": "out"})
        assert response.status_code == 200

    def test_the_key_is_compared_in_constant_time(self, client, players, monkeypatch):
        calls = []
        original = secrets.compare_digest

        def spy(a, b):
            calls.append((a, b))
            return original(a, b)

        monkeypatch.setattr(deps.secrets, "compare_digest", spy)
        assert post(client, {"player_id": players[0].player_id, "status": "out"}).status_code == 200
        assert calls == [
            (ADMIN_KEY.encode(), ADMIN_KEY.encode())
        ]  # bytes: ασφαλές και για μη-ASCII

    def test_reading_the_list_is_public(self, client):
        assert client.get("/availability").status_code == 200

    def test_unauthenticated_requests_cannot_probe_for_players(self, client):
        for path in ("/availability/P999999", "/availability/abc", "/availability/P000006"):
            assert client.delete(path).status_code == 401, path

    def test_the_key_never_appears_in_responses(self, client, players):
        responses = [
            client.get("/availability"),
            client.post("/availability", json={"player_id": "P000006", "status": "out"}),
            post(client, {"player_id": players[0].player_id, "status": "out"}),
            client.get("/openapi.json"),
            client.get("/health"),
        ]
        for response in responses:
            assert ADMIN_KEY not in response.text
            assert ADMIN_KEY not in str(response.headers)


class TestAdminDisabled:
    """Χωρίς `ADMIN_API_KEY` τα προστατευμένα endpoints είναι κλειστά, ποτέ ανοιχτά."""

    @pytest.fixture(params=["", "   "])
    def disabled(self, request, api_engine, api_predictor, api_clock):
        app = create_app(
            settings=make_settings(admin_api_key=request.param),
            engine=api_engine,
            predictor=api_predictor,
            clock=api_clock,
        )
        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client

    @pytest.mark.parametrize(("method", "path", "kwargs"), PROTECTED)
    @pytest.mark.parametrize("headers", [{}, ADMIN_HEADERS, {"X-API-Key": ""}, {"X-API-Key": " "}])
    def test_the_protected_endpoints_answer_503(self, disabled, method, path, kwargs, headers):
        response = getattr(disabled, method)(path, headers=headers, **kwargs)
        assert response.status_code == 503
        assert response.json() == {"detail": "admin API is disabled"}

    def test_nothing_is_written_and_reading_still_works(self, disabled, api_engine):
        post(disabled, {"player_id": "P000006", "status": "out"})
        assert rows_in_database(api_engine) == 0
        assert disabled.get("/availability").json() == {"total": 0, "items": []}
        assert disabled.get("/health").status_code == 200


class TestUpsert:
    def test_a_new_record(self, client, players, api_clock, api_engine):
        player = players[0]
        response = post(
            client,
            {
                "player_id": player.player_id,
                "status": "out",
                "source": "club statement",
                "note": "ankle sprain",
                "expected_return": "2026-10-20",
            },
        )
        assert response.status_code == 200
        assert response.json() == {
            "player_id": player.player_id,
            "name": player.name,
            "status": "out",
            "source": "club statement",
            "note": "ankle sprain",
            "expected_return": "2026-10-20",
            "updated_at": api_clock.moment.isoformat().replace("+00:00", "Z"),
        }
        assert rows_in_database(api_engine) == 1

    def test_only_the_status_is_required(self, client, players):
        body = post(client, {"player_id": players[0].player_id, "status": "doubtful"}).json()
        assert body["status"] == "doubtful"
        assert body["source"] is None and body["note"] is None and body["expected_return"] is None

    def test_a_second_post_replaces_the_whole_record(self, client, players, api_clock, api_engine):
        player_id = players[0].player_id
        post(
            client,
            {
                "player_id": player_id,
                "status": "out",
                "source": "club",
                "note": "knee",
                "expected_return": "2026-11-01",
            },
        )
        api_clock.advance(hours=5)
        second = post(client, {"player_id": player_id, "status": "doubtful"}).json()
        assert second["status"] == "doubtful"
        assert second["source"] is None and second["note"] is None  # δεν συγχωνεύεται
        assert second["expected_return"] is None
        assert second["updated_at"] == api_clock.moment.isoformat().replace("+00:00", "Z")
        assert rows_in_database(api_engine) == 1  # ένα upsert, όχι δεύτερη γραμμή
        listed = client.get("/availability").json()
        assert listed["total"] == 1 and listed["items"][0] == second

    def test_the_same_request_twice_is_idempotent(self, client, players):
        body = {"player_id": players[0].player_id, "status": "out", "note": "x"}
        first, second = post(client, body).json(), post(client, body).json()
        assert first == second
        assert client.get("/availability").json()["total"] == 1

    def test_the_player_id_is_normalised(self, client, players):
        player = players[0]
        body = post(client, {"player_id": f"  {player.player_id.lower()} ", "status": "out"}).json()
        assert body["player_id"] == player.player_id

    def test_updated_at_is_set_by_the_server_and_cannot_be_supplied(self, client, players):
        response = post(
            client,
            {
                "player_id": players[0].player_id,
                "status": "out",
                "updated_at": "2001-01-01T00:00:00Z",
            },
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["type"] == "extra_forbidden"

    def test_text_fields_are_trimmed(self, client, players):
        body = post(
            client,
            {
                "player_id": players[0].player_id,
                "status": "out",
                "source": "  club ",
                "note": "  hi  ",
            },
        ).json()
        assert body["source"] == "club" and body["note"] == "hi"

    def test_a_blank_text_becomes_null(self, client, players):
        body = post(
            client,
            {"player_id": players[0].player_id, "status": "out", "source": "", "note": "   \n"},
        ).json()
        assert body["source"] is None and body["note"] is None

    def test_explicit_nulls_are_accepted(self, client, players):
        body = post(
            client,
            {
                "player_id": players[0].player_id,
                "status": "out",
                "source": None,
                "note": None,
                "expected_return": None,
            },
        )
        assert body.status_code == 200

    def test_the_updated_at_of_the_record_is_in_utc_whatever_the_clock_zone(
        self, client, players, api_clock
    ):
        api_clock.moment = datetime(2026, 10, 3, 15, 30, tzinfo=timezone(timedelta(hours=3)))
        body = post(client, {"player_id": players[0].player_id, "status": "out"}).json()
        assert body["updated_at"] == "2026-10-03T12:30:00Z"
        assert api_clock.moment.astimezone(UTC).hour == 12


class TestValidation:
    def test_an_unknown_player_is_404_and_nothing_is_stored(self, client, api_engine):
        response = post(client, {"player_id": "P999999", "status": "out"})
        assert response.status_code == 404
        assert response.json()["detail"].startswith("player not found")
        assert rows_in_database(api_engine) == 0

    @pytest.mark.parametrize("bad_id", ["abc", "P12", "P1234567", "", "   ", "X000001", "P00-001"])
    def test_a_player_id_of_the_wrong_shape_is_422(self, client, bad_id):
        response = post(client, {"player_id": bad_id, "status": "out"})
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "player_id"]

    @pytest.mark.parametrize("bad_id", [None, 7, ["P000006"], {"id": 1}])
    def test_a_player_id_of_the_wrong_type_is_422(self, client, bad_id):
        assert post(client, {"player_id": bad_id, "status": "out"}).status_code == 422

    @pytest.mark.parametrize("status", ["injured", "OUT", "Out", "", None, 1, True, ["out"]])
    def test_an_invalid_status_is_422(self, client, players, status):
        response = post(client, {"player_id": players[0].player_id, "status": status})
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "status"]

    def test_a_missing_status_or_player_is_422(self, client, players):
        assert post(client, {"player_id": players[0].player_id}).status_code == 422
        assert post(client, {"status": "out"}).status_code == 422

    def test_the_note_is_limited_to_500_characters(self, client, players):
        player_id = players[0].player_id
        ok = post(client, {"player_id": player_id, "status": "out", "note": "x" * 500})
        assert ok.status_code == 200 and len(ok.json()["note"]) == 500
        too_long = post(client, {"player_id": player_id, "status": "out", "note": "x" * 501})
        assert too_long.status_code == 422
        assert too_long.json()["detail"][0]["loc"] == ["body", "note"]
        # τα κενά στα άκρα δεν μετρούν: το όριο ελέγχεται μετά το strip
        padded = post(client, {"player_id": player_id, "status": "out", "note": f" {'x' * 500} "})
        assert padded.status_code == 200

    def test_a_failed_validation_does_not_change_the_stored_record(self, client, players):
        player_id = players[0].player_id
        post(client, {"player_id": player_id, "status": "out", "note": "original"})
        assert (
            post(client, {"player_id": player_id, "status": "out", "note": "x" * 501}).status_code
            == 422
        )
        assert client.get("/availability").json()["items"][0]["note"] == "original"

    def test_the_source_is_limited_to_100_characters(self, client, players):
        player_id = players[0].player_id
        assert (
            post(client, {"player_id": player_id, "status": "out", "source": "s" * 100}).status_code
            == 200
        )
        assert (
            post(client, {"player_id": player_id, "status": "out", "source": "s" * 101}).status_code
            == 422
        )

    @pytest.mark.parametrize("value", ["2026-10-20", "2099-12-31", "2000-01-01", " 2026-10-20 "])
    def test_valid_dates(self, client, players, value):
        body = post(
            client, {"player_id": players[0].player_id, "status": "out", "expected_return": value}
        )
        assert body.status_code == 200
        assert body.json()["expected_return"] == value.strip()

    @pytest.mark.parametrize(
        "value",
        [
            "2026-13-45",
            "2026-02-30",
            "tomorrow",
            "10/10/2026",
            "2026-10-20T00:00:00",
            "",
            20261020,
            1.5,
            True,
            ["2026-10-20"],
        ],
    )
    def test_an_invalid_date_is_422(self, client, players, value):
        response = post(
            client, {"player_id": players[0].player_id, "status": "out", "expected_return": value}
        )
        assert response.status_code == 422, value
        assert response.json()["detail"][0]["loc"] == ["body", "expected_return"]

    def test_unknown_fields_are_rejected(self, client, players):
        response = post(
            client, {"player_id": players[0].player_id, "status": "out", "notes": "typo"}
        )
        assert response.status_code == 422

    @pytest.mark.parametrize("content", [b"not json", b"[1, 2]", b'"text"', b"", b"null"])
    def test_the_body_must_be_a_json_object(self, client, content):
        response = client.post(
            "/availability",
            content=content,
            headers={**ADMIN_HEADERS, "Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_the_content_type_is_checked(self, client, players):
        response = client.post(
            "/availability",
            content=f'{{"player_id": "{players[0].player_id}", "status": "out"}}',
            headers={**ADMIN_HEADERS, "Content-Type": "text/plain"},
        )
        assert response.status_code == 422


class TestList:
    def test_an_empty_list(self, client):
        assert client.get("/availability").json() == {"total": 0, "items": []}

    def test_the_order_is_out_then_doubtful_then_available_and_by_name(self, client, players):
        by_name = sorted(players[:6], key=lambda p: p.name)
        a, b, c, d = by_name[:4]
        for player, status in ((d, "available"), (b, "out"), (c, "doubtful"), (a, "out")):
            post(client, {"player_id": player.player_id, "status": status})
        body = client.get("/availability").json()
        assert body["total"] == 4
        assert [(i["player_id"], i["status"]) for i in body["items"]] == [
            (a.player_id, "out"),
            (b.player_id, "out"),
            (c.player_id, "doubtful"),
            (d.player_id, "available"),
        ]
        assert [i["name"] for i in body["items"]] == [a.name, b.name, c.name, d.name]

    def test_the_status_filter(self, client, players):
        for player, status in zip(
            players[:4], ("out", "out", "doubtful", "available"), strict=True
        ):
            post(client, {"player_id": player.player_id, "status": status})
        counts = {
            status: client.get(f"/availability?status={status}").json()["total"]
            for status in ("out", "doubtful", "available")
        }
        assert counts == {"out": 2, "doubtful": 1, "available": 1}
        only_out = client.get("/availability?status=out").json()["items"]
        assert {item["status"] for item in only_out} == {"out"}
        assert client.get("/availability").json()["total"] == 4

    @pytest.mark.parametrize("status", ["injured", "OUT", ""])
    def test_an_invalid_status_filter_is_422(self, client, status):
        assert client.get(f"/availability?status={status}").status_code == 422


class TestDelete:
    def test_a_record_is_removed(self, client, players, api_engine):
        player_id = players[0].player_id
        post(client, {"player_id": player_id, "status": "out"})
        response = client.delete(f"/availability/{player_id}", headers=ADMIN_HEADERS)
        assert response.status_code == 204
        assert response.content == b""
        assert rows_in_database(api_engine) == 0
        assert client.get("/availability").json()["total"] == 0

    def test_only_the_given_player_is_removed(self, client, players):
        first, second = players[0].player_id, players[1].player_id
        post(client, {"player_id": first, "status": "out"})
        post(client, {"player_id": second, "status": "doubtful"})
        client.delete(f"/availability/{first}", headers=ADMIN_HEADERS)
        assert [i["player_id"] for i in client.get("/availability").json()["items"]] == [second]

    def test_removing_twice_is_404_the_second_time(self, client, players):
        player_id = players[0].player_id
        post(client, {"player_id": player_id, "status": "out"})
        assert client.delete(f"/availability/{player_id}", headers=ADMIN_HEADERS).status_code == 204
        again = client.delete(f"/availability/{player_id}", headers=ADMIN_HEADERS)
        assert again.status_code == 404
        assert again.json() == {"detail": "no availability record for this player"}

    def test_a_player_without_a_record_is_404(self, client, players):
        response = client.delete(f"/availability/{players[0].player_id}", headers=ADMIN_HEADERS)
        assert response.status_code == 404

    def test_an_unknown_player_is_404(self, client):
        response = client.delete("/availability/P999999", headers=ADMIN_HEADERS)
        assert response.status_code == 404
        assert response.json()["detail"].startswith("player not found")

    def test_an_invalid_id_is_422(self, client):
        response = client.delete("/availability/abc", headers=ADMIN_HEADERS)
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["path", "player_id"]

    def test_the_id_is_normalised(self, client, players):
        player_id = players[0].player_id
        post(client, {"player_id": player_id, "status": "out"})
        response = client.delete(f"/availability/%20{player_id.lower()}%20", headers=ADMIN_HEADERS)
        assert response.status_code == 204


class TestEffect:
    def test_the_override_is_effective_immediately_in_predict_and_rankings(self, client, players):
        top = players[0]
        assert client.get(f"/predict/{top.player_id}").json()["predicted_fantasy"] > 0
        post(client, {"player_id": top.player_id, "status": "out"})
        assert client.get(f"/predict/{top.player_id}").json()["predicted_fantasy"] == 0.0
        ranking_ids = [i["player_id"] for i in client.get("/rankings?limit=500").json()["items"]]
        assert top.player_id not in ranking_ids
        client.delete(f"/availability/{top.player_id}", headers=ADMIN_HEADERS)
        assert client.get(f"/predict/{top.player_id}").json()["predicted_fantasy"] > 0
        ranking_ids = [i["player_id"] for i in client.get("/rankings?limit=500").json()["items"]]
        assert ranking_ids[0] == top.player_id

    def test_the_override_does_not_reach_the_model(self, client, players, monkeypatch):
        """Το μοντέλο δεν βλέπει τη διαθεσιμότητα: η πρόβλεψη είναι ίδια με και χωρίς override."""
        seen = []
        original = Predictor.predict_player

        def spy(self, player_id, as_of=None):
            result = original(self, player_id, as_of)
            seen.append(result.predicted_fantasy)
            return result

        monkeypatch.setattr(Predictor, "predict_player", spy)
        player_id = players[0].player_id
        client.get(f"/predict/{player_id}")
        post(client, {"player_id": player_id, "status": "out"})
        client.get(f"/predict/{player_id}")
        assert len(seen) == 2 and seen[0] == seen[1] > 0


class TestOldStyleIds:
    """Παίκτες με ID παλαιάς μορφής (`PLRU`): αντίγραφο της βάσης των πραγματικών fixtures."""

    @pytest.fixture
    def copy_client(self, fixture_database_url, tmp_path, tiny_xgb_bundle):
        source = make_url(fixture_database_url).database
        target = tmp_path / "copy.db"
        shutil.copy(source, target)
        engine = get_engine(f"sqlite:///{target.as_posix()}")
        predictor = Predictor(tiny_xgb_bundle, engine, today=lambda: REAL_TODAY)
        app = create_app(
            settings=make_settings(),
            engine=engine,
            predictor=predictor,
            clock=FakeClock(noon_utc(REAL_TODAY)),
        )
        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client
        engine.dispose()

    def test_an_old_style_id_can_be_marked_and_removed(self, copy_client):
        body = post(copy_client, {"player_id": " plru ", "status": "out"})
        assert body.status_code == 200
        assert body.json()["player_id"] == "PLRU"
        assert body.json()["name"] == "SIMONOVIC, MARKO"
        predicted = copy_client.get("/predict/PLRU").json()
        assert predicted["predicted_fantasy"] == 0.0
        assert predicted["availability"]["status"] == "out"
        assert copy_client.delete("/availability/plru", headers=ADMIN_HEADERS).status_code == 204
        assert copy_client.get("/predict/PLRU").json()["availability"]["status"] == "available"

    def test_the_table_is_created_at_startup_when_it_is_missing(self, copy_client):
        # Το αντίγραφο ξεκίνησε από βάση που δημιουργήθηκε από το ingestion (με τον πίνακα): το
        # `ensure_table` είναι ασφαλές να ξανατρέξει και δεν χαλά τα δεδομένα.
        post(copy_client, {"player_id": "P003469", "status": "doubtful"})
        assert copy_client.get("/availability").json()["total"] == 1
        engine = copy_client.app.state.api.engine
        store.ensure_table(engine)
        store.ensure_table(engine)
        assert copy_client.get("/availability").json()["total"] == 1
