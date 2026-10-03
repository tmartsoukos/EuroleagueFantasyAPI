"""Tests του `GET /rankings`: προεπιλογές, σελιδοποίηση και συνεχόμενο rank, φίλτρο ομάδας,
ταξινόμηση, εξαίρεση των `out`, `include_unavailable`, `active_only`, κενό αποτέλεσμα, όρια
παραμέτρων και ακριβείς κανόνες ταξινόμησης με ψεύτικο Predictor.
"""

from __future__ import annotations

import math
from datetime import date

import pytest
from api_support import (
    ADMIN_HEADERS,
    FakeClock,
    FakePredictor,
    make_next_game,
    make_prediction,
    make_settings,
    make_small_database,
    noon_utc,
)
from fastapi.testclient import TestClient

from elfantasy.api.main import create_app
from elfantasy.model.predict import Predictor


@pytest.fixture(scope="module")
def active(api_predictor, api_today):
    """Οι ενεργοί παίκτες, ταξινομημένοι όπως τους δίνει ο Predictor (φθίνουσα κατά fantasy)."""
    return api_predictor.predict_all(as_of=api_today)


def mark(client, player_id, status, **extra):
    response = client.post(
        "/availability",
        json={"player_id": player_id, "status": status, **extra},
        headers=ADMIN_HEADERS,
    )
    assert response.status_code == 200, response.text


class TestDefaults:
    def test_the_defaults(self, client, active, api_today):
        response = client.get("/rankings")
        assert response.status_code == 200
        body = response.json()
        assert body["meta"] == {
            "as_of": api_today.isoformat(),
            "model_version": "test-xgb-v1",
            "total": len(active),
            "limit": 50,
            "offset": 0,
        }
        items = body["items"]
        assert len(items) == min(50, len(active))
        assert [item["rank"] for item in items] == list(range(1, len(items) + 1))
        assert [item["player_id"] for item in items] == [p.player_id for p in active[:50]]
        assert all(item["is_active"] for item in items)
        assert all(item["availability_status"] == "available" for item in items)

    def test_the_fields_of_an_item(self, client, active):
        top = active[0]
        item = client.get("/rankings?limit=1").json()["items"][0]
        assert item == {
            "rank": 1,
            "player_id": top.player_id,
            "name": top.name,
            "team_code": top.team_code,
            "predicted_fantasy": round(top.predicted_fantasy, 2),
            "model_predicted_fantasy": round(top.predicted_fantasy, 2),
            "predicted_pir": round(top.predicted_pir, 2),
            "next_opponent_code": top.next_game.opp_code,
            "next_home": top.next_game.home,
            "next_game_date": top.next_game.game_date.isoformat(),
            "availability_status": "available",
            "is_active": True,
        }

    def test_the_order_is_descending_by_predicted_fantasy(self, client):
        items = client.get("/rankings?limit=500").json()["items"]
        values = [item["predicted_fantasy"] for item in items]
        assert values == sorted(values, reverse=True)

    def test_without_a_next_game_the_next_fields_are_null(self, client, api_clock):
        api_clock.advance(days=1000)
        body = client.get("/rankings?limit=3").json()
        assert len(body["items"]) == 3
        for item in body["items"]:
            assert item["next_opponent_code"] is None
            assert item["next_home"] is None
            assert item["next_game_date"] is None


class TestPagination:
    def test_limit_and_offset_with_a_continuous_rank(self, client, active):
        body = client.get("/rankings?limit=5&offset=3").json()
        assert body["meta"]["limit"] == 5 and body["meta"]["offset"] == 3
        assert body["meta"]["total"] == len(active)
        items = body["items"]
        assert [item["rank"] for item in items] == [4, 5, 6, 7, 8]
        assert [item["player_id"] for item in items] == [p.player_id for p in active[3:8]]

    def test_consecutive_pages_cover_the_list_without_gaps(self, client, active):
        seen = []
        for offset in range(0, len(active), 10):
            page = client.get(f"/rankings?limit=10&offset={offset}").json()
            seen.extend((item["rank"], item["player_id"]) for item in page["items"])
        assert seen == [(index + 1, p.player_id) for index, p in enumerate(active)]

    def test_an_offset_beyond_the_end_is_an_empty_page(self, client, active):
        body = client.get(f"/rankings?offset={len(active) + 10}").json()
        assert body["items"] == []
        assert body["meta"]["total"] == len(active)

    def test_the_maximum_limit_is_accepted(self, client, active):
        response = client.get("/rankings?limit=500")
        assert response.status_code == 200
        assert len(response.json()["items"]) == len(active)

    @pytest.mark.parametrize(
        "query",
        [
            "limit=0",
            "limit=-1",
            "limit=501",
            "limit=abc",
            "limit=",
            "offset=-1",
            "offset=abc",
            "include_unavailable=maybe",
            "active_only=2",
            "team=AB",
            "team=ABCD",
            "team=T-3",
            "team=%20T03",
        ],
    )
    def test_invalid_parameters_are_422(self, client, query):
        response = client.get(f"/rankings?{query}")
        assert response.status_code == 422, query
        assert isinstance(response.json()["detail"], list)


class TestTeamFilter:
    def test_a_team_filter_in_any_case(self, client, active):
        team = active[0].team_code
        expected = [p.player_id for p in active if p.team_code == team]
        for variant in (team, team.lower()):
            body = client.get(f"/rankings?team={variant}&limit=500").json()
            assert body["meta"]["total"] == len(expected)
            assert [item["player_id"] for item in body["items"]] == expected
            assert {item["team_code"] for item in body["items"]} == {team}
            assert [item["rank"] for item in body["items"]] == list(range(1, len(expected) + 1))

    def test_an_unknown_team_code_is_404_and_lists_the_valid_codes(self, client, synthetic_league):
        response = client.get("/rankings?team=ZZZ")
        assert response.status_code == 404
        detail = response.json()["detail"]
        assert detail.startswith("unknown team code 'ZZZ'; valid codes: ")
        assert detail.endswith(", ".join(sorted(synthetic_league.teams)))

    def test_a_team_with_everybody_out_is_an_empty_result(self, client, active):
        team = active[0].team_code
        for prediction in active:
            if prediction.team_code == team:
                mark(client, prediction.player_id, "out")
        response = client.get(f"/rankings?team={team}")
        assert response.status_code == 200
        assert response.json()["items"] == []
        assert response.json()["meta"]["total"] == 0
        with_out = client.get(f"/rankings?team={team}&include_unavailable=true").json()
        assert with_out["meta"]["total"] == sum(1 for p in active if p.team_code == team)


class TestAvailabilityInRankings:
    def test_out_players_are_excluded_by_default(self, client, active):
        top, second = active[0], active[1]
        mark(client, top.player_id, "out")
        body = client.get("/rankings?limit=500").json()
        ids = [item["player_id"] for item in body["items"]]
        assert top.player_id not in ids
        assert ids[0] == second.player_id
        assert body["meta"]["total"] == len(active) - 1
        assert [item["rank"] for item in body["items"]] == list(range(1, len(active)))

    def test_include_unavailable_lists_out_players_last_with_a_zero_value(self, client, active):
        top = active[0]
        mark(client, top.player_id, "out", note="ankle")
        body = client.get("/rankings?include_unavailable=true&limit=500").json()
        items = body["items"]
        assert body["meta"]["total"] == len(active)
        last = items[-1]
        assert last["player_id"] == top.player_id
        assert last["rank"] == len(active)
        assert last["predicted_fantasy"] == 0.0
        assert last["model_predicted_fantasy"] == round(top.predicted_fantasy, 2)
        assert last["predicted_pir"] == round(top.predicted_pir, 2)
        assert last["availability_status"] == "out"
        assert items[0]["player_id"] == active[1].player_id

    def test_doubtful_players_stay_in_place_and_are_flagged(self, client, active):
        top = active[0]
        mark(client, top.player_id, "doubtful")
        body = client.get("/rankings?limit=3").json()
        first = body["items"][0]
        assert first["player_id"] == top.player_id and first["rank"] == 1
        assert first["availability_status"] == "doubtful"
        assert first["predicted_fantasy"] == first["model_predicted_fantasy"]
        assert body["items"][1]["availability_status"] == "available"

    def test_an_available_record_changes_nothing(self, client, active):
        mark(client, active[0].player_id, "available")
        body = client.get("/rankings?limit=500").json()
        assert body["meta"]["total"] == len(active)
        assert [item["player_id"] for item in body["items"]] == [p.player_id for p in active]

    def test_the_override_applies_immediately_and_can_be_removed(self, client, active):
        top = active[0]
        before = client.get("/rankings?limit=500").json()
        mark(client, top.player_id, "out")
        assert top.player_id not in [
            i["player_id"] for i in client.get("/rankings").json()["items"]
        ]
        assert (
            client.delete(f"/availability/{top.player_id}", headers=ADMIN_HEADERS).status_code
            == 204
        )
        assert client.get("/rankings?limit=500").json() == before


class TestActiveOnly:
    def test_everybody_with_active_only_false(self, client, api_predictor, api_today, active):
        everyone = api_predictor.predict_all(as_of=api_today, active_only=False)
        assert len(everyone) > len(active)
        body = client.get("/rankings?active_only=false&limit=500").json()
        assert body["meta"]["total"] == len(everyone)
        assert {item["player_id"] for item in body["items"]} == {p.player_id for p in everyone}
        assert any(item["is_active"] is False for item in body["items"])
        values = [item["predicted_fantasy"] for item in body["items"]]
        assert values == sorted(values, reverse=True)


class TestWarmCache:
    def test_the_startup_computes_once_and_requests_do_not_recompute(
        self, api_engine, tiny_xgb_bundle, api_today, monkeypatch
    ):
        predictor = Predictor(tiny_xgb_bundle, api_engine, today=lambda: api_today)
        calls = []
        original = predictor._compute

        def counting(as_of):
            calls.append(as_of)
            return original(as_of)

        monkeypatch.setattr(predictor, "_compute", counting)
        app = create_app(
            settings=make_settings(),
            engine=api_engine,
            predictor=predictor,
            clock=FakeClock(noon_utc(api_today)),
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            assert calls == [api_today]  # η προθέρμανση του startup
            first_player = predictor.predict_all(as_of=api_today)[0].player_id
            assert client.get("/rankings").status_code == 200
            assert client.get(f"/predict/{first_player}").status_code == 200
            assert client.get("/players?search=player").status_code == 200
            assert client.get("/rankings?team=T01&active_only=false").status_code == 200
        assert calls == [api_today]  # καμία νέα πρόβλεψη ανά αίτημα


class TestSortingRules:
    """Ακριβείς κανόνες ταξινόμησης με χειροποίητες προβλέψεις."""

    @pytest.fixture
    def fake(self, tmp_path):
        players = [(f"P00000{index}", f"PLAYER {index}, TEST") for index in range(1, 8)]
        engine = make_small_database(tmp_path / "fake.db", players)
        game = make_next_game("AAA", "BBB")
        predictor = FakePredictor(
            [
                make_prediction("P000001", 10.0, team="AAA", next_game=game),
                make_prediction("P000002", 10.0, team="AAA", next_game=game),  # ισοβαθμία
                make_prediction("P000003", 3.14159, team="BBB", pir=-0.001),
                make_prediction("P000004", -1.5, team="BBB"),  # αρνητική πρόβλεψη
                make_prediction("P000005", 25.0, team="AAA", next_game=game),  # θα γίνει out
                make_prediction("P000006", 7.0, team="AAA", active=False),
                make_prediction("P000007", 40.0, team="BBB"),  # θα γίνει out
            ]
        )
        app = create_app(
            settings=make_settings(),
            engine=engine,
            predictor=predictor,
            clock=FakeClock(noon_utc(date(2026, 10, 3))),
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, predictor
        engine.dispose()

    def test_ties_are_ordered_by_player_id(self, fake):
        client, _ = fake
        ids = [
            item["player_id"] for item in client.get("/rankings?active_only=false").json()["items"]
        ]
        assert ids[:4] == ["P000007", "P000005", "P000001", "P000002"]

    def test_out_players_are_always_last_even_when_others_are_negative(self, fake):
        client, _ = fake
        mark(client, "P000005", "out")
        mark(client, "P000007", "out")
        items = client.get("/rankings?include_unavailable=true").json()["items"]
        assert [item["player_id"] for item in items] == [
            "P000001",
            "P000002",
            "P000003",
            "P000004",  # -1,5 < 0, αλλά ΠΡΙΝ από τους out
            "P000007",  # οι out ταξινομούνται μεταξύ τους κατά πρόβλεψη του μοντέλου
            "P000005",
        ]
        by_id = {item["player_id"]: item for item in items}
        assert by_id["P000004"]["predicted_fantasy"] == -1.5
        assert by_id["P000007"]["predicted_fantasy"] == 0.0
        assert by_id["P000007"]["model_predicted_fantasy"] == 40.0
        assert [item["rank"] for item in items] == [1, 2, 3, 4, 5, 6]

    def test_values_are_rounded_and_never_negative_zero(self, fake):
        client, _ = fake
        items = {i["player_id"]: i for i in client.get("/rankings").json()["items"]}
        assert items["P000003"]["predicted_fantasy"] == 3.14
        assert items["P000003"]["predicted_pir"] == 0.0
        assert math.copysign(1.0, items["P000003"]["predicted_pir"]) == 1.0

    def test_the_filters_are_passed_to_the_predictor_with_the_date_of_the_clock(self, fake):
        client, predictor = fake
        client.get("/rankings?team=bbb&active_only=false")
        assert predictor.predict_all_calls[-1] == {
            "as_of": date(2026, 10, 3),
            "team_code": "BBB",
            "active_only": False,
        }
        client.get("/rankings")
        assert predictor.predict_all_calls[-1] == {
            "as_of": date(2026, 10, 3),
            "team_code": None,
            "active_only": True,
        }

    def test_inactive_players_are_excluded_by_default(self, fake):
        client, _ = fake
        ids = [item["player_id"] for item in client.get("/rankings").json()["items"]]
        assert "P000006" not in ids
        assert "P000006" in [
            item["player_id"] for item in client.get("/rankings?active_only=false").json()["items"]
        ]
