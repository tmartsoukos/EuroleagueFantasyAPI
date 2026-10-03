"""Tests του `GET /predict/{player_id}`: έγκυρος και άγνωστος παίκτης, μορφές ID, παίκτης χωρίς
προγραμματισμένο αγώνα, ανενεργός παίκτης, features, override διαθεσιμότητας και στρογγυλοποίηση.

Τα περισσότερα tests τρέχουν πάνω στο συνθετικό πρωτάθλημα. Η κλάση `TestRealFixtures` χρησιμοποιεί
τους πραγματικούς αγώνες των fixtures (παλαιά και νέα IDs, παίκτης χωρίς μελλοντικό αγώνα).
"""

from __future__ import annotations

import re

import pytest
from api_support import ADMIN_HEADERS
from sqlalchemy import select

from elfantasy.db import models
from elfantasy.db.session import get_engine
from elfantasy.features.build import FEATURE_COLUMNS

ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def first(predictions, **conditions):
    """Η πρώτη πρόβλεψη του Predictor που ικανοποιεί τις συνθήκες (π.χ. `is_active=False`)."""
    for prediction in predictions:
        if all(getattr(prediction, name) == value for name, value in conditions.items()):
            return prediction
    raise AssertionError(f"no prediction with {conditions}")


@pytest.fixture(scope="module")
def everyone(api_predictor, api_today):
    return api_predictor.predict_all(as_of=api_today, active_only=False)


class TestPredictResponse:
    def test_the_response_matches_the_predictor(self, client, everyone, api_today):
        for prediction in everyone[:: max(1, len(everyone) // 12)]:
            response = client.get(f"/predict/{prediction.player_id}")
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["player_id"] == prediction.player_id
            assert body["name"] == prediction.name
            assert body["team_code"] == prediction.team_code
            assert body["team_name"] == f"TEAM {prediction.team_code}"
            assert body["is_active"] is prediction.is_active
            assert body["predicted_fantasy"] == round(prediction.predicted_fantasy, 2)
            assert body["model_predicted_fantasy"] == round(prediction.predicted_fantasy, 2)
            assert body["predicted_pir"] == round(prediction.predicted_pir, 2)
            assert body["n_prior_appearances"] == prediction.n_prior_appearances
            expected_last = prediction.last_appearance_date
            assert body["last_appearance_date"] == (
                None if expected_last is None else expected_last.isoformat()
            )
            assert body["model_version"] == "test-xgb-v1"

    def test_the_next_game_and_the_names_of_the_teams(self, client, everyone):
        prediction = first(everyone, is_active=True)
        game = prediction.next_game
        body = client.get(f"/predict/{prediction.player_id}").json()
        assert body["next_game"] == {
            "season": game.season,
            "gamecode": game.gamecode,
            "game_date": game.game_date.isoformat(),
            "tipoff_utc": game.tipoff_utc.isoformat() + "Z",
            "opponent_code": game.opp_code,
            "opponent_name": f"TEAM {game.opp_code}",
            "home": game.home,
        }
        assert ISO_UTC.match(body["next_game"]["tipoff_utc"])

    def test_both_home_and_away_games_occur(self, client, everyone):
        homes = {
            client.get(f"/predict/{p.player_id}").json()["next_game"]["home"] for p in everyone[:40]
        }
        assert homes == {True, False}

    def test_predictions_are_rounded_to_two_decimals(self, client, everyone):
        for prediction in everyone[::7]:
            body = client.get(f"/predict/{prediction.player_id}").json()
            for field in ("predicted_fantasy", "model_predicted_fantasy", "predicted_pir"):
                assert body[field] == round(body[field], 2), field

    def test_a_player_with_a_game_and_history_has_no_warnings(self, client, everyone):
        prediction = first(everyone, is_active=True)
        assert prediction.n_prior_appearances >= 5 and prediction.next_game is not None
        body = client.get(f"/predict/{prediction.player_id}").json()
        assert body["notes"] == []
        assert body["availability"] == {
            "status": "available",
            "source": None,
            "note": None,
            "expected_return": None,
            "updated_at": None,
        }

    def test_features_are_included_only_on_request(self, client, everyone):
        player_id = first(everyone, is_active=True).player_id
        default = client.get(f"/predict/{player_id}").json()
        assert "features" not in default
        explicit_false = client.get(f"/predict/{player_id}?include_features=false").json()
        assert "features" not in explicit_false
        with_features = client.get(f"/predict/{player_id}?include_features=true").json()
        assert list(with_features["features"]) == list(FEATURE_COLUMNS)
        assert len(with_features["features"]) == 42
        for value in with_features["features"].values():
            assert value is None or value == round(value, 4)
        without = {key: value for key, value in with_features.items() if key != "features"}
        assert without == default  # τα υπόλοιπα πεδία είναι ίδια

    def test_a_bad_include_features_value_is_a_validation_error(self, client, everyone):
        player_id = everyone[0].player_id
        assert client.get(f"/predict/{player_id}?include_features=maybe").status_code == 422


class TestPlayerIds:
    def test_an_unknown_player_is_404(self, client):
        response = client.get("/predict/P999999")
        assert response.status_code == 404
        assert response.json() == {
            "detail": "player not found; use /players?search=<name> to look up a player_id"
        }

    @pytest.mark.parametrize(
        "bad_id",
        ["abc", "P12", "P1234567", "X000001", "P00-001", "P0000 1", "%20", "P00000%C3%A9", "1234"],
    )
    def test_ids_of_the_wrong_shape_are_422(self, client, bad_id):
        response = client.get(f"/predict/{bad_id}")
        assert response.status_code == 422, bad_id
        (error,) = response.json()["detail"]
        assert error["loc"] == ["path", "player_id"]
        assert error["type"] == "string_pattern_mismatch"
        assert "^P[A-Z0-9]{3,6}$" in error["msg"]

    def test_lowercase_ids_and_surrounding_spaces_are_accepted(self, client, everyone):
        player_id = first(everyone, is_active=True).player_id
        canonical = client.get(f"/predict/{player_id}").json()
        for variant in (player_id.lower(), f" {player_id} ", f"%20{player_id.lower()}%20"):
            response = client.get(f"/predict/{variant}")
            assert response.status_code == 200, variant
            assert response.json() == canonical

    def test_a_valid_but_unknown_old_style_id_is_404_not_422(self, client):
        assert client.get("/predict/PXYZ").status_code == 404

    def test_other_methods_are_not_allowed(self, client):
        response = client.post("/predict/P000001")
        assert response.status_code == 405
        assert response.json() == {"detail": "Method Not Allowed"}


class TestContext:
    def test_a_player_without_a_scheduled_game_gets_a_neutral_context(
        self, client, api_clock, everyone
    ):
        player_id = first(everyone, is_active=True).player_id
        api_clock.advance(days=1000)  # το πρόγραμμα έχει τελειώσει
        body = client.get(f"/predict/{player_id}").json()
        assert body["next_game"] is None
        assert "no scheduled game: neutral context" in body["notes"]
        assert isinstance(body["predicted_fantasy"], float)
        assert isinstance(body["predicted_pir"], float)

    def test_an_inactive_player_still_gets_a_prediction_and_a_warning(self, client, everyone):
        prediction = first(everyone, is_active=False)
        body = client.get(f"/predict/{prediction.player_id}").json()
        assert body["is_active"] is False
        assert any(note.startswith("player is not active") for note in body["notes"])
        assert isinstance(body["predicted_fantasy"], float)
        assert body["team_code"] == prediction.team_code  # η ομάδα της τελευταίας γραμμής

    def test_a_player_without_history_is_flagged(self, client, everyone):
        prediction = first(everyone, n_prior_appearances=0)
        body = client.get(f"/predict/{prediction.player_id}").json()
        assert body["n_prior_appearances"] == 0
        assert body["last_appearance_date"] is None
        assert (
            "player has no previous appearances: prediction is based on the game context only"
            in body["notes"]
        )

    def test_the_warnings_are_listed_in_a_stable_order(self, client, api_clock, everyone):
        prediction = first(everyone, is_active=False)
        client.post(
            "/availability",
            json={"player_id": prediction.player_id, "status": "doubtful"},
            headers=ADMIN_HEADERS,
        )
        api_clock.advance(days=1000)
        notes = client.get(f"/predict/{prediction.player_id}").json()["notes"]
        assert notes[0] == "player is marked doubtful: prediction assumes the player plays"
        assert notes[1] == "no scheduled game: neutral context"
        assert notes[2].startswith("player is not active")


class TestAvailabilityOverride:
    def post(self, client, player_id, status, **extra):
        response = client.post(
            "/availability",
            json={"player_id": player_id, "status": status, **extra},
            headers=ADMIN_HEADERS,
        )
        assert response.status_code == 200, response.text
        return response

    def test_out_sets_the_effective_prediction_to_zero_only(self, client, everyone, api_clock):
        prediction = first(everyone, is_active=True)
        before = client.get(f"/predict/{prediction.player_id}").json()
        assert before["predicted_fantasy"] > 0
        self.post(
            client,
            prediction.player_id,
            "out",
            source="club",
            note="ankle",
            expected_return="2099-01-01",
        )
        body = client.get(f"/predict/{prediction.player_id}").json()
        assert body["predicted_fantasy"] == 0.0
        assert body["model_predicted_fantasy"] == before["model_predicted_fantasy"]
        assert body["predicted_pir"] == before["predicted_pir"]  # το PIR δεν αλλάζει
        assert body["availability"] == {
            "status": "out",
            "source": "club",
            "note": "ankle",
            "expected_return": "2099-01-01",
            "updated_at": api_clock.moment.isoformat().replace("+00:00", "Z"),
        }
        assert body["notes"] == ["player is marked out: effective prediction is 0"]
        assert body["next_game"] == before["next_game"]  # ο υπόλοιπος κόσμος ίδιος

    def test_doubtful_keeps_the_prediction_and_adds_a_warning(self, client, everyone):
        prediction = first(everyone, is_active=True)
        before = client.get(f"/predict/{prediction.player_id}").json()
        self.post(client, prediction.player_id, "doubtful", note="knee")
        body = client.get(f"/predict/{prediction.player_id}").json()
        assert body["predicted_fantasy"] == before["predicted_fantasy"] > 0
        assert body["availability"]["status"] == "doubtful"
        assert body["notes"] == ["player is marked doubtful: prediction assumes the player plays"]

    def test_available_changes_nothing_but_shows_the_record(self, client, everyone):
        prediction = first(everyone, is_active=True)
        before = client.get(f"/predict/{prediction.player_id}").json()
        self.post(client, prediction.player_id, "available", note="cleared to play")
        body = client.get(f"/predict/{prediction.player_id}").json()
        assert body["predicted_fantasy"] == before["predicted_fantasy"]
        assert body["notes"] == []
        assert body["availability"]["status"] == "available"
        assert body["availability"]["note"] == "cleared to play"
        assert body["availability"]["updated_at"] is not None

    def test_removing_the_override_restores_the_prediction(self, client, everyone):
        prediction = first(everyone, is_active=True)
        before = client.get(f"/predict/{prediction.player_id}").json()
        self.post(client, prediction.player_id, "out")
        assert client.get(f"/predict/{prediction.player_id}").json()["predicted_fantasy"] == 0.0
        deleted = client.delete(f"/availability/{prediction.player_id}", headers=ADMIN_HEADERS)
        assert deleted.status_code == 204
        assert client.get(f"/predict/{prediction.player_id}").json() == before

    def test_an_expected_return_in_the_past_adds_a_warning(self, client, everyone, api_clock):
        prediction = first(everyone, is_active=True)
        self.post(client, prediction.player_id, "out", expected_return="2000-01-01")
        notes = client.get(f"/predict/{prediction.player_id}").json()["notes"]
        assert "expected return date has passed: the availability record may be outdated" in notes
        # την ημέρα της επιστροφής δεν θεωρείται ακόμη παλιά
        self.post(
            client, prediction.player_id, "out", expected_return=api_clock.moment.date().isoformat()
        )
        notes = client.get(f"/predict/{prediction.player_id}").json()["notes"]
        assert notes == ["player is marked out: effective prediction is 0"]


class TestRealFixtures:
    def test_an_old_style_id_with_spaces_and_lowercase(self, real_client):
        response = real_client.get("/predict/%20plru%20")
        assert response.status_code == 200
        body = response.json()
        assert body["player_id"] == "PLRU"
        assert body["name"] == "SIMONOVIC, MARKO"

    def test_a_new_style_id_and_a_scheduled_game(self, real_client, fixture_database_url):
        body = real_client.get("/predict/P006835").json()  # Jaylen Hoard (TEL)
        assert body["name"] == "HOARD, JAYLEN"
        assert body["team_code"] == "TEL"
        game = body["next_game"]
        assert (game["season"], game["gamecode"]) == (2026, 32)
        assert (game["opponent_code"], game["home"]) == ("MIL", True)
        assert game["game_date"] == "2026-10-08"
        assert game["tipoff_utc"] == "2026-10-08T16:00:00Z"  # 18:00 CEST = 16:00 UTC
        engine = get_engine(fixture_database_url)
        with engine.connect() as connection:
            names = dict(
                connection.execute(select(models.teams.c.team_code, models.teams.c.name)).all()
            )
        engine.dispose()
        assert body["team_name"] == names["TEL"]
        assert game["opponent_name"] == names["MIL"]
        assert body["n_prior_appearances"] == 2
        assert body["last_appearance_date"] == "2025-09-30"

    def test_a_player_whose_team_has_no_scheduled_game(self, real_client):
        body = real_client.get("/predict/P007200").json()  # Shane Larkin (IST)
        assert body["name"] == "LARKIN, SHANE"
        assert body["team_code"] == "IST"
        assert body["next_game"] is None
        assert body["is_active"] is False
        assert "no scheduled game: neutral context" in body["notes"]
        assert any(note.startswith("player is not active") for note in body["notes"])
        assert "player has only 1 previous appearance: prediction is less reliable" in body["notes"]

    def test_a_player_who_did_not_play_has_no_history(self, real_client):
        body = real_client.get("/predict/P006590").json()  # Beaubois: DNP στον 2025/1
        assert body["n_prior_appearances"] == 0
        assert body["last_appearance_date"] is None
        assert any("no previous appearances" in note for note in body["notes"])
