"""Unit tests των dependencies (`elfantasy.api.deps`) και των μοντέλων (`elfantasy.api.schemas`),
χωρίς HTTP: έλεγχος κλειδιού, κανονικοποίηση `player_id`, μηνύματα 503 και επικύρωση εισόδου.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest
from api_support import ADMIN_KEY, make_settings
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from elfantasy.api.availability import AvailabilityStatus
from elfantasy.api.deps import get_engine, get_service, get_state, require_admin, valid_player_id
from elfantasy.api.schemas import (
    AvailabilityIn,
    AvailabilityOut,
    NextGameOut,
    PredictionOut,
    normalise_player_id,
)
from elfantasy.api.security import MIN_ADMIN_KEY_LENGTH
from elfantasy.api.state import AppState


class TestRequireAdmin:
    def state(self, key: str | None) -> AppState:
        settings = None if key is None else make_settings(admin_api_key=key)
        return AppState(settings=settings)

    def test_the_correct_key_passes(self):
        assert require_admin(self.state(ADMIN_KEY), ADMIN_KEY) is None

    @pytest.mark.parametrize("sent", [None, "", "wrong", ADMIN_KEY + "x", ADMIN_KEY.swapcase()])
    def test_a_missing_or_wrong_key_is_401(self, sent):
        with pytest.raises(HTTPException) as error:
            require_admin(self.state(ADMIN_KEY), sent)
        assert error.value.status_code == 401
        assert error.value.detail == "invalid or missing API key"

    @pytest.mark.parametrize("configured", [None, "", "   ", "\t\n"])
    @pytest.mark.parametrize("sent", [None, "", ADMIN_KEY, " "])
    def test_without_a_configured_key_everything_is_503(self, configured, sent):
        with pytest.raises(HTTPException) as error:
            require_admin(self.state(configured), sent)
        assert error.value.status_code == 503
        assert error.value.detail == "admin API is disabled"

    def test_a_non_ascii_key_works_on_both_sides(self):
        key = "κλειδί-ασφαλείας-δοκιμής-2026"
        state = self.state(key)
        assert require_admin(state, key) is None
        with pytest.raises(HTTPException) as error:
            require_admin(state, "κλειδί")
        assert error.value.status_code == 401

    def test_a_non_ascii_key_sent_as_utf_8_bytes_matches_like_a_real_client(self):
        """Το Starlette αποκωδικοποιεί τις επικεφαλίδες ως latin-1: ένας πραγματικός client
        στέλνει το κλειδί σε UTF-8 και ο server βλέπει latin-1 «μπερδεμένο» κείμενο."""
        key = "κλειδί-ασφαλείας-δοκιμής-2026"
        as_seen_by_starlette = key.encode("utf-8").decode("latin-1")
        assert as_seen_by_starlette != key
        assert require_admin(self.state(key), as_seen_by_starlette) is None
        with pytest.raises(HTTPException) as error:
            require_admin(self.state(key), as_seen_by_starlette[:-1] + "x")
        assert error.value.status_code == 401

    def test_a_key_with_surrounding_spaces_must_match_exactly(self):
        state = self.state(" spaced-key-with-enough-length ")
        assert require_admin(state, " spaced-key-with-enough-length ") is None
        with pytest.raises(HTTPException):
            require_admin(state, "spaced-key-with-enough-length")

    @pytest.mark.parametrize("configured", ["1234", "short", "x" * 23, " " + "x" * 22 + " "])
    def test_a_too_short_key_keeps_the_admin_api_closed(self, configured):
        """Ένα σύντομο κλειδί δεν ανοίγει τα endpoints, ούτε με το σωστό κλειδί (503, όχι 401)."""
        with pytest.raises(HTTPException) as error:
            require_admin(self.state(configured), configured)
        assert error.value.status_code == 503
        assert error.value.detail == "admin API is disabled"

    def test_the_minimum_length_is_accepted(self):
        key = "k" * MIN_ADMIN_KEY_LENGTH
        assert require_admin(self.state(key), key) is None


class TestValidPlayerId:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("P007200", "P007200"),
            (" p007200 ", "P007200"),
            ("padf", "PADF"),
            ("PLRU", "PLRU"),
            ("P123", "P123"),
            ("P123456", "P123456"),
        ],
    )
    def test_valid_ids_are_normalised(self, raw, expected):
        assert valid_player_id(raw) == expected

    @pytest.mark.parametrize(
        "raw", ["", " ", "abc", "P12", "P1234567", "X007200", "P00-200", "Π007200"]
    )
    def test_invalid_ids_raise_a_validation_error_in_the_fastapi_format(self, raw):
        with pytest.raises(RequestValidationError) as error:
            valid_player_id(raw)
        (detail,) = error.value.errors()
        assert detail["loc"] == ("path", "player_id")
        assert detail["type"] == "string_pattern_mismatch"
        assert detail["input"] == raw
        assert detail["ctx"] == {"pattern": "^P[A-Z0-9]{3,6}$"}


class TestServiceDependencies:
    def test_get_state_reads_the_application_state(self):
        state = AppState()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(api=state)))
        assert get_state(request) is state

    def test_the_service_is_returned_when_it_exists(self):
        marker = object()
        assert get_service(AppState(service=marker)) is marker

    @pytest.mark.parametrize(
        ("problems", "message"),
        [
            ({"model": "x"}, "prediction model is not available"),
            ({"database": "x"}, "database is not available"),
            ({"model": "x", "database": "y"}, "prediction model is not available"),
            ({}, "service is not ready"),
        ],
    )
    def test_a_missing_service_is_503_with_a_clean_message(self, problems, message):
        with pytest.raises(HTTPException) as error:
            get_service(AppState(problems=problems))
        assert error.value.status_code == 503 and error.value.detail == message

    def test_the_engine(self):
        engine = object()
        assert get_engine(AppState(engine=engine)) is engine
        for state in (AppState(), AppState(engine=engine, problems={"database": "x"})):
            with pytest.raises(HTTPException) as error:
                get_engine(state)
            assert error.value.status_code == 503
            assert error.value.detail == "database is not available"


class TestAvailabilityInput:
    def make(self, **overrides):
        return AvailabilityIn(**{"player_id": "P007200", "status": "out", **overrides})

    def test_a_minimal_body(self):
        body = self.make()
        assert body.player_id == "P007200" and body.status is AvailabilityStatus.OUT
        assert (body.source, body.note, body.expected_return) == (None, None, None)

    def test_the_id_is_normalised_and_checked(self):
        assert self.make(player_id=" p007200 ").player_id == "P007200"
        assert self.make(player_id="padf").player_id == "PADF"
        for bad in ("abc", "P1", "", "P0072001"):
            with pytest.raises(ValidationError):
                self.make(player_id=bad)

    def test_the_status_is_one_of_three(self):
        for good in ("out", "doubtful", "available"):
            assert self.make(status=good).status.value == good
        for bad in ("OUT", "injured", "", None, 3):
            with pytest.raises(ValidationError):
                self.make(status=bad)

    def test_text_is_stripped_and_blank_becomes_none(self):
        assert self.make(note="  hi  ", source=" s ").note == "hi"
        assert self.make(note="   ").note is None
        assert self.make(source="").source is None
        assert self.make(note=None).note is None

    def test_the_length_limits_apply_after_stripping(self):
        assert len(self.make(note=" " + "x" * 500 + " ").note) == 500
        with pytest.raises(ValidationError):
            self.make(note="x" * 501)
        assert len(self.make(source="s" * 100).source) == 100
        with pytest.raises(ValidationError):
            self.make(source="s" * 101)

    def test_the_date(self):
        assert self.make(expected_return="2026-10-20").expected_return == date(2026, 10, 20)
        assert self.make(expected_return=date(2026, 10, 20)).expected_return == date(2026, 10, 20)
        assert self.make(expected_return=" 2026-10-20 ").expected_return == date(2026, 10, 20)
        assert self.make(expected_return=None).expected_return is None
        for bad in ("2026-13-45", "tomorrow", "", 20261020, 1.5, True, ["2026-10-20"]):
            with pytest.raises(ValidationError):
                self.make(expected_return=bad)

    def test_unknown_fields_are_rejected(self):
        with pytest.raises(ValidationError):
            self.make(updated_at="2026-01-01T00:00:00Z")


class TestOutputModels:
    def game(self):
        return NextGameOut(
            season=2026,
            gamecode=1,
            game_date=date(2026, 10, 7),
            tipoff_utc=datetime(2026, 10, 7, 18, 45, tzinfo=UTC),
            opponent_code="BBB",
            opponent_name="TEAM BBB",
            home=True,
        )

    def prediction(self, **extra):
        return PredictionOut(
            player_id="P1",
            name="N",
            team_code="AAA",
            team_name="TEAM AAA",
            is_active=True,
            next_game=None,
            predicted_fantasy=1.0,
            model_predicted_fantasy=1.0,
            predicted_pir=1.0,
            availability=AvailabilityOut(),
            n_prior_appearances=1,
            last_appearance_date=None,
            model_version="v",
            notes=[],
            **extra,
        )

    def test_the_default_availability_is_available_with_nulls(self):
        assert AvailabilityOut().model_dump() == {
            "status": AvailabilityStatus.AVAILABLE,
            "source": None,
            "note": None,
            "expected_return": None,
            "updated_at": None,
        }

    def test_features_are_omitted_from_the_unset_dump_and_present_when_given(self):
        assert "features" not in self.prediction().model_dump(exclude_unset=True)
        dumped = self.prediction(features={"a": 1.0, "b": None}).model_dump(exclude_unset=True)
        assert dumped["features"] == {"a": 1.0, "b": None}
        assert self.prediction().model_dump(exclude_unset=True)["next_game"] is None

    def test_a_tipoff_with_utc_is_serialised_with_z(self):
        assert self.game().model_dump_json().count('"tipoff_utc":"2026-10-07T18:45:00Z"') == 1

    def test_normalise_player_id(self):
        assert normalise_player_id("  p007200\t") == "P007200"
        assert normalise_player_id("") == ""
