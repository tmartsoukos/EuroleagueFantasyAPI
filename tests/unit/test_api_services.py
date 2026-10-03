"""Unit tests της υπηρεσίας του API (`elfantasy.api.services`): κανονικοποίηση αναζήτησης,
στρογγυλοποίηση, σημειώσεις, ονόματα ομάδων, αναζήτηση παικτών και ανανέωση, χωρίς HTTP.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from api_support import (
    FakeClock,
    FakePredictor,
    make_next_game,
    make_prediction,
    make_small_database,
    noon_utc,
)
from sqlalchemy import update

from elfantasy.api.services import (
    NOTE_INACTIVE,
    NOTE_NO_GAME,
    NOTE_NO_HISTORY,
    PlayerNotFound,
    PredictionService,
    UnknownTeam,
    clock_today,
    database_summary,
    normalise_text,
    round2,
    utc_now,
)
from elfantasy.db import models

TODAY = date(2026, 10, 3)


class TestNormaliseText:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("VEZENKOV, SASHA", "vezenkov sasha"),
            ("Vezenkóv   sasha", "vezenkov sasha"),
            ("ĐORĐEVIĆ, NEMANJA", "dordevic nemanja"),
            ("MÜLLER, KARL", "muller karl"),
            ("O'BRIEN, SEAN", "obrien sean"),
            ("O’BRIEN", "obrien"),
            ("LAWSON, A.J.", "lawson aj"),
            ("LUWAWU-CABARROT, TIMOTHE", "luwawu cabarrot timothe"),
            ("Straße", "strasse"),
            ("ŁUKASZ", "lukasz"),
            ("Øystein", "oystein"),
            ("Œuvre", "oeuvre"),
            ("İSTANBUL", "istanbul"),
            ("ıstanbul", "istanbul"),
            ("Γιάννης", "γιαννησ"),  # τόνοι αφαιρούνται· το τελικό ς γίνεται σ (casefold)
            ("  a   b  ", "a b"),
            ("", ""),
            ("'", ""),
        ],
    )
    def test_normalisation(self, raw, expected):
        assert normalise_text(raw) == expected

    def test_a_query_and_a_name_normalise_to_comparable_keys(self):
        name = normalise_text("ŠIMONOVIĆ, MARKO")
        for query in ("simonovic", "ŠIMONOVIĆ", "marko simonovic", "SIMONOVIC, MARKO"):
            assert all(token in name for token in normalise_text(query).split())


class TestSmallHelpers:
    def test_round2(self):
        assert round2(20.1285400390625) == 20.13
        assert round2(5) == 5.0 and isinstance(round2(5), float)
        assert round2(np.float32(7.8395)) == 7.84 and type(round2(np.float32(7.8395))) is float

    def test_round2_never_returns_negative_zero(self):
        for value in (-0.001, -0.0, -0.004):
            result = round2(value)
            assert result == 0.0 and math.copysign(1.0, result) == 1.0
        assert round2(-1.5) == -1.5

    def test_clock_today_uses_the_utc_date(self):
        assert clock_today(lambda: datetime(2026, 10, 3, 23, 59, tzinfo=UTC)) == TODAY
        assert clock_today(lambda: datetime(2026, 10, 3, 23, 59)) == TODAY  # naive = UTC
        zone = timezone(timedelta(hours=3))
        assert clock_today(lambda: datetime(2026, 10, 4, 2, 0, tzinfo=zone)) == TODAY  # 23:00 UTC

    def test_the_default_clock_is_utc(self):
        now = utc_now()
        assert now.tzinfo is not None and now.utcoffset() == timedelta(0)
        assert abs(datetime.now(UTC) - now) < timedelta(seconds=5)


def add_game(engine, gamecode, day, played, home="AAA", away="BBB"):
    with engine.begin() as connection:
        connection.execute(
            models.games.insert(),
            {
                "season": 2026,
                "gamecode": gamecode,
                "phase": "RS",
                "round": 1,
                "game_date": day,
                "tipoff_utc": datetime(day.year, day.month, day.day, 18),
                "home_code": home,
                "away_code": away,
                "home_score": 80 if played else None,
                "away_score": 70 if played else None,
                "played": played,
                "winner_code": home if played else None,
            },
        )


class TestDatabaseSummary:
    def test_an_empty_database(self, tmp_path):
        engine = make_small_database(tmp_path / "empty.db", [])
        summary = database_summary(engine, TODAY)
        assert (summary.players, summary.latest_played_game_date) == (0, None)
        assert summary.next_scheduled_game_date is None
        engine.dispose()

    def test_players_and_the_latest_and_next_game_dates(self, tmp_path):
        engine = make_small_database(tmp_path / "s.db", [("P000001", "A, A"), ("P000002", "B, B")])
        add_game(engine, 1, date(2026, 9, 28), played=True)
        add_game(engine, 2, date(2026, 10, 1), played=True)
        add_game(engine, 3, date(2026, 9, 30), played=False)  # ακυρωμένος αγώνας του παρελθόντος
        add_game(engine, 4, date(2026, 10, 8), played=False)
        add_game(engine, 5, date(2026, 10, 7), played=False)
        summary = database_summary(engine, TODAY)
        assert summary.players == 2
        assert summary.latest_played_game_date == date(2026, 10, 1)
        assert summary.next_scheduled_game_date == date(2026, 10, 7)  # όχι ο ακυρωμένος
        later = database_summary(engine, date(2026, 10, 7))
        assert later.next_scheduled_game_date == date(2026, 10, 7)  # η ίδια ημέρα μετρά
        assert database_summary(engine, date(2026, 10, 9)).next_scheduled_game_date is None
        engine.dispose()


@pytest.fixture
def parts(tmp_path):
    players = [(f"P00000{index}", f"PLAYER {index}, TEST") for index in range(1, 7)]
    engine = make_small_database(tmp_path / "service.db", players)
    game = make_next_game("AAA", "BBB")
    predictor = FakePredictor(
        [
            make_prediction("P000001", 14.0, team="AAA", next_game=game),
            make_prediction("P000002", 9.0, team="AAA", appearances=0, last_appearance=None),
            make_prediction(
                "P000003", 5.0, team="BBB", active=False, appearances=3, next_game=game
            ),
            make_prediction("P000004", 6.0, team="BBB", appearances=1, next_game=game),
            make_prediction("P000005", 8.0, team="BBB", appearances=4, next_game=game),
            make_prediction("P000006", 8.5, team="BBB", appearances=5, next_game=game),
        ]
    )
    clock = FakeClock(noon_utc(TODAY))
    service = PredictionService(predictor, engine, clock)
    yield SimpleNamespace(service=service, predictor=predictor, engine=engine, clock=clock)
    engine.dispose()


class TestPredictNotes:
    def test_a_normal_player_has_no_notes(self, parts):
        assert parts.service.predict("P000001").notes == []

    def test_no_game_and_no_history(self, parts):
        assert parts.service.predict("P000002").notes == [NOTE_NO_GAME, NOTE_NO_HISTORY]

    def test_an_inactive_player_with_few_appearances(self, parts):
        assert parts.service.predict("P000003").notes == [
            NOTE_INACTIVE,
            "player has only 3 previous appearances: prediction is less reliable",
        ]

    def test_a_single_appearance_is_singular(self, parts):
        assert parts.service.predict("P000004").notes == [
            "player has only 1 previous appearance: prediction is less reliable"
        ]

    def test_the_threshold_of_few_appearances_is_five(self, parts):
        assert len(parts.service.predict("P000005").notes) == 1  # 4 συμμετοχές
        assert parts.service.predict("P000006").notes == []  # 5 συμμετοχές

    def test_the_inactive_note_mentions_the_window(self):
        assert "45 days" in NOTE_INACTIVE and NOTE_INACTIVE.startswith("player is not active")


class TestPredict:
    def test_features_are_rounded_to_four_decimals_and_nulls_are_kept(self, parts):
        without = parts.service.predict("P000001")
        assert "features" not in without.model_fields_set
        with_features = parts.service.predict("P000001", include_features=True)
        assert with_features.features == {"pir_mean_5": 7.1235, "home": None}
        assert "features" in with_features.model_fields_set

    def test_the_fields(self, parts):
        prediction = parts.service.predict(" p000001 ")
        assert prediction.player_id == "P000001"
        assert prediction.team_name == "TEAM AAA"
        assert prediction.next_game.opponent_name == "TEAM BBB"
        assert prediction.next_game.tipoff_utc == datetime(2026, 10, 7, 18, 45, tzinfo=UTC)
        assert prediction.predicted_fantasy == prediction.model_predicted_fantasy == 14.0
        assert prediction.predicted_pir == 13.5
        assert prediction.model_version == "fake-v1"

    def test_an_unknown_player_raises_a_domain_error(self, parts):
        with pytest.raises(PlayerNotFound) as error:
            parts.service.predict("P999999")
        assert error.value.message.startswith("player not found")

    def test_a_game_without_a_known_tipoff_time(self, parts):
        game = make_next_game()
        game = type(game)(**{**game.__dict__, "tipoff_utc": None})
        parts.predictor._predictions["P000001"] = make_prediction("P000001", 14.0, next_game=game)
        assert parts.service.predict("P000001").next_game.tipoff_utc is None

    def test_the_date_comes_from_the_clock(self, parts):
        parts.clock.advance(days=3)
        parts.service.predict("P000001")
        parts.service.rankings(
            limit=5, offset=0, team=None, include_unavailable=False, active_only=True
        )
        assert parts.service.today() == date(2026, 10, 6)


class TestTeams:
    def test_team_names_come_from_the_database(self, parts):
        assert parts.service.team_name("AAA") == "TEAM AAA"
        assert parts.service.team_name("ZZZ") == "ZZZ"  # άγνωστη ομάδα: ο κωδικός
        assert parts.service.team_codes == ["AAA", "BBB"]

    def test_require_team_normalises_the_code(self, parts):
        assert parts.service.require_team("aaa") == "AAA"
        assert parts.service.require_team(" Bbb ") == "BBB"

    def test_an_unknown_team_lists_the_valid_codes(self, parts):
        with pytest.raises(UnknownTeam) as error:
            parts.service.require_team("zzz")
        assert error.value.message == "unknown team code 'zzz'; valid codes: AAA, BBB"

    def test_the_names_are_cached_until_the_reference_data_is_reloaded(self, parts):
        with parts.engine.begin() as connection:
            connection.execute(
                update(models.teams)
                .where(models.teams.c.team_code == "AAA")
                .values(name="NEW NAME")
            )
        assert parts.service.team_name("AAA") == "TEAM AAA"  # cache
        parts.service.reload_reference_data()
        assert parts.service.team_name("AAA") == "NEW NAME"


class TestSearch:
    @pytest.fixture
    def names(self, tmp_path):
        players = [
            ("P000001", "ĐORĐEVIĆ, NEMANJA"),
            ("P000002", "MÜLLER, KARL"),
            ("P000003", "O'BRIEN, SEAN"),
            ("P000004", "NÚÑEZ, ÁLVARO"),
            ("P000005", "SMITH, JOHN"),
            ("P000006", "SMITH, JANE"),
        ]
        engine = make_small_database(tmp_path / "names.db", players)
        predictions = [
            make_prediction(player_id, 10.0 - index, name=name, team="AAA" if index % 2 else "BBB")
            for index, (player_id, name) in enumerate(players)
        ]
        predictions[5] = make_prediction(
            "P000006", 1.0, name="SMITH, JANE", team="BBB", active=False
        )
        service = PredictionService(FakePredictor(predictions), engine, FakeClock(noon_utc(TODAY)))
        yield service
        engine.dispose()

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("djordjevic", []),  # «đ» → «d», όχι «dj»: τεκμηριωμένος περιορισμός
            ("dordevic", ["P000001"]),
            ("ĐORĐEVIĆ", ["P000001"]),
            ("muller", ["P000002"]),
            ("MÜLLER", ["P000002"]),
            ("o'brien", ["P000003"]),
            ("obrien", ["P000003"]),
            ("sean o brien", ["P000003"]),
            ("nunez", ["P000004"]),
            ("alvaro nunez", ["P000004"]),
            ("smith", ["P000005", "P000006"]),
            ("smith jo", ["P000005"]),
            ("jane smith", ["P000006"]),
            ("zzz", []),
        ],
    )
    def test_queries(self, names, query, expected):
        result = names.search_players(search=query, team=None, limit=20)
        assert sorted(item.player_id for item in result.items) == sorted(expected)
        assert result.total == len(expected)

    def test_active_players_come_first_then_alphabetical(self, names):
        result = names.search_players(search="smith", team=None, limit=20)
        assert [item.player_id for item in result.items] == ["P000005", "P000006"]
        assert [item.is_active for item in result.items] == [True, False]
        everyone = names.search_players(search=None, team=None, limit=20)
        assert everyone.items[-1].player_id == "P000006"  # ο ανενεργός τελευταίος

    def test_the_limit_truncates_but_the_total_is_complete(self, names):
        result = names.search_players(search=None, team=None, limit=2)
        assert len(result.items) == 2 and result.total == 6 and result.limit == 2

    def test_the_team_filter(self, names):
        result = names.search_players(search=None, team="aaa", limit=20)
        assert {item.team_code for item in result.items} == {"AAA"}
        with pytest.raises(UnknownTeam):
            names.search_players(search=None, team="ZZZ", limit=20)

    def test_a_player_missing_from_the_catalogue_is_skipped(self, names):
        names._players.pop("P000001")
        result = names.search_players(search=None, team=None, limit=20)
        assert "P000001" not in [item.player_id for item in result.items]


class TestRefresh:
    def test_the_refresh_reloads_the_predictor_and_reports_the_cutoff(self, parts):
        add_game(parts.engine, 1, date(2026, 10, 1), played=True)
        parts.service.reload_reference_data()
        assert parts.service.loaded_through == date(2026, 10, 1)
        add_game(parts.engine, 2, date(2026, 10, 2), played=True)
        add_game(parts.engine, 3, date(2026, 10, 9), played=False)
        parts.clock.advance(minutes=5)
        result = parts.service.refresh()
        assert parts.predictor.refresh_calls == 1
        assert result.status == "refreshed"
        assert result.model_version == "fake-v1"
        assert result.players == 6
        assert result.previous_latest_played_game_date == date(2026, 10, 1)
        assert result.latest_played_game_date == date(2026, 10, 2)
        assert result.next_scheduled_game_date == date(2026, 10, 9)
        assert result.refreshed_at == parts.clock.moment
        assert parts.service.loaded_through == date(2026, 10, 2)
        assert parts.predictor.predict_all_calls[-1]["active_only"] is False  # προθέρμανση

    def test_the_warm_up_computes_the_predictions_of_the_day(self, parts):
        parts.predictor.predict_all_calls.clear()
        parts.service.warm_up()
        assert parts.predictor.predict_all_calls == [
            {"as_of": TODAY, "team_code": None, "active_only": False}
        ]
