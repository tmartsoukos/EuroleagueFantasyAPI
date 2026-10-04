"""Integration test του `Predictor` πάνω σε βάση που φτιάχνεται από τα ΠΡΑΓΜΑΤΙΚΑ fixtures
(7 αγώνες του 2016-2026 και 3 μελλοντικοί αγώνες του 2026, από το ingestion pipeline).

Το μοντέλο είναι το μικρό συνθετικό μοντέλο των tests (οι τιμές των προβλέψεων δεν ελέγχονται):
εδώ ελέγχονται η φόρτωση από τη βάση, τα πραγματικά ID (παλαιά και νέα μορφή), ο επόμενος αγώνας
από το πραγματικό schedule και οι ενεργοί παίκτες. Δεν γίνεται καμία κλήση δικτύου.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytest
from sqlalchemy import select

from elfantasy.db import models
from elfantasy.db.session import get_engine
from elfantasy.features.build import FEATURE_COLUMNS
from elfantasy.model.artifact import save_bundle
from elfantasy.model.predict import Predictor

AS_OF = date(
    2026, 10, 3
)  # η ημέρα του πλήρους τρεξίματος: τα 3 fixtures του 2026/31-33 είναι μελλοντικά


@pytest.fixture(scope="module")
def engine(fixture_database_url):
    engine = get_engine(fixture_database_url)
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def predictor(tiny_xgb_bundle, engine, tmp_path_factory):
    path = tmp_path_factory.mktemp("predict_db_model") / "model.joblib"
    save_bundle(tiny_xgb_bundle, path)
    return Predictor.load(path, engine)


def players_in(engine, season, gamecode=None) -> set[str]:
    columns = models.player_games.c
    statement = select(columns.player_id).where(columns.season == season)
    if gamecode is not None:
        statement = statement.where(columns.gamecode == gamecode)
    with engine.connect() as connection:
        return {row[0] for row in connection.execute(statement)}


class TestRealPlayers:
    def test_a_player_whose_team_has_a_scheduled_game(self, predictor):
        # Jaylen Hoard (TEL): ο επόμενος αγώνας της TEL είναι ο 2026/32 με γηπεδούχο τη TEL.
        prediction = predictor.predict_player("P006835", as_of=AS_OF)
        assert prediction.name == "HOARD, JAYLEN"
        assert prediction.team_code == "TEL"
        game = prediction.next_game
        assert (game.season, game.gamecode) == (2026, 32)
        assert (game.team_code, game.opp_code, game.home) == ("TEL", "MIL", True)
        assert game.game_date == date(2026, 10, 8)
        assert game.tipoff_utc == datetime(2026, 10, 8, 16, 0)  # 18:00 CEST = 16:00 UTC
        assert prediction.n_prior_appearances == 2  # οι αγώνες 2024/175 (PIR 4) και 2025/1 (PIR 22)
        assert prediction.last_appearance_date == date(2025, 9, 30)
        assert prediction.features["home"] == 1.0
        assert prediction.features["pir_mean_5"] == 13.0  # (4 + 22) / 2
        assert (
            prediction.features["fantasy_mean_5"] == 13.0
        )  # και οι δύο αγώνες χάθηκαν: χωρίς μπόνους
        assert prediction.features["min_last"] == pytest.approx(31 + 34 / 60)  # 31:34 στον 2025/1

    def test_an_away_game(self, predictor, engine):
        # Παίκτης της ASV (2023/1): ο επόμενος αγώνας είναι ο 2026/31 με φιλοξενούμενη την ASV.
        asv_players = sorted(
            player_id
            for player_id in players_in(engine, 2023, 1)
            if predictor.predict_player(player_id, as_of=AS_OF).team_code == "ASV"
        )
        assert asv_players
        game = predictor.predict_player(asv_players[0], as_of=AS_OF).next_game
        assert (game.season, game.gamecode, game.opp_code, game.home) == (2026, 31, "PRS", False)
        assert game.game_date == date(2026, 10, 7)
        assert game.tipoff_utc == datetime(2026, 10, 7, 18, 45)

    def test_a_player_whose_team_has_no_scheduled_game_gets_neutral_context(self, predictor):
        # Shane Larkin (IST): κανένας μελλοντικός αγώνας της IST στα fixtures.
        larkin = predictor.predict_player("P007200", as_of=AS_OF)
        assert larkin.name == "LARKIN, SHANE"
        assert larkin.team_code == "IST" and larkin.next_game is None
        assert larkin.n_prior_appearances == 1
        assert larkin.last_appearance_date == date(2025, 9, 30)
        assert larkin.features["home"] is None and larkin.features["opp_def_pir_5"] is None
        assert larkin.features["min_last"] == pytest.approx(33 + 21 / 60)  # 33:21
        assert larkin.features["pir_mean_5"] == 12.0
        assert larkin.features["fantasy_mean_5"] == pytest.approx(13.2)  # νίκη: PIR + |PIR|/10
        assert np.isfinite(larkin.predicted_fantasy) and np.isfinite(larkin.predicted_pir)

    def test_negative_pir_and_a_player_who_did_not_play(self, predictor):
        hazer = predictor.predict_player("P011201", as_of=AS_OF)  # PIR -4 σε νίκη
        assert hazer.features["pir_mean_5"] == -4.0
        assert hazer.features["fantasy_mean_5"] == pytest.approx(-3.6)  # νίκη: −4 + 0,4
        beaubois = predictor.predict_player("P006590", as_of=AS_OF)  # DNP στον 2025/1
        assert beaubois.n_prior_appearances == 0
        assert beaubois.last_appearance_date is None
        assert beaubois.features["pir_mean_5"] is None
        assert beaubois.features["dnp_streak"] == 1.0
        assert np.isfinite(beaubois.predicted_fantasy)

    def test_the_latest_name_and_old_style_ids(self, predictor):
        vezenkov = predictor.predict_player("P003469", as_of=AS_OF)
        assert vezenkov.name == "VEZENKOV, SASHA"  # το πιο πρόσφατο όνομα (2016: «ALEKSANDAR»)
        assert vezenkov.team_code == "OLY"
        assert vezenkov.n_prior_appearances == 2
        old_style = predictor.predict_player(" plru ", as_of=AS_OF)  # παλαιά μορφή ID, με κενά
        assert old_style is not None and old_style.player_id == "PLRU"
        assert old_style.name == "SIMONOVIC, MARKO"
        assert predictor.predict_player("PXXX") is None

    def test_a_game_on_the_day_of_as_of_is_the_next_game_but_the_following_day_is_not(
        self, predictor
    ):
        on_the_day = predictor.predict_player("P006835", as_of=date(2026, 10, 8))
        assert on_the_day.next_game.gamecode == 32
        after = predictor.predict_player("P006835", as_of=date(2026, 10, 9))
        assert after.next_game is None


class TestRankingsOnRealFixtures:
    def test_active_players_are_the_ones_of_the_current_season(self, predictor, engine):
        expected = players_in(engine, 2026)
        active = predictor.predict_all(as_of=AS_OF)
        assert {p.player_id for p in active} == expected
        assert all(p.team_code in {"BAS", "OLY", "BES", "PAM"} for p in active)
        values = [p.predicted_fantasy for p in active]
        assert values == sorted(values, reverse=True)

    def test_all_players_can_be_listed(self, predictor, engine):
        everyone = predictor.predict_all(as_of=AS_OF, active_only=False)
        with engine.connect() as connection:
            total = connection.execute(select(models.players.c.player_id)).all()
        assert len(everyone) == len(total) == 152
        assert all(
            np.isfinite(p.predicted_fantasy) and np.isfinite(p.predicted_pir) for p in everyone
        )
        assert all(set(p.features) == set(FEATURE_COLUMNS) for p in everyone)

    def test_the_team_filter(self, predictor):
        assert predictor.predict_all(as_of=AS_OF, team_code="TEL") == []  # κανένας ενεργός
        tel = predictor.predict_all(as_of=AS_OF, team_code="tel", active_only=False)
        assert tel and all(p.team_code == "TEL" for p in tel)
        assert all(p.next_game is not None and p.next_game.gamecode == 32 for p in tel)

    def test_predict_player_matches_predict_all(self, predictor):
        for prediction in predictor.predict_all(as_of=AS_OF, active_only=False)[::9]:
            assert predictor.predict_player(prediction.player_id, as_of=AS_OF) == prediction

    def test_refresh_after_new_data_does_not_break(self, predictor):
        before = predictor.predict_all(as_of=AS_OF)
        predictor.refresh()
        assert predictor.predict_all(as_of=AS_OF) == before
