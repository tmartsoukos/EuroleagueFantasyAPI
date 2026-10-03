"""Tests του model/evaluate_recorded.py: καταγεγραμμένες προβλέψεις έναντι αποτελεσμάτων."""

import math
from datetime import date, timedelta

import pandas as pd
import pytest
from recorded_support import (
    busiest_game,
    make_legacy_predictions,
    play_game,
    predicted_for_game,
    recorded_rows,
)
from sqlalchemy import select

from elfantasy.db import models
from elfantasy.db.session import get_engine
from elfantasy.model import evaluate_recorded as evaluator
from elfantasy.model.evaluate_recorded import (
    APPEARED,
    DNP,
    NOT_IN_BOXSCORE,
    PENDING,
    EvaluationError,
    classify,
    evaluate,
    evaluate_recorded,
    format_evaluation,
    load_recorded,
)
from elfantasy.model.record_predictions import record_predictions

SECRET = "S3cr3t-Pa55"


# ----------------------------------------------------------------------------------------------
# Καθαρή λογική πάνω σε χειροποίητους πίνακες
# ----------------------------------------------------------------------------------------------


def frame(*rows) -> pd.DataFrame:
    """Γραμμές `(model_version, game_date, played, predicted, actual, dnp, minutes)`."""
    return pd.DataFrame(
        [
            {
                "player_id": f"P{index}",
                "season": 2026,
                "gamecode": 31,
                "model_version": version,
                "as_of": date(2026, 10, 3),
                "predicted_fantasy": predicted,
                "game_date": day,
                "played": played,
                "actual_fantasy": actual,
                "dnp": dnp,
                "minutes": minutes,
            }
            for index, (version, day, played, predicted, actual, dnp, minutes) in enumerate(rows)
        ]
    )


D1, D2 = date(2026, 10, 7), date(2026, 10, 8)


class TestClassify:
    def test_every_category(self):
        data = frame(
            ("v", D1, False, 10.0, None, None, None),  # δεν έχει παιχτεί
            ("v", D1, True, 10.0, 12.0, False, 22.5),  # αγωνίστηκε
            ("v", D1, True, 10.0, 0.0, True, 0.0),  # γραμμή DNP
            ("v", D1, True, 10.0, None, None, None),  # δεν υπάρχει στο boxscore
            ("v", D1, True, 10.0, -1.0, False, 0.0),  # γραμμή με 0 λεπτά και dnp = false
        )
        assert classify(data).tolist() == [PENDING, APPEARED, DNP, NOT_IN_BOXSCORE, DNP]

    def test_a_row_of_a_game_that_is_not_played_is_always_pending(self):
        data = frame(("v", D1, False, 1.0, 5.0, False, 30.0))
        assert classify(data).tolist() == [PENDING]


class TestEvaluate:
    def test_metrics_for_one_day(self):
        data = frame(
            ("v", D1, True, 10.0, 14.0, False, 20.0),  # σφάλμα −4
            ("v", D1, True, 20.0, 15.0, False, 25.0),  # σφάλμα +5
            ("v", D1, True, 8.0, 0.0, True, 0.0),  # DNP: δεν μετρά στο MAE, μετρά στο mae_incl_dnp
            ("v", D1, True, 9.0, None, None, None),  # όχι στο boxscore
            ("v", D1, False, 7.0, None, None, None),  # εκκρεμεί
        )
        result = evaluate(data)
        assert (result.recorded, result.pending) == (5, 1)
        day = result.daily.loc[("v", D1)]
        assert (day["predictions"], day["appeared"], day["dnp"], day["not_in_boxscore"]) == (
            4,
            2,
            1,
            1,
        )
        assert day["mae"] == pytest.approx((4 + 5) / 2)
        assert day["bias"] == pytest.approx((-4 + 5) / 2)
        assert day["mae_incl_dnp"] == pytest.approx((4 + 5 + 8) / 3)
        overall = result.overall.loc["v"]
        assert overall["mae"] == pytest.approx(4.5) and overall["predictions"] == 4

    def test_days_and_model_versions_are_grouped_separately(self):
        data = frame(
            ("v1", D1, True, 10.0, 12.0, False, 20.0),  # |e| = 2
            ("v1", D2, True, 10.0, 16.0, False, 20.0),  # |e| = 6
            ("v2", D1, True, 10.0, 11.0, False, 20.0),  # |e| = 1
        )
        result = evaluate(data)
        assert list(result.daily.index) == [("v1", D1), ("v1", D2), ("v2", D1)]
        assert result.daily["mae"].tolist() == pytest.approx([2.0, 6.0, 1.0])
        assert result.overall.loc["v1", "mae"] == pytest.approx(4.0)
        assert result.overall.loc["v2", "mae"] == pytest.approx(1.0)
        assert result.overall.loc["v1", "bias"] == pytest.approx(-4.0)  # το μοντέλο υποεκτιμά

    def test_a_day_without_appearances_has_no_mae_but_reports_the_counts(self):
        data = frame(
            ("v", D1, True, 10.0, 0.0, True, 0.0),
            ("v", D1, True, 10.0, None, None, None),
        )
        day = evaluate(data).daily.loc[("v", D1)]
        assert math.isnan(day["mae"]) and math.isnan(day["bias"])
        assert (day["appeared"], day["dnp"], day["not_in_boxscore"]) == (0, 1, 1)
        assert day["mae_incl_dnp"] == pytest.approx(10.0)

    def test_a_perfect_prediction_has_zero_error(self):
        data = frame(("v", D1, True, 12.5, 12.5, False, 10.0))
        assert evaluate(data).overall.loc["v", "mae"] == 0.0

    def test_nothing_played_yet(self):
        result = evaluate(
            frame(("v", D1, False, 1.0, None, None, None), ("v", D2, False, 2.0, None, None, None))
        )
        assert (result.recorded, result.pending) == (2, 2)
        assert result.daily.empty and result.overall.empty

    def test_an_empty_frame(self):
        result = evaluate(pd.DataFrame(columns=evaluator._FRAME_COLUMNS))
        assert (result.recorded, result.pending) == (0, 0)
        assert list(result.overall.columns) == evaluator.SUMMARY_COLUMNS

    def test_the_report_text(self):
        result = evaluate(
            frame(
                ("v", D1, True, 10.0, 14.0, False, 20.0),
                ("v", D1, True, 8.0, 0.0, True, 0.0),
                ("v", D2, False, 8.0, None, None, None),
            )
        )
        text_ = format_evaluation(result)
        assert "Recorded predictions: 3 (2 with a played game, 1 waiting" in text_
        assert "Overall" in text_ and "Per game day" in text_ and "2026-10-07" in text_
        assert "4.000" in text_  # MAE με τρία δεκαδικά
        assert "dnp" in text_ and "not_in_boxscore" in text_

    def test_the_report_when_nothing_is_played(self):
        text_ = format_evaluation(evaluate(frame(("v", D1, False, 1.0, None, None, None))))
        assert "Nothing to evaluate yet" in text_ and "Overall" not in text_


# ----------------------------------------------------------------------------------------------
# Με πραγματική βάση: καταγραφή → αγώνες που παίζονται → αξιολόγηση
# ----------------------------------------------------------------------------------------------


@pytest.fixture
def recorded(league_db, league_predictor, api_today):
    """Προβλέψεις της ημέρας `api_today` καταγεγραμμένες, χωρίς αγώνα που έχει παιχτεί."""
    record_predictions(league_db, league_predictor, as_of=api_today)
    return league_db


class TestAgainstTheDatabase:
    def test_nothing_is_evaluated_before_any_game_is_played(self, recorded):
        result = evaluate_recorded(recorded)
        assert result.recorded == len(recorded_rows(recorded)) == result.pending
        assert result.overall.empty and result.daily.empty

    def test_counts_and_errors_after_a_game_is_played(self, recorded, league_predictor):
        season, gamecode = busiest_game(recorded)
        predicted = {
            row["player_id"]: row["predicted_fantasy"]
            for row in predicted_for_game(recorded, season, gamecode)
        }
        players = sorted(predicted)
        assert len(players) >= 6
        actual = {players[0]: 14.0, players[1]: 3.5, players[2]: 22.0}
        dnp = [players[3]]  # players[4] και οι επόμενοι: χωρίς γραμμή στο boxscore
        play_game(recorded, league_predictor, season, gamecode, appeared=actual, dnp=dnp)

        result = evaluate_recorded(recorded)
        day = result.daily.iloc[0]
        missing = len(players) - 3 - 1
        assert (day["appeared"], day["dnp"], day["not_in_boxscore"]) == (3, 1, missing)
        assert day["predictions"] == len(players)
        errors = [predicted[p] - actual[p] for p in actual]
        assert day["mae"] == pytest.approx(sum(abs(e) for e in errors) / 3)
        assert day["bias"] == pytest.approx(sum(errors) / 3)
        incl = [abs(e) for e in errors] + [abs(predicted[players[3]] - 0.0)]
        assert day["mae_incl_dnp"] == pytest.approx(sum(incl) / 4)
        assert result.pending == result.recorded - len(players)
        assert result.overall.iloc[0]["mae"] == pytest.approx(day["mae"])

    def test_the_game_date_is_the_day_of_the_row(self, recorded, league_predictor):
        season, gamecode = busiest_game(recorded)
        players = sorted(row["player_id"] for row in predicted_for_game(recorded, season, gamecode))
        play_game(recorded, league_predictor, season, gamecode, appeared={players[0]: 10.0}, dnp=[])
        frame_ = load_recorded(recorded)
        game_day = frame_.loc[frame_["played"].astype(bool), "game_date"].iloc[0]
        with recorded.connect() as connection:
            expected = connection.execute(
                select(models.games.c.game_date).where(
                    (models.games.c.season == season) & (models.games.c.gamecode == gamecode)
                )
            ).scalar_one()
        assert pd.Timestamp(game_day).date() == expected
        assert evaluate_recorded(recorded).daily.index[0][1] == expected

    def test_a_played_game_without_any_boxscore_counts_everyone_as_missing(self, recorded):
        season, gamecode = busiest_game(recorded)
        with recorded.begin() as connection:
            connection.execute(
                models.games.update()
                .where((models.games.c.season == season) & (models.games.c.gamecode == gamecode))
                .values(played=True, home_score=80, away_score=70)
            )
        n = len(predicted_for_game(recorded, season, gamecode))
        day = evaluate_recorded(recorded).daily.iloc[0]
        assert (day["appeared"], day["dnp"], day["not_in_boxscore"], day["predictions"]) == (
            0,
            0,
            n,
            n,
        )
        assert math.isnan(day["mae"])

    def test_two_game_days_give_two_rows_and_a_combined_total(self, recorded, league_predictor):
        by_game: dict[tuple[int, int], list[str]] = {}
        for row in recorded_rows(recorded):
            by_game.setdefault((row["season"], row["gamecode"]), []).append(row["player_id"])
        ranked = sorted(by_game.items(), key=lambda item: -len(item[1]))
        (first, first_players), (second, second_players) = ranked[0], ranked[1]
        dates = {}
        with recorded.connect() as connection:
            for season, gamecode in (first, second):
                dates[(season, gamecode)] = connection.execute(
                    select(models.games.c.game_date).where(
                        (models.games.c.season == season) & (models.games.c.gamecode == gamecode)
                    )
                ).scalar_one()
        if dates[first] == dates[second]:  # δύο αγώνες την ίδια ημέρα: ο δεύτερος μετατίθεται
            with recorded.begin() as connection:
                connection.execute(
                    models.games.update()
                    .where(
                        (models.games.c.season == second[0])
                        & (models.games.c.gamecode == second[1])
                    )
                    .values(game_date=dates[second] + timedelta(days=1))
                )
        for key, players in ((first, first_players), (second, second_players)):
            actual = {players[0]: 10.0, players[1]: 20.0}
            play_game(recorded, league_predictor, key[0], key[1], appeared=actual, dnp=[])
        result = evaluate_recorded(recorded)
        assert len(result.daily) == 2
        assert result.overall.iloc[0]["appeared"] == 4
        # ίδιο πλήθος εμφανίσεων σε κάθε ημέρα: ο συνολικός MAE είναι ο μέσος όρος των ημερήσιων
        assert result.overall.iloc[0]["mae"] == pytest.approx(result.daily["mae"].mean())

    def test_the_filters(self, recorded, league_predictor):
        season, gamecode = busiest_game(recorded)
        players = sorted(row["player_id"] for row in predicted_for_game(recorded, season, gamecode))
        play_game(recorded, league_predictor, season, gamecode, appeared={players[0]: 10.0}, dnp=[])
        with recorded.connect() as connection:
            day = connection.execute(
                select(models.games.c.game_date).where(
                    (models.games.c.season == season) & (models.games.c.gamecode == gamecode)
                )
            ).scalar_one()
        assert not evaluate_recorded(recorded, since=day, until=day).daily.empty
        assert evaluate_recorded(recorded, since=day + timedelta(days=1)).recorded == (
            evaluate_recorded(recorded, since=day + timedelta(days=1)).pending
        )
        assert evaluate_recorded(recorded, until=day - timedelta(days=1)).recorded == 0
        assert evaluate_recorded(recorded, model_version="no-such-model").recorded == 0
        assert evaluate_recorded(recorded, model_version="test-xgb-v1").recorded == len(
            recorded_rows(recorded)
        )

    def test_two_model_versions_are_never_mixed(
        self, recorded, league_predictor, league_db, tiny_ridge_bundle, api_today
    ):
        from elfantasy.model.predict import Predictor

        ridge = Predictor(tiny_ridge_bundle, league_db, today=lambda: api_today)
        record_predictions(league_db, ridge, as_of=api_today)
        season, gamecode = busiest_game(league_db)
        players = sorted(
            {row["player_id"] for row in predicted_for_game(league_db, season, gamecode)}
        )
        play_game(
            league_db,
            league_predictor,
            season,
            gamecode,
            appeared={players[0]: 12.0, players[1]: 6.0},
            dnp=[],
        )
        result = evaluate_recorded(league_db)
        assert set(result.overall.index) == {"test-xgb-v1", "test-ridge-v1"}
        assert set(result.daily.index.get_level_values(0)) == {"test-xgb-v1", "test-ridge-v1"}
        xgb = evaluate_recorded(league_db, model_version="test-xgb-v1")
        assert list(xgb.overall.index) == ["test-xgb-v1"]
        assert xgb.overall.iloc[0]["mae"] == pytest.approx(result.overall.loc["test-xgb-v1", "mae"])

    def test_the_players_who_did_not_play_do_not_count_as_errors(self, recorded, league_predictor):
        season, gamecode = busiest_game(recorded)
        players = sorted(row["player_id"] for row in predicted_for_game(recorded, season, gamecode))
        predicted = {
            r["player_id"]: r["predicted_fantasy"]
            for r in predicted_for_game(recorded, season, gamecode)
        }
        play_game(
            recorded,
            league_predictor,
            season,
            gamecode,
            appeared={players[0]: predicted[players[0]]},
            dnp=players[1:4],
        )
        day = evaluate_recorded(recorded).daily.iloc[0]
        assert day["mae"] == pytest.approx(
            0.0
        )  # ο μοναδικός παίκτης που αγωνίστηκε προβλέφθηκε ακριβώς
        assert day["dnp"] == 3
        assert day["mae_incl_dnp"] > 0  # αλλά οι DNP φαίνονται στην ενημερωτική μετρική


class TestLoading:
    def test_a_missing_table(self, tmp_path):
        engine = get_engine(f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")
        with pytest.raises(EvaluationError, match="predictions table does not exist"):
            load_recorded(engine)
        engine.dispose()

    def test_an_empty_legacy_table_has_nothing_recorded(self, league_db):
        make_legacy_predictions(league_db)
        result = evaluate_recorded(league_db)
        assert (result.recorded, result.pending) == (0, 0)

    def test_the_loaded_frame_has_the_documented_columns(self, recorded):
        assert list(load_recorded(recorded).columns) == evaluator._FRAME_COLUMNS


class TestCli:
    @pytest.fixture
    def db_url(self, league_db):
        return league_db.url.render_as_string(hide_password=False)

    def test_a_report_is_printed(self, recorded, league_predictor, db_url, capsys):
        season, gamecode = busiest_game(recorded)
        players = sorted(row["player_id"] for row in predicted_for_game(recorded, season, gamecode))
        play_game(
            recorded,
            league_predictor,
            season,
            gamecode,
            appeared={players[0]: 10.0},
            dnp=[players[1]],
        )
        assert evaluator.main(["--db", db_url]) == 0
        out = capsys.readouterr().out
        assert "Recorded predictions:" in out and "Overall" in out and "Per game day" in out
        assert "test-xgb-v1" in out

    def test_nothing_to_evaluate_exits_with_0(self, recorded, db_url, capsys):
        assert evaluator.main(["--db", db_url]) == 0
        assert "Nothing to evaluate yet" in capsys.readouterr().out

    def test_the_filters_are_passed_on(self, recorded, db_url, capsys):
        assert (
            evaluator.main(
                [
                    "--db",
                    db_url,
                    "--model-version",
                    "nope",
                    "--since",
                    "2026-01-01",
                    "--until",
                    "2026-12-31",
                ]
            )
            == 0
        )
        assert "Recorded predictions: 0" in capsys.readouterr().out

    def test_a_missing_table_exits_with_1(self, tmp_path, capsys):
        url = f"sqlite:///{(tmp_path / 'none.db').as_posix()}"
        assert evaluator.main(["--db", url]) == 1
        assert "predictions table does not exist" in capsys.readouterr().out

    def test_the_default_database_comes_from_the_settings(self, db_url, monkeypatch, capsys):
        monkeypatch.setenv("DATABASE_URL", db_url)
        evaluator.get_settings.cache_clear()
        assert evaluator.main([]) == 0
        capsys.readouterr()

    def test_an_invalid_date_is_a_usage_error(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            evaluator.main(["--since", "yesterday"])
        assert excinfo.value.code == 2

    def test_an_invalid_url_exits_with_2(self, capsys):
        assert evaluator.main(["--db", f"postgresql://u:{SECRET}@h:badport/db"]) == 2
        assert SECRET not in capsys.readouterr().out

    def test_an_unexpected_error_exits_with_1_and_hides_the_password(
        self, db_url, monkeypatch, capsys
    ):
        def failing(*args, **kwargs):
            raise RuntimeError(f"boom postgresql://u:{SECRET}@h/db")

        monkeypatch.setattr(evaluator, "evaluate_recorded", failing)
        assert evaluator.main(["--db", db_url]) == 1
        out = capsys.readouterr().out
        assert "the evaluation failed" in out and SECRET not in out
