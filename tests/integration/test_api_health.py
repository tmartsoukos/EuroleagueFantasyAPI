"""Tests του `GET /health` και της φόρτωσης της εφαρμογής (lifespan).

Η υπηρεσία είναι `ok` (HTTP 200) μόνο αν η βάση φτάνει και το μοντέλο φορτώθηκε· αλλιώς είναι
`degraded` (HTTP 503) με σύντομα μηνύματα και χωρίς διαρροή διαδρομών αρχείων, URL βάσης ή stack
traces. Το import του module της εφαρμογής δεν πρέπει να αποτυγχάνει όταν λείπει το μοντέλο ή η
βάση.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from api_support import FakePredictor, make_prediction, make_settings
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from elfantasy.api.health import model_info
from elfantasy.api.main import create_app
from elfantasy.config import get_settings
from elfantasy.db.session import create_all, get_engine
from elfantasy.model.artifact import save_bundle
from elfantasy.model.predict import Predictor

REPO_ROOT = Path(__file__).resolve().parents[2]


def database_url(engine) -> str:
    return engine.url.render_as_string(hide_password=False)


def assert_no_leaks(text: str, *forbidden: object) -> None:
    """Η απάντηση δεν περιέχει διαδρομές, URL βάσης, stack traces ή εσωτερικά ονόματα."""
    for item in (*forbidden, "Traceback", 'File "', ".joblib", ".db", "sqlite", "sqlalchemy"):
        assert str(item) not in text, f"{item!r} leaked in {text!r}"


class TestHealthOk:
    def test_ok_with_database_and_model(self, client, synthetic_league, api_today):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["problems"] == []
        assert body["model"]["version"] == "test-xgb-v1"
        assert body["model"]["threshold"] == 9.99
        assert body["database"] == {
            "ok": True,
            "players": len(synthetic_league.players),
            "latest_played_game_date": synthetic_league.last_played_date.isoformat(),
            "next_scheduled_game_date": synthetic_league.first_future_date.isoformat(),
        }
        # Το ρολόι των tests είναι η ημέρα μετά τον τελευταίο αγώνα.
        assert body["data_age_days"] == 1
        assert body["data_loaded_through"] == synthetic_league.last_played_date.isoformat()

    def test_head_requests_get_the_same_status_for_monitors(self, client, api_clock, monkeypatch):
        assert client.head("/health").status_code == 200
        monkeypatch.setattr(
            "elfantasy.api.health.database_summary",
            lambda engine, today: (_ for _ in ()).throw(
                OperationalError("SELECT 1", {}, Exception())
            ),
        )
        assert client.head("/health").status_code == 503
        assert "head" not in client.get("/openapi.json").json()["paths"]["/health"]

    def test_the_check_does_not_compute_predictions(self, client, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("health must not compute predictions")

        monkeypatch.setattr(Predictor, "predict_all", fail)
        monkeypatch.setattr(Predictor, "predict_player", fail)
        assert client.get("/health").status_code == 200

    def test_without_future_games_there_is_no_next_game_date(self, client, api_clock):
        api_clock.advance(days=1000)
        body = client.get("/health").json()
        assert body["database"]["next_scheduled_game_date"] is None  # κανένας αγώνας στο μέλλον
        assert body["status"] == "ok"  # offseason: δεν είναι πρόβλημα

    def test_the_model_block_is_built_from_the_training_metrics(self):
        metrics = json.loads((REPO_ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
        info = model_info(SimpleNamespace(metrics=metrics, model_version="v1"))
        assert info.version == "v1"
        assert info.selected_model == metrics["selected_model"]["name"]
        assert info.test_mae == pytest.approx(metrics["threshold"]["test_mae"])
        assert info.threshold == pytest.approx(metrics["threshold"]["value"])
        assert info.trained_through_season == metrics["protocol"]["final_fit_through_season"]

    def test_a_later_refit_wins_over_the_final_fit_season(self):
        metrics = {"protocol": {"final_fit_through_season": 2024, "final_refit_through": 2026}}
        info = model_info(SimpleNamespace(metrics=metrics, model_version="v1"))
        assert info.trained_through_season == 2026

    @pytest.mark.parametrize("metrics", [{}, None, {"threshold": "bad", "selected_model": 3}])
    def test_missing_or_odd_metrics_become_null(self, metrics):
        info = model_info(SimpleNamespace(metrics=metrics, model_version="v1"))
        assert info.version == "v1"
        assert info.selected_model is None
        assert info.test_mae is None and info.threshold is None
        assert info.trained_through_season is None


class TestDegraded:
    def run(self, settings, **kwargs):
        app = create_app(settings=settings, **kwargs)
        return TestClient(app, raise_server_exceptions=False)

    def test_a_missing_model_is_degraded_but_the_application_runs(
        self, api_engine, tmp_path, caplog
    ):
        missing = tmp_path / "secret-folder" / "missing-model.joblib"
        settings = make_settings(database_url=database_url(api_engine), model_path=str(missing))
        with self.run(settings) as client, caplog.at_level("ERROR"):
            response = client.get("/health")
            assert response.status_code == 503
            body = response.json()
            assert body["status"] == "degraded"
            assert body["model"] is None
            assert body["database"]["ok"] is True
            assert body["problems"] == ["model: model artifact is missing or incompatible"]
            assert_no_leaks(response.text, tmp_path, "missing-model", "secret-folder")
            # τα endpoints που χρειάζονται το μοντέλο απαντούν 503 με καθαρό μήνυμα
            for path in ("/predict/P000001", "/rankings", "/players"):
                reply = client.get(path)
                assert reply.status_code == 503, path
                assert reply.json() == {"detail": "prediction model is not available"}
            # η διαθεσιμότητα χρειάζεται μόνο τη βάση
            assert client.get("/availability").status_code == 200
        assert "missing-model" in caplog.text  # η λεπτομέρεια γράφεται μόνο στο log του server

    def test_a_corrupt_model_is_degraded(self, api_engine, tmp_path):
        broken = tmp_path / "model.joblib"
        broken.write_bytes(b"this is not a joblib file")
        settings = make_settings(database_url=database_url(api_engine), model_path=str(broken))
        with self.run(settings) as client:
            response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["problems"] == ["model: model artifact is missing or incompatible"]
        assert_no_leaks(response.text, tmp_path, "this is not")

    def test_a_missing_database_file_is_degraded_and_nothing_is_created(self, tmp_path):
        target = tmp_path / "nowhere" / "missing.db"
        settings = make_settings(database_url=f"sqlite:///{target.as_posix()}")
        with self.run(settings) as client:
            response = client.get("/health")
            assert response.status_code == 503
            body = response.json()
            assert body["status"] == "degraded"
            assert body["database"] == {
                "ok": False,
                "players": None,
                "latest_played_game_date": None,
                "next_scheduled_game_date": None,
            }
            assert body["model"] is None
            assert "database: database is not available or not initialised" in body["problems"]
            assert_no_leaks(response.text, tmp_path, "nowhere")
            for path in ("/predict/P000001", "/rankings", "/availability"):
                reply = client.get(path)
                assert reply.status_code == 503, path
                assert reply.json() == {"detail": "database is not available"}
        assert not target.exists() and not target.parent.exists()  # καμία παρενέργεια στον δίσκο

    def test_a_corrupt_database_file_is_degraded(self, tmp_path):
        broken = tmp_path / "broken.db"
        broken.write_bytes(b"definitely not a sqlite database" * 20)
        settings = make_settings(database_url=f"sqlite:///{broken.as_posix()}")
        with self.run(settings) as client:
            response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["database"]["ok"] is False
        assert_no_leaks(response.text, tmp_path, "definitely")

    def test_an_invalid_database_url_is_degraded(self, caplog):
        settings = make_settings(database_url="notadialect://user:s3cret@host/db")
        with self.run(settings) as client, caplog.at_level("ERROR"):
            response = client.get("/health")
        assert response.status_code == 503
        assert_no_leaks(response.text, "s3cret", "notadialect")
        assert "s3cret" not in caplog.text  # ούτε στο log: τα credentials δεν γράφονται

    def test_a_database_without_players_is_degraded(self, tmp_path, tiny_xgb_bundle):
        engine = get_engine(f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")
        create_all(engine)
        model_file = tmp_path / "model.joblib"
        save_bundle(tiny_xgb_bundle, model_file)
        settings = make_settings(database_url=database_url(engine), model_path=str(model_file))
        with self.run(settings) as client:
            response = client.get("/health")
            body = response.json()
            assert response.status_code == 503
            assert body["model"]["version"] == "test-xgb-v1"
            assert body["database"]["ok"] is False
            assert body["database"]["players"] == 0
            assert body["problems"] == [
                "database: database contains no players (run the ingestion pipeline)"
            ]
            assert client.get("/predict/P000001").status_code == 404  # η υπηρεσία δουλεύει
            assert client.get("/rankings").json()["items"] == []
        engine.dispose()

    def test_the_database_going_away_after_startup_is_reported(self, client, monkeypatch):
        def fail(engine, today):
            raise OperationalError("SELECT secret FROM somewhere", {}, Exception("server down"))

        monkeypatch.setattr("elfantasy.api.health.database_summary", fail)
        response = client.get("/health")
        assert response.status_code == 503
        body = response.json()
        assert body["database"]["ok"] is False
        assert body["model"]["version"] == "test-xgb-v1"
        assert body["problems"] == ["database: database is not available or not initialised"]
        assert_no_leaks(response.text, "SELECT", "secret", "server down")

    def test_a_failure_of_the_first_predictions_is_degraded(self, api_engine, caplog):
        class Broken(FakePredictor):
            def predict_all(self, *args, **kwargs):
                raise RuntimeError("feature bug in C:/secret/path")

        app = create_app(
            settings=make_settings(),
            engine=api_engine,
            predictor=Broken([make_prediction("P1", 1)]),
        )
        with TestClient(app, raise_server_exceptions=False) as client, caplog.at_level("ERROR"):
            response = client.get("/health")
            assert response.status_code == 503
            assert response.json()["problems"] == ["model: predictions could not be computed"]
            assert response.json()["model"]["version"] == "fake-v1"
            assert_no_leaks(response.text, "feature bug", "secret")
            assert client.get("/rankings").json() == {"detail": "prediction model is not available"}
        assert "feature bug" in caplog.text

    def test_a_database_error_while_building_the_predictor_is_a_database_problem(
        self, api_engine, tmp_path, tiny_xgb_bundle, monkeypatch
    ):
        model_file = tmp_path / "model.joblib"
        save_bundle(tiny_xgb_bundle, model_file)

        def broken_load(cls, *args, **kwargs):
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))

        monkeypatch.setattr(Predictor, "load", classmethod(broken_load))
        settings = make_settings(database_url=database_url(api_engine), model_path=str(model_file))
        with self.run(settings) as client:
            response = client.get("/health")
        assert response.status_code == 503
        assert (
            "database: database is not available or not initialised" in response.json()["problems"]
        )
        assert_no_leaks(response.text, "connection refused")

    def test_an_application_whose_lifespan_did_not_run_is_not_ready(self, api_engine):
        app = create_app(settings=make_settings(), engine=api_engine)
        client = TestClient(app, raise_server_exceptions=False)  # χωρίς `with`: χωρίς lifespan
        response = client.get("/health")
        assert response.status_code == 503
        assert "application: application startup has not completed" in response.json()["problems"]
        assert client.get("/predict/P000001").json() == {"detail": "service is not ready"}


class TestLoading:
    def test_one_shared_engine_is_passed_to_the_predictor(
        self, api_engine, api_predictor, tmp_path, monkeypatch
    ):
        captured = {}

        def fake_load(cls, model_path=None, engine=None, **kwargs):
            captured.update(model_path=model_path, engine=engine)
            return api_predictor

        monkeypatch.setattr(Predictor, "load", classmethod(fake_load))
        model_file = tmp_path / "model.joblib"
        url = database_url(api_engine)
        app = create_app(settings=make_settings(database_url=url, model_path=str(model_file)))
        with TestClient(app, raise_server_exceptions=False) as client:
            state = app.state.api
            assert client.get("/health").status_code == 200
            assert captured["engine"] is state.engine  # ΜΙΑ κοινή engine
            assert captured["model_path"] == str(model_file)
            assert state.engine is not api_engine  # δημιουργήθηκε από το DATABASE_URL
            assert state.owns_engine and state.owns_predictor
        assert state.engine is None and state.predictor is None  # το shutdown τα έκλεισε

    def test_the_model_and_the_database_are_loaded_from_the_environment(
        self, api_engine, tmp_path, tiny_xgb_bundle, monkeypatch
    ):
        model_file = tmp_path / "model.joblib"
        save_bundle(tiny_xgb_bundle, model_file)
        monkeypatch.setenv("DATABASE_URL", database_url(api_engine))
        monkeypatch.setenv("MODEL_PATH", str(model_file))
        monkeypatch.setenv("ADMIN_API_KEY", "key-from-the-environment")
        get_settings.cache_clear()
        with TestClient(create_app(), raise_server_exceptions=False) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/rankings?limit=1").status_code == 200
            denied = client.post("/admin/refresh", headers={"X-API-Key": "wrong"})
            assert denied.status_code == 401
            allowed = client.post(
                "/admin/refresh", headers={"X-API-Key": "key-from-the-environment"}
            )
            assert allowed.status_code == 200

    def test_an_injected_engine_and_predictor_are_not_closed_by_the_application(
        self, api_engine, api_predictor
    ):
        app = create_app(settings=make_settings(), engine=api_engine, predictor=api_predictor)
        with TestClient(app, raise_server_exceptions=False):
            pass
        state = app.state.api
        assert state.engine is api_engine and state.predictor is api_predictor
        assert state.service is None and not state.started
        with TestClient(app, raise_server_exceptions=False) as client:  # νέο startup: ξανά ok
            assert client.get("/health").status_code == 200

    def test_the_version_falls_back_when_the_package_is_not_installed(self, monkeypatch):
        import importlib
        from importlib import metadata

        import elfantasy.api.main as main

        def not_installed(name):
            raise metadata.PackageNotFoundError(name)

        monkeypatch.setattr(metadata, "version", not_installed)
        try:
            assert importlib.reload(main).API_VERSION == "0.1.0"
        finally:
            monkeypatch.undo()
            importlib.reload(main)  # επαναφορά: το module δεν πρέπει να μείνει αλλαγμένο
        assert main.API_VERSION == metadata.version("elfantasy")

    def test_importing_the_module_works_without_a_model_or_a_database(self, tmp_path):
        env = {
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "DATABASE_URL": f"sqlite:///{(tmp_path / 'absent' / 'absent.db').as_posix()}",
            "MODEL_PATH": str(tmp_path / "absent.joblib"),
            "ADMIN_API_KEY": "",
        }
        code = (
            "import elfantasy.api.main as m; "
            "assert m.app.title and m.app.state.api.started is False; print('imported')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "imported" in result.stdout
        assert not (tmp_path / "absent").exists()  # το import δεν δημιουργεί αρχεία
