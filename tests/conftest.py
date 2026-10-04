"""Κοινά fixtures των tests: πραγματικά boxscores, results και schedule από τον φάκελο fixtures.

Τα αρχεία `tests/fixtures/*.csv` προέρχονται από πραγματικές κλήσεις στο euroleague_api (7 αγώνες
των σεζόν 2016, 2023, 2024, 2025 και 2026 και 3 μελλοντικοί αγώνες του 2026), χωρίς καμία αλλαγή
στις τιμές. Οι πραγματικοί αγώνες είναι:

- 2016/1 και 2016/62: παλαιά IDs (π.χ. `PLRU`), DNP, αρνητικό PIR, ο Vezenkov ως «ALEKSANDAR».
- 2023/1: ο Simonovic με νέο ID (`P012711`) ενώ το 2016 είχε το `PLRU`.
- 2024/175 και 2026/8: ο DeJulius με διαφορετική γραφή ονόματος.
- 2025/1: ο αγώνας των παραδειγμάτων του FANTASY_RULES.md (Larkin, Hoard, Hazer, Beaubois).
- 2026/5: ο Vezenkov ως «SASHA» (πιο πρόσφατο όνομα).
- 2026/31, 32, 33: μελλοντικοί αγώνες (`played = false` στο schedule).
"""

import os
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from api_support import REAL_TODAY, FakeClock, make_settings, noon_utc
from pg_support import (
    ForbidSilentPostgresSkips,
    PostgresServer,
    create_database,
    drop_database,
    is_disposable,
    is_local_url,
    require_postgres,
)
from sqlalchemy import Engine, delete
from sqlalchemy.exc import SQLAlchemyError
from synthetic_league import League, make_league, write_to_database

from elfantasy.config import get_settings
from elfantasy.db import models
from elfantasy.db.migrate import Migrator
from elfantasy.db.session import get_engine
from elfantasy.db.urls import normalize_database_url
from elfantasy.features.build import FEATURE_COLUMNS, build_features
from elfantasy.ingest import clean, pipeline
from elfantasy.ingest.fetch import RawCache
from elfantasy.model.artifact import ModelBundle, library_versions
from elfantasy.model.predict import Predictor
from elfantasy.model.train import DEFAULT_XGB, ModelSpec, fit_ridge, fit_xgboost_final

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path_factory, monkeypatch):
    """Κανένα test δεν γράφει στον πραγματικό φάκελο `data/` ή στην πραγματική βάση."""
    folder = tmp_path_factory.mktemp("settings")
    monkeypatch.setenv("DATA_DIR", str(folder / "data"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(folder / 'data' / 'settings.db').as_posix()}")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def restore_logging():
    """Το `pipeline.main()` ρυθμίζει τον root logger: τον επαναφέρουμε και κλείνουμε τα νέα
    handlers (αλλιώς ένα handler πάνω σε stdout που έκλεισε το pytest θα έβγαζε σφάλματα)."""
    import logging

    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)


def read_raw_boxscores() -> pd.DataFrame:
    """Ακατέργαστο boxscore, όπως το επιστρέφει το euroleague_api (με trailing spaces στα IDs)."""
    text_columns = ["Player_ID", "Team", "Dorsal", "Player", "Minutes"]
    return pd.read_csv(
        FIXTURES_DIR / "boxscores.csv",
        dtype={column: str for column in text_columns},
        keep_default_na=False,
        na_values=[""],
    )


def read_raw_results() -> dict[int, pd.DataFrame]:
    """Results ανά σεζόν (όπως το `get_gamecodes_season`)."""
    frame = pd.read_csv(FIXTURES_DIR / "results.csv", keep_default_na=False)
    return {
        int(season): part.drop(columns=["season"]).reset_index(drop=True)
        for season, part in frame.groupby("season")
    }


def read_raw_schedule() -> dict[int, pd.DataFrame]:
    """Schedule ανά σεζόν (όπως το `Schedule.get_schedule`: όλα κείμενο εκτός από το gameday)."""
    frame = pd.read_csv(FIXTURES_DIR / "schedule.csv", dtype=str, keep_default_na=False)
    schedules = {}
    for season, part in frame.groupby("season"):
        part = part.drop(columns=["season"]).reset_index(drop=True)
        part["gameday"] = part["gameday"].astype(int)
        schedules[int(season)] = part
    return schedules


@pytest.fixture
def raw_boxscores() -> pd.DataFrame:
    return read_raw_boxscores()


@pytest.fixture
def raw_results() -> dict[int, pd.DataFrame]:
    return read_raw_results()


@pytest.fixture
def raw_schedule() -> dict[int, pd.DataFrame]:
    return read_raw_schedule()


def _populate_cache(cache: RawCache) -> RawCache:
    boxscores = read_raw_boxscores()
    for season, results in read_raw_results().items():
        cache.save_results(season, results)
    for season, schedule in read_raw_schedule().items():
        cache.save_schedule(season, schedule)
    for season, part in boxscores.groupby("Season"):
        cache.append_boxscores(int(season), [part.reset_index(drop=True)])
    return cache


@pytest.fixture
def fixture_cache(tmp_path) -> RawCache:
    """Cache με τα δεδομένα των fixtures, στη διάταξη του `data/raw/` (για --no-fetch)."""
    return _populate_cache(RawCache(tmp_path / "data" / "raw"))


@pytest.fixture(scope="session")
def fixture_database_url(tmp_path_factory) -> str:
    """URL βάσης SQLite που φτιάχνεται μία φορά από τα fixtures (ingestion pipeline).

    Περιέχει τους 7 πραγματικούς αγώνες και τους 3 μελλοντικούς των fixtures (σεζόν 2016, 2023,
    2024, 2025, 2026). Μόνο για ανάγνωση: κανένα test δεν πρέπει να την τροποποιεί.
    """
    root = tmp_path_factory.mktemp("fixture_database")
    _populate_cache(RawCache(root / "data" / "raw"))
    url = f"sqlite:///{(root / 'fixtures.db').as_posix()}"
    pipeline.run_pipeline(
        [2016, 2023, 2024, 2025, 2026], db_url=url, data_dir=root / "data", fetch=False
    )
    return url


@pytest.fixture(scope="session")
def fixture_dataset() -> clean.CleanData:
    """Το καθαρισμένο dataset των fixtures (υπολογίζεται μία φορά). Μόνο για ανάγνωση."""
    return clean.clean_all(read_raw_boxscores(), read_raw_results(), read_raw_schedule())


# --------------------------------------------------------------------------------------
# Φάση 3: συνθετικό πρωτάθλημα, features και μικρά μοντέλα για tests (χωρίς την πραγματική βάση)
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_league() -> League:
    """Ντετερμινιστικό συνθετικό πρωτάθλημα (4 σεζόν, 8 ομάδες, μελλοντικοί αγώνες)."""
    return make_league(seed=7)


@pytest.fixture(scope="session")
def synthetic_frame(synthetic_league: League) -> pd.DataFrame:
    """Τα features όλων των γραμμών του συνθετικού ιστορικού."""
    return build_features(synthetic_league.history, games=synthetic_league.games)


def _tiny_bundle(frame: pd.DataFrame, kind: str, version: str) -> ModelBundle:
    rows = frame[frame["is_appearance"]]
    features = rows[FEATURE_COLUMNS].to_numpy(dtype=float)
    targets = {"fantasy": rows["fantasy_score"].to_numpy(float), "pir": rows["pir"].to_numpy(float)}
    if kind == "ridge":
        models = {name: fit_ridge(features, y, alpha=10.0) for name, y in targets.items()}
    else:
        spec = ModelSpec("tiny", "xgb_absoluteerror", dict(DEFAULT_XGB))
        models = {
            name: fit_xgboost_final(spec, (features, y), rounds=25, seed=1)
            for name, y in targets.items()
        }
    return ModelBundle(
        model_version=version,
        feature_columns=list(FEATURE_COLUMNS),
        models=models,
        library_versions=library_versions(),
        trained_at="2026-01-01T00:00:00+00:00",
        metrics={"model_version": version, "threshold": {"value": 9.99, "passed": True}},
    )


@pytest.fixture(scope="session")
def tiny_xgb_bundle(synthetic_frame: pd.DataFrame) -> ModelBundle:
    """Μικρό μοντέλο XGBoost (25 δέντρα) εκπαιδευμένο στα συνθετικά δεδομένα."""
    return _tiny_bundle(synthetic_frame, "xgboost", "test-xgb-v1")


@pytest.fixture(scope="session")
def tiny_ridge_bundle(synthetic_frame: pd.DataFrame) -> ModelBundle:
    """Μικρό μοντέλο Ridge εκπαιδευμένο στα συνθετικά δεδομένα."""
    return _tiny_bundle(synthetic_frame, "ridge", "test-ridge-v1")


# --------------------------------------------------------------------------------------
# Φάση 4: API (TestClient πάνω σε προσωρινή βάση SQLite και μικρό μοντέλο)
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def api_today(synthetic_league: League) -> date:
    """Η ημέρα μετά τον τελευταίο παιγμένο αγώνα του συνθετικού πρωταθλήματος: οι επόμενοι
    αγώνες των ομάδων είναι στο πρόγραμμα."""
    return synthetic_league.last_played_date + timedelta(days=1)


@pytest.fixture(scope="session")
def api_engine(synthetic_league: League, tmp_path_factory):
    """Βάση SQLite με το συνθετικό πρωτάθλημα (όλοι οι πίνακες, και ο κενός player_availability)."""
    path = tmp_path_factory.mktemp("api") / "league.db"
    engine = get_engine(f"sqlite:///{path.as_posix()}")
    write_to_database(engine, synthetic_league)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def api_predictor(tiny_xgb_bundle: ModelBundle, api_engine, api_today: date) -> Predictor:
    """Ένας κοινός Predictor για όλα τα tests του API (η cache του κρατά τον υπολογισμό)."""
    return Predictor(tiny_xgb_bundle, api_engine, today=lambda: api_today)


@pytest.fixture
def api_clock(api_today: date) -> FakeClock:
    """Ρολόι των tests: 12:00 UTC της ημέρας `api_today`."""
    return FakeClock(noon_utc(api_today))


@pytest.fixture
def client(api_engine, api_predictor: Predictor, api_clock: FakeClock):
    """TestClient της εφαρμογής πάνω στο συνθετικό πρωτάθλημα, με κλειδί διαχειριστή `ADMIN_KEY`.

    Τα σφάλματα του server επιστρέφονται ως απαντήσεις (`raise_server_exceptions=False`). Η
    διαθεσιμότητα που γράφει το test σβήνεται στο τέλος, ώστε η κοινή βάση να μένει καθαρή.
    """
    from fastapi.testclient import TestClient

    from elfantasy.api.main import create_app

    app = create_app(
        settings=make_settings(), engine=api_engine, predictor=api_predictor, clock=api_clock
    )
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    with api_engine.begin() as connection:
        connection.execute(delete(models.player_availability))


@pytest.fixture(scope="session")
def real_client(fixture_database_url: str, tiny_xgb_bundle: ModelBundle):
    """TestClient πάνω στη βάση των ΠΡΑΓΜΑΤΙΚΩΝ fixtures (7 αγώνες και 3 μελλοντικοί του 2026).

    Μόνο για ανάγνωση: κανένα test δεν πρέπει να γράφει στη βάση (π.χ. διαθεσιμότητα). Το ρολόι
    είναι σταθερό στις 12:00 UTC της 2026-10-03.
    """
    from fastapi.testclient import TestClient

    from elfantasy.api.main import create_app

    engine = get_engine(fixture_database_url)
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


# --------------------------------------------------------------------------------------
# Φάση 5: εγγράψιμη βάση SQLite με το συνθετικό πρωτάθλημα (καταγραφή προβλέψεων)
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def league_template(tmp_path_factory, synthetic_league: League) -> Path:
    """Αρχείο SQLite με το συνθετικό πρωτάθλημα (πρότυπο): κάθε test παίρνει δικό του αντίγραφο."""
    path = tmp_path_factory.mktemp("league_template") / "league.db"
    engine = get_engine(f"sqlite:///{path.as_posix()}")
    write_to_database(engine, synthetic_league)
    engine.dispose()
    return path


@pytest.fixture
def league_db(league_template: Path, tmp_path: Path) -> Iterator[Engine]:
    """Φρέσκο ΕΓΓΡΑΨΙΜΟ αντίγραφο της βάσης του συνθετικού πρωταθλήματος (χωρίς predictions)."""
    import shutil

    copy = tmp_path / "league_copy.db"
    shutil.copyfile(league_template, copy)
    engine = get_engine(f"sqlite:///{copy.as_posix()}")
    yield engine
    engine.dispose()


@pytest.fixture
def league_predictor(league_db: Engine, tiny_xgb_bundle: ModelBundle, api_today: date) -> Predictor:
    """Predictor με το μικρό μοντέλο πάνω στο `league_db`, με σταθερή ημέρα `api_today`."""
    return Predictor(tiny_xgb_bundle, league_db, today=lambda: api_today)


# --------------------------------------------------------------------------------------
# Φάση 5: πραγματικό Postgres (ΜΟΝΟ τοπικό: TEST_DATABASE_URL ή ενσωματωμένος pgserver)
# --------------------------------------------------------------------------------------


def _postgres_unavailable(reason: str):
    if require_postgres():
        pytest.fail(
            f"PostgreSQL is required (ELFANTASY_REQUIRE_POSTGRES) but unavailable: {reason}"
        )
    pytest.skip(reason)


def pytest_configure(config):
    """Με ELFANTASY_REQUIRE_POSTGRES=1 (στο CI) κάθε test με marker `postgres` που παραλείπεται, για
    οποιονδήποτε λόγο, αποτυγχάνει ολόκληρη την εκτέλεση (βλ. `ForbidSilentPostgresSkips`)."""
    config.pluginmanager.register(ForbidSilentPostgresSkips(), "forbid-silent-postgres-skips")


@pytest.fixture(scope="session")
def postgres_server(tmp_path_factory) -> Iterator[PostgresServer]:
    """Ένας τοπικός server Postgres για τα tests που τον χρειάζονται (βλ. `tests/pg_support.py`).

    Παραλείπει τα tests αν δεν υπάρχει (ούτε `TEST_DATABASE_URL` ούτε `pixeltable-pgserver`) ή αν
    το `TEST_DATABASE_URL` δεν δείχνει σε τοπικό server: ποτέ remote βάση.
    """
    configured = os.environ.get("TEST_DATABASE_URL", "").strip()
    if configured:
        if not is_local_url(configured):
            _postgres_unavailable(
                "TEST_DATABASE_URL does not point to a local server: the tests never connect "
                "to remote databases"
            )
        yield PostgresServer(normalize_database_url(configured), disposable=is_disposable())
        return
    try:
        import pixeltable_pgserver
    except ImportError:
        _postgres_unavailable(
            "no PostgreSQL available: set TEST_DATABASE_URL (local server) or install "
            "pixeltable-pgserver (requirements-dev.txt)"
        )
    try:
        server = pixeltable_pgserver.get_server(
            tmp_path_factory.mktemp("pgdata"), cleanup_mode="stop"
        )
    except Exception as exc:
        _postgres_unavailable(f"the embedded PostgreSQL could not start ({type(exc).__name__})")
    try:
        yield PostgresServer(normalize_database_url(server.get_uri()), disposable=True)
    finally:
        server.cleanup()


@pytest.fixture
def pg_url(postgres_server: PostgresServer) -> Iterator[str]:
    """Το URL μιας ΚΕΝΗΣ βάσης στον τοπικό server, που διαγράφεται στο τέλος του test."""
    try:
        url = create_database(postgres_server)
    except SQLAlchemyError as exc:
        pytest.skip(f"cannot create a test database on the local server ({type(exc).__name__})")
    try:
        yield url
    finally:
        drop_database(postgres_server, url)


@pytest.fixture
def pg_engine(pg_url: str) -> Iterator[Engine]:
    """Engine (με τις ρυθμίσεις του project για Postgres) προς την κενή βάση `pg_url`."""
    engine = get_engine(pg_url)
    yield engine
    engine.dispose()


@pytest.fixture
def pg_migrated(pg_engine: Engine) -> Engine:
    """Η κενή βάση μετά την εφαρμογή όλων των migrations (όπως θα γίνει στο Supabase)."""
    Migrator(pg_engine).apply()
    return pg_engine


@pytest.fixture(scope="session")
def pg_league_url(postgres_server: PostgresServer, synthetic_league: League) -> Iterator[str]:
    """Βάση Postgres με τα migrations και το συνθετικό πρωτάθλημα (μόνο για ανάγνωση)."""
    url = create_database(postgres_server, "elfantasy_league")
    engine = get_engine(url)
    try:
        Migrator(engine).apply()
        write_to_database(engine, synthetic_league)
        engine.dispose()
        yield url
    finally:
        engine.dispose()
        drop_database(postgres_server, url)
