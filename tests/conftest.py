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

from pathlib import Path

import pandas as pd
import pytest
from synthetic_league import League, make_league

from elfantasy.config import get_settings
from elfantasy.features.build import FEATURE_COLUMNS, build_features
from elfantasy.ingest import clean, pipeline
from elfantasy.ingest.fetch import RawCache
from elfantasy.model.artifact import ModelBundle, library_versions
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
