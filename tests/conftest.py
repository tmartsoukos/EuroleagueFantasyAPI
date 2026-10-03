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

from elfantasy.config import get_settings
from elfantasy.ingest import clean
from elfantasy.ingest.fetch import RawCache

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


@pytest.fixture
def fixture_cache(tmp_path) -> RawCache:
    """Cache με τα δεδομένα των fixtures, στη διάταξη του `data/raw/` (για --no-fetch)."""
    cache = RawCache(tmp_path / "data" / "raw")
    boxscores = read_raw_boxscores()
    for season, results in read_raw_results().items():
        cache.save_results(season, results)
    for season, schedule in read_raw_schedule().items():
        cache.save_schedule(season, schedule)
    for season, part in boxscores.groupby("Season"):
        cache.append_boxscores(int(season), [part.reset_index(drop=True)])
    return cache


@pytest.fixture(scope="session")
def fixture_dataset() -> clean.CleanData:
    """Το καθαρισμένο dataset των fixtures (υπολογίζεται μία φορά). Μόνο για ανάγνωση."""
    return clean.clean_all(read_raw_boxscores(), read_raw_results(), read_raw_schedule())
