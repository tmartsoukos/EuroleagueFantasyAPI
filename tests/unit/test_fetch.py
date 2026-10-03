"""Tests του ingest/fetch.py (offline): ρυθμός, retry, 429, κενό σώμα, cache, missing."""

import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from xml.parsers.expat import ExpatError

import pandas as pd
import pytest
import requests

from elfantasy.ingest import fetch
from elfantasy.ingest.fetch import (
    DataUnavailableError,
    EuroleagueApiSource,
    FetchError,
    RateLimitedError,
    RateLimiter,
    RawCache,
    RequestRunner,
    RetryPolicy,
    SeasonReport,
    expected_gamecodes,
    fetch_season,
    fetch_seasons,
    parse_retry_after,
    season_in_progress,
)

SRC_DIR = Path(__file__).resolve().parents[2] / "src"


class FakeClock:
    """Ψεύτικο ρολόι: το sleep προχωρά τον χρόνο χωρίς να περιμένει πραγματικά."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def http_error(status: int | None, retry_after: object = None) -> requests.HTTPError:
    if status is None:
        return requests.HTTPError("no response")
    response = requests.Response()
    response.status_code = status
    if retry_after is not None:
        response.headers["Retry-After"] = str(retry_after)
    return requests.HTTPError(f"HTTP {status}", response=response)


def empty_body_error() -> json.JSONDecodeError:
    return json.JSONDecodeError("Expecting value", "", 0)


class Script:
    """Καλούμενο που επιστρέφει ή σηκώνει διαδοχικά τα αποτελέσματα, με επανάληψη του τελευταίου."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self):
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def make_runner(clock: FakeClock, rps: float = 1.0, policy: RetryPolicy | None = None):
    limiter = RateLimiter(rps, clock=clock.monotonic, sleep=clock.sleep)
    return RequestRunner(limiter, policy, sleep=clock.sleep)


# ----------------------------------------------------------------------------------------------
# RateLimiter
# ----------------------------------------------------------------------------------------------


class TestRateLimiter:
    def test_first_request_is_immediate_and_the_next_ones_are_spaced(self):
        clock = FakeClock()
        limiter = RateLimiter(1.0, clock=clock.monotonic, sleep=clock.sleep)
        assert limiter.wait() == 0
        assert limiter.wait() == pytest.approx(1.0)
        assert limiter.wait() == pytest.approx(1.0)
        assert clock.sleeps == [1.0, 1.0]

    def test_interval_follows_the_rate(self):
        clock = FakeClock()
        limiter = RateLimiter(0.5, clock=clock.monotonic, sleep=clock.sleep)
        limiter.wait()
        assert limiter.wait() == pytest.approx(2.0)
        fast = RateLimiter(1.5, clock=clock.monotonic, sleep=clock.sleep)
        fast.wait()
        assert fast.wait() == pytest.approx(1 / 1.5)

    def test_time_spent_in_the_request_counts_towards_the_interval(self):
        clock = FakeClock()
        limiter = RateLimiter(1.0, clock=clock.monotonic, sleep=clock.sleep)
        limiter.wait()
        clock.now += 0.4  # διάρκεια του αιτήματος
        assert limiter.wait() == pytest.approx(0.6)

    def test_slow_responses_do_not_cause_a_burst_afterwards(self):
        clock = FakeClock()
        limiter = RateLimiter(1.0, clock=clock.monotonic, sleep=clock.sleep)
        limiter.wait()
        clock.now += 5.0  # αργή απάντηση
        assert limiter.wait() == 0
        assert limiter.wait() == pytest.approx(1.0)  # δεν «μαζεύονται» αιτήματα

    def test_slow_down_halves_the_rate_down_to_a_floor(self):
        clock = FakeClock()
        limiter = RateLimiter(1.0, clock=clock.monotonic, sleep=clock.sleep)
        assert limiter.slow_down() == 0.5
        assert limiter.rps == 0.5
        for _ in range(10):
            limiter.slow_down()
        assert limiter.rps == fetch.MIN_RPS

    def test_slow_down_applies_the_new_interval_from_the_last_request(self):
        clock = FakeClock()
        limiter = RateLimiter(1.0, clock=clock.monotonic, sleep=clock.sleep)
        limiter.wait()
        limiter.slow_down()
        assert limiter.wait() == pytest.approx(2.0)

    def test_block_for_delays_the_next_request(self):
        clock = FakeClock()
        limiter = RateLimiter(1.0, clock=clock.monotonic, sleep=clock.sleep)
        limiter.wait()
        limiter.block_for(30)
        assert limiter.wait() == pytest.approx(30.0)

    @pytest.mark.parametrize("rps", [0, -1.0, 1.6, 2.2])
    def test_rate_must_be_positive_and_not_above_the_maximum(self, rps):
        with pytest.raises(ValueError, match="rps"):
            RateLimiter(rps)

    def test_the_documented_defaults(self):
        assert fetch.DEFAULT_RPS == 1.0
        assert fetch.MAX_RPS == 1.5
        assert RateLimiter(fetch.MAX_RPS).rps == 1.5


class TestRetryAfter:
    def test_seconds(self):
        assert parse_retry_after("243") == 243.0
        assert parse_retry_after(" 7.5 ") == 7.5
        assert parse_retry_after("0") == 0.0

    def test_negative_values_become_zero(self):
        assert parse_retry_after("-5") == 0.0

    def test_http_date(self):
        future = datetime.now(UTC) + timedelta(seconds=120)
        value = parse_retry_after(format_datetime(future, usegmt=True))
        assert value is not None and 100 < value <= 120
        past = datetime.now(UTC) - timedelta(seconds=120)
        assert parse_retry_after(format_datetime(past, usegmt=True)) == 0.0

    def test_http_date_without_a_time_zone_is_treated_as_utc(self):
        # Το «-0000» σημαίνει άγνωστη ζώνη: το parsedate_to_datetime επιστρέφει naive datetime.
        assert parse_retry_after("Wed, 21 Oct 2099 07:28:00 -0000") > 10**9

    @pytest.mark.parametrize("value", [None, "", "soon", "12 parsecs"])
    def test_invalid_values(self, value):
        assert parse_retry_after(value) is None


# ----------------------------------------------------------------------------------------------
# RequestRunner
# ----------------------------------------------------------------------------------------------


class TestRequestRunner:
    def test_success(self):
        runner = make_runner(FakeClock())
        assert runner.call(lambda: "ok") == "ok"
        assert runner.stats.requests == 1

    def test_requests_are_spaced_by_the_limiter(self):
        clock = FakeClock()
        runner = make_runner(clock, rps=1.0)
        starts = []
        for _ in range(4):
            runner.call(lambda: starts.append(clock.now))
        assert [b - a for a, b in zip(starts, starts[1:], strict=False)] == pytest.approx([1.0] * 3)

    def test_429_waits_retry_after_plus_margin_and_halves_the_rate(self):
        clock = FakeClock()
        runner = make_runner(clock)
        script = Script(http_error(429, 7), "ok")
        assert runner.call(script) == "ok"
        assert script.calls == 2
        assert clock.sleeps == [pytest.approx(12.0)]  # 7 s Retry-After + 5 s περιθώριο
        assert runner.limiter.rps == 0.5
        assert runner.stats.rate_limited == 1
        assert runner.stats.rate_limit_wait_seconds == pytest.approx(12.0)

    def test_the_slower_rate_applies_to_the_following_requests(self):
        clock = FakeClock()
        runner = make_runner(clock)
        runner.call(Script(http_error(429, 1), "ok"))
        before = len(clock.sleeps)
        runner.call(lambda: "next")
        assert clock.sleeps[before:] == [pytest.approx(2.0)]  # 1/0.5 s ανάμεσα στα αιτήματα

    def test_429_without_retry_after_backs_off_exponentially(self):
        clock = FakeClock()
        runner = make_runner(clock)
        runner.call(Script(http_error(429), http_error(429), "ok"))
        assert clock.sleeps == [pytest.approx(65.0), pytest.approx(125.0)]  # 60+5, 120+5
        assert runner.limiter.rps == 0.25

    def test_huge_retry_after_is_capped(self):
        clock = FakeClock()
        runner = make_runner(clock)
        runner.call(Script(http_error(429, 100000), "ok"))
        assert clock.sleeps == [pytest.approx(905.0)]  # 900 s ανώτατο όριο + 5 s περιθώριο

    def test_persistent_429_gives_up_with_rate_limited_error(self):
        clock = FakeClock()
        policy = RetryPolicy(max_rate_limit_retries=2)
        runner = make_runner(clock, policy=policy)
        script = Script(http_error(429, 1))
        with pytest.raises(RateLimitedError):
            runner.call(script, label="boxscore 2025/9")
        assert script.calls == 3  # 1 + 2 επαναλήψεις
        assert runner.stats.rate_limited == 3

    def test_server_errors_use_exponential_backoff(self):
        clock = FakeClock()
        runner = make_runner(clock)
        assert runner.call(Script(http_error(503), http_error(502), "ok")) == "ok"
        assert clock.sleeps == [2.0, 4.0]
        assert runner.stats.server_errors == 2

    def test_server_errors_give_up_after_a_finite_number_of_attempts(self):
        clock = FakeClock()
        runner = make_runner(clock)
        script = Script(http_error(500))
        with pytest.raises(FetchError, match="HTTP 500 after 5 attempts"):
            runner.call(script)
        assert script.calls == 5
        assert clock.sleeps == [2.0, 4.0, 8.0, 16.0]

    @pytest.mark.parametrize("error", [requests.Timeout(), requests.ConnectionError("reset")])
    def test_timeouts_and_connection_errors_are_retried(self, error):
        clock = FakeClock()
        runner = make_runner(clock)
        assert runner.call(Script(error, "ok")) == "ok"
        assert clock.sleeps == [2.0]
        assert runner.stats.network_errors == 1

    def test_http_error_without_response_is_treated_as_a_network_error(self):
        runner = make_runner(FakeClock())
        assert runner.call(Script(http_error(None), "ok")) == "ok"
        assert runner.stats.network_errors == 1

    def test_persistent_timeouts_raise_fetch_error(self):
        runner = make_runner(FakeClock(), policy=RetryPolicy(max_attempts=3))
        script = Script(requests.Timeout())
        with pytest.raises(FetchError, match="Timeout after 3 attempts"):
            runner.call(script)
        assert script.calls == 3

    @pytest.mark.parametrize(
        "error",
        [
            empty_body_error(),
            requests.exceptions.JSONDecodeError("Expecting value", "", 0),
            ExpatError("no element found"),
        ],
        ids=["json", "requests-json", "xml"],
    )
    def test_empty_body_is_retried_a_few_times_and_then_reported_unavailable(self, error):
        clock = FakeClock()
        runner = make_runner(clock)
        script = Script(error)
        with pytest.raises(DataUnavailableError, match="empty or invalid response body"):
            runner.call(script)
        assert script.calls == 3  # η πρώτη προσπάθεια και 2 ξαναδοκιμές, όχι ατέρμονα
        assert clock.sleeps == [3.0, 3.0]
        assert runner.stats.empty_responses == 3

    def test_empty_body_can_recover(self):
        runner = make_runner(FakeClock())
        assert runner.call(Script(empty_body_error(), "data")) == "data"

    def test_other_http_statuses_are_unavailable_without_retries(self):
        script = Script(http_error(404))
        with pytest.raises(DataUnavailableError, match="HTTP 404"):
            make_runner(FakeClock()).call(script)
        assert script.calls == 1

    def test_classified_errors_of_the_source_pass_through(self):
        script = Script(DataUnavailableError("no statistics"))
        with pytest.raises(DataUnavailableError, match="no statistics"):
            make_runner(FakeClock()).call(script)
        assert script.calls == 1

    def test_unexpected_errors_are_not_retried(self):
        script = Script(KeyError("Stats"))
        with pytest.raises(FetchError, match="unexpected response format"):
            make_runner(FakeClock()).call(script)
        assert script.calls == 1


# ----------------------------------------------------------------------------------------------
# Πηγή δεδομένων: το πραγματικό πακέτο euroleague_api με προσομοίωση HTTP
# ----------------------------------------------------------------------------------------------


def fake_response(status: int, body: bytes = b"", headers: dict | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = body
    response.url = "https://live.euroleague.net/api/Boxscore"
    response.headers.update(headers or {})
    return response


def boxscore_json(game: pd.DataFrame) -> bytes:
    """Το JSON του Boxscore endpoint, φτιαγμένο από τις γραμμές ενός πραγματικού αγώνα."""
    stats = []
    for home in (1, 0):
        part = game[game["Home"] == home].drop(columns=["Season", "Gamecode", "Home"])
        ids = part["Player_ID"].str.strip()

        def records(frame):
            return json.loads(frame.to_json(orient="records"))

        stats.append(
            {
                "Team": "TEAM",
                "Coach": "COACH",
                "PlayersStats": records(part[~ids.isin(["Team", "Total"])]),
                "tmr": records(part[ids == "Team"])[0],
                "totr": records(part[ids == "Total"])[0],
            }
        )
    return json.dumps({"Live": False, "Stats": stats, "ByQuarter": [], "EndOfQuarter": []}).encode()


class TestEuroleagueApiSource:
    def test_package_http_errors_are_handled_by_the_runner(self, monkeypatch, raw_boxscores):
        """429, κενό σώμα και έγκυρη απάντηση μέσα από τον πραγματικό κώδικα του πακέτου."""
        game = raw_boxscores[(raw_boxscores["Season"] == 2025) & (raw_boxscores["Gamecode"] == 1)]
        responses = [
            fake_response(429, b'{"title": "Error 1015"}', {"Retry-After": "243"}),
            fake_response(200, b""),
            fake_response(500),
            fake_response(200, boxscore_json(game)),
        ]
        requested = []

        def fake_get(url, params=None, headers=None, timeout=None):
            requested.append((url, params))
            return responses.pop(0)

        monkeypatch.setattr(requests, "get", fake_get)
        clock = FakeClock()
        runner = make_runner(clock)
        source = EuroleagueApiSource()
        frame = runner.call(lambda: source.boxscore(2025, 1), label="boxscore 2025/1")

        assert not responses  # καταναλώθηκαν και οι τέσσερις απαντήσεις
        assert requested[0] == (
            "https://live.euroleague.net/api/Boxscore",
            {"gamecode": 1, "seasoncode": "E2025"},
        )
        assert runner.stats.rate_limited == 1
        assert runner.stats.empty_responses == 1
        assert runner.stats.server_errors == 1
        assert clock.sleeps[0] == pytest.approx(248.0)  # Retry-After 243 s + 5 s
        assert runner.limiter.rps == 0.5
        # Το DataFrame του πακέτου έχει τις ίδιες γραμμές και στήλες με τον πραγματικό αγώνα.
        assert list(frame.columns) == list(game.columns)
        assert frame["Valuation"].tolist() == game["Valuation"].tolist()
        assert frame["Player_ID"].str.strip().tolist() == game["Player_ID"].str.strip().tolist()

    def test_empty_skeleton_boxscore_is_reported_as_unavailable(self, monkeypatch):
        """Αγώνας 2018/21: το API απαντά με ομάδες «N/D», κενούς παίκτες και `tmr` null."""
        totals = {"Minutes": "", "Points": 0, "Valuation": 0}
        skeleton_team = {
            "Team": "N/D",
            "Coach": "",
            "PlayersStats": [],
            "tmr": None,
            "totr": totals,
        }
        body = json.dumps({"Live": False, "Stats": [skeleton_team, skeleton_team]}).encode()
        requested = []

        def fake_get(url, params=None, headers=None, timeout=None):
            requested.append(params)
            return fake_response(200, body)

        monkeypatch.setattr(requests, "get", fake_get)
        clock = FakeClock()
        runner = make_runner(clock)
        source = EuroleagueApiSource()
        with pytest.raises(DataUnavailableError, match="2018/21 has no player statistics"):
            runner.call(lambda: source.boxscore(2018, 21), label="boxscore 2018/21")
        assert len(requested) == 1  # καμία επανάληψη: η απάντηση δεν θα αλλάξει
        assert clock.sleeps == []

    def test_results_and_schedule_are_delegated_to_the_package(self):
        source = EuroleagueApiSource()
        source._data.get_gamecodes_season = lambda season: ("results", season)
        source._schedule.get_schedule = lambda season: ("schedule", season)
        assert source.results(2025) == ("results", 2025)
        assert source.schedule(2026) == ("schedule", 2026)


# ----------------------------------------------------------------------------------------------
# Root logger
# ----------------------------------------------------------------------------------------------


class TestRootLogger:
    def test_context_manager_restores_handlers_and_level(self):
        root = logging.getLogger()
        handlers, level = list(root.handlers), root.level
        with fetch.preserved_root_logger():
            root.addHandler(logging.NullHandler())
            root.setLevel(logging.DEBUG)
        assert root.handlers == handlers
        assert root.level == level

    def test_importing_the_package_does_not_change_the_root_logger(self):
        """Σε καθαρό interpreter, γιατί το πακέτο καλεί logging.basicConfig κατά το import."""
        code = (
            "import logging\n"
            "from elfantasy.ingest.fetch import load_euroleague_api\n"
            "root = logging.getLogger()\n"
            "before = (list(root.handlers), root.level)\n"
            "api = load_euroleague_api()\n"
            "after = (list(root.handlers), root.level)\n"
            "assert before == after, (before, after)\n"
            "assert logging.getLogger('euroleague_api').level == logging.CRITICAL\n"
            "assert hasattr(api, 'BoxScoreData')\n"
            "print('OK')\n"
        )
        env = {**os.environ, "PYTHONPATH": str(SRC_DIR), "PYTHONUTF8": "1"}
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "OK"

    def test_the_module_does_not_import_the_package_at_import_time(self):
        code = (
            "import sys\n"
            "import elfantasy.ingest.fetch\n"
            "assert 'euroleague_api' not in sys.modules\n"
            "print('OK')\n"
        )
        env = {**os.environ, "PYTHONPATH": str(SRC_DIR), "PYTHONUTF8": "1"}
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
        )
        assert done.returncode == 0, done.stderr


# ----------------------------------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------------------------------


def box(season: int, gamecode: int, marker: int = 0) -> pd.DataFrame:
    """Μικρό «boxscore» με τρεις γραμμές για τα tests του cache και της άντλησης."""
    return pd.DataFrame(
        {
            "Season": season,
            "Gamecode": gamecode,
            "Player_ID": ["P1       ", "Team", "Total"],
            "Points": [marker, 0, marker],
        }
    )


def results_frame(*scores: tuple[int, int, int]) -> pd.DataFrame:
    """Results (gameCode, homescore, awayscore) με το `played` του πακέτου πάντα True."""
    return pd.DataFrame(
        {
            "gameCode": [s[0] for s in scores],
            "homescore": [s[1] for s in scores],
            "awayscore": [s[2] for s in scores],
            "played": True,
        }
    )


def schedule_frame(**played: str) -> pd.DataFrame:
    """Schedule: `g1="true", g2="false"` -> αγώνες 1 και 2 με το αντίστοιχο played."""
    return pd.DataFrame(
        {"game": [name[1:] for name in played], "played": list(played.values()), "gameday": 1}
    )


class TestReplaceWithRetry:
    def test_retries_when_the_file_is_briefly_locked(self, tmp_path, monkeypatch):
        source, target = tmp_path / "a.tmp", tmp_path / "a.parquet"
        source.write_text("new", encoding="utf-8")
        real_replace = os.replace
        calls = []

        def flaky(src, dst):
            calls.append(src)
            if len(calls) < 3:
                raise PermissionError("[WinError 5] Access is denied")
            return real_replace(src, dst)

        monkeypatch.setattr(fetch.os, "replace", flaky)
        sleeps = []
        fetch.replace_with_retry(source, target, sleep=sleeps.append)
        assert target.read_text(encoding="utf-8") == "new"
        assert len(calls) == 3 and sleeps == [0.25, 0.25]

    def test_gives_up_after_the_last_attempt(self, tmp_path, monkeypatch):
        def locked(src, dst):
            raise PermissionError("locked")

        monkeypatch.setattr(fetch.os, "replace", locked)
        with pytest.raises(PermissionError):
            fetch.replace_with_retry(
                tmp_path / "x.tmp", tmp_path / "x", attempts=3, sleep=lambda seconds: None
            )

    def test_cache_writes_use_the_retrying_replace(self, tmp_path, monkeypatch):
        attempts = []
        real_replace = os.replace

        def flaky(src, dst):
            attempts.append(1)
            if len(attempts) == 1:
                raise PermissionError("locked")
            return real_replace(src, dst)

        monkeypatch.setattr(fetch.os, "replace", flaky)
        cache = RawCache(tmp_path)
        cache.append_boxscores(2025, [box(2025, 1)])
        assert cache.boxscore_gamecodes(2025) == {1}


class TestRawCache:
    def test_boxscores_are_appended_and_replaced_per_game(self, tmp_path):
        cache = RawCache(tmp_path / "raw")
        assert cache.load_boxscores(2025) is None
        assert cache.boxscore_gamecodes(2025) == set()
        cache.append_boxscores(2025, [box(2025, 2, 5), box(2025, 1, 5)])
        cache.append_boxscores(2025, [box(2025, 3), box(2025, 2, 9)])  # ο αγώνας 2 ξαναγράφεται
        frame = cache.load_boxscores(2025)
        assert frame["Gamecode"].tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3]
        assert frame[frame["Gamecode"] == 2]["Points"].tolist() == [9, 0, 9]
        assert cache.boxscore_gamecodes(2025) == {1, 2, 3}
        assert sorted(p.name for p in (tmp_path / "raw").iterdir()) == ["boxscores_2025.parquet"]

    def test_empty_append_writes_nothing(self, tmp_path):
        cache = RawCache(tmp_path)
        cache.append_boxscores(2025, [])
        assert list(tmp_path.iterdir()) == []

    def test_results_and_schedule_roundtrip(self, tmp_path):
        cache = RawCache(tmp_path)
        results, schedule = results_frame((1, 80, 70)), schedule_frame(g1="true")
        cache.save_results(2025, results)
        cache.save_schedule(2025, schedule)
        pd.testing.assert_frame_equal(cache.load_results(2025), results)
        pd.testing.assert_frame_equal(cache.load_schedule(2025), schedule)
        assert cache.load_results(2024) is None and cache.load_schedule(2024) is None

    def test_missing_report_accumulates_seasons(self, tmp_path):
        cache = RawCache(tmp_path)
        assert cache.load_missing() == {}
        cache.record_season(SeasonReport(2024, expected=3, fetched=3))
        cache.record_season(
            SeasonReport(2025, expected=3, fetched=2, missing=[3], reasons={3: "empty body"})
        )
        data = cache.load_missing()
        assert set(data) == {"2024", "2025"}
        assert data["2025"]["missing"] == [3]
        assert data["2025"]["reasons"] == {"3": "empty body"}
        assert data["2024"]["missing"] == []
        assert not list(tmp_path.glob("*.tmp"))


# ----------------------------------------------------------------------------------------------
# Άντληση σεζόν
# ----------------------------------------------------------------------------------------------


class FakeSource:
    """Πηγή που καταγράφει τις κλήσεις και ακολουθεί σενάρια ανά αγώνα."""

    def __init__(self, results=None, schedule=None, behaviour=None):
        self._results = results
        self._schedule = schedule
        self.behaviour: dict[int, Script] = behaviour or {}
        self.calls: list[tuple] = []

    def results(self, season):
        self.calls.append(("results", season))
        if isinstance(self._results, BaseException):
            raise self._results
        return self._results

    def schedule(self, season):
        self.calls.append(("schedule", season))
        if isinstance(self._schedule, BaseException):
            raise self._schedule
        return self._schedule

    def boxscore(self, season, gamecode):
        self.calls.append(("boxscore", season, gamecode))
        script = self.behaviour.get(gamecode)
        if script is not None:
            outcome = script()
            if outcome is not None:
                return outcome
        return box(season, gamecode, gamecode)

    def boxscore_calls(self) -> list[int]:
        return [call[2] for call in self.calls if call[0] == "boxscore"]


FIVE_PLAYED = results_frame((1, 80, 70), (2, 70, 75), (3, 90, 60), (4, 71, 72), (5, 88, 80))
ALL_PLAYED = schedule_frame(g1="true", g2="true", g3="true", g4="true", g5="true")
IN_PROGRESS = schedule_frame(g1="true", g2="true", g3="true", g4="true", g5="true", g6="false")


def run_season(source, cache, clock=None, **options):
    clock = clock or FakeClock()
    runner = make_runner(clock)
    options.setdefault("retry_pause", 0)
    return fetch_season(2025, source, runner, cache, sleep=clock.sleep, **options), runner, clock


class TestExpectedGames:
    def test_scores_and_schedule_decide_not_the_played_flag_of_the_package(self):
        results = results_frame(
            (1, 80, 70), (2, 0, 0), (3, 75, 74)
        )  # το «played» είναι True παντού
        schedule = schedule_frame(g3="true", g4="true", g5="false")
        # Ο αγώνας 2 έχει σκορ 0-0 (δεν έχει παιχτεί), ο 4 είναι παιγμένος μόνο στο schedule.
        assert expected_gamecodes(results, schedule) == [1, 3, 4]

    def test_missing_inputs(self):
        assert expected_gamecodes(None, None) == []
        assert expected_gamecodes(results_frame((7, 1, 2)), None) == [7]
        assert expected_gamecodes(None, schedule_frame(g8="true")) == [8]

    def test_season_in_progress(self):
        assert season_in_progress(None)
        assert season_in_progress(IN_PROGRESS)
        assert not season_in_progress(ALL_PLAYED)


class TestFetchSeason:
    def test_downloads_everything_and_reports_the_counts(self, tmp_path):
        cache = RawCache(tmp_path)
        source = FakeSource(FIVE_PLAYED, ALL_PLAYED)
        report, runner, _ = run_season(source, cache)
        assert (report.expected, report.fetched, report.new, report.missing) == (5, 5, 5, [])
        assert report.metadata_refreshed
        assert cache.boxscore_gamecodes(2025) == {1, 2, 3, 4, 5}
        assert source.calls[:2] == [("results", 2025), ("schedule", 2025)]
        assert source.boxscore_calls() == [1, 2, 3, 4, 5]
        assert cache.load_missing()["2025"]["missing"] == []
        assert runner.stats.requests == 7

    def test_a_complete_season_needs_no_network_at_all(self, tmp_path):
        cache = RawCache(tmp_path)
        run_season(FakeSource(FIVE_PLAYED, ALL_PLAYED), cache)
        second = FakeSource(FIVE_PLAYED, ALL_PLAYED)
        report, runner, _ = run_season(second, cache)
        assert second.calls == []
        assert runner.stats.requests == 0
        assert (report.expected, report.fetched, report.new) == (5, 5, 0)

    def test_resumes_from_the_cache_after_an_interruption(self, tmp_path):
        cache = RawCache(tmp_path)
        cache.save_results(2025, FIVE_PLAYED)
        cache.save_schedule(2025, ALL_PLAYED)
        cache.append_boxscores(2025, [box(2025, 1, 1), box(2025, 2, 2)])
        source = FakeSource(FIVE_PLAYED, ALL_PLAYED)
        report, _, _ = run_season(source, cache)
        assert source.boxscore_calls() == [3, 4, 5]
        assert ("results", 2025) not in source.calls  # το metadata υπάρχει και η σεζόν τελείωσε
        assert (report.fetched, report.new, report.missing) == (5, 3, [])

    def test_season_in_progress_refreshes_the_metadata_and_fetches_only_new_games(self, tmp_path):
        cache = RawCache(tmp_path)
        early = schedule_frame(g1="true", g2="true", g3="false", g4="false", g5="false", g6="false")
        run_season(FakeSource(results_frame((1, 80, 70), (2, 70, 75)), early), cache)
        assert cache.boxscore_gamecodes(2025) == {1, 2}
        grown = FakeSource(FIVE_PLAYED, IN_PROGRESS)
        report, _, _ = run_season(grown, cache)
        assert ("results", 2025) in grown.calls and ("schedule", 2025) in grown.calls
        assert grown.boxscore_calls() == [3, 4, 5]
        assert (report.expected, report.fetched, report.new) == (5, 5, 3)

    def test_refresh_metadata_can_be_forced(self, tmp_path):
        cache = RawCache(tmp_path)
        run_season(FakeSource(FIVE_PLAYED, ALL_PLAYED), cache)
        source = FakeSource(FIVE_PLAYED, ALL_PLAYED)
        run_season(source, cache, refresh_metadata=True)
        assert source.calls == [("results", 2025), ("schedule", 2025)]

    def test_games_that_stay_unavailable_are_reported_and_not_retried_forever(self, tmp_path):
        cache = RawCache(tmp_path)
        behaviour = {3: Script(empty_body_error())}  # πάντα κενό σώμα HTTP 200
        source = FakeSource(FIVE_PLAYED, ALL_PLAYED, behaviour)
        report, _, _ = run_season(source, cache)
        assert report.missing == [3]
        assert "empty or invalid response body" in report.reasons[3]
        assert (report.expected, report.fetched, report.new) == (5, 4, 4)
        assert behaviour[3].calls == 6  # 3 προσπάθειες στο κύριο πέρασμα και 3 στο τελικό
        assert cache.boxscore_gamecodes(2025) == {1, 2, 4, 5}
        saved = cache.load_missing()["2025"]
        assert saved["missing"] == [3] and saved["expected"] == 5 and saved["fetched"] == 4
        assert "3" in saved["reasons"]

    def test_failed_games_are_retried_at_the_end_of_the_season(self, tmp_path):
        cache = RawCache(tmp_path)
        # Αποτυγχάνει και στις τρεις προσπάθειες του κύριου περάσματος και πετυχαίνει στο τελικό.
        behaviour = {2: Script(*[empty_body_error()] * 3, None)}
        source = FakeSource(FIVE_PLAYED, ALL_PLAYED, behaviour)
        report, _, _ = run_season(source, cache)
        assert report.missing == []
        assert report.fetched == 5
        assert source.boxscore_calls() == [1, 2, 2, 2, 3, 4, 5, 2]

    def test_without_retry_passes_the_game_stays_missing(self, tmp_path):
        behaviour = {2: Script(*[empty_body_error()] * 3, None)}
        report, _, _ = run_season(
            FakeSource(FIVE_PLAYED, ALL_PLAYED, behaviour), RawCache(tmp_path), retry_passes=0
        )
        assert report.missing == [2]

    def test_transient_server_errors_are_recovered_without_a_second_pass(self, tmp_path):
        behaviour = {4: Script(http_error(503), None)}
        source = FakeSource(FIVE_PLAYED, ALL_PLAYED, behaviour)
        report, runner, _ = run_season(source, RawCache(tmp_path))
        assert report.missing == [] and runner.stats.server_errors == 1
        assert source.boxscore_calls().count(4) == 2

    def test_persistent_rate_limiting_aborts_but_keeps_the_games_already_fetched(self, tmp_path):
        cache = RawCache(tmp_path)
        behaviour = {3: Script(http_error(429, 1))}
        source = FakeSource(FIVE_PLAYED, ALL_PLAYED, behaviour)
        clock = FakeClock()
        runner = RequestRunner(
            RateLimiter(1.0, clock=clock.monotonic, sleep=clock.sleep),
            RetryPolicy(max_rate_limit_retries=2),
            sleep=clock.sleep,
        )
        with pytest.raises(RateLimitedError):
            fetch_season(2025, source, runner, cache, checkpoint_every=50, sleep=clock.sleep)
        assert cache.boxscore_gamecodes(2025) == {1, 2}  # αποθηκεύτηκαν πριν τον τερματισμό
        assert runner.limiter.rps < 0.5  # ο ρυθμός μειώθηκε σε κάθε 429

    def test_an_interruption_flushes_the_buffer_to_the_cache(self, tmp_path):
        cache = RawCache(tmp_path)

        class Interrupt(KeyboardInterrupt):
            pass

        behaviour = {4: Script(Interrupt())}
        source = FakeSource(FIVE_PLAYED, ALL_PLAYED, behaviour)
        with pytest.raises(Interrupt):
            run_season(source, cache, checkpoint_every=100)
        assert cache.boxscore_gamecodes(2025) == {1, 2, 3}

    def test_checkpoints_are_written_every_n_games(self, tmp_path, monkeypatch):
        cache = RawCache(tmp_path)
        sizes = []
        original = RawCache.append_boxscores

        def spy(self, season, frames):
            sizes.append(len(frames))
            return original(self, season, frames)

        monkeypatch.setattr(RawCache, "append_boxscores", spy)
        run_season(FakeSource(FIVE_PLAYED, ALL_PLAYED), cache, checkpoint_every=2)
        assert sizes == [2, 2, 1]

    def test_empty_tables_count_as_missing(self, tmp_path):
        behaviour = {5: Script(pd.DataFrame())}
        report, _, _ = run_season(
            FakeSource(FIVE_PLAYED, ALL_PLAYED, behaviour), RawCache(tmp_path)
        )
        assert report.missing == [5]
        assert report.reasons[5] == "empty boxscore table"

    def test_failure_of_the_results_does_not_stop_the_season(self, tmp_path):
        failing = http_error(500)  # τελικά FetchError μετά από 5 προσπάθειες
        source = FakeSource(failing, ALL_PLAYED)
        report, _, _ = run_season(source, RawCache(tmp_path))
        assert (report.expected, report.fetched, report.missing) == (5, 5, [])

    def test_failure_of_the_schedule_does_not_stop_the_season(self, tmp_path):
        source = FakeSource(FIVE_PLAYED, http_error(500))
        report, _, _ = run_season(source, RawCache(tmp_path))
        assert (report.expected, report.fetched) == (5, 5)

    @pytest.mark.parametrize("failing", ["results", "schedule"])
    def test_rate_limiting_while_refreshing_the_metadata_stops_the_run(self, tmp_path, failing):
        source = FakeSource(FIVE_PLAYED, ALL_PLAYED)
        setattr(source, f"_{failing}", http_error(429, 1))
        clock = FakeClock()
        runner = RequestRunner(
            RateLimiter(1.0, clock=clock.monotonic, sleep=clock.sleep),
            RetryPolicy(max_rate_limit_retries=1),
            sleep=clock.sleep,
        )
        with pytest.raises(RateLimitedError):
            fetch_season(2025, source, runner, RawCache(tmp_path), sleep=clock.sleep)
        assert not any(call[0] == "boxscore" for call in source.calls)

    def test_season_fails_when_neither_results_nor_schedule_are_available(self, tmp_path):
        source = FakeSource(http_error(500), http_error(500))
        with pytest.raises(FetchError, match="neither results nor schedule"):
            run_season(source, RawCache(tmp_path))

    def test_progress_is_logged(self, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="elfantasy.ingest.fetch")
        run_season(FakeSource(FIVE_PLAYED, ALL_PLAYED), RawCache(tmp_path), progress_every=2)
        messages = [record.getMessage() for record in caplog.records]
        assert any("2/5 games done" in message for message in messages)
        assert any("complete, 5 of 5 games" in message for message in messages)

    def test_missing_games_are_logged_as_errors(self, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="elfantasy.ingest.fetch")
        behaviour = {3: Script(empty_body_error())}
        run_season(FakeSource(FIVE_PLAYED, ALL_PLAYED, behaviour), RawCache(tmp_path))
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors and "still missing: [3]" in errors[0].getMessage()


class TestFetchSeasons:
    def test_seasons_are_fetched_in_order_and_failures_do_not_stop_the_rest(self, tmp_path):
        cache = RawCache(tmp_path)
        clock = FakeClock()
        runner = make_runner(clock)

        class TwoSeasons(FakeSource):
            def results(self, season):
                if season == 2024:
                    raise http_error(500)
                return super().results(season)

            def schedule(self, season):
                if season == 2024:
                    raise http_error(500)
                return super().schedule(season)

        source = TwoSeasons(FIVE_PLAYED, ALL_PLAYED)
        reports = fetch_seasons(
            [2024, 2025], source, runner, cache, retry_pause=0, sleep=clock.sleep
        )
        assert [r.season for r in reports] == [2024, 2025]
        assert reports[0].error and "neither results nor schedule" in reports[0].error
        assert reports[1].error is None and reports[1].fetched == 5

    def test_refresh_seasons_force_the_metadata_download(self, tmp_path):
        cache = RawCache(tmp_path)
        clock = FakeClock()
        runner = make_runner(clock)
        source = FakeSource(FIVE_PLAYED, ALL_PLAYED)
        fetch_seasons([2025], source, runner, cache, retry_pause=0, sleep=clock.sleep)
        source.calls.clear()
        fetch_seasons(
            [2025], source, runner, cache, refresh_seasons=[2025], retry_pause=0, sleep=clock.sleep
        )
        assert source.calls == [("results", 2025), ("schedule", 2025)]

    def test_rate_limit_abort_propagates(self, tmp_path):
        clock = FakeClock()
        runner = RequestRunner(
            RateLimiter(1.0, clock=clock.monotonic, sleep=clock.sleep),
            RetryPolicy(max_rate_limit_retries=1),
            sleep=clock.sleep,
        )
        source = FakeSource(FIVE_PLAYED, ALL_PLAYED, {1: Script(http_error(429, 1))})
        with pytest.raises(RateLimitedError):
            fetch_seasons([2025, 2026], source, runner, RawCache(tmp_path), sleep=clock.sleep)
        assert ("boxscore", 2026, 1) not in source.calls  # η εκτέλεση σταμάτησε

    def test_final_summary_is_logged(self, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="elfantasy.ingest.fetch")
        clock = FakeClock()
        runner = make_runner(clock)
        source = FakeSource(FIVE_PLAYED, ALL_PLAYED, {2: Script(http_error(429, 1), None)})
        fetch_seasons([2025], source, runner, RawCache(tmp_path), retry_pause=0, sleep=clock.sleep)
        summary = [
            r.getMessage() for r in caplog.records if r.getMessage().startswith("Fetch finished")
        ]
        assert summary and "1 HTTP 429" in summary[0]
