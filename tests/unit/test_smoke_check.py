"""Tests του `scripts/smoke_check.py`: ο έλεγχος μετά το deploy στο Render.

Η λογική (backoff, λήξη χρόνου, έλεγχος commit) δοκιμάζεται με ψεύτικο ρολόι και δίκτυο· η γραμμή
εντολών δοκιμάζεται και πάνω σε πραγματικό HTTP server στο loopback.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import smoke_check as smoke

SHA = "9a94d7d0e5b6c1f2a3b4c5d6e7f8091a2b3c4d5e"
OK_BODY = {
    "status": "ok",
    "model": {"version": "20261003T140043Z-7ae47948"},
    "database": {"ok": True},
    "data_loaded_through": "2026-10-02",
    "problems": [],
}


@pytest.fixture(autouse=True)
def no_real_step_summary(monkeypatch):
    """Στο CI η μεταβλητή GITHUB_STEP_SUMMARY υπάρχει: τα tests δεν πρέπει να γράφουν στην
    πραγματική περίληψη του job."""
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)


def ok_response(**extra) -> smoke.Response:
    return smoke.Response(200, json.dumps({**OK_BODY, **extra}))


class FakeTime:
    """Ψεύτικο ρολόι: το `sleep` προχωρά την ώρα χωρίς πραγματική αναμονή."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def run_with(responses, timeout=480.0, expect_commit=None, **kwargs):
    """Τρέχει το `run` με σενάριο απαντήσεων (η τελευταία επαναλαμβάνεται)."""
    fake = FakeTime()
    sequence = list(responses)
    seen: list[tuple[str, float]] = []
    logs: list[str] = []

    def fetcher(url, request_timeout):
        seen.append((url, request_timeout))
        return sequence.pop(0) if len(sequence) > 1 else sequence[0]

    result = smoke.run(
        "https://example.test",
        timeout,
        expect_commit,
        fetcher=fetcher,
        clock=fake.clock,
        sleeper=fake.sleep,
        log=logs.append,
        **kwargs,
    )
    return result, fake, seen, logs


# --------------------------------------------------------------------------------------
# Λογική
# --------------------------------------------------------------------------------------


class TestPolling:
    def test_a_healthy_service_passes_on_the_first_attempt(self):
        (ok, verdict, _response, attempts), fake, seen, _ = run_with([ok_response()])
        assert ok and verdict.ok and attempts == 1
        assert fake.sleeps == []
        assert seen == [("https://example.test/health", 60.0)]

    def test_it_retries_through_cold_start_errors_until_the_service_is_ok(self):
        refused = smoke.Response(None, "", "URLError: [Errno 111] Connection refused")
        warming = smoke.Response(503, '{"detail": "starting"}')
        degraded = smoke.Response(
            503, json.dumps({"status": "degraded", "problems": ["model: not loaded"]})
        )
        (ok, _verdict, _response, attempts), fake, _seen, logs = run_with(
            [refused, warming, degraded, ok_response()]
        )
        assert ok and attempts == 4
        assert fake.sleeps == [5.0, 7.5, 11.25]  # backoff ×1,5
        assert logs[0].startswith("attempt 1 after 0s: no response (")
        assert "HTTP 503" in logs[1]

    def test_the_wait_grows_up_to_a_ceiling(self):
        result, fake, _seen, _logs = run_with([smoke.Response(503, "")], timeout=400.0)
        assert not result[0]
        assert fake.sleeps[:6] == [5.0, 7.5, 11.25, 16.875, 25.3125, 30.0]
        assert set(fake.sleeps[5:-1]) == {30.0}

    def test_on_timeout_it_fails_and_returns_the_last_response(self):
        down = smoke.Response(503, '{"detail": "database unavailable"}')
        (ok, verdict, response, attempts), fake, _, _ = run_with([down], timeout=100.0)
        assert not ok and not verdict.ok
        assert verdict.reason == "HTTP 503"
        assert response is down
        assert attempts > 3
        assert sum(fake.sleeps) <= 100.0  # δεν ξεπερνά τον συνολικό χρόνο

    def test_the_request_timeout_never_exceeds_the_time_left(self):
        _, _fake, seen, _ = run_with([smoke.Response(503, "")], timeout=100.0)
        assert all(1.0 <= value <= 60.0 for _, value in seen)
        assert seen[-1][1] < 60.0

    def test_status_must_be_ok_even_with_http_200(self):
        body = json.dumps({"status": "degraded", "problems": ["database: down"]})
        (ok, verdict, _, _), _, _, _ = run_with([smoke.Response(200, body)], timeout=10.0)
        assert not ok
        assert "status is 'degraded'" in verdict.reason and "database: down" in verdict.reason

    @pytest.mark.parametrize(
        ("response", "reason"),
        [
            (smoke.Response(200, "<html>proxy</html>"), "not JSON"),
            (smoke.Response(200, "[1, 2]"), "not an object"),
            (smoke.Response(404, "{}"), "HTTP 404"),
            (smoke.Response(200, '{"status": null}'), "status is None"),
        ],
    )
    def test_malformed_or_unexpected_answers_do_not_pass(self, response, reason):
        (ok, verdict, _, _), _, _, _ = run_with([response], timeout=10.0)
        assert not ok and reason in verdict.reason


class TestCommit:
    def test_the_old_version_does_not_pass_while_the_new_one_is_building(self):
        old = ok_response(commit="1" * 40)
        new = ok_response(commit=SHA)
        (ok, _, _, attempts), fake, _, logs = run_with([old, old, new], expect_commit=SHA)
        assert ok and attempts == 3 and len(fake.sleeps) == 2
        assert "runs commit 111111111111 (waiting for 9a94d7d0e5b6)" in logs[0]

    def test_the_old_version_forever_is_a_failure(self):
        old = ok_response(commit="1" * 40)
        (ok, verdict, response, _), _, _, _ = run_with([old], timeout=60.0, expect_commit=SHA)
        assert not ok and "runs commit" in verdict.reason
        assert response is old

    @pytest.mark.parametrize(
        ("live", "expected", "same"),
        [
            (SHA, SHA, True),
            (SHA.upper(), SHA, True),
            (SHA[:7], SHA, True),  # συντομευμένο SHA στην υπηρεσία
            (SHA, SHA[:10], True),  # συντομευμένο αναμενόμενο
            (" " + SHA + "\n", SHA, True),
            ("1" * 40, SHA, False),
            (SHA[:6], SHA, False),  # πολύ κοντό για να είναι αξιόπιστο
            ("abc", "abc", True),
            ("abc", "abd", False),
        ],
    )
    def test_commit_comparison(self, live, expected, same):
        assert smoke.commits_match(live, expected) is same

    @pytest.mark.parametrize("missing", [None, "", "   ", 5])
    def test_a_service_that_does_not_report_its_commit_passes_with_a_notice(self, missing):
        (ok, verdict, _, _), _, _, logs = run_with([ok_response(commit=missing)], expect_commit=SHA)
        assert ok and verdict.note and "cannot confirm" in verdict.note
        assert any(line.startswith("notice: ") for line in logs)

    def test_without_an_expected_commit_the_field_is_ignored(self):
        (ok, verdict, _, _), _, _, logs = run_with([ok_response(commit="1" * 40)])
        assert ok and verdict.note is None
        assert not any(line.startswith("notice") for line in logs)


class TestHelpers:
    @pytest.mark.parametrize(
        ("url", "normalized"),
        [
            ("https://svc.onrender.com", "https://svc.onrender.com"),
            ("https://svc.onrender.com/", "https://svc.onrender.com"),
            ("  http://127.0.0.1:8000/  ", "http://127.0.0.1:8000"),
        ],
    )
    def test_normalize_url(self, url, normalized):
        assert smoke.normalize_url(url) == normalized

    @pytest.mark.parametrize(
        "url", ["", "svc.onrender.com", "ftp://svc.test", "file:///etc/passwd", "https://"]
    )
    def test_urls_that_are_not_http_are_rejected(self, url):
        with pytest.raises(ValueError, match="http"):
            smoke.normalize_url(url)

    def test_urls_with_credentials_are_rejected_without_echoing_them(self):
        with pytest.raises(ValueError) as caught:
            smoke.normalize_url("https://user:hunter2@svc.test")
        assert "hunter2" not in str(caught.value)

    def test_connection_errors_do_not_include_the_url(self):
        free = socket.socket()
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
        free.close()  # κανείς δεν ακούει πια σε αυτή τη θύρα
        response = smoke.fetch(f"http://127.0.0.1:{port}/health?key=SECRET", timeout=2.0)
        assert response.status is None and response.error
        assert "SECRET" not in response.error and str(port) not in response.error


# --------------------------------------------------------------------------------------
# Γραμμή εντολών πάνω σε πραγματικό HTTP server
# --------------------------------------------------------------------------------------


class Scripted:
    """Server στο loopback που απαντά με σενάριο (status, σώμα)· η τελευταία επαναλαμβάνεται."""

    def __init__(self, script):
        self.script = list(script)
        self.requests: list[str] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                owner.requests.append(self.path)
                status, body = owner.script.pop(0) if len(owner.script) > 1 else owner.script[0]
                payload = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


FAST = ["--initial-delay", "0.02", "--max-delay", "0.05", "--request-timeout", "2"]


class TestCommandLine:
    def test_passes_against_a_healthy_service_and_writes_the_summary(self, tmp_path, capsys):
        summary = tmp_path / "summary.md"
        with Scripted([(200, json.dumps({**OK_BODY, "commit": SHA}))]) as server:
            code = smoke.main(
                ["--url", server.url, "--expect-commit", SHA, "--summary-file", str(summary), *FAST]
            )
        assert code == 0
        assert "smoke check passed" in capsys.readouterr().out
        text = summary.read_text(encoding="utf-8")
        assert "passed" in text and "20261003T140043Z-7ae47948" in text and SHA in text
        assert server.requests == ["/health"]

    def test_waits_through_a_503_and_then_passes(self, tmp_path):
        with Scripted([(503, '{"detail": "starting"}'), (200, json.dumps(OK_BODY))]) as server:
            code = smoke.main(["--url", server.url, "--timeout-seconds", "10", *FAST])
        assert code == 0 and len(server.requests) == 2

    def test_fails_with_the_last_response_when_the_service_never_gets_healthy(
        self, tmp_path, capsys
    ):
        summary = tmp_path / "summary.md"
        body = json.dumps({"status": "degraded", "problems": ["database: down"]})
        with Scripted([(503, body)]) as server:
            code = smoke.main(
                [
                    "--url",
                    server.url,
                    "--timeout-seconds",
                    "0.5",
                    "--summary-file",
                    str(summary),
                    *FAST,
                ]
            )
        assert code == 1
        error = capsys.readouterr().err
        assert "smoke check FAILED" in error and "HTTP 503" in error and "database: down" in error
        assert "Traceback" not in error
        text = summary.read_text(encoding="utf-8")
        assert "FAILED" in text and "database: down" in text

    def test_fails_when_nothing_listens(self, tmp_path, capsys):
        free = socket.socket()
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
        free.close()
        code = smoke.main(["--url", f"http://127.0.0.1:{port}", "--timeout-seconds", "0.3", *FAST])
        assert code == 1
        assert "no response" in capsys.readouterr().err

    def test_a_long_body_is_cut(self, capsys):
        with Scripted([(502, "x" * 10_000)]) as server:
            code = smoke.main(["--url", server.url, "--timeout-seconds", "0.3", *FAST])
        assert code == 1
        error = capsys.readouterr().err
        assert error.count("x") == smoke.BODY_LIMIT

    @pytest.mark.parametrize("url", ["", "ftp://x.test", "https://user:pw@x.test", "x.test"])
    def test_a_bad_url_is_exit_code_2(self, url, capsys):
        assert smoke.main(["--url", url]) == 2
        assert capsys.readouterr().err.startswith("error: ")

    def test_non_positive_timeouts_are_exit_code_2(self, capsys):
        assert smoke.main(["--url", "http://127.0.0.1:1", "--timeout-seconds", "0"]) == 2
        assert "positive" in capsys.readouterr().err

    def test_the_summary_is_appended_not_overwritten(self, tmp_path):
        summary = tmp_path / "summary.md"
        summary.write_text("previous step\n", encoding="utf-8")
        with Scripted([(200, json.dumps(OK_BODY))]) as server:
            smoke.main(["--url", server.url, "--summary-file", str(summary), *FAST])
        text = summary.read_text(encoding="utf-8")
        assert text.startswith("previous step\n") and "smoke check: passed" in text

    def test_the_summary_file_defaults_to_github_step_summary(self, tmp_path, monkeypatch):
        target = tmp_path / "gh-summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))
        with Scripted([(200, json.dumps(OK_BODY))]) as server:
            assert smoke.main(["--url", server.url, *FAST]) == 0
        assert "passed" in target.read_text(encoding="utf-8")
