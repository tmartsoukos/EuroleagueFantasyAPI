#!/usr/bin/env python3
"""Smoke check μετά το deploy: περιμένει το `GET /health` της υπηρεσίας να απαντήσει υγιές.

Το job `deploy` του CI το καλεί αφού το Deploy Hook του Render δεχτεί το αίτημα. Το script
ρωτά το `<url>/health` με αυξανόμενη καθυστέρηση (backoff) μέχρι να ισχύουν ΟΛΑ τα παρακάτω:

* HTTP 200 και σώμα JSON,
* `"status": "ok"` (η βάση απαντά και το μοντέλο φορτώθηκε),
* αν δόθηκε `--expect-commit` και η υπηρεσία δηλώνει το commit της (`commit` στο JSON): ταυτίζεται
  με το αναμενόμενο. Έτσι δεν περνά ο έλεγχος από τη ΠΑΛΙΑ έκδοση που εξυπηρετεί ακόμη το Render
  όσο χτίζεται η νέα (ζωντανές αναβαθμίσεις χωρίς διακοπή). Αν η υπηρεσία δεν δηλώνει commit
  (`null` ή απουσία), ο έλεγχος αυτός παραλείπεται με σχετικό μήνυμα.

Στο free tier του Render η υπηρεσία «κοιμάται» και η εκκίνηση παίρνει περίπου ένα λεπτό, γι' αυτό
τα αιτήματα έχουν μεγάλο timeout και η συνολική αναμονή είναι λεπτά.

Αν λήξει ο χρόνος, το script τελειώνει με κωδικό 1 και τυπώνει την τελευταία απάντηση (κωδικός
και σώμα, κομμένο σε 2000 χαρακτήρες)· δεν περιέχει μυστικά, γιατί το `/health` δεν έχει. Το URL
του Deploy Hook δεν περνά ποτέ από εδώ. Η περίληψη γράφεται και στο `$GITHUB_STEP_SUMMARY`, αν
υπάρχει.

Κωδικοί εξόδου: 0 υγιής υπηρεσία, 1 λήξη χρόνου, 2 άκυρα ορίσματα.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

BODY_LIMIT = 2000
USER_AGENT = "elfantasy-smoke-check"


@dataclass(frozen=True)
class Response:
    """Το αποτέλεσμα ενός αιτήματος: κωδικός και σώμα, ή περιγραφή σφάλματος σύνδεσης."""

    status: int | None
    body: str
    error: str | None = None


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str
    health: dict | None = None
    note: str | None = None


def normalize_url(url: str) -> str:
    """Το βασικό URL χωρίς τελικό `/`. Μόνο http(s), χωρίς διαπιστευτήρια μέσα στο URL."""
    parsed = urllib.parse.urlsplit(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("the service URL must start with http:// or https://")
    if parsed.username or parsed.password:
        raise ValueError("the service URL must not contain credentials")
    return url.strip().rstrip("/")


def fetch(url: str, timeout: float) -> Response:
    """GET του url. Τα σφάλματα σύνδεσης γίνονται `Response` (χωρίς το URL στο μήνυμα)."""
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return Response(response.status, response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as error:
        return Response(error.code, error.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
        reason = getattr(error, "reason", error)
        return Response(None, "", f"{type(error).__name__}: {reason}")


def commits_match(live: str, expected: str) -> bool:
    """Ίδιο commit (πεζά/κεφαλαία αδιάφορα, δέχεται και συντομευμένο SHA τουλάχιστον 7 ψηφίων)."""
    live, expected = live.strip().lower(), expected.strip().lower()
    if min(len(live), len(expected)) < 7:
        return live == expected
    return live.startswith(expected) or expected.startswith(live)


def evaluate(response: Response, expect_commit: str | None) -> Verdict:
    if response.error is not None:
        return Verdict(False, f"no response ({response.error})")
    if response.status != 200:
        return Verdict(False, f"HTTP {response.status}")
    try:
        health = json.loads(response.body)
    except ValueError:
        return Verdict(False, "HTTP 200 but the body is not JSON")
    if not isinstance(health, dict):
        return Verdict(False, "HTTP 200 but the JSON is not an object")
    if health.get("status") != "ok":
        problems = health.get("problems")
        detail = f", problems: {problems}" if problems else ""
        return Verdict(False, f"status is {health.get('status')!r}{detail}", health)
    live = health.get("commit")
    if expect_commit:
        if isinstance(live, str) and live.strip():
            if not commits_match(live, expect_commit):
                return Verdict(
                    False,
                    f"healthy, but it runs commit {live[:12]} (waiting for {expect_commit[:12]})",
                    health,
                )
        else:
            return Verdict(
                True,
                "ok",
                health,
                "the service does not report its commit: cannot confirm that the new version "
                "is the one answering",
            )
    return Verdict(True, "ok", health)


def run(
    url: str,
    timeout_seconds: float,
    expect_commit: str | None = None,
    request_timeout: float = 60.0,
    initial_delay: float = 5.0,
    max_delay: float = 30.0,
    *,
    fetcher: Callable[[str, float], Response] = fetch,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> tuple[bool, Verdict, Response, int]:
    """Επαναλαμβάνει τον έλεγχο μέχρι επιτυχία ή λήξη χρόνου. Επιστρέφει (ok, verdict, τελευταία
    απάντηση, πλήθος προσπαθειών)."""
    health_url = f"{normalize_url(url)}/health"
    started = clock()
    delay = initial_delay
    attempt = 0
    while True:
        attempt += 1
        remaining = timeout_seconds - (clock() - started)
        response = fetcher(health_url, max(1.0, min(request_timeout, remaining)))
        verdict = evaluate(response, expect_commit)
        elapsed = clock() - started
        log(f"attempt {attempt} after {elapsed:.0f}s: {verdict.reason}")
        if verdict.ok:
            if verdict.note:
                log(f"notice: {verdict.note}")
            return True, verdict, response, attempt
        remaining = timeout_seconds - (clock() - started)
        if remaining <= 0:
            return False, verdict, response, attempt
        sleeper(min(delay, remaining))
        delay = min(delay * 1.5, max_delay)


def summary_markdown(
    ok: bool, verdict: Verdict, response: Response, attempts: int, url: str
) -> str:
    if ok:
        health = verdict.health or {}
        model = health.get("model") if isinstance(health.get("model"), dict) else {}
        lines = [
            "### Post-deploy smoke check: passed",
            "",
            f"- service: {url}",
            f"- attempts: {attempts}",
            f"- model version: {model.get('version')}",
            f"- data loaded through: {health.get('data_loaded_through')}",
            f"- commit reported by the service: {health.get('commit')}",
        ]
        if verdict.note:
            lines.append(f"- notice: {verdict.note}")
        return "\n".join(lines) + "\n"
    shown = response.body[:BODY_LIMIT] if response.body else (response.error or "(empty)")
    return (
        "### Post-deploy smoke check: FAILED\n\n"
        f"- service: {url}\n- attempts: {attempts}\n- last result: {verdict.reason}\n\n"
        f"Last response (HTTP {response.status}):\n\n```\n{shown}\n```\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--url", required=True, help="βασικό URL της υπηρεσίας (https://….onrender.com)"
    )
    parser.add_argument("--timeout-seconds", type=float, default=480.0, help="συνολική αναμονή")
    parser.add_argument("--expect-commit", default="", help="SHA που πρέπει να τρέχει η υπηρεσία")
    parser.add_argument("--request-timeout", type=float, default=60.0, help="timeout ανά αίτημα")
    parser.add_argument("--initial-delay", type=float, default=5.0, help="πρώτη αναμονή (s)")
    parser.add_argument("--max-delay", type=float, default=30.0, help="μέγιστη αναμονή (s)")
    parser.add_argument(
        "--summary-file",
        default=os.environ.get("GITHUB_STEP_SUMMARY", ""),
        help="αρχείο όπου προστίθεται η περίληψη (προεπιλογή: $GITHUB_STEP_SUMMARY)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        url = normalize_url(args.url)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.timeout_seconds <= 0 or args.request_timeout <= 0:
        print("error: the timeouts must be positive", file=sys.stderr)
        return 2
    ok, verdict, response, attempts = run(
        url,
        args.timeout_seconds,
        args.expect_commit.strip() or None,
        args.request_timeout,
        args.initial_delay,
        args.max_delay,
    )
    text = summary_markdown(ok, verdict, response, attempts, url)
    if args.summary_file:
        with open(args.summary_file, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    if ok:
        print("smoke check passed")
        return 0
    shown = response.body[:BODY_LIMIT] if response.body else (response.error or "(empty)")
    print(
        f"smoke check FAILED after {attempts} attempts: {verdict.reason}\n"
        f"last response: HTTP {response.status}\n{shown}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
