"""Unit tests του `BodySizeLimitMiddleware` (api/limits.py) πάνω σε καθαρή εφαρμογή ASGI, χωρίς το
FastAPI: έτσι φαίνεται ακριβώς τι διαβάζει και τι στέλνει το middleware.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from elfantasy.api.limits import MAX_REQUEST_BODY_BYTES, BodySizeLimitMiddleware


class RecordingApp:
    """Εφαρμογή ASGI που διαβάζει ολόκληρο το σώμα και απαντά με το μέγεθός του."""

    def __init__(self, respond_before_reading: bool = False) -> None:
        self.calls = 0
        self.bytes_read = 0
        self.respond_before_reading = respond_before_reading

    async def __call__(self, scope, receive, send):
        self.calls += 1
        if scope["type"] != "http":
            return
        if self.respond_before_reading:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"early"})
        body = b""
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        self.bytes_read = len(body)
        if self.respond_before_reading:
            return
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": str(len(body)).encode()})


def drive(app, *, chunks=(), headers=(), scope_type="http"):
    """Εκτελεί ένα αίτημα και επιστρέφει τα μηνύματα που στάλθηκαν προς τον client."""
    sent = []
    queue = list(chunks)

    async def receive():
        if queue:
            body = queue.pop(0)
            return {"type": "http.request", "body": body, "more_body": bool(queue)}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": scope_type,
        "method": "POST",
        "path": "/",
        "headers": [(name.encode("latin-1"), value.encode("utf-8")) for name, value in headers],
    }
    asyncio.run(app(scope, receive, send))
    return sent


def status_of(sent) -> int:
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def body_of(sent) -> bytes:
    return b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")


def headers_of(sent) -> dict[bytes, bytes]:
    start = next(m for m in sent if m["type"] == "http.response.start")
    return dict(start["headers"])


class TestDeclaredLength:
    def test_a_small_body_passes_untouched(self):
        app = RecordingApp()
        sent = drive(
            BodySizeLimitMiddleware(app, max_bytes=100),
            chunks=[b"x" * 40],
            headers=[("content-length", "40")],
        )
        assert status_of(sent) == 200 and body_of(sent) == b"40"
        assert app.calls == 1

    def test_a_body_of_exactly_the_limit_passes(self):
        app = RecordingApp()
        sent = drive(
            BodySizeLimitMiddleware(app, max_bytes=100),
            chunks=[b"x" * 100],
            headers=[("content-length", "100")],
        )
        assert status_of(sent) == 200 and body_of(sent) == b"100"

    def test_one_byte_over_the_limit_is_rejected_without_calling_the_app(self):
        app = RecordingApp()
        sent = drive(
            BodySizeLimitMiddleware(app, max_bytes=100),
            chunks=[b"x" * 101],
            headers=[("content-length", "101")],
        )
        assert status_of(sent) == 413
        assert json.loads(body_of(sent)) == {"detail": "request body too large"}
        assert app.calls == 0  # ούτε routing ούτε έλεγχος κλειδιού ούτε ανάγνωση σώματος

    def test_a_huge_declared_length_is_rejected_before_any_byte_is_read(self):
        app = RecordingApp()
        sent = drive(
            BodySizeLimitMiddleware(app, max_bytes=100),
            chunks=[],  # ο client δεν έχει στείλει τίποτα ακόμη
            headers=[("content-length", str(60 * 1024 * 1024))],
        )
        assert status_of(sent) == 413 and app.calls == 0

    def test_the_rejection_closes_the_connection(self):
        sent = drive(
            BodySizeLimitMiddleware(RecordingApp(), max_bytes=10),
            headers=[("content-length", "11")],
        )
        headers = headers_of(sent)
        assert headers[b"connection"] == b"close"
        assert headers[b"content-type"] == b"application/json"
        assert int(headers[b"content-length"]) == len(body_of(sent))

    @pytest.mark.parametrize("value", ["abc", "-5", "1.5", "1e3", "", " ", "0x10", "١٢٣"])
    def test_an_invalid_content_length_is_a_400(self, value):
        app = RecordingApp()
        sent = drive(
            BodySizeLimitMiddleware(app, max_bytes=100), headers=[("content-length", value)]
        )
        assert status_of(sent) == 400
        assert json.loads(body_of(sent)) == {"detail": "invalid Content-Length header"}
        assert app.calls == 0

    def test_a_request_without_a_body_passes(self):
        app = RecordingApp()
        sent = drive(BodySizeLimitMiddleware(app, max_bytes=100))
        assert status_of(sent) == 200 and app.calls == 1


class TestStreamedBodies:
    def test_a_streamed_body_under_the_limit_passes(self):
        app = RecordingApp()
        sent = drive(BodySizeLimitMiddleware(app, max_bytes=100), chunks=[b"x" * 30] * 3)
        assert status_of(sent) == 200 and body_of(sent) == b"90"

    def test_a_streamed_body_over_the_limit_is_rejected_while_it_is_read(self):
        app = RecordingApp()
        sent = drive(BodySizeLimitMiddleware(app, max_bytes=100), chunks=[b"x" * 30] * 10)
        assert status_of(sent) == 413
        assert json.loads(body_of(sent)) == {"detail": "request body too large"}
        # Η εφαρμογή είδε «αποσύνδεση» πριν διαβάσει όλο το σώμα και η δική της απάντηση αγνοήθηκε.
        assert app.bytes_read <= 100
        assert [m["type"] for m in sent].count("http.response.start") == 1

    def test_the_applications_late_response_is_dropped_after_a_rejection(self):
        class LateResponder(RecordingApp):
            async def __call__(self, scope, receive, send):
                await super().__call__(scope, receive, send)
                await send({"type": "http.response.start", "status": 500, "headers": []})
                await send({"type": "http.response.body", "body": b"late"})

        sent = drive(
            BodySizeLimitMiddleware(LateResponder(), max_bytes=100), chunks=[b"x" * 60] * 4
        )
        assert status_of(sent) == 413
        assert b"late" not in body_of(sent)

    def test_an_application_that_already_answered_is_not_overridden(self):
        app = RecordingApp(respond_before_reading=True)
        sent = drive(BodySizeLimitMiddleware(app, max_bytes=100), chunks=[b"x" * 60] * 4)
        assert status_of(sent) == 200 and body_of(sent) == b"early"


class TestOtherScopes:
    @pytest.mark.parametrize("scope_type", ["lifespan", "websocket"])
    def test_non_http_scopes_pass_through(self, scope_type):
        app = RecordingApp()
        drive(BodySizeLimitMiddleware(app, max_bytes=1), scope_type=scope_type)
        assert app.calls == 1


class TestConfiguration:
    @pytest.mark.parametrize("value", [0, -1])
    def test_the_limit_must_be_positive(self, value):
        with pytest.raises(ValueError, match="positive"):
            BodySizeLimitMiddleware(RecordingApp(), max_bytes=value)

    def test_the_default_limit_is_64_kib(self):
        assert MAX_REQUEST_BODY_BYTES == 64 * 1024
        assert BodySizeLimitMiddleware(RecordingApp()).max_bytes == MAX_REQUEST_BODY_BYTES
