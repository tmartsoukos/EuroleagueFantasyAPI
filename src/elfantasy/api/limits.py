"""Όριο μεγέθους σώματος αιτήματος (pure ASGI middleware).

Το FastAPI διαβάζει και αναλύει ολόκληρο το σώμα ενός αιτήματος ΠΡΙΝ εκτελέσει τις εξαρτήσεις
του endpoint, άρα και πριν από τον έλεγχο του `X-API-Key`. Χωρίς όριο, ένα ανώνυμο `POST` με σώμα
δεκάδων MB (το review της Φάσης 7 μέτρησε 60 MB → +163 MB μνήμης, που δεν επιστρέφεται) θα
μπορούσε να εξαντλήσει τη μνήμη του free plan του Render (512 MB). Το middleware απορρίπτει το
αίτημα με HTTP 413 χωρίς να το διαβάσει, ανεξάρτητα από την αυθεντικοποίηση:

* αν η επικεφαλίδα `Content-Length` ξεπερνά το όριο, η απόρριψη γίνεται αμέσως, πριν διαβαστεί
  οποιοδήποτε byte του σώματος·
* αν το σώμα έρχεται χωρίς `Content-Length` (chunked), μετριέται καθώς διαβάζεται και το αίτημα
  απορρίπτεται μόλις περάσει το όριο.

Όλα τα σώματα που δέχεται το API είναι μικρά (το μεγαλύτερο, το `POST /availability`, δεν ξεπερνά
τα λίγα KB), γι' αυτό το όριο είναι γενναιόδωρο.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

# 64 KiB: πολύ πάνω από κάθε νόμιμο σώμα του API, πολύ κάτω από οτιδήποτε απειλεί τη μνήμη.
MAX_REQUEST_BODY_BYTES = 64 * 1024

_TOO_LARGE = json.dumps({"detail": "request body too large"}).encode("utf-8")
_BAD_LENGTH = json.dumps({"detail": "invalid Content-Length header"}).encode("utf-8")


class BodySizeLimitMiddleware:
    """Απορρίπτει με HTTP 413 αιτήματα με σώμα μεγαλύτερο από `max_bytes`."""

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_REQUEST_BODY_BYTES) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _declared_length(scope)
        if declared is not None and declared < 0:
            await _respond(send, 400, _BAD_LENGTH)
            return
        if declared is not None and declared > self.max_bytes:
            await _respond(send, 413, _TOO_LARGE)
            return

        received = 0
        started = False  # η εφαρμογή έχει ήδη ξεκινήσει να στέλνει απάντηση
        rejected = False  # εμείς στείλαμε το 413: ό,τι στείλει η εφαρμογή μετά αγνοείται

        async def guarded_receive() -> Message:
            nonlocal received, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes and not started:
                    rejected = True
                    await _respond(send, 413, _TOO_LARGE)
                    # Η εφαρμογή βλέπει «αποσύνδεση» και σταματά να διαβάζει το σώμα.
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal started
            if rejected:
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        await self.app(scope, guarded_receive, guarded_send)


def _declared_length(scope: Scope) -> int | None:
    """Η τιμή του `Content-Length`: None αν λείπει, -1 αν δεν είναι μη αρνητικός ακέραιος."""
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            text = value.decode("latin-1").strip()
            if text.isascii() and text.isdigit():
                return int(text)
            return -1
    return None


async def _respond(send: Send, status: int, body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                # Το υπόλοιπο σώμα δεν διαβάζεται: η σύνδεση κλείνει μετά την απάντηση.
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
