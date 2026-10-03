"""Κοινά βοηθητικά των εργαλείων γραμμής εντολών που μιλούν με τη βάση (migrate, transfer,
record_predictions, evaluate_recorded).

* `console_logging`: logging στην κονσόλα (stdout) μόνο για τη διάρκεια της εντολής. Δεν αλλάζει
  τον root logger, άρα δεν επηρεάζει τα tests ή τη βιβλιοθήκη που καλεί την εντολή.
* `report_failure`: καταγράφει μια αποτυχία με ΚΡΥΜΜΕΝΟΥΣ τους κωδικούς των URL βάσης. Τα μηνύματα
  των drivers και το traceback περνούν πάντα από το `redact_secrets`.
"""

from __future__ import annotations

import logging
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager

from elfantasy.db.urls import redact_secrets

logger = logging.getLogger(__name__)

PACKAGE_LOGGER = "elfantasy"


class _StdoutHandler(logging.StreamHandler):
    """Γράφει πάντα στο τρέχον `sys.stdout` (όχι σε αυτό που ίσχυε όταν δημιουργήθηκε ο handler)."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value) -> None:  # η StreamHandler αναθέτει stream στον constructor
        pass


def use_utf8_output() -> None:
    """Στα ελληνικά Windows η κονσόλα είναι cp1253: το UTF-8 αποφεύγει σφάλματα εκτύπωσης."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def write_text(text: str) -> None:
    """Γράφει το `text` στο stdout ως UTF-8 με αλλαγές γραμμής LF, ανεξάρτητα από την κονσόλα.

    Χρειάζεται όταν η έξοδος ανακατευθύνεται σε αρχείο (π.χ. `--print-sql > 001_init.sql`): στα
    Windows το κείμενο θα έβγαινε αλλιώς με CRLF και στην κωδικοσελίδα της κονσόλας.
    """
    stream = sys.stdout
    stream.flush()
    buffer = getattr(stream, "buffer", None)
    if buffer is None:
        stream.write(text)
        return
    buffer.write(text.encode("utf-8"))
    buffer.flush()


@contextmanager
def console_logging(level: str | int = "INFO") -> Iterator[None]:
    """Logging του πακέτου `elfantasy` στην κονσόλα, μόνο μέσα στο `with`."""
    use_utf8_output()
    package_logger = logging.getLogger(PACKAGE_LOGGER)
    handler = _StdoutHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    previous_level = package_logger.level
    package_logger.addHandler(handler)
    package_logger.setLevel(level)
    try:
        yield
    finally:
        package_logger.removeHandler(handler)
        package_logger.setLevel(previous_level)


def report_failure(message: str, error: BaseException, *urls: str | None) -> None:
    """Καταγράφει το σφάλμα `error` χωρίς κωδικούς: ERROR με το (κρυμμένο) μήνυμα και, αν το
    επίπεδο logging είναι DEBUG, ολόκληρο το traceback (επίσης κρυμμένο)."""
    text = redact_secrets(str(error) or type(error).__name__, *urls)
    logger.error("%s: %s", message, text)
    if logger.isEnabledFor(logging.DEBUG):
        details = "".join(traceback.format_exception(error))
        logger.debug("traceback:\n%s", redact_secrets(details, *urls))
