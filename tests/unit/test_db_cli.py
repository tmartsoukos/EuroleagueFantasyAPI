"""Tests του db/cli.py: logging κονσόλας των εργαλείων, κρυμμένα μυστικά και έξοδος UTF-8."""

import logging

from elfantasy.db import cli

SECRET = "S3cr3t-Pa55"
URL = f"postgresql://postgres:{SECRET}@db.example.com:5432/postgres"


def test_console_logging_writes_to_the_current_stdout_and_cleans_up(capsys):
    package_logger = logging.getLogger("elfantasy")
    before = (list(package_logger.handlers), package_logger.level)
    with cli.console_logging("INFO"):
        logging.getLogger("elfantasy.db.example").info("hello from the tool")
        logging.getLogger("elfantasy.db.example").debug("hidden debug")
    out = capsys.readouterr().out
    assert "hello from the tool" in out and "hidden debug" not in out
    assert "INFO" in out
    assert (list(package_logger.handlers), package_logger.level) == before  # τίποτα δεν διέρρευσε


def test_console_logging_follows_a_replaced_stdout(capsys):
    # το handler δεν κρατά το stdout που ίσχυε όταν δημιουργήθηκε: γράφει στο τρέχον
    with cli.console_logging("INFO"):
        logging.getLogger("elfantasy.x").info("first")
        capsys.readouterr()
        logging.getLogger("elfantasy.x").info("second")
        assert "second" in capsys.readouterr().out


def test_console_logging_accepts_a_numeric_level(capsys):
    with cli.console_logging(logging.DEBUG):
        logging.getLogger("elfantasy.x").debug("verbose")
    assert "verbose" in capsys.readouterr().out


def test_report_failure_hides_the_password_in_the_message(caplog):
    error = RuntimeError(f"connection to {URL} failed")
    with caplog.at_level(logging.INFO, logger="elfantasy.db.cli"):
        cli.report_failure("could not connect", error, URL)
    text = caplog.text
    assert "could not connect" in text and SECRET not in text
    assert "postgres:***@db.example.com" in text


def test_report_failure_hides_the_password_in_the_traceback_at_debug_level(caplog):
    try:
        raise RuntimeError(f"bad url {URL}")
    except RuntimeError as error:
        with caplog.at_level(logging.DEBUG, logger="elfantasy.db.cli"):
            cli.report_failure("failed", error, URL)
    assert "traceback" in caplog.text and "RuntimeError" in caplog.text
    assert SECRET not in caplog.text


def test_report_failure_without_a_message_names_the_exception_type(caplog):
    with caplog.at_level(logging.INFO, logger="elfantasy.db.cli"):
        cli.report_failure("failed", ValueError(), None)
    assert "ValueError" in caplog.text


def test_write_text_emits_utf8_with_lf_endings(capsysbinary):
    cli.write_text("a\nΓειά σου\n")
    assert capsysbinary.readouterr().out == "a\nΓειά σου\n".encode()


def test_write_text_falls_back_to_text_streams(monkeypatch):
    class TextOnly:
        def __init__(self):
            self.parts = []

        def flush(self):
            pass

        def write(self, text):
            self.parts.append(text)

    stream = TextOnly()
    monkeypatch.setattr(cli.sys, "stdout", stream)
    cli.write_text("plain")
    assert stream.parts == ["plain"]


def test_use_utf8_output_tolerates_streams_without_reconfigure(monkeypatch):
    class Bare:
        pass

    monkeypatch.setattr(cli.sys, "stdout", Bare())
    monkeypatch.setattr(cli.sys, "stderr", Bare())
    cli.use_utf8_output()  # δεν σηκώνει εξαίρεση
