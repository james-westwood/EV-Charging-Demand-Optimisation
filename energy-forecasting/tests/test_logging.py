import json
import logging
import os

import pytest

from src.logging_config import get_logger


def _capture_log(logger: logging.Logger, level: int, message: str) -> dict:
    """Capture a single log record and return parsed JSON."""
    import io

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)

    # Copy formatter from existing handler
    formatter = logger.handlers[0].formatter if logger.handlers else None
    if formatter:
        handler.setFormatter(formatter)

    logger.addHandler(handler)
    try:
        logger.log(level, message)
    finally:
        logger.removeHandler(handler)

    output = stream.getvalue().strip()
    return json.loads(output)


def test_json_has_required_keys(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    logger = get_logger("test_keys")
    record = _capture_log(logger, logging.INFO, "hello world")
    assert "level" in record
    assert "message" in record
    assert "timestamp" in record


def test_message_value(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    logger = get_logger("test_message")
    record = _capture_log(logger, logging.WARNING, "test message")
    assert record["message"] == "test message"
    assert record["level"] == "WARNING"


def test_log_level_info_suppresses_debug(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    logger = get_logger("test_info_level")
    import io

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    if logger.handlers:
        handler.setFormatter(logger.handlers[0].formatter)
    logger.addHandler(handler)
    try:
        logger.debug("should not appear")
    finally:
        logger.removeHandler(handler)

    assert stream.getvalue().strip() == ""


def test_log_level_debug_enables_debug_messages(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    logger = get_logger("test_debug_level")
    record = _capture_log(logger, logging.DEBUG, "debug msg")
    assert record["message"] == "debug msg"
    assert record["level"] == "DEBUG"
