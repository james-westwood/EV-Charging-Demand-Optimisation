"""Smoke test: ensure all required packages import without error."""


def test_lightgbm():
    import lightgbm  # noqa: F401


def test_duckdb():
    import duckdb  # noqa: F401


def test_pandas():
    import pandas  # noqa: F401


def test_pyarrow():
    import pyarrow  # noqa: F401


def test_sklearn():
    import sklearn  # noqa: F401


def test_shap():
    import shap  # noqa: F401


def test_fastapi():
    import fastapi  # noqa: F401


def test_uvicorn():
    import uvicorn  # noqa: F401


def test_httpx():
    import httpx  # noqa: F401


def test_pytest():
    import pytest  # noqa: F401


def test_pytest_asyncio():
    import pytest_asyncio  # noqa: F401


def test_workalendar():
    import workalendar  # noqa: F401


def test_pulp():
    import pulp  # noqa: F401


def test_scipy():
    import scipy  # noqa: F401


def test_numpy():
    import numpy  # noqa: F401
