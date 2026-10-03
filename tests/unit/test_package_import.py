"""Έλεγχος ότι το πακέτο elfantasy και τα υποπακέτα του εισάγονται σωστά (src layout)."""

import importlib

import pytest

PACKAGES = [
    "elfantasy",
    "elfantasy.ingest",
    "elfantasy.db",
    "elfantasy.features",
    "elfantasy.model",
    "elfantasy.api",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name):
    assert importlib.import_module(name) is not None
