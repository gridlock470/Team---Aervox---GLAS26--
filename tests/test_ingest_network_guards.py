"""Network-fetch functions must import and fail cleanly without connectivity.

These tests never touch the network: the underlying client libraries are
monkeypatched to simulate an offline / unauthenticated environment.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path():
    """Isolated temp dir (the shared pytest tmp root is permission-locked here)."""
    d = Path(tempfile.mkdtemp(prefix="sih-net-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_imerg_module_imports_and_exposes_api():
    from nowcast.ingest import imerg

    assert hasattr(imerg, "fetch_imerg")
    assert hasattr(imerg, "load_imerg")
    assert issubclass(imerg.ImergAuthError, Exception)


def test_fetch_imerg_without_credentials_raises_typed_error(tmp_path, monkeypatch):
    import earthaccess

    from nowcast.ingest import imerg

    def _no_auth(*_a, **_k):
        raise RuntimeError("no EDL credentials / offline")

    monkeypatch.setattr(earthaccess, "login", _no_auth)
    with pytest.raises(imerg.ImergAuthError):
        imerg.fetch_imerg("2020-06-01", "2020-06-02", tmp_path)


def test_fetch_imerg_login_returns_none_raises(tmp_path, monkeypatch):
    import earthaccess

    from nowcast.ingest import imerg

    monkeypatch.setattr(earthaccess, "login", lambda *a, **k: None)
    with pytest.raises(imerg.ImergAuthError):
        imerg.fetch_imerg("2020-06-01", "2020-06-02", tmp_path)


def test_dem_fetch_module_api():
    from nowcast.ingest import dem

    assert hasattr(dem, "fetch_glo30")
    assert issubclass(dem.DemDownloadError, Exception)


def test_fetch_glo30_network_failure_is_typed(tmp_path, monkeypatch):
    import pooch

    from nowcast.ingest import dem

    monkeypatch.setattr(
        pooch, "retrieve", lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
    )
    with pytest.raises(dem.DemDownloadError):
        dem.fetch_glo30(bbox=(76.0, 27.0, 77.0, 28.0), out_dir=tmp_path)
