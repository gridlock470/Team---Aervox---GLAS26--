"""Tests for :mod:`nowcast.ingest.run_wget` using dummy shell scripts."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from nowcast.ingest.run_wget import _resolve_shell, run_wget_scripts


def _find_functional_bash() -> str | None:
    candidate = _resolve_shell("bash")
    if candidate:
        try:
            res = subprocess.run([candidate, "-c", "exit 0"], capture_output=True, timeout=5)
            if res.returncode == 0:
                return candidate
        except Exception:
            pass
    return None


_BASH = _find_functional_bash()
pytestmark = pytest.mark.skipif(_BASH is None, reason="functional bash not available on PATH")


@pytest.fixture
def tmp_path():
    """Isolated temp dir (the shared pytest tmp root is permission-locked here)."""
    d = Path(tempfile.mkdtemp(prefix="sih-wget-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_runs_script_and_reports_new_file(tmp_path):
    script = tmp_path / "download_data.sh"
    script.write_text("#!/usr/bin/env bash\nprintf 'abcdef' > payload.bin\n")
    results = run_wget_scripts(tmp_path)
    assert len(results) == 1
    res = results[0]
    assert res.ok and res.returncode == 0
    assert (tmp_path / "payload.bin").exists()
    names = [p.name for p in res.new_files]
    assert "payload.bin" in names
    assert res.total_bytes >= 6


def test_no_scripts_returns_empty(tmp_path):
    assert run_wget_scripts(tmp_path) == []


def test_missing_directory_raises():
    with pytest.raises(FileNotFoundError):
        run_wget_scripts("does/not/exist/anywhere")


def test_failing_script_captured(tmp_path):
    (tmp_path / "bad.sh").write_text("#!/usr/bin/env bash\nexit 3\n")
    (res,) = run_wget_scripts(tmp_path)
    assert not res.ok
    assert res.returncode == 3
