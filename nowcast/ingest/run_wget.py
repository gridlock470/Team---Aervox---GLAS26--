"""Run the portal-provided download scripts (wget / shell) and log results.

Portals such as NCMRWF RDS and NASA GES DISC hand out ``*.sh`` wget scripts
instead of a direct API. :func:`run_wget_scripts` executes every script in a
directory through ``subprocess`` and reports the files that appeared plus
their sizes.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ScriptResult", "run_wget_scripts"]

_LOG = logging.getLogger(__name__)

_SCRIPT_GLOBS = ("*.sh", "*.bash", "wget*.txt")


def _resolve_shell(shell: str = "bash") -> str:
    """Resolve a functional shell executable across platforms.

    On Windows:
    1. Check for Git Bash (e.g. C:\\Program Files\\Git\\bin\\bash.exe).
    2. Avoid the WindowsApps stub which emits REGDB_E_CLASSNOTREG when WSL is not active.
    """
    if shell == "bash":
        for candidate in (
            Path(r"C:\Program Files\Git\bin\bash.exe"),
            Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
            Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
        ):
            if candidate.is_file():
                return str(candidate)
        found = shutil.which("bash")
        if found and "WindowsApps" not in found:
            return found
    return shell


@dataclass
class ScriptResult:
    """Outcome of running one download script."""

    script: Path
    returncode: int
    stdout: str
    stderr: str
    new_files: list[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if the script exited 0."""
        return self.returncode == 0

    @property
    def total_bytes(self) -> int:
        """Sum of sizes of files that appeared while the script ran."""
        return sum(p.stat().st_size for p in self.new_files if p.exists())


def _snapshot(directory: Path) -> dict[Path, int]:
    return {
        p: p.stat().st_size
        for p in directory.rglob("*")
        if p.is_file()
    }


def run_wget_scripts(
    directory: str | Path,
    *,
    shell: str = "bash",
    timeout: float | None = 3600.0,
) -> list[ScriptResult]:
    """Execute every download script found in ``directory``.

    Parameters
    ----------
    directory:
        Folder holding the portal scripts (and where downloads land).
    shell:
        Interpreter used to run a script (``"bash"`` by default).
    timeout:
        Per-script timeout in seconds (``None`` disables it).

    Returns
    -------
    list[ScriptResult]
        One entry per script, in sorted filename order.

    Raises
    ------
    FileNotFoundError
        If ``directory`` does not exist.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"script directory not found: {directory}")

    scripts: list[Path] = []
    for pattern in _SCRIPT_GLOBS:
        scripts.extend(sorted(directory.glob(pattern)))
    scripts = sorted(set(scripts))
    if not scripts:
        _LOG.warning("no download scripts (%s) in %s", ", ".join(_SCRIPT_GLOBS), directory)
        return []

    resolved_shell = _resolve_shell(shell)
    results: list[ScriptResult] = []
    for script in scripts:
        before = _snapshot(directory)
        _LOG.info("running download script %s", script.name)
        try:
            cmd = (
                [resolved_shell, script.name]
                if Path(directory) == script.parent
                else [resolved_shell, script.as_posix()]
            )
            proc = subprocess.run(
                cmd,
                cwd=str(directory),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            rc, out, err = proc.returncode, proc.stdout, proc.stderr
        except FileNotFoundError as err:
            rc, out, err = 127, "", f"interpreter '{resolved_shell}' not found: {err}"
        except subprocess.TimeoutExpired as err:
            rc, out, err = 124, err.stdout or "", f"timed out after {timeout}s"

        after = _snapshot(directory)
        new_files = sorted(p for p, size in after.items() if before.get(p) != size)
        for nf in new_files:
            _LOG.info("  -> %s (%d bytes)", nf.name, nf.stat().st_size if nf.exists() else 0)
        results.append(
            ScriptResult(script=script, returncode=rc, stdout=out, stderr=err, new_files=new_files)
        )

    return results
