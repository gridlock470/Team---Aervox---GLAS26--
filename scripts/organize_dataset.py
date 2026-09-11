"""Arrange the downloaded data into DataSet/organized/ for a readable overview.

    python scripts/organize_dataset.py

Files are HARDLINKED, not copied or moved. Three reasons:

* a hardlink adds no bytes -- the 1.2 GB of DEM tiles is not duplicated;
* the originals under DataSet/raw/ stay exactly where nowcast.config expects
  them, so ingest and the fetch scripts keep working;
* a fetch that is still running keeps writing to its own directory without
  finding it moved out from under it.

Re-run it whenever you want: it links whatever is new and rewrites the
inventory. Nothing under raw/ is ever modified or deleted.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from nowcast import config

ORGANIZED = Path(config.DATA_DIR) / "organized"


@dataclass(frozen=True)
class Source:
    order: int
    slug: str
    title: str
    directory: Path
    pattern: str
    expected: int | None
    feeds: str

    @property
    def dest(self) -> Path:
        return ORGANIZED / f"{self.order:02d}_{self.slug}"


def sources() -> list[Source]:
    raw = Path(config.RAW_DIR)
    return [
        Source(1, "terrain_dem", "Copernicus GLO-30 DEM", config.RAW_DEM_DIR, "*.tif", 30,
               "elevation, slope, flow accumulation, HAND"),
        Source(2, "reanalysis_era5", "ERA5 reanalysis", raw / "era5", "*.nc", 24,
               "t2m, winds, mslp, tcwv, CAPE, CIN + pressure-level profiles"),
        Source(3, "rainfall_imerg", "GPM IMERG rainfall", config.RAW_IMERG_DIR, "*.nc", 183,
               "rainfall labels"),
        Source(4, "reanalysis_imdaa", "IMDAA reanalysis", config.RAW_IMDAA_DIR, "*", None,
               "preferred source for the ERA5 fields"),
        Source(5, "rainfall_mera", "MERA merged rainfall", config.RAW_MERA_DIR, "*", None,
               "4 km rainfall labels"),
        Source(6, "satellite_insat3d", "INSAT-3D imagery", config.RAW_INSAT3D_DIR, "*", None,
               "cloud-top temperature, water vapour"),
        Source(7, "satellite_insat3dr", "INSAT-3DR imagery", config.RAW_INSAT3DR_DIR, "*", None,
               "cloud-top temperature, water vapour"),
    ]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def link_source(src: Source) -> tuple[int, int, int]:
    """Hardlink new files into the organized folder. Returns (linked, already, bytes)."""
    src.dest.mkdir(parents=True, exist_ok=True)
    linked = already = total = 0
    if not src.directory.exists():
        return 0, 0, 0

    for path in sorted(src.directory.glob(src.pattern)):
        if path.is_dir() or path.suffix == ".part":
            continue  # in-flight temp files are not finished data
        try:
            total += path.stat().st_size
        except OSError:
            continue
        target = src.dest / path.name
        if target.exists():
            already += 1
            continue
        try:
            os.link(path, target)
            linked += 1
        except OSError:
            # Different volume, or the file vanished mid-rename. Skip rather
            # than silently copying gigabytes.
            pass
    return linked, already, total


def status_for(count: int, expected: int | None) -> str:
    if expected is None:
        return "not started" if count == 0 else "partial"
    if count == 0:
        return "not started"
    if count >= expected:
        return "complete"
    return f"{100 * count / expected:.0f}% downloaded"


def write_inventory(rows: list[tuple[Source, int, int]]) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    grand = sum(b for _, _, b in rows)
    have = sum(1 for _s, c, _b in rows if c)

    lines = [
        "# Dataset inventory",
        "",
        f"Generated {stamp} by `scripts/organize_dataset.py`.",
        "",
        f"**{have} of {len(rows)} sources have data on disk - {human(grand)} total.**",
        "",
        "Files here are hardlinks to `DataSet/raw/`. They occupy no extra space, and",
        "the originals stay where the pipeline expects them.",
        "",
        "| # | Source | Status | Files | Size | Feeds |",
        "|---|--------|--------|-------|------|-------|",
    ]
    for s, count, nbytes in rows:
        expected = f"/{s.expected}" if s.expected else ""
        lines.append(
            f"| {s.order:02d} | {s.title} | {status_for(count, s.expected)} | "
            f"{count}{expected} | {human(nbytes) if nbytes else '-'} | {s.feeds} |"
        )

    pending = [s.title for s, c, _b in rows if s.expected is None and c == 0]
    partial = [
        f"{s.title} ({c}/{s.expected})"
        for s, c, _b in rows if s.expected and 0 < c < s.expected
    ]
    lines += ["", "## Still to come", ""]
    if partial:
        lines.append(f"- Downloading now: {', '.join(partial)}")
    if pending:
        lines.append(f"- Not started (queued at the provider): {', '.join(pending)}")
    if not partial and not pending:
        lines.append("- Nothing outstanding.")
    lines += [
        "",
        "Re-run `python scripts/organize_dataset.py` to pick up newly downloaded files.",
        "",
    ]

    out = ORGANIZED / "INVENTORY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    ORGANIZED.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[Source, int, int]] = []

    for src in sources():
        linked, already, nbytes = link_source(src)
        count = linked + already
        rows.append((src, count, nbytes))
        note = f"+{linked} new" if linked else "no change"
        print(f"  {src.order:02d} {src.title:<26} {count:>4} files  "
              f"{human(nbytes):>9}  ({note})")

    inv = write_inventory(rows)
    total = sum(b for _s, _c, b in rows)
    print(f"\n  organized -> {ORGANIZED}")
    print(f"  inventory -> {inv}")
    print(f"  {human(total)} linked, 0 bytes duplicated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
