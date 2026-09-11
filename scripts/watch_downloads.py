"""Live progress for the dataset downloads. Read-only; safe to run anytime.

    python scripts/watch_downloads.py

Redraws in place and reports a new arrival the moment a file lands. Rates and
ETAs are measured from arrivals observed while this is running, so they reflect
what the download is doing now rather than a long-run average -- which matters
because throughput here has varied by an order of magnitude.

Ctrl-C to quit; it never touches the files.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from nowcast import config

REFRESH_S = 2.0
BAR_W = 28
# Arrivals kept per dataset for the rate estimate. Short on purpose: a long
# window would hide a slowdown that has already happened.
WINDOW = 12


@dataclass
class Track:
    name: str
    directory: Path
    pattern: str
    target: int
    seen: set[str] = field(default_factory=set)
    arrivals: list[float] = field(default_factory=list)

    def poll(self) -> list[str]:
        """Return names of files that appeared since the last poll."""
        now = sorted(p.name for p in self.directory.glob(self.pattern))
        fresh = [n for n in now if n not in self.seen]
        self.seen.update(now)
        for _ in fresh:
            self.arrivals.append(time.time())
        del self.arrivals[:-WINDOW]
        return fresh

    @property
    def count(self) -> int:
        return len(self.seen)

    @property
    def bytes(self) -> int:
        total = 0
        for p in self.directory.glob(self.pattern):
            try:
                total += p.stat().st_size
            except OSError:
                pass  # file may vanish mid-rename
        return total

    def seconds_per_file(self) -> float | None:
        """Mean gap between recent arrivals, or None until two are seen."""
        if len(self.arrivals) < 2:
            return None
        span = self.arrivals[-1] - self.arrivals[0]
        return span / (len(self.arrivals) - 1) if span > 0 else None

    def eta_s(self) -> float | None:
        rate = self.seconds_per_file()
        remaining = max(0, self.target - self.count)
        if rate is None or remaining == 0:
            return None
        return rate * remaining


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{int(n)}B" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GB"


def human_time(s: float | None) -> str:
    if s is None:
        return "--"
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def bar(done: int, total: int, width: int = BAR_W) -> str:
    if total <= 0:
        return " " * width
    filled = round(width * min(done, total) / total)
    return "#" * filled + "." * (width - filled)


def render(tracks: list[Track], events: list[str], started: float) -> str:
    out = [
        "  SIH dataset downloads      elapsed " + human_time(time.time() - started),
        "",
    ]
    grand = 0
    for t in tracks:
        pct = 100 * t.count / t.target if t.target else 100.0
        rate = t.seconds_per_file()
        rate_s = f"{human_time(rate)}/file" if rate else "measuring"
        state = "done" if t.count >= t.target else f"eta {human_time(t.eta_s())}"
        out.append(
            f"  {t.name:<6} [{bar(t.count, t.target)}] "
            f"{t.count:>4}/{t.target:<4} {pct:5.1f}%  "
            f"{human_bytes(t.bytes):>9}  {rate_s:>13}  {state}"
        )
        grand += t.bytes

    out += ["", f"  total on disk  {human_bytes(grand)}", ""]
    if events:
        out.append("  recent arrivals")
        for line in events[-6:]:
            out.append(f"    {line}")
    out.append("")
    out.append("  Ctrl-C to quit. Counts only -- this never opens the files.")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imerg-total", type=int, default=183,
                    help="expected IMERG days (default: Apr-Sep 2018)")
    ap.add_argument("--era5-total", type=int, default=24)
    ap.add_argument("--dem-total", type=int, default=30)
    ap.add_argument("--once", action="store_true", help="print one frame and exit")
    args = ap.parse_args()

    tracks = [
        Track("DEM", config.RAW_DEM_DIR, "*.tif", args.dem_total),
        Track("ERA5", Path(config.RAW_DIR) / "era5", "*.nc", args.era5_total),
        Track("IMERG", config.RAW_IMERG_DIR, "*.nc", args.imerg_total),
    ]
    for t in tracks:
        t.directory.mkdir(parents=True, exist_ok=True)
        t.poll()           # seed the baseline
        t.arrivals.clear()  # files already present are not arrivals

    started = time.time()
    events: list[str] = []

    if args.once:
        print(render(tracks, events, started))
        return 0

    try:
        while True:
            for t in tracks:
                for name in t.poll():
                    events.append(f"{time.strftime('%H:%M:%S')}  {t.name:<6} {name}")

            frame = render(tracks, events, started)
            rows = shutil.get_terminal_size((100, 30)).lines
            # Clear, home, then pad so stale text below cannot linger.
            sys.stdout.write("\033[H\033[J" + frame + "\n" * max(0, rows - frame.count("\n") - 3))
            sys.stdout.flush()

            if all(t.count >= t.target for t in tracks):
                print("\n  All targets reached.")
                return 0
            time.sleep(REFRESH_S)
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        return 130


if __name__ == "__main__":
    sys.exit(main())
