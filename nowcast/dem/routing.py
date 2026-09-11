"""Terrain routing from a DEM: D8 flow direction, accumulation, streams, HAND.

:func:`compute_routing` prefers :mod:`pysheds` (pit-fill + flat resolution +
D8). ``pysheds`` is frequently broken on Windows, so a self-contained NumPy
implementation (:func:`_d8_numpy`) is used automatically when pysheds cannot be
imported or fails. Both paths return the same :class:`xarray.Dataset`.

Output variables (dims ``(lat, lon)``), matching :data:`nowcast.schema.STATIC_VARS`:

* ``flow_direction`` -- D8 pointer, codes ``1,2,4,...,128`` (``0`` at
  outlets/sinks). See "D8 code convention" below.
* ``flow_accumulation`` -- number of cells draining through each cell (>= 1)
* ``streams`` -- boolean channel mask (accumulation over a threshold)
* ``hand`` -- height above nearest drainage (m, >= 0)
* ``slope`` -- terrain slope in degrees

D8 code convention
------------------
The numeric codes are the ESRI D8 *values* paired with the array-offset table
below. That offset table is **shared verbatim** with
:data:`nowcast.features.labels._D8_OFFSETS` (the label-side flow accumulator)
and matches the pysheds default ``dirmap`` in array space, so the numpy and
pysheds branches and the downstream label router are all mutually consistent.

The codes are **not** re-mapped for the lat-ascending project grid: rows
increase northward, so e.g. code ``4`` -> offset ``(+1, 0)`` means "flows
toward increasing latitude" (geographic *north* here), even though ESRI names
value 4 "South". Only the drainage *topology* is guaranteed meaningful; the
compass names attached to the raw ESRI values are not.
"""

from __future__ import annotations

import heapq

import numpy as np
import xarray as xr

from nowcast import config

__all__ = ["compute_routing", "routing_backend"]

# D8 table: (drow, dcol, ESRI code, distance-in-cells). The (drow, dcol) ->
# code mapping is identical to nowcast.features.labels._D8_OFFSETS and to the
# pysheds default dirmap in array space. See the module docstring: codes are
# NOT reinterpreted for the lat-ascending grid, so code 4 == offset (+1, 0) ==
# "toward increasing latitude" (north here), not ESRI's "South".
_D8: tuple[tuple[int, int, int, float], ...] = (
    (0, 1, 1, 1.0),
    (1, 1, 2, np.sqrt(2)),
    (1, 0, 4, 1.0),
    (1, -1, 8, np.sqrt(2)),
    (0, -1, 16, 1.0),
    (-1, -1, 32, np.sqrt(2)),
    (-1, 0, 64, 1.0),
    (-1, 1, 128, np.sqrt(2)),
)
_CODE_TO_DELTA: dict[int, tuple[int, int]] = {c: (dr, dc) for dr, dc, c, _ in _D8}

_CELL_SIZE_M = config.GRID_RESOLUTION_DEG * 111_000.0


def routing_backend() -> str:
    """Return the backend :func:`compute_routing` will use (``"pysheds"`` or
    ``"numpy"``).

    Probes pysheds on a tiny array so a merely-importable-but-broken pysheds
    (e.g. the numpy 2.x / pysheds 0.5 ``np.in1d`` incompatibility) is reported
    as ``"numpy"``.
    """
    probe = np.array(
        [[9, 8, 7, 6, 5], [8, 7, 6, 5, 4], [7, 6, 5, 4, 3], [6, 5, 4, 3, 2], [5, 4, 3, 2, 1]],
        dtype="float64",
    )
    try:
        _d8_pysheds(probe, None)
        return "pysheds"
    except Exception:  # noqa: BLE001
        return "numpy"


# ---------------------------------------------------------------------------
# NumPy D8 implementation
# ---------------------------------------------------------------------------
def _fill_depressions(dem: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    """Priority-flood depression filling with a tiny gradient across flats."""
    nr, nc = dem.shape
    filled = np.array(dem, dtype="float64")
    closed = np.zeros((nr, nc), dtype=bool)
    heap: list[tuple[float, int, int]] = []

    for i in range(nr):
        for j in (0, nc - 1):
            if not closed[i, j]:
                closed[i, j] = True
                heapq.heappush(heap, (filled[i, j], i, j))
    for j in range(nc):
        for i in (0, nr - 1):
            if not closed[i, j]:
                closed[i, j] = True
                heapq.heappush(heap, (filled[i, j], i, j))

    while heap:
        z, i, j = heapq.heappop(heap)
        for dr, dc, _code, _dist in _D8:
            ni, nj = i + dr, j + dc
            if 0 <= ni < nr and 0 <= nj < nc and not closed[ni, nj]:
                closed[ni, nj] = True
                filled[ni, nj] = max(filled[ni, nj], z + epsilon)
                heapq.heappush(heap, (filled[ni, nj], ni, nj))
    return filled


def _flow_direction(filled: np.ndarray) -> np.ndarray:
    """Steepest-descent D8 pointer grid (0 where no lower neighbour exists)."""
    nr, nc = filled.shape
    fdir = np.zeros((nr, nc), dtype="int32")
    for i in range(nr):
        for j in range(nc):
            best_slope = 0.0
            best_code = 0
            for dr, dc, code, dist in _D8:
                ni, nj = i + dr, j + dc
                if 0 <= ni < nr and 0 <= nj < nc:
                    drop = (filled[i, j] - filled[ni, nj]) / dist
                    if drop > best_slope:
                        best_slope = drop
                        best_code = code
            fdir[i, j] = best_code
    return fdir


def _flow_accumulation(filled: np.ndarray, fdir: np.ndarray) -> np.ndarray:
    """Number of cells draining through each cell (self included, >= 1)."""
    nr, nc = filled.shape
    acc = np.ones((nr, nc), dtype="float64")
    order = np.argsort(filled, axis=None)[::-1]  # high -> low
    for flat in order:
        i, j = divmod(int(flat), nc)
        code = int(fdir[i, j])
        if code == 0:
            continue
        dr, dc = _CODE_TO_DELTA[code]
        ni, nj = i + dr, j + dc
        if 0 <= ni < nr and 0 <= nj < nc:
            acc[ni, nj] += acc[i, j]
    return acc


def _hand(elev: np.ndarray, fdir: np.ndarray, stream: np.ndarray) -> np.ndarray:
    """Height above nearest drainage: drop downslope to the first stream cell."""
    nr, nc = elev.shape
    hand = np.zeros((nr, nc), dtype="float64")
    max_steps = nr * nc
    for i in range(nr):
        for j in range(nc):
            ci, cj = i, j
            steps = 0
            while not stream[ci, cj] and int(fdir[ci, cj]) != 0 and steps < max_steps:
                dr, dc = _CODE_TO_DELTA[int(fdir[ci, cj])]
                ni, nj = ci + dr, cj + dc
                if not (0 <= ni < nr and 0 <= nj < nc):
                    break
                ci, cj = ni, nj
                steps += 1
            hand[i, j] = max(0.0, elev[i, j] - elev[ci, cj])
    return hand


def _slope_deg(elev: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(elev.astype("float64"))
    return np.degrees(np.arctan(np.hypot(gy, gx) / _CELL_SIZE_M))


def _d8_numpy(elev: np.ndarray, stream_threshold: float | None) -> dict[str, np.ndarray]:
    """Full NumPy routing stack for a 2-D elevation array."""
    filled = _fill_depressions(elev)
    fdir = _flow_direction(filled)
    acc = _flow_accumulation(filled, fdir)
    if stream_threshold is None:
        stream_threshold = float(max(2.0, np.nanpercentile(acc, 90.0)))
    stream = acc >= stream_threshold
    if not stream.any():  # guarantee a non-empty channel network
        stream = acc >= np.nanmax(acc)
    hand = _hand(elev, fdir, stream)
    slope = _slope_deg(elev)
    return {
        "flow_direction": fdir.astype("float32"),
        "flow_accumulation": acc.astype("float32"),
        "streams": stream,
        "hand": hand.astype("float32"),
        "slope": slope.astype("float32"),
    }


# ---------------------------------------------------------------------------
# pysheds implementation
# ---------------------------------------------------------------------------
def _d8_pysheds(elev: np.ndarray, stream_threshold: float | None) -> dict[str, np.ndarray]:
    """Routing via pysheds. Raises on any failure so the caller can fall back."""
    from affine import Affine  # noqa: PLC0415
    from pysheds.grid import Grid  # noqa: PLC0415
    from pysheds.view import Raster, ViewFinder  # noqa: PLC0415

    nr, nc = elev.shape
    res = config.GRID_RESOLUTION_DEG
    affine = Affine(res, 0.0, float(config.BBOX_WEST), 0.0, res, float(config.BBOX_SOUTH))
    view = ViewFinder(affine=affine, shape=(nr, nc), nodata=np.float64(-9999.0))
    dem = Raster(elev.astype("float64"), viewfinder=view)
    grid = Grid.from_raster(dem)

    pit_filled = grid.fill_pits(dem)
    flooded = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(flooded)
    # pysheds default dirmap; its array-space offsets equal ``_D8`` above, so the
    # numpy and pysheds ``flow_direction`` codes agree cell-for-cell.
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated, dirmap=dirmap)
    acc = grid.accumulation(fdir, dirmap=dirmap)

    acc_arr = np.asarray(acc, dtype="float64") + 1.0
    if stream_threshold is None:
        stream_threshold = float(max(2.0, np.nanpercentile(acc_arr, 90.0)))
    stream = acc_arr >= stream_threshold
    if not stream.any():
        stream = acc_arr >= np.nanmax(acc_arr)

    hand = grid.compute_hand(fdir, inflated, acc_arr > stream_threshold, dirmap=dirmap)
    hand_arr = np.nan_to_num(np.asarray(hand, dtype="float64"), nan=0.0)
    hand_arr = np.clip(hand_arr, 0.0, None)

    return {
        "flow_direction": np.asarray(fdir, dtype="float32"),
        "flow_accumulation": acc_arr.astype("float32"),
        "streams": stream,
        "hand": hand_arr.astype("float32"),
        "slope": _slope_deg(elev).astype("float32"),
    }


def compute_routing(
    elevation_da: xr.DataArray, *, stream_threshold: float | None = None
) -> xr.Dataset:
    """Compute the D8 routing stack for ``elevation_da``.

    Parameters
    ----------
    elevation_da:
        2-D elevation :class:`xarray.DataArray` with ``lat`` / ``lon`` coords
        (e.g. from :func:`nowcast.ingest.dem.load_dem` or
        :func:`nowcast.testing.synthetic.make_dem`).
    stream_threshold:
        Flow-accumulation threshold (in cells) that defines a channel. When
        ``None`` the 90th percentile of the accumulation is used so the
        stream network is always non-empty.

    Returns
    -------
    xarray.Dataset
        ``flow_direction``, ``flow_accumulation``, ``streams``, ``hand``,
        ``slope`` on the same grid as the input.
    """
    if elevation_da.ndim != 2:
        raise ValueError(f"expected a 2-D elevation field, got dims {elevation_da.dims}")

    elev = np.asarray(elevation_da.values, dtype="float64")
    elev = np.where(np.isfinite(elev), elev, np.nanmin(elev[np.isfinite(elev)]))

    try:
        result = _d8_pysheds(elev, stream_threshold)
        backend = "pysheds"
    except Exception:  # noqa: BLE001 - pysheds missing or broken -> NumPy fallback
        result = _d8_numpy(elev, stream_threshold)
        backend = "numpy"

    lat = elevation_da["lat"]
    lon = elevation_da["lon"]
    ds = xr.Dataset(
        {
            "flow_direction": (("lat", "lon"), result["flow_direction"]),
            "flow_accumulation": (("lat", "lon"), result["flow_accumulation"]),
            "streams": (("lat", "lon"), result["streams"]),
            "hand": (("lat", "lon"), result["hand"]),
            "slope": (("lat", "lon"), result["slope"]),
        },
        coords={"lat": lat, "lon": lon},
    )
    ds["flow_direction"].attrs.update(
        units="1",
        long_name="D8 flow-direction code (ESRI values, array-offset convention)",
        note="codes not remapped for lat-ascending grid; see nowcast.dem.routing docstring",
    )
    ds["flow_accumulation"].attrs.update(units="cells", long_name="flow accumulation")
    ds["streams"].attrs.update(units="bool", long_name="channel network mask")
    ds["hand"].attrs.update(units="m", long_name="height above nearest drainage")
    ds["slope"].attrs.update(units="deg", long_name="terrain slope")
    ds.attrs["routing_backend"] = backend
    return ds
