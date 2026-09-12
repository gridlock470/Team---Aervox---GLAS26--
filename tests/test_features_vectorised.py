"""Accuracy and chunked-array regressions for the vectorised feature maths.

Two defects blocked the first real feature build and both are pinned here:

* ``kinematics._interp_to_height`` used ``apply_ufunc(..., vectorize=True)``
  with no ``dask=`` argument, so any lazily-opened Zarr raised
  ``ValueError: apply_ufunc encountered a chunked array``;
* ``thermo.lifted_index`` looped MetPy once per column (~4.3 ms each, ~9 h for
  the pilot cube). The vectorised replacement is validated against that MetPy
  path here, which is why ``method="metpy"`` still exists.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from nowcast import config
from nowcast.features import kinematics, thermo
from nowcast.testing import synthetic

_HAS_METPY = importlib.util.find_spec("metpy") is not None
metpy_only = pytest.mark.skipif(not _HAS_METPY, reason="metpy is required")

# Agreement demanded of the vectorised lifted index against MetPy, in K. A fast
# but wrong instability proxy is worse than a slow right one.
_MAX_MEDIAN_DIFF_K = 0.5
_MAX_P95_DIFF_K = 0.5

# Columns drawn from the real datacube for the accuracy comparison.
_SAMPLE_COLUMNS = 300


def _cube(n_hours: int = 6, seed: int = 3) -> xr.Dataset:
    return synthetic.make_datacube(n_hours=n_hours, seed=seed).isel(
        lat=slice(0, 8), lon=slice(0, 8)
    )


# ---------------------------------------------------------------------------
# Defect 1 - chunked arrays
# ---------------------------------------------------------------------------
def test_interp_level_last_matches_numpy_interp():
    rng = np.random.default_rng(7)
    heights = np.sort(rng.uniform(0.0, 12_000.0, size=(5, 7)), axis=-1)
    values = rng.normal(size=(5, 7))
    for target in (-100.0, 0.0, 1_000.0, 6_000.0, 99_000.0):
        got = kinematics._interp_level_last(values, heights, target)
        want = np.array(
            [np.interp(target, heights[i], values[i]) for i in range(heights.shape[0])]
        )
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_interp_level_last_handles_unsorted_columns():
    heights = np.array([[3000.0, 0.0, 1500.0]])
    values = np.array([[30.0, 0.0, 15.0]])
    got = kinematics._interp_level_last(values, heights, 750.0)
    np.testing.assert_allclose(got, [7.5])


def test_bulk_shear_on_chunked_dataset_matches_eager():
    """Regression: a lazily-opened (chunked) cube used to raise in apply_ufunc."""
    cube = _cube()
    eager = kinematics.bulk_shear(cube, 0.0, 6000.0)
    lazy = kinematics.bulk_shear(cube.chunk({"time": 2}), 0.0, 6000.0)
    assert lazy.chunks is not None, "expected the lazy path to stay chunked"
    np.testing.assert_allclose(lazy.compute().values, eager.values, rtol=1e-6)


# ---------------------------------------------------------------------------
# Defect 2 - vectorised lifted index
# ---------------------------------------------------------------------------
@metpy_only
def test_thermo_constants_match_metpy():
    """The vectorised path hard-codes MetPy's constants; keep them honest."""
    import metpy.constants as mpconst

    assert thermo._RD == pytest.approx(float(mpconst.Rd.magnitude))
    assert thermo._CP_D == pytest.approx(float(mpconst.Cp_d.magnitude))
    assert thermo._LV == pytest.approx(float(mpconst.Lv.magnitude))
    assert thermo._EPSILON == pytest.approx(float(mpconst.epsilon.magnitude))


@metpy_only
def test_dewpoint_matches_metpy():
    """Bolton dewpoint vs MetPy's own inversion.

    MetPy 1.7 inverts its saturation curve with a slightly different constant
    set, so the two differ by up to ~0.12 K. That is the largest single term in
    the end-to-end lifted-index residual measured below (median 0.07 K), and it
    is an order of magnitude inside the 0.5 K accuracy budget.
    """
    from metpy.calc import dewpoint_from_relative_humidity
    from metpy.units import units

    temperature = np.linspace(250.0, 315.0, 40)
    rh = np.linspace(0.05, 0.99, 40)
    want = (
        dewpoint_from_relative_humidity(
            temperature * units.K, rh * units.dimensionless
        )
        .to("K")
        .magnitude
    )
    np.testing.assert_allclose(thermo.dewpoint_from_rh(temperature, rh), want, atol=0.2)


def test_lifted_index_vectorised_shape_and_dtype():
    li = thermo.lifted_index(_cube())
    assert set(li.dims) == {"time", "lat", "lon"}
    assert li.dtype == np.float32
    assert li.name == "lifted_index"
    assert li.attrs["units"] == "K"
    assert np.isfinite(li.values).mean() > 0.5


def test_lifted_index_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown lifted_index method"):
        thermo.lifted_index(_cube(), method="nope")


def _abs_diff_stats(cube: xr.Dataset) -> tuple[np.ndarray, float, float]:
    fast = thermo.lifted_index(cube, method="vectorised").values.ravel()
    slow = thermo.lifted_index(cube, method="metpy").values.ravel()
    both = np.isfinite(fast) & np.isfinite(slow)
    diff = np.abs(fast[both] - slow[both]).astype("float64")
    return diff, float(np.median(diff)), float(np.percentile(diff, 95))


@metpy_only
def test_lifted_index_vectorised_matches_metpy_on_synthetic():
    cube = _cube(n_hours=6).isel(lat=slice(0, 6), lon=slice(0, 6))
    diff, median, p95 = _abs_diff_stats(cube)
    assert diff.size >= 100
    assert median < _MAX_MEDIAN_DIFF_K, f"median |diff| = {median:.3f} K"
    assert p95 < _MAX_P95_DIFF_K, f"p95 |diff| = {p95:.3f} K"


@metpy_only
@pytest.mark.skipif(
    not config.DATACUBE_PATH.exists(), reason="real datacube not built on this machine"
)
def test_lifted_index_vectorised_matches_metpy_on_real_columns():
    """The accuracy claim that matters: real columns, not synthetic ones."""
    cube = xr.open_zarr(config.DATACUBE_PATH)
    rng = np.random.default_rng(config.RANDOM_SEED)
    picker = {
        dim: xr.DataArray(
            rng.integers(0, cube.sizes[dim], _SAMPLE_COLUMNS), dims="column"
        )
        for dim in ("time", "lat", "lon")
    }
    # Re-label the gathered columns as a fake ``lon`` axis so both code paths
    # see the (level, lon) layout they expect; the physics is per-column anyway.
    sample = (
        cube[["t", "rh"]]
        .isel(picker)
        .drop_vars(["time", "lat", "lon"])
        .rename({"column": "lon"})
        .assign_coords(lon=np.arange(_SAMPLE_COLUMNS, dtype="float64"))
        .load()
    )

    diff, median, p95 = _abs_diff_stats(sample)
    assert diff.size >= _SAMPLE_COLUMNS * 0.9, f"only {diff.size} columns compared"
    assert median < _MAX_MEDIAN_DIFF_K, f"median |diff| = {median:.3f} K"
    assert p95 < _MAX_P95_DIFF_K, f"p95 |diff| = {p95:.3f} K"


def test_parcel_temperature_saturated_parcel_cools_less_than_dry():
    """A saturated parcel releases latent heat, so it must stay warmer aloft."""
    t0 = np.array([300.0])
    moist = thermo.parcel_temperature_at(1000.0, t0, np.array([0.95]), 500.0)
    dry = t0 * (500.0 / 1000.0) ** thermo._KAPPA
    assert moist[0] > dry[0]
    assert moist[0] < t0[0]
