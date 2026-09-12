"""Thermodynamic instability predictors: lifted index and CAPE/CIN.

:func:`lifted_index` is **vectorised**: every column in the cube is lifted at
once with closed-form thermodynamics plus a fixed-step RK4 moist ascent. The
original per-column MetPy loop is kept as ``method="metpy"`` so the two can be
compared on real data (see ``tests/test_features_vectorised.py``); it costs
~4.3 ms per column, i.e. ~9 h for the 8.2 M columns of the pilot datacube, and
is a reference implementation only.

:func:`ensure_cape_cin` still loops MetPy per column, but only on the fallback
branch: the datacube ships ``cape``/``cin`` from the reanalysis, so the loop is
not reached by the real feature build. If a cube ever lands without them, that
branch has the same ~9 h cost and needs the same treatment.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

_ORDER = ("time", "level", "lat", "lon")
_LEVEL_DIM = "level"

# ---------------------------------------------------------------------------
# Constants, taken from ``metpy.constants`` (MetPy 1.7) so the vectorised path
# and the MetPy reference path integrate the *same* equation. Hard-coded rather
# than imported so this module stays importable without MetPy;
# ``tests/test_features_vectorised.py`` asserts they still match MetPy's.
# ---------------------------------------------------------------------------
_RD = 287.04749097718457  # dry-air gas constant, J kg-1 K-1
_CP_D = 1004.6662184201462  # dry-air specific heat at constant p, J kg-1 K-1
_LV = 2_500_840.0  # latent heat of vaporisation at 0 C, J kg-1
_EPSILON = 0.6219569100577033  # Rd / Rv
_KAPPA = _RD / _CP_D

# Steps of the RK4 moist ascent from the LCL to the target level. The residual
# against MetPy is dominated by Bolton's closed-form LCL, not by the integrator:
# 20, 40 and 120 steps all land within 1e-4 K of each other.
_MOIST_STEPS = 40

_LI_TARGET_HPA = 500.0


def _prep_columns(
    ds: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[int]]:
    """Return ``(pressure, temp_cols, rh_cols, out_dims, out_shape)``.

    ``pressure`` is 1-D descending (surface first). ``temp_cols`` / ``rh_cols``
    have shape ``(n_level, n_column)`` with columns flattened in C order over
    ``out_dims``. Relative humidity is normalised to a 0-1 fraction.
    """
    dims = [d for d in _ORDER if d in ds["t"].dims]
    t = ds["t"].transpose(*dims)
    rh = ds["rh"].transpose(*dims)
    p = ds[_LEVEL_DIM].values.astype("float64")
    order = np.argsort(-p)
    p_sorted = p[order]
    level_axis = dims.index(_LEVEL_DIM)

    tv = np.moveaxis(np.asarray(t.values, dtype="float64"), level_axis, 0)[order]
    rv = np.moveaxis(np.asarray(rh.values, dtype="float64"), level_axis, 0)[order]
    if np.isfinite(rv).any() and np.nanmax(rv) > 1.5:
        rv = rv / 100.0
    rv = np.clip(rv, 1e-4, 1.0)

    n_level = tv.shape[0]
    out_dims = [d for d in dims if d != _LEVEL_DIM]
    out_shape = [ds.sizes[d] for d in out_dims]
    return p_sorted, tv.reshape(n_level, -1), rv.reshape(n_level, -1), out_dims, out_shape


def _column_da(
    values: np.ndarray, shape: list[int], out_dims: list[str], ds: xr.Dataset, name: str, ln: str
) -> xr.DataArray:
    return xr.DataArray(
        values.reshape(shape).astype("float32"),
        dims=out_dims,
        coords={d: ds[d] for d in out_dims},
        name=name,
        attrs={"long_name": ln},
    )


def saturation_vapor_pressure_hpa(temperature_k: np.ndarray) -> np.ndarray:
    """Bolton (1980) eq. 10 saturation vapour pressure in hPa.

    This is the same expression MetPy uses, so the dewpoint, the LCL and the
    moist ascent below all sit on MetPy's own saturation curve.
    """
    t_c = temperature_k - 273.15
    return 6.112 * np.exp(17.67 * t_c / (temperature_k - 29.65))


def dewpoint_from_rh(temperature_k: np.ndarray, rh_frac: np.ndarray) -> np.ndarray:
    """Dewpoint (K) from temperature and a 0-1 relative-humidity fraction.

    Closed-form inversion of :func:`saturation_vapor_pressure_hpa`, matching
    ``metpy.calc.dewpoint_from_relative_humidity`` analytically.
    """
    vapour = np.clip(rh_frac, 1e-6, 1.0) * saturation_vapor_pressure_hpa(temperature_k)
    val = np.log(vapour / 6.112)
    return 243.5 * val / (17.67 - val) + 273.15


def lcl_temperature(temperature_k: np.ndarray, dewpoint_k: np.ndarray) -> np.ndarray:
    """LCL temperature (K) via Bolton (1980) eq. 15 - closed form, no iteration.

    MetPy solves the LCL by fixed-point iteration; Bolton's formula agrees with
    it to ~0.1 K, which is the dominant term in the residual reported by
    ``tests/test_features_vectorised.py``.
    """
    return 1.0 / (1.0 / (dewpoint_k - 56.0) + np.log(temperature_k / dewpoint_k) / 800.0) + 56.0


def _moist_lapse_dt_dlnp(pressure_hpa: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
    """``dT / d(ln p)`` along a saturated pseudoadiabat.

    Identical to the right-hand side ``metpy.calc.moist_lapse`` integrates,
    rewritten in ``ln p`` so a fixed step size spans the ascent evenly.
    """
    e_s = saturation_vapor_pressure_hpa(temperature_k)
    r_s = _EPSILON * e_s / np.maximum(pressure_hpa - e_s, 1e-6)
    numerator = _RD * temperature_k + _LV * r_s
    denominator = _CP_D + (_LV * _LV * r_s * _EPSILON) / (_RD * temperature_k * temperature_k)
    return numerator / denominator


def parcel_temperature_at(
    p_start_hpa: float,
    temperature_k: np.ndarray,
    rh_frac: np.ndarray,
    p_target_hpa: float,
    *,
    steps: int = _MOIST_STEPS,
) -> np.ndarray:
    """Temperature (K) at ``p_target_hpa`` of a parcel lifted from ``p_start_hpa``.

    Fully vectorised over whatever shape ``temperature_k`` / ``rh_frac`` have:
    dry adiabat to the LCL (Poisson), then a fixed-step RK4 moist ascent of the
    pseudoadiabat. Columns whose LCL is already above the target never saturate,
    so they stay on the dry adiabat the whole way.
    """
    dewpoint_k = dewpoint_from_rh(temperature_k, rh_frac)
    t_lcl = lcl_temperature(temperature_k, dewpoint_k)
    p_lcl = p_start_hpa * (t_lcl / temperature_k) ** (1.0 / _KAPPA)

    stays_dry = p_lcl <= p_target_hpa
    ln_p = np.log(np.where(stays_dry, p_start_hpa, p_lcl))
    t_parcel = np.where(stays_dry, temperature_k, t_lcl).astype("float64")

    step = (np.log(p_target_hpa) - ln_p) / steps
    for _ in range(steps):
        k1 = _moist_lapse_dt_dlnp(np.exp(ln_p), t_parcel)
        k2 = _moist_lapse_dt_dlnp(np.exp(ln_p + step / 2.0), t_parcel + step / 2.0 * k1)
        k3 = _moist_lapse_dt_dlnp(np.exp(ln_p + step / 2.0), t_parcel + step / 2.0 * k2)
        k4 = _moist_lapse_dt_dlnp(np.exp(ln_p + step), t_parcel + step * k3)
        t_parcel = t_parcel + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        ln_p = ln_p + step

    t_dry = temperature_k * (p_target_hpa / p_start_hpa) ** _KAPPA
    return np.where(stays_dry, t_dry, t_parcel)


def _rh_fraction(values: np.ndarray) -> np.ndarray:
    """Normalise relative humidity to a 0-1 fraction (datacube ships percent)."""
    out = np.asarray(values, dtype="float64")
    if np.isfinite(out).any() and np.nanmax(out) > 1.5:
        out = out / 100.0
    return np.clip(out, 1e-4, 1.0)


def _lifted_index_vectorised(ds: xr.Dataset) -> xr.DataArray:
    """Vectorised 500 hPa lifted index - every column lifted in one pass."""
    levels = np.asarray(ds[_LEVEL_DIM].values, dtype="float64")
    sfc = int(np.argmax(levels))
    target_idx = int(np.argmin(np.abs(levels - _LI_TARGET_HPA)))
    if abs(levels[target_idx] - _LI_TARGET_HPA) > 1e-6:
        raise KeyError(
            f"lifted_index needs a {_LI_TARGET_HPA:.0f} hPa level; got {levels.tolist()}"
        )

    t_sfc = ds["t"].isel({_LEVEL_DIM: sfc})
    out_dims = list(t_sfc.dims)
    t_sfc_vals = np.asarray(t_sfc.values, dtype="float64")
    rh_sfc = _rh_fraction(ds["rh"].isel({_LEVEL_DIM: sfc}).transpose(*out_dims).values)
    t_env = np.asarray(
        ds["t"].isel({_LEVEL_DIM: target_idx}).transpose(*out_dims).values, dtype="float64"
    )

    finite = np.isfinite(t_sfc_vals) & np.isfinite(rh_sfc) & np.isfinite(t_env)
    safe_t = np.where(finite, t_sfc_vals, 300.0)
    safe_rh = np.where(finite, rh_sfc, 0.5)
    t_parcel = parcel_temperature_at(
        float(levels[sfc]), safe_t, safe_rh, float(levels[target_idx])
    )
    values = np.where(finite, t_env - t_parcel, np.nan)

    return xr.DataArray(
        values.astype("float32"),
        dims=out_dims,
        coords={d: ds[d] for d in out_dims},
        name="lifted_index",
        attrs={"long_name": "500 hPa lifted index", "units": "K", "method": "vectorised"},
    )


def lifted_index(ds: xr.Dataset, *, method: str = "vectorised") -> xr.DataArray:
    """500 hPa lifted index in K (positive = stable, negative = unstable).

    ``method="vectorised"`` (default) lifts every column at once - minutes for
    the whole pilot cube. ``method="metpy"`` runs the original per-column MetPy
    loop; it is the reference the vectorised path is validated against and is
    far too slow (~4.3 ms/column) for a real build.
    """
    if method == "vectorised":
        return _lifted_index_vectorised(ds)
    if method != "metpy":
        raise ValueError(f"unknown lifted_index method {method!r}")

    from metpy.calc import dewpoint_from_relative_humidity, parcel_profile
    from metpy.calc import lifted_index as _mp_lifted_index
    from metpy.units import units

    p, temp_cols, rh_cols, out_dims, shape = _prep_columns(ds)
    p_q = p * units.hPa
    n_col = temp_cols.shape[1]
    out = np.full(n_col, np.nan, dtype="float64")

    for c in range(n_col):
        tc = temp_cols[:, c]
        rc = rh_cols[:, c]
        if not (np.isfinite(tc).all() and np.isfinite(rc).all()):
            continue
        t_q = tc * units.K
        td_q = dewpoint_from_relative_humidity(t_q, rc * units.dimensionless)
        try:
            profile = parcel_profile(p_q, t_q[0], td_q[0])
            li = _mp_lifted_index(p_q, t_q, profile)
            out[c] = float(np.atleast_1d(li.to("K").magnitude)[0])
        except (ValueError, IndexError):
            continue

    da = _column_da(out, shape, out_dims, ds, "lifted_index", "500 hPa lifted index")
    return da.assign_attrs(units="K", method="metpy")


def ensure_cape_cin(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Return ``(cape, cin)`` in J kg-1.

    Existing ``cape``/``cin`` fields are passed through; otherwise surface-based
    values are computed column-wise with MetPy. ``cin`` is non-positive.
    """
    if {"cape", "cin"} <= set(ds.data_vars):
        return (
            ds["cape"].astype("float32").rename("cape"),
            ds["cin"].astype("float32").rename("cin"),
        )

    from metpy.calc import dewpoint_from_relative_humidity, surface_based_cape_cin
    from metpy.units import units

    p, temp_cols, rh_cols, out_dims, shape = _prep_columns(ds)
    p_q = p * units.hPa
    n_col = temp_cols.shape[1]
    cape = np.full(n_col, np.nan, dtype="float64")
    cin = np.full(n_col, np.nan, dtype="float64")

    for c in range(n_col):
        tc = temp_cols[:, c]
        rc = rh_cols[:, c]
        if not (np.isfinite(tc).all() and np.isfinite(rc).all()):
            continue
        t_q = tc * units.K
        td_q = dewpoint_from_relative_humidity(t_q, rc * units.dimensionless)
        try:
            cc, ci = surface_based_cape_cin(p_q, t_q, td_q)
            cape[c] = float(cc.to("J/kg").magnitude)
            cin[c] = float(ci.to("J/kg").magnitude)
        except (ValueError, IndexError):
            continue

    cape_da = _column_da(cape, shape, out_dims, ds, "cape", "surface-based CAPE")
    cin_da = _column_da(cin, shape, out_dims, ds, "cin", "surface-based CIN")
    return cape_da.assign_attrs(units="J kg-1"), cin_da.assign_attrs(units="J kg-1")
