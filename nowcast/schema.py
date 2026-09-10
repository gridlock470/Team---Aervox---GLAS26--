"""Data contract for the datacube, the feature matrix and the model tensors.

This module is authoritative for:

* the variable names, units, dtypes and dimension order in the Zarr datacube;
* the exact ordered list of model-input channels (:data:`FEATURE_CHANNELS`);
* the shape of a single training sample.

Ingestion, feature engineering and the datamodule all import from here so the
three slices cannot drift apart. Conformance is checked by
:func:`validate_datacube` and :func:`validate_sample`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nowcast import config


class SchemaError(ValueError):
    """Raised when data does not conform to the contract in this module."""


# ---------------------------------------------------------------------------
# Dimension order
# ---------------------------------------------------------------------------
LEVEL_DIMS: tuple[str, ...] = ("time", "level", "lat", "lon")
SURFACE_DIMS: tuple[str, ...] = ("time", "lat", "lon")
STATIC_DIMS: tuple[str, ...] = ("lat", "lon")
TARGET_DIMS: tuple[str, ...] = ("time", "hazard", "lead", "lat", "lon")


@dataclass(frozen=True)
class VarSpec:
    """One variable in the datacube."""

    name: str
    dims: tuple[str, ...]
    units: str
    dtype: str = "float32"
    description: str = ""


# ---------------------------------------------------------------------------
# Standardised variables that ingestion must produce in the datacube
# ---------------------------------------------------------------------------
SURFACE_VARS: tuple[VarSpec, ...] = (
    VarSpec("t2m", SURFACE_DIMS, "K", description="2 m air temperature"),
    VarSpec("u10", SURFACE_DIMS, "m s-1", description="10 m zonal wind"),
    VarSpec("v10", SURFACE_DIMS, "m s-1", description="10 m meridional wind"),
    VarSpec("prmsl", SURFACE_DIMS, "Pa", description="mean sea level pressure"),
    VarSpec("tcwv", SURFACE_DIMS, "kg m-2", description="total column water vapour"),
    VarSpec("cape", SURFACE_DIMS, "J kg-1", description="CAPE (IMDAA or MetPy-derived)"),
    VarSpec("cin", SURFACE_DIMS, "J kg-1", description="CIN (IMDAA or MetPy-derived)"),
    VarSpec("precip", SURFACE_DIMS, "mm h-1", description="merged precip rate; label source"),
)

LEVEL_VARS: tuple[VarSpec, ...] = (
    VarSpec("t", LEVEL_DIMS, "K", description="air temperature on pressure levels"),
    VarSpec("rh", LEVEL_DIMS, "%", description="relative humidity"),
    VarSpec("z", LEVEL_DIMS, "m2 s-2", description="geopotential"),
    VarSpec("u", LEVEL_DIMS, "m s-1", description="zonal wind"),
    VarSpec("v", LEVEL_DIMS, "m s-1", description="meridional wind"),
)

SATELLITE_VARS: tuple[VarSpec, ...] = (
    VarSpec("ctt", SURFACE_DIMS, "K", description="INSAT cloud-top temperature (TIR-1 BT)"),
    VarSpec("wv_bt", SURFACE_DIMS, "K", description="INSAT water-vapour channel brightness temp"),
)

STATIC_VARS: tuple[VarSpec, ...] = (
    VarSpec("elevation", STATIC_DIMS, "m", description="Copernicus GLO-30 DEM resampled to grid"),
    VarSpec("slope", STATIC_DIMS, "deg", description="terrain slope"),
    VarSpec("flow_accumulation", STATIC_DIMS, "cells", description="pysheds flow accumulation"),
    VarSpec("flow_direction", STATIC_DIMS, "1", description="D8 flow-direction pointer"),
    VarSpec("hand", STATIC_DIMS, "m", description="height above nearest drainage"),
)

ALL_DATACUBE_VARS: tuple[VarSpec, ...] = (
    SURFACE_VARS + LEVEL_VARS + SATELLITE_VARS + STATIC_VARS
)

# ---------------------------------------------------------------------------
# Ordered model-input channels
# ---------------------------------------------------------------------------
# The model consumes a float32 tensor of shape (C, T, H, W). This tuple is the
# AUTHORITATIVE channel order. ``features.assemble`` writes channels in this
# order and ``data.datamodule`` reads them in this order — never reorder without
# updating both and retraining.
FEATURE_CHANNELS: tuple[str, ...] = (
    # --- dynamic surface / thermodynamics ---
    "t2m",
    "prmsl",
    "tcwv",
    "cape",
    "cin",
    "iwv",
    "iwv_tendency_1h",
    "lifted_index",
    # --- kinematics ---
    "conv_850",
    "shear_0_6km",
    "shear_0_1km",
    "u10",
    "v10",
    # --- satellite lead signal ---
    "ctt",
    "ctt_drop_rate_1h",
    "wv_bt",
    # --- static terrain (broadcast over the time axis) ---
    "elevation",
    "slope",
    "flow_accumulation",
    "hand",
)

# Channels that carry no time dimension in the datacube and are broadcast.
STATIC_CHANNELS: tuple[str, ...] = (
    "elevation",
    "slope",
    "flow_accumulation",
    "hand",
)

# Channels the flash-flood head receives in addition to the shared backbone
# features (routed terrain context).
FLASH_FLOOD_EXTRA_CHANNELS: tuple[str, ...] = (
    "flow_accumulation",
    "flow_direction",
    "hand",
)

N_CHANNELS: int = len(FEATURE_CHANNELS)
N_HAZARDS: int = len(config.HAZARDS)
N_LEADS: int = len(config.LEAD_TIMES_H)

# Shapes of one training sample.
INPUT_SHAPE: tuple[int, int, int, int] = (
    N_CHANNELS,
    config.INPUT_SEQ_LEN,
    config.GRID_SHAPE[0],
    config.GRID_SHAPE[1],
)
TARGET_SHAPE: tuple[int, int, int, int] = (
    N_HAZARDS,
    N_LEADS,
    config.GRID_SHAPE[0],
    config.GRID_SHAPE[1],
)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
def _coords_match(actual: np.ndarray, expected: np.ndarray, name: str) -> None:
    actual = np.asarray(actual, dtype="float64")
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=1e-4):
        raise SchemaError(
            f"coordinate '{name}' does not match the target grid "
            f"(got shape {actual.shape}, expected {expected.shape})"
        )


def validate_datacube(ds, *, require_static: bool = True, require_satellite: bool = False) -> None:
    """Validate an xarray ``Dataset`` against the datacube contract.

    Parameters
    ----------
    ds:
        The dataset to check (duck-typed; must expose ``coords``/``data_vars``).
    require_static:
        If ``True``, the terrain variables must be present.
    require_satellite:
        If ``True``, the INSAT variables must be present.
    """
    for dim in ("time", "lat", "lon"):
        if dim not in ds.coords:
            raise SchemaError(f"missing coordinate '{dim}'")

    _coords_match(ds["lat"].values, config.GRID_LAT, "lat")
    _coords_match(ds["lon"].values, config.GRID_LON, "lon")

    if "level" in ds.coords:
        _coords_match(
            ds["level"].values,
            np.asarray(config.PRESSURE_LEVELS_HPA, dtype="float64"),
            "level",
        )

    required: list[VarSpec] = list(SURFACE_VARS) + list(LEVEL_VARS)
    if require_static:
        required += list(STATIC_VARS)
    if require_satellite:
        required += list(SATELLITE_VARS)

    for spec in required:
        if spec.name not in ds.data_vars:
            raise SchemaError(f"missing variable '{spec.name}'")
        got = tuple(ds[spec.name].dims)
        if got != spec.dims:
            raise SchemaError(
                f"variable '{spec.name}' has dims {got}, expected {spec.dims}"
            )


def validate_sample(x: np.ndarray, y: np.ndarray) -> None:
    """Validate one ``(x, y)`` training pair against :data:`INPUT_SHAPE` /
    :data:`TARGET_SHAPE`."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != INPUT_SHAPE:
        raise SchemaError(f"input tensor has shape {x.shape}, expected {INPUT_SHAPE}")
    if y.shape != TARGET_SHAPE:
        raise SchemaError(f"target tensor has shape {y.shape}, expected {TARGET_SHAPE}")
    if np.isnan(x).any():
        raise SchemaError("input tensor contains NaNs")
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    if ymin < 0.0 or ymax > 1.0:
        raise SchemaError(f"target values out of [0, 1] range: [{ymin}, {ymax}]")
