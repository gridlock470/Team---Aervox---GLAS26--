"""Merge multiple precipitation estimates into one ``precip`` field.

Priority order (highest first): MERA, then IMERG, then INSAT QPE. For every
grid cell / time step the first source with a finite value wins; gaps fall
through to the next source. All inputs must already be on the target grid and
in ``mm h-1`` (use the source loaders in this package).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = ["merge_precip"]


def _as_precip_da(obj: xr.Dataset | xr.DataArray | None) -> xr.DataArray | None:
    if obj is None:
        return None
    if isinstance(obj, xr.Dataset):
        if "precip" not in obj:
            raise KeyError("dataset passed to merge_precip has no 'precip' variable")
        return obj["precip"]
    return obj


def merge_precip(
    mera: xr.Dataset | xr.DataArray | None = None,
    imerg: xr.Dataset | xr.DataArray | None = None,
    insat_qpe: xr.Dataset | xr.DataArray | None = None,
) -> xr.DataArray:
    """Combine precip sources, preferring MERA > IMERG > INSAT QPE.

    Parameters
    ----------
    mera, imerg, insat_qpe:
        Datasets (with a ``precip`` variable) or bare DataArrays on the target
        grid. Any subset may be ``None``.

    Returns
    -------
    xarray.DataArray
        Name ``precip``, units ``mm h-1``. Cells left unfilled by every source
        are ``0.0``.
    """
    ordered = [
        _as_precip_da(mera),
        _as_precip_da(imerg),
        _as_precip_da(insat_qpe),
    ]
    ordered = [da for da in ordered if da is not None]
    if not ordered:
        raise ValueError("merge_precip needs at least one of mera / imerg / insat_qpe")

    merged = ordered[0].astype("float32")
    for nxt in ordered[1:]:
        nxt = nxt.astype("float32")
        merged, nxt = xr.align(merged, nxt, join="outer")
        merged = xr.where(np.isfinite(merged), merged, nxt)

    merged = xr.where(np.isfinite(merged), merged, 0.0).clip(min=0.0)
    merged = merged.astype("float32")
    merged.name = "precip"
    merged.attrs.update(units="mm h-1", long_name="merged precipitation rate")
    merged.attrs["merge_priority"] = "MERA>IMERG>INSAT_QPE"
    return merged
