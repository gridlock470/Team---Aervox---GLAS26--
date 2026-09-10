"""Raw-variable name maps and unit conversions for every ingest source.

Each source publishes its own variable names and units. The dicts below map a
raw name to a :data:`nowcast.schema` name plus an affine unit conversion
``value_SI = raw * scale + offset``. Loaders look a variable up here, rename it
and apply the conversion so the datacube is always in the schema units.

Conventions
-----------
* ``scale`` / ``offset`` convert the raw stored value to the schema unit.
* ``accumulated`` marks precip stored as an accumulation (needs a time
  derivative / division by the accumulation window to become ``mm h-1``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "VarMap",
    "IMDAA_SINGLE_LEVEL",
    "IMDAA_PRESSURE_LEVEL",
    "MERA_VARS",
    "IMERG_VARS",
    "INSAT_L1C_VARS",
    "INSAT_QPE_VARS",
    "DEM_VARS",
    "lookup",
]


@dataclass(frozen=True)
class VarMap:
    """One raw -> schema variable mapping with an affine unit conversion."""

    schema_name: str
    scale: float = 1.0
    offset: float = 0.0
    raw_units: str = ""
    accumulated: bool = False
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def convert(self, values):
        """Apply ``values * scale + offset`` (numpy / xarray friendly)."""
        return values * self.scale + self.offset


# ---------------------------------------------------------------------------
# IMDAA (NCMRWF regional reanalysis) -- single level
# ---------------------------------------------------------------------------
# IMDAA GRIB/NetCDF short names. t2m/u10/v10 already SI. APCP is an accumulated
# precip depth in kg m-2 (== mm); the loader divides by the accumulation hours.
IMDAA_SINGLE_LEVEL: dict[str, VarMap] = {
    "TMP_2m": VarMap("t2m", raw_units="K", aliases=("2t", "t2m", "TMP_GDS0_HTGL")),
    "UGRD_10m": VarMap("u10", raw_units="m s-1", aliases=("10u", "u10", "UGRD_GDS0_HTGL")),
    "VGRD_10m": VarMap("v10", raw_units="m s-1", aliases=("10v", "v10", "VGRD_GDS0_HTGL")),
    "PRMSL_msl": VarMap("prmsl", raw_units="Pa", aliases=("msl", "prmsl", "PRMSL_GDS0_MSL")),
    "PWAT_eatm": VarMap(
        "tcwv", raw_units="kg m-2", aliases=("tcwv", "pwat", "PWAT_GDS0_EATM")
    ),
    "APCP_sfc": VarMap(
        "precip",
        raw_units="kg m-2",
        accumulated=True,
        notes="accumulated depth; divide by accumulation window -> mm h-1",
        aliases=("tp", "APCP_GDS0_SFC", "apcp"),
    ),
    "CAPE_sfc": VarMap("cape", raw_units="J kg-1", aliases=("cape", "CAPE_GDS0_SFC")),
    "CIN_sfc": VarMap("cin", raw_units="J kg-1", aliases=("cin", "CIN_GDS0_SFC")),
}

# ---------------------------------------------------------------------------
# IMDAA -- pressure levels
# ---------------------------------------------------------------------------
IMDAA_PRESSURE_LEVEL: dict[str, VarMap] = {
    "TMP_prl": VarMap("t", raw_units="K", aliases=("t", "air_temperature")),
    "RH_prl": VarMap("rh", raw_units="%", aliases=("r", "rh", "relative_humidity")),
    "HGT_prl": VarMap(
        "z",
        scale=9.80665,
        raw_units="gpm",
        notes="IMDAA stores geopotential height (m); schema wants geopotential (m2 s-2)",
        aliases=("gh", "geopotential_height"),
    ),
    "UGRD_prl": VarMap("u", raw_units="m s-1", aliases=("u", "eastward_wind")),
    "VGRD_prl": VarMap("v", raw_units="m s-1", aliases=("v", "northward_wind")),
}

# ---------------------------------------------------------------------------
# MERA (Met Eireann / here: generic reanalysis precip product used as backup)
# ---------------------------------------------------------------------------
MERA_VARS: dict[str, VarMap] = {
    "PRATE": VarMap("precip", scale=3600.0, raw_units="kg m-2 s-1", notes="rate -> mm h-1"),
    "APCP": VarMap(
        "precip",
        raw_units="kg m-2",
        accumulated=True,
        notes="accumulated depth -> divide by window",
    ),
    "tp": VarMap("precip", scale=1000.0, raw_units="m", accumulated=True, notes="m -> mm"),
}

# ---------------------------------------------------------------------------
# GPM IMERG
# ---------------------------------------------------------------------------
IMERG_VARS: dict[str, VarMap] = {
    "precipitation": VarMap("precip", raw_units="mm hr-1", aliases=("precipitationCal",)),
    "precipitationCal": VarMap("precip", raw_units="mm hr-1"),
    "HQprecipitation": VarMap("precip", raw_units="mm hr-1"),
}

# ---------------------------------------------------------------------------
# INSAT-3D / 3DR imager L1C + QPE
# ---------------------------------------------------------------------------
INSAT_L1C_VARS: dict[str, VarMap] = {
    "IMG_TIR1_TEMP": VarMap("ctt", raw_units="K", aliases=("TIR1_TEMP", "tir1_bt")),
    "IMG_TIR1": VarMap(
        "ctt",
        raw_units="count",
        notes="raw counts require the file's radiance+BT LUT; prefer *_TEMP",
    ),
    "IMG_WV_TEMP": VarMap("wv_bt", raw_units="K", aliases=("WV_TEMP", "wv_bt")),
    "IMG_WV": VarMap("wv_bt", raw_units="count", notes="raw counts; prefer *_TEMP"),
}

INSAT_QPE_VARS: dict[str, VarMap] = {
    "HEM_GPI": VarMap("precip", raw_units="mm hr-1", aliases=("IMR", "rain_rate")),
    "RainRate": VarMap("precip", raw_units="mm hr-1"),
    "precipitation": VarMap("precip", raw_units="mm hr-1"),
}

# ---------------------------------------------------------------------------
# Copernicus GLO-30 DEM
# ---------------------------------------------------------------------------
DEM_VARS: dict[str, VarMap] = {
    "elevation": VarMap("elevation", raw_units="m"),
    "band_data": VarMap("elevation", raw_units="m"),
    "Band1": VarMap("elevation", raw_units="m"),
}


def lookup(table: dict[str, VarMap], raw_name: str) -> VarMap | None:
    """Return the :class:`VarMap` for ``raw_name`` in ``table``.

    Matches the dict key first, then any entry whose ``aliases`` contains the
    name (case-insensitive).
    """
    if raw_name in table:
        return table[raw_name]
    low = raw_name.lower()
    for key, vm in table.items():
        if key.lower() == low or low in {a.lower() for a in vm.aliases}:
            return vm
    return None
