"""Tests for :mod:`nowcast.ingest.names`."""

from __future__ import annotations

import numpy as np

from nowcast import schema
from nowcast.ingest import names


def _schema_names() -> set[str]:
    return {v.name for v in schema.ALL_DATACUBE_VARS}


def test_every_map_targets_a_schema_name():
    valid = _schema_names()
    for table in (
        names.IMDAA_SINGLE_LEVEL,
        names.IMDAA_PRESSURE_LEVEL,
        names.MERA_VARS,
        names.IMERG_VARS,
        names.INSAT_L1C_VARS,
        names.INSAT_QPE_VARS,
        names.DEM_VARS,
    ):
        for vm in table.values():
            assert vm.schema_name in valid


def test_lookup_by_key_and_alias():
    vm = names.lookup(names.IMDAA_SINGLE_LEVEL, "TMP_2m")
    assert vm is not None and vm.schema_name == "t2m"
    assert names.lookup(names.IMDAA_SINGLE_LEVEL, "2t").schema_name == "t2m"
    assert names.lookup(names.IMDAA_SINGLE_LEVEL, "unknown_var") is None


def test_geopotential_height_to_geopotential_conversion():
    vm = names.lookup(names.IMDAA_PRESSURE_LEVEL, "HGT_prl")
    # 100 gpm -> ~981 m2 s-2
    assert np.isclose(vm.convert(np.array([100.0]))[0], 980.665)


def test_apcp_marked_accumulated():
    assert names.lookup(names.IMDAA_SINGLE_LEVEL, "APCP_sfc").accumulated is True
    assert names.lookup(names.IMDAA_SINGLE_LEVEL, "TMP_2m").accumulated is False


def test_mera_rate_scale_to_mm_per_hour():
    vm = names.MERA_VARS["PRATE"]
    assert np.isclose(vm.convert(np.array([1.0e-3]))[0], 3.6)
