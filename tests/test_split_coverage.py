"""Pins the train/val/test date-range boundaries in ``nowcast.config`` to the
per-hazard coverage they were chosen for.

Cloudburst fires on 8 dates and flash flood on 3 dates in the whole ingested
record (2018-04-01..2018-09-30) -- both are documented and reasoned about at
length next to the constants themselves. That reasoning is only a comment: it
does not stop a future edit (a "round" date, a wider val window, a rebuilt
label store with different thresholds) from silently taking a hazard back to
zero coverage in some split. This test makes the invariant executable instead
of advisory: every (split, hazard) pair must have at least one positive date.

Skipped where the real label store is not built (matches the existing
``DATACUBE_PATH.exists()`` skip pattern in test_features_vectorised.py) -- no
test in the rest of the suite requires real data, and this one is the
exception by design.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcast import config
from nowcast.features.labels import open_labels

_LABELS_PATH = config.PROCESSED_DIR / "labels.zarr"

pytestmark = pytest.mark.skipif(
    not _LABELS_PATH.exists(), reason="real label store not built on this machine"
)


def _positive_dates(labels, date_range: tuple[str, str], hazard: str) -> set[str]:
    """Calendar dates within ``date_range`` where ``hazard`` occurs (lead 0)."""
    times = pd.DatetimeIndex(labels["time"].values)
    start = pd.Timestamp(date_range[0])
    end_exclusive = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    mask = (times >= start) & (times < end_exclusive)
    occ = labels.sel(hazard=hazard).isel(lead=0) >= config.LABEL_OCCURRENCE_THRESHOLD
    hit = np.asarray(occ.isel(time=np.where(mask)[0]).any(dim=["lat", "lon"]).values)
    return set(times[mask][hit].normalize().strftime("%Y-%m-%d"))


@pytest.fixture(scope="module")
def labels():
    return open_labels(_LABELS_PATH)


@pytest.mark.parametrize(
    "split_name,date_range",
    [
        ("train", config.TRAIN_DATE_RANGE),
        ("val", config.VAL_DATE_RANGE),
        ("test", config.TEST_DATE_RANGE),
    ],
)
def test_every_split_has_every_hazard(labels, split_name, date_range):
    for hazard in config.HAZARDS:
        dates = _positive_dates(labels, date_range, hazard)
        assert dates, (
            f"{split_name} ({date_range[0]}..{date_range[1]}) has zero {hazard} "
            f"dates -- that hazard has no held-out signal in this split"
        )


def test_splits_stay_disjoint_and_chronological():
    """A cheap guard against the boundary shift overlapping or reordering."""
    train_end = pd.Timestamp(config.TRAIN_DATE_RANGE[1])
    val_start = pd.Timestamp(config.VAL_DATE_RANGE[0])
    val_end = pd.Timestamp(config.VAL_DATE_RANGE[1])
    test_start = pd.Timestamp(config.TEST_DATE_RANGE[0])
    assert train_end < val_start
    assert val_end < test_start
