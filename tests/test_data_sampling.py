"""Tests for the event-balanced window sampler (storm catalogue + train weights).

Positives are ~1e-4 of all cell-hours, so a uniformly shuffled loader serves
thousands of consecutive all-negative windows. These tests pin the *between*-
window fix: the per-window storm catalogue, the per-hazard inverse-frequency
weights, and the fact that only the train loader is rebalanced.

Fixtures blank ``synthetic.make_targets`` and inject occurrences at chosen
analysis times, so every expected catalogue entry is known exactly rather than
inferred from random label noise.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import (
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)

from nowcast import config
from nowcast.data.datamodule import NowcastDataModule, NowcastDataset
from nowcast.testing import synthetic

_SEQ = config.INPUT_SEQ_LEN
_MAX_LEAD = max(config.LEAD_TIMES_H)
_N_HOURS = 60
_THUNDER = list(config.HAZARDS).index("thunderstorm")
_CLOUDBURST = list(config.HAZARDS).index("cloudburst")


def _blank_labels(n_hours: int = _N_HOURS):
    """Schema-valid targets with every cell forced to zero (no occurrence)."""
    labels = synthetic.make_targets(n_hours=n_hours, seed=0)
    labels.values[:] = 0.0
    return labels


def _inject(labels, analysis_time: int, hazard: int) -> None:
    """Put one unambiguous occurrence in the label field at ``analysis_time``."""
    labels.values[analysis_time, hazard, 0, 5:8, 5:8] = 1.0


def _dataset(labels, n_hours: int = _N_HOURS) -> NowcastDataset:
    features = synthetic.make_feature_datacube(n_hours=n_hours, seed=0)
    return NowcastDataset(features, labels)


def _valid_analysis_times(n_hours: int = _N_HOURS) -> np.ndarray:
    """Analysis times that survive the F19 label-horizon filter."""
    return np.arange(_SEQ - 1, n_hours - _MAX_LEAD)


def _datamodule(dataset: NowcastDataset, **kwargs) -> NowcastDataModule:
    return NowcastDataModule(
        batch_size=1,
        train_dataset=dataset,
        val_dataset=dataset,
        test_dataset=dataset,
        **kwargs,
    )


def test_catalogue_flags_exactly_the_windows_holding_a_positive():
    labels = _blank_labels()
    injected = [_SEQ - 1, 20, 35]
    for t in injected:
        _inject(labels, t, _THUNDER)
    dataset = _dataset(labels)

    assert dataset.event_flags.shape == (len(dataset), len(config.HAZARDS))

    # Brute force: recompute the flag straight from the label field that
    # __getitem__ would serve for this window.
    expected = np.zeros(len(dataset), dtype=bool)
    for i in range(len(dataset)):
        analysis = int(dataset._window_starts[i]) + _SEQ - 1
        served = dataset.labels.isel(time=analysis).values
        expected[i] = bool((served >= config.LABEL_OCCURRENCE_THRESHOLD).any())

    assert np.array_equal(dataset.has_event(), expected)
    assert int(expected.sum()) == len(injected)
    # Only the injected hazard is catalogued, not the other two.
    assert dataset.event_flags[:, _THUNDER].sum() == len(injected)
    assert dataset.event_flags[:, _CLOUDBURST].sum() == 0


def test_rarer_hazard_windows_outweigh_common_hazard_windows():
    labels = _blank_labels()
    valid = _valid_analysis_times()
    thunder_times = valid[:8]
    cloudburst_time = int(valid[20])
    for t in thunder_times:
        _inject(labels, int(t), _THUNDER)
    _inject(labels, cloudburst_time, _CLOUDBURST)
    dataset = _dataset(labels)

    weights = dataset.sample_weights()
    thunder_w = weights[dataset.event_flags[:, _THUNDER]]
    cloudburst_w = weights[dataset.event_flags[:, _CLOUDBURST]]

    assert thunder_w.size == thunder_times.size
    assert cloudburst_w.size == 1
    # Cloudburst is 8x rarer here, so it must be strictly -- not marginally --
    # heavier. An "any event" flag would give both the identical weight.
    assert cloudburst_w.min() > thunder_w.max()
    assert cloudburst_w.min() >= 4.0 * thunder_w.max()
    assert weights.max() <= config.EVENT_SAMPLER_MAX_REPLICATION


def test_quiet_windows_keep_a_non_zero_floor_weight():
    labels = _blank_labels()
    for t in _valid_analysis_times()[:5]:
        _inject(labels, int(t), _THUNDER)
    dataset = _dataset(labels)

    weights = dataset.sample_weights()
    quiet = ~dataset.has_event()
    assert quiet.any()
    # A model that never sees calm air cries wolf: genuine negatives stay in.
    assert np.all(weights[quiet] == 1.0)
    assert np.all(weights > 0.0)
    assert np.isfinite(weights).all()


def test_train_loader_oversamples_event_windows_relative_to_uniform():
    labels = _blank_labels()
    for t in _valid_analysis_times()[:6]:
        _inject(labels, int(t), _THUNDER)
    dataset = _dataset(labels)
    datamodule = _datamodule(dataset)

    sampler = datamodule.train_sampler()
    assert isinstance(sampler, WeightedRandomSampler)

    is_event = dataset.has_event()
    uniform_fraction = float(is_event.mean())

    torch.manual_seed(config.RANDOM_SEED)
    drawn = [idx for _ in range(12) for idx in sampler]
    assert len(drawn) >= 300
    observed = float(is_event[np.asarray(drawn)].mean())

    assert observed > 1.5 * uniform_fraction
    # Near the configured target, but never a balanced 50/50 batch.
    assert abs(observed - config.EVENT_SAMPLER_TARGET_FRACTION) < 0.1
    assert observed < 0.5


def test_val_and_test_loaders_are_not_weighted():
    labels = _blank_labels()
    for t in _valid_analysis_times()[:6]:
        _inject(labels, int(t), _THUNDER)
    datamodule = _datamodule(_dataset(labels))

    assert isinstance(datamodule.train_dataloader().sampler, WeightedRandomSampler)
    for loader in (datamodule.val_dataloader(), datamodule.test_dataloader()):
        # Rebalancing these would make every reported metric describe a climate
        # that does not exist.
        assert not isinstance(loader.sampler, WeightedRandomSampler)
        assert isinstance(loader.sampler, SequentialSampler)


def test_sampling_can_be_disabled_for_the_ablation():
    datamodule = NowcastDataModule.from_synthetic(
        n_hours=90, batch_size=2, event_balanced_sampling=False
    )
    datamodule.setup("fit")
    assert datamodule.train_sampler() is None
    loader = datamodule.train_dataloader()
    assert not isinstance(loader.sampler, WeightedRandomSampler)
    assert isinstance(loader.sampler, RandomSampler)

    on = NowcastDataModule.from_synthetic(n_hours=90, batch_size=2)
    on.setup("fit")
    assert isinstance(on.train_dataloader().sampler, WeightedRandomSampler)


def test_weights_index_window_starts_not_raw_time():
    """F19 guard: a sampler built on unfiltered time indices would silently
    re-admit the windows the label-horizon filter dropped."""
    labels = _blank_labels()
    kept = int(_valid_analysis_times()[3])
    # Past the horizon: this analysis time has no full, in-bounds label and its
    # window is excluded, so it must never reach the catalogue or the weights.
    dropped = _N_HOURS - 2
    assert dropped > _valid_analysis_times()[-1]
    _inject(labels, kept, _THUNDER)
    _inject(labels, dropped, _CLOUDBURST)
    dataset = _dataset(labels)

    weights = dataset.sample_weights()
    assert weights.shape == (len(dataset),)
    assert weights.size == dataset._window_starts.size
    assert weights.size != dataset.n_times  # raw-time indexing would be this long

    flagged = np.flatnonzero(dataset.has_event())
    analysis_times = dataset._window_starts[flagged] + _SEQ - 1
    assert analysis_times.tolist() == [kept]
    assert dataset.event_flags[:, _CLOUDBURST].sum() == 0


def test_degrades_gracefully_when_a_split_has_no_positives():
    dataset = _dataset(_blank_labels())
    assert not dataset.has_event().any()

    weights = dataset.sample_weights()
    assert weights.shape == (len(dataset),)
    assert np.isfinite(weights).all()
    assert np.all(weights == 1.0)  # uniform, no divide-by-zero

    datamodule = _datamodule(dataset)
    sampler = datamodule.train_sampler()
    assert sampler is not None
    assert len(list(sampler)) == len(dataset)
    batch = next(iter(datamodule.train_dataloader()))
    assert len(batch) == 3


def test_degrades_gracefully_when_every_window_is_an_event():
    labels = _blank_labels()
    for t in _valid_analysis_times():
        _inject(labels, int(t), _THUNDER)
    dataset = _dataset(labels)

    assert dataset.has_event().all()
    weights = dataset.sample_weights()
    # No quiet window to weigh events against -- fall back to uniform rather
    # than dividing by a zero negative count.
    assert np.all(weights == 1.0)
