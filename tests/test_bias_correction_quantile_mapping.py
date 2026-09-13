"""Tests for empirical quantile mapping (nowcast/bias_correction)."""

from __future__ import annotations

import numpy as np

from nowcast.bias_correction import fit_quantile_map


def _shifted_case(n: int = 4000, seed: int = 0):
    """A `source` distribution shifted and scaled relative to `target`."""
    rng = np.random.default_rng(seed)
    target = rng.gamma(shape=2.0, scale=1.0, size=n)  # right-skewed, precip-like
    source = target * 1.8 + 3.0  # systematically biased -- too high, too spread out
    return source, target


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Max gap between two empirical CDFs, evaluated at their pooled sample points."""
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), grid, side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def test_quantile_map_matches_target_distribution_better_than_uncorrected():
    source, target = _shifted_case()
    mapper = fit_quantile_map(source, target)
    corrected = mapper.apply(source)

    ks_before = _ks_statistic(source, target)
    ks_after = _ks_statistic(corrected, target)
    assert ks_after < ks_before
    assert ks_after < 0.05  # near-exact match: same samples used to fit and apply


def test_quantile_map_preserves_shape_and_generalises_to_new_samples():
    source, target = _shifted_case(seed=1)
    mapper = fit_quantile_map(source, target)

    held_out_source, held_out_target = _shifted_case(n=1000, seed=2)
    corrected = mapper.apply(held_out_source)
    assert corrected.shape == held_out_source.shape

    ks_before = _ks_statistic(held_out_source, held_out_target)
    ks_after = _ks_statistic(corrected, held_out_target)
    assert ks_after < ks_before


def test_quantile_map_clamps_rather_than_extrapolates():
    source = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    target = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    mapper = fit_quantile_map(source, target, n_quantiles=5)

    below = mapper.apply(np.array([-100.0]))
    above = mapper.apply(np.array([1e6]))
    assert below[0] == mapper.target_quantiles.min()
    assert above[0] == mapper.target_quantiles.max()
