"""Empirical quantile mapping (CDF matching) between two distributions."""

from __future__ import annotations

import numpy as np


class QuantileMapper:
    """Map values from a source distribution onto a target distribution's quantiles.

    Built by :func:`fit_quantile_map`. Each new source value is mapped to the
    target-distribution value at the same percentile rank -- e.g. a value at
    the source's 90th percentile becomes the target's 90th-percentile value.
    """

    def __init__(self, source_quantiles: np.ndarray, target_quantiles: np.ndarray) -> None:
        self.source_quantiles = source_quantiles
        self.target_quantiles = target_quantiles

    def apply(self, values) -> np.ndarray:
        """Map `values` (any shape) through the fitted quantile correspondence.

        Values outside the fitted range are clamped to the nearest fitted
        quantile's target value (``numpy.interp``'s default behaviour), not
        extrapolated -- a value more extreme than anything seen while fitting
        should not be corrected by extrapolating a monotone curve past where
        it was ever validated.
        """
        arr = np.asarray(values, dtype=float)
        return np.interp(arr, self.source_quantiles, self.target_quantiles).reshape(arr.shape)


def fit_quantile_map(source_samples, target_samples, n_quantiles: int = 100) -> QuantileMapper:
    """Fit a :class:`QuantileMapper` correcting `source_samples` onto `target_samples`.

    Both are flattened to 1-D; they need not be the same length or paired in
    time -- only their marginal distributions are compared. `n_quantiles`
    percentile points (0..100) approximate each distribution's CDF; 100 is
    enough for a smooth map without being sensitive to individual outliers
    the way using every raw sample as a quantile point would be.
    """
    source = np.asarray(source_samples, dtype=float).reshape(-1)
    target = np.asarray(target_samples, dtype=float).reshape(-1)
    percentiles = np.linspace(0, 100, n_quantiles)
    source_q = np.percentile(source, percentiles)
    target_q = np.percentile(target, percentiles)
    return QuantileMapper(source_q, target_q)
