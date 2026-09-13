"""Bias-correction between two distributions of the same physical quantity.

Phase 6: this module is a proof of mechanism, not yet the full deployment
bias-correction layer the project's build order calls for (bridging
train-on-IMDAA / infer-on-operational NCUM-GFS-GDAS). Neither IMDAA nor
GFS/NCUM/GDAS data exists on disk yet -- see docs/PHASE_6_FINDINGS.md for
why, and what unblocks the real version of this step.
"""

from __future__ import annotations

from nowcast.bias_correction.quantile_mapping import QuantileMapper, fit_quantile_map

__all__ = ["QuantileMapper", "fit_quantile_map"]
