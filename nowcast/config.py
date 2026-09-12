"""Global configuration — the single source of truth for the nowcast project.

Pilot region, target analysis grid, time window, vertical levels, hazard
definitions and on-disk paths live here and *only* here. Every module under
``nowcast`` imports these values; nothing re-defines them locally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "DataSet"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
CATALOG_DIR: Path = DATA_DIR / "catalog"

RAW_IMDAA_DIR: Path = RAW_DIR / "imdaa"
RAW_MERA_DIR: Path = RAW_DIR / "mera"
RAW_IMERG_DIR: Path = RAW_DIR / "imerg"
RAW_INSAT3D_DIR: Path = RAW_DIR / "insat3d"
RAW_INSAT3DR_DIR: Path = RAW_DIR / "insat3dr"
RAW_DEM_DIR: Path = RAW_DIR / "dem"

DATACUBE_PATH: Path = PROCESSED_DIR / "datacube.zarr"
DEM_ROUTING_PATH: Path = PROCESSED_DIR / "dem_routing.zarr"
NORM_STATS_PATH: Path = PROCESSED_DIR / "norm_stats.json"

# ---------------------------------------------------------------------------
# Pilot region — Uttarakhand + Delhi NCR
# ---------------------------------------------------------------------------
# Exact analysis bounding box (decimal degrees).
BBOX_NORTH: float = 31.5
BBOX_SOUTH: float = 27.8
BBOX_WEST: float = 76.3
BBOX_EAST: float = 81.1

# The NCMRWF RDS "Select Coordinates" form accepts WHOLE NUMBERS ONLY. Order
# rounded OUTWARD, then crop back to the exact bbox in the ingest pipeline.
BBOX_ORDER_NORTH: int = 32
BBOX_ORDER_SOUTH: int = 27
BBOX_ORDER_WEST: int = 76
BBOX_ORDER_EAST: int = 82

# ---------------------------------------------------------------------------
# Target grid — common grid for the datacube and every derived product
# ---------------------------------------------------------------------------
GRID_RESOLUTION_DEG: float = 0.1
GRID_CRS: str = "EPSG:4326"

# Cell-centre coordinates, strictly ascending.
GRID_LAT: np.ndarray = np.round(
    np.arange(BBOX_SOUTH, BBOX_NORTH + 1e-9, GRID_RESOLUTION_DEG), 4
)
GRID_LON: np.ndarray = np.round(
    np.arange(BBOX_WEST, BBOX_EAST + 1e-9, GRID_RESOLUTION_DEG), 4
)
GRID_SHAPE: tuple[int, int] = (len(GRID_LAT), len(GRID_LON))  # (H, W) == (38, 49)

# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
TIME_START: str = "2018-01-01"
TIME_END: str = "2020-12-31"
TIMESTEP: str = "1h"  # datacube temporal resolution (pandas offset alias)

# Pressure levels ordered surface -> top, hPa.
PRESSURE_LEVELS_HPA: tuple[int, ...] = (1000, 925, 850, 700, 500, 300, 250)

# ---------------------------------------------------------------------------
# Model task definition
# ---------------------------------------------------------------------------
# Multi-task: one shared backbone -> one probability-map head per hazard.
HAZARDS: tuple[str, ...] = ("thunderstorm", "cloudburst", "flash_flood")

# Forecast lead times (hours ahead) that each head predicts.
LEAD_TIMES_H: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

# Hours of history fed to the model as the input sequence.
INPUT_SEQ_LEN: int = 12

# ---------------------------------------------------------------------------
# Label thresholds (used by nowcast.features.labels to build targets)
# ---------------------------------------------------------------------------
# Precip-rate thresholds in mm/h unless noted. Tuned for data density on the
# 0.1 deg grid, not operational IMD criteria — documented in DATA_CONTRACT.md.
THUNDERSTORM_PRECIP_MM_H: float = 10.0
THUNDERSTORM_CTT_MAX_K: float = 241.0
CLOUDBURST_PRECIP_MM_H: float = 50.0
CLOUDBURST_ACCUM_MM_2H: float = 60.0
FLASH_FLOOD_UPSTREAM_ACCUM_MM: float = 80.0  # rain routed over the drainage network

# Spatial Gaussian smoothing sigma (grid cells) applied to the occurrence mask.
# The smoothed field is PEAK-NORMALISED in features.labels so an isolated event
# still reaches 1.0 (see LABEL_OCCURRENCE_THRESHOLD).
LABEL_SMOOTH_SIGMA: float = 1.0

# A grid cell counts as a positive occurrence (for CSI/POD/FAR, the LightGBM
# baseline, and calibration) when its (peak-normalised) label >= this value.
LABEL_OCCURRENCE_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Rare-event loss weighting (consumed by nowcast.training.losses)
# ---------------------------------------------------------------------------
# Positive cells are ~1e-4 to 1e-5 of all cell-hours, so an unweighted BCE is
# minimised by predicting "no event" everywhere. The three focal constants and
# the BCE positive weight below counteract that; they are defaults only — the
# LightningModule may override them from its YAML config.

# Weight of the focal term inside MultiTaskLoss (alongside bce=1.0, dice=0.5).
# With alpha=0.25 and gamma=2.0 the focal term is roughly an order of magnitude
# smaller than plain BCE on the same batch, so a unit weight makes it a real
# contributor to the gradient without letting it dominate BCE or Dice.
FOCAL_WEIGHT: float = 1.0

# Focal alpha: static weight given to positive cells (1 - alpha goes to
# negatives). 0.25 is the Lin et al. (2017) value; it is deliberately small
# because gamma already supplies most of the rare-class emphasis, and a larger
# alpha on a 1e-4 base rate over-forecasts badly.
FOCAL_ALPHA: float = 0.25

# Focal gamma: exponent that down-weights easy, confidently-correct cells by
# (1 - p_t) ** gamma. 2.0 is the Lin et al. (2017) value and cuts the loss of a
# p_t = 0.9 easy negative by 100x, which is exactly the regime a field of
# almost-all-negative cells lives in.
FOCAL_GAMMA: float = 2.0

# pos_weight for masked_bce_with_logits: multiplier on the positive term of the
# BCE. Full inverse frequency would be ~1e4 and makes the gradient explode and
# the model over-forecast; 20 is a capped value that lifts the positive-cell
# share of the gradient to a trainable level while keeping the output close
# enough to calibrated for the post-hoc calibration stage to fix the rest.
BCE_POS_WEIGHT: float = 20.0

# ---------------------------------------------------------------------------
# Event-balanced window sampling (consumed by nowcast.data.datamodule)
# ---------------------------------------------------------------------------
# The focal/pos_weight constants above fix the imbalance *within* a window. The
# two below fix the imbalance *between* windows: with events in ~1e-4 of all
# cell-hours, a uniformly shuffled loader serves thousands of consecutive
# all-negative windows and the model never receives a positive gradient at all.
# They are complementary — do not trade one off against the other.

# Target share of the TRAIN sampler's probability mass that lands on windows
# containing at least one occurrence. 0.35 leaves ~2/3 of every batch genuinely
# quiet, which the model needs in order to learn when *not* to fire; pushing
# this to 0.5+ balances the batch but produces a model that cries wolf on calm
# air, and no amount of post-hoc calibration recovers the lost specificity.
EVENT_SAMPLER_TARGET_FRACTION: float = 0.35

# Ceiling on how much more often a single event window may be drawn than a
# quiet one. Without it, a split holding three cloudburst days would give each
# of them a ~1e4 weight and the model would simply memorise those three fields.
# 50 is roughly one appearance per batch-of-50 for the rarest window, which is
# enough signal to learn from and too little to overfit to.
EVENT_SAMPLER_MAX_REPLICATION: float = 50.0

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 1234

# Chronological train / val / test split — DISJOINT ISO date ranges, both ends
# inclusive. Model selection uses VAL, final held-out evaluation uses TEST.
# Consumed by data.datamodule and baseline.dataset (filter on the ``time`` coord;
# do NOT split by calendar year — val and test would overlap).
# Chronological, disjoint, and inside the data that actually exists. These
# previously pointed at 2019-2020, which matched nothing on disk: every sample
# fell into train while val and test were empty, so no metric had anything to
# compute on and nothing would have complained.
#
# Available: IMERG 2018-04-01..2018-09-30 complete, verified by date coverage
# rather than file count (2019 was never fetched).
# Split chronologically so no future information reaches training. Train keeps
# July, the most event-dense month; August is event-dense as well, so val and
# test are not starved of positives -- which matters at a 1-in-200 base rate.
#
# Known tradeoff: the 2-3 May 2018 demo outbreak falls in TRAIN, so skill
# quoted on it is in-sample. Move May into TEST if a held-out demo matters
# more than training signal.
TRAIN_DATE_RANGE: tuple[str, str] = ("2018-04-01", "2018-07-31")
VAL_DATE_RANGE: tuple[str, str] = ("2018-08-01", "2018-08-25")
TEST_DATE_RANGE: tuple[str, str] = ("2018-08-26", "2018-09-30")
