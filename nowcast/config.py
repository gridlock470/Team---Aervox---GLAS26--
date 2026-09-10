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
DATA_DIR: Path = PROJECT_ROOT / "data"
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

# Gaussian smoothing sigma (grid cells) applied to binary occurrence -> prob map.
LABEL_SMOOTH_SIGMA: float = 1.0

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 1234

# Train / val / test split by calendar year.
TRAIN_YEARS: tuple[int, ...] = (2018, 2019)
VAL_YEARS: tuple[int, ...] = (2020,)
TEST_YEARS: tuple[int, ...] = (2020,)  # held-out event windows carved out in code
