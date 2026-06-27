"""Shared configuration for all experiments.

To ensure FAIR comparison across metrics, every experiment imports
the same hyperparameters from this file.
"""
from __future__ import annotations
import os
import random
from pathlib import Path

import numpy as np
import torch

# =============================================================================
# PATHS
# =============================================================================
# User can override by setting env var CPV_DATA_ROOT
DATA_ROOT = Path(os.environ.get(
    "CPV_DATA_ROOT",
    "/kaggle/input/datasets/kurt54/sod-tinypeopleinsea"
))
TRAIN_DIR = DATA_ROOT / "train"
VALID_DIR = DATA_ROOT / "valid"
TEST_DIR  = DATA_ROOT / "test"

# When running locally, fallback to local TOD data
LOCAL_TOD_ROOT = Path("/home/ttung05/Desktop/CPV/Tiny_Object_Detection/data/TOD")
if not TRAIN_DIR.exists() and LOCAL_TOD_ROOT.exists():
    DATA_ROOT = LOCAL_TOD_ROOT
    TRAIN_DIR = DATA_ROOT / "train"
    VALID_DIR = DATA_ROOT / "valid"
    TEST_DIR  = DATA_ROOT / "test"

NUM_CLASSES = 2
CLASS_NAMES = {0: "dry", 1: "wet"}  # NOTE: TOD uses dry-person, wet-swimmer
# Map for compatibility
CLASS_DISPLAY = {0: "dry-person", 1: "wet-swimmer"}

# =============================================================================
# TRAINING SCHEDULE  (giống nhau cho mọi metric)
# =============================================================================
EPOCHS              = 20
LR                  = 0.005
MOMENTUM            = 0.9
WEIGHT_DECAY        = 1e-4
WARMUP_EPOCHS       = 2
WARMUP_START_LR     = 1e-4
LR_SCHEDULER        = "cosine"
LR_STEPS            = [14, 18]   # dùng nếu LR_SCHEDULER="multistep"
LR_GAMMA            = 0.1
EARLY_STOP_PATIENCE = 5
BEST_METRIC         = "mAP_50"
BEST_MODE           = "max"
MIN_DELTA           = 1e-4

# =============================================================================
# DATA / TILING
# =============================================================================
BATCH_SIZE       = 4
NUM_WORKERS      = 4
TILE_SIZE        = 512
TILE_OVERLAP     = 64
MIN_SIZE         = 640
MAX_SIZE         = 800
CACHE_IMAGES     = True

# =============================================================================
# TINY OBJECT
# =============================================================================
TINY_THRESHOLD_PX     = 16.0
TINY_AREA_THR         = TINY_THRESHOLD_PX ** 2
TINY_TILE_OVERSAMPLE  = 2.0
USE_COPY_PASTE        = True
COPY_PASTE_PROB       = 0.30
COPY_PASTE_MAX_PER    = 3
COPY_PASTE_SCALE_JIT  = (0.8, 1.2)

# =============================================================================
# RFLA
# =============================================================================
RFLA_K    = 3
RFLA_BETA = 0.9
RFLA_DYNAMIC_K_MICRO = 6
RFLA_DYNAMIC_K_TINY  = 5
RFLA_DYNAMIC_K_SMALL = 4
RFLA_DYNAMIC_K_LARGE = 3
RFLA_QUALITY_RATIO   = 0.60
RFLA_MIN_SIM         = 1e-6

# =============================================================================
# METRIC HYPERPARAMS (chung cho nhiều metric)
# =============================================================================
METRIC_BETA = 8.0     # β trong exp(-β·d), giống paper IGWD

# ALW-specific (shared by alw_full and variants)
ALW_SHAPE_LAMBDA_MIN    = 0.15
ALW_SHAPE_LAMBDA_POWER  = 1.5
ALW_CHARBONNIER_EPS_MIN = 1e-3
ALW_CHARBONNIER_EPS_MAX = 0.35
# ALW_SHAPE_RELIABILITY_THR sẽ được compute từ dataset (adaptive P25)

# =============================================================================
# RPN / RoI
# =============================================================================
RPN_NUM_PROPOSALS_TRAIN = 3000
RPN_NUM_PROPOSALS_TEST  = 1500
RPN_NMS_THRESH          = 0.7
RPN_FG_IOU              = 0.5
RPN_BG_IOU              = 0.4
ROI_FG_IOU_THRESH       = 0.5
ROI_BG_IOU_THRESH       = 0.5
BOX_DETECTIONS_PER_IMG  = 200

SCORE_THRESH_TRAIN = 0.05
SCORE_THRESH_TEST  = 0.30
NMS_THRESH_TEST    = 0.5
NMS_METRIC_THRESH  = 0.5  # ngưỡng metric cho NMS (lower = stricter)

# =============================================================================
# EMA / CHECKPOINT
# =============================================================================
USE_EMA   = True
EMA_DECAY = 0.9998

EVAL_EVERY        = 1
EMPTY_CACHE_EVERY = 50

# =============================================================================
# SPEED OPTIMIZATIONS (opt-in; default off for reproducibility)
# =============================================================================
# Enable on PyTorch 2.0+ with Triton for ~1.3-1.8x speedup.
# NOTE: First epoch will be slow due to compilation; subsequent epochs benefit.
USE_TORCH_COMPILE = False
# Convert backbone to channels_last memory format for ~5-10% backbone speedup.
# Caller must still pass image tensors in channels_last format.
USE_CHANNELS_LAST = False
# DataLoader prefetch_factor (number of batches each worker preloads).
# Higher = more GPU saturation but more RAM. Default PyTorch = 2.
PREFETCH_FACTOR = 4

# =============================================================================
# DEVICE / SEED
# =============================================================================
SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def seed_all(s: int = SEED) -> None:
    """Set all seeds for reproducibility."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def make_output_dir(metric: str, placement: str, seed: int) -> Path:
    """Build consistent output directory."""
    name = f"{metric}__{placement}__seed{seed}"
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)
    return out