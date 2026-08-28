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
def _resolve_data_root() -> Path:
    if "CPV_DATA_ROOT" in os.environ:
        return Path(os.environ["CPV_DATA_ROOT"])
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for candidate in list(kaggle_input.glob("**/train")):
            if (candidate / "images").is_dir():
                return candidate.parent
        for candidate in list(kaggle_input.glob("*tinyperson*")) + list(kaggle_input.glob("*sod*")):
            if (candidate / "train/images").is_dir():
                return candidate
    for local_root in [
        Path(r"C:\Users\ADMIN\_Project\tiny-object-detection\data"),
        Path(r"C:\Users\ADMIN\OneDrive\Documents\_Project\tiny-object-detection\data"),
        Path("/home/ttung05/Desktop/CPV/Tiny_Object_Detection/data/TOD"),
    ]:
        if local_root.exists():
            return local_root
    return Path("/kaggle/input/datasets/kurt54/sod-tinypeopleinsea")

DATA_ROOT = _resolve_data_root()
TRAIN_DIR = DATA_ROOT / "train"
VALID_DIR = DATA_ROOT / "valid" if (DATA_ROOT / "valid").exists() else DATA_ROOT / "validation"
TEST_DIR  = DATA_ROOT / "test"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATCH_ROOT = PROJECT_ROOT / "data" / "patches"

NUM_CLASSES = 2
CLASS_NAMES = {0: "dry", 1: "wet"}  # NOTE: TOD uses dry-person, wet-swimmer
CLASS_DISPLAY = {0: "dry-person", 1: "wet-swimmer"}

# =============================================================================
# TRAINING SCHEDULE  (giống nhau cho mọi metric)
# =============================================================================
EPOCHS              = int(os.environ.get("TOD_EPOCHS", "20"))
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
BATCH_SIZE       = int(os.environ.get("TOD_BATCH_SIZE", "4"))
NUM_WORKERS      = int(os.environ.get("TOD_NUM_WORKERS", "2"))   # Windows can set 0 after eval slowdowns
TILE_SIZE        = 512
TILE_OVERLAP     = 64
MIN_SIZE         = 640
MAX_SIZE         = 800
CACHE_IMAGES     = False   # True sẽ preload 7.3GB RAM, gây MemoryError trên Windows

# =============================================================================
# TINY OBJECT
# =============================================================================
TINY_THRESHOLD_PX     = 16.0
TINY_AREA_THR         = TINY_THRESHOLD_PX ** 2
TINY_TILE_OVERSAMPLE  = float(os.environ.get("TOD_TINY_TILE_OVERSAMPLE", "2.0"))
USE_COPY_PASTE        = os.environ.get("TOD_USE_COPY_PASTE", "1").lower() not in ("0", "false", "no")
COPY_PASTE_PROB       = float(os.environ.get("TOD_COPY_PASTE_PROB", "0.30"))
COPY_PASTE_MAX_PER    = int(os.environ.get("TOD_COPY_PASTE_MAX_PER", "3"))
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
NWD_C = 12.0          # NWD normalization constant (≈ median sqrt-area ≈ 11.5)

# ALW-specific (shared by alw_full and variants)
ALW_SHAPE_LAMBDA_MIN    = 0.15
ALW_SHAPE_LAMBDA_POWER  = 1.5
ALW_CHARBONNIER_EPS_MIN = 1e-3
ALW_CHARBONNIER_EPS_MAX = 0.35

# SA-ALW: Scale-Adaptive parameters (Phase 2.7-2.8)
SA_ALW_BETA_MIN    = 8.0   # β cho object lớn nhất
SA_ALW_BETA_MAX    = 10.0  # β cho object nhỏ nhất
SA_ALW_S_MIN       = 5.6   # P10 từ Phase 0
SA_ALW_S_MAX       = 28.7  # P90 từ Phase 0
SA_ALW_POS_WEIGHT_MIN = 1.0   # w_pos cho object lớn
SA_ALW_POS_WEIGHT_MAX = 1.5   # w_pos cho object siêu nhỏ
SA_ALW_LOG_CLAMP      = 3.0   # clamp cho log-ratio, cần ablation H2.4

# =============================================================================
# BOX REGRESSION LOSS (decoupled assignment–regression breakthrough)
# =============================================================================
BOX_LOSS_TYPE = "metric"
BOX_LOSS_METRIC_WEIGHT = 0.25   # auxiliary weight for (1-sim) when using ciou/diou/smooth_l1
BOX_LOSS_WARMUP_EPOCHS = 3      # pure metric loss for first N epochs, then ramp new loss
CBL_ALPHA = 5.0                  # normalized RoI delta range [-alpha, alpha]
CBL_NUM_BINS = 6                # paper R-CNN setting: grid number n=5 -> n+1 logits
CBL_GRID_BETA = 1.0             # interval-nonuniform density around zero
CBL_UM_WEIGHT = 1.0             # entropy-matching uncertainty loss weight

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
USE_EMA   = os.environ.get("TOD_USE_EMA", "1").lower() not in ("0", "false", "no")
EMA_DECAY = 0.9998

EVAL_EVERY        = 1
EMPTY_CACHE_EVERY = int(os.environ.get("TOD_EMPTY_CACHE_EVERY", "0"))

# =============================================================================
# SPEED OPTIMIZATIONS (opt-in; default off for reproducibility)
# =============================================================================
USE_TORCH_COMPILE = False
USE_CHANNELS_LAST = False
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
