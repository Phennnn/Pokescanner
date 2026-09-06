"""Central configuration for PokeScanner.

Every module reads paths and constants from here so there is exactly one place
to change when the model, the image size or the folder layout moves.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR       = ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
IMAGES_DIR     = DATA_DIR / "images"
BACKGROUNDS_DIR = DATA_DIR / "backgrounds"

LABEL_MAP_PATH = PROCESSED_DIR / "label_map.json"
STATS_CSV      = RAW_DIR / "pokemon_stats.csv"
TYPES_CSV      = RAW_DIR / "pokemon_types.csv"

WEIGHTS_DIR    = ROOT / "model" / "weights"

# ── Normalisation (ImageNet) ──────────────────────────────────────────────────
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# ── Model registry ────────────────────────────────────────────────────────────
# arch key -> (timm model name, native input size)
ARCHITECTURES = {
    "efficientnet_b0": ("efficientnet_b0", 224),
    "efficientnet_b2": ("efficientnet_b2", 260),
    "efficientnet_b3": ("efficientnet_b3", 300),
    "convnext_tiny":   ("convnext_tiny",   224),
    "convnext_small":  ("convnext_small",  224),
}

DEFAULT_ARCH = "efficientnet_b2"

# Checkpoints are looked up in this order when no explicit path is given.
WEIGHT_CANDIDATES = (
    "best_model_convnext_tiny.pth",
    "best_model_convnext_small.pth",
    "best_model_b2.pth",
    "best_model_efficientnet_b2.pth",
    "best_model.pth",
)


def default_weights() -> Path:
    """The checkpoint to load: $POKESCANNER_WEIGHTS wins, else first found.

    POKESCANNER_WEIGHTS takes either a bare filename (looked up in
    model/weights/) or a full path, so a newly trained model can be tried
    in any app without editing code.
    """
    override = os.environ.get("POKESCANNER_WEIGHTS")
    if override:
        p = Path(override)
        return p if p.is_absolute() or p.exists() else WEIGHTS_DIR / override
    for name in WEIGHT_CANDIDATES:
        p = WEIGHTS_DIR / name
        if p.exists():
            return p
    return WEIGHTS_DIR / "best_model_b2.pth"


# ── Inference defaults ────────────────────────────────────────────────────────
TOP_K = 5

# Softmax temperature. >1 softens over-confident predictions, <1 sharpens.
# The model was trained with label smoothing so raw probabilities read high;
# 1.0 keeps the original behaviour and is overridable per-app.
TEMPERATURE = float(os.environ.get("POKESCANNER_TEMPERATURE", "1.0"))

# Below this top-1 probability the scanner reports "unsure" rather than a name.
UNSURE_THRESHOLD = float(os.environ.get("POKESCANNER_UNSURE", "0.18"))

# Isolate the subject from its background before classifying. This is the
# single biggest win against the sprite -> real-photo domain gap.
ISOLATE_SUBJECT = os.environ.get("POKESCANNER_ISOLATE", "1") != "0"

# Background colour the subject is composited onto (matches training, which
# pasted RGBA sprites onto white).
CANVAS_COLOR = (255, 255, 255)

# Fraction of the square canvas the subject should occupy after cropping.
# Swept over both clean sprites and synthetic scenes (tools/benchmark.py): the
# curve is flat from 0.85 to 1.0 and peaks at 0.95 on both, so this keeps a
# little padding as insurance against a slightly tight bounding box.
SUBJECT_FILL = 0.95
