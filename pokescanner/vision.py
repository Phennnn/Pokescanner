"""Image preparation for PokeScanner.

The classifier was trained on game sprites: the creature sits centred on a flat
white canvas with generous padding. A webcam frame is the opposite - a small
card or figure somewhere inside a cluttered room, squashed to a square. Most of
the misidentifications this project sees come from that mismatch rather than
from the weights themselves.

This module closes the gap at inference time by rebuilding a sprite-like image
from whatever the camera saw:

    find the subject -> crop it -> composite it centred on white -> letterbox

Nothing here needs a GPU and nothing here needs a retrain.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import config

BBox = Tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)


# -- Small helpers ------------------------------------------------------------
def to_pil(image) -> Image.Image:
    """Accept a PIL image, an HxWx3 RGB array or an HxWx4 RGBA array."""
    if isinstance(image, Image.Image):
        return image
    arr = np.asarray(image)
    if arr.ndim == 2:
        return Image.fromarray(arr, "L").convert("RGB")
    if arr.shape[2] == 4:
        return Image.fromarray(arr, "RGBA")
    return Image.fromarray(arr[:, :, :3], "RGB")


def flatten_alpha(img: Image.Image, color=config.CANVAS_COLOR) -> Image.Image:
    """RGBA -> RGB composited onto a flat colour (matches the training paste)."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, color)
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        return canvas
    return img.convert("RGB")


def letterbox(img: Image.Image, size: int, color=config.CANVAS_COLOR) -> Image.Image:
    """Resize preserving aspect ratio, then pad to a square.

    The old pipeline called Resize((S, S)), which stretches a 4:3 webcam frame
    into a square and distorts every proportion the model learned.
    """
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (size, size), color)
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), color)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


# -- Subject detection --------------------------------------------------------
def _alpha_bbox(img: Image.Image, thresh: int = 8) -> Optional[BBox]:
    """Bounding box of the opaque pixels, for sprites that still carry alpha."""
    if img.mode not in ("RGBA", "LA") and not (
        img.mode == "P" and "transparency" in img.info
    ):
        return None
    alpha = np.asarray(img.convert("RGBA").getchannel("A"))
    ys, xs = np.nonzero(alpha > thresh)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _largest_central_component(mask: np.ndarray, center_prior: float = 0.62) -> np.ndarray:
    """Keep the component that best combines size with centrality.

    A plain "largest blob" rule latches onto a bright window or a table edge
    running off-frame. Scoring area x centre-overlap keeps the thing the user
    actually pointed the camera at.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask

    h, w = mask.shape
    cy0, cy1 = int(h * (1 - center_prior) / 2), int(h * (1 + center_prior) / 2)
    cx0, cx1 = int(w * (1 - center_prior) / 2), int(w * (1 + center_prior) / 2)
    center_slice = (slice(cy0, cy1), slice(cx0, cx1))
    center_area = max(1, (cy1 - cy0) * (cx1 - cx0))

    best_label, best_score = 0, -1.0
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 0.002 * h * w:                      # ignore speckle
            continue
        overlap = np.count_nonzero(labels[center_slice] == label) / center_area
        score = area * (0.15 + overlap)               # size, weighted by centrality
        if score > best_score:
            best_label, best_score = label, score

    if best_label == 0:
        return mask
    return (labels == best_label).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Flood the outside and invert - turns an outline into a solid silhouette."""
    h, w = mask.shape
    flood = mask.copy().astype(np.uint8)
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, pad, (0, 0), 1)
    return (mask | (1 - flood)).astype(np.uint8)


def subject_mask(rgb: np.ndarray) -> np.ndarray:
    """Foreground mask for an RGB array, with no model and no extra dependency.

    Two weak cues are combined because either one alone fails often:
      * colour distance from the frame border (walls and desks are uniform)
      * edge density (printed cards and figures are busy, backgrounds are not)
    """
    h, w = rgb.shape[:2]
    scale = 256 / max(h, w)
    if scale < 1:
        small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = rgb.copy()
    sh, sw = small.shape[:2]

    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)

    # 1. distance from the median border colour
    border = max(2, int(min(sh, sw) * 0.06))
    ring = np.concatenate([
        lab[:border].reshape(-1, 3), lab[-border:].reshape(-1, 3),
        lab[:, :border].reshape(-1, 3), lab[:, -border:].reshape(-1, 3),
    ])
    bg_color = np.median(ring, axis=0)
    dist = np.linalg.norm(lab - bg_color, axis=2)
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, color_mask = cv2.threshold(dist_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. edge density
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edge_mask = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=2) > 0

    mask = (color_mask.astype(bool) | edge_mask).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = _fill_holes(mask)
    mask = _largest_central_component(mask)

    if scale < 1:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.uint8)


def _bbox_from_mask(mask: np.ndarray) -> Optional[BBox]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _usable_bbox(box: Optional[BBox], min_side: int = 4) -> bool:
    """Non-degenerate: big enough to be worth cropping to."""
    if box is None:
        return False
    x0, y0, x1, y1 = box
    return (x1 - x0) >= min_side and (y1 - y0) >= min_side


def _sane_bbox(box: Optional[BBox], w: int, h: int) -> bool:
    """Reject boxes covering almost everything or almost nothing."""
    if box is None:
        return False
    x0, y0, x1, y1 = box
    frac = ((x1 - x0) * (y1 - y0)) / float(w * h)
    return 0.01 < frac < 0.97 and (x1 - x0) > 8 and (y1 - y0) > 8


# -- The main entry point -----------------------------------------------------
def isolate_subject(img: Image.Image,
                    fill: float = config.SUBJECT_FILL,
                    remove_background: bool = False,
                    color=config.CANVAS_COLOR) -> Image.Image:
    """Rebuild a sprite-like square image: subject centred on a white canvas.

    ``remove_background=True`` also erases the pixels outside the silhouette,
    which matches the training data most closely but is unforgiving when the
    mask is wrong. Cropping alone is the safer default.
    """
    alpha_box = _alpha_bbox(img)
    rgb_img = flatten_alpha(img, color)
    arr = np.asarray(rgb_img)
    h, w = arr.shape[:2]

    mask = None
    if alpha_box is not None and _usable_bbox(alpha_box):
        # Alpha is ground truth, so it is trusted at any size - only the
        # heuristic mask has to clear the area-fraction sanity check.
        box = alpha_box
    else:
        mask = subject_mask(arr)
        box = _bbox_from_mask(mask)
        if not _sane_bbox(box, w, h):
            # No trustworthy subject: fall back to a plain centre crop, which
            # still beats squashing the whole frame.
            side = int(min(h, w) * 0.9)
            cx, cy = w // 2, h // 2
            box = (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)
            mask = None

    x0, y0, x1, y1 = box
    if remove_background and mask is not None:
        arr = np.where(mask[:, :, None].astype(bool), arr, np.array(color, np.uint8))

    crop = Image.fromarray(arr[y0:y1, x0:x1])

    # Re-pad so the subject occupies `fill` of a square canvas, mirroring the
    # padding ratio of the source sprites.
    cw, ch = crop.size
    side = int(round(max(cw, ch) / max(0.1, min(1.0, fill))))
    canvas = Image.new("RGB", (side, side), color)
    canvas.paste(crop, ((side - cw) // 2, (side - ch) // 2))
    return canvas


def prepare(image, size: int,
            isolate: bool = config.ISOLATE_SUBJECT,
            remove_background: bool = False) -> Image.Image:
    """Full camera-frame -> model-input chain. Returns a size x size RGB image."""
    img = to_pil(image)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if isolate:
        try:
            img = isolate_subject(img, remove_background=remove_background)
        except Exception:
            img = flatten_alpha(img)
    else:
        img = flatten_alpha(img)
    return letterbox(img, size)
