"""Read Pokemon trading cards instead of guessing at them.

A card has the species name printed on it in large clean type. Reading that
text beats classifying the artwork: the classifier has to separate 809 lookalike
creatures from a handful of sprites each, while the text says "Charizard".

The pipeline:

    frame -> find the card quad -> flatten it -> OCR -> match against the
    known species list

The last step is what makes this robust. OCR on a glossy, angled, badly lit
card produces things like "Charlzard" or "CHARIZARO". Matching against a fixed
vocabulary of 809 names repairs most of that, because there is usually exactly
one species within a small edit distance.

Card detection is optional. If no card quad is found the whole frame is read
anyway, which still works for a card filling the shot or a name on a screen.
"""

from __future__ import annotations

import difflib
import functools
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from . import dex, vision

# A standard card is 63mm x 88mm. Rectify to that ratio at a readable size.
CARD_W, CARD_H = 440, 614
CARD_RATIO = CARD_W / CARD_H

# Words that appear on cards but are not part of the species name.
_TCG_NOISE = {
    "basic", "stage", "hp", "ex", "gx", "v", "vmax", "vstar", "vunion",
    "lv", "lvx", "prime", "break", "tag", "team", "star", "shining",
    "radiant", "dark", "light", "shadow", "rocket", "team", "delta",
    "species", "owner", "pokemon", "pokmon", "trainer", "energy", "item",
    "supporter", "stadium", "tool", "ancient", "future", "tera",
    "illustration", "rare", "holo", "reverse", "promo", "full", "art",
    "no", "of", "the", "and", "evolves", "from", "put", "onto",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’.-]{1,}")


@dataclass
class TextBox:
    text: str
    confidence: float
    box: np.ndarray = field(repr=False)          # 4x2 corner array

    @property
    def height(self) -> float:
        ys = self.box[:, 1]
        return float(ys.max() - ys.min())

    @property
    def top(self) -> float:
        return float(self.box[:, 1].min())


@dataclass
class CardReading:
    label: Optional[str]                 # matched class label, None if no match
    score: float                         # 0..1 match quality
    raw_text: str                        # the token OCR actually produced
    all_text: List[str] = field(default_factory=list)
    card_found: bool = False
    rectified: Optional[Image.Image] = field(repr=False, default=None)

    @property
    def display_name(self) -> str:
        if not self.label:
            return ""
        return dex.get(self.label)["display_name"]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "display_name": self.display_name,
            "score": round(self.score, 3),
            "raw_text": self.raw_text,
            "card_found": self.card_found,
            "all_text": self.all_text[:20],
        }


# -- OCR backend --------------------------------------------------------------
class _OCR:
    """Thin wrapper so the rest of the module does not care which engine runs.

    RapidOCR is the default: pip-installable, ONNX, no system binary. Tesseract
    is used if it happens to be available instead.
    """

    def __init__(self):
        self._engine = None
        self._kind = None

    def _load(self):
        if self._engine is not None:
            return
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._engine, self._kind = RapidOCR(), "rapidocr"
            return
        except Exception:
            pass
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._engine, self._kind = pytesseract, "tesseract"
            return
        except Exception:
            pass
        self._kind = "none"

    @property
    def available(self) -> bool:
        self._load()
        return self._kind not in (None, "none")

    @property
    def kind(self) -> str:
        self._load()
        return self._kind or "none"

    def read(self, rgb: np.ndarray) -> List[TextBox]:
        self._load()
        if self._kind == "rapidocr":
            # use_cls runs an orientation classifier over every text box and
            # costs ~1.4s per card. Card text is already upright once the card
            # is rectified, so it buys nothing here.
            result, _ = self._engine(rgb, use_cls=False)
            out = []
            for item in (result or []):
                box, text, conf = item[0], item[1], item[2]
                out.append(TextBox(str(text), float(conf),
                                   np.asarray(box, dtype=np.float32)))
            return out
        if self._kind == "tesseract":
            from PIL import Image as _Image
            data = self._engine.image_to_data(
                _Image.fromarray(rgb), output_type=self._engine.Output.DICT)
            out = []
            for i, text in enumerate(data["text"]):
                if not text.strip():
                    continue
                x, y = data["left"][i], data["top"][i]
                w, h = data["width"][i], data["height"][i]
                box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                               dtype=np.float32)
                conf = max(0.0, float(data["conf"][i])) / 100.0
                out.append(TextBox(text.strip(), conf, box))
            return out
        return []


OCR = _OCR()


# -- Vocabulary ---------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _vocabulary() -> dict:
    """{comparable name: label}. Form variants also register their species."""
    vocab: dict[str, str] = {}
    for label in dex.load():
        key = label.replace("-", "")
        vocab.setdefault(key, label)
        # "giratina-altered" should also answer to the printed name "Giratina"
        species = label.split("-")[0]
        if len(species) >= 4:
            vocab.setdefault(species, label)
    return vocab


def _comparable(text: str) -> str:
    """Lowercase, accent-free, letters only. 'Farfetch'd' -> 'farfetchd'."""
    return re.sub(r"[^a-z0-9]", "", dex.strip_accents(str(text)).lower())


def _threshold_for(candidate: str) -> float:
    """Short names need to match more exactly.

    At a fixed ratio, a short word is far easier to hit by accident: the
    3-letter fragment "eee" scores 0.75 against "eevee", which is enough to
    turn OCR noise into a confident wrong answer. Longer names carry enough
    signal that 0.80 is safe.
    """
    return 0.88 if len(candidate) <= 6 else 0.80


def match_species(text: str, min_score: Optional[float] = None
                  ) -> Tuple[Optional[str], float]:
    """Best species label for one OCR token, plus a 0..1 match score."""
    key = _comparable(text)
    # Fewer than four characters is noise, not a name. Every species in the
    # vocabulary is at least five characters long.
    if len(key) < 4:
        return None, 0.0
    vocab = _vocabulary()
    if key in vocab:
        return vocab[key], 1.0

    best_label, best_score = None, 0.0
    for candidate, label in vocab.items():
        # cheap length filter before the expensive ratio
        if abs(len(candidate) - len(key)) > max(2, len(key) // 3):
            continue
        score = difflib.SequenceMatcher(None, key, candidate).ratio()
        if score > best_score:
            best_label, best_score, best_candidate = label, score, candidate

    if best_label is None:
        return None, 0.0
    floor = min_score if min_score is not None else _threshold_for(best_candidate)
    if best_score >= floor:
        return best_label, best_score
    return None, best_score


# -- Card detection -----------------------------------------------------------
def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def find_card(rgb: np.ndarray, min_area_frac: float = 0.10) -> Optional[np.ndarray]:
    """Corners of the most card-like quadrilateral, or None."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, min_area_frac * w * h
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        area = cv2.contourArea(approx)
        if area < best_area:
            continue
        corners = _order_corners(approx)
        width = np.linalg.norm(corners[1] - corners[0])
        height = np.linalg.norm(corners[3] - corners[0])
        if width < 20 or height < 20:
            continue
        ratio = min(width, height) / max(width, height)
        # A card is 63x88mm, so 0.72. The band allows perspective squash but
        # is tight enough to reject the near-square artwork window inside the
        # card, which used to win and get rectified in place of the card.
        if not 0.55 < ratio < 0.88:
            continue
        best, best_area = corners, area
    return best


def rectify(rgb: np.ndarray, corners: np.ndarray) -> Image.Image:
    """Flatten a detected card quad into an upright card image."""
    width = np.linalg.norm(corners[1] - corners[0])
    height = np.linalg.norm(corners[3] - corners[0])
    if width > height:                       # card photographed on its side
        corners = np.array([corners[3], corners[0], corners[1], corners[2]],
                           dtype=np.float32)
    dst = np.array([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]],
                   dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(rgb, matrix, (CARD_W, CARD_H))
    return Image.fromarray(warped)


# -- Reading ------------------------------------------------------------------
_EVOLVES_RE = re.compile(r"evolve[sd]?\s*fr[o0]m", re.IGNORECASE)


def _candidate_tokens(boxes: Sequence[TextBox], image_h: int):
    """Yield (token, weight) pairs worth testing against the species list.

    Two things stop the wrong name winning:

    * Size. The card name is the biggest text on the card, so weight is scaled
      by height relative to the largest box in view, squared to make the
      preference decisive rather than a tiebreak.
    * The evolution line. "Stage 2 / Evolves from Charmeleon" contains a real
      species name that OCR reads perfectly, so it used to beat a slightly
      misread "Charizard". Any box carrying that phrase is skipped outright.
    """
    if not boxes:
        return
    max_h = max((tb.height for tb in boxes), default=1.0) or 1.0

    for tb in boxes:
        if _EVOLVES_RE.search(tb.text):
            continue
        for word in _WORD_RE.findall(tb.text):
            if _comparable(word) in _TCG_NOISE or len(word) < 3:
                continue
            size_w = (min(1.0, tb.height / max_h)) ** 2
            pos_w = 1.0 - min(1.0, tb.top / max(1.0, image_h)) * 0.5
            yield word, max(0.05, size_w) * pos_w * max(0.3, tb.confidence)


def _drop_evolution_line(boxes: Sequence[TextBox]) -> List[TextBox]:
    """Remove the pre-evolution name, which OCR reads more cleanly than the
    card's own name. The phrase and the name it refers to are sometimes split
    across two boxes, so the box following the phrase goes too."""
    out, skip_next = [], False
    for tb in boxes:
        if _EVOLVES_RE.search(tb.text):
            skip_next = True
            continue
        if skip_next:
            skip_next = False
            # only drop it if it is short, i.e. plausibly just the name
            if len(tb.text.strip().split()) <= 2:
                continue
        out.append(tb)
    return out


def _best_species(boxes: Sequence[TextBox], image_h: int):
    """Highest-scoring species across every token OCR returned.

    Two different numbers are involved and conflating them was a bug. The
    weighted score decides which token on the card is the name; the raw string
    similarity is how sure we are that the token names that species. Only the
    latter is meaningful as a confidence, so that is what gets reported:
    "Charitard" is a 0.89 match for Charizard whether it was printed large or
    small.
    """
    best_label, best_rank, best_score, best_raw = None, 0.0, 0.0, ""
    for token, weight in _candidate_tokens(_drop_evolution_line(boxes), image_h):
        label, score = match_species(token)
        if label is None:
            continue
        rank = score * (0.55 + 0.45 * weight)
        if rank > best_rank:
            best_label, best_rank, best_score, best_raw = label, rank, score, token
    return best_label, best_score, best_raw


NAME_STRIP = 0.20      # the name sits in the top fifth on every card layout

# Each OCR pass costs roughly 1 to 3 seconds on CPU. The cascade usually stops
# at the first pass; this stops a pathological frame from running all four.
TIME_BUDGET = 6.0


def read_card(image, detect_card: bool = True,
              time_budget: float = TIME_BUDGET) -> CardReading:
    """Identify a Pokemon from printed text. Cheap, and exact when it works.

    Rather than trusting card detection, each candidate view is verified by
    whether OCR actually finds a species name in it. Views are tried cheapest
    first and the cascade stops at the first hit, so the common case costs one
    OCR pass over a thin strip.
    """
    pil = vision.flatten_alpha(vision.to_pil(image))
    rgb = np.asarray(pil)

    if not OCR.available:
        return CardReading(None, 0.0, "", [], False, None)

    corners = find_card(rgb) if detect_card else None
    rectified = rectify(rgb, corners) if corners is not None else None

    # cheapest first: name strip of the rectified card, then name strip of the
    # raw frame (rescues a mis-detected card), then the whole frame
    views = []
    if rectified is not None:
        card_arr = np.asarray(rectified)
        views.append(("card-strip", card_arr[:int(card_arr.shape[0] * NAME_STRIP)]))
    views.append(("frame-strip", rgb[:int(rgb.shape[0] * NAME_STRIP)]))
    if rectified is not None:
        views.append(("card-full", np.asarray(rectified)))
    views.append(("frame-full", rgb))

    seen_text: List[str] = []
    started = time.perf_counter()
    for index, (_name, arr) in enumerate(views):
        if arr.size == 0 or min(arr.shape[:2]) < 16:
            continue
        # always run the first pass; later ones only while there is time left
        if index and time.perf_counter() - started > time_budget:
            break
        arr = np.ascontiguousarray(arr)
        boxes = OCR.read(arr)
        seen_text.extend(tb.text for tb in boxes)
        label, score, raw = _best_species(boxes, arr.shape[0])
        if label is not None:
            return CardReading(label, min(1.0, score), raw, seen_text,
                               corners is not None, rectified)

    return CardReading(None, 0.0, "", seen_text, corners is not None, rectified)


def looks_like_a_card(image) -> bool:
    """Quick check with no OCR, for deciding which path to take."""
    rgb = np.asarray(vision.flatten_alpha(vision.to_pil(image)))
    return find_card(rgb) is not None
