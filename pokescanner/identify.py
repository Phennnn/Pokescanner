"""One entry point that decides how to identify what the camera is seeing.

Two paths, with very different strengths:

  * Reading the card. Near-exact when there is printed text, useless otherwise.
    Measured 86.7% on photographed cards against 0.0% for the classifier on the
    same images (tools/benchmark_cards.py).
  * Classifying the image. Handles sprites, artwork, figures and screens, but
    has never seen a trading card in training.

So: try to read it, and fall back to looking at it. That ordering matters,
because on a card the classifier is not merely worse, it is confidently wrong.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from . import cards, dex

# Below this the card reading is treated as a guess and the classifier is used.
CARD_TRUST = 0.80


@dataclass
class Identification:
    label: Optional[str]
    confidence: float                    # 0..1, comparable across sources
    source: str                          # "card" | "model" | "none"
    display_name: str = ""
    stats: dict = field(default_factory=dict)
    alternatives: List[dict] = field(default_factory=list)
    card_text: str = ""                  # what OCR read, when it was used
    card_found: bool = False
    elapsed_ms: float = 0.0

    @property
    def is_confident(self) -> bool:
        if self.source == "card":
            return self.confidence >= CARD_TRUST
        from . import config
        return self.confidence >= config.UNSURE_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "name": self.label,
            "display_name": self.display_name,
            "confidence": round(self.confidence * 100, 1),
            "source": self.source,
            "confident": self.is_confident,
            "stats": self.stats,
            "alternatives": self.alternatives,
            "card_text": self.card_text,
            "card_found": self.card_found,
            "elapsed_ms": round(self.elapsed_ms),
        }


def _from_label(label: str, confidence: float, source: str,
                **extra) -> Identification:
    entry = dex.get(label)
    return Identification(label=label, confidence=confidence, source=source,
                          display_name=entry["display_name"], stats=entry,
                          **extra)


def identify(image, classifier=None, mode: str = "auto",
             top_k: int = 4) -> Identification:
    """Identify a Pokemon from a frame.

    mode:
      "auto"  read the text first, fall back to the classifier (default)
      "card"  card reading only
      "model" classifier only
    """
    started = time.perf_counter()

    reading = None
    if mode in ("auto", "card") and cards.OCR.available:
        reading = cards.read_card(image)
        if reading.label and reading.score >= CARD_TRUST:
            return _from_label(
                reading.label, reading.score, "card",
                card_text=reading.raw_text, card_found=reading.card_found,
                elapsed_ms=(time.perf_counter() - started) * 1000)

    if mode == "card":
        elapsed = (time.perf_counter() - started) * 1000
        if reading and reading.label:
            return _from_label(reading.label, reading.score, "card",
                               card_text=reading.raw_text,
                               card_found=reading.card_found,
                               elapsed_ms=elapsed)
        return Identification(None, 0.0, "none", elapsed_ms=elapsed,
                              card_found=bool(reading and reading.card_found))

    if classifier is None:
        elapsed = (time.perf_counter() - started) * 1000
        return Identification(None, 0.0, "none", elapsed_ms=elapsed)

    result = classifier.predict(image, top_k=top_k)
    elapsed = (time.perf_counter() - started) * 1000
    if not result.predictions:
        return Identification(None, 0.0, "none", elapsed_ms=elapsed)

    top = result.predictions[0]
    return _from_label(
        top.label, top.confidence, "model",
        alternatives=[p.to_dict() for p in result.predictions[1:]],
        card_text=(reading.raw_text if reading else ""),
        card_found=bool(reading and reading.card_found),
        elapsed_ms=elapsed)
