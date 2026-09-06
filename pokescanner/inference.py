"""The classifier, shared by the OpenCV scanner, the Gradio app and the Pokedex.

All three used to carry their own copy of the model loading, the TTA list and
the softmax averaging. They are one class now, so a change to the inference
recipe reaches every UI at once.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

from . import config, dex, vision


# -- Prediction record --------------------------------------------------------
@dataclass
class Prediction:
    label: str
    confidence: float                 # 0..1
    stats: dict = field(repr=False, default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.label.replace("-", " ").title()

    @property
    def percent(self) -> float:
        return round(self.confidence * 100, 1)

    def to_dict(self) -> dict:
        return {
            "name": self.label,
            "display_name": self.display_name,
            "confidence": self.percent,
            "stats": self.stats,
        }


@dataclass
class ScanResult:
    predictions: List[Prediction]
    entropy: float = 0.0              # nats; high means the model is spread thin
    frames: int = 1
    elapsed_ms: float = 0.0

    @property
    def top(self) -> Optional[Prediction]:
        return self.predictions[0] if self.predictions else None

    @property
    def is_confident(self) -> bool:
        return bool(self.predictions) and \
            self.predictions[0].confidence >= config.UNSURE_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "predictions": [p.to_dict() for p in self.predictions],
            "confident": self.is_confident,
            "entropy": round(self.entropy, 3),
            "frames": self.frames,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# -- Checkpoint handling ------------------------------------------------------
def _detect_arch(state: dict) -> str:
    """Work out which backbone a bare state_dict came from."""
    keys = state.keys()
    if "conv_stem.weight" in keys:
        head = state.get("conv_head.weight")
        width = head.shape[0] if head is not None else 1408
        return {1280: "efficientnet_b0",
                1408: "efficientnet_b2",
                1536: "efficientnet_b3"}.get(int(width), "efficientnet_b2")
    if any(k.startswith("stem.") or k.startswith("stages.") for k in keys):
        deep = any("stages.2.blocks.26" in k for k in keys)
        return "convnext_small" if deep else "convnext_tiny"
    return config.DEFAULT_ARCH


def _num_classes_from_state(state: dict) -> Optional[int]:
    for key in ("classifier.weight", "head.fc.weight", "fc.weight", "head.weight"):
        if key in state:
            return int(state[key].shape[0])
    return None


def load_checkpoint(path: Path):
    """Return (state_dict, metadata). Accepts bare state dicts and rich ones."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta = {}
    if isinstance(blob, dict) and any(
        k in blob for k in ("model", "state_dict", "model_state_dict")
    ):
        state = blob.get("model") or blob.get("state_dict") or blob["model_state_dict"]
        meta = {k: v for k, v in blob.items()
                if k not in ("model", "state_dict", "model_state_dict")}
    else:
        state = blob
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    if "arch" not in meta:
        # a filename hint beats key-sniffing when both are available
        stem = Path(path).stem.lower()
        for key in config.ARCHITECTURES:
            if key in stem or key.split("_")[-1] == stem.rsplit("_", 1)[-1]:
                meta["arch"] = key
                break
    meta.setdefault("arch", _detect_arch(state))
    return state, meta


# -- Test-time augmentation ---------------------------------------------------
def _normalise():
    return transforms.Normalize(config.MEAN, config.STD)


def build_tta(size: int, level: int = 4) -> List[transforms.Compose]:
    """TTA views, cheapest and most reliable first.

    Inputs reach these already square and letterboxed, so the views vary scale
    and colour rather than aspect ratio.
    """
    to_tensor = [transforms.ToTensor(), _normalise()]
    views = [
        # 1. the prepared image as-is
        transforms.Compose([transforms.Resize((size, size)), *to_tensor]),
        # 2. slight zoom in - trims the padding
        transforms.Compose([transforms.Resize((int(size * 1.12), int(size * 1.12))),
                            transforms.CenterCrop(size), *to_tensor]),
        # 3. mirrored
        transforms.Compose([transforms.Resize((size, size)),
                            transforms.RandomHorizontalFlip(p=1.0), *to_tensor]),
        # 4. stronger zoom
        transforms.Compose([transforms.Resize((int(size * 1.25), int(size * 1.25))),
                            transforms.CenterCrop(size), *to_tensor]),
        # 5. brightness/contrast shift - cameras are dimmer than sprites
        transforms.Compose([transforms.Resize((size, size)),
                            transforms.ColorJitter(brightness=0.25, contrast=0.25),
                            *to_tensor]),
        # 6. mirrored zoom
        transforms.Compose([transforms.Resize((int(size * 1.12), int(size * 1.12))),
                            transforms.CenterCrop(size),
                            transforms.RandomHorizontalFlip(p=1.0), *to_tensor]),
    ]
    return views[:max(1, min(level, len(views)))]


# -- Classifier ---------------------------------------------------------------
class PokemonClassifier:
    """Loads a checkpoint once and answers scans.

    >>> clf = PokemonClassifier()
    >>> clf.predict(pil_image).top.display_name
    'Pikachu'
    """

    def __init__(self,
                 weights: Optional[Path] = None,
                 arch: Optional[str] = None,
                 device: Optional[str] = None,
                 tta_level: int = 4,
                 isolate: Optional[bool] = None,
                 temperature: float = config.TEMPERATURE,
                 verbose: bool = True):
        self.weights = Path(weights) if weights else config.default_weights()
        if not self.weights.exists():
            raise FileNotFoundError(
                f"No model weights at {self.weights}. Train one with "
                f"`python model/train.py`, or drop a .pth into "
                f"{config.WEIGHTS_DIR}."
            )

        with open(config.LABEL_MAP_PATH, encoding="utf-8") as fh:
            label_map = json.load(fh)
        self.idx_to_label = {int(k): v for k, v in label_map["idx_to_label"].items()}
        self.num_classes = int(label_map["num_classes"])

        state, meta = load_checkpoint(self.weights)
        self.arch = arch or meta.get("arch", config.DEFAULT_ARCH)
        timm_name, native_size = config.ARCHITECTURES.get(
            self.arch, (self.arch, 260))
        self.img_size = int(meta.get("img_size") or native_size)

        ckpt_classes = _num_classes_from_state(state) or self.num_classes
        if ckpt_classes != self.num_classes:
            raise ValueError(
                f"Checkpoint has {ckpt_classes} classes but label_map.json has "
                f"{self.num_classes}. They must come from the same run."
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = timm.create_model(timm_name, pretrained=False,
                                       num_classes=self.num_classes)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)
        if self.device.type == "cpu":
            torch.set_num_threads(max(1, torch.get_num_threads()))

        self.tta = build_tta(self.img_size, tta_level)
        self.isolate = config.ISOLATE_SUBJECT if isolate is None else isolate
        self.temperature = max(1e-3, float(temperature))
        self.val_accuracy = meta.get("val_accuracy")

        if verbose:
            if not self.val_accuracy:
                print(f"[pokescanner] WARNING: {self.weights.name} reports no "
                      f"validation accuracy, so it was never checked against "
                      f"held-out data. Set POKESCANNER_WEIGHTS to pick another.")
            acc = f", val acc {self.val_accuracy:.1%}" if self.val_accuracy else ""
            print(f"[pokescanner] {self.arch} @ {self.img_size}px on "
                  f"{self.device}, {self.num_classes} classes, "
                  f"{len(self.tta)} TTA views{acc}")

    # -- internals ------------------------------------------------------------
    def _probs_for(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """Averaged softmax over every (image, TTA view) pair. Shape [C]."""
        batch = torch.stack([t(img) for img in images for t in self.tta])
        batch = batch.to(self.device)
        with torch.inference_mode():
            logits = self.model(batch) / self.temperature
            probs = F.softmax(logits, dim=1)
        return probs.mean(dim=0).cpu()

    def _to_result(self, probs: torch.Tensor, top_k: int,
                   frames: int, elapsed: float) -> ScanResult:
        k = max(1, min(top_k, probs.numel()))
        top_probs, top_idxs = torch.topk(probs, k)
        preds = [
            Prediction(label=self.idx_to_label[int(i)],
                       confidence=float(p),
                       stats=dex.get(self.idx_to_label[int(i)]))
            for p, i in zip(top_probs, top_idxs)
        ]
        p = probs.clamp_min(1e-12)
        entropy = float(-(p * p.log()).sum())
        return ScanResult(preds, entropy=entropy, frames=frames,
                          elapsed_ms=elapsed * 1000)

    # -- public API -----------------------------------------------------------
    def prepare(self, image, remove_background: bool = False) -> Image.Image:
        return vision.prepare(image, self.img_size, isolate=self.isolate,
                              remove_background=remove_background)

    def predict(self, image, top_k: int = config.TOP_K,
                remove_background: bool = False) -> ScanResult:
        """Identify a single image (PIL, numpy RGB/RGBA, or a path)."""
        started = time.perf_counter()
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        prepared = self.prepare(image, remove_background=remove_background)
        probs = self._probs_for([prepared])
        return self._to_result(probs, top_k, 1, time.perf_counter() - started)

    def predict_frames(self, frames: Iterable, top_k: int = config.TOP_K,
                       remove_background: bool = False) -> ScanResult:
        """Aggregate several webcam frames into one answer.

        Averaging the probabilities across frames (rather than voting on each
        frame's argmax) keeps the runner-up information, so a Pokemon that is
        second-best in every frame can still win.
        """
        started = time.perf_counter()
        prepared = [self.prepare(f, remove_background=remove_background)
                    for f in frames]
        if not prepared:
            return ScanResult([], frames=0)
        probs = self._probs_for(prepared)
        return self._to_result(probs, top_k, len(prepared),
                               time.perf_counter() - started)

    def predict_batch(self, images: Sequence, top_k: int = 1) -> List[ScanResult]:
        """One result per image. Used by the evaluation tooling."""
        return [self.predict(img, top_k=top_k) for img in images]

    def warmup(self) -> None:
        """Run one dummy pass so the first real scan is not the slow one."""
        blank = Image.new("RGB", (self.img_size, self.img_size),
                          config.CANVAS_COLOR)
        self._probs_for([blank])


_SHARED: Optional[PokemonClassifier] = None


def get_classifier(**kwargs) -> PokemonClassifier:
    """Process-wide singleton, so importing two UIs does not load two models."""
    global _SHARED
    if _SHARED is None:
        _SHARED = PokemonClassifier(**kwargs)
    return _SHARED
