"""Synthetic "in the wild" scenes built from transparent sprites.

The training set is 809 sprites on transparent backgrounds; the test set in
practice is a phone or webcam pointed at a card on a desk. This module bridges
that by pasting a sprite into a plausible scene - random background, scale,
placement, rotation, lighting, blur, sensor noise and JPEG artefacts.

It has two customers:

  * ``tools/benchmark.py`` - builds a fixed, seeded evaluation set so changes to
    the inference pipeline can be measured instead of guessed at.
  * ``model/train.py`` - applies the same corruptions as training augmentation,
    so the next model sees cluttered backgrounds during training.

Backgrounds are procedural by default, so nothing needs downloading. Drop real
photos into ``data/backgrounds/`` and they will be used instead - real photos
are better, this is just the zero-setup path.
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from . import config

_BG_CACHE: Optional[List[Path]] = None


def background_files() -> List[Path]:
    """Real background photos, if the user supplied any."""
    global _BG_CACHE
    if _BG_CACHE is None:
        d = config.BACKGROUNDS_DIR
        if d.exists():
            _BG_CACHE = sorted(
                p for p in d.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            )
        else:
            _BG_CACHE = []
    return _BG_CACHE


# -- Procedural backgrounds ---------------------------------------------------
def _gradient(size, rng: random.Random) -> Image.Image:
    w, h = size
    c0 = np.array([rng.randint(20, 235) for _ in range(3)], np.float32)
    c1 = np.array([rng.randint(20, 235) for _ in range(3)], np.float32)
    if rng.random() < 0.5:
        ramp = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    else:
        ramp = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    arr = c0 * (1 - ramp) + c1 * ramp
    arr = np.broadcast_to(arr, (h, w, 3)).copy()
    return Image.fromarray(arr.astype(np.uint8))


def _noise_wall(size, rng: random.Random) -> Image.Image:
    """Blurred noise - reads as a painted wall, carpet or cloth."""
    w, h = size
    base = np.array([rng.randint(30, 225) for _ in range(3)], np.float32)
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
        0, rng.uniform(8, 40), (h, w, 3))
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).filter(
        ImageFilter.GaussianBlur(rng.uniform(1.5, 6)))
    return img


def _stripes(size, rng: random.Random) -> Image.Image:
    """Wood grain, table edges, shelf slats."""
    w, h = size
    base = np.array([rng.randint(60, 200), rng.randint(40, 160),
                     rng.randint(20, 130)], np.float32)
    freq = rng.uniform(0.02, 0.14)
    phase = rng.uniform(0, 6.28)
    axis = np.arange(h if rng.random() < 0.5 else w, dtype=np.float32)
    wave = (np.sin(axis * freq + phase) * rng.uniform(6, 26))
    if wave.shape[0] == h:
        arr = base[None, None, :] + wave[:, None, None]
        arr = np.broadcast_to(arr, (h, w, 3)).copy()
    else:
        arr = base[None, None, :] + wave[None, :, None]
        arr = np.broadcast_to(arr, (h, w, 3)).copy()
    arr += np.random.default_rng(rng.randrange(1 << 30)).normal(0, 6, (h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _clutter(size, rng: random.Random, sprites: Sequence[Path]) -> Image.Image:
    """Other sprites, blurred and blown up - stands in for a busy desk."""
    img = _noise_wall(size, rng)
    if not sprites:
        return img
    for _ in range(rng.randint(1, 3)):
        try:
            other = Image.open(rng.choice(sprites)).convert("RGBA")
        except Exception:
            continue
        scale = rng.uniform(1.5, 4.0)
        other = other.resize((max(1, int(other.width * scale)),
                              max(1, int(other.height * scale))), Image.BICUBIC)
        other = other.filter(ImageFilter.GaussianBlur(rng.uniform(2, 7)))
        img.paste(other, (rng.randint(-other.width // 2, size[0]),
                          rng.randint(-other.height // 2, size[1])), other)
    return img


def random_background(size, rng: random.Random,
                      sprites: Sequence[Path] = ()) -> Image.Image:
    """A background of the requested size, from photos if available."""
    files = background_files()
    if files and rng.random() < 0.8:
        try:
            img = Image.open(rng.choice(files)).convert("RGB")
            # random crop then resize, so the same photo is not always framed alike
            w, h = img.size
            side = int(min(w, h) * rng.uniform(0.5, 1.0))
            x = rng.randint(0, max(0, w - side))
            y = rng.randint(0, max(0, h - side))
            return img.crop((x, y, x + side, y + side)).resize(size, Image.BICUBIC)
        except Exception:
            pass

    pick = rng.random()
    if pick < 0.3:
        return _gradient(size, rng)
    if pick < 0.6:
        return _noise_wall(size, rng)
    if pick < 0.8:
        return _stripes(size, rng)
    return _clutter(size, rng, sprites)


# -- Corruptions --------------------------------------------------------------
def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def camera_effects(img: Image.Image, rng: random.Random,
                   strength: float = 1.0) -> Image.Image:
    """Lighting, focus and sensor artefacts a real capture would carry."""
    if rng.random() < 0.9:
        img = ImageEnhance.Brightness(img).enhance(
            1 + rng.uniform(-0.35, 0.35) * strength)
    if rng.random() < 0.9:
        img = ImageEnhance.Contrast(img).enhance(
            1 + rng.uniform(-0.3, 0.3) * strength)
    if rng.random() < 0.7:
        img = ImageEnhance.Color(img).enhance(
            1 + rng.uniform(-0.4, 0.3) * strength)
    if rng.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 2.0) * strength))
    if rng.random() < 0.6:
        arr = np.asarray(img, np.float32)
        arr += np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, rng.uniform(2, 14) * strength, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.7:
        img = _jpeg(img, rng.randint(35, 92))
    return img


def composite_scene(sprite: Image.Image,
                    size=(640, 480),
                    rng: Optional[random.Random] = None,
                    sprites: Sequence[Path] = (),
                    fill_range=(0.22, 0.62),
                    strength: float = 1.0) -> Image.Image:
    """Paste one RGBA sprite into a random scene and return an RGB frame."""
    rng = rng or random.Random()
    sprite = sprite.convert("RGBA")

    # trim the sprite's transparent margin so `fill` means what it says
    bbox = sprite.getbbox()
    if bbox:
        sprite = sprite.crop(bbox)

    bg = random_background(size, rng, sprites)

    if rng.random() < 0.8:
        sprite = sprite.rotate(rng.uniform(-18, 18), Image.BICUBIC, expand=True)

    fill = rng.uniform(*fill_range)
    target = fill * min(size)
    scale = target / max(1, max(sprite.size))
    new_size = (max(2, int(sprite.width * scale)), max(2, int(sprite.height * scale)))
    sprite = sprite.resize(new_size, Image.LANCZOS)

    max_x = max(1, size[0] - sprite.width)
    max_y = max(1, size[1] - sprite.height)
    # bias towards the centre - people do roughly aim at the thing
    cx = int(np.clip(rng.gauss(max_x / 2, max_x / 5), 0, max_x))
    cy = int(np.clip(rng.gauss(max_y / 2, max_y / 5), 0, max_y))

    bg.paste(sprite, (cx, cy), sprite)
    return camera_effects(bg, rng, strength)
