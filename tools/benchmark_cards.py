"""Compare card OCR against the image classifier on photographed cards.

Real card scans are downloaded once from api.pokemontcg.io, then each one is
turned into something a webcam would actually see: perspective tilt, a
cluttered desk behind it, uneven lighting, glare, blur, sensor noise and JPEG
compression. Both identification paths then run on the identical image.

    python tools/benchmark_cards.py --fetch      # download cards first
    python tools/benchmark_cards.py --n 40

The classifier was trained on sprites and has never seen a card, so this is the
comparison that decides whether reading the text is worth it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokescanner import cards, config, synth  # noqa: E402

CARD_DIR = config.ROOT / "reports" / "cards"
MANIFEST = CARD_DIR / "_manifest.json"

SPECIES = [
    "charizard", "pikachu", "blastoise", "gengar", "snorlax", "mewtwo", "eevee",
    "lucario", "greninja", "rayquaza", "gardevoir", "umbreon", "garchomp",
    "sylveon", "dragonite", "tyranitar", "metagross", "zoroark", "aegislash",
    "decidueye", "venusaur", "alakazam", "machamp", "gyarados", "arcanine",
    "scizor", "houndoom", "salamence", "infernape", "darkrai",
]


def _get(url: str, tries: int = 4) -> bytes:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pokescanner"})
            return urllib.request.urlopen(req, timeout=40).read()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 + 3 * attempt)
    raise RuntimeError("unreachable")


def fetch_cards(per_species: int = 2) -> list:
    """Download a spread of real card images. Free API, so be gentle with it."""
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for name in SPECIES:
        try:
            blob = _get(f"https://api.pokemontcg.io/v2/cards?q=name:{name}"
                        f"&pageSize={per_species}&orderBy=set.releaseDate")
            data = json.loads(blob)
        except Exception as exc:
            print(f"  skip {name}: {type(exc).__name__}")
            continue
        for card in data.get("data", [])[:per_species]:
            images = card.get("images", {})
            url = images.get("large") or images.get("small")
            if not url:
                continue
            path = CARD_DIR / f"{name}__{card['id']}.png"
            if not path.exists():
                try:
                    path.write_bytes(_get(url))
                except Exception as exc:
                    print(f"  image fail {card['id']}: {type(exc).__name__}")
                    continue
            saved.append([str(path), name])
        time.sleep(0.4)
    MANIFEST.write_text(json.dumps(saved), encoding="utf-8")
    print(f"{len(saved)} card images in {CARD_DIR}")
    return saved


def photograph(card: Image.Image, rng: random.Random,
               sprites=()) -> Image.Image:
    """Make a clean card scan look like a webcam photo of that card."""
    card = card.convert("RGB")
    cw, ch = card.size

    # perspective tilt
    arr = np.asarray(card)
    jitter = min(cw, ch) * rng.uniform(0.02, 0.11)
    src = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    dst = np.float32([[rng.uniform(0, jitter), rng.uniform(0, jitter)],
                      [cw - rng.uniform(0, jitter), rng.uniform(0, jitter)],
                      [cw - rng.uniform(0, jitter), ch - rng.uniform(0, jitter)],
                      [rng.uniform(0, jitter), ch - rng.uniform(0, jitter)]])
    warped = cv2.warpPerspective(arr, cv2.getPerspectiveTransform(src, dst),
                                 (cw, ch), borderValue=(255, 255, 255))

    # glare: a soft bright band across the foil
    if rng.random() < 0.65:
        glare = np.zeros((ch, cw), np.float32)
        x0, y0 = rng.randint(0, cw), rng.randint(0, ch)
        cv2.line(glare, (x0, y0),
                 (x0 + rng.randint(-cw, cw), y0 + rng.randint(-ch, ch)),
                 1.0, thickness=rng.randint(40, 160))
        glare = cv2.GaussianBlur(glare, (0, 0), rng.uniform(30, 80))
        warped = np.clip(warped.astype(np.float32)
                         + glare[:, :, None] * rng.uniform(40, 130), 0, 255)
        warped = warped.astype(np.uint8)

    card_img = Image.fromarray(warped)

    # place it on a desk, taking up a realistic share of the frame
    frame_w, frame_h = 800, 600
    scene = synth.random_background((frame_w, frame_h), rng, sprites)
    fill = rng.uniform(0.45, 0.85)
    scale = (frame_h * fill) / ch
    new = (max(8, int(cw * scale)), max(8, int(ch * scale)))
    card_img = card_img.resize(new, Image.LANCZOS)
    if rng.random() < 0.7:
        card_img = card_img.rotate(rng.uniform(-12, 12), Image.BICUBIC,
                                   expand=True, fillcolor=(255, 255, 255))

    x = rng.randint(0, max(1, frame_w - card_img.width))
    y = rng.randint(0, max(1, frame_h - card_img.height))
    scene.paste(card_img, (x, y))
    return synth.camera_effects(scene, rng, strength=0.75)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="download cards first")
    ap.add_argument("--n", type=int, default=0, help="limit how many to test")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--clean", action="store_true",
                    help="score the raw scans instead of photographing them")
    ap.add_argument("--no-cnn", action="store_true", help="skip the classifier")
    ap.add_argument("--dump", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.fetch or not MANIFEST.exists():
        manifest = fetch_cards()
    else:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not manifest:
        raise SystemExit("No cards. Run with --fetch")

    rng = random.Random(args.seed)
    if args.n:
        manifest = rng.sample(manifest, min(args.n, len(manifest)))

    clf = None
    if not args.no_cnn:
        from pokescanner.inference import PokemonClassifier
        clf = PokemonClassifier(tta_level=4)
        clf.warmup()

    print(f"OCR backend: {cards.OCR.kind}")
    print(f"{len(manifest)} cards, "
          f"{'clean scans' if args.clean else 'photographed'}\n")

    sprite_pool = [next(iter(sorted(d.glob('*.png'))), None)
                   for d in sorted(config.IMAGES_DIR.iterdir()) if d.is_dir()]
    sprite_pool = [p for p in sprite_pool if p]

    ocr_ok = ocr_wrong = ocr_none = 0
    cnn_ok = 0
    ocr_time = cnn_time = 0.0
    rows = []

    for i, (path, expected) in enumerate(manifest, 1):
        try:
            card = Image.open(path)
        except Exception:
            continue
        image = card if args.clean else photograph(card, rng, sprite_pool)

        if args.dump and i <= args.dump:
            out_dir = config.ROOT / "reports" / "card_scenes"
            out_dir.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").save(out_dir / f"{expected}_{i}.jpg", quality=88)

        t0 = time.perf_counter()
        reading = cards.read_card(image)
        ocr_time += time.perf_counter() - t0

        ocr_hit = reading.label is not None and \
            reading.label.split("-")[0] == expected
        if ocr_hit:
            ocr_ok += 1
        elif reading.label is None:
            ocr_none += 1
        else:
            ocr_wrong += 1

        cnn_hit = None
        if clf is not None:
            t0 = time.perf_counter()
            result = clf.predict(image, top_k=1)
            cnn_time += time.perf_counter() - t0
            cnn_hit = bool(result.top and
                           result.top.label.split("-")[0] == expected)
            cnn_ok += cnn_hit

        rows.append({"expected": expected, "ocr": reading.label,
                     "ocr_hit": ocr_hit, "cnn_hit": cnn_hit})
        flag = "ok " if ocr_hit else ("-- " if reading.label is None else "XX ")
        print(f"  {flag}{expected:<12} ocr={str(reading.label):<20}"
              f"{'cnn ok' if cnn_hit else ('cnn miss' if clf else '')}")

    n = max(1, len(rows))
    print(f"\ncard OCR   {ocr_ok}/{n} = {ocr_ok / n:.1%}   "
          f"(wrong {ocr_wrong}, no match {ocr_none})   "
          f"{ocr_time / n:.2f}s per card")
    if clf is not None:
        print(f"classifier {cnn_ok}/{n} = {cnn_ok / n:.1%}   "
              f"{cnn_time / n:.2f}s per card")
        print(f"\nOCR beats the classifier by "
              f"{(ocr_ok - cnn_ok) / n:+.1%} on cards")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"n": n, "ocr_top1": ocr_ok / n,
             "cnn_top1": cnn_ok / n if clf else None,
             "clean": args.clean, "rows": rows}, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
