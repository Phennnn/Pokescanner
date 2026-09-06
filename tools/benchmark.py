"""Measure the sprite -> real-world domain gap, and what closes it.

Builds a seeded set of synthetic "photos" (sprite pasted into a cluttered scene,
then blurred, dimmed, noised and JPEG-compressed) and scores several inference
pipelines on the identical images:

    legacy      Resize((S, S)) on the raw frame - what the apps used to do
    letterbox   aspect-preserving resize, no cropping
    isolate     find the subject, crop it, centre it on white  (the new default)
    isolate+bg  same, but also erase everything outside the silhouette

Usage:
    python tools/benchmark.py                    # 200 classes, quick
    python tools/benchmark.py --n 809 --views 4  # full sweep
    python tools/benchmark.py --dump 12          # also save sample scenes

Read the numbers as relative, not absolute: the sprites were part of training,
so every pipeline scores higher here than it would on genuine card photos. What
transfers is the ordering and the size of the gaps between pipelines.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokescanner import config, synth, vision          # noqa: E402
from pokescanner.inference import PokemonClassifier    # noqa: E402

PIPELINES = ("legacy", "letterbox", "isolate", "isolate+bg")


def build_eval_set(n_classes: int, per_class: int, seed: int):
    """[(pil_scene, label)] - deterministic for a given seed."""
    rng = random.Random(seed)
    class_dirs = sorted(p for p in config.IMAGES_DIR.iterdir() if p.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class folders in {config.IMAGES_DIR}")
    all_sprites = [next(iter(sorted(d.glob("*.png"))), None) for d in class_dirs]
    all_sprites = [p for p in all_sprites if p]

    chosen = class_dirs if n_classes >= len(class_dirs) else \
        rng.sample(class_dirs, n_classes)

    samples = []
    for d in sorted(chosen):
        files = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))
        if not files:
            continue
        for i in range(per_class):
            try:
                sprite = Image.open(files[i % len(files)])
            except Exception:
                continue
            scene = synth.composite_scene(sprite, rng=rng, sprites=all_sprites)
            samples.append((scene, d.name))
    return samples


def prep(pipeline: str, image: Image.Image, size: int) -> Image.Image:
    if pipeline == "legacy":
        return vision.flatten_alpha(image).resize((size, size), Image.BILINEAR)
    if pipeline == "letterbox":
        return vision.letterbox(vision.flatten_alpha(image), size)
    if pipeline == "isolate":
        return vision.prepare(image, size, isolate=True, remove_background=False)
    if pipeline == "isolate+bg":
        return vision.prepare(image, size, isolate=True, remove_background=True)
    raise ValueError(pipeline)


def score(clf: PokemonClassifier, samples, pipeline: str, batch: int = 8):
    """top-1, top-5, mean top-1 probability, seconds per image."""
    label_to_idx = {v: k for k, v in clf.idx_to_label.items()}
    top1 = top5 = seen = 0
    conf_sum = 0.0
    started = time.perf_counter()

    for start in range(0, len(samples), batch):
        chunk = samples[start:start + batch]
        views = []
        for image, _ in chunk:
            prepared = prep(pipeline, image, clf.img_size)
            views.extend(t(prepared) for t in clf.tta)

        tensor = torch.stack(views).to(clf.device)
        with torch.inference_mode():
            probs = F.softmax(clf.model(tensor) / clf.temperature, dim=1)
        probs = probs.view(len(chunk), len(clf.tta), -1).mean(dim=1)

        top = probs.topk(5, dim=1)
        for row, (_, label) in enumerate(chunk):
            gold = label_to_idx.get(label)
            if gold is None:
                continue
            ranked = top.indices[row].tolist()
            seen += 1
            conf_sum += float(top.values[row][0])
            if ranked[0] == gold:
                top1 += 1
            if gold in ranked:
                top5 += 1

    elapsed = time.perf_counter() - started
    return {
        "pipeline": pipeline,
        "n": seen,
        "top1": top1 / seen if seen else 0.0,
        "top5": top5 / seen if seen else 0.0,
        "mean_conf": conf_sum / seen if seen else 0.0,
        "sec_per_image": elapsed / max(1, seen),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="number of classes to sample")
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--views", type=int, default=4, help="TTA views")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--pipelines", nargs="*", default=list(PIPELINES))
    ap.add_argument("--dump", type=int, default=0,
                    help="save this many example scenes for eyeballing")
    ap.add_argument("--out", type=Path, default=None, help="write results as JSON")
    args = ap.parse_args()

    clf = PokemonClassifier(weights=args.weights, tta_level=args.views)

    print(f"Building {args.n} x {args.per_class} synthetic scenes (seed {args.seed})...")
    samples = build_eval_set(args.n, args.per_class, args.seed)
    print(f"{len(samples)} scenes ready\n")

    if args.dump:
        out_dir = config.ROOT / "reports" / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        for image, label in samples[:args.dump]:
            image.save(out_dir / f"{label}_scene.jpg", quality=90)
            prep("isolate", image, clf.img_size).save(
                out_dir / f"{label}_isolated.jpg", quality=90)
        print(f"Wrote {args.dump} example pairs to {out_dir}\n")

    rows = []
    for pipeline in args.pipelines:
        row = score(clf, samples, pipeline)
        rows.append(row)
        print(f"{row['pipeline']:12} top1 {row['top1']:6.1%}   "
              f"top5 {row['top5']:6.1%}   conf {row['mean_conf']:5.1%}   "
              f"{row['sec_per_image']*1000:5.0f} ms/img")

    if rows:
        base = next((r for r in rows if r["pipeline"] == "legacy"), rows[0])
        best = max(rows, key=lambda r: r["top1"])
        delta = best["top1"] - base["top1"]
        print(f"\nbest: {best['pipeline']}  ({delta:+.1%} top-1 vs {base['pipeline']})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"config": vars(args) | {"weights": str(args.weights)},
             "results": rows}, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
