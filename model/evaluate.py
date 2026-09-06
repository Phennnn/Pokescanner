"""Score a checkpoint and say where it actually fails.

    python model/evaluate.py                          # held-out split, clean sprites
    python model/evaluate.py --scenes                 # same images, put in cluttered scenes
    python model/evaluate.py --weights model/weights/best_model_convnext_tiny.pth

A single val-accuracy number hides the interesting part. This prints the
confusion pairs the model keeps making, the classes it never gets right, and a
calibration table showing whether its confidence means anything - which is what
you need before deciding what to fix next.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokescanner import config, synth, vision          # noqa: E402
from pokescanner.inference import PokemonClassifier    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import collect_samples, stratified_split    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--images", type=Path, default=config.IMAGES_DIR)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--scenes", action="store_true",
                    help="composite each image into a random scene first")
    ap.add_argument("--no-isolate", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    clf = PokemonClassifier(weights=args.weights, tta_level=args.views,
                            isolate=not args.no_isolate)
    label_to_idx = {v: k for k, v in clf.idx_to_label.items()}

    with open(config.LABEL_MAP_PATH, encoding="utf-8") as fh:
        label_map = json.load(fh)

    by_class = collect_samples(args.images, label_map["label_to_idx"])
    _, val = stratified_split(by_class, args.val_split, args.seed)
    if not val:
        # Every class has a single image (the repo ships one sprite per class),
        # so there is no held-out data; score the training images instead and
        # say so, rather than reporting nothing.
        val = [(paths[0], idx) for idx, paths in by_class.items()]
        print("No held-out split available (one image per class). "
              "Scoring the training images - treat these numbers as a sanity "
              "check, not as generalisation.\n")

    rng = random.Random(args.seed)
    if args.limit:
        val = rng.sample(val, min(args.limit, len(val)))

    print(f"evaluating {len(val)} images"
          f"{' as synthetic scenes' if args.scenes else ''}\n")

    sprite_pool = [paths[0] for paths in by_class.values()]
    top1 = top5 = 0
    confusions = Counter()
    per_class_wrong = defaultdict(int)
    per_class_total = defaultdict(int)
    bins = defaultdict(lambda: [0, 0])          # confidence decile -> [correct, n]

    for i, (path, gold_idx) in enumerate(val, 1):
        gold = clf.idx_to_label[gold_idx]
        try:
            img = Image.open(path)
        except Exception:
            continue
        if args.scenes:
            img = synth.composite_scene(img.convert("RGBA"), rng=rng,
                                        sprites=sprite_pool)

        prepared = clf.prepare(img)
        batch = torch.stack([t(prepared) for t in clf.tta]).to(clf.device)
        with torch.inference_mode():
            probs = F.softmax(clf.model(batch) / clf.temperature, dim=1).mean(0)

        vals, idxs = probs.topk(5)
        ranked = [clf.idx_to_label[int(x)] for x in idxs]
        conf = float(vals[0])

        per_class_total[gold] += 1
        correct = ranked[0] == gold
        top1 += correct
        top5 += gold in ranked
        if not correct:
            confusions[(gold, ranked[0])] += 1
            per_class_wrong[gold] += 1

        bucket = min(9, int(conf * 10))
        bins[bucket][0] += correct
        bins[bucket][1] += 1

        if i % 100 == 0:
            print(f"  {i}/{len(val)}  running top-1 {top1 / i:.1%}", flush=True)

    n = max(1, len(val))
    print(f"\ntop-1 {top1 / n:.2%}    top-5 {top5 / n:.2%}    n={n}")

    print("\nmost frequent confusions (true -> predicted)")
    for (gold, pred), count in confusions.most_common(15):
        print(f"  {gold:<24} -> {pred:<24} x{count}")

    always_wrong = sorted(c for c in per_class_total
                          if per_class_wrong[c] == per_class_total[c])
    print(f"\nclasses never predicted correctly: {len(always_wrong)}")
    print("  " + ", ".join(always_wrong[:25]) + (" ..." if len(always_wrong) > 25 else ""))

    print("\ncalibration (is the confidence meaningful?)")
    print("  confidence   accuracy    n")
    for bucket in sorted(bins):
        correct, total = bins[bucket]
        print(f"  {bucket * 10:>3}-{bucket * 10 + 10:<3}%   "
              f"{correct / total:>7.1%}  {total:>5}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "weights": str(clf.weights),
            "arch": clf.arch,
            "scenes": args.scenes,
            "isolate": clf.isolate,
            "n": n,
            "top1": top1 / n,
            "top5": top5 / n,
            "confusions": [[g, p, c] for (g, p), c in confusions.most_common(50)],
            "never_correct": always_wrong,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
