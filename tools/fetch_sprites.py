"""Add more images per class from the PokeAPI sprite repository.

The single biggest limit on accuracy is that each class has only a handful of
training images. This pulls up to nine additional views per species straight
from GitHub (no API key, no Kaggle account) and drops them into the existing
class folders:

    front, back, shiny, shiny back, official artwork, HOME render,
    Black/White, Emerald and Platinum game sprites

Official artwork and HOME renders matter most: they are large, shaded, 3D-ish
images, much closer to a photograph of a figure than a 96x96 game sprite is.

    python tools/fetch_sprites.py                # every class
    python tools/fetch_sprites.py --limit 20     # try it on 20 first
    python tools/fetch_sprites.py --sources official-artwork home

Safe to re-run: anything already downloaded is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokescanner import config, dex  # noqa: E402

BASE = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon"

SOURCES = {
    "front":             "{n}.png",
    "back":              "back/{n}.png",
    "shiny":             "shiny/{n}.png",
    "shiny-back":        "back/shiny/{n}.png",
    "official-artwork":  "other/official-artwork/{n}.png",
    "home":              "other/home/{n}.png",
    "black-white":       "versions/generation-v/black-white/{n}.png",
    "emerald":           "versions/generation-iii/emerald/{n}.png",
    "platinum":          "versions/generation-iv/platinum/{n}.png",
}


def dex_numbers() -> dict:
    """{label: national dex number} for all 809 classes.

    The stats CSV stops at 801 and omits eight late Gen 7 species, but the
    types CSV is in national dex order, so its row index fills the gaps.
    """
    numbers = {}
    if config.TYPES_CSV.exists():
        types = pd.read_csv(config.TYPES_CSV, encoding="utf-8")
        for i, row in types.iterrows():
            numbers[dex.normalise(row["Name"])] = i + 1

    for label, entry in dex.load().items():
        if entry.get("pokedex_number"):
            numbers.setdefault(label, entry["pokedex_number"])
    return numbers


def download(url: str, dest: Path, tries: int = 3) -> bool:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pokescanner"})
            data = urllib.request.urlopen(req, timeout=30).read()
            if len(data) < 200:                 # empty or placeholder
                return False
            dest.write_bytes(data)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:                 # this form simply has no sprite
                return False
            if attempt == tries - 1:
                return False
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            if attempt == tries - 1:
                return False
            time.sleep(1.5 * (attempt + 1))
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=config.IMAGES_DIR)
    ap.add_argument("--sources", nargs="*", default=list(SOURCES),
                    choices=list(SOURCES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.05)
    args = ap.parse_args()

    numbers = dex_numbers()
    labels = sorted(dex.load())
    if args.limit:
        labels = labels[:args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    got = skipped = missing = 0
    no_number = []

    for i, label in enumerate(labels, 1):
        number = numbers.get(label)
        if not number:
            no_number.append(label)
            continue
        folder = args.out / label
        folder.mkdir(parents=True, exist_ok=True)

        for source in args.sources:
            dest = folder / f"{label}_{source}.png"
            if dest.exists():
                skipped += 1
                continue
            if download(f"{BASE}/{SOURCES[source].format(n=number)}", dest):
                got += 1
            else:
                missing += 1
            time.sleep(args.delay)

        if i % 50 == 0 or i == len(labels):
            print(f"  {i}/{len(labels)} classes   "
                  f"+{got} new, {skipped} already had, {missing} unavailable",
                  flush=True)

    counts = [len(list((args.out / l).glob('*'))) for l in labels
              if (args.out / l).exists()]
    print(f"\ndownloaded {got} images")
    if no_number:
        print(f"no dex number for {len(no_number)}: {', '.join(no_number[:8])}")
    if counts:
        print(f"images per class now: min {min(counts)}, "
              f"median {sorted(counts)[len(counts) // 2]}, max {max(counts)}")
        print(f"total images: {sum(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
