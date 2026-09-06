"""Zip the dataset for upload to Colab, with paths that survive the trip.

Windows separates paths with backslashes, and a zip built naively on Windows
stores them that way. Linux unzip does not treat a backslash as a separator, so
the archive extracts into one flat directory of files literally named
``data\\images\\pikachu\\pikachu.png`` and training then finds zero classes.

Every path written here is normalised to forward slashes, which is what the zip
format actually specifies, so the same archive extracts correctly on Windows,
macOS, Linux and Colab.

    python tools/make_colab_bundle.py
    python tools/make_colab_bundle.py --out C:/tmp/pokescanner_data.zip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokescanner import config  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=config.ROOT / "pokescanner_data.zip")
    ap.add_argument("--images", type=Path, default=config.IMAGES_DIR)
    ap.add_argument("--compress", type=int, default=6, choices=range(0, 10))
    args = ap.parse_args()

    if not args.images.exists():
        raise SystemExit(f"No images at {args.images}")

    files: list[tuple[Path, str]] = []
    for path in sorted(args.images.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            rel = path.relative_to(config.ROOT)
            files.append((path, rel.as_posix()))

    for extra in (config.LABEL_MAP_PATH, config.STATS_CSV, config.TYPES_CSV):
        if extra.exists():
            files.append((extra, extra.relative_to(config.ROOT).as_posix()))

    if not files:
        raise SystemExit("Nothing to bundle")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=args.compress) as zf:
        for i, (path, arcname) in enumerate(files, 1):
            # arcname is already POSIX, which is what the zip spec wants and
            # what makes this extract correctly on Colab
            assert "\\" not in arcname, arcname
            zf.write(path, arcname)
            if i % 1000 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    size_mb = args.out.stat().st_size / 1e6
    classes = len({f[1].split("/")[2] for f in files
                   if f[1].startswith("data/images/") and f[1].count("/") > 2})
    print(f"\n{args.out}")
    print(f"{len(files)} files, {classes} classes, {size_mb:.1f} MB")
    if size_mb > 100:
        print("Over 100 MB, so the Colab upload widget will be slow. Consider "
              "putting it in Google Drive and mounting that instead, or use "
              "Option A in the notebook and skip the upload entirely.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
