# ── PokéScanner · data/data_pipeline.py ─────────────────────────────────────
# Step 2: Load, clean, and merge both Kaggle datasets into one master CSV.
#
# Datasets needed (download from Kaggle and place in data/raw/):
#   1. rounakbanik/pokemon          → pokemon.csv
#   2. vishalsubbiah/pokemon-images-and-types → images/ folder + pokemon.csv
#
# Quick Kaggle download (requires kaggle API token in ~/.kaggle/kaggle.json):
#   kaggle datasets download -d rounakbanik/pokemon -p data/raw/ --unzip
#   kaggle datasets download -d vishalsubbiah/pokemon-images-and-types -p data/raw/ --unzip
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent   
RAW_DIR     = ROOT / "raw"
PROCESSED   = ROOT / "processed"
IMAGES_OUT  = ROOT / "images"

STATS_CSV = RAW_DIR / "pokemon_stats.csv"   # rounakbanik (full stats)
IMG_CSV   = RAW_DIR / "pokemon_types.csv"   # vishalsubbiah (types + names)
IMG_DIR   = RAW_DIR / "images"              # vishalsubbiah images                         # vishalsubbiah images folder

PROCESSED.mkdir(parents=True, exist_ok=True)
IMAGES_OUT.mkdir(parents=True, exist_ok=True)


# ── 1. Load Stats Dataset (rounakbanik) ──────────────────────────────────────
def load_stats(path: Path) -> pd.DataFrame:
    """
    Loads the rounakbanik stats CSV.
    Keeps the columns we actually need for the app.
    """
    print("[1/4] Loading stats dataset...")

    df = pd.read_csv(path)

    keep_cols = [
        "name", "pokedex_number", "type1", "type2",
        "hp", "attack", "defense", "sp_attack", "sp_defense", "speed",
        "base_total", "generation", "is_legendary",
        "against_fire", "against_water", "against_electric", "against_grass",
        "against_ice", "against_fight", "against_poison", "against_ground",
        "against_flying", "against_psychic", "against_bug", "against_rock",
        "against_ghost", "against_dragon", "against_dark", "against_steel",
        "against_fairy", "capture_rate", "base_egg_steps",
        "height_m", "weight_kg", "classfication", "abilities",
    ]

    # only keep columns that actually exist in this version of the CSV
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # normalize name for joining
    df["name_clean"] = df["name"].apply(normalize_name)

    print(f"    → {len(df)} Pokémon loaded, {len(df.columns)} columns kept")
    return df


# ── 2. Load Image/Type Dataset (vishalsubbiah) ───────────────────────────────
def load_image_labels(path: Path) -> pd.DataFrame:
    """
    Loads the vishalsubbiah type CSV.
    Gives us the canonical name ↔ image filename mapping.
    """
    print("[2/4] Loading image labels dataset...")

    df = pd.read_csv(path)

    # expected columns: Name, Type1, Type2
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"name": "img_name", "type1": "img_type1", "type2": "img_type2"})
    df["name_clean"] = df["img_name"].apply(normalize_name)

    print(f"    → {len(df)} Pokémon image labels loaded")
    return df


# ── 3. Merge ──────────────────────────────────────────────────────────────────
def merge_datasets(stats: pd.DataFrame, img_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Left-joins stats onto image labels using the cleaned name as key.
    Image labels are the left table because we need every Pokémon
    that has an image to be in the training set.
    """
    print("[3/4] Merging datasets...")

    merged = img_labels.merge(stats, on="name_clean", how="left")

    # fill missing type2 with empty string (single-type Pokémon)
    merged["type2"] = merged.get("type2", pd.Series([""] * len(merged))).fillna("")
    merged["img_type2"] = merged["img_type2"].fillna("")

    # use stats types as primary, fall back to img types
    merged["final_type1"] = merged["type1"].fillna(merged["img_type1"])
    merged["final_type2"] = merged["type2"].fillna(merged["img_type2"])

    # flag rows where the join failed (no stats data found)
    unmatched = merged["hp"].isna().sum()
    if unmatched > 0:
        print(f"    ⚠  {unmatched} Pokémon had no stats match — check name mismatches")
        merged[merged["hp"].isna()][["img_name", "name_clean"]].to_csv(
            PROCESSED / "unmatched.csv", index=False
        )
        print(f"       Saved to data/processed/unmatched.csv for manual review")

    print(f"    → {len(merged)} rows in merged dataset")
    return merged


# ── 4. Organise Images ────────────────────────────────────────────────────────
def organise_images(merged: pd.DataFrame, img_dir: Path, out_dir: Path):
    """
    Copies images into a class-folder structure ready for PyTorch ImageFolder:
        data/images/<pokemon_name>/<pokemon_name>.png
    This is the standard layout for torchvision.datasets.ImageFolder.
    """
    print("[4/4] Organising images into class folders...")

    if not img_dir.exists():
        print(f"    ⚠  Image directory not found at {img_dir} — skipping copy")
        print(f"       Make sure you've downloaded the vishalsubbiah dataset")
        return

    copied, skipped = 0, 0

    for _, row in merged.iterrows():
        name = row["img_name"]
        class_dir = out_dir / normalize_name(name)
        class_dir.mkdir(parents=True, exist_ok=True)

        # images in vishalsubbiah dataset are named e.g. "Bulbasaur.png"
        for ext in [".png", ".jpg", ".jpeg"]:
            src = img_dir / f"{name}{ext}"
            if src.exists():
                dst = class_dir / f"{name}{ext}"
                shutil.copy2(src, dst)
                copied += 1
                break
        else:
            skipped += 1

    print(f"    → {copied} images copied, {skipped} missing")


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_name(name: str) -> str:
    """
    Lowercase, strip whitespace, remove special characters.
    Handles edge cases like:
        'Nidoran♀' → 'nidoran-f'
        'Mr. Mime'  → 'mr-mime'
        'Farfetch'd' → 'farfetchd'
    """
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    name = name.replace("♀", "-f").replace("♂", "-m")
    name = name.replace("'", "").replace("'", "")
    name = re.sub(r"[^a-z0-9\-]", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return name


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("\n🔴 PokéScanner · Data Pipeline")
    print("=" * 45)

    # check raw files exist
    if not STATS_CSV.exists():
        print(f"\n❌ Stats CSV not found at: {STATS_CSV}")
        print("   Run: kaggle datasets download -d rounakbanik/pokemon -p data/raw/ --unzip")
        return

    if not IMG_CSV.exists():
        print(f"\n❌ Image CSV not found at: {IMG_CSV}")
        print("   Run: kaggle datasets download -d vishalsubbiah/pokemon-images-and-types -p data/raw/ --unzip")
        return

    stats       = load_stats(STATS_CSV)
    img_labels  = load_image_labels(IMG_CSV)
    merged      = merge_datasets(stats, img_labels)

    # save master CSV
    out_path = PROCESSED / "master.csv"
    merged.to_csv(out_path, index=False)
    print(f"\n✅ Master dataset saved → {out_path}")
    print(f"   Shape: {merged.shape}")

    organise_images(merged, IMG_DIR, IMAGES_OUT)

    print("\n🎉 Pipeline complete!")
    print(f"   Images ready at : data/images/<pokemon_name>/")
    print(f"   Master CSV at   : data/processed/master.csv")
    print(f"\n   Next → run preprocess.py to verify images + build label map")


if __name__ == "__main__":
    run()
