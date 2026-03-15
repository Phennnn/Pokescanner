# ── PokéScanner · data/preprocess.py ────────────────────────────────────────
# Step 2b: Image preprocessing + PyTorch Dataset class.
#
# Run this AFTER data_pipeline.py:
#   python data/preprocess.py
#
# What it does:
#   1. Scans data/images/ and builds a label → index map
#   2. Verifies all images can open (catches corrupted files early)
#   3. Defines the PokemonDataset class used in training (Step 3)
#   4. Defines train/val/test transforms with augmentation
#   5. Saves label_map.json for inference (Step 4)
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images"
PROCESSED  = ROOT / "data" / "processed"
MASTER_CSV = PROCESSED / "master.csv"
LABEL_MAP  = PROCESSED / "label_map.json"

# ── Image Config ──────────────────────────────────────────────────────────────
IMG_SIZE   = 224        # EfficientNet-B0 default input size
MEAN       = [0.485, 0.456, 0.406]   # ImageNet mean (we're fine-tuning on it)
STD        = [0.229, 0.224, 0.225]   # ImageNet std


# ── 1. Build Label Map ────────────────────────────────────────────────────────
def build_label_map(images_dir: Path) -> dict:
    """
    Scans the class-folder structure and returns:
        {
          "label_to_idx": {"bulbasaur": 0, "ivysaur": 1, ...},
          "idx_to_label": {"0": "bulbasaur", "1": "ivysaur", ...},
          "num_classes": 800
        }
    """
    classes = sorted([
        d.name for d in images_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    label_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_label = {idx: cls for cls, idx in label_to_idx.items()}

    label_map = {
        "label_to_idx": label_to_idx,
        "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
        "num_classes": len(classes)
    }

    with open(LABEL_MAP, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"[1/3] Label map built → {len(classes)} classes")
    print(f"      Saved to {LABEL_MAP}")
    return label_map


# ── 2. Verify Images ──────────────────────────────────────────────────────────
def verify_images(images_dir: Path) -> list[Path]:
    """
    Walks every image file and tries to open it.
    Returns list of valid image paths.
    Logs corrupted/unreadable files to data/processed/bad_images.txt
    """
    print("[2/3] Verifying images...")

    all_images = list(images_dir.rglob("*.png")) + \
                 list(images_dir.rglob("*.jpg")) + \
                 list(images_dir.rglob("*.jpeg"))

    good, bad = [], []

    for img_path in tqdm(all_images, desc="    Checking"):
        try:
            with Image.open(img_path) as img:
                img.verify()           # detects truncated files
            good.append(img_path)
        except Exception as e:
            bad.append((str(img_path), str(e)))

    if bad:
        bad_log = PROCESSED / "bad_images.txt"
        with open(bad_log, "w") as f:
            for path, err in bad:
                f.write(f"{path}\t{err}\n")
        print(f"    ⚠  {len(bad)} corrupted images logged → {bad_log}")

    print(f"    → {len(good)} valid images, {len(bad)} bad")
    return good


# ── 3. Transforms ─────────────────────────────────────────────────────────────
def get_transforms(split: str) -> transforms.Compose:
    """
    Returns augmentation pipeline for each split.

    Train:  heavy augmentation to help the model generalise from clean sprites
            to real-world photos of cards, figures, and screens.
            This is the key technique for bridging the domain gap.

    Val/Test: just resize + normalize (no augmentation).
    """

    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),

            # ── simulate real-world conditions ─────────────────────────────
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.4,   # cards vary in lighting
                contrast=0.4,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # slight position shifts
                scale=(0.85, 1.15)
            ),
            transforms.RandomGrayscale(p=0.05),  # handle b&w photos

            # ── convert + normalize ─────────────────────────────────────────
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),

            # ── random erasing: simulates partial occlusion (finger, glare) ─
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])

    else:  # val or test
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])


# ── 4. Dataset Class ──────────────────────────────────────────────────────────
class PokemonDataset(Dataset):
    """
    PyTorch Dataset for Pokémon image classification.

    Folder structure expected:
        data/images/
            bulbasaur/
                Bulbasaur.png
            ivysaur/
                Ivysaur.png
            ...

    Usage:
        label_map = json.load(open("data/processed/label_map.json"))
        dataset = PokemonDataset(
            images_dir = Path("data/images"),
            label_map  = label_map,
            split      = "train",
            val_split  = 0.15,
            test_split = 0.05,
            seed       = 42
        )
    """

    def __init__(
        self,
        images_dir: Path,
        label_map:  dict,
        split:      str  = "train",    # "train" | "val" | "test"
        val_split:  float = 0.15,
        test_split: float = 0.05,
        seed:       int   = 42,
        transform = None
    ):
        self.images_dir  = images_dir
        self.label_to_idx = label_map["label_to_idx"]
        self.transform   = transform or get_transforms(split)
        self.split       = split

        # collect all (image_path, label_idx) pairs
        all_samples = self._collect_samples()

        # deterministic train/val/test split
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(all_samples))
        n = len(indices)
        n_test  = int(n * test_split)
        n_val   = int(n * val_split)
        n_train = n - n_test - n_val

        if split == "train":
            selected = indices[:n_train]
        elif split == "val":
            selected = indices[n_train:n_train + n_val]
        else:
            selected = indices[n_train + n_val:]

        self.samples = [all_samples[i] for i in selected]
        print(f"    PokemonDataset [{split}] → {len(self.samples)} samples")

    def _collect_samples(self) -> list[tuple[Path, int]]:
        samples = []
        for class_dir in sorted(self.images_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            if label not in self.label_to_idx:
                continue
            idx = self.label_to_idx[label]
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    samples.append((img_file, idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert("RGBA")

        # handle transparent PNG backgrounds (sprites have alpha channels)
        # paste onto white background before normalizing
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
        img = bg

        if self.transform:
            img = self.transform(img)

        return img, label


# ── 5. Quick sanity check ─────────────────────────────────────────────────────
def run():
    print("\n🔴 PokéScanner · Preprocessing")
    print("=" * 45)

    if not IMAGES_DIR.exists() or not any(IMAGES_DIR.iterdir()):
        print(f"❌ No images found at {IMAGES_DIR}")
        print("   Run data_pipeline.py first")
        return

    label_map = build_label_map(IMAGES_DIR)
    valid_images = verify_images(IMAGES_DIR)

    if not valid_images:
        print("❌ No valid images found — check data_pipeline.py ran correctly")
        return

    # quick dataloader smoke test
    print("[3/3] Running DataLoader smoke test...")
    try:
        dataset = PokemonDataset(IMAGES_DIR, label_map, split="train")
        loader  = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
        imgs, labels = next(iter(loader))
        print(f"    → Batch shape : {imgs.shape}")   # [8, 3, 224, 224]
        print(f"    → Labels      : {labels.tolist()}")
        print(f"    → dtype       : {imgs.dtype}")
    except Exception as e:
        print(f"    ❌ DataLoader test failed: {e}")
        return

    print("\n✅ Preprocessing complete!")
    print(f"   Classes    : {label_map['num_classes']}")
    print(f"   Label map  : data/processed/label_map.json")
    print(f"\n   Next → Step 3 · model/train.py (run on Colab with GPU)")


if __name__ == "__main__":
    run()
