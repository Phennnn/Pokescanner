"""Train a PokeScanner classifier.

Runs the same way locally and on Colab/Kaggle:

    python model/train.py --arch convnext_tiny --epochs 30
    python model/train.py --arch efficientnet_b2 --resume

What this does differently from the original notebook recipe:

  * Stratified split. The old split shuffled every image globally, so with ~7
    images per class a good number of classes ended up with zero validation
    images and a few with almost no training images. Splitting per class fixes
    both, and makes the reported accuracy mean something.
  * Background augmentation. Sprites get composited into cluttered synthetic
    scenes for a share of the training samples, which is the training-side half
    of the domain-gap fix (the inference-side half lives in pokescanner.vision).
  * Architecture is a flag, not an edit. ConvNeXt-Tiny is the recommended next
    step: it is a 2022 design that holds up better than EfficientNet in the
    low-data regime this project is in.
  * Checkpoints carry their own metadata (arch, image size, val accuracy), so
    the apps can load any of them without being told what they are.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokescanner import config, synth  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# -- Data ---------------------------------------------------------------------
def collect_samples(images_dir: Path, label_to_idx: dict):
    """{label_idx: [paths]} for every class folder that has images."""
    by_class = defaultdict(list)
    for class_dir in sorted(images_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name not in label_to_idx:
            continue
        idx = label_to_idx[class_dir.name]
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                by_class[idx].append(f)
    return by_class


def stratified_split(by_class: dict, val_split: float, seed: int):
    """Split inside each class, so every class appears in train and (if it has
    more than one image) in val."""
    rng = random.Random(seed)
    train, val = [], []
    for idx, paths in by_class.items():
        paths = list(paths)
        rng.shuffle(paths)
        if len(paths) == 1:
            # a single image cannot be in both; training matters more
            train.append((paths[0], idx))
            continue
        n_val = max(1, int(round(len(paths) * val_split)))
        n_val = min(n_val, len(paths) - 1)
        val.extend((p, idx) for p in paths[:n_val])
        train.extend((p, idx) for p in paths[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class PokemonDataset(Dataset):
    """Class-folder dataset with optional synthetic-background augmentation."""

    def __init__(self, samples, transform, train: bool = True,
                 background_prob: float = 0.0, sprite_pool=(), seed: int = 0):
        self.samples = samples
        self.transform = transform
        self.train = train
        self.background_prob = background_prob
        self.sprite_pool = list(sprite_pool)
        self.seed = seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            img = Image.open(path)
        except Exception:
            return self[(index + 1) % len(self)]

        has_alpha = img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info)

        if self.train and has_alpha and self.background_prob > 0 and \
                random.random() < self.background_prob:
            # A fresh scene each epoch: the same sprite is seen on many
            # backgrounds over a run, which is the point.
            rng = random.Random((self.seed, index, random.random()).__hash__())
            img = synth.composite_scene(
                img.convert("RGBA"), size=(320, 320), rng=rng,
                sprites=self.sprite_pool, fill_range=(0.35, 0.85),
                strength=0.7)
        else:
            rgba = img.convert("RGBA")
            canvas = Image.new("RGB", rgba.size, config.CANVAS_COLOR)
            canvas.paste(rgba, mask=rgba.getchannel("A"))
            img = canvas

        return self.transform(img), label


def build_transforms(size: int, train: bool):
    norm = transforms.Normalize(config.MEAN, config.STD)
    if not train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(), norm,
        ])
    return transforms.Compose([
        transforms.Resize((int(size * 1.15), int(size * 1.15))),
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(25)], p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.35, hue=0.08),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply(
            [transforms.GaussianBlur(5, sigma=(0.1, 1.8))], p=0.25),
        transforms.ToTensor(), norm,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


# -- Training pieces ----------------------------------------------------------
def mixup(x, y, alpha: float):
    """Returns mixed inputs and the pair of targets with their weight."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam


def accuracy_topk(logits, target, ks=(1, 5)):
    maxk = max(ks)
    _, pred = logits.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1))
    return [float(correct[:, :k].any(dim=1).sum().item()) for k in ks]


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum = top1 = top5 = seen = 0.0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device)
        logits = model(images)
        loss_sum += float(criterion(logits, targets).item()) * images.size(0)
        a1, a5 = accuracy_topk(logits, targets)
        top1 += a1
        top5 += a5
        seen += images.size(0)
    seen = max(1.0, seen)
    return loss_sum / seen, top1 / seen, top5 / seen


def save_checkpoint(path: Path, model, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), **meta}, path)


# -- Main ---------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="convnext_tiny",
                    choices=sorted(config.ARCHITECTURES))
    ap.add_argument("--images", type=Path, default=config.IMAGES_DIR)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup-epochs", type=int, default=3,
                    help="epochs training the classifier head only")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--background-prob", type=float, default=0.5,
                    help="share of training sprites pasted into random scenes")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit-classes", type=int, default=0,
                    help="debug: train on the first N classes only")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timm_name, native_size = config.ARCHITECTURES[args.arch]
    img_size = args.img_size or native_size
    out_path = args.out or (config.WEIGHTS_DIR / f"best_model_{args.arch}.pth")

    with open(config.LABEL_MAP_PATH, encoding="utf-8") as fh:
        label_map = json.load(fh)
    label_to_idx = label_map["label_to_idx"]
    num_classes = int(label_map["num_classes"])

    by_class = collect_samples(args.images, label_to_idx)
    if args.limit_classes:
        keep = sorted(by_class)[:args.limit_classes]
        by_class = {k: by_class[k] for k in keep}
    if not by_class:
        raise SystemExit(f"No training images found under {args.images}")

    train_samples, val_samples = stratified_split(by_class, args.val_split, args.seed)
    singletons = sum(1 for v in by_class.values() if len(v) == 1)
    sprite_pool = [p for paths in by_class.values() for p in paths[:1]]

    print(f"device        {device}")
    print(f"arch          {args.arch} @ {img_size}px")
    print(f"classes       {len(by_class)} of {num_classes}")
    print(f"train / val   {len(train_samples)} / {len(val_samples)}")
    print(f"singletons    {singletons} classes have only one image "
          f"(train-only, no val)")
    print(f"background    {args.background_prob:.0%} of train sprites get a scene")
    print(f"output        {out_path}\n")

    train_ds = PokemonDataset(train_samples, build_transforms(img_size, True),
                              train=True, background_prob=args.background_prob,
                              sprite_pool=sprite_pool, seed=args.seed)
    val_ds = PokemonDataset(val_samples, build_transforms(img_size, False),
                            train=False)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin,
                              drop_last=len(train_ds) > args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin)

    model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
    start_epoch, best_acc = 0, 0.0
    if args.resume and out_path.exists():
        blob = torch.load(out_path, map_location="cpu", weights_only=False)
        state = blob.get("model", blob)
        model.load_state_dict(state)
        best_acc = float(blob.get("val_accuracy", 0.0))
        start_epoch = int(blob.get("epoch", 0))
        print(f"resumed from {out_path} at epoch {start_epoch}, "
              f"best {best_acc:.2%}\n")
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    classifier = model.get_classifier()
    head_params = {id(p) for p in classifier.parameters()}

    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if id(p) not in head_params],
          "lr": args.lr},
         {"params": list(classifier.parameters()), "lr": args.head_lr}],
        weight_decay=args.weight_decay)

    total_epochs = args.epochs
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr, args.head_lr],
        total_steps=total_epochs * steps_per_epoch, pct_start=0.25)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def set_backbone_frozen(frozen: bool):
        for param in model.parameters():
            param.requires_grad = not frozen
        for param in classifier.parameters():
            param.requires_grad = True

    print("epoch   train loss   val loss   val top1   val top5   lr        time")
    for epoch in range(start_epoch, total_epochs):
        frozen = epoch < args.warmup_epochs
        set_backbone_frozen(frozen)

        model.train()
        started = time.perf_counter()
        running, seen = 0.0, 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            mixed, ya, yb, lam = mixup(images, targets, args.mixup)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(mixed)
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += float(loss.item()) * images.size(0)
            seen += images.size(0)

        train_loss = running / max(1, seen)
        val_loss, val_top1, val_top5 = (
            evaluate(model, val_loader, device, criterion)
            if len(val_ds) else (float("nan"), 0.0, 0.0))
        lr_now = optimizer.param_groups[0]["lr"]
        flag = ""

        # With no validation set at all (every class a singleton) there is
        # nothing to select on, so keep the latest epoch rather than nothing.
        improved = val_top1 > best_acc if len(val_ds) else True
        if improved:
            best_acc = max(best_acc, val_top1)
            flag = "  <- best" if len(val_ds) else "  <- saved"
            save_checkpoint(out_path, model, {
                "arch": args.arch,
                "img_size": img_size,
                "val_accuracy": val_top1,
                "val_top5": val_top5,
                "epoch": epoch + 1,
                "num_classes": num_classes,
                "background_prob": args.background_prob,
            })

        print(f"{epoch+1:3d}/{total_epochs}  {train_loss:10.4f}  "
              f"{val_loss:9.4f}  {val_top1:8.2%}  {val_top5:8.2%}  "
              f"{lr_now:.2e}  {time.perf_counter()-started:5.0f}s"
              f"{'  [frozen]' if frozen else ''}{flag}")

    print(f"\nbest val top-1 {best_acc:.2%} -> {out_path}")
    print(f"Point the apps at it with:  POKESCANNER_WEIGHTS={out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
