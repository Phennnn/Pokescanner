# ── PokéScanner · app/scanner.py ─────────────────────────────────────────────
# Step 4: Real-time webcam inference.
# Run from project root:
#   python app/scanner.py
#
# Controls:
#   SPACE  → capture current frame and identify Pokémon
#   A      → add detected Pokémon to team
#   C      → clear team
#   Q      → quit
# ─────────────────────────────────────────────────────────────────────────────

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
WEIGHTS    = ROOT / "model" / "weights" / "best_model.pth"
LABEL_MAP  = ROOT / "data" / "processed" / "label_map.json"
STATS_CSV  = ROOT / "data" / "raw" / "pokemon_stats.csv"

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]
CONF_THRESHOLD = 0.15          # min confidence to show a result
TOP_K          = 3             # show top-3 candidates

# ── Type colours (BGR for OpenCV) ─────────────────────────────────────────────
TYPE_COLORS = {
    "fire":     (0,  100, 255),
    "water":    (255, 150,  50),
    "grass":    (50,  200,  50),
    "electric": (0,  220, 255),
    "psychic":  (180,  50, 255),
    "ice":      (255, 220, 150),
    "dragon":   (180,  50, 100),
    "dark":     (80,   50,  50),
    "fairy":    (180, 130, 255),
    "normal":   (150, 150, 150),
    "fighting": (50,   50, 200),
    "flying":   (220, 180, 100),
    "poison":   (130,  50, 180),
    "ground":   (50,  150, 200),
    "rock":     (80,  120, 150),
    "bug":      (50,  180,  80),
    "ghost":    (130,  80, 100),
    "steel":    (180, 180, 180),
}


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(weights_path: Path, num_classes: int, device: torch.device):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    model.to(device)
    print(f"✅ Model loaded from {weights_path}")
    return model


# ── Transforms ────────────────────────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ── Stats loader ──────────────────────────────────────────────────────────────
def load_stats(stats_csv: Path) -> dict:
    """Returns dict of {name_clean: {hp, attack, defense, speed, type1, type2}}"""
    import pandas as pd
    if not stats_csv.exists():
        return {}
    df = pd.read_csv(stats_csv)
    stats = {}
    for _, row in df.iterrows():
        key = str(row["name"]).strip().lower().replace(" ", "-").replace("'", "")
        stats[key] = {
            "hp":      int(row.get("hp", 0)),
            "attack":  int(row.get("attack", 0)),
            "defense": int(row.get("defense", 0)),
            "speed":   int(row.get("speed", 0)),
            "type1":   str(row.get("type1", "")).lower(),
            "type2":   str(row.get("type2", "")).lower(),
            "legendary": bool(row.get("is_legendary", 0)),
        }
    return stats


# ── TTA Transforms (Test-Time Augmentation) ───────────────────────────────────
# We run the same image through 6 different augmentations and average the
# predictions. This simulates seeing the Pokémon from slightly different
# angles/crops/flips and makes the model much more stable.
tta_transforms = [
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 30, IMG_SIZE + 30)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
]


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(frame_bgr, model, idx_to_label, device):
    """
    Takes a BGR OpenCV frame, returns list of (name, confidence) sorted by conf.
    Uses Test-Time Augmentation (TTA): runs 6 augmented versions of the image
    and averages the probabilities for a more stable prediction.
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # run all TTA transforms and stack into one batch
    tensors = torch.stack([t(pil_img) for t in tta_transforms]).to(device)

    with torch.no_grad():
        logits = model(tensors)               # [6, num_classes]
        probs  = F.softmax(logits, dim=1)     # [6, num_classes]
        probs  = probs.mean(dim=0)            # average across 6 augmentations

    top_probs, top_idxs = torch.topk(probs, TOP_K)
    results = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_idxs.cpu().numpy()):
        name = idx_to_label.get(str(idx), f"unknown_{idx}")
        results.append((name, float(prob)))
    return results


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_scanner_overlay(frame, scanning: bool):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    box_size = min(w, h) // 2

    x1 = cx - box_size // 2
    y1 = cy - box_size // 2
    x2 = cx + box_size // 2
    y2 = cy + box_size // 2

    color = (0, 80, 220) if not scanning else (0, 220, 80)
    corner = 24
    thick  = 2

    # corner brackets
    for (px, py, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (px, py), (px + dx*corner, py), color, thick)
        cv2.line(frame, (px, py), (px, py + dy*corner), color, thick)

    # scan line animation
    if scanning:
        t   = time.time() % 2 / 2
        sy  = int(y1 + (y2 - y1) * abs(2*t - 1))
        cv2.line(frame, (x1, sy), (x2, sy), (0, 255, 100), 1)

    return (x1, y1, x2, y2)


def draw_result_panel(frame, predictions, stats_db, team):
    h, w = frame.shape[:2]
    panel_x = w - 280
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, h), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    y = 20
    cv2.putText(frame, "POKESCANNER", (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 220), 1)
    y += 30

    if not predictions:
        cv2.putText(frame, "Press SPACE to scan", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        y += 20
    else:
        name, conf = predictions[0]
        display_name = name.replace("-", " ").title()

        # confidence bar
        conf_w = int(240 * conf)
        cv2.rectangle(frame, (panel_x + 10, y), (panel_x + 250, y + 14), (40, 40, 60), -1)
        cv2.rectangle(frame, (panel_x + 10, y), (panel_x + 10 + conf_w, y + 14),
                      (0, 200, 80) if conf > 0.5 else (0, 160, 220), -1)
        cv2.putText(frame, f"{conf*100:.1f}%", (panel_x + 15, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        y += 22

        cv2.putText(frame, display_name, (panel_x + 10, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y += 28

        # stats
        s = stats_db.get(name, {})
        if s:
            type1 = s.get("type1", "")
            type2 = s.get("type2", "")
            t1_col = TYPE_COLORS.get(type1, (150,150,150))
            t2_col = TYPE_COLORS.get(type2, (150,150,150))

            cv2.rectangle(frame, (panel_x+10, y), (panel_x+90, y+16), t1_col, -1)
            cv2.putText(frame, type1.upper(), (panel_x+14, y+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
            if type2 and type2 != "nan":
                cv2.rectangle(frame, (panel_x+96, y), (panel_x+176, y+16), t2_col, -1)
                cv2.putText(frame, type2.upper(), (panel_x+100, y+12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
            y += 24

            for stat_name, key in [("HP","hp"),("ATK","attack"),("DEF","defense"),("SPD","speed")]:
                val = s.get(key, 0)
                bar_w = int(200 * min(val / 200, 1.0))
                cv2.putText(frame, stat_name, (panel_x+10, y+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140,140,140), 1)
                cv2.rectangle(frame, (panel_x+44, y+2), (panel_x+244, y+12), (40,40,60), -1)
                cv2.rectangle(frame, (panel_x+44, y+2), (panel_x+44+bar_w, y+12), (0,180,80), -1)
                cv2.putText(frame, str(val), (panel_x+250, y+11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200,200,200), 1)
                y += 18

            if s.get("legendary"):
                cv2.putText(frame, "* LEGENDARY", (panel_x+10, y+12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1)
                y += 20

        y += 10
        cv2.putText(frame, "Other candidates:", (panel_x+10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100,100,100), 1)
        y += 16
        for alt_name, alt_conf in predictions[1:]:
            cv2.putText(frame, f"{alt_name.replace('-',' ').title()} {alt_conf*100:.1f}%",
                        (panel_x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100,100,100), 1)
            y += 14

    # team display
    y = h - 180
    cv2.putText(frame, f"TEAM ({len(team)}/6)", (panel_x+10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,80,220), 1)
    y += 18
    for i, mon in enumerate(team):
        name_display = mon.replace("-", " ").title()
        cv2.putText(frame, f"{i+1}. {name_display}", (panel_x+10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
        y += 16

    # controls
    y = h - 25
    cv2.putText(frame, "SPACE:scan  A:add  C:clear  Q:quit", (panel_x+6, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80,80,80), 1)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("\n🔴 PokéScanner · Webcam Inference")
    print("=" * 40)

    # load label map
    if not LABEL_MAP.exists():
        print(f"❌ label_map.json not found at {LABEL_MAP}")
        return
    with open(LABEL_MAP) as f:
        label_map = json.load(f)
    idx_to_label = label_map["idx_to_label"]
    num_classes  = label_map["num_classes"]

    # load model
    if not WEIGHTS.exists():
        print(f"❌ Model weights not found at {WEIGHTS}")
        print("   Copy best_model.pth from Colab to model/weights/")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    model = load_model(WEIGHTS, num_classes, device)

    # load stats
    print("   Loading Pokémon stats...")
    stats_db = load_stats(STATS_CSV)
    print(f"   Stats loaded for {len(stats_db)} Pokémon")

    # open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    print("✅ Webcam opened — press SPACE to scan!\n")

    predictions = []
    team        = []
    scanning    = False
    scan_timer  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # mirror

        # scanner overlay
        scanning_anim = (time.time() - scan_timer) < 0.5
        draw_scanner_overlay(frame, scanning_anim)

        # result panel
        draw_result_panel(frame, predictions, stats_db, team)

        cv2.imshow("PokéScanner", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            # multi-frame voting: grab 5 frames, average predictions for stability
            scan_timer = time.time()
            print("Scanning (5-frame vote)...")
            from collections import defaultdict
            vote_scores = defaultdict(float)
            for _ in range(5):
                ret2, frame2 = cap.read()
                if not ret2: continue
                frame2 = cv2.flip(frame2, 1)
                preds = predict(frame2, model, idx_to_label, device)
                for pname, pconf in preds:
                    vote_scores[pname] += pconf
            total_score = sum(vote_scores.values()) or 1.0
            predictions = sorted(vote_scores.items(), key=lambda x: x[1], reverse=True)
            predictions = [(n, s / total_score) for n, s in predictions[:TOP_K]]
            if predictions:
                name, conf = predictions[0]
                print(f"   -> {name.replace('-',' ').title()} ({conf*100:.1f}% vote share)")

        elif key == ord('a'):
            if predictions:
                name, conf = predictions[0]
                if conf >= CONF_THRESHOLD:
                    if len(team) >= 6:
                        print("⚠️  Team is full (max 6)!")
                    elif name in team:
                        print(f"⚠️  {name} is already in your team!")
                    else:
                        team.append(name)
                        print(f"✅ Added {name.replace('-',' ').title()} to team! ({len(team)}/6)")
                        if len(team) == 6:
                            print(f"   Team: {', '.join(t.replace('-',' ').title() for t in team)}")

        elif key == ord('c'):
            team = []
            print("🗑️  Team cleared")

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 PokéScanner closed")
    if team:
        print(f"   Final team: {', '.join(t.replace('-',' ').title() for t in team)}")


if __name__ == "__main__":
    main()