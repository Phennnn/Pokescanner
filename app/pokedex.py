# ── PokéScanner · app/pokedex.py ─────────────────────────────────────────────
# Anime-style Pokédex UI — Flask backend
# Run from project root:
#   python app/pokedex.py
# Opens http://localhost:5000
# ─────────────────────────────────────────────────────────────────────────────

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from flask import Flask, jsonify, render_template_string, request
from PIL import Image
from torchvision import transforms

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
WEIGHTS   = ROOT / "model" / "weights" / "best_model_b2.pth"
LABEL_MAP = ROOT / "data" / "processed" / "label_map.json"
STATS_CSV = ROOT / "data" / "raw" / "pokemon_stats.csv"

IMG_SIZE  = 260
MEAN      = [0.485, 0.456, 0.406]
STD       = [0.229, 0.224, 0.225]
TOP_K     = 3

# ── TTA ───────────────────────────────────────────────────────────────────────
tta_transforms = [
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),                              transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    transforms.Compose([transforms.Resize((IMG_SIZE+20, IMG_SIZE+20)), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    transforms.Compose([transforms.Resize((IMG_SIZE+30, IMG_SIZE+30)), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
]

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
with open(LABEL_MAP) as f:
    label_map = json.load(f)
idx_to_label = label_map["idx_to_label"]
NUM_CLASSES  = label_map["num_classes"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = timm.create_model("efficientnet_b2", pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval().to(device)
print(f"Model ready on {device}")

# ── Stats ─────────────────────────────────────────────────────────────────────
stats_db = {}
if STATS_CSV.exists():
    df = pd.read_csv(STATS_CSV)
    for _, row in df.iterrows():
        key = str(row["name"]).strip().lower().replace(" ", "-").replace("'","").replace("'","")
        stats_db[key] = {
            "hp":          int(row.get("hp", 0)),
            "attack":      int(row.get("attack", 0)),
            "defense":     int(row.get("defense", 0)),
            "sp_attack":   int(row.get("sp_attack", 0)),
            "sp_defense":  int(row.get("sp_defense", 0)),
            "speed":       int(row.get("speed", 0)),
            "base_total":  int(row.get("base_total", 0)),
            "type1":       str(row.get("type1", "")).lower().strip(),
            "type2":       str(row.get("type2", "")).lower().strip(),
            "legendary":   bool(row.get("is_legendary", 0)),
            "generation":  int(row.get("generation", 0)),
            "pokedex_number": int(row.get("pokedex_number", 0)),
            "capture_rate": str(row.get("capture_rate", "?")),
            "classfication": str(row.get("classfication", "")),
            "height_m":    float(row.get("height_m", 0) or 0),
            "weight_kg":   float(row.get("weight_kg", 0) or 0),
        }

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("image", "")
    if not data:
        return jsonify({"error": "no image"}), 400

    # decode base64 image from browser
    img_bytes = base64.b64decode(data.split(",")[1] if "," in data else data)
    pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    tensors = torch.stack([t(pil_img) for t in tta_transforms]).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensors), dim=1).mean(dim=0)

    top_probs, top_idxs = torch.topk(probs, TOP_K)
    results = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_idxs.cpu().numpy()):
        name = idx_to_label[str(idx)]
        s    = stats_db.get(name, {})
        results.append({
            "name":       name,
            "confidence": round(float(prob) * 100, 1),
            "stats":      s,
        })

    return jsonify({"predictions": results})


# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pokédex</title>
<link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323:wght@400&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --red:       #cc1c1c;
    --red-dark:  #8b0000;
    --red-light: #e83030;
    --red-shine: #ff6060;
    --blue:      #1a3a8f;
    --blue-dark: #0d1f5c;
    --blue-mid:  #1e4aa8;
    --hinge:     #2a2a2a;
    --screen-bg: #0a1a0a;
    --phosphor:  #39ff14;
    --phosphor2: #7fff00;
    --amber:     #ffb000;
    --crt-lines: rgba(0,0,0,0.18);
    --btn-gray:  #3a3a3a;
  }

  body {
    background: #1a1a1a;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    font-family: 'Press Start 2P', monospace;
    overflow: hidden;
  }

  /* ── Device shell ── */
  .pokedex {
    display: flex;
    width: 900px;
    height: 560px;
    filter: drop-shadow(0 30px 60px rgba(0,0,0,.9));
    position: relative;
  }

  /* ── LEFT HALF ── */
  .left-half {
    width: 420px;
    background: linear-gradient(145deg, var(--red-light) 0%, var(--red) 40%, var(--red-dark) 100%);
    border-radius: 20px 0 0 20px;
    padding: 22px 18px 22px 22px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    position: relative;
    border: 2px solid #ff8080;
    border-right: none;
  }

  /* plastic texture */
  .left-half::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 20px 0 0 20px;
    background: repeating-linear-gradient(
      135deg,
      transparent,
      transparent 2px,
      rgba(255,255,255,0.015) 2px,
      rgba(255,255,255,0.015) 4px
    );
    pointer-events: none;
  }

  /* shine strip */
  .left-half::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 45%;
    border-radius: 20px 0 0 0;
    background: linear-gradient(180deg, rgba(255,255,255,0.12) 0%, transparent 100%);
    pointer-events: none;
  }

  /* ── top indicator row ── */
  .indicator-row {
    display: flex;
    align-items: center;
    gap: 10px;
    z-index: 2;
  }

  .big-light {
    width: 42px; height: 42px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #88eeff, #0088cc);
    border: 3px solid #004466;
    box-shadow: 0 0 0 3px #002233, 0 0 12px #0af;
    flex-shrink: 0;
    transition: box-shadow .15s;
  }
  .big-light.scanning {
    animation: bigblink .4s steps(1) infinite;
  }
  @keyframes bigblink {
    0%   { box-shadow: 0 0 0 3px #002233, 0 0 28px #0ff, 0 0 50px #0ff; background: radial-gradient(circle at 35% 35%, #ffffff, #00ccff); }
    50%  { box-shadow: 0 0 0 3px #002233, 0 0 6px #0af; background: radial-gradient(circle at 35% 35%, #88eeff, #0088cc); }
    100% { box-shadow: 0 0 0 3px #002233, 0 0 28px #0ff, 0 0 50px #0ff; }
  }

  .small-lights { display: flex; gap: 5px; }
  .dot {
    width: 12px; height: 12px; border-radius: 50%;
    border: 1.5px solid rgba(0,0,0,.4);
  }
  .dot.r { background: #ff4444; box-shadow: 0 0 4px #f00; }
  .dot.y { background: #ffcc00; box-shadow: 0 0 4px #fa0; }
  .dot.g { background: #44ff44; box-shadow: 0 0 4px #0f0; animation: gpulse 2s ease-in-out infinite; }
  @keyframes gpulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  /* ── main screen ── */
  .screen-wrap {
    flex: 1;
    background: #111;
    border-radius: 8px;
    padding: 8px;
    border: 3px solid #1a1a1a;
    box-shadow: inset 0 0 12px rgba(0,0,0,.8), 0 0 0 2px #333;
    position: relative;
    overflow: hidden;
    z-index: 2;
  }

  .screen-inner {
    width: 100%;
    height: 100%;
    background: var(--screen-bg);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
  }

  #webcam {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    transform: scaleX(-1);
  }

  /* CRT scanlines */
  .scanlines {
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
      to bottom,
      transparent 0px,
      transparent 3px,
      var(--crt-lines) 3px,
      var(--crt-lines) 4px
    );
    pointer-events: none;
    z-index: 3;
  }

  /* scan beam */
  .scan-beam {
    position: absolute;
    left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--phosphor), transparent);
    opacity: 0;
    z-index: 4;
    pointer-events: none;
  }
  .scan-beam.active {
    animation: beam .6s ease-in-out 3;
  }
  @keyframes beam {
    0%   { top: 0%;    opacity: .9; }
    100% { top: 100%;  opacity: 0; }
  }

  /* corner brackets */
  .bracket { position: absolute; width: 18px; height: 18px; z-index: 5; }
  .bracket.tl { top: 8px;  left: 8px;  border-top: 2px solid var(--phosphor); border-left: 2px solid var(--phosphor); }
  .bracket.tr { top: 8px;  right: 8px; border-top: 2px solid var(--phosphor); border-right: 2px solid var(--phosphor); }
  .bracket.bl { bottom: 8px; left: 8px;  border-bottom: 2px solid var(--phosphor); border-left: 2px solid var(--phosphor); }
  .bracket.br { bottom: 8px; right: 8px; border-bottom: 2px solid var(--phosphor); border-right: 2px solid var(--phosphor); }

  /* crosshair */
  .crosshair {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    z-index: 4; pointer-events: none;
  }
  .crosshair::before, .crosshair::after {
    content: ''; position: absolute; background: rgba(57,255,20,.25);
  }
  .crosshair::before { width: 1px; height: 40%; }
  .crosshair::after  { width: 40%; height: 1px; }

  /* ── bottom buttons ── */
  .btn-row {
    display: flex;
    gap: 8px;
    align-items: center;
    z-index: 2;
  }

  .scan-btn {
    flex: 1;
    padding: 10px 0;
    background: linear-gradient(180deg, #222 0%, #111 100%);
    border: 2px solid #444;
    border-bottom: 3px solid #000;
    border-radius: 6px;
    color: var(--phosphor);
    font-family: 'Press Start 2P', monospace;
    font-size: 9px;
    cursor: pointer;
    letter-spacing: 1px;
    transition: .1s;
    text-shadow: 0 0 8px var(--phosphor);
  }
  .scan-btn:active { transform: translateY(2px); border-bottom-width: 1px; }
  .scan-btn:hover  { background: linear-gradient(180deg, #333 0%, #1a1a1a 100%); }

  .dpad {
    width: 50px; height: 50px;
    background: linear-gradient(145deg, #444, #222);
    border-radius: 4px;
    border: 2px solid #555;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: repeat(3, 1fr);
    gap: 1px;
    flex-shrink: 0;
  }
  .dpad-btn {
    background: #333;
    border-radius: 2px;
    cursor: pointer;
  }
  .dpad-btn:nth-child(2),.dpad-btn:nth-child(4),.dpad-btn:nth-child(6),.dpad-btn:nth-child(8) { background: #3a3a3a; }
  .dpad-btn:nth-child(5) { background: #222; border-radius: 50%; }

  /* ── HINGE ── */
  .hinge {
    width: 22px;
    background: linear-gradient(90deg, #1a1a1a, #3a3a3a, #1a1a1a);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 8px;
    border-top: 2px solid #555;
    border-bottom: 2px solid #555;
    flex-shrink: 0;
  }
  .hinge-screw {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #888, #333);
    border: 1px solid #555;
  }

  /* ── RIGHT HALF ── */
  .right-half {
    flex: 1;
    background: linear-gradient(145deg, #2a50b8 0%, var(--blue) 50%, var(--blue-dark) 100%);
    border-radius: 0 20px 20px 0;
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    border: 2px solid #6080e0;
    border-left: none;
    position: relative;
    overflow: hidden;
  }

  .right-half::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 0 20px 20px 0;
    background: repeating-linear-gradient(
      135deg, transparent, transparent 2px,
      rgba(255,255,255,0.02) 2px, rgba(255,255,255,0.02) 4px
    );
    pointer-events: none;
  }

  .right-half::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 50%;
    border-radius: 0 20px 0 0;
    background: linear-gradient(180deg, rgba(255,255,255,0.1) 0%, transparent 100%);
    pointer-events: none;
  }

  /* ── info screen ── */
  .info-screen {
    background: var(--screen-bg);
    border-radius: 6px;
    border: 3px solid #0a0a1a;
    box-shadow: inset 0 0 16px rgba(0,0,0,.9), 0 0 0 2px #334;
    padding: 12px;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 8px;
    position: relative;
    z-index: 2;
    overflow: hidden;
  }

  /* CRT scanlines on info screen too */
  .info-screen::after {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
      to bottom, transparent 0px, transparent 3px,
      rgba(0,0,0,.15) 3px, rgba(0,0,0,.15) 4px
    );
    pointer-events: none;
    z-index: 10;
    border-radius: 4px;
  }

  .idle-msg {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 16px;
    color: var(--phosphor);
    text-align: center;
  }
  .idle-msg .blink { animation: textblink 1s steps(1) infinite; }
  @keyframes textblink { 0%,100%{opacity:1} 50%{opacity:0} }
  .idle-msg p { font-size: 8px; line-height: 1.8; color: #3a7a3a; }

  .result-view { display: none; flex-direction: column; gap: 6px; height: 100%; }
  .result-view.show { display: flex; }

  .mon-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    border-bottom: 1px solid #0a3a0a;
    padding-bottom: 6px;
  }

  .mon-name {
    font-family: 'Press Start 2P', monospace;
    font-size: 11px;
    color: var(--phosphor);
    text-shadow: 0 0 10px var(--phosphor);
    line-height: 1.4;
  }
  .mon-num {
    font-family: 'VT323', monospace;
    font-size: 20px;
    color: #2a6a2a;
  }

  .type-row { display: flex; gap: 6px; }
  .type-pill {
    font-family: 'Press Start 2P', monospace;
    font-size: 6px;
    padding: 3px 7px;
    border-radius: 3px;
    letter-spacing: .5px;
    border: 1px solid rgba(255,255,255,.2);
  }

  .stats-grid {
    display: flex;
    flex-direction: column;
    gap: 4px;
    flex: 1;
  }

  .stat-line {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .stat-lbl {
    font-family: 'VT323', monospace;
    font-size: 15px;
    color: #2a6a2a;
    width: 46px;
    flex-shrink: 0;
  }
  .stat-track {
    flex: 1;
    height: 6px;
    background: #0a1a0a;
    border: 1px solid #0a3a0a;
    border-radius: 2px;
    overflow: hidden;
  }
  .stat-fill {
    height: 100%;
    border-radius: 2px;
    width: 0%;
    transition: width .8s ease;
  }
  .stat-num {
    font-family: 'VT323', monospace;
    font-size: 15px;
    color: var(--phosphor);
    width: 28px;
    text-align: right;
  }

  .meta-row {
    display: flex;
    justify-content: space-between;
    border-top: 1px solid #0a3a0a;
    padding-top: 6px;
  }
  .meta-item {
    font-family: 'VT323', monospace;
    font-size: 14px;
    color: #2a7a2a;
    text-align: center;
  }
  .meta-item span {
    display: block;
    font-size: 11px;
    color: var(--phosphor);
    margin-top: 1px;
  }

  .legendary-tag {
    font-family: 'Press Start 2P', monospace;
    font-size: 7px;
    color: var(--amber);
    text-shadow: 0 0 8px var(--amber);
    letter-spacing: 1px;
    animation: amberpulse 1.5s ease-in-out infinite;
  }
  @keyframes amberpulse { 0%,100%{opacity:1; text-shadow:0 0 8px #ffb000} 50%{opacity:.7; text-shadow:0 0 20px #ffb000, 0 0 40px #ff8000} }

  .conf-bar-wrap {
    background: #0a1a0a;
    border: 1px solid #0a3a0a;
    border-radius: 2px;
    height: 8px;
    overflow: hidden;
  }
  .conf-bar-fill {
    height: 100%;
    background: var(--phosphor);
    border-radius: 2px;
    transition: width .6s ease;
    box-shadow: 0 0 6px var(--phosphor);
  }

  .others-row {
    display: flex;
    flex-direction: column;
    gap: 2px;
    border-top: 1px solid #0a3a0a;
    padding-top: 4px;
  }
  .other-item {
    display: flex;
    justify-content: space-between;
    font-family: 'VT323', monospace;
    font-size: 13px;
    color: #1a4a1a;
  }

  /* ── bottom right controls ── */
  .right-btns {
    display: flex;
    gap: 8px;
    z-index: 2;
  }
  .r-btn {
    flex: 1;
    padding: 8px 0;
    background: linear-gradient(180deg, #1a2a6a 0%, #0d1a4a 100%);
    border: 1.5px solid #334499;
    border-bottom: 3px solid #0a0a2a;
    border-radius: 5px;
    color: #88aaff;
    font-family: 'Press Start 2P', monospace;
    font-size: 7px;
    cursor: pointer;
    letter-spacing: .5px;
    transition: .1s;
    text-align: center;
  }
  .r-btn:active { transform: translateY(2px); border-bottom-width: 1px; }
  .r-btn:hover  { background: linear-gradient(180deg, #223380 0%, #111a5c 100%); color: #aaccff; }
  .r-btn.active-team { border-color: var(--phosphor); color: var(--phosphor); text-shadow: 0 0 6px var(--phosphor); }

  /* ── Team overlay ── */
  .team-overlay {
    position: absolute;
    inset: 0;
    background: var(--screen-bg);
    border-radius: 4px;
    padding: 10px;
    display: none;
    flex-direction: column;
    gap: 6px;
    z-index: 20;
    overflow: hidden;
  }
  .team-overlay.show { display: flex; }
  .team-overlay::after {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
      to bottom, transparent 0px, transparent 3px,
      rgba(0,0,0,.15) 3px, rgba(0,0,0,.15) 4px
    );
    pointer-events: none;
    z-index: 10;
  }
  .team-title {
    font-family: 'Press Start 2P', monospace;
    font-size: 8px;
    color: var(--phosphor);
    text-shadow: 0 0 8px var(--phosphor);
    border-bottom: 1px solid #0a3a0a;
    padding-bottom: 6px;
  }
  .team-slots {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 6px;
    flex: 1;
  }
  .team-slot-card {
    background: #0a150a;
    border: 1px solid #0a3a0a;
    border-radius: 3px;
    padding: 6px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .team-slot-card .slot-num { font-family:'VT323',monospace; font-size:11px; color:#1a4a1a; }
  .team-slot-card .slot-name { font-family:'Press Start 2P',monospace; font-size:7px; color:var(--phosphor); text-shadow:0 0 6px var(--phosphor); line-height:1.4; }
  .team-slot-card .slot-type { font-family:'VT323',monospace; font-size:12px; color:#2a7a2a; }
  .team-slot-empty { opacity:.3; }
  .team-slot-empty .slot-name { color:#1a4a1a; text-shadow:none; }

  /* type colors */
  .t-fire     { background:#8b3a00; color:#ff9944; }
  .t-water    { background:#003a8b; color:#44aaff; }
  .t-grass    { background:#1a5a00; color:#66ff44; }
  .t-electric { background:#5a5a00; color:#ffee00; }
  .t-psychic  { background:#5a0044; color:#ff88dd; }
  .t-ice      { background:#004a4a; color:#88ffee; }
  .t-dragon   { background:#2a008b; color:#aa88ff; }
  .t-dark     { background:#1a1400; color:#aa8855; }
  .t-fairy    { background:#5a0055; color:#ff99ee; }
  .t-normal   { background:#3a3a1a; color:#ccccaa; }
  .t-fighting { background:#5a0000; color:#ff6644; }
  .t-flying   { background:#1a2a5a; color:#88aaff; }
  .t-poison   { background:#3a005a; color:#cc66ff; }
  .t-ground   { background:#4a3a00; color:#ddbb44; }
  .t-rock     { background:#3a2a00; color:#bbaa44; }
  .t-bug      { background:#2a3a00; color:#99cc00; }
  .t-ghost    { background:#1a0055; color:#9966ff; }
  .t-steel    { background:#2a2a3a; color:#aabbcc; }

  /* typewriter */
  .typewriter { overflow: hidden; white-space: nowrap; animation: type .5s steps(20, end) forwards; }
  @keyframes type { from { width: 0 } to { width: 100% } }

  /* notification */
  .notif {
    position: fixed;
    bottom: 30px; left: 50%;
    transform: translateX(-50%) translateY(20px);
    background: rgba(0,20,0,.95);
    border: 1px solid var(--phosphor);
    color: var(--phosphor);
    font-family: 'Press Start 2P', monospace;
    font-size: 8px;
    padding: 10px 18px;
    border-radius: 4px;
    opacity: 0;
    transition: opacity .3s, transform .3s;
    text-shadow: 0 0 8px var(--phosphor);
    z-index: 999;
    white-space: nowrap;
  }
  .notif.show { opacity: 1; transform: translateX(-50%) translateY(0); }
</style>
</head>
<body>

<div class="pokedex">

  <!-- LEFT HALF -->
  <div class="left-half">
    <div class="indicator-row">
      <div class="big-light" id="bigLight"></div>
      <div class="small-lights">
        <div class="dot r"></div>
        <div class="dot y"></div>
        <div class="dot g"></div>
      </div>
    </div>

    <div class="screen-wrap">
      <div class="screen-inner">
        <video id="webcam" autoplay playsinline muted></video>
        <div class="scanlines"></div>
        <div class="scan-beam" id="scanBeam"></div>
        <div class="bracket tl"></div>
        <div class="bracket tr"></div>
        <div class="bracket bl"></div>
        <div class="bracket br"></div>
        <div class="crosshair"></div>
      </div>
    </div>

    <div class="btn-row">
      <button class="scan-btn" onclick="doScan()">[ SCAN ]</button>
      <div class="dpad">
        <div></div><div class="dpad-btn"></div><div></div>
        <div class="dpad-btn"></div><div class="dpad-btn"></div><div class="dpad-btn"></div>
        <div></div><div class="dpad-btn"></div><div></div>
      </div>
    </div>
  </div>

  <!-- HINGE -->
  <div class="hinge">
    <div class="hinge-screw"></div>
    <div class="hinge-screw"></div>
    <div class="hinge-screw"></div>
  </div>

  <!-- RIGHT HALF -->
  <div class="right-half">
    <div class="info-screen" id="infoScreen">

      <!-- idle state -->
      <div class="idle-msg" id="idleMsg">
        <div style="font-size:10px;color:#1a5a1a">POKEDEX</div>
        <div style="font-size:8px;color:var(--phosphor);text-shadow:0 0 10px var(--phosphor)" class="blink">READY</div>
        <p>POINT CAMERA<br>AT A POKEMON<br>AND PRESS SCAN</p>
      </div>

      <!-- result state -->
      <div class="result-view" id="resultView">
        <div class="mon-header">
          <div>
            <div class="mon-name" id="monName">---</div>
            <div class="type-row" id="typeRow"></div>
          </div>
          <div style="text-align:right">
            <div class="mon-num" id="monNum">#000</div>
            <div id="legendaryTag"></div>
          </div>
        </div>

        <div style="font-family:VT323,monospace;font-size:12px;color:#1a5a1a;margin-bottom:2px">CONFIDENCE</div>
        <div class="conf-bar-wrap"><div class="conf-bar-fill" id="confBar" style="width:0%"></div></div>

        <div class="stats-grid" id="statsGrid"></div>

        <div class="meta-row" id="metaRow"></div>

        <div class="others-row" id="othersRow"></div>
      </div>

      <!-- team overlay -->
      <div class="team-overlay" id="teamOverlay">
        <div class="team-title">MY TEAM</div>
        <div class="team-slots" id="teamSlots"></div>
      </div>
    </div>

    <div class="right-btns">
      <button class="r-btn" onclick="addToTeam()">+ TEAM</button>
      <button class="r-btn" id="teamBtn" onclick="toggleTeam()">VIEW TEAM</button>
      <button class="r-btn" onclick="clearTeam()">CLEAR</button>
    </div>
  </div>
</div>

<div class="notif" id="notif"></div>

<script>
const TYPE_COLORS = {
  fire:'t-fire',water:'t-water',grass:'t-grass',electric:'t-electric',
  psychic:'t-psychic',ice:'t-ice',dragon:'t-dragon',dark:'t-dark',
  fairy:'t-fairy',normal:'t-normal',fighting:'t-fighting',flying:'t-flying',
  poison:'t-poison',ground:'t-ground',rock:'t-rock',bug:'t-bug',
  ghost:'t-ghost',steel:'t-steel'
};

const STAT_COLORS = {
  HP:'#ff5959', Attack:'#f5ac78', Defense:'#fae078',
  'Sp.Atk':'#9db7f5', 'Sp.Def':'#a7db8d', Speed:'#fa92b2'
};

let currentPred = null;
let team = [];
let teamVisible = false;

// ── Webcam ──
async function initCam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width:640, height:480, facingMode:'user' } });
    document.getElementById('webcam').srcObject = stream;
  } catch(e) { console.error('Cam error:', e); }
}
initCam();

// ── Scan ──
async function doScan() {
  const video = document.getElementById('webcam');
  if (!video.srcObject) return notify('NO CAMERA FEED');

  // animate
  document.getElementById('bigLight').classList.add('scanning');
  const beam = document.getElementById('scanBeam');
  beam.classList.remove('active');
  void beam.offsetWidth;
  beam.classList.add('active');

  // capture frame
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const b64 = canvas.toDataURL('image/jpeg', .85);

  try {
    const res  = await fetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ image: b64 }) });
    const data = await res.json();
    if (data.predictions) showResult(data.predictions);
  } catch(e) { notify('SCAN ERROR'); }

  setTimeout(() => document.getElementById('bigLight').classList.remove('scanning'), 1200);
}

// ── Show result ──
function showResult(preds) {
  currentPred = preds[0];
  const p = preds[0];
  const s = p.stats || {};

  document.getElementById('idleMsg').style.display = 'none';
  document.getElementById('resultView').classList.add('show');

  // name with typewriter
  const nameEl = document.getElementById('monName');
  nameEl.textContent = '';
  const name = p.name.replace(/-/g,' ').toUpperCase();
  let i = 0;
  const tw = setInterval(() => {
    nameEl.textContent += name[i++];
    if (i >= name.length) clearInterval(tw);
  }, 50);

  // dex num + legendary
  const dex = s.pokedex_number || '?';
  document.getElementById('monNum').textContent = '#' + String(dex).padStart(3,'0');
  document.getElementById('legendaryTag').innerHTML = s.legendary ? '<div class="legendary-tag">* LEGENDARY</div>' : '';

  // types
  const tr = document.getElementById('typeRow');
  tr.innerHTML = '';
  [s.type1, s.type2].filter(t => t && t !== 'nan').forEach(t => {
    const span = document.createElement('span');
    span.className = 'type-pill ' + (TYPE_COLORS[t] || '');
    span.textContent = t.toUpperCase();
    tr.appendChild(span);
  });

  // confidence bar
  setTimeout(() => {
    document.getElementById('confBar').style.width = p.confidence + '%';
  }, 100);

  // stats
  const statDefs = [
    ['HP', s.hp], ['Attack', s.attack], ['Defense', s.defense],
    ['Sp.Atk', s.sp_attack], ['Sp.Def', s.sp_defense], ['Speed', s.speed]
  ];
  const grid = document.getElementById('statsGrid');
  grid.innerHTML = '';
  statDefs.forEach(([lbl, val]) => {
    if (!val && val !== 0) return;
    const pct = Math.min(100, Math.round(val / 255 * 100));
    const div = document.createElement('div');
    div.className = 'stat-line';
    div.innerHTML = `
      <span class="stat-lbl">${lbl}</span>
      <div class="stat-track"><div class="stat-fill" style="background:${STAT_COLORS[lbl]};width:0%" data-pct="${pct}"></div></div>
      <span class="stat-num">${val}</span>`;
    grid.appendChild(div);
  });
  setTimeout(() => {
    grid.querySelectorAll('.stat-fill').forEach(el => el.style.width = el.dataset.pct + '%');
  }, 200);

  // meta
  const meta = document.getElementById('metaRow');
  const gen  = s.generation || '?';
  const bst  = s.base_total || '?';
  const cap  = s.capture_rate || '?';
  const ht   = s.height_m ? s.height_m + 'm' : '?';
  const wt   = s.weight_kg ? s.weight_kg + 'kg' : '?';
  meta.innerHTML = `
    <div class="meta-item">GEN<span>${gen}</span></div>
    <div class="meta-item">BST<span>${bst}</span></div>
    <div class="meta-item">CATCH<span>${cap}</span></div>
    <div class="meta-item">HT<span>${ht}</span></div>
    <div class="meta-item">WT<span>${wt}</span></div>`;

  // others
  const others = document.getElementById('othersRow');
  others.innerHTML = preds.slice(1).map(p2 =>
    `<div class="other-item"><span>${p2.name.replace(/-/g,' ').toUpperCase()}</span><span>${p2.confidence}%</span></div>`
  ).join('');
}

// ── Team ──
function addToTeam() {
  if (!currentPred) return notify('SCAN FIRST');
  if (team.length >= 6) return notify('TEAM FULL');
  if (team.find(t => t.name === currentPred.name)) return notify('ALREADY IN TEAM');
  team.push(currentPred);
  notify('ADDED: ' + currentPred.name.replace(/-/g,' ').toUpperCase());
  renderTeam();
}

function clearTeam() {
  team = [];
  renderTeam();
  notify('TEAM CLEARED');
}

function toggleTeam() {
  teamVisible = !teamVisible;
  document.getElementById('teamOverlay').classList.toggle('show', teamVisible);
  document.getElementById('teamBtn').classList.toggle('active-team', teamVisible);
}

function renderTeam() {
  const slots = document.getElementById('teamSlots');
  slots.innerHTML = '';
  for (let i = 0; i < 6; i++) {
    const mon = team[i];
    const div = document.createElement('div');
    if (mon) {
      const t1 = (mon.stats?.type1 || '').toLowerCase();
      div.className = 'team-slot-card';
      div.innerHTML = `
        <span class="slot-num">${i+1}.</span>
        <span class="slot-name">${mon.name.replace(/-/g,' ').toUpperCase()}</span>
        <span class="slot-type">${t1.toUpperCase() || '---'}</span>`;
    } else {
      div.className = 'team-slot-card team-slot-empty';
      div.innerHTML = `<span class="slot-num">${i+1}.</span><span class="slot-name">EMPTY</span>`;
    }
    slots.appendChild(div);
  }
}
renderTeam();

// ── Notify ──
function notify(msg) {
  const el = document.getElementById('notif');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 2200);
}

// keyboard shortcut
document.addEventListener('keydown', e => {
  if (e.code === 'Space') { e.preventDefault(); doScan(); }
  if (e.code === 'KeyA') addToTeam();
  if (e.code === 'KeyT') toggleTeam();
  if (e.code === 'KeyC') clearTeam();
});
</script>
</body>
</html>"""

if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)