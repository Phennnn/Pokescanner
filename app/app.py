# ── PokéScanner · app/app.py ──────────────────────────────────────────────────
# Step 5: Full Gradio web app.
# Run from project root:
#   python app/app.py
#
# Opens a browser at http://localhost:7860
# Features:
#   - Upload image OR use webcam to identify Pokémon
#   - Confidence bar + top-3 candidates
#   - Full stats display (HP, ATK, DEF, SP.ATK, SP.DEF, SPD)
#   - Type badges with weakness chart
#   - Team builder (up to 6 Pokémon)
#   - Team analysis (type coverage + weaknesses)
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path
from collections import defaultdict

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
WEIGHTS   = ROOT / "model" / "weights" / "best_model_b2.pth"
LABEL_MAP = ROOT / "data" / "processed" / "label_map.json"
STATS_CSV = ROOT / "data" / "raw" / "pokemon_stats.csv"

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE = 260
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]
TOP_K    = 5

# ── Type colours (hex) ────────────────────────────────────────────────────────
TYPE_COLORS = {
    "fire":     "#FF6B35", "water":    "#4A9EFF", "grass":    "#5DBE6E",
    "electric": "#FFD700", "psychic":  "#FF6EB4", "ice":      "#96D9D6",
    "dragon":   "#6F35FC", "dark":     "#705746", "fairy":    "#D685AD",
    "normal":   "#A8A878", "fighting": "#C22E28", "flying":   "#89AAE3",
    "poison":   "#A33EA1", "ground":   "#E2BF65", "rock":     "#B6A136",
    "bug":      "#A6B91A", "ghost":    "#735797", "steel":    "#B7B7CE",
}

# ── TTA transforms ────────────────────────────────────────────────────────────
tta_transforms = [
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    transforms.Compose([transforms.Resize((IMG_SIZE+20, IMG_SIZE+20)), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    transforms.Compose([transforms.Resize((IMG_SIZE+30, IMG_SIZE+30)), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
]


# ── Load everything at startup ────────────────────────────────────────────────
print("Loading PokéScanner...")

with open(LABEL_MAP) as f:
    label_map = json.load(f)
idx_to_label = label_map["idx_to_label"]
NUM_CLASSES  = label_map["num_classes"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = timm.create_model("efficientnet_b2", pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval().to(device)
print(f"Model loaded on {device}")

# stats db
stats_db = {}
if STATS_CSV.exists():
    df = pd.read_csv(STATS_CSV)
    for _, row in df.iterrows():
        key = str(row["name"]).strip().lower().replace(" ", "-").replace("'", "").replace("'", "")
        stats_db[key] = {
            "hp": int(row.get("hp", 0)),
            "attack": int(row.get("attack", 0)),
            "defense": int(row.get("defense", 0)),
            "sp_attack": int(row.get("sp_attack", 0)),
            "sp_defense": int(row.get("sp_defense", 0)),
            "speed": int(row.get("speed", 0)),
            "base_total": int(row.get("base_total", 0)),
            "type1": str(row.get("type1", "")).lower().strip(),
            "type2": str(row.get("type2", "")).lower().strip(),
            "legendary": bool(row.get("is_legendary", 0)),
            "generation": int(row.get("generation", 0)),
            "pokedex_number": int(row.get("pokedex_number", 0)),
            "capture_rate": str(row.get("capture_rate", "?")),
            "classfication": str(row.get("classfication", "")),
        }

# team state
team = []

print(f"Ready! {NUM_CLASSES} classes, {len(stats_db)} with stats")


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_image(pil_img):
    img = pil_img.convert("RGB")
    tensors = torch.stack([t(img) for t in tta_transforms]).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensors), dim=1).mean(dim=0)
    top_probs, top_idxs = torch.topk(probs, TOP_K)
    return [(idx_to_label[str(i.item())], float(p)) for p, i in zip(top_probs, top_idxs)]


# ── Stat bar HTML ─────────────────────────────────────────────────────────────
def stat_bar(label, value, max_val=255, color="#4CAF50"):
    pct = min(100, int(value / max_val * 100))
    return f"""
    <div style="display:flex;align-items:center;gap:8px;margin:3px 0">
      <span style="font-family:'Rajdhani',sans-serif;font-size:12px;color:#888;width:64px;flex-shrink:0">{label}</span>
      <div style="flex:1;height:8px;background:#1a1a2e;border-radius:4px;overflow:hidden">
        <div style="width:{pct}%;height:100%;background:{color};border-radius:4px;transition:width .4s ease"></div>
      </div>
      <span style="font-family:'Rajdhani',sans-serif;font-size:13px;color:#ddd;width:32px;text-align:right">{value}</span>
    </div>"""


def type_badge(t):
    if not t or t == "nan": return ""
    color = TYPE_COLORS.get(t, "#888")
    return f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:12px;font-size:11px;font-family:Rajdhani,sans-serif;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-right:4px">{t}</span>'


# ── Build result card HTML ────────────────────────────────────────────────────
def build_result_html(predictions):
    if not predictions:
        return '<div style="color:#555;text-align:center;padding:40px">No prediction yet</div>'

    name, conf = predictions[0]
    s = stats_db.get(name, {})
    display_name = name.replace("-", " ").title()
    t1 = s.get("type1", "")
    t2 = s.get("type2", "")
    dex_num = s.get("pokedex_number", "?")
    gen     = s.get("generation", "?")
    cls     = s.get("classfication", "")
    cap     = s.get("capture_rate", "?")
    legendary = s.get("legendary", False)

    conf_color = "#4CAF50" if conf > 0.5 else "#FF9800" if conf > 0.25 else "#f44336"
    conf_pct   = int(conf * 100)

    # confidence bar
    conf_bar = f"""
    <div style="margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px">
        <span style="font-size:11px;color:#666;font-family:Rajdhani,sans-serif">CONFIDENCE</span>
        <span style="font-size:13px;font-weight:700;color:{conf_color};font-family:Rajdhani,sans-serif">{conf_pct}%</span>
      </div>
      <div style="height:6px;background:#1a1a2e;border-radius:3px;overflow:hidden">
        <div style="width:{conf_pct}%;height:100%;background:{conf_color};border-radius:3px"></div>
      </div>
    </div>"""

    # name + dex
    header = f"""
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px">
      <div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:28px;font-weight:700;color:#fff;line-height:1">{display_name}</div>
        <div style="font-size:11px;color:#555;font-family:Rajdhani,sans-serif;margin-top:2px">{cls}</div>
      </div>
      <div style="text-align:right">
        <div style="font-family:'Rajdhani',sans-serif;font-size:22px;font-weight:700;color:#333">#{str(dex_num).zfill(3) if str(dex_num).isdigit() else dex_num}</div>
        <div style="font-size:11px;color:#444;font-family:Rajdhani,sans-serif">GEN {gen}</div>
      </div>
    </div>"""

    types = f'<div style="margin-bottom:12px">{type_badge(t1)}{type_badge(t2)}</div>'

    legendary_badge = '<div style="display:inline-block;background:#1a1400;border:1px solid #FFD700;color:#FFD700;padding:2px 10px;border-radius:4px;font-size:11px;font-family:Rajdhani,sans-serif;font-weight:700;letter-spacing:1px;margin-bottom:12px">* LEGENDARY</div>' if legendary else ""

    # stats
    stats_html = ""
    if s:
        stats_html = f"""
        <div style="margin:12px 0">
          {stat_bar("HP",       s.get("hp",0),        255, "#FF5959")}
          {stat_bar("Attack",   s.get("attack",0),     255, "#F5AC78")}
          {stat_bar("Defense",  s.get("defense",0),    255, "#FAE078")}
          {stat_bar("Sp. Atk",  s.get("sp_attack",0),  255, "#9DB7F5")}
          {stat_bar("Sp. Def",  s.get("sp_defense",0), 255, "#A7DB8D")}
          {stat_bar("Speed",    s.get("speed",0),       255, "#FA92B2")}
        </div>
        <div style="display:flex;gap:16px;font-family:Rajdhani,sans-serif;font-size:12px;color:#555;margin-top:8px">
          <span>BST <strong style="color:#aaa">{s.get("base_total",0)}</strong></span>
          <span>Catch rate <strong style="color:#aaa">{cap}</strong></span>
        </div>"""

    # other candidates
    others = ""
    if len(predictions) > 1:
        others = '<div style="margin-top:14px;padding-top:12px;border-top:1px solid #1a1a2e"><div style="font-size:11px;color:#444;font-family:Rajdhani,sans-serif;margin-bottom:6px">OTHER CANDIDATES</div>'
        for alt_name, alt_conf in predictions[1:]:
            alt_pct = int(alt_conf * 100)
            alt_display = alt_name.replace("-", " ").title()
            others += f'<div style="display:flex;justify-content:space-between;font-family:Rajdhani,sans-serif;font-size:13px;color:#555;margin:3px 0"><span>{alt_display}</span><span>{alt_pct}%</span></div>'
        others += "</div>"

    return f"""
    <div style="background:#0d0d1a;border:1px solid #1e1e3a;border-radius:12px;padding:18px;font-family:sans-serif">
      {conf_bar}{header}{types}{legendary_badge}{stats_html}{others}
    </div>"""


# ── Build team HTML ───────────────────────────────────────────────────────────
def build_team_html():
    if not team:
        return '<div style="color:#333;text-align:center;padding:20px;font-family:Rajdhani,sans-serif">No Pokémon in team yet</div>'

    slots = ""
    for i, name in enumerate(team):
        s = stats_db.get(name, {})
        t1 = s.get("type1", "")
        color = TYPE_COLORS.get(t1, "#333")
        display = name.replace("-", " ").title()
        dex = s.get("pokedex_number", "?")
        slots += f"""
        <div style="background:#0d0d1a;border:1px solid #1e1e3a;border-radius:8px;padding:10px;text-align:center">
          <div style="font-size:10px;color:#333;font-family:Rajdhani,sans-serif;margin-bottom:2px">{i+1}</div>
          <div style="font-size:13px;font-weight:700;color:#ddd;font-family:Rajdhani,sans-serif;line-height:1.2">{display}</div>
          <div style="font-size:10px;color:#444;font-family:Rajdhani,sans-serif">#{dex}</div>
          <div style="margin-top:4px">{type_badge(t1)}</div>
        </div>"""

    # empty slots
    for i in range(len(team), 6):
        slots += f'<div style="background:#08080f;border:1px dashed #1a1a2e;border-radius:8px;padding:10px;text-align:center;color:#222;font-family:Rajdhani,sans-serif;font-size:11px">{i+1}<br>empty</div>'

    return f'<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px">{slots}</div>'


def build_analysis_html():
    if not team:
        return ""

    type_count = defaultdict(int)
    weaknesses = defaultdict(float)

    for name in team:
        s = stats_db.get(name, {})
        t1 = s.get("type1", "")
        t2 = s.get("type2", "")
        if t1: type_count[t1] += 1
        if t2 and t2 != "nan": type_count[t2] += 1

        # weakness from type matchup columns
        df_row = pd.read_csv(STATS_CSV) if STATS_CSV.exists() else pd.DataFrame()
        if not df_row.empty:
            match = df_row[df_row["name"].str.lower().str.replace(" ", "-").str.replace("'", "") == name]
            if not match.empty:
                row = match.iloc[0]
                for col in [c for c in df_row.columns if c.startswith("against_")]:
                    etype = col.replace("against_", "")
                    val = float(row.get(col, 1.0))
                    if val > 1.0:
                        weaknesses[etype] += 1

    type_html = "".join(f'{type_badge(t)}<span style="font-size:11px;color:#555;font-family:Rajdhani,sans-serif">x{c} </span>' for t, c in sorted(type_count.items(), key=lambda x: -x[1]))
    weak_html  = "".join(f'{type_badge(t)}<span style="font-size:11px;color:#555;font-family:Rajdhani,sans-serif">({int(c)}) </span>' for t, c in sorted(weaknesses.items(), key=lambda x: -x[1])[:6])

    return f"""
    <div style="background:#0d0d1a;border:1px solid #1e1e3a;border-radius:12px;padding:16px;margin-top:10px">
      <div style="font-family:Rajdhani,sans-serif;font-size:11px;color:#444;margin-bottom:8px">TYPE COVERAGE</div>
      <div style="margin-bottom:12px">{type_html or '<span style="color:#333">—</span>'}</div>
      <div style="font-family:Rajdhani,sans-serif;font-size:11px;color:#444;margin-bottom:8px">COMMON WEAKNESSES</div>
      <div>{weak_html or '<span style="color:#444">None detected</span>'}</div>
    </div>"""


# ── Gradio handlers ───────────────────────────────────────────────────────────
current_predictions = []

def on_scan(image):
    global current_predictions
    if image is None:
        return build_result_html([]), build_team_html(), build_analysis_html()
    pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    current_predictions = predict_image(pil)
    return build_result_html(current_predictions), build_team_html(), build_analysis_html()


def on_add():
    global team
    if not current_predictions:
        return build_team_html(), build_analysis_html(), "Scan a Pokémon first!"
    name, conf = current_predictions[0]
    if len(team) >= 6:
        return build_team_html(), build_analysis_html(), "Team is full (6/6)!"
    if name in team:
        return build_team_html(), build_analysis_html(), f"{name.replace('-',' ').title()} is already in your team!"
    team.append(name)
    msg = f"Added {name.replace('-',' ').title()} to team! ({len(team)}/6)"
    return build_team_html(), build_analysis_html(), msg


def on_clear():
    global team
    team = []
    return build_team_html(), build_analysis_html(), "Team cleared!"


# ── Custom CSS ────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=DM+Sans:wght@300;400;500&display=swap');

body, .gradio-container { background: #07070f !important; font-family: 'DM Sans', sans-serif !important; }
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }

h1 { font-family: 'Rajdhani', sans-serif !important; font-size: 32px !important; font-weight: 700 !important;
     letter-spacing: 2px !important; color: #fff !important; margin: 0 !important; }

.scan-btn { background: #e8362a !important; border: none !important; color: #fff !important;
            font-family: 'Rajdhani', sans-serif !important; font-weight: 700 !important;
            letter-spacing: 1px !important; font-size: 15px !important; }
.scan-btn:hover { background: #c42a1f !important; }

.add-btn  { background: transparent !important; border: 1px solid #1e1e3a !important; color: #aaa !important;
            font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important; }
.add-btn:hover  { border-color: #4CAF50 !important; color: #4CAF50 !important; }

.clear-btn { background: transparent !important; border: 1px solid #1e1e3a !important; color: #555 !important;
             font-family: 'Rajdhani', sans-serif !important; }
.clear-btn:hover { border-color: #e8362a !important; color: #e8362a !important; }

.gr-panel, .gr-box { background: #0d0d1a !important; border: 1px solid #1e1e3a !important; border-radius: 12px !important; }
.gr-input { background: #07070f !important; border: 1px solid #1e1e3a !important; color: #fff !important; }
label { color: #555 !important; font-family: 'Rajdhani', sans-serif !important; font-size: 11px !important; letter-spacing: 1px !important; }
.status-msg { font-family: 'Rajdhani', sans-serif !important; font-size: 13px !important; color: #4CAF50 !important; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="PokéScanner") as demo:

    gr.HTML("""
    <div style="display:flex;align-items:center;gap:14px;padding:20px 0 10px">
      <div style="width:36px;height:36px;background:#e8362a;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:18px">&#9673;</div>
      <div>
        <h1>Poke<span style="color:#e8362a">Scanner</span></h1>
        <div style="font-family:Rajdhani,sans-serif;font-size:11px;color:#333;letter-spacing:2px">REAL-TIME POKEMON IDENTIFIER</div>
      </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="UPLOAD IMAGE OR USE WEBCAM",
                sources=["upload", "webcam"],
                type="numpy",
                height=320,
            )
            scan_btn  = gr.Button("SCAN",        elem_classes="scan-btn")
            with gr.Row():
                add_btn   = gr.Button("+ ADD TO TEAM", elem_classes="add-btn")
                clear_btn = gr.Button("CLEAR TEAM",    elem_classes="clear-btn")
            status_box = gr.Textbox(label="", interactive=False, elem_classes="status-msg")

        with gr.Column(scale=1):
            result_html = gr.HTML(build_result_html([]))

    gr.HTML('<div style="font-family:Rajdhani,sans-serif;font-size:11px;color:#333;letter-spacing:2px;margin:16px 0 8px">MY TEAM</div>')
    team_html     = gr.HTML(build_team_html())
    analysis_html = gr.HTML("")

    # wire up
    scan_btn.click(on_scan,  inputs=image_input,    outputs=[result_html, team_html, analysis_html])
    add_btn.click( on_add,   inputs=None,            outputs=[team_html, analysis_html, status_box])
    clear_btn.click(on_clear, inputs=None,           outputs=[team_html, analysis_html, status_box])


if __name__ == "__main__":
    demo.launch(inbrowser=True)