# ── PokéScanner · app/pokedex.py ─────────────────────────────────────────────
# Anime-style Pokédex UI — Flask backend
# Run from project root:
#   python app/pokedex.py
# Opens http://localhost:5000
# ─────────────────────────────────────────────────────────────────────────────

import base64
import binascii
import io
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokescanner import dex
from pokescanner.inference import PokemonClassifier

TOP_K = 4
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

print("Loading model...")
CLASSIFIER = PokemonClassifier()
CLASSIFIER.warmup()

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/health")
def health():
    return jsonify({
        "arch": CLASSIFIER.arch,
        "img_size": CLASSIFIER.img_size,
        "device": str(CLASSIFIER.device),
        "classes": CLASSIFIER.num_classes,
        "isolate": CLASSIFIER.isolate,
    })


def _decode(data: str) -> Image.Image:
    payload = data.split(",", 1)[1] if "," in data else data
    return Image.open(io.BytesIO(base64.b64decode(payload)))


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    data = payload.get("image", "")
    if not data:
        return jsonify({"error": "no image"}), 400

    try:
        pil_img = _decode(data)
    except (binascii.Error, ValueError, OSError):
        return jsonify({"error": "could not read that image"}), 400

    CLASSIFIER.isolate = bool(payload.get("isolate", CLASSIFIER.isolate))
    result = CLASSIFIER.predict(pil_img, top_k=TOP_K)

    predictions = []
    for p in result.predictions:
        entry = p.to_dict()
        entry["weaknesses"] = dex.weaknesses(p.label)
        entry["resistances"] = dex.resistances(p.label)
        predictions.append(entry)

    body = {
        "predictions": predictions,
        "confident": result.is_confident,
        "elapsed_ms": round(result.elapsed_ms),
    }

    if payload.get("want_preview"):
        buf = io.BytesIO()
        CLASSIFIER.prepare(pil_img).save(buf, "JPEG", quality=82)
        body["model_input"] = ("data:image/jpeg;base64,"
                               + base64.b64encode(buf.getvalue()).decode())

    return jsonify(body)


@app.route("/team/analyse", methods=["POST"])
def analyse_team():
    payload = request.get_json(silent=True) or {}
    labels = [str(x) for x in (payload.get("team") or [])][:6]
    return jsonify(dex.team_report(labels))


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
  .scan-btn.alt    { flex: 0 0 74px; color: var(--amber); text-shadow: 0 0 8px var(--amber); }

  /* drop target state while a file is dragged over the screen */
  .screen-wrap.dropping { outline: 2px dashed var(--amber); outline-offset: 3px; }

  /* the frame the model actually receives, shown bottom-left of the screen */
  .model-input {
    position: absolute;
    left: 6px; bottom: 6px;
    width: 54px; height: 54px;
    border: 1px solid #0a3a0a;
    border-radius: 2px;
    object-fit: cover;
    opacity: .85;
    display: none;
    z-index: 4;
  }
  .model-input.show { display: block; }

  .low-conf {
    font-family: 'Press Start 2P', monospace;
    font-size: 6px;
    color: var(--amber);
    text-shadow: 0 0 6px var(--amber);
    letter-spacing: .5px;
    line-height: 1.6;
    margin-bottom: 4px;
  }

  .matchup-row {
    display: flex;
    flex-direction: column;
    gap: 3px;
    border-top: 1px solid #0a3a0a;
    padding-top: 5px;
  }
  .matchup-line { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
  .matchup-lbl {
    font-family: 'Press Start 2P', monospace;
    font-size: 6px;
    color: #1a5a1a;
    letter-spacing: .5px;
    min-width: 42px;
  }
  .mult-pill {
    font-family: 'VT323', monospace;
    font-size: 11px;
    line-height: 1;
    padding: 2px 4px;
    border-radius: 2px;
    color: #051405;
    font-weight: 700;
  }

  .team-analysis {
    border-top: 1px solid #0a3a0a;
    padding-top: 6px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .ta-line { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
  .ta-lbl {
    font-family: 'Press Start 2P', monospace;
    font-size: 6px;
    color: #1a5a1a;
    letter-spacing: .5px;
  }
  .ta-ok { font-family: 'VT323', monospace; font-size: 13px; color: var(--phosphor); }

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
        <img class="model-input" id="modelInput" alt="model input">
      </div>
    </div>

    <div class="btn-row">
      <button class="scan-btn" onclick="doScan()">[ SCAN ]</button>
      <button class="scan-btn alt" onclick="document.getElementById('fileInput').click()">[ FILE ]</button>
      <input type="file" id="fileInput" accept="image/*" hidden onchange="handleFile(this.files[0])">
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
        <div class="low-conf" id="lowConf" style="display:none"></div>
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

        <div class="matchup-row" id="matchupRow"></div>

        <div class="others-row" id="othersRow"></div>
      </div>

      <!-- team overlay -->
      <div class="team-overlay" id="teamOverlay">
        <div class="team-title">MY TEAM</div>
        <div class="team-slots" id="teamSlots"></div>
        <div class="team-analysis" id="teamAnalysis"></div>
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
let scanning = false;

// ── Webcam ──
async function initCam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width:640, height:480, facingMode:'user' } });
    document.getElementById('webcam').srcObject = stream;
  } catch(e) { console.error('Cam error:', e); }
}
initCam();

// ── Scan ──
function beginScanAnimation() {
  document.getElementById('bigLight').classList.add('scanning');
  const beam = document.getElementById('scanBeam');
  beam.classList.remove('active');
  void beam.offsetWidth;
  beam.classList.add('active');
}

function endScanAnimation() {
  setTimeout(() => document.getElementById('bigLight').classList.remove('scanning'), 1200);
}

// Send a data-URL to the backend and render whatever comes back.
async function classify(b64) {
  if (scanning) return;
  scanning = true;
  beginScanAnimation();
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: b64, want_preview: true })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      notify((err.error || 'SCAN FAILED').toUpperCase());
      return;
    }
    const data = await res.json();
    if (data.model_input) {
      const el = document.getElementById('modelInput');
      el.src = data.model_input;
      el.classList.add('show');
    }
    if (data.predictions && data.predictions.length) {
      showResult(data.predictions, data.confident, data.elapsed_ms);
    }
  } catch (e) {
    notify('SCAN ERROR');
  } finally {
    scanning = false;
    endScanAnimation();
  }
}

async function doScan() {
  const video = document.getElementById('webcam');
  if (!video.srcObject) return notify('NO CAMERA - USE [ FILE ]');

  // The visible feed is mirrored for the user's benefit; undo that before
  // sending, so the model sees the scene the right way round.
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0);
  await classify(canvas.toDataURL('image/jpeg', .9));
}

// ── File upload / drag and drop ──
function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) return notify('NOT AN IMAGE');
  if (file.size > 12 * 1024 * 1024) return notify('FILE TOO BIG (12MB MAX)');
  const reader = new FileReader();
  reader.onload = () => classify(reader.result);
  reader.onerror = () => notify('COULD NOT READ FILE');
  reader.readAsDataURL(file);
}

(function enableDropTarget() {
  const wrap = document.querySelector('.screen-wrap');
  if (!wrap) return;
  ['dragenter', 'dragover'].forEach(evt => wrap.addEventListener(evt, e => {
    e.preventDefault();
    wrap.classList.add('dropping');
  }));
  ['dragleave', 'drop'].forEach(evt => wrap.addEventListener(evt, e => {
    e.preventDefault();
    wrap.classList.remove('dropping');
  }));
  wrap.addEventListener('drop', e => {
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    handleFile(file);
  });
  window.addEventListener('paste', e => {
    const item = [...(e.clipboardData ? e.clipboardData.items : [])]
      .find(i => i.type.startsWith('image/'));
    if (item) handleFile(item.getAsFile());
  });
})();

// ── Show result ──
function showResult(preds, confident, elapsedMs) {
  currentPred = preds[0];
  const p = preds[0];
  const s = p.stats || {};

  document.getElementById('idleMsg').style.display = 'none';
  document.getElementById('resultView').classList.add('show');

  const lc = document.getElementById('lowConf');
  if (confident === false) {
    lc.textContent = 'LOW CONFIDENCE - FILL MORE OF THE FRAME, ADD LIGHT, OR USE A PLAINER BACKGROUND';
    lc.style.display = 'block';
  } else {
    lc.style.display = 'none';
  }

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
  meta.innerHTML = s.has_stats === false
    ? `<div class="meta-item" style="flex:1">NO BASE STATS ON FILE FOR THIS FORM</div>`
    : `
    <div class="meta-item">GEN<span>${gen}</span></div>
    <div class="meta-item">BST<span>${bst}</span></div>
    <div class="meta-item">CATCH<span>${cap}</span></div>
    <div class="meta-item">HT<span>${ht}</span></div>
    <div class="meta-item">WT<span>${wt}</span></div>`;

  // type matchups
  renderMatchups(p);

  // others
  const others = document.getElementById('othersRow');
  others.innerHTML = preds.slice(1).map(p2 =>
    `<div class="other-item"><span>${p2.name.replace(/-/g,' ').toUpperCase()}</span><span>${p2.confidence}%</span></div>`
  ).join('');
}

// ── Type matchups ──
const TYPE_HEX = {
  normal:'#A8A878', fire:'#FF6B35', water:'#4A9EFF', electric:'#FFD700',
  grass:'#5DBE6E', ice:'#96D9D6', fighting:'#C22E28', poison:'#A33EA1',
  ground:'#E2BF65', flying:'#89AAE3', psychic:'#FF6EB4', bug:'#A6B91A',
  rock:'#B6A136', ghost:'#735797', dragon:'#6F35FC', dark:'#705746',
  steel:'#B7B7CE', fairy:'#D685AD'
};

function multPill(type, mult) {
  const bg = TYPE_HEX[type] || '#888';
  const suffix = (mult === '' || mult === undefined) ? '' : ` x${mult}`;
  return `<span class="mult-pill" style="background:${bg}">${type.slice(0,4).toUpperCase()}${suffix}</span>`;
}

function renderMatchups(p) {
  const row = document.getElementById('matchupRow');
  const weak = p.weaknesses || {};
  const res  = p.resistances || {};
  const byMult = (a, b) => b[1] - a[1];

  const weakPills = Object.entries(weak).sort(byMult).map(([t, m]) => multPill(t, m)).join('');
  const resPills  = Object.entries(res).sort((a, b) => a[1] - b[1]).map(([t, m]) => multPill(t, m)).join('');

  row.innerHTML = `
    <div class="matchup-line"><span class="matchup-lbl">WEAK</span>${weakPills || '<span class="ta-ok">none</span>'}</div>
    <div class="matchup-line"><span class="matchup-lbl">RESIST</span>${resPills || '<span class="ta-ok">none</span>'}</div>`;
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
  renderTeamAnalysis();
}

// Coverage report for the current team, computed server-side from the type chart.
async function renderTeamAnalysis() {
  const box = document.getElementById('teamAnalysis');
  if (!box) return;
  if (!team.length) { box.innerHTML = ''; return; }

  try {
    const res = await fetch('/team/analyse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ team: team.map(m => m.name) })
    });
    if (!res.ok) return;
    const r = await res.json();

    const shared = Object.entries(r.weaknesses || {}).filter(([, c]) => c >= 2);
    const sharedPills = shared.map(([t, c]) => multPill(t, c)).join('');
    const gaps = (r.uncovered_types || []).map(t => multPill(t, '')).join('');

    box.innerHTML = `
      <div class="ta-line"><span class="ta-lbl">SHARED WEAK</span>
        ${sharedPills || '<span class="ta-ok">none - balanced</span>'}</div>
      <div class="ta-line"><span class="ta-lbl">NO COVERAGE</span>
        ${gaps || '<span class="ta-ok">all types covered</span>'}</div>`;
  } catch (e) { /* analysis is optional, never block the UI */ }
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
  if (e.code === 'KeyF') document.getElementById('fileInput').click();
});
</script>
</body>
</html>"""

if __name__ == "__main__":
    import webbrowser  # noqa: F401
    webbrowser.open("http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)