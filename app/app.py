"""PokeScanner - Gradio web app.

    python app/app.py        ->  http://localhost:7860

Upload a photo or use the webcam, get an identification with full stats and
type matchups, and build a team of six with a coverage report.

Two things changed from the first version:
  * Team state lives in a gr.State, not module globals, so two people on the
    same server no longer share (and overwrite) one team. That matters as soon
    as this is deployed anywhere public.
  * The card shows the image the model actually receives after subject
    isolation, which makes a wrong answer diagnosable instead of mysterious.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokescanner import cards, dex, identify           # noqa: E402
from pokescanner.inference import PokemonClassifier    # noqa: E402

MAX_TEAM = 6

print("Loading PokeScanner...")
CLASSIFIER = PokemonClassifier()
CLASSIFIER.warmup()


# -- HTML fragments -----------------------------------------------------------
def type_badge(t: str) -> str:
    if not t:
        return ""
    return (f'<span style="background:{dex.TYPE_COLORS.get(t, "#888")};color:#fff;'
            'padding:2px 10px;border-radius:12px;font-size:11px;'
            'font-family:Rajdhani,sans-serif;font-weight:700;letter-spacing:1px;'
            f'text-transform:uppercase;margin-right:4px">{t}</span>')


def mult_badge(t: str, mult: float) -> str:
    label = f"{t} x{mult:g}"
    return (f'<span style="background:{dex.TYPE_COLORS.get(t, "#888")};color:#fff;'
            'padding:2px 8px;border-radius:10px;font-size:10px;'
            'font-family:Rajdhani,sans-serif;font-weight:700;letter-spacing:.5px;'
            f'text-transform:uppercase;margin:0 4px 4px 0;display:inline-block">'
            f'{label}</span>')


def stat_bar(label: str, value: int, color: str) -> str:
    pct = min(100, int(value / 255 * 100))
    return f"""
    <div style="display:flex;align-items:center;gap:8px;margin:3px 0">
      <span style="font-family:Rajdhani,sans-serif;font-size:12px;color:#888;
                   width:64px;flex-shrink:0">{label}</span>
      <div style="flex:1;height:8px;background:#1a1a2e;border-radius:4px;overflow:hidden">
        <div style="width:{pct}%;height:100%;background:{color};border-radius:4px"></div>
      </div>
      <span style="font-family:Rajdhani,sans-serif;font-size:13px;color:#ddd;
                   width:32px;text-align:right">{value}</span>
    </div>"""


def empty_card(message="No scan yet") -> str:
    return (f'<div style="color:#555;text-align:center;padding:40px;'
            f'font-family:Rajdhani,sans-serif">{message}</div>')


def build_result_html(result) -> str:
    if result is None or not result.predictions:
        return empty_card()

    top = result.predictions[0]
    s = top.stats
    conf = top.confidence
    conf_pct = int(conf * 100)
    conf_color = "#4CAF50" if conf > 0.5 else "#FF9800" if conf > 0.25 else "#f44336"

    unsure = ""
    if not result.is_confident:
        unsure = ('<div style="background:#2a1a00;border:1px solid #FF9800;'
                  'color:#FF9800;padding:6px 10px;border-radius:6px;font-size:11px;'
                  'font-family:Rajdhani,sans-serif;margin-bottom:10px">'
                  'LOW CONFIDENCE - try filling more of the frame, better light, '
                  'or a plainer background</div>')

    conf_bar = f"""
    <div style="margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px">
        <span style="font-size:11px;color:#666;font-family:Rajdhani,sans-serif">CONFIDENCE</span>
        <span style="font-size:13px;font-weight:700;color:{conf_color};
                     font-family:Rajdhani,sans-serif">{conf_pct}%</span>
      </div>
      <div style="height:6px;background:#1a1a2e;border-radius:3px;overflow:hidden">
        <div style="width:{conf_pct}%;height:100%;background:{conf_color};
                    border-radius:3px"></div>
      </div>
    </div>"""

    dex_num = s.get("pokedex_number", 0)
    dex_str = f"#{dex_num:03d}" if dex_num else "#???"
    header = f"""
    <div style="display:flex;justify-content:space-between;align-items:flex-start;
                margin-bottom:6px">
      <div>
        <div style="font-family:Rajdhani,sans-serif;font-size:28px;font-weight:700;
                    color:#fff;line-height:1">{top.display_name}</div>
        <div style="font-size:11px;color:#555;font-family:Rajdhani,sans-serif;
                    margin-top:2px">{s.get('classfication', '')}</div>
      </div>
      <div style="text-align:right">
        <div style="font-family:Rajdhani,sans-serif;font-size:22px;font-weight:700;
                    color:#333">{dex_str}</div>
        <div style="font-size:11px;color:#444;font-family:Rajdhani,sans-serif">
          GEN {s.get('generation', '?') or '?'}</div>
      </div>
    </div>"""

    types = ('<div style="margin-bottom:12px">'
             + "".join(type_badge(t) for t in s.get("types", [])) + "</div>")

    legendary = ""
    if s.get("legendary"):
        legendary = ('<div style="display:inline-block;background:#1a1400;'
                     'border:1px solid #FFD700;color:#FFD700;padding:2px 10px;'
                     'border-radius:4px;font-size:11px;font-family:Rajdhani,sans-serif;'
                     'font-weight:700;letter-spacing:1px;margin-bottom:12px">'
                     '* LEGENDARY</div>')

    if s.get("has_stats"):
        stats_html = f"""
        <div style="margin:12px 0">
          {stat_bar("HP", s.get("hp", 0), "#FF5959")}
          {stat_bar("Attack", s.get("attack", 0), "#F5AC78")}
          {stat_bar("Defense", s.get("defense", 0), "#FAE078")}
          {stat_bar("Sp. Atk", s.get("sp_attack", 0), "#9DB7F5")}
          {stat_bar("Sp. Def", s.get("sp_defense", 0), "#A7DB8D")}
          {stat_bar("Speed", s.get("speed", 0), "#FA92B2")}
        </div>
        <div style="display:flex;gap:16px;font-family:Rajdhani,sans-serif;
                    font-size:12px;color:#555;margin-top:8px">
          <span>BST <strong style="color:#aaa">{s.get('base_total', 0)}</strong></span>
          <span>Catch <strong style="color:#aaa">{s.get('capture_rate', '?')}</strong></span>
          <span>HT <strong style="color:#aaa">{s.get('height_m', 0)}m</strong></span>
          <span>WT <strong style="color:#aaa">{s.get('weight_kg', 0)}kg</strong></span>
        </div>"""
    else:
        stats_html = ('<div style="font-family:Rajdhani,sans-serif;font-size:12px;'
                      'color:#555;margin:12px 0">Base stats are not in the dataset '
                      'for this form.</div>')

    weak = dex.weaknesses(top.label)
    resist = dex.resistances(top.label)
    matchup_html = ""
    if weak or resist:
        weak_row = "".join(mult_badge(t, m) for t, m in
                           sorted(weak.items(), key=lambda kv: -kv[1]))
        res_row = "".join(mult_badge(t, m) for t, m in
                          sorted(resist.items(), key=lambda kv: kv[1]))
        matchup_html = f"""
        <div style="margin-top:14px;padding-top:12px;border-top:1px solid #1a1a2e">
          <div style="font-size:11px;color:#444;font-family:Rajdhani,sans-serif;
                      margin-bottom:6px">TAKES MORE DAMAGE FROM</div>
          <div>{weak_row or '<span style="color:#333">nothing</span>'}</div>
          <div style="font-size:11px;color:#444;font-family:Rajdhani,sans-serif;
                      margin:10px 0 6px">RESISTS</div>
          <div>{res_row or '<span style="color:#333">nothing</span>'}</div>
        </div>"""

    others = ""
    if len(result.predictions) > 1:
        rows = "".join(
            f'<div style="display:flex;justify-content:space-between;'
            f'font-family:Rajdhani,sans-serif;font-size:13px;color:#555;margin:3px 0">'
            f'<span>{p.display_name}</span><span>{p.percent}%</span></div>'
            for p in result.predictions[1:])
        others = ('<div style="margin-top:14px;padding-top:12px;'
                  'border-top:1px solid #1a1a2e"><div style="font-size:11px;'
                  'color:#444;font-family:Rajdhani,sans-serif;margin-bottom:6px">'
                  f'OTHER CANDIDATES</div>{rows}</div>')

    footer = (f'<div style="font-size:10px;color:#2e2e46;'
              f'font-family:Rajdhani,sans-serif;margin-top:12px">'
              f'{len(CLASSIFIER.tta)} TTA views  ·  {result.elapsed_ms:.0f} ms</div>')

    return f"""
    <div style="background:#0d0d1a;border:1px solid #1e1e3a;border-radius:12px;
                padding:18px;font-family:sans-serif">
      {unsure}{conf_bar}{header}{types}{legendary}{stats_html}{matchup_html}{others}{footer}
    </div>"""


def build_team_html(team) -> str:
    if not team:
        return ('<div style="color:#333;text-align:center;padding:20px;'
                'font-family:Rajdhani,sans-serif">No Pokemon in your team yet</div>')

    slots = ""
    for i, label in enumerate(team):
        s = dex.get(label)
        slots += f"""
        <div style="background:#0d0d1a;border:1px solid #1e1e3a;border-radius:8px;
                    padding:10px;text-align:center">
          <div style="font-size:10px;color:#333;font-family:Rajdhani,sans-serif;
                      margin-bottom:2px">{i + 1}</div>
          <div style="font-size:13px;font-weight:700;color:#ddd;
                      font-family:Rajdhani,sans-serif;line-height:1.2">
            {s['display_name']}</div>
          <div style="font-size:10px;color:#444;font-family:Rajdhani,sans-serif">
            #{s.get('pokedex_number', 0) or '???'}</div>
          <div style="margin-top:4px">
            {''.join(type_badge(t) for t in s.get('types', [])[:1])}</div>
        </div>"""
    for i in range(len(team), MAX_TEAM):
        slots += (f'<div style="background:#08080f;border:1px dashed #1a1a2e;'
                  f'border-radius:8px;padding:10px;text-align:center;color:#222;'
                  f'font-family:Rajdhani,sans-serif;font-size:11px">{i + 1}<br>'
                  f'empty</div>')
    return (f'<div style="display:grid;grid-template-columns:repeat({MAX_TEAM},1fr);'
            f'gap:8px">{slots}</div>')


def build_analysis_html(team) -> str:
    if not team:
        return ""
    report = dex.team_report(team)

    coverage = "".join(
        f'{type_badge(t)}<span style="font-size:11px;color:#555;'
        f'font-family:Rajdhani,sans-serif">x{c} </span>'
        for t, c in report["type_counts"].items())

    shared = [(t, c) for t, c in report["weaknesses"].items() if c >= 2]
    weak_html = "".join(
        f'{type_badge(t)}<span style="font-size:11px;color:#555;'
        f'font-family:Rajdhani,sans-serif">{c} members </span>'
        for t, c in shared[:6])

    gaps = report["uncovered_types"]
    gaps_html = "".join(type_badge(t) for t in gaps)

    return f"""
    <div style="background:#0d0d1a;border:1px solid #1e1e3a;border-radius:12px;
                padding:16px;margin-top:10px">
      <div style="font-family:Rajdhani,sans-serif;font-size:11px;color:#444;
                  margin-bottom:8px">TYPE COVERAGE</div>
      <div style="margin-bottom:12px">{coverage or '<span style="color:#333">-</span>'}</div>

      <div style="font-family:Rajdhani,sans-serif;font-size:11px;color:#444;
                  margin-bottom:8px">SHARED WEAKNESSES (2+ members)</div>
      <div style="margin-bottom:12px">
        {weak_html or '<span style="color:#4CAF50;font-size:12px;font-family:Rajdhani,sans-serif">None - the team is well spread</span>'}</div>

      <div style="font-family:Rajdhani,sans-serif;font-size:11px;color:#444;
                  margin-bottom:8px">NO SUPER-EFFECTIVE COVERAGE AGAINST</div>
      <div>{gaps_html or '<span style="color:#4CAF50;font-size:12px;font-family:Rajdhani,sans-serif">Every type is covered</span>'}</div>
    </div>"""


# -- handlers -----------------------------------------------------------------
def on_scan(image, isolate, team):
    if image is None:
        return (empty_card("Upload an image or take a webcam shot first"),
                None, None, build_team_html(team), build_analysis_html(team))
    CLASSIFIER.isolate = bool(isolate)
    result = CLASSIFIER.predict(image)
    model_input = CLASSIFIER.prepare(image)
    return (build_result_html(result), result, model_input,
            build_team_html(team), build_analysis_html(team))


def on_add(result, team):
    team = list(team or [])
    if result is None or not result.predictions:
        return team, build_team_html(team), build_analysis_html(team), "Scan something first"
    top = result.predictions[0]
    if not result.is_confident:
        return (team, build_team_html(team), build_analysis_html(team),
                "Confidence is too low to trust - try another shot")
    if len(team) >= MAX_TEAM:
        return team, build_team_html(team), build_analysis_html(team), "Team is full (6/6)"
    if top.label in team:
        return (team, build_team_html(team), build_analysis_html(team),
                f"{top.display_name} is already on the team")
    team.append(top.label)
    return (team, build_team_html(team), build_analysis_html(team),
            f"Added {top.display_name} ({len(team)}/6)")


def on_undo(team):
    team = list(team or [])
    if not team:
        return team, build_team_html(team), build_analysis_html(team), "Team is empty"
    removed = dex.get(team.pop())["display_name"]
    return (team, build_team_html(team), build_analysis_html(team),
            f"Removed {removed}")


def on_clear():
    return [], build_team_html([]), build_analysis_html([]), "Team cleared"


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=DM+Sans:wght@300;400;500&display=swap');
body, .gradio-container { background:#07070f !important; font-family:'DM Sans',sans-serif !important; }
.gradio-container { max-width:1100px !important; margin:0 auto !important; }
h1 { font-family:'Rajdhani',sans-serif !important; font-size:32px !important; font-weight:700 !important;
     letter-spacing:2px !important; color:#fff !important; margin:0 !important; }
.scan-btn { background:#e8362a !important; border:none !important; color:#fff !important;
            font-family:'Rajdhani',sans-serif !important; font-weight:700 !important;
            letter-spacing:1px !important; font-size:15px !important; }
.scan-btn:hover { background:#c42a1f !important; }
.add-btn { background:transparent !important; border:1px solid #1e1e3a !important; color:#aaa !important;
           font-family:'Rajdhani',sans-serif !important; font-weight:600 !important; }
.add-btn:hover { border-color:#4CAF50 !important; color:#4CAF50 !important; }
.clear-btn { background:transparent !important; border:1px solid #1e1e3a !important; color:#555 !important;
             font-family:'Rajdhani',sans-serif !important; }
.clear-btn:hover { border-color:#e8362a !important; color:#e8362a !important; }
.gr-panel, .gr-box { background:#0d0d1a !important; border:1px solid #1e1e3a !important; border-radius:12px !important; }
label { color:#555 !important; font-family:'Rajdhani',sans-serif !important; font-size:11px !important; letter-spacing:1px !important; }
.status-msg { font-family:'Rajdhani',sans-serif !important; font-size:13px !important; color:#4CAF50 !important; }
"""


# Gradio 6 moved `css` from the Blocks constructor to launch(); older
# versions only accept it on Blocks. Pass it wherever this version wants it.
GRADIO_MAJOR = int(gr.__version__.split(".")[0])
_blocks_kwargs = {"title": "PokeScanner"}
if GRADIO_MAJOR < 6:
    _blocks_kwargs["css"] = CSS

with gr.Blocks(**_blocks_kwargs) as demo:
    team_state = gr.State([])
    result_state = gr.State(None)

    gr.HTML("""
    <div style="display:flex;align-items:center;gap:14px;padding:20px 0 10px">
      <div style="width:36px;height:36px;background:#e8362a;border-radius:50%;
                  display:flex;align-items:center;justify-content:center;
                  font-size:18px">&#9673;</div>
      <div>
        <h1>Poke<span style="color:#e8362a">Scanner</span></h1>
        <div style="font-family:Rajdhani,sans-serif;font-size:11px;color:#333;
                    letter-spacing:2px">REAL-TIME POKEMON IDENTIFIER</div>
      </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="UPLOAD IMAGE OR USE WEBCAM",
                                   sources=["upload", "webcam"],
                                   type="numpy", height=320)
            scan_btn = gr.Button("SCAN", elem_classes="scan-btn")
            isolate_cb = gr.Checkbox(
                value=CLASSIFIER.isolate, label="ISOLATE SUBJECT",
                info="Crop to the Pokemon before classifying. Turn off to compare.")
            with gr.Row():
                add_btn = gr.Button("+ ADD TO TEAM", elem_classes="add-btn")
                undo_btn = gr.Button("UNDO", elem_classes="add-btn")
                clear_btn = gr.Button("CLEAR", elem_classes="clear-btn")
            status_box = gr.Textbox(label="", interactive=False,
                                    elem_classes="status-msg")
            model_input_img = gr.Image(label="WHAT THE MODEL SEES", height=180,
                                       interactive=False)

        with gr.Column(scale=1):
            result_html = gr.HTML(empty_card())

    gr.HTML('<div style="font-family:Rajdhani,sans-serif;font-size:11px;color:#333;'
            'letter-spacing:2px;margin:16px 0 8px">MY TEAM</div>')
    team_html = gr.HTML(build_team_html([]))
    analysis_html = gr.HTML("")

    scan_btn.click(on_scan, [image_input, isolate_cb, team_state],
                   [result_html, result_state, model_input_img, team_html, analysis_html])
    add_btn.click(on_add, [result_state, team_state],
                  [team_state, team_html, analysis_html, status_box])
    undo_btn.click(on_undo, [team_state],
                   [team_state, team_html, analysis_html, status_box])
    clear_btn.click(on_clear, None,
                    [team_state, team_html, analysis_html, status_box])


if __name__ == "__main__":
    launch_kwargs = {"inbrowser": True}
    if GRADIO_MAJOR >= 6:
        launch_kwargs["css"] = CSS
    demo.launch(**launch_kwargs)
