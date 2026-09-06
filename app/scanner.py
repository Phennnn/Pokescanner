"""PokeScanner - real-time webcam scanner (OpenCV window).

    python app/scanner.py

Controls
    SPACE   scan (averages several frames)
    A       add the current Pokemon to your team
    C       clear the team
    T       team type analysis on/off
    I       subject isolation on/off (see what the model actually receives)
    M       cycle scan mode: auto (read card, else classify) / card / model
    S       save the last scan to reports/scans/
    Q       quit

The targeting brackets are now real: only what is inside them is classified.
The old version drew the box but sent the entire frame to the model, so a
Pokemon card held in the middle of a messy desk competed with the desk.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokescanner import cards, config, dex, identify   # noqa: E402
from pokescanner.inference import PokemonClassifier    # noqa: E402

SCAN_FRAMES = 5
ROI_FRACTION = 0.62      # side of the targeting box, as a share of frame height

PANEL_W = 300
BG = (18, 16, 28)
FG = (235, 235, 240)
DIM = (120, 118, 130)
ACCENT = (60, 60, 232)
GOOD = (90, 200, 110)
WARN = (60, 180, 250)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


TYPE_BGR = {t: hex_to_bgr(c) for t, c in dex.TYPE_COLORS.items()}


# -- drawing ------------------------------------------------------------------
def roi_box(frame):
    h, w = frame.shape[:2]
    side = int(min(h, w) * ROI_FRACTION)
    cx, cy = (w - PANEL_W) // 2, h // 2
    return (max(0, cx - side // 2), max(0, cy - side // 2),
            min(w - PANEL_W, cx + side // 2), min(h, cy + side // 2))


def draw_brackets(frame, box, scanning: bool):
    x1, y1, x2, y2 = box
    color = GOOD if scanning else ACCENT
    corner, thick = 26, 2
    for px, py, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                           (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (px, py), (px + dx * corner, py), color, thick)
        cv2.line(frame, (px, py), (px, py + dy * corner), color, thick)
    if scanning:
        t = time.time() % 1.2 / 1.2
        sy = int(y1 + (y2 - y1) * abs(2 * t - 1))
        cv2.line(frame, (x1 + 4, sy), (x2 - 4, sy), (120, 255, 160), 1)


def text(frame, s, xy, scale=0.4, color=FG, thick=1):
    cv2.putText(frame, s, xy, FONT, scale, color, thick, cv2.LINE_AA)


def bar(frame, x, y, width, height, value, color, bg=(48, 46, 62)):
    cv2.rectangle(frame, (x, y), (x + width, y + height), bg, -1)
    filled = int(width * max(0.0, min(1.0, value)))
    if filled:
        cv2.rectangle(frame, (x, y), (x + filled, y + height), color, -1)


def draw_panel(frame, result, team, show_team_analysis, isolate_on):
    h, w = frame.shape[:2]
    x0 = w - PANEL_W
    cv2.rectangle(frame, (x0, 0), (w, h), BG, -1)
    cv2.line(frame, (x0, 0), (x0, h), (44, 42, 58), 1)

    px = x0 + 14
    y = 28
    text(frame, "POKESCANNER", (px, y), 0.55, ACCENT, 2)
    y += 22
    text(frame, f"isolate {'ON' if isolate_on else 'OFF'}   "
                f"ocr {cards.OCR.kind}", (px, y), 0.34, DIM)
    y += 22

    if result is None or not result.label:
        text(frame, "Aim inside the brackets", (px, y), 0.38, DIM)
        text(frame, "and press SPACE", (px, y + 16), 0.38, DIM)
        y += 46
    else:
        conf = result.confidence
        conf_color = GOOD if conf > 0.5 else WARN if conf > 0.25 else (70, 70, 220)

        bar(frame, px, y, PANEL_W - 28, 14, conf, conf_color)
        text(frame, f"{conf * 100:.1f}%", (px + 6, y + 11), 0.38, (255, 255, 255))
        y += 30

        if not result.is_confident:
            text(frame, "NOT SURE - best guess:", (px, y), 0.36, (80, 170, 250))
            y += 18

        text(frame, result.display_name[:20], (px, y + 14), 0.62, FG, 2)
        y += 30

        # say which path answered: reading the card is a different kind of
        # evidence from classifying the picture, and the user should know
        if result.source == "card":
            label = f"READ FROM CARD: \"{result.card_text}\""
            text(frame, label[:34], (px, y), 0.33, (120, 220, 140))
            y += 18

        s = result.stats
        types = s.get("types", [])
        if types:
            tx = px
            for t in types:
                cv2.rectangle(frame, (tx, y), (tx + 82, y + 17), TYPE_BGR.get(t, DIM), -1)
                text(frame, t.upper()[:9], (tx + 5, y + 13), 0.33, (255, 255, 255))
                tx += 88
            y += 26

        if s.get("has_stats"):
            for label, key in (("HP", "hp"), ("ATK", "attack"), ("DEF", "defense"),
                               ("SPA", "sp_attack"), ("SPD", "sp_defense"),
                               ("SPE", "speed")):
                val = s.get(key, 0)
                text(frame, label, (px, y + 10), 0.34, DIM)
                bar(frame, px + 40, y + 2, 180, 10, val / 200.0, (80, 180, 90))
                text(frame, str(val), (px + 228, y + 10), 0.33, (200, 200, 200))
                y += 16
            text(frame, f"BST {s.get('base_total', 0)}   "
                        f"#{s.get('pokedex_number', 0):03d}   "
                        f"GEN {s.get('generation', '?')}", (px, y + 12), 0.34, DIM)
            y += 24
        else:
            text(frame, "no stats on file for this form", (px, y + 10), 0.33, DIM)
            y += 22

        if s.get("legendary"):
            text(frame, "* LEGENDARY", (px, y + 10), 0.42, (0, 215, 255))
            y += 20

        weak = dex.weaknesses(result.label)
        if weak:
            worst = sorted(weak.items(), key=lambda kv: -kv[1])[:4]
            text(frame, "WEAK TO", (px, y + 10), 0.32, DIM)
            y += 18
            tx = px
            for t, mult in worst:
                cv2.rectangle(frame, (tx, y), (tx + 62, y + 15), TYPE_BGR.get(t, DIM), -1)
                text(frame, f"{t[:5].upper()}x{mult:g}", (tx + 3, y + 11), 0.28,
                     (255, 255, 255))
                tx += 68
            y += 24

        if result.alternatives:
            text(frame, "also considered", (px, y + 10), 0.32, DIM)
            y += 20
            for alt in result.alternatives[:3]:
                text(frame, f"{alt['display_name'][:18]}  {alt['confidence']:.1f}%",
                     (px, y), 0.34, (128, 126, 140))
                y += 15
            y += 6

        text(frame, f"via {result.source}  {result.elapsed_ms:.0f} ms",
             (px, y + 8), 0.31, (86, 84, 96))
        y += 22

    # team
    ty = h - 190
    text(frame, f"TEAM ({len(team)}/6)", (px, ty), 0.45, ACCENT)
    ty += 20
    for i, label in enumerate(team):
        entry = dex.get(label)
        t1 = entry["types"][0] if entry["types"] else None
        if t1:
            cv2.rectangle(frame, (px, ty - 9), (px + 5, ty + 2), TYPE_BGR[t1], -1)
        text(frame, f"{i + 1}. {entry['display_name'][:19]}", (px + 12, ty), 0.37,
             (205, 205, 210))
        ty += 16

    if show_team_analysis and team:
        report = dex.team_report(team)
        ty += 6
        text(frame, "SHARED WEAKNESSES", (px, ty), 0.32, DIM)
        ty += 15
        shared = [(t, c) for t, c in report["weaknesses"].items() if c >= 2][:4]
        if shared:
            tx = px
            for t, count in shared:
                cv2.rectangle(frame, (tx, ty - 9), (tx + 62, ty + 4),
                              TYPE_BGR.get(t, DIM), -1)
                text(frame, f"{t[:5].upper()}x{count}", (tx + 3, ty + 1), 0.28,
                     (255, 255, 255))
                tx += 68
        else:
            text(frame, "none - well balanced", (px, ty + 1), 0.32, GOOD)

    text(frame, "SPACE scan  A add  C clear", (x0 + 8, h - 26), 0.31, (92, 90, 104))
    text(frame, "T team  I isolate  M mode  S save  Q quit", (x0 + 8, h - 12),
         0.31, (92, 90, 104))


# -- main ---------------------------------------------------------------------
def main() -> int:
    print("PokeScanner - webcam scanner")
    try:
        clf = PokemonClassifier()
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        return 1
    clf.warmup()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
    if not cap.isOpened():
        print("Could not open the webcam. Is another app using it?")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Ready. Aim inside the brackets and press SPACE.\n")

    result = None
    team: list[str] = []
    last_scan_at = 0.0
    show_team_analysis = False
    last_crop = None
    scan_mode = "auto"      # auto | card | model

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Not mirrored on purpose: this is a scanner, not a selfie camera.
        # A mirrored frame shows every card name backwards and makes the OCR
        # path fail on cards it would otherwise read.
        box = roi_box(frame)
        draw_brackets(frame, box, (time.time() - last_scan_at) < 0.6)
        draw_panel(frame, result, team, show_team_analysis, clf.isolate)
        cv2.imshow("PokeScanner", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

        if key == ord(" "):
            last_scan_at = time.time()
            x1, y1, x2, y2 = box
            crops = []
            for _ in range(SCAN_FRAMES):
                ok2, frame2 = cap.read()
                if not ok2:
                    continue
                crop = frame2[y1:y2, x1:x2]
                if crop.size:
                    crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if crops:
                last_crop = crops[-1]
                result = identify.identify(crops[-1], clf, mode=scan_mode)
                if result.label:
                    mark = "" if result.is_confident else "  (low confidence)"
                    via = (f' [read "{result.card_text}" off the card]'
                           if result.source == "card" else "")
                    print(f"  -> {result.display_name} "
                          f"{result.confidence * 100:.1f}%{mark}{via}")
                else:
                    print("  -> nothing recognised")

        elif key == ord("a"):
            if result and result.label:
                label = result.label
                if not result.is_confident:
                    print("  too unsure to add - scan again")
                elif len(team) >= 6:
                    print("  team is full")
                elif label in team:
                    print(f"  {result.display_name} is already on the team")
                else:
                    team.append(label)
                    print(f"  added {result.display_name} ({len(team)}/6)")

        elif key == ord("c"):
            team.clear()
            print("  team cleared")

        elif key == ord("t"):
            show_team_analysis = not show_team_analysis

        elif key == ord("m"):
            order = ["auto", "card", "model"]
            scan_mode = order[(order.index(scan_mode) + 1) % len(order)]
            print(f"  scan mode: {scan_mode}")

        elif key == ord("i"):
            clf.isolate = not clf.isolate
            print(f"  subject isolation {'on' if clf.isolate else 'off'}")

        elif key == ord("s") and last_crop is not None and result:
            out = config.ROOT / "reports" / "scans"
            out.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            name = result.label or "unknown"
            cv2.imwrite(str(out / f"{stamp}_{name}_raw.jpg"),
                        cv2.cvtColor(last_crop, cv2.COLOR_RGB2BGR))
            prepared = np.asarray(clf.prepare(last_crop))
            cv2.imwrite(str(out / f"{stamp}_{name}_model_input.jpg"),
                        cv2.cvtColor(prepared, cv2.COLOR_RGB2BGR))
            print(f"  saved to {out}")

    cap.release()
    cv2.destroyAllWindows()
    if team:
        report = dex.team_report(team)
        print("\nFinal team: " +
              ", ".join(dex.get(t)["display_name"] for t in team))
        shared = [f"{t} x{c}" for t, c in report["weaknesses"].items() if c >= 2]
        if shared:
            print("Shared weaknesses: " + ", ".join(shared))
        if report["uncovered_types"]:
            print("No super-effective coverage against: " +
                  ", ".join(report["uncovered_types"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
