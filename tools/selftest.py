"""Fast checks on the non-model parts of PokeScanner.

    python tools/selftest.py

No model weights and no pytest needed - this covers the type chart, the
form-aware name matching and the image preparation, which are the parts most
likely to break silently when the data files change.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokescanner import config, dex, vision  # noqa: E402

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# -- type chart ---------------------------------------------------------------
@check
def type_chart_is_symmetric_in_size():
    assert len(dex.TYPES) == 18
    assert set(dex._CHART) == set(dex.TYPES)
    for attacker, row in dex._CHART.items():
        unknown = set(row) - set(dex.TYPES)
        assert not unknown, f"{attacker} hits unknown types {unknown}"


@check
def known_matchups_are_right():
    cases = [
        ("electric", ["ground"], 0.0),          # immunity
        ("water", ["fire", "ground"], 4.0),     # double weakness
        ("fighting", ["ghost"], 0.0),
        ("rock", ["fire", "flying"], 4.0),      # Charizard's famous 4x
        ("ground", ["flying", "steel"], 0.0),   # immunity beats weakness
        ("normal", ["rock", "steel"], 0.25),
        ("dragon", ["fairy"], 0.0),
        ("psychic", ["dark"], 0.0),
        ("grass", ["water"], 2.0),
    ]
    for attacker, defenders, expected in cases:
        got = dex.effectiveness(attacker, defenders)
        assert got == expected, f"{attacker} vs {defenders}: {got} != {expected}"


@check
def charizard_is_four_times_weak_to_rock():
    weak = dex.weaknesses("charizard")
    assert weak.get("rock") == 4.0, weak
    assert "grass" not in weak                  # Fire resists Grass


@check
def team_report_counts_members_not_types():
    report = dex.team_report(["charizard", "moltres"])
    # both are Fire/Flying, so both are weak to Rock
    assert report["weaknesses"]["rock"] == 2, report["weaknesses"]
    assert report["size"] == 2
    assert isinstance(report["uncovered_types"], list)


@check
def empty_team_report_is_safe():
    report = dex.team_report([])
    assert report["size"] == 0
    assert report["type_counts"] == {}


# -- name matching ------------------------------------------------------------
@check
def normalise_handles_awkward_names():
    cases = {
        "Mr. Mime": "mr-mime",
        "Farfetch'd": "farfetchd",
        "Flabébé": "flabebe",
        "Nidoran♀": "nidoran-f",
        "Ho-Oh": "ho-oh",
        "Type: Null": "type-null",
    }
    for raw, expected in cases.items():
        got = dex.normalise(raw)
        assert got == expected, f"{raw!r} -> {got!r}, wanted {expected!r}"


@check
def form_variants_resolve_to_their_species():
    if not config.STATS_CSV.exists():
        return "skipped (no stats csv)"
    for label, expected_dex in [("giratina-altered", 487),
                                ("aegislash-blade", 681),
                                ("deoxys-normal", 386),
                                ("zygarde-50", 718),
                                ("meowstic-male", 678)]:
        entry = dex.get(label)
        assert entry["has_stats"], f"{label} still has no stats"
        assert entry["pokedex_number"] == expected_dex, \
            f"{label} -> #{entry['pokedex_number']}, wanted #{expected_dex}"


@check
def every_class_has_a_typing():
    if not config.TYPES_CSV.exists():
        return "skipped (no types csv)"
    db = dex.load()
    missing = [k for k, v in db.items() if not v["types"]]
    assert not missing, f"{len(missing)} classes have no type: {missing[:5]}"


@check
def unknown_label_does_not_crash():
    entry = dex.get("not-a-pokemon")
    assert entry["types"] == []
    assert entry["has_stats"] is False


# -- image preparation --------------------------------------------------------
@check
def letterbox_preserves_aspect_ratio():
    img = Image.new("RGB", (400, 100), (10, 20, 30))
    out = vision.letterbox(img, 260)
    assert out.size == (260, 260)
    arr = np.asarray(out)
    # the padding is the canvas colour, the middle band is the image
    assert tuple(arr[0, 0]) == config.CANVAS_COLOR
    assert tuple(arr[130, 130]) == (10, 20, 30)


@check
def alpha_is_flattened_onto_the_canvas():
    img = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    img.paste((255, 0, 0, 255), (20, 20, 30, 30))
    out = vision.flatten_alpha(img)
    assert out.mode == "RGB"
    assert tuple(np.asarray(out)[0, 0]) == config.CANVAS_COLOR


@check
def isolate_crops_to_the_alpha_bbox():
    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    img.paste((0, 200, 0, 255), (90, 90, 110, 110))     # 20x20 subject
    out = vision.isolate_subject(img, fill=1.0)
    assert out.size == (20, 20), out.size
    assert tuple(np.asarray(out)[10, 10]) == (0, 200, 0)


@check
def isolate_finds_a_subject_without_alpha():
    arr = np.full((240, 320, 3), 200, np.uint8)         # flat background
    arr[90:150, 120:200] = (30, 60, 180)                # a block in the middle
    out = vision.isolate_subject(Image.fromarray(arr), fill=1.0)
    w, h = out.size
    # should land near the 80x60 block rather than the whole 320x240 frame
    assert 40 < w < 160 and 30 < h < 140, out.size


@check
def prepare_accepts_arrays_and_paths_and_rgba():
    for source in (np.zeros((120, 160, 3), np.uint8),
                   np.zeros((120, 160, 4), np.uint8),
                   Image.new("RGB", (160, 120), (5, 5, 5))):
        out = vision.prepare(source, 224)
        assert out.size == (224, 224) and out.mode == "RGB"


@check
def prepare_survives_a_degenerate_image():
    # a single flat colour has no subject at all; must not raise
    out = vision.prepare(Image.new("RGB", (64, 64), (128, 128, 128)), 224)
    assert out.size == (224, 224)


# -- runner -------------------------------------------------------------------
def main() -> int:
    passed = failed = skipped = 0
    for fn in CHECKS:
        name = fn.__name__.replace("_", " ")
        try:
            note = fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}\n      {exc}")
        except Exception:
            failed += 1
            print(f"ERROR {name}")
            traceback.print_exc()
        else:
            if isinstance(note, str) and note.startswith("skipped"):
                skipped += 1
                print(f"skip  {name}  ({note})")
            else:
                passed += 1
                print(f"ok    {name}")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
