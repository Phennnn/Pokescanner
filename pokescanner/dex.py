"""Pokedex data: stats, types and type matchups.

Loaded once and shared by every app. Two things it does that the old per-app
copies did not:

  * Form-aware name matching. Labels like ``giratina-altered`` or
    ``aegislash-blade`` do not appear in the stats CSV, which is why 30 species
    used to render an empty card. Trailing form tokens are stripped until a
    match is found.
  * A real 18x18 type chart, so weaknesses and resistances are computed for all
    809 classes instead of only the ones with ``against_*`` columns.
"""

from __future__ import annotations

import functools
import json
import unicodedata
from typing import Dict, List, Optional

import pandas as pd

from . import config

# -- Type chart (generation 6+) -----------------------------------------------
TYPES: List[str] = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison",
    "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark",
    "steel", "fairy",
]

# attacker -> {defender: multiplier}; anything unlisted is 1.0
_CHART: Dict[str, Dict[str, float]] = {
    "normal":   {"rock": .5, "steel": .5, "ghost": 0},
    "fire":     {"grass": 2, "ice": 2, "bug": 2, "steel": 2,
                 "fire": .5, "water": .5, "rock": .5, "dragon": .5},
    "water":    {"fire": 2, "ground": 2, "rock": 2,
                 "water": .5, "grass": .5, "dragon": .5},
    "electric": {"water": 2, "flying": 2,
                 "electric": .5, "grass": .5, "dragon": .5, "ground": 0},
    "grass":    {"water": 2, "ground": 2, "rock": 2,
                 "fire": .5, "grass": .5, "poison": .5, "flying": .5,
                 "bug": .5, "dragon": .5, "steel": .5},
    "ice":      {"grass": 2, "ground": 2, "flying": 2, "dragon": 2,
                 "fire": .5, "water": .5, "ice": .5, "steel": .5},
    "fighting": {"normal": 2, "ice": 2, "rock": 2, "dark": 2, "steel": 2,
                 "poison": .5, "flying": .5, "psychic": .5, "bug": .5,
                 "fairy": .5, "ghost": 0},
    "poison":   {"grass": 2, "fairy": 2,
                 "poison": .5, "ground": .5, "rock": .5, "ghost": .5, "steel": 0},
    "ground":   {"fire": 2, "electric": 2, "poison": 2, "rock": 2, "steel": 2,
                 "grass": .5, "bug": .5, "flying": 0},
    "flying":   {"grass": 2, "fighting": 2, "bug": 2,
                 "electric": .5, "rock": .5, "steel": .5},
    "psychic":  {"fighting": 2, "poison": 2,
                 "psychic": .5, "steel": .5, "dark": 0},
    "bug":      {"grass": 2, "psychic": 2, "dark": 2,
                 "fire": .5, "fighting": .5, "poison": .5, "flying": .5,
                 "ghost": .5, "steel": .5, "fairy": .5},
    "rock":     {"fire": 2, "ice": 2, "flying": 2, "bug": 2,
                 "fighting": .5, "ground": .5, "steel": .5},
    "ghost":    {"psychic": 2, "ghost": 2, "dark": .5, "normal": 0},
    "dragon":   {"dragon": 2, "steel": .5, "fairy": 0},
    "dark":     {"psychic": 2, "ghost": 2,
                 "fighting": .5, "dark": .5, "fairy": .5},
    "steel":    {"ice": 2, "rock": 2, "fairy": 2,
                 "fire": .5, "water": .5, "electric": .5, "steel": .5},
    "fairy":    {"fighting": 2, "dragon": 2, "dark": 2,
                 "fire": .5, "poison": .5, "steel": .5},
}

TYPE_COLORS: Dict[str, str] = {
    "normal": "#A8A878", "fire": "#FF6B35", "water": "#4A9EFF",
    "electric": "#FFD700", "grass": "#5DBE6E", "ice": "#96D9D6",
    "fighting": "#C22E28", "poison": "#A33EA1", "ground": "#E2BF65",
    "flying": "#89AAE3", "psychic": "#FF6EB4", "bug": "#A6B91A",
    "rock": "#B6A136", "ghost": "#735797", "dragon": "#6F35FC",
    "dark": "#705746", "steel": "#B7B7CE", "fairy": "#D685AD",
}


def effectiveness(attacker: str, defender_types) -> float:
    """Damage multiplier of one attacking type against a defending typing."""
    row = _CHART.get(attacker, {})
    mult = 1.0
    for t in defender_types:
        if t in TYPES:
            mult *= row.get(t, 1.0)
    return mult


def matchups(defender_types) -> Dict[str, float]:
    """{attacking type: multiplier} for every type that is not neutral."""
    types = [t for t in defender_types if t in TYPES]
    if not types:
        return {}
    out = {}
    for atk in TYPES:
        m = effectiveness(atk, types)
        if m != 1.0:
            out[atk] = m
    return out


# -- Name normalisation -------------------------------------------------------
# Trailing tokens that denote a form rather than a species.
_FORM_TOKENS = {
    "normal", "altered", "land", "plant", "sandy", "trash", "standard", "zen",
    "incarnate", "therian", "ordinary", "resolute", "aria", "pirouette",
    "male", "female", "blade", "shield", "average", "small", "large", "super",
    "confined", "unbound", "baile", "pau", "pompom", "sensu", "midday",
    "midnight", "dusk", "solo", "school", "meteor", "core", "red", "striped",
    "blue", "white", "black", "attack", "defense", "speed", "sky", "origin",
    "sunny", "rainy", "snowy", "west", "east", "mega", "primal", "alola",
    "galar", "hisui", "complete", "50", "10", "eternal", "ultra", "dawn",
    "sunshine", "overcast", "active", "archipelago", "continental", "elegant",
    "fancy", "garden", "high", "plains", "icy", "jungle", "marine", "meadow",
    "modern", "monsoon", "ocean", "poke", "polar", "river", "sandstorm",
    "savanna", "sun", "tundra", "natural", "heat", "wash", "frost", "fan",
    "mow", "spring", "summer", "autumn", "winter", "shock", "burn", "chill",
    "douse", "disguised", "busted", "single", "rapid", "gulping", "gorging",
    "amped", "low", "key", "noice", "hangry", "crowned", "eternamax",
}

# Species whose canonical name genuinely contains one of the tokens above,
# or that need an outright override.
_EXPLICIT_ALIASES = {
    "nidoran-f": "nidoran",
    "nidoran-m": "nidoran",
    "mr-mime": "mr. mime",
    "mime-jr": "mime jr.",
    "type-null": "type: null",
    "porygon-z": "porygon-z",
    "ho-oh": "ho-oh",
    "jangmo-o": "jangmo-o",
    "hakamo-o": "hakamo-o",
    "kommo-o": "kommo-o",
    "farfetchd": "farfetch'd",
    "sirfetchd": "sirfetch'd",
    "flabebe": "flabebe",
}


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


def normalise(name: str) -> str:
    """'Flabebe' / 'Mr. Mime' / 'Nidoran-F' -> a stable lowercase-hyphen key."""
    text = strip_accents(str(name)).lower().strip()
    text = text.replace("♀", "-f").replace("♂", "-m")
    # Apostrophes are dropped rather than treated as separators, so
    # "Farfetch'd" becomes "farfetchd" and matches the folder name.
    text = text.replace("'", "").replace("’", "")
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")


def _candidates(label: str):
    """Progressively shorter keys to try, longest first."""
    yield label
    alias = _EXPLICIT_ALIASES.get(label)
    if alias:
        yield normalise(alias)
    parts = label.split("-")
    while len(parts) > 1 and parts[-1] in _FORM_TOKENS:
        parts = parts[:-1]
        yield "-".join(parts)


# -- Loading ------------------------------------------------------------------
_EMPTY = {
    "hp": 0, "attack": 0, "defense": 0, "sp_attack": 0, "sp_defense": 0,
    "speed": 0, "base_total": 0, "type1": "", "type2": "", "legendary": False,
    "generation": 0, "pokedex_number": 0, "capture_rate": "?",
    "classfication": "", "height_m": 0.0, "weight_kg": 0.0,
    "has_stats": False,
}


def _as_int(value, default=0) -> int:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value, default=0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_type(value) -> str:
    text = str(value).lower().strip()
    return text if text in TYPES else ""


@functools.lru_cache(maxsize=1)
def load() -> Dict[str, dict]:
    """{label: stats dict} covering every class in the label map."""
    labels: List[str] = []
    if config.LABEL_MAP_PATH.exists():
        with open(config.LABEL_MAP_PATH, encoding="utf-8") as fh:
            labels = list(json.load(fh)["label_to_idx"].keys())

    # 1. types for all 809 classes (this CSV is keyed by the same slugs)
    type_by_label: Dict[str, tuple] = {}
    if config.TYPES_CSV.exists():
        tdf = pd.read_csv(config.TYPES_CSV, encoding="utf-8")
        for _, row in tdf.iterrows():
            type_by_label[normalise(row["Name"])] = (
                _clean_type(row.get("Type1")), _clean_type(row.get("Type2")))

    # 2. full stats, keyed by normalised species name
    stats_by_name: Dict[str, dict] = {}
    if config.STATS_CSV.exists():
        sdf = pd.read_csv(config.STATS_CSV, encoding="utf-8")
        for _, row in sdf.iterrows():
            stats_by_name[normalise(row["name"])] = {
                "hp": _as_int(row.get("hp")),
                "attack": _as_int(row.get("attack")),
                "defense": _as_int(row.get("defense")),
                "sp_attack": _as_int(row.get("sp_attack")),
                "sp_defense": _as_int(row.get("sp_defense")),
                "speed": _as_int(row.get("speed")),
                "base_total": _as_int(row.get("base_total")),
                "type1": _clean_type(row.get("type1")),
                "type2": _clean_type(row.get("type2")),
                "legendary": bool(_as_int(row.get("is_legendary"))),
                "generation": _as_int(row.get("generation")),
                "pokedex_number": _as_int(row.get("pokedex_number")),
                "capture_rate": str(row.get("capture_rate", "?")),
                "classfication": str(row.get("classfication", "") or ""),
                "height_m": _as_float(row.get("height_m")),
                "weight_kg": _as_float(row.get("weight_kg")),
                "has_stats": True,
            }

    if not labels:
        labels = sorted(set(type_by_label) | set(stats_by_name))

    db: Dict[str, dict] = {}
    for label in labels:
        entry = None
        for key in _candidates(label):
            if key in stats_by_name:
                entry = dict(stats_by_name[key])
                break
        if entry is None:
            entry = dict(_EMPTY)

        # types from the per-label CSV win: they exist for every class and are
        # correct for form variants the stats CSV does not carry.
        t1, t2 = type_by_label.get(label, ("", ""))
        if t1:
            entry["type1"], entry["type2"] = t1, t2

        entry["label"] = label
        entry["display_name"] = label.replace("-", " ").title()
        entry["types"] = [t for t in (entry["type1"], entry["type2"]) if t]
        db[label] = entry
    return db


def get(label: str) -> dict:
    """Stats for a label, always a dict (never None)."""
    entry = load().get(label)
    if entry is None:
        entry = dict(_EMPTY)
        entry["label"] = label
        entry["display_name"] = str(label).replace("-", " ").title()
        entry["types"] = []
    return entry


def weaknesses(label: str) -> Dict[str, float]:
    """{attacking type: multiplier > 1} for one Pokemon."""
    return {t: m for t, m in matchups(get(label)["types"]).items() if m > 1}


def resistances(label: str) -> Dict[str, float]:
    """{attacking type: multiplier < 1} for one Pokemon, immunities included."""
    return {t: m for t, m in matchups(get(label)["types"]).items() if m < 1}


def team_report(labels: List[str]) -> dict:
    """Shared type coverage / weakness analysis used by every UI.

    Returns counts of how many team members are weak to (or resist) each type,
    plus the types nobody on the team can hit for super-effective damage.
    """
    weak_counts: Dict[str, int] = {t: 0 for t in TYPES}
    resist_counts: Dict[str, int] = {t: 0 for t in TYPES}
    type_counts: Dict[str, int] = {}

    for label in labels:
        entry = get(label)
        for t in entry["types"]:
            type_counts[t] = type_counts.get(t, 0) + 1
        for atk, mult in matchups(entry["types"]).items():
            if mult > 1:
                weak_counts[atk] += 1
            elif mult < 1:
                resist_counts[atk] += 1

    # offensive coverage: types the team can hit super-effectively with STAB
    team_types = set(type_counts)
    covered = {d for atk in team_types for d in TYPES
               if effectiveness(atk, [d]) > 1}
    uncovered = sorted(set(TYPES) - covered) if team_types else []

    return {
        "type_counts": dict(sorted(type_counts.items(), key=lambda kv: -kv[1])),
        "weaknesses": {t: c for t, c in sorted(weak_counts.items(),
                                               key=lambda kv: -kv[1]) if c},
        "resistances": {t: c for t, c in sorted(resist_counts.items(),
                                                key=lambda kv: -kv[1]) if c},
        "uncovered_types": uncovered,
        "size": len(labels),
    }
