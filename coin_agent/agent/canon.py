"""Value normalization + confusability classes (spec §3.3).

This table is what prevents the single largest source of false conflicts: a `FAR` pair may
produce a decisive CONFLICT; a `NEAR` pair only ever produces a WEAK_CONFLICT (never decisive
alone). Relations are stored explicitly here — never inferred from string distance (spec is
explicit on this point).

Coverage is a first pass over the spec's own examples plus obvious siblings, not exhaustive.
Unlisted pairs default to NEAR (see `relation()`), which is the conservative choice: an
incomplete FAR table would fabricate decisive conflicts (wrong `False` conclusions), whereas an
incomplete NEAR default only costs a wasted question. Expand via `scripts/build_priors.py`
findings, not guesswork.
"""

import re

SAME = "SAME"
NEAR = "NEAR"
FAR = "FAR"

# --- synonym maps: raw/free text -> canonical vocab value ---------------------------------------
# Keys are lowercased, whitespace-collapsed before lookup (see `normalize`).

_COLOR_SYNONYMS = {
    "navy": "navy", "navy blue": "navy", "dark navy": "navy",
    "dark blue": "dark_blue", "midnight blue": "dark_blue",
    "blue": "blue", "light blue": "light_blue", "sky blue": "light_blue", "baby blue": "light_blue",
    "white": "white", "bright white": "white",
    "off white": "off_white", "off-white": "off_white", "eggshell": "off_white", "ivory": "off_white",
    "cream": "cream", "cream colored": "cream",
    "beige": "beige", "tan": "tan", "sand": "tan",
    "grey": "grey", "gray": "grey",
    "light grey": "light_grey", "light gray": "light_grey",
    "dark grey": "dark_grey", "dark gray": "dark_grey", "charcoal": "dark_grey",
    "black": "black",
    "brown": "brown", "dark brown": "brown", "chocolate": "brown",
    "red": "red", "maroon": "red", "burgundy": "red",
    "orange": "orange", "burnt orange": "orange",
    "yellow": "yellow", "mustard": "yellow",
    "green": "green", "sage green": "green", "mint": "green", "emerald": "green",
    "olive": "olive", "olive green": "olive",
    "teal": "teal", "turquoise": "teal",
    "purple": "purple", "lavender": "purple", "violet": "purple",
    "pink": "pink", "rose": "pink", "blush": "pink",
    "gold": "gold", "golden": "gold",
    "silver": "silver", "metallic silver": "silver",
    "multicolor": "multicolor", "multi-colored": "multicolor", "multicolored": "multicolor",
    "multi-color": "multicolor", "multicoloured": "multicolor", "patterned": "multicolor",
}

# Live-model regression test (Qwen3-VL-30B-A3B-Instruct via vllm) surfaced this: the model
# correctly said "light green" / "dark red" / "pale yellow" for real dataset descriptions, but
# _COLOR_SYNONYMS only enumerates a handful of specific "modifier + color" phrases (e.g. "light
# blue", "dark grey") — every other modifier+color combination silently failed to normalize,
# discarding a genuinely correct extraction. Enumerating every combination is unbounded; instead,
# strip a leading modifier and retry against the base color once, in both normalize() and
# find_in_text().
_COLOR_MODIFIERS = ("light", "dark", "pale", "deep", "bright", "muted", "soft", "vivid", "dull")

_MATERIAL_SYNONYMS = {
    "painted": "painted", "painted wood": "painted",
    "bare wood": "bare_wood", "raw wood": "bare_wood", "unfinished wood": "bare_wood",
    "oak": "oak", "oak wood": "oak",
    "butcher block": "butcher_block", "butcherblock": "butcher_block",
    "laminate": "laminate", "formica": "laminate",
    "metal": "metal", "stainless steel": "metal", "steel": "metal", "aluminum": "metal",
    "glass": "glass", "tempered glass": "glass",
    "stone": "stone",
    "granite": "granite",
    "marble": "marble",
    "tile": "tile", "ceramic tile": "tile", "porcelain tile": "tile",
    "fabric": "fabric", "upholstered": "fabric", "cloth": "fabric",
    "leather": "leather", "faux leather": "leather",
    "concrete": "concrete",
    "carpet": "carpet", "carpeted": "carpet",
    "vinyl": "vinyl",
}

_HARDWARE_SYNONYMS = {
    "knob": "knob", "knobs": "knob", "round knob": "knob",
    "bar pull": "bar_pull", "bar handle": "bar_pull", "pull": "bar_pull", "pulls": "bar_pull",
    "recessed": "recessed", "finger pull": "recessed", "no handle": "recessed",
    "none": "none", "no hardware": "none",
}

_FINISH_SYNONYMS = {
    "brass": "brass", "antique brass": "brass", "gold brass": "brass",
    "chrome": "chrome", "polished chrome": "chrome",
    "black": "black", "matte black": "black", "black finish": "black",
    "nickel": "nickel", "brushed nickel": "nickel", "satin nickel": "nickel",
    "wood": "wood", "wooden": "wood",
    "stainless": "stainless", "stainless steel": "stainless",
    "bronze": "bronze", "oil rubbed bronze": "bronze",
    "gold": "gold",
    "none": "none", "no hardware": "none",
}

_STATE_SYNONYMS = {
    "open": "open", "opened": "open", "ajar": "open",
    "closed": "closed", "shut": "closed",
    "on": "on", "turned on": "on", "lit": "on",
    "off": "off", "turned off": "off",
}

_ROOM_SYNONYMS = {
    "kitchen": "kitchen",
    "bedroom": "bedroom",
    "bathroom": "bathroom", "washroom": "bathroom",
    "living room": "living_room", "lounge": "living_room", "family room": "living_room",
    "office": "office", "home office": "office", "study": "office",
    "dining room": "dining_room",
    "hallway": "hallway", "corridor": "hallway", "entryway": "hallway",
    "laundry room": "laundry_room", "utility room": "laundry_room",
    "garage": "garage",
    "outdoor": "outdoor", "outside": "outdoor", "patio": "outdoor", "yard": "outdoor",
}

_SYNONYMS_BY_TYPE = {
    "COLOR": _COLOR_SYNONYMS,
    "MATERIAL": _MATERIAL_SYNONYMS,
    "HARDWARE": _HARDWARE_SYNONYMS,
    "FINISH": _FINISH_SYNONYMS,
    "STATE": _STATE_SYNONYMS,
    "ROOM": _ROOM_SYNONYMS,
}


def _key(text: str) -> str:
    return " ".join(text.strip().lower().split())


def normalize(raw: str | None, value_type: str) -> str | None:
    """Map raw free text onto a canonical vocab value for `value_type`, or None if unrecognized.
    Non-enum types ("str", "int", "bool") pass through unchanged (lightly cleaned for str).

    Exact-match only — use this when `raw` is already an isolated value (e.g. extract.py's VLM
    JSON `value` field, which the prompt asks to fill with just the value). For prose answers
    (oracle responses are full sentences, not bare values), use `find_in_text` instead.
    """
    if raw is None:
        return None
    if value_type not in _SYNONYMS_BY_TYPE:
        return _key(raw) if value_type == "str" else raw
    key = _key(raw)
    hit = _SYNONYMS_BY_TYPE[value_type].get(key)
    if hit is not None:
        return hit
    if value_type == "COLOR":
        for modifier in _COLOR_MODIFIERS:
            if key.startswith(modifier + " "):
                return _COLOR_SYNONYMS.get(key[len(modifier) + 1:])
    return None


def find_in_text(text: str | None, value_type: str) -> str | None:
    """Scan a prose answer for any known synonym phrase, longest-phrase-first so e.g. "dark blue"
    wins over the substring "blue". Returns the canonical value of the first (longest) match, or
    None if nothing recognizable is present. Word-boundary aware so "tan" doesn't match inside
    "stand".
    """
    if text is None:
        return None
    if value_type not in _SYNONYMS_BY_TYPE:
        return None
    hay = _key(text)
    synonyms = _SYNONYMS_BY_TYPE[value_type]
    for phrase in sorted(synonyms.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", hay):
            return synonyms[phrase]
    if value_type == "COLOR":
        for modifier in _COLOR_MODIFIERS:
            m = re.search(rf"\b{modifier}\s+(\w+)\b", hay)
            if m and m.group(1) in _COLOR_SYNONYMS:
                return _COLOR_SYNONYMS[m.group(1)]
    return None


# --- confusability relations --------------------------------------------------------------------
# Undirected pairs. Only pairs explicitly worth calling out are listed; `relation()` handles
# SAME (identical canon) and the NEAR default itself.

_NEAR_CLUSTERS = {
    "COLOR": [
        {"navy", "dark_blue", "blue"},
        {"white", "off_white", "cream"},
        {"grey", "light_grey", "white"},
        {"grey", "dark_grey", "black"},
        {"beige", "tan", "cream"},
        {"green", "olive"},
        {"teal", "blue"},
        {"gold", "yellow"},
        {"silver", "grey", "light_grey"},
    ],
    "MATERIAL": [
        {"bare_wood", "oak", "butcher_block"},
        {"stone", "granite", "marble"},
        {"laminate", "tile"},
        {"vinyl", "laminate"},
    ],
    "FINISH": [
        {"nickel", "chrome", "stainless"},
        {"brass", "gold"},
        {"black", "bronze"},
    ],
    "HARDWARE": [],
    "STATE": [],
    "ROOM": [
        {"living_room", "dining_room"},
        {"hallway", "laundry_room"},
    ],
}

_FAR_PAIRS = {
    "COLOR": [
        ("navy", "white"), ("navy", "cream"), ("navy", "off_white"),
        ("black", "white"), ("black", "off_white"), ("black", "cream"),
        ("red", "green"), ("red", "blue"), ("red", "navy"),
        ("white", "brown"), ("white", "black"),
        ("yellow", "purple"), ("orange", "blue"),
    ],
    "MATERIAL": [
        ("painted", "bare_wood"), ("painted", "oak"), ("painted", "butcher_block"),
        ("metal", "fabric"), ("metal", "bare_wood"),
        ("glass", "fabric"), ("glass", "carpet"),
        ("carpet", "tile"), ("carpet", "stone"), ("carpet", "marble"),
        ("concrete", "carpet"),
    ],
    "FINISH": [
        ("brass", "chrome"), ("brass", "nickel"), ("brass", "black"), ("brass", "stainless"),
        ("chrome", "black"), ("wood", "chrome"), ("wood", "brass"),
        ("gold", "chrome"), ("gold", "nickel"),
    ],
    "HARDWARE": [
        ("knob", "bar_pull"), ("knob", "none"), ("bar_pull", "none"), ("knob", "recessed"),
    ],
    "STATE": [("open", "closed"), ("on", "off")],
    "ROOM": [
        ("kitchen", "bedroom"), ("kitchen", "bathroom"), ("bathroom", "bedroom"),
        ("bathroom", "living_room"), ("garage", "bedroom"), ("outdoor", "bathroom"),
    ],
}


def _pair_in(pairs, a, b) -> bool:
    return (a, b) in pairs or (b, a) in pairs


def relation(value_type: str, a: str | None, b: str | None) -> str:
    """SAME / NEAR / FAR between two already-canonicalized values of the same type.
    Only FAR may produce a decisive conflict (compare.py). Unlisted-but-differing pairs
    default to NEAR — see module docstring for why.
    """
    if a is None or b is None:
        raise ValueError("relation() requires two canonicalized (non-None) values")
    if a == b:
        return SAME
    far = _FAR_PAIRS.get(value_type, [])
    if _pair_in(far, a, b):
        return FAR
    clusters = _NEAR_CLUSTERS.get(value_type, [])
    if any(a in c and b in c for c in clusters):
        return NEAR
    return NEAR
