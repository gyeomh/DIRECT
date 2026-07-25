"""Slot ontology, canonical vocabularies, and tiers (spec §3).

A fixed typed frame, not an open-ended scene graph — the slot set below is closed and depth-capped
at 2 hops, matching what the benchmark's descriptions actually contain. `canon.py` owns
normalization and confusability over the vocabularies defined here; this module only owns *what
slots exist* and *what values are legal*.
"""

from dataclasses import dataclass

TIER_A = "A"  # decisive-eligible: a conflict here can terminate deliberation with False
TIER_B = "B"  # evidence only: accumulates toward the adjudicator, never decisive alone
TIER_C = "C"  # never decisive, never queried: transient/movable, generates false conflicts


@dataclass(frozen=True)
class SlotSpec:
    key: str
    type: str          # one of the *_VALUES vocab names below, or "str"/"int"/"bool"
    tier: str           # TIER_A / TIER_B / TIER_C
    region: str         # grouping used for front-load coverage and question bundling (§5.3)
    notes: str = ""


# --- canonical vocabularies -------------------------------------------------------------------
# These are the *closed* value sets for enum-typed slots. `canon.py` maps raw VLM/oracle text onto
# these via synonym tables, and defines SAME/NEAR/FAR relations over them. Lists are a first-pass
# derived from the spec's own examples plus the obvious siblings; expand via build_priors.py runs
# over real extractions rather than guessing further.

COLOR_VALUES = [
    "white", "off_white", "cream", "beige", "tan",
    "grey", "light_grey", "dark_grey", "black",
    "brown", "red", "orange", "yellow",
    "green", "olive", "blue", "navy", "dark_blue", "light_blue", "teal",
    "purple", "pink", "gold", "silver", "multicolor",
]

MATERIAL_VALUES = [
    "painted", "bare_wood", "oak", "butcher_block", "laminate",
    "metal", "glass", "stone", "granite", "marble", "tile",
    "fabric", "leather", "concrete", "carpet", "vinyl",
]

HARDWARE_VALUES = ["knob", "bar_pull", "recessed", "none"]

FINISH_VALUES = ["brass", "chrome", "black", "nickel", "wood", "stainless", "bronze", "gold", "none"]

STATE_VALUES = ["open", "closed", "on", "off"]

ROOM_VALUES = [
    "kitchen", "bedroom", "bathroom", "living_room", "office",
    "dining_room", "hallway", "laundry_room", "garage", "outdoor",
]

VOCAB = {
    "COLOR": COLOR_VALUES,
    "MATERIAL": MATERIAL_VALUES,
    "HARDWARE": HARDWARE_VALUES,
    "FINISH": FINISH_VALUES,
    "STATE": STATE_VALUES,
    "ROOM": ROOM_VALUES,
}


# --- slot table (spec §3.1 / §3.2) ------------------------------------------------------------
# `region` groups slots for front-load coverage (§5.3: "pick the highest-scoring slot from each
# distinct region in turn") and for bundling (§5.3: "never bundle across regions").

SLOTS: dict[str, SlotSpec] = {
    "obj.category": SlotSpec("obj.category", "str", TIER_A, "obj", "always known from the description"),
    "obj.color_primary": SlotSpec("obj.color_primary", "COLOR", TIER_A, "obj", "dominant color of the target object"),
    "obj.color_secondary": SlotSpec("obj.color_secondary", "COLOR", TIER_B, "obj"),
    "obj.material": SlotSpec("obj.material", "MATERIAL", TIER_A, "obj"),
    "obj.hardware_type": SlotSpec("obj.hardware_type", "HARDWARE", TIER_B, "obj"),
    "obj.hardware_finish": SlotSpec("obj.hardware_finish", "FINISH", TIER_A, "obj"),
    "obj.state": SlotSpec("obj.state", "STATE", TIER_C, "obj", "transient"),
    "obj.style": SlotSpec("obj.style", "str", TIER_C, "obj", "free text, e.g. shaker, flat-panel"),
    "obj.count": SlotSpec("obj.count", "int", TIER_B, "obj", "how many of this object type are visible"),
    "ctx.above.object": SlotSpec("ctx.above.object", "str", TIER_A, "ctx.above"),
    "ctx.above.material": SlotSpec("ctx.above.material", "MATERIAL", TIER_A, "ctx.above", "strongest single discriminator in kitchen scenes"),
    "ctx.above.color": SlotSpec("ctx.above.color", "COLOR", TIER_A, "ctx.above"),
    "ctx.support.object": SlotSpec("ctx.support.object", "str", TIER_A, "ctx.support", "what the object rests on / is set into"),
    # ctx.adjacent[i].* and ctx.contains[i] are expanded at runtime up to config.schema.max_adjacent —
    # see `adjacent_slot_keys()` / `contains_slot_keys()` below rather than being listed individually.
    "room.type": SlotSpec("room.type", "ROOM", TIER_A, "room"),
    "room.floor_material": SlotSpec("room.floor_material", "MATERIAL", TIER_A, "room", "very stable across viewpoints"),
    "room.floor_color": SlotSpec("room.floor_color", "COLOR", TIER_B, "room", "lighting-sensitive"),
    "room.wall_color": SlotSpec("room.wall_color", "COLOR", TIER_B, "room", "lighting-sensitive"),
    "room.window_present": SlotSpec("room.window_present", "bool", TIER_B, "room"),
    "room.notable_appliance": SlotSpec("room.notable_appliance", "str", TIER_B, "room", "e.g. range, dishwasher"),
}

# Template specs for the indexed families (§3.1). `key` uses `{i}` as a 0-based placeholder.
_ADJACENT_TEMPLATE = {
    "object": SlotSpec("ctx.adjacent[{i}].object", "str", TIER_B, "ctx.adjacent"),
    "color": SlotSpec("ctx.adjacent[{i}].color", "COLOR", TIER_B, "ctx.adjacent"),
}
_CONTAINS_TEMPLATE = SlotSpec("ctx.contains[{i}]", "str", TIER_C, "ctx.contains", "movable contents")


def adjacent_slot_keys(max_adjacent: int) -> list[str]:
    return [
        _ADJACENT_TEMPLATE[field].key.format(i=i)
        for i in range(max_adjacent)
        for field in ("object", "color")
    ]


def contains_slot_keys(max_adjacent: int) -> list[str]:
    # contains list is sized the same as adjacent by convention; spec doesn't give it a separate cap.
    return [_CONTAINS_TEMPLATE.key.format(i=i) for i in range(max_adjacent)]


def spec_for(slot_key: str) -> SlotSpec:
    """Resolve a slot key, including indexed ctx.adjacent[i]/ctx.contains[i] instances, to its SlotSpec."""
    if slot_key in SLOTS:
        return SLOTS[slot_key]
    if slot_key.startswith("ctx.adjacent[") and slot_key.endswith(".object"):
        return _ADJACENT_TEMPLATE["object"]
    if slot_key.startswith("ctx.adjacent[") and slot_key.endswith(".color"):
        return _ADJACENT_TEMPLATE["color"]
    if slot_key.startswith("ctx.contains["):
        return _CONTAINS_TEMPLATE
    raise KeyError(f"Unknown slot key: {slot_key}")


def all_slot_keys(max_adjacent: int) -> list[str]:
    return list(SLOTS.keys()) + adjacent_slot_keys(max_adjacent) + contains_slot_keys(max_adjacent)


def queryable_slot_keys(max_adjacent: int) -> list[str]:
    """Slots actually worth asking a VLM about — everything `all_slot_keys` has, minus:
    - Tier C (obj.style, obj.state, ctx.contains[i]): never decisive (compare.py), never queried
      (select.py), never passed to the adjudicator (adjudicate.py) — zero consumers anywhere.
    - obj.category: every candidate in an episode is guaranteed to share the target's category
      (verified empirically — 0 mismatches across all 528 training distractors, filenames all
      share the category prefix), and it's already known with full confidence from
      info["category"] (parse.py). Re-asking a VLM to re-derive it per candidate is wasted
      output tokens, and risks a spurious FAR conflict from pure phrasing (e.g. "wardrobe" vs
      "closet") on a slot that's actually guaranteed to match.

    Used by both extract.py's response_schema() (per candidate image) and parse.py's
    DESCRIPTION_PARSE_PROMPT schema (once, from the description text) so the two prompts request
    exactly the same vocabulary.
    """
    return [
        k for k in all_slot_keys(max_adjacent)
        if spec_for(k).tier != TIER_C and k != "obj.category"
    ]


def regions(max_adjacent: int) -> list[str]:
    """Distinct regions, in a stable order, for front-load coverage (§5.3)."""
    seen = []
    for key in all_slot_keys(max_adjacent):
        r = spec_for(key).region
        if r not in seen:
            seen.append(r)
    return seen
