"""Question ranking + wh-templating (spec §5.3).

Two things happen here: (1) pick which unresolved slot is worth asking about, given what's
readable in *this* candidate and what a plausible distractor is likely to share; (2) turn that
slot into an oracle-facing wh-question. These templates go to the Oracle (Gemini), not to our
Qwen VLM — the Qwen prompts (extract.py / adjudicate.py) are a separate, still-open piece.
"""

from dataclasses import dataclass

from . import schema
from .priors import PriorsTable
from .state import ObservationFrame, TargetBelief

# --- wh-question templates ----------------------------------------------------------------------
# One fixed template per slot key. `{np}` is filled with the noun phrase from the description
# (never "it" / "the target" / "the object in your image" — spec §5.3 templating rules).
# All are open wh-form, under 20 words, and never reference the candidate image or its quality.

_TEMPLATES = {
    "obj.color_primary": "What is the primary color of {np}?",
    "obj.color_secondary": "Besides its main color, what secondary color does {np} have, if any?",
    "obj.material": "What material is {np} made of?",
    "obj.hardware_type": "What type of hardware, such as knobs or bar pulls, does {np} have?",
    "obj.hardware_finish": "What finish is the hardware on {np}?",
    "obj.count": "How many objects like {np} are visible?",
    "ctx.above.object": "What object sits directly above {np}?",
    "ctx.above.material": "What material is the surface directly above {np}?",
    "ctx.above.color": "What color is the surface directly above {np}?",
    "ctx.support.object": "What does {np} rest on or sit inside?",
    "room.type": "What type of room is {np} located in?",
    "room.floor_material": "What material is the floor in the room with {np}?",
    "room.floor_color": "What color is the floor in the room with {np}?",
    "room.wall_color": "What color are the walls in the room with {np}?",
    "room.window_present": "How many windows, if any, are visible in the room with {np}?",
    "room.notable_appliance": "What large appliances, if any, are visible in the room with {np}?",
}


def _adjacent_template(slot_key: str) -> str:
    if slot_key.endswith(".object"):
        return "What object is immediately next to {np}?"
    return "What color is the object immediately next to {np}?"


def template_for(slot_key: str) -> str:
    if slot_key in _TEMPLATES:
        return _TEMPLATES[slot_key]
    if slot_key.startswith("ctx.adjacent["):
        return _adjacent_template(slot_key)
    raise KeyError(f"No question template for slot '{slot_key}' (Tier-C slots are never queried)")


def render_question(slot_key: str, noun_phrase: str) -> str:
    return template_for(slot_key).format(np=noun_phrase)


def render_bundled_question(slot_keys: list[str], noun_phrase: str) -> str:
    """Join up to `max_bundle_slots` same-region templates into one question with 'and'.
    Caller (top()) is responsible for only bundling slots in the same region.
    """
    parts = [render_question(k, noun_phrase).rstrip("?") for k in slot_keys]
    # first part keeps its full wh-form; subsequent parts are stitched as a second clause
    return parts[0] + "? And " + "? And ".join(p[0].lower() + p[1:] for p in parts[1:]) + "?"


# --- candidate pool + scoring --------------------------------------------------------------------


@dataclass
class ScoredSlot:
    slot_key: str
    score: float
    region: str


def candidate_pool(
    frame: ObservationFrame,
    belief: TargetBelief,
    *,
    max_adjacent: int,
    tau_obs: float,
) -> list[str]:
    """§5.3: slots where ALL hold — Tier A/B (never C); belief unknown (never re-ask, the oracle
    is deterministic); frame resolved and confident (must be readable in *this* candidate); and
    not already implied by the description.
    """
    pool = []
    for slot_key in schema.all_slot_keys(max_adjacent):
        spec = schema.spec_for(slot_key)
        if spec.tier == schema.TIER_C:
            continue
        belief_value = belief.get(slot_key)
        if belief_value.certainty != "unknown" or belief_value.provenance == "description":
            continue  # already known (from description) or already asked/resolved
        if belief.has_asked(slot_key):
            continue
        frame_value = frame.get(slot_key)
        if frame_value.certainty != "resolved" or frame_value.confidence < tau_obs:
            continue  # can't read it in the candidate -> worthless even if oracle answers well
        pool.append(slot_key)
    return pool


def has_hedged_discriminative_slot(belief: TargetBelief, max_adjacent: int) -> bool:
    """True if some Tier-A/B, not-implied-by-description slot came back `hedged` rather than
    `resolved` or `unknown`. Such a slot is permanently excluded from `candidate_pool` (never
    re-asked — the oracle is deterministic) yet never decisive in `compare.py` either, so if it's
    the *only* thing keeping the pool non-empty, an empty pool doesn't mean "nothing left to
    check" — it means "the one thing left to check came back uncertain." Caller
    (questioner.py) uses this to route that case to adjudicate() instead of a blind conclude(True).
    """
    for slot_key in schema.all_slot_keys(max_adjacent):
        spec = schema.spec_for(slot_key)
        if spec.tier == schema.TIER_C:
            continue
        value = belief.get(slot_key)
        if value.provenance == "description":
            continue
        if value.certainty == "hedged":
            return True
    return False


def score_slot(
    slot_key: str,
    frame: ObservationFrame,
    category: str,
    priors: PriorsTable,
    *,
    tier_weight: dict[str, float],
    stability: dict[str, float],
) -> float:
    spec = schema.spec_for(slot_key)
    frame_value = frame.get(slot_key)
    disc = priors.disc(slot_key, frame_value.canon, category)
    tw = tier_weight.get(spec.tier, 0.5)
    # ctx.adjacent[i].* keys aren't in the stability table verbatim; fall back to the family key.
    stab_key = slot_key
    if slot_key.startswith("ctx.adjacent[") and slot_key not in stability:
        stab_key = "ctx.adjacent.object" if slot_key.endswith(".object") else "ctx.adjacent.color"
    stab = stability.get(stab_key, 0.7)
    return frame_value.confidence * disc * tw * stab


def rank(
    pool: list[str],
    frame: ObservationFrame,
    category: str,
    priors: PriorsTable,
    *,
    tier_weight: dict[str, float],
    stability: dict[str, float],
) -> list[ScoredSlot]:
    scored = [
        ScoredSlot(
            slot_key=k,
            score=score_slot(k, frame, category, priors, tier_weight=tier_weight, stability=stability),
            region=schema.spec_for(k).region,
        )
        for k in pool
    ]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


def front_load_order(scored: list[ScoredSlot], max_adjacent: int) -> list[ScoredSlot]:
    """§5.3: on candidate #1, prefer maximum region coverage — highest-scoring slot from each
    distinct region in turn, rather than a flat score sort.
    """
    by_region: dict[str, list[ScoredSlot]] = {}
    for s in scored:
        by_region.setdefault(s.region, []).append(s)
    for region_list in by_region.values():
        region_list.sort(key=lambda s: s.score, reverse=True)

    ordered_regions = [r for r in schema.regions(max_adjacent) if r in by_region]
    result = []
    round_idx = 0
    while any(by_region[r][round_idx:round_idx + 1] for r in ordered_regions):
        for r in ordered_regions:
            bucket = by_region[r]
            if round_idx < len(bucket):
                result.append(bucket[round_idx])
        round_idx += 1
    return result


def top(
    pool: list[str],
    frame: ObservationFrame,
    belief: TargetBelief,
    category: str,
    priors: PriorsTable,
    *,
    candidates_seen: int,
    max_adjacent: int,
    tier_weight: dict[str, float],
    stability: dict[str, float],
    allow_bundle: bool,
    max_bundle_slots: int,
) -> tuple[str, list[str]]:
    """Returns (question_text, slot_keys_covered). `candidates_seen == 0` triggers front-loading."""
    scored = rank(pool, frame, category, priors, tier_weight=tier_weight, stability=stability)
    if candidates_seen == 0:
        scored = front_load_order(scored, max_adjacent)

    if not scored:
        raise ValueError("top() called with an empty pool — caller must check candidate_pool() first")

    best = scored[0]
    slots_to_ask = [best.slot_key]

    if allow_bundle and max_bundle_slots >= 2:
        for other in scored[1:]:
            if len(slots_to_ask) >= max_bundle_slots:
                break
            if other.region == best.region:
                slots_to_ask.append(other.slot_key)

    noun_phrase = belief.noun_phrase
    if len(slots_to_ask) > 1:
        question = render_bundled_question(slots_to_ask, noun_phrase)
    else:
        question = render_question(slots_to_ask[0], noun_phrase)
    return question, slots_to_ask
