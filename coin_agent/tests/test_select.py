import pytest

from agent import schema, select
from agent.priors import PriorsTable
from agent.state import ObservationFrame, SlotValue, TargetBelief

BANNED_SUBSTRINGS = ["target", "image i", "artifact"]


def _resolved_frame(*slot_keys, confidence=0.9):
    frame = ObservationFrame(image_hash="h")
    for k in slot_keys:
        frame.slots[k] = SlotValue(raw="x", canon="x", confidence=confidence, certainty="resolved")
    return frame


def test_pool_excludes_tier_c():
    tier_c_keys = [k for k in schema.all_slot_keys(3) if schema.spec_for(k).tier == schema.TIER_C]
    assert tier_c_keys, "expected at least one Tier-C slot to exist"
    frame = _resolved_frame(*tier_c_keys)
    belief = TargetBelief(description="d")
    pool = select.candidate_pool(frame, belief, max_adjacent=3, tau_obs=0.8)
    assert not (set(pool) & set(tier_c_keys))


def test_pool_excludes_already_asked():
    frame = _resolved_frame("obj.material")
    belief = TargetBelief(description="d")
    belief.record_question("obj.material", "What material is the cabinet?")
    pool = select.candidate_pool(frame, belief, max_adjacent=3, tau_obs=0.8)
    assert "obj.material" not in pool


def test_pool_excludes_slot_implied_by_description():
    frame = _resolved_frame("obj.color_primary")
    belief = TargetBelief(description="d")
    belief.set_slot("obj.color_primary", SlotValue(canon="white", confidence=1.0, certainty="resolved", provenance="description"))
    pool = select.candidate_pool(frame, belief, max_adjacent=3, tau_obs=0.8)
    assert "obj.color_primary" not in pool


def test_pool_excludes_low_confidence_or_unresolved_frame_slots():
    frame = ObservationFrame(image_hash="h")
    frame.slots["obj.material"] = SlotValue(raw="x", canon="x", confidence=0.5, certainty="resolved")  # below tau_obs
    frame.slots["obj.hardware_finish"] = SlotValue(certainty="unknown")  # unresolved
    belief = TargetBelief(description="d")
    pool = select.candidate_pool(frame, belief, max_adjacent=3, tau_obs=0.8)
    assert "obj.material" not in pool
    assert "obj.hardware_finish" not in pool


_QUERYABLE_SLOTS = [
    k for k in schema.SLOTS
    if schema.SLOTS[k].tier != schema.TIER_C and k != "obj.category"  # obj.category is always known
    # upfront from info["category"] (env.py:135) and therefore always excluded by
    # candidate_pool()'s "not implied by description" filter — it has no template because it's
    # never actually queried in practice.
]


@pytest.mark.parametrize("slot_key", _QUERYABLE_SLOTS)
def test_templates_have_no_banned_substrings_and_are_questions(slot_key):
    q = select.render_question(slot_key, "the cabinet")
    low = q.lower()
    for banned in BANNED_SUBSTRINGS:
        assert banned not in low, f"{slot_key} template contains banned substring {banned!r}: {q!r}"
    assert q.strip().endswith("?"), f"{slot_key} template is not a question: {q!r}"
    assert len(q.split()) <= 20, f"{slot_key} template exceeds 20 words: {q!r}"


def test_top_never_returns_a_tier_c_slot():
    pool = ["obj.material"]  # Tier C slots are filtered out before top() is ever called
    frame = _resolved_frame("obj.material")
    belief = TargetBelief(description="d")
    priors = PriorsTable.empty()
    question, slots = select.top(
        pool, frame, belief, "cabinet", priors, candidates_seen=1, max_adjacent=3,
        tier_weight={"A": 1.0, "B": 0.6}, stability={"obj.material": 0.9},
        allow_bundle=True, max_bundle_slots=2,
    )
    assert all(schema.spec_for(s).tier != schema.TIER_C for s in slots)


def test_bundle_stays_within_one_region():
    # Two same-region ("obj") slots score higher than one different-region ("room") slot, so the
    # bundle should pick up the "obj" pair and explicitly exclude the "room" slot even though it's
    # in the pool — bundling never crosses regions (spec §5.3).
    frame = _resolved_frame("obj.color_primary", "obj.hardware_finish", "room.floor_material")
    belief = TargetBelief(description="d")
    priors = PriorsTable.empty()
    pool = ["obj.color_primary", "obj.hardware_finish", "room.floor_material"]
    _, slots = select.top(
        pool, frame, belief, "cabinet", priors, candidates_seen=1, max_adjacent=3,
        tier_weight={"A": 1.0, "B": 0.6},
        stability={"obj.color_primary": 0.75, "obj.hardware_finish": 0.85, "room.floor_material": 0.1},
        allow_bundle=True, max_bundle_slots=2,
    )
    regions = {schema.spec_for(s).region for s in slots}
    assert len(regions) == 1, f"bundled slots span multiple regions: {slots} -> {regions}"
    assert len(slots) == 2, f"expected the two same-region slots to bundle together, got: {slots}"
    assert "room.floor_material" not in slots
