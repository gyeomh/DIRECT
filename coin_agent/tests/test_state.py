import pytest

from agent.state import MonotonicityViolation, SlotValue, TargetBelief


def test_monotonicity_blocks_overwrite_with_different_value():
    belief = TargetBelief(description="d")
    belief.set_slot("obj.color_primary", SlotValue(canon="white", confidence=1.0, certainty="resolved", provenance="description"))
    with pytest.raises(MonotonicityViolation):
        belief.set_slot("obj.color_primary", SlotValue(canon="navy", confidence=0.9, certainty="resolved", provenance="oracle"))


def test_monotonicity_allows_reasserting_the_same_value():
    belief = TargetBelief(description="d")
    belief.set_slot("obj.material", SlotValue(canon="oak", confidence=1.0, certainty="resolved", provenance="description"))
    belief.set_slot("obj.material", SlotValue(canon="oak", confidence=0.95, certainty="resolved", provenance="oracle"))
    assert belief.get("obj.material").canon == "oak"


def test_monotonicity_allows_filling_an_unknown_slot():
    belief = TargetBelief(description="d")
    belief.set_slot("obj.material", SlotValue(canon="oak", confidence=0.9, certainty="resolved", provenance="oracle"))
    assert belief.get("obj.material").canon == "oak"


def test_hedged_oracle_slots_are_also_protected_by_monotonicity():
    # Spec §4: "A slot filled from description or oracle is never overwritten" — this doesn't
    # carve out hedged answers, so a hedged oracle value is just as locked as a resolved one.
    # (select.candidate_pool never re-asks a hedged slot either way, since its certainty isn't
    # "unknown" — this is the defensive backstop, not the primary enforcement.)
    belief = TargetBelief(description="d")
    belief.set_slot("obj.material", SlotValue(canon="oak", confidence=0.5, certainty="hedged", provenance="oracle"))
    with pytest.raises(MonotonicityViolation):
        belief.set_slot("obj.material", SlotValue(canon="laminate", confidence=0.9, certainty="resolved", provenance="oracle"))


def test_render_text_never_repeats_description_verbatim():
    description = "White cabinet standing against a light green wall"
    belief = TargetBelief(description=description)
    belief.set_slot("obj.color_primary", SlotValue(canon="white", confidence=1.0, certainty="resolved", provenance="description"))
    rendered = belief.render_text()
    assert description not in rendered
