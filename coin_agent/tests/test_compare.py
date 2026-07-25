from agent import compare, schema
from agent.state import ObservationFrame, SlotValue, TargetBelief


def _frame(**slots):
    f = ObservationFrame(image_hash="h")
    f.slots.update(slots)
    return f


def _belief(**slots):
    b = TargetBelief(description="d")
    b.slots.update(slots)
    return b


def test_tier_b_far_disagreement_never_decisive():
    # obj.color_secondary is Tier B
    frame = _frame(**{"obj.color_secondary": SlotValue(canon="navy", confidence=0.95, certainty="resolved")})
    belief = _belief(**{"obj.color_secondary": SlotValue(canon="white", confidence=1.0, certainty="resolved", provenance="description")})
    result = compare.compare(frame, belief, tau_obs=0.8)
    assert not result.decisive_conflict


def test_hedged_belief_never_decisive():
    # obj.color_primary is Tier A, and navy/white is FAR, but the belief slot is hedged
    frame = _frame(**{"obj.color_primary": SlotValue(canon="navy", confidence=0.95, certainty="resolved")})
    belief = _belief(**{"obj.color_primary": SlotValue(canon="white", confidence=1.0, certainty="hedged", provenance="oracle")})
    result = compare.compare(frame, belief, tau_obs=0.8)
    assert not result.decisive_conflict


def test_low_confidence_observation_never_decisive():
    frame = _frame(**{"obj.color_primary": SlotValue(canon="navy", confidence=0.5, certainty="resolved")})
    belief = _belief(**{"obj.color_primary": SlotValue(canon="white", confidence=1.0, certainty="resolved", provenance="description")})
    result = compare.compare(frame, belief, tau_obs=0.8)
    assert not result.decisive_conflict


def test_tier_a_far_high_confidence_resolved_is_decisive():
    frame = _frame(**{"obj.color_primary": SlotValue(canon="navy", confidence=0.95, certainty="resolved")})
    belief = _belief(**{"obj.color_primary": SlotValue(canon="white", confidence=1.0, certainty="resolved", provenance="description")})
    result = compare.compare(frame, belief, tau_obs=0.8)
    assert result.decisive_conflict
    assert result.decisive_slot == "obj.color_primary"


def test_weak_conflicts_for_decisive_off_by_default():
    # Three genuinely Tier-B weak conflicts (all COLOR/FAR pairs, all Tier B slots — ctx.above.color
    # is deliberately NOT used here since it's Tier A and would be independently decisive on its own).
    frame = _frame(**{
        "obj.color_secondary": SlotValue(canon="navy", confidence=0.9, certainty="resolved"),
        "room.floor_color": SlotValue(canon="navy", confidence=0.9, certainty="resolved"),
        "room.wall_color": SlotValue(canon="navy", confidence=0.9, certainty="resolved"),
    })
    belief = _belief(**{
        "obj.color_secondary": SlotValue(canon="white", confidence=1.0, certainty="resolved", provenance="description"),
        "room.floor_color": SlotValue(canon="white", confidence=1.0, certainty="resolved", provenance="description"),
        "room.wall_color": SlotValue(canon="white", confidence=1.0, certainty="resolved", provenance="description"),
    })
    for slot_key in frame.slots:
        assert schema.spec_for(slot_key).tier == schema.TIER_B
    result = compare.compare(frame, belief, tau_obs=0.8, weak_conflicts_for_decisive=None)
    assert not result.decisive_conflict


def test_weak_conflicts_for_decisive_when_enabled():
    frame = _frame(**{
        "obj.color_secondary": SlotValue(canon="navy", confidence=0.9, certainty="resolved"),
        "room.floor_color": SlotValue(canon="navy", confidence=0.9, certainty="resolved"),
        "room.wall_color": SlotValue(canon="navy", confidence=0.9, certainty="resolved"),
    })
    belief = _belief(**{
        "obj.color_secondary": SlotValue(canon="white", confidence=1.0, certainty="resolved", provenance="description"),
        "room.floor_color": SlotValue(canon="white", confidence=1.0, certainty="resolved", provenance="description"),
        "room.wall_color": SlotValue(canon="white", confidence=1.0, certainty="resolved", provenance="description"),
    })
    result = compare.compare(frame, belief, tau_obs=0.8, weak_conflicts_for_decisive=3)
    assert result.decisive_conflict
