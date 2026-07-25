from agent.budget import MAX_TIME_S, BudgetController
from agent.state import SlotValue, TargetBelief

BASE_CONFIG = {
    "assumed_n_candidates": 8,
    "hard_cap_per_image": 6,
    "soft_time_frac": 0.60,
    "hard_time_frac": 0.85,
    "front_load": {"weights": {0: 2.0, 1: 1.0, "default": 0.8}},
}


def _clock(sequence):
    """Deterministic fake clock: yields the given values in order, then repeats the last one."""
    it = iter(sequence)
    last = [sequence[0]]

    def _fn():
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]

    return _fn


def _belief_with_no_known_tier_a_slots():
    # An "empty" belief: every Tier-A slot counts toward ambiguity_allowance, so the cap is
    # governed purely by the time/step formula, not clipped by ambiguity_allowance.
    return TargetBelief(description="d")


def test_reserves_one_step_per_remaining_candidate():
    config = {**BASE_CONFIG, "assumed_n_candidates": 3}
    bc = BudgetController(config, max_adjacent=3, _time_fn=_clock([0.0]))
    bc.per_candidate_cap = 10  # isolate the reserve check from the per-candidate-cap check
    bc.steps_used = 57  # 60 - 3 remaining reserved
    assert bc._reserve() == 3
    assert bc.may_ask() is False  # steps_used >= MAX_STEPS - reserve (57 >= 57)


def test_cap_drops_after_candidate_zero():
    # hard_cap_per_image raised well above what either candidate's formula produces, so the
    # front-load difference isn't masked by both candidates hitting the same ceiling.
    config = {**BASE_CONFIG, "hard_cap_per_image": 20}
    bc = BudgetController(config, max_adjacent=3, _time_fn=_clock([0.0]))
    belief = _belief_with_no_known_tier_a_slots()

    bc.start_candidate(belief)
    cap_candidate_0 = bc.per_candidate_cap

    bc.advance_candidate()
    bc.start_candidate(belief)
    cap_candidate_1 = bc.per_candidate_cap

    assert cap_candidate_0 > cap_candidate_1 > 0


def test_hard_cap_per_image_is_respected():
    config = {**BASE_CONFIG, "hard_cap_per_image": 2}
    bc = BudgetController(config, max_adjacent=3, _time_fn=_clock([0.0]))
    bc.start_candidate(_belief_with_no_known_tier_a_slots())
    assert bc.per_candidate_cap <= 2


def test_ambiguity_allowance_clips_cap():
    """A description that already resolves every Tier-A slot (e.g. 'category' type on a
    trivial object) should clip the cap to ~0 regardless of the time/step budget.
    """
    from agent import schema

    belief = TargetBelief(description="d")
    for slot_key in schema.all_slot_keys(3):
        if schema.spec_for(slot_key).tier == schema.TIER_A:
            belief.set_slot(slot_key, SlotValue(canon="x", confidence=1.0, certainty="resolved", provenance="description"))

    bc = BudgetController(BASE_CONFIG, max_adjacent=3, _time_fn=_clock([0.0]))
    bc.start_candidate(belief)
    assert bc.per_candidate_cap == 0


def test_soft_stop_fires_past_soft_time_frac():
    # First clock tick (0.0) is consumed by __init__'s start_time; start_candidate() doesn't read
    # the clock at all, so may_ask()'s `elapsed` property is the one that consumes the second tick.
    soft_deadline = BASE_CONFIG["soft_time_frac"] * MAX_TIME_S
    bc = BudgetController(BASE_CONFIG, max_adjacent=3, _time_fn=_clock([0.0, soft_deadline + 1]))
    bc.start_candidate(_belief_with_no_known_tier_a_slots())
    assert bc.may_ask() is False
    assert bc.soft_stop_fired is True


def test_hard_stop_fires_past_hard_time_frac():
    # First clock tick (0.0) is consumed by __init__'s start_time, so the *first* hard_stop()
    # call already consumes the second tick — needs three ticks, not two, to observe both a
    # False and a True reading.
    hard_deadline = BASE_CONFIG["hard_time_frac"] * MAX_TIME_S
    bc = BudgetController(BASE_CONFIG, max_adjacent=3, _time_fn=_clock([0.0, 0.0, hard_deadline + 1]))
    assert bc.hard_stop() is False  # elapsed = 0.0 - 0.0
    assert bc.hard_stop() is True  # elapsed = (hard_deadline+1) - 0.0
    assert bc.hard_stop_fired is True
