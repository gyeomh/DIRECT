"""Integration tests (spec §9): a scripted-oracle fixture covering the three named scenarios —
decisive conflict on candidate #1; true match requiring 3 questions; hedged answer forcing
adjudication — plus the general "loop terminates, actions are valid, never both None" checks
that would otherwise use env.py's MockOracle.

`extract.extract` / `adjudicate.adjudicate` are monkeypatched rather than calling a real VLM:
their prompts (extract.EXTRACTION_PROMPT / adjudicate.ADJUDICATION_PROMPT) don't exist yet, but
everything *around* them — budget tracking, compare/select wiring, monotonic belief
accumulation, idempotency — is real and worth testing now.

Note: upstream `env.MockOracle.ask()` is itself missing `self` in its signature (a real bug,
`m.ask(prompt=..., images=...)` raises `TypeError` — try it) and independent of the three issues
in patches/README.md. We don't route through it here for exactly that reason.
"""

import numpy as np

from agent import adjudicate as adjudicate_mod
from agent import extract as extract_mod
from agent.llm import LLMCallFailed, LLMClient
from agent.questioner import GraphQuestioner
from agent.state import ObservationFrame, SlotValue

IMG = np.zeros((8, 8, 3), dtype=np.uint8)


def make_questioner(monkeypatch, description="White cabinet standing against a light green wall", category="Cabinet"):
    # No vllm server is running in tests, and DESCRIPTION_PARSE_PROMPT now makes a real call
    # inside __init__ (via parse_description) — fail it instantly instead of hitting a real
    # (slow, doomed) network connection. This exercises the same degrade path as a genuinely
    # unreachable server: belief keeps only obj.category.
    monkeypatch.setattr(LLMClient, "call", lambda *a, **k: (_ for _ in ()).throw(LLMCallFailed("no server in tests")))
    info = {"task_image": "SHOULD_BE_DISCARDED", "target_description": description, "category": category}
    q = GraphQuestioner(info)
    q.reset_time()
    return q


def _frame(image_hash="h", **slots):
    f = ObservationFrame(image_hash=image_hash)
    f.slots.update(slots)
    return f


def _assert_valid_action(action):
    assert (action["question"] is None) != (action["conclusion"] is None), (
        f"action must return exactly one of question/conclusion, got: {action}"
    )


def test_decisive_conflict_on_first_candidate(monkeypatch):
    # make_questioner() forces the description-parse LLM call to fail fast (no server in tests),
    # so parse_description() only seeds obj.category here, same as a real degrade. Seed
    # obj.color_primary by hand to simulate what a live DESCRIPTION_PARSE_PROMPT call would
    # produce, so there's something for the candidate frame to conflict with.
    q = make_questioner(monkeypatch)
    q.belief.set_slot(
        "obj.color_primary",
        SlotValue(raw="White cabinet", canon="white", confidence=1.0, certainty="resolved", provenance="description"),
    )
    conflicting_frame = _frame(**{
        "obj.color_primary": SlotValue(raw="navy", canon="navy", confidence=0.95, certainty="resolved"),
    })
    monkeypatch.setattr(extract_mod, "extract", lambda *a, **k: conflicting_frame)

    action = q.ask_or_conclude(dict(image=IMG, answer=None))
    _assert_valid_action(action)
    assert action["conclusion"] == 0
    assert "obj.color_primary" in action["reasoning"]


def test_true_match_requires_three_questions(monkeypatch):
    # Three slots in three DIFFERENT regions (obj / ctx.above / room) so config.yaml's default
    # allow_bundle=true can't legitimately collapse two of them into one question — same-region
    # bundling is correct behavior (select.py), but it isn't what this scenario is testing.
    q = make_questioner(monkeypatch)
    frame = _frame(**{
        "obj.material": SlotValue(raw="oak", canon="oak", confidence=0.9, certainty="resolved"),
        "ctx.above.material": SlotValue(raw="granite", canon="granite", confidence=0.9, certainty="resolved"),
        "room.type": SlotValue(raw="kitchen", canon="kitchen", confidence=0.9, certainty="resolved"),
    })
    monkeypatch.setattr(extract_mod, "extract", lambda *a, **k: frame)

    obs = dict(image=IMG, answer=None)
    n_asked = 0
    for _ in range(10):
        action = q.ask_or_conclude(obs)
        _assert_valid_action(action)
        if action["question"] is not None:
            n_asked += 1
            answered_slot = q._last_asked_slots[0]
            obs = dict(image=IMG, answer=f"It is {frame.get(answered_slot).canon}.")
        else:
            assert action["conclusion"] == 1
            break
    else:
        raise AssertionError("loop did not terminate within 10 steps")

    assert n_asked == 3


def test_hedged_answer_forces_adjudication(monkeypatch):
    q = make_questioner(monkeypatch)
    frame = _frame(**{
        "ctx.above.material": SlotValue(raw="granite", canon="granite", confidence=0.9, certainty="resolved"),
    })
    monkeypatch.setattr(extract_mod, "extract", lambda *a, **k: frame)
    monkeypatch.setattr(adjudicate_mod, "adjudicate", lambda *a, **k: (True, "adjudicator says match", None))

    action = q.ask_or_conclude(dict(image=IMG, answer=None))
    _assert_valid_action(action)
    assert action["question"] is not None  # asks about the only readable Tier-A slot

    action = q.ask_or_conclude(dict(image=IMG, answer="I'm not sure, maybe granite?"))
    _assert_valid_action(action)
    assert action["conclusion"] == 1
    assert action["reasoning"] == "adjudicator says match"


def test_never_returns_both_none_and_terminates(monkeypatch):
    monkeypatch.setattr(extract_mod, "extract", lambda *a, **k: ObservationFrame(image_hash="h"))
    monkeypatch.setattr(adjudicate_mod, "adjudicate", lambda *a, **k: (False, "no evidence", None))
    q = make_questioner(monkeypatch)
    action = q.ask_or_conclude(dict(image=IMG, answer=None))
    _assert_valid_action(action)


def test_extraction_failure_routes_to_adjudicate_not_blind_true(monkeypatch):
    """Regression test for the bug caught while wiring questioner.py: an extraction failure
    degrading to an all-unknown frame must NOT be indistinguishable from "nothing left to check".
    """
    def _raise(*a, **k):
        raise extract_mod.ExtractionParseError("boom")

    monkeypatch.setattr(extract_mod, "extract", _raise)
    monkeypatch.setattr(adjudicate_mod, "adjudicate", lambda *a, **k: (False, "degraded: extraction failed", None))
    q = make_questioner(monkeypatch)
    action = q.ask_or_conclude(dict(image=IMG, answer=None))
    _assert_valid_action(action)
    assert action["conclusion"] == 0
    assert action["reasoning"] == "degraded: extraction failed"
