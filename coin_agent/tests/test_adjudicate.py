import numpy as np

from agent.adjudicate import (
    AdjudicationParseError,
    _build_adjudication_prompt,
    adjudicate,
    parse_response,
    qa_history_text,
    score_to_conclusion,
)
from agent.llm import LLMCallFailed, LLMClient, LLMResult
from agent.state import SlotValue, TargetBelief

IMG = np.zeros((8, 8, 3), dtype=np.uint8)


def test_parse_response_extracts_motivation_and_score():
    text = "<motivation>Colors and material match.</motivation><score>2</score>"
    motivation, score = parse_response(text)
    assert motivation == "Colors and material match."
    assert score == 2


def test_parse_response_handles_surrounding_noise():
    text = "Sure!\n<motivation>Looks the same</motivation>\n<score>1</score>\nDone."
    motivation, score = parse_response(text)
    assert motivation == "Looks the same"
    assert score == 1


def test_parse_response_malformed_raises():
    try:
        parse_response("no tags here at all")
        raise AssertionError("expected AdjudicationParseError")
    except AdjudicationParseError:
        pass


def test_score_to_conclusion_mapping():
    assert score_to_conclusion(2) is True
    assert score_to_conclusion(0) is False
    assert score_to_conclusion(1) is False  # residual "unsure" defaults to no-match, not a coin flip


def test_qa_history_empty():
    belief = TargetBelief(description="d")
    assert qa_history_text(belief) == "(no questions asked yet)"


def test_qa_history_single_question():
    belief = TargetBelief(description="d")
    belief.set_slot("obj.material", SlotValue(raw="It's oak.", canon="oak", confidence=0.9, certainty="resolved", provenance="oracle"))
    belief.record_question("obj.material", "What material is the cabinet?")
    text = qa_history_text(belief)
    assert text == "Q: What material is the cabinet?\nA: It's oak."


def test_qa_history_dedupes_bundled_question():
    """Regression test: select.py bundles up to 2 same-region slots into one question, and
    questioner.py records that question once per bundled slot in belief.asked — same question
    text, two slot_keys. Without dedup this printed the identical Q&A pair twice.
    """
    belief = TargetBelief(description="d")
    bundled_q = "What is the primary color of the cabinet? And what finish is its hardware?"
    belief.set_slot("obj.color_primary", SlotValue(raw="white, brass", canon="white", confidence=0.9, certainty="resolved", provenance="oracle"))
    belief.set_slot("obj.hardware_finish", SlotValue(raw="white, brass", canon="brass", confidence=0.9, certainty="resolved", provenance="oracle"))
    belief.record_question("obj.color_primary", bundled_q)
    belief.record_question("obj.hardware_finish", bundled_q)

    text = qa_history_text(belief)
    assert text.count("Q: " + bundled_q) == 1


def test_build_adjudication_prompt_includes_all_context():
    belief = TargetBelief(description="White cabinet with plush toys")
    belief.set_slot("obj.color_primary", SlotValue(raw="white", canon="white", confidence=1.0, certainty="resolved", provenance="description"))
    belief.record_question("room.type", "What room is the cabinet in?")
    belief.set_slot("room.type", SlotValue(raw="kitchen", canon="kitchen", confidence=0.9, certainty="resolved", provenance="oracle", source_question="What room is the cabinet in?"))

    prompt = _build_adjudication_prompt(belief)
    assert "White cabinet with plush toys" in prompt
    assert "obj.color_primary: white" in prompt
    assert "What room is the cabinet in?" in prompt
    assert "<motivation>" in prompt and "<score>" in prompt


def test_adjudicate_end_to_end(monkeypatch):
    scripted_text = "<motivation>Same white cabinet, same kitchen.</motivation><score>2</score>"
    monkeypatch.setattr(
        LLMClient, "call",
        lambda *a, **k: LLMResult(text=scripted_text, logprobs=None, latency_s=0.0, cached=False),
    )
    client = LLMClient("fake-model")
    belief = TargetBelief(description="White cabinet")
    conclusion, motivation, fallback = adjudicate(IMG, belief, client)
    assert conclusion is True
    assert motivation == "Same white cabinet, same kitchen."
    assert fallback is None


def test_adjudicate_retries_once_on_parse_failure(monkeypatch):
    responses = iter([
        LLMResult(text="garbage, no tags", logprobs=None, latency_s=0.0, cached=False),
        LLMResult(text="<motivation>ok now</motivation><score>0</score>", logprobs=None, latency_s=0.0, cached=False),
    ])
    monkeypatch.setattr(LLMClient, "call", lambda *a, **k: next(responses))
    client = LLMClient("fake-model")
    belief = TargetBelief(description="White cabinet")
    conclusion, motivation, fallback = adjudicate(IMG, belief, client)
    assert conclusion is False
    assert motivation == "ok now"
    assert fallback == "retry_parse"


def test_adjudicate_double_parse_failure_defaults_to_false(monkeypatch):
    monkeypatch.setattr(
        LLMClient, "call",
        lambda *a, **k: LLMResult(text="garbage", logprobs=None, latency_s=0.0, cached=False),
    )
    client = LLMClient("fake-model")
    belief = TargetBelief(description="White cabinet")
    conclusion, motivation, fallback = adjudicate(IMG, belief, client)
    assert conclusion is False
    assert fallback == "double_parse_failure"


def test_adjudicate_skip_call_never_touches_llm(monkeypatch):
    def _fail_if_called(*a, **k):
        raise AssertionError("llm_client.call should never be invoked when skip_call=True")

    monkeypatch.setattr(LLMClient, "call", _fail_if_called)
    client = LLMClient("fake-model")
    belief = TargetBelief(description="White cabinet")
    conclusion, motivation, fallback = adjudicate(IMG, belief, client, skip_call=True)
    assert conclusion is False
    assert fallback == "hard_time_budget"


def test_adjudicate_degrades_when_both_attempts_fail_to_call(monkeypatch):
    def _fail(*a, **k):
        raise LLMCallFailed("no server")

    monkeypatch.setattr(LLMClient, "call", _fail)
    client = LLMClient("fake-model")
    belief = TargetBelief(description="White cabinet")
    conclusion, motivation, fallback = adjudicate(IMG, belief, client)
    assert conclusion is False
    assert fallback == "double_parse_failure"
