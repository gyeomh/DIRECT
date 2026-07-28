import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from llm import LLMClient
from self_check import (
    SELF_CHECK_PROMPT,
    SELF_CHECK_SCHEMA,
    SelfCheckResult,
    build_prompt,
    is_failure,
    self_check,
)

IMG = np.zeros((8, 8, 3), dtype=np.uint8)

# region_for lives in templates.py (shared with the question path -- tests/test_templates.py has
# full per-key grammar coverage). self_check.py itself takes an already-assembled region string.


# --- prompt construction: verbatim prefix + variable suffix ------------------------------------


def test_build_prompt_contains_verbatim_system_prompt_unchanged():
    prompt, schema = build_prompt("left of the cabinet", "wooden tiles on a black wall")
    assert prompt.startswith(SELF_CHECK_PROMPT)
    assert schema == SELF_CHECK_SCHEMA


def test_build_prompt_embeds_region_and_assertion_as_separate_fields():
    prompt, _ = build_prompt("left of the cabinet", "wooden tiles on a black wall")
    assert "region: left of the cabinet" in prompt
    assert "assertion: wooden tiles on a black wall" in prompt


def test_build_prompt_is_byte_identical_prefix_across_different_assertions():
    """SPEC.md §7 item 5: the fixed system prompt must be a byte-identical prefix across every
    call, regardless of image or assertion, so prefix caching covers the rule text too.
    """
    prompt_a, _ = build_prompt("left of the cabinet", "assertion one")
    prompt_b, _ = build_prompt("above of the desk", "a completely different assertion")
    prefix_len = len(SELF_CHECK_PROMPT)
    assert prompt_a[:prefix_len] == prompt_b[:prefix_len] == SELF_CHECK_PROMPT


def test_schema_field_order_is_evidence_then_verdict():
    # dict insertion order matters here: it's what the guided-decoding backend is expected to
    # honor as generation order (SPEC.md §7 item 4 -- evidence acts as a short CoT before verdict).
    assert list(SELF_CHECK_SCHEMA["properties"].keys()) == ["evidence", "verdict"]


# --- self_check(): parsing, failure classification ---------------------------------------------


def test_is_failure_no_is_a_failure():
    assert is_failure("no") is True


def test_is_failure_yes_is_not_a_failure():
    assert is_failure("yes") is False


def test_is_failure_parse_error_counts_as_failure():
    assert is_failure("PARSE_ERROR") is True


def test_self_check_parses_evidence_and_verdict(tmp_path, monkeypatch):
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    monkeypatch.setattr(
        client._backend, "generate",
        lambda *a, **k: '{"evidence": "cabinet is clearly white, not navy", "verdict": "no"}',
    )
    result = self_check(client, IMG, "the cabinet itself", "it is navy blue")
    assert isinstance(result, SelfCheckResult)
    assert result.verdict == "no"
    assert result.evidence == "cabinet is clearly white, not navy"


def test_self_check_returns_parse_error_on_malformed_json(tmp_path, monkeypatch):
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: "not json at all")
    result = self_check(client, IMG, "region", "assertion")
    assert result.verdict == "PARSE_ERROR"


def test_self_check_returns_parse_error_on_missing_verdict_key(tmp_path, monkeypatch):
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: '{"evidence": "something"}')
    result = self_check(client, IMG, "region", "assertion")
    assert result.verdict == "PARSE_ERROR"


def test_self_check_with_fake_backend_schema_filler(tmp_path):
    # FakeVLMBackend's generic schema filler picks each property's first plausible value:
    # "evidence" (a plain string, no enum) falls back to the "fake" placeholder, "verdict"
    # (an enum) takes its first listed value, "yes".
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    result = self_check(client, IMG, "region", "assertion")
    assert result.verdict == "yes"
