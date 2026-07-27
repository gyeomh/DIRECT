import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from llm import LLMClient
from self_check import (
    CONTRADICTS_SCHEMA,
    POLARITY_CONTRADICTS,
    POLARITY_YES_NO,
    YES_NO_SCHEMA,
    build_prompt,
    is_failure,
    self_check,
)

IMG = np.zeros((8, 8, 3), dtype=np.uint8)


def test_build_prompt_yes_no_embeds_statement():
    prompt, schema = build_prompt(POLARITY_YES_NO, "The cabinet is white.")
    assert "The cabinet is white." in prompt
    assert schema == YES_NO_SCHEMA


def test_build_prompt_contradicts_embeds_statement():
    prompt, schema = build_prompt(POLARITY_CONTRADICTS, "The cabinet is white.")
    assert "The cabinet is white." in prompt
    assert schema == CONTRADICTS_SCHEMA


def test_build_prompt_unknown_polarity_raises():
    try:
        build_prompt("nonsense", "statement")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_is_failure_yes_no():
    assert is_failure(POLARITY_YES_NO, "no") is True
    assert is_failure(POLARITY_YES_NO, "yes") is False


def test_is_failure_contradicts_cant_tell_is_not_a_failure():
    """SPEC.md §7: cant_tell counts as neither pass nor fail -- for this experiment's
    false-failure-rate metric, that means NOT a failure (it wouldn't wrongly terminate a real
    episode the way an actual 'contradicts' would).
    """
    assert is_failure(POLARITY_CONTRADICTS, "contradicts") is True
    assert is_failure(POLARITY_CONTRADICTS, "consistent") is False
    assert is_failure(POLARITY_CONTRADICTS, "cant_tell") is False


def test_is_failure_parse_error_always_fails():
    assert is_failure(POLARITY_YES_NO, "PARSE_ERROR") is True
    assert is_failure(POLARITY_CONTRADICTS, "PARSE_ERROR") is True


def test_self_check_parses_fake_backend_response(tmp_path):
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    verdict = self_check(client, IMG, "some statement", POLARITY_YES_NO)
    assert verdict == "yes"  # FakeVLMBackend's schema filler picks the first enum value


def test_self_check_returns_parse_error_on_malformed_json(tmp_path, monkeypatch):
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: "not json at all")
    verdict = self_check(client, IMG, "statement", POLARITY_YES_NO)
    assert verdict == "PARSE_ERROR"


def test_self_check_returns_parse_error_on_missing_verdict_key(tmp_path, monkeypatch):
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: '{"wrong_key": "yes"}')
    verdict = self_check(client, IMG, "statement", POLARITY_YES_NO)
    assert verdict == "PARSE_ERROR"
