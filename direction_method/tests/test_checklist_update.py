import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from checklist_update import (
    CHECKLIST_UPDATE_PROMPT,
    CHECKLIST_UPDATE_SCHEMA,
    ChecklistUpdateError,
    build_prompt,
    checklist_update,
    merge_checklist,
)
from llm import LLMClient
from templates import CHECKLIST_KEYS


def fake_client(tmp_path) -> LLMClient:
    return LLMClient("fake-model", backend="fake", cache_dir=tmp_path)


# --- prompt construction ------------------------------------------------------------------


def test_build_prompt_contains_verbatim_system_prompt():
    prompt, schema = build_prompt({}, [])
    assert prompt.startswith(CHECKLIST_UPDATE_PROMPT)
    assert schema == CHECKLIST_UPDATE_SCHEMA


def test_build_prompt_formats_current_checklist_one_line_per_assertion():
    checklist = {"Target": ["it is navy blue", "it has brass handles"], "above": ["a white sink"]}
    prompt, _ = build_prompt(checklist, [])
    assert "current checklist:\nTarget: it is navy blue\nTarget: it has brass handles\nabove: a white sink" in prompt


def test_build_prompt_formats_new_answers_by_relation():
    prompt, _ = build_prompt({}, [("left", "a wooden table with a plant on it"), ("on", "nothing visible")])
    assert "new answers:\nleft: a wooden table with a plant on it\non: nothing visible" in prompt


def test_build_prompt_empty_checklist_and_answers():
    prompt, _ = build_prompt({}, [])
    assert prompt.endswith("current checklist:\n\n\nnew answers:\n")


# --- schema shape --------------------------------------------------------------------------


def test_schema_pins_additions_keys_to_the_11_key_enum():
    additions_schema = CHECKLIST_UPDATE_SCHEMA["properties"]["additions"]
    assert additions_schema["additionalProperties"] is False
    assert set(additions_schema["properties"].keys()) == set(CHECKLIST_KEYS)
    assert len(CHECKLIST_KEYS) == 11


# --- merge_checklist: append-only, dedup, superset invariant --------------------------------


def test_merge_appends_new_assertions_under_existing_key():
    checklist = {"Target": ["it is navy blue"]}
    additions = {"Target": ["it has brass handles"]}
    merged = merge_checklist(checklist, additions)
    assert merged == {"Target": ["it is navy blue", "it has brass handles"]}


def test_merge_creates_new_key_not_previously_present():
    checklist = {}
    additions = {"left": ["a wooden table"]}
    merged = merge_checklist(checklist, additions)
    assert merged == {"left": ["a wooden table"]}


def test_merge_does_not_mutate_the_input_checklist():
    checklist = {"Target": ["it is navy blue"]}
    merge_checklist(checklist, {"Target": ["it has brass handles"]})
    assert checklist == {"Target": ["it is navy blue"]}  # unchanged


def test_merge_never_reorders_or_removes_existing_assertions():
    checklist = {"Target": ["a", "b", "c"]}
    merged = merge_checklist(checklist, {"Target": ["d"]})
    assert merged["Target"] == ["a", "b", "c", "d"]


def test_merge_dedups_exact_match_against_existing():
    checklist = {"next to": ["a nightstand"]}
    merged = merge_checklist(checklist, {"next to": ["a nightstand"]})
    assert merged["next to"] == ["a nightstand"]


def test_merge_dedups_after_normalizing_case_and_whitespace():
    checklist = {"next to": ["a nightstand"]}
    additions = {"next to": ["  A   Nightstand  "]}
    merged = merge_checklist(checklist, additions)
    assert merged["next to"] == ["a nightstand"]  # the new variant is dropped, original untouched


def test_merge_dedups_duplicates_within_the_same_additions_batch():
    checklist = {}
    additions = {"on": ["a blue blanket", "a BLUE blanket", "a blue blanket"]}
    merged = merge_checklist(checklist, additions)
    assert merged["on"] == ["a blue blanket"]


def test_merge_result_is_superset_of_pre_merge_checklist():
    checklist = {"Target": ["it is navy blue"], "above": ["a white sink"]}
    merged = merge_checklist(checklist, {"Target": ["it has brass handles"]})
    for key, assertions in checklist.items():
        assert merged[key][: len(assertions)] == assertions


def test_merge_with_no_additions_is_a_no_op():
    checklist = {"Target": ["it is navy blue"]}
    merged = merge_checklist(checklist, {})
    assert merged == checklist
    assert merged is not checklist  # still a fresh dict, per the "never mutate input" contract


# --- checklist_update(): end to end ---------------------------------------------------------


def test_checklist_update_parses_and_merges_additions(tmp_path, monkeypatch):
    client = fake_client(tmp_path)
    payload = {"additions": {"left": ["a wooden table"], "on": ["nothing visible"]}}
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: json.dumps(payload))

    checklist = {"Target": ["it is navy blue"]}
    updated = checklist_update(client, checklist, [("left", "a wooden table"), ("on", "there is nothing on it")])
    assert updated == {
        "Target": ["it is navy blue"],
        "left": ["a wooden table"],
        "on": ["nothing visible"],
    }


def test_checklist_update_skips_the_vlm_call_when_round_answers_is_empty(tmp_path, monkeypatch):
    client = fake_client(tmp_path)
    calls = []
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: calls.append(1) or "{}")

    checklist = {"Target": ["it is navy blue"]}
    result = checklist_update(client, checklist, [])
    assert result is checklist  # untouched, returned as-is
    assert calls == []


def test_checklist_update_raises_on_malformed_json(tmp_path, monkeypatch):
    client = fake_client(tmp_path)
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: "not json")
    with pytest.raises(ChecklistUpdateError):
        checklist_update(client, {}, [("left", "a table")])


def test_checklist_update_raises_on_missing_additions_key(tmp_path, monkeypatch):
    client = fake_client(tmp_path)
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: json.dumps({"wrong_key": {}}))
    with pytest.raises(ChecklistUpdateError):
        checklist_update(client, {}, [("left", "a table")])


def test_checklist_update_with_fake_backend_schema_filler_does_not_crash(tmp_path):
    client = fake_client(tmp_path)
    updated = checklist_update(client, {}, [("left", "a wooden table")])
    # FakeVLM's filler fills every additions property with a 1-item list -- all 11 keys appear
    assert set(updated.keys()) == set(CHECKLIST_KEYS)
