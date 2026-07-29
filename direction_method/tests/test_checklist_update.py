import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checklist_update import checklist_update, merge_checklist


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


# --- checklist_update(): files each answer under its OWN already-known key, verbatim --------
#
# No LLM call, no key classification -- `relation` in each (relation, answer) pair IS the
# checklist key. Confirmed against the real full sweep: the previous LLM-driven design misfiled
# content under the wrong key 51.5% of the time content grew at all. These tests exercise the
# code-only replacement.


def test_checklist_update_files_answer_under_its_own_relation_key():
    checklist = {}
    updated = checklist_update(checklist, [("left", "a mirror with a white frame")])
    assert updated == {"left": ["a mirror with a white frame"]}


def test_checklist_update_keeps_the_answer_verbatim_no_rephrasing():
    checklist = {}
    raw_answer = "There is nothing on top of the black TV. It is a flat screen with a black frame."
    updated = checklist_update(checklist, [("on", raw_answer)])
    assert updated["on"] == [raw_answer]


def test_checklist_update_never_files_an_answer_under_a_different_key_than_asked():
    # The exact real-sweep failure mode this replaces: an answer to "what is on the left" must
    # never end up under "left-bottom" or any other key.
    checklist = {}
    updated = checklist_update(checklist, [("left", "a mirror"), ("right", "a potted plant")])
    assert updated == {"left": ["a mirror"], "right": ["a potted plant"]}
    assert "left-bottom" not in updated
    assert "right-bottom" not in updated


def test_checklist_update_appends_to_an_existing_key():
    checklist = {"Target": ["it is black"]}
    updated = checklist_update(checklist, [("below", "a wooden dresser")])
    assert updated == {"Target": ["it is black"], "below": ["a wooden dresser"]}


def test_checklist_update_handles_multiple_answers_for_the_same_relation_in_one_round():
    checklist = {}
    updated = checklist_update(checklist, [("on", "a lamp"), ("on", "a book")])
    assert updated["on"] == ["a lamp", "a book"]


def test_checklist_update_dedups_an_exact_repeat_answer():
    checklist = {"left": ["a mirror"]}
    updated = checklist_update(checklist, [("left", "a mirror")])
    assert updated["left"] == ["a mirror"]


def test_checklist_update_is_a_no_op_on_empty_round_answers():
    checklist = {"Target": ["it is navy blue"]}
    result = checklist_update(checklist, [])
    assert result is checklist  # untouched, returned as-is, no work done


def test_checklist_update_does_not_mutate_the_input_checklist():
    checklist = {"Target": ["it is navy blue"]}
    checklist_update(checklist, [("left", "a table")])
    assert checklist == {"Target": ["it is navy blue"]}


def test_checklist_update_takes_no_llm_client_argument():
    # The whole point of the fix: there is no classification step left to call an LLM for.
    import inspect

    params = list(inspect.signature(checklist_update).parameters)
    assert params == ["checklist", "round_answers"]
