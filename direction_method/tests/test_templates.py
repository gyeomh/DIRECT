import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from templates import (
    CHECKLIST_KEYS,
    EMPTY_REGION_ASSERTION,
    LATERAL_KEYS,
    QUESTION_SUFFIX,
    QUESTION_TEMPLATES,
    REGION_TEMPLATES,
    VIEWER_CONVENTION_CLAUSE,
    assertion_for_answer,
    is_empty_answer,
    question_for,
    region_for,
)
from zone_gen import REGION_KEYS as ZONE_GEN_KEYS

TARGET = "kitchen lower cabinet"


# --- region_for grammar, per key -----------------------------------------------------------


@pytest.mark.parametrize(
    "relation,expected",
    [
        ("left", "left of the kitchen lower cabinet"),
        ("right", "right of the kitchen lower cabinet"),
        ("above", "above the kitchen lower cabinet"),
        ("below", "below the kitchen lower cabinet"),
        ("left-top", "above and to the left of the kitchen lower cabinet"),
        ("right-top", "above and to the right of the kitchen lower cabinet"),
        ("left-bottom", "below and to the left of the kitchen lower cabinet"),
        ("right-bottom", "below and to the right of the kitchen lower cabinet"),
        ("on", "on top of the kitchen lower cabinet"),
        ("next to", "next to the kitchen lower cabinet"),
        ("Target", "the kitchen lower cabinet itself"),
    ],
)
def test_region_for_every_key_is_grammatical(relation, expected):
    assert region_for(relation, TARGET) == expected


def test_region_for_above_is_no_longer_the_ungrammatical_generic_formula():
    # the bug this change fixes: the old "{relation} of the {target}" formula produced
    # "above of the cabinet", which is not English.
    assert region_for("above", "cabinet") == "above the cabinet"
    assert "above of" not in region_for("above", "cabinet")


def test_region_for_is_deterministic_no_vlm_call():
    assert region_for("left", "cabinet") == region_for("left", "cabinet")


def test_region_for_unknown_key_raises():
    with pytest.raises(KeyError):
        region_for("diagonal", TARGET)


# --- question_for grammar, per key -----------------------------------------------------------


VIEWER = VIEWER_CONVENTION_CLAUSE.format(t=TARGET)


@pytest.mark.parametrize(
    "relation,expected_question",
    [
        ("left", "What is on the left of the kitchen lower cabinet?" + VIEWER),
        ("right", "What is on the right of the kitchen lower cabinet?" + VIEWER),
        ("above", "What is above the kitchen lower cabinet?"),
        ("below", "What is below the kitchen lower cabinet?"),
        ("left-top", "What is above and to the left of the kitchen lower cabinet?" + VIEWER),
        ("right-top", "What is above and to the right of the kitchen lower cabinet?" + VIEWER),
        ("left-bottom", "What is below and to the left of the kitchen lower cabinet?" + VIEWER),
        ("right-bottom", "What is below and to the right of the kitchen lower cabinet?" + VIEWER),
        ("on", "What is resting on the kitchen lower cabinet's top surface?"),
    ],
)
def test_question_for_every_zone_gen_key(relation, expected_question):
    assert question_for(relation, TARGET) == expected_question + QUESTION_SUFFIX


# --- viewer convention on lateral questions -----------------------------------------------------


def test_lateral_keys_are_exactly_the_ones_with_a_left_or_right_component():
    assert set(LATERAL_KEYS) == {
        "left", "right", "left-top", "right-top", "left-bottom", "right-bottom",
    }


@pytest.mark.parametrize("relation", ["left", "right", "left-top", "right-top", "left-bottom", "right-bottom"])
def test_lateral_questions_state_the_viewer_convention(relation):
    """zone_gen and self_check both pin screen-relative left/right; the oracle was never told, which
    is the mirroring risk behind `left`'s 7.7% self_check failure rate vs `right`'s 4.0%.
    """
    question = question_for(relation, TARGET)
    assert "as you see them looking at the image" in question
    assert f"not from the {TARGET}'s own point of view" in question


@pytest.mark.parametrize("relation", ["above", "below", "on"])
def test_vertical_only_questions_omit_the_viewer_convention(relation):
    # nothing to mirror on the vertical axis -- the clause would be noise in these questions.
    assert "point of view" not in question_for(relation, TARGET)


def test_question_for_appends_shape_and_color_suffix():
    assert question_for("left", TARGET).endswith(" Can you describe the shape and color?")


@pytest.mark.parametrize("relation", ["next to", "Target"])
def test_question_for_next_to_and_target_have_no_question_template(relation):
    # they only ever arrive from context_parser, never from zone_gen -- no live question is ever
    # generated for them via this table.
    with pytest.raises(KeyError):
        question_for(relation, TARGET)


# --- cross-module enum coverage: this is the actual regression test for template drift --------


def test_every_zone_gen_relation_key_has_a_question_template():
    missing = [k for k in ZONE_GEN_KEYS if k not in QUESTION_TEMPLATES]
    assert missing == [], f"zone_gen can emit these keys with no question template: {missing}"


def test_every_zone_gen_relation_key_has_a_region_template():
    missing = [k for k in ZONE_GEN_KEYS if k not in REGION_TEMPLATES]
    assert missing == [], f"zone_gen can emit these keys with no region template: {missing}"


def test_every_checklist_key_has_a_region_template():
    # CHECKLIST_KEYS is derived directly from REGION_TEMPLATES, so this is definitionally true --
    # kept as an explicit regression guard in case CHECKLIST_KEYS is ever redefined independently.
    missing = [k for k in CHECKLIST_KEYS if k not in REGION_TEMPLATES]
    assert missing == []


def test_checklist_keys_is_exactly_zone_gen_keys_plus_next_to_and_target():
    assert set(CHECKLIST_KEYS) == set(ZONE_GEN_KEYS) | {"next to", "Target"}


def test_question_templates_has_no_keys_outside_zone_gen_vocabulary():
    # the inverse direction: QUESTION_TEMPLATES must not silently grow a key zone_gen never emits,
    # since that would be dead code hiding a vocabulary mismatch.
    assert set(QUESTION_TEMPLATES.keys()) == set(ZONE_GEN_KEYS)


# --- empty-answer normalization -----------------------------------------------------------------
#
# The first five positive cases are verbatim answer strings from the 2026-08-11 27B run logs, which
# is every distinct emptiness wording in those 19 answers; the rest are near variants the pattern
# should also cover. Every negative case is a real answer from the same logs that merely contains a
# negation.


@pytest.mark.parametrize(
    "answer",
    [
        # --- observed verbatim in the run logs
        "nothing",
        " nothing",
        "Nothing.",
        " Nothing.",
        "Nothing is on top of the teal blanket. It is smooth and unadorned.",
        # --- near variants, not observed
        "none",
        "no objects",
        "There is nothing there.",
        "no items visible",
    ],
)
def test_is_empty_answer_detects_real_emptiness_answers(answer):
    assert is_empty_answer(answer) is True
    assert assertion_for_answer(answer) == EMPTY_REGION_ASSERTION


@pytest.mark.parametrize(
    "answer",
    [
        # describes content, merely contains a negation -- must survive verbatim
        "Below the clock is a plain, light beige wall with no distinct shape or color variations.",
        "nothing but a white wall",
        "Below the clock is a beige wall with a faint, indistinct shadow or mark.",
        "white sink",
        "vase with greenery, round white object, brown bowl",
    ],
)
def test_is_empty_answer_leaves_content_answers_verbatim(answer):
    assert is_empty_answer(answer) is False
    assert assertion_for_answer(answer) == answer


def test_empty_region_assertion_is_region_stripped():
    """self_check gets `region` as its own field and assertions never restate it (SPEC.md §2, §7).
    Embedding a region string in the negation is also what breaks it: "there is nothing left of the
    tv" reads as "nothing remaining".
    """
    for relation in CHECKLIST_KEYS:
        assert region_for(relation, "tv") not in EMPTY_REGION_ASSERTION
