import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from templates import (
    CHECKLIST_KEYS,
    QUESTION_SUFFIX,
    QUESTION_TEMPLATES,
    REGION_TEMPLATES,
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


@pytest.mark.parametrize(
    "relation,expected_question",
    [
        ("left", "What is on the left of the kitchen lower cabinet?"),
        ("right", "What is on the right of the kitchen lower cabinet?"),
        ("above", "What is above the kitchen lower cabinet?"),
        ("below", "What is below the kitchen lower cabinet?"),
        ("left-top", "What is above and to the left of the kitchen lower cabinet?"),
        ("right-top", "What is above and to the right of the kitchen lower cabinet?"),
        ("left-bottom", "What is below and to the left of the kitchen lower cabinet?"),
        ("right-bottom", "What is below and to the right of the kitchen lower cabinet?"),
        ("on", "What is on top of the kitchen lower cabinet?"),
    ],
)
def test_question_for_every_zone_gen_key(relation, expected_question):
    assert question_for(relation, TARGET) == expected_question + QUESTION_SUFFIX


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
