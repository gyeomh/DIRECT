import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from color_family import COLOR_TERMS, NON_COLOR_MATERIALS, colors_in, reconcile, same_family

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- the real failures this exists for --------------------------------------------------------
#
# assertion / evidence pairs copied verbatim from self_check output on the 2026-08-11 35B run.
# The "true target" ones are false-"no"s (the answer is true of that image by construction); the
# distractor ones are correct rejections that must survive.


@pytest.mark.parametrize(
    "assertion,evidence",
    [
        ("it is light grey", "The shelves are clearly white, not light grey."),
        ("it is beige", "The lamp stand is white or off-white, not beige."),
        ("it is silver and mirrored", "The wardrobe is white, not silver or mirrored."),
        ("it is light gray", "The cabinet is painted a light, off-white or cream color."),
        ("The burgundy chair is on a white carpet",
         "The floor beneath the chair is a light beige carpet, not white."),
    ],
)
def test_rescues_within_family_false_negatives(assertion, evidence):
    assert reconcile(assertion, evidence) is True


@pytest.mark.parametrize(
    "assertion,evidence",
    [
        # genuinely different hue -- the claim really is contradicted
        ("it is beige", "The lamp stand is clearly green, not beige."),
        # correct distractor rejections, measured on distractor images
        ("it is beige", "The lamp stand is a dark, metallic brown, not beige."),
        ("it is black", "The lamp stand is gold or brass, not black."),
    ],
)
def test_leaves_real_contradictions_alone(assertion, evidence):
    assert reconcile(assertion, evidence) is False


def test_reconcile_needs_a_color_on_both_sides():
    assert reconcile("a wooden table", "The region holds a refrigerator.") is False
    assert reconcile("it is on the left", "Nothing is there.") is False


def test_reconcile_ignores_the_restated_claim_colour():
    # the evidence almost always repeats the claimed colour after "not"; if that counted as
    # perception, every colour verdict would reconcile with itself
    assert reconcile("it is beige", "The dresser is beige.") is False


# --- the family model -------------------------------------------------------------------------


def test_neutrals_separate_by_lightness_not_hue():
    assert same_family("white", "light grey") is True
    assert same_family("white", "beige") is True
    assert same_family("beige", "light brown") is True
    assert same_family("beige", "brown") is False         # bare "brown" is mid, beige is light
    assert same_family("beige", "dark brown") is False    # the distractor case
    assert same_family("white", "black") is False
    assert same_family("black", "dark grey") is True


def test_chromatic_hues_match_across_their_whole_range():
    assert same_family("navy", "light blue") is True
    assert same_family("deep blue", "blue") is True
    assert same_family("dark green", "sage") is True
    assert same_family("burgundy", "red") is True


def test_chromatic_never_matches_across_hue_or_into_neutral():
    assert same_family("green", "red") is False
    assert same_family("blue", "orange") is False
    assert same_family("gold", "black") is False
    assert same_family("beige", "green") is False


def test_longest_term_wins():
    assert colors_in("a light grey shelf") == ["light grey"]
    assert colors_in("a dark blue cabinet") == ["dark blue"]
    assert colors_in("dark brown wooden dresser") == ["dark brown", "wooden"]


def test_materials_without_a_hue_are_not_colors():
    for word in NON_COLOR_MATERIALS:
        assert word not in COLOR_TERMS, f"{word!r} has no hue and must not reconcile"


# --- corpus coverage: this is the test that matters as the eval set changes --------------------


def _corpus_texts():
    texts = []
    episodes = REPO_ROOT / "episodes_train.jsonl"
    if episodes.exists():
        for line in episodes.open():
            texts.extend(json.loads(line)["tasks"].values())
    for path in (REPO_ROOT / "direction_method" / "artifacts").glob("qwen36_35b_fixes*/[a-z]*.gzip.json"):
        data = json.load(gzip.open(path))
        for candidate in data["answers"]:
            for turn in candidate:
                texts.extend(re.sub(r"^A:\s*", "", a) for a in turn)
    return texts


# Every color-ish word the corpus actually uses. Kept as a literal list so the test states its own
# expectation rather than deriving it from the table it is checking.
CORPUS_COLOR_WORDS = [
    "white", "brown", "wooden", "green", "black", "blue", "red", "silver", "beige", "gray",
    "gold", "orange", "yellow", "stainless", "purple", "wood", "terracotta", "bronze",
    "marble", "brick", "cream", "navy", "granite", "teal", "tan", "golden", "maroon", "burgundy",
    "grey", "copper", "emerald", "mint", "brass", "rust", "forest", "pink", "metallic",
    "light brown", "dark brown", "dark gray", "light blue", "dark blue", "light gray", "light wood",
    "dark green", "light beige", "dark red", "dark wooden", "deep blue", "light wooden", "dark wood",
    "dark bronze", "light grey", "light tan", "natural wood", "light green", "bright red",
]


@pytest.mark.parametrize("word", CORPUS_COLOR_WORDS)
def test_every_corpus_color_word_is_in_the_table(word):
    """The table is only useful if it covers what the data actually says. A new eval set that
    introduces a colour this misses would silently stop reconciling it, so the miss should be a
    test failure, not a quiet regression.
    """
    assert word in COLOR_TERMS, f"{word!r} occurs in the corpus but has no family"


@pytest.mark.skipif(not (REPO_ROOT / "episodes_train.jsonl").exists(), reason="corpus not present")
def test_corpus_word_list_is_still_accurate():
    """Guards the list above against the corpus changing under it."""
    texts = _corpus_texts()
    assert texts, "expected to find episode descriptions"
    seen = Counter()
    for text in texts:
        for term in colors_in(text):
            seen[term] += 1
    missing = [w for w in CORPUS_COLOR_WORDS if w not in seen]
    assert not missing, f"listed as corpus words but no longer found: {missing}"
