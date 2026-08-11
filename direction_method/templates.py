"""Shared region/question string templates.

Used by both the checklist path (self_check's `region` field, via `region_for`) and the question
path (what is actually asked to the oracle, via `question_for`). A single shared table is
required: the old formula `"{relation} of the {target}"` was only grammatical for `left`/`right`
("above of the cabinet" is wrong), and letting the two paths format independently risks them
drifting apart -- self_check's region string must always describe the same thing the oracle was
actually asked about.

`next to` and `Target` only ever arrive from `context_parser` (never from `zone_gen`), so they have
no entry in `QUESTION_TEMPLATES` -- no question is ever generated for them via this table; the
mandatory first/target-appearance question is a separate, fixed string (SPEC.md §6).

Also holds the empty-answer normalizer (`is_empty_answer` / `EMPTY_REGION_ASSERTION`): another
deterministic, no-VLM string decision shared by the self_check path and the checklist path, which
is why it lives beside the tables rather than in `questioner.py`.
"""

import re

REGION_TEMPLATES = {
    "left": "left of the {t}",
    "right": "right of the {t}",
    "above": "above the {t}",
    "below": "below the {t}",
    "left-top": "above and to the left of the {t}",
    "right-top": "above and to the right of the {t}",
    "left-bottom": "below and to the left of the {t}",
    "right-bottom": "below and to the right of the {t}",
    "on": "on top of the {t}",
    "next to": "next to the {t}",
    "Target": "the {t} itself",
}

QUESTION_TEMPLATES = {
    "left": "What is on the left of the {t}?",
    "right": "What is on the right of the {t}?",
    "above": "What is above the {t}?",
    "below": "What is below the {t}?",
    "left-top": "What is above and to the left of the {t}?",
    "right-top": "What is above and to the right of the {t}?",
    "left-bottom": "What is below and to the left of the {t}?",
    "right-bottom": "What is below and to the right of the {t}?",
    # Not "What is on top of the {t}?" -- that phrasing is loose enough in ordinary English that the
    # oracle answers it as `above`, naming the object higher up rather than the one resting on the
    # target. Measured on the 2026-08-11 27B run: `on` failed self_check 33/269 (12.3%), the worst of
    # any relation and 3x `above`'s 3.9%, with answers like "a large framed painting" for a piano
    # (it is on the wall) and "orange cabinet with silver handles" for an oven (an upper cabinet).
    # self_check then reads `region = "on top of the {t}"` strictly and calls it a contradiction.
    #
    # The asymmetry is why only this key is reworded: the oracle conflating the two costs nothing in
    # the `above` direction (a resting object still satisfies "above the {t}" under self_check's
    # coarse-position rule 4) and a false `"no"` in the `on` direction. Wording deliberately reuses
    # zone_gen's own words for this relation ("objects resting on it", zone_gen.py) so the relation
    # is described identically at the point it is chosen and at the point it is asked about.
    # Singular "What is", not "What objects are": measured on the 2026-08-11 35B run, the plural
    # form made the oracle enumerate the whole surface (mean 8.9 -> 10.6 words, 2.03 -> 2.33 listed
    # items, while every other relation's answers stayed flat or got shorter). A longer conjunctive
    # list is a stricter claim -- self_check has to find every item -- so `on`'s false-`"no"` rate on
    # the TRUE match went 2 -> 11 while its distractor rejections fell 18 -> 13: worse on both axes.
    # The "top surface" pinning itself worked and is kept: the oracle stopped naming wall-mounted
    # and overhead objects ("washing machine" -> "White cabinets", "stove" -> "A silver microwave"
    # both disappeared), which was the original failure this template was changed to fix.
    "on": "What is resting on the {t}'s top surface?",
}

QUESTION_SUFFIX = " Can you describe the shape and color?"

# The viewer convention was pinned everywhere the pipeline reasons about direction -- zone_gen's
# prompt ("screen-left is 'left' ... never mirror this as if the target object itself were a person
# facing the camera") and self_check's rule 4 ("always screen-left/screen-right as the image is
# viewed") -- but never in the question the oracle actually receives, which left the oracle free to
# answer from the object's own point of view and mirror the two. Consistent with that: on the
# 2026-08-11 27B run `left` failed self_check 54/698 (7.7%) against `right`'s 24/597 (4.0%), and the
# hand-read failure list in SPEC.md §14 calls its one left/right flip "the most concerning of the
# five, since left/right is a hard binary, not a coarse judgment call".
#
# Appended only to the keys whose question contains a lateral component -- the vertical-only keys
# (`above`, `below`, `on`) are not mirrorable, and a clause about left/right in their question would
# be noise the oracle has to read past.
VIEWER_CONVENTION_CLAUSE = (
    " Judge left and right as you see them looking at the image, not from the {t}'s own point of view."
)

# The full 11-key checklist enum (context_parser's region keys, §10) -- every key REGION_TEMPLATES
# knows about. zone_gen's own REGION_KEYS (§5) is the 9-key subset that excludes "next to"/"Target",
# since those two can never come from a zone_gen call.
CHECKLIST_KEYS = tuple(REGION_TEMPLATES.keys())

# Derived from the templates rather than hand-listed, so a key added to QUESTION_TEMPLATES with a
# lateral component cannot silently miss the viewer-convention clause.
LATERAL_KEYS = tuple(
    key for key, template in QUESTION_TEMPLATES.items()
    if "left" in template or "right" in template
)


def region_for(relation: str, target_phrase: str) -> str:
    """Deterministic string assembly, no VLM call. `relation` is a checklist parent key (any of
    the 11 `CHECKLIST_KEYS`) or a zone_gen relation key (a 9-key subset of the same vocabulary) --
    both vocabularies are covered by the same `REGION_TEMPLATES` table, so the checklist path and
    the oracle-answer path always produce identical region strings for the same relation.
    """
    return REGION_TEMPLATES[relation].format(t=target_phrase)


def question_for(relation: str, target_phrase: str) -> str:
    """Only defined for zone_gen's 9 relation keys -- `next to`/`Target` raise KeyError, since a
    live question is never generated for them (SPEC.md §6). Lateral relations additionally carry the
    viewer-convention clause, so the oracle is told the same screen-relative convention that
    zone_gen and self_check already use.
    """
    question = QUESTION_TEMPLATES[relation].format(t=target_phrase)
    if relation in LATERAL_KEYS:
        question += VIEWER_CONVENTION_CLAUSE.format(t=target_phrase)
    return question + QUESTION_SUFFIX


# --- empty-answer normalization ----------------------------------------------------------------
#
# The oracle sometimes answers a relation question with a bare "nothing". Passed through verbatim
# it becomes `assertion="nothing"`, which self_check has nothing to look for: there is no object in
# it, so the verdict turns on whatever the model thinks it sees in the region and contradiction
# rule 5 ("REGION CLAIMED EMPTY BUT IS NOT") fires. Measured on the 2026-08-11 27B run: 19 such
# answers, 10 of them scored `"no"` -- and every one of the 10 was an `on` answer, where the region
# is the target's own top surface. `checklist_update` files answers verbatim, so an un-normalized
# "nothing" is then re-checked against every later candidate in the episode too.
#
# Rewriting it into an explicit, region-stripped emptiness sentence keeps the discriminative signal
# (a candidate whose region plainly holds objects still contradicts, by that same rule 5) while
# giving self_check a claim it can actually judge. Region-stripped because self_check receives
# `region` as a separate field and assertions never restate it (SPEC.md §2, §7) -- and because the
# region strings do not survive being embedded in a negation: "there is nothing left of the tv"
# reads as "nothing remaining", not as "the left region is empty".
EMPTY_REGION_ASSERTION = "this region is empty and holds no objects"

# Deliberately tight, and anchored with fullmatch on the FIRST sentence only: the answer must be an
# emptiness statement outright, not merely contain a negation. "Below the clock is a plain wall with
# no distinct shape" and "nothing but a white wall" both stay verbatim -- they describe content.
_EMPTY_ANSWER_RE = re.compile(
    r"\W*"
    # optional lead-in
    r"(?:there\s+(?:is|are)\s+|i\s+(?:see|can\s+see)\s+)?"
    # the emptiness word itself
    r"(?:nothing|none|no\s+(?:objects?|items?|things?))"
    # optional tail, in one of three shapes -- anything else and the answer is left verbatim
    r"(?:"
    r"\s+(?:is|are)\s+[^.!?]*"  # "Nothing is on top of the teal blanket."
    r"|"
    # a locative prepositional phrase restating the region: "nothing on top of the black TV".
    # Restricted to an explicit preposition list precisely so that "nothing but a white wall" --
    # which describes content -- does not match: "but" is not on it.
    r"\s+(?:on|in|inside|under|underneath|beneath|above|below|behind|beside|near|next\s+to|"
    r"to\s+the|at)\b[^.!?]*"
    r"|"
    r"(?:\s+(?:there|here|visible|at\s+all))*"  # "There is nothing there.", "no items visible"
    r")"
    r"[\s.!?]*",
    re.IGNORECASE,
)


def is_empty_answer(answer: str) -> bool:
    """True when the oracle's answer asserts the region holds nothing, rather than describing
    something in it. Only the first sentence is tested -- the observed long form trails a clause
    about the target itself ("Nothing is on top of the teal blanket. It is smooth and unadorned."),
    which is not region content and is dropped with the rest of the answer.
    """
    first_sentence = re.split(r"(?<=[.!?])\s+", answer.strip(), maxsplit=1)[0]
    return bool(_EMPTY_ANSWER_RE.fullmatch(first_sentence.strip()))


def assertion_for_answer(answer: str) -> str:
    """The oracle answer as self_check and the checklist should both see it: verbatim, unless it is
    an emptiness statement, in which case the canonical region-stripped form."""
    return EMPTY_REGION_ASSERTION if is_empty_answer(answer) else answer
