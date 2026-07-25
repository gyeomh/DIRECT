"""Text -> SlotValue parsing: oracle answers (hedge detection) and the initial description parse.

Two different trust levels live here (spec §4, §5):
- The description (and `info["category"]`, see below) is authoritative — never hedged, always
  `certainty="resolved"`, `confidence=1.0`.
- Oracle answers can hedge ("it's hard to tell, maybe white") — those must NOT be treated as
  resolved, or a low-confidence guess could later read as a decisive CONFLICT (compare.py checks
  `belief.certainty == "resolved"` for exactly this reason).

Ground-truth note (found reading env.py, not in the original spec): `env.reset()` sets
`info["category"]` to `episode["tasks"]["category"]` *unconditionally*, regardless of which
`--description-type` is active for `target_description` (env.py:135). This is not the
`task_image` leak the spec warns against in §0.6 — it's a label the organizers add to `info` for
every task type — so `obj.category` should be seeded straight from `info["category"]` rather than
guessed from `target_description` text. Only fall back to the regex/LLM path if a future
integration point doesn't have `info["category"]` available.
"""

import re

from . import canon, schema
from .state import SlotValue, TargetBelief

HEDGE_MARKERS = [
    "maybe", "possibly", "perhaps", "i think", "i believe", "looks like", "seems",
    "might be", "could be", "hard to tell", "difficult to tell", "not entirely sure",
    "not sure", "appears to be", "probably", "kind of", "sort of",
]


def _norm_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def detect_hedge(answer: str) -> bool:
    a = _norm_text(answer)
    return any(marker in a for marker in HEDGE_MARKERS)


_WORD_NUMBERS = {
    "no": 0, "none": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _extract_count(text: str) -> int | None:
    key = _norm_text(text)
    if re.search(r"\b(no|none|not any)\b", key):
        return 0
    m = re.search(r"\d+", key)
    if m:
        return int(m.group())
    for word, n in _WORD_NUMBERS.items():
        if re.search(rf"\b{word}\b", key):
            return n
    return None


def _extract_bool(text: str) -> bool | None:
    count = _extract_count(text)
    if count is not None:
        return count > 0
    key = _norm_text(text)
    if re.search(r"\byes\b", key):
        return True
    if re.search(r"\bno\b", key):
        return False
    return None


def _clean_free_text(text: str) -> str | None:
    # First clause only — oracle answers are capped under 15 words but often still a full
    # sentence ("It's a stainless steel range."); a free-text slot canon value should be the
    # noun phrase, not the whole sentence. This is a first pass — no NP chunker.
    cleaned = re.sub(r"^(it'?s|it is|there is|there'?s|the .*? (is|are|rests? on|sits? on))\s+", "", _norm_text(text))
    return cleaned.split(",")[0].split(".")[0].strip().rstrip(".") or None


def _canon_for_slot(raw_answer: str, value_type: str) -> str | None:
    if value_type in schema.VOCAB:
        # oracle answers are prose ("The countertop is white."), not an isolated value — use
        # find_in_text (substring search), not normalize() (exact match only).
        return canon.find_in_text(raw_answer, value_type)
    if value_type == "int":
        count = _extract_count(raw_answer)
        return None if count is None else str(count)
    if value_type == "bool":
        b = _extract_bool(raw_answer)
        return None if b is None else str(b).lower()
    return _clean_free_text(raw_answer)  # "str"


def parse_oracle_answer(raw_answer: str, slot_key: str, question: str) -> SlotValue:
    """Oracle answers are capped under 15 words and the oracle is deterministic at
    temperature=1e-6 (spec §0.2) — never re-ask an identical question expecting a different
    answer; a hedge is a property of the answer text, not noise to average out.
    """
    value_type = schema.spec_for(slot_key).type
    canon_value = _canon_for_slot(raw_answer, value_type)

    # NOTE: `detect_no_info` deliberately does NOT gate this — HEDGE_MARKERS and NO_INFO_MARKERS
    # overlap ("not sure" is in both), and "I'm not sure, maybe granite?" must resolve to a
    # *hedged* granite, not be discarded as no-info just because it also hedges. If no value
    # could be extracted at all (a genuine no-info answer has nothing for find_in_text to match),
    # canon_value is already None here regardless of which marker list matched.
    if canon_value is None:
        return SlotValue(
            raw=raw_answer, canon=None, confidence=0.0, certainty="unknown",
            provenance="oracle", source_question=question,
        )
    if detect_hedge(raw_answer):
        return SlotValue(
            raw=raw_answer, canon=canon_value, confidence=0.5, certainty="hedged",
            provenance="oracle", source_question=question,
        )
    return SlotValue(
        raw=raw_answer, canon=canon_value, confidence=0.9, certainty="resolved",
        provenance="oracle", source_question=question,
    )


# --- initial description parse (spec §4) ---------------------------------------------------------
# TODO(together): the non-category slots below are meant to be filled by "one text-only LLM call
# using the same JSON schema" (spec §4) — that call needs a prompt, which we haven't written yet.
# Until DESCRIPTION_PARSE_PROMPT is filled in, only `obj.category` (from `info["category"]`, or the
# regex fallback) is seeded here; everything else starts unknown and gets filled by extract()/the
# oracle during the episode, same as if the description hadn't mentioned it at all. This is a
# conservative gap, not a correctness bug: worst case it costs a few wasted early questions.
DESCRIPTION_PARSE_PROMPT = None

_CATEGORY_HEURISTIC = re.compile(r"^(?:[a-z]+ ){0,3}?([a-z_]+)$")


def _category_fallback(description: str) -> str | None:
    """Best-effort guess only used if `info["category"]` isn't available. The dataset's own
    category strings are short common-noun phrases; this is intentionally weak — do not lean on
    it once info["category"] is wired through questioner.py.
    """
    words = _norm_text(description).split()
    return words[-1] if words else None


def parse_description(description: str, info_category: str | None = None) -> TargetBelief:
    belief = TargetBelief(description=description, noun_phrase=(info_category or description))

    category_text = info_category or _category_fallback(description)
    if category_text:
        belief.set_slot(
            "obj.category",
            SlotValue(
                raw=category_text, canon=_norm_text(category_text), confidence=1.0,
                certainty="resolved", provenance="description", source_question=None,
            ),
        )
        # Prefer "the {category}" over the raw category string as the noun phrase — reads more
        # naturally in the wh-templates ("What is the primary color of the cabinet?").
        belief.noun_phrase = f"the {_norm_text(category_text)}"

    return belief
