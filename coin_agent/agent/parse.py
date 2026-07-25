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

import json
import re

from . import canon, schema
from .extract import ExtractionParseError, _extract_json
from .llm import LLMClient, LLMCallFailed
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
# The one text-only LLM call that reads target_description and seeds everything it states into
# the belief BEFORE any candidate image is seen — this is what makes candidate_pool()'s "not
# implied by the description" filter and budget.ambiguity_allowance actually mean something.
# obj.category is handled separately (info["category"], see module docstring) and Tier-C slots
# (obj.style, obj.state, ctx.contains) are deliberately excluded — they're never decisive and
# never queried (schema.py §3.2), so parsing them here would be pure overhead with no downstream
# use. Only the *first* adjacent object/color is requested (ctx.adjacent[0].*) since every example
# description in this dataset mentions at most one adjacent object.

_DESCRIPTION_SLOT_KEYS = [
    k for k in schema.SLOTS if k != "obj.category" and schema.spec_for(k).tier != schema.TIER_C
] + ["ctx.adjacent[0].object", "ctx.adjacent[0].color"]

NOT_MENTIONED = "not_mentioned"

DESCRIPTION_PARSE_PROMPT = """You are extracting structured facts from one short description of \
an object and its immediate surroundings.

Description: "{description}"

The object this sentence describes IS the target object — extract facts ABOUT it and about what \
is spatially around it. Anything else the sentence names (a picture, a doorway, a wall) is \
surrounding context, not the target itself.

Extract ONLY facts explicitly stated or directly, unambiguously implied by this exact sentence. \
Never guess, never fill in a plausible-sounding default, never use outside knowledge about what \
this kind of object usually looks like. If the description does not mention a field, its value \
must be exactly "{not_mentioned}".

Rules (field names below are exactly the JSON keys you must use — see the schema at the bottom):
- "X and Y <object>" (e.g. "white and multicolored clock") -> the first color is \
obj.color_primary, the second is obj.color_secondary.
- Object-name fields (ctx.above.object, ctx.support.object, ctx.adjacent[0].object, \
room.notable_appliance) must be a single bare noun with no adjectives — "doorway", not "open \
doorway"; "picture", not "black framed picture". Put color separately in the matching *.color \
field. Drop any adjective that doesn't fit one of these fields (e.g. "open", "framed", "faceted") \
rather than folding it into the noun.
- "beneath/under/below X" -> the object is BELOW X, so X is ctx.above.object (plus \
ctx.above.material/ctx.above.color if X's material/color is also given).
- "standing on/set into/resting on/hanging on X" -> X is ctx.support.object.
- "next to/beside/against X" -> X is ctx.adjacent[0].object (plus ctx.adjacent[0].color if X's \
color is given).
- "with <material> accents/trim" (e.g. "red tile accents") -> that material is obj.material, and \
its color is obj.color_secondary.
- A wall's color mentioned anywhere -> room.wall_color. A floor's color/material -> \
room.floor_color / room.floor_material.
- Loose or movable items (stuffed animals, plush toys, towels) are contents, not context — there \
is no field for them; leave every other field "{not_mentioned}" if this is all the sentence says.

Return ONLY strict JSON, no other text, with exactly these keys and no others:
{schema_keys}
"""


def _build_description_prompt(description: str) -> str:
    schema_keys_skeleton = json.dumps({k: NOT_MENTIONED for k in _DESCRIPTION_SLOT_KEYS}, indent=2)
    return DESCRIPTION_PARSE_PROMPT.format(
        description=description, not_mentioned=NOT_MENTIONED, schema_keys=schema_keys_skeleton,
    )


def _slot_value_from_description_field(slot_key: str, raw_value) -> SlotValue | None:
    if not isinstance(raw_value, str) or raw_value.strip().lower() == NOT_MENTIONED:
        return None
    value_type = schema.spec_for(slot_key).type
    if value_type in schema.VOCAB:
        canon_value = canon.normalize(raw_value, value_type) or canon.find_in_text(raw_value, value_type)
    elif value_type == "int":
        canon_value = str(_extract_count(raw_value)) if _extract_count(raw_value) is not None else None
    elif value_type == "bool":
        canon_value = str(_extract_bool(raw_value)).lower() if _extract_bool(raw_value) is not None else None
    else:
        canon_value = _norm_text(raw_value) or None
    if canon_value is None:
        return None
    return SlotValue(
        raw=raw_value, canon=canon_value, confidence=1.0, certainty="resolved",
        provenance="description", source_question=None,
    )


def parse_description_llm_response(text: str) -> dict[str, SlotValue]:
    """Raises ExtractionParseError (via _extract_json) on malformed JSON — caller degrades by
    keeping whatever was already seeded from info["category"], per this module's docstring: a
    missing/failed description parse costs a few wasted early questions, not a correctness bug.
    """
    parsed = _extract_json(text)
    slots = {}
    for slot_key in _DESCRIPTION_SLOT_KEYS:
        value = _slot_value_from_description_field(slot_key, parsed.get(slot_key))
        if value is not None:
            slots[slot_key] = value
    return slots


_CATEGORY_HEURISTIC = re.compile(r"^(?:[a-z]+ ){0,3}?([a-z_]+)$")


def _category_fallback(description: str) -> str | None:
    """Best-effort guess only used if `info["category"]` isn't available. The dataset's own
    category strings are short common-noun phrases; this is intentionally weak — do not lean on
    it once info["category"] is wired through questioner.py.
    """
    words = _norm_text(description).split()
    return words[-1] if words else None


def parse_description(
    description: str, info_category: str | None = None, llm_client: LLMClient | None = None,
) -> TargetBelief:
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

    if llm_client is not None:
        try:
            result = llm_client.call(_build_description_prompt(description), temperature=0.0)
            for slot_key, value in parse_description_llm_response(result.text).items():
                belief.set_slot(slot_key, value)
        except (LLMCallFailed, ExtractionParseError):
            pass  # degrade: belief keeps only obj.category, same as a plain "category" description

    return belief
