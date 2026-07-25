"""Candidate image -> ObservationFrame via the Qwen VLM (spec §5.1).

Hard requirements this prompt satisfies (spec §5.1, non-negotiable regardless of wording):
  - strict JSON matching `response_schema(max_adjacent)` below;
  - an explicit "unclear" option on every field;
  - a per-field self-reported `visibility` in {clear, partial, not_visible};
  - description-blind: no mention of "target", no reference to the description at all — this
    call must not be biased toward confirming what the description says (that bias is exactly
    what compare.py relies on *not* existing);
  - explicit instruction to ignore digital artifacts/distortions and never report them as content.

Same bare-noun / color-split-out convention as parse.py's DESCRIPTION_PARSE_PROMPT, and the same
`schema.queryable_slot_keys()` vocabulary — so a slot means the same thing whether it came from
the description or from a candidate image (obj.category and Tier-C slots excluded from both, see
that function's docstring).
"""

import json

from . import canon, schema
from .llm import LLMClient, LLMCallFailed, _image_hash
from .state import ObservationFrame, SlotValue

UNCLEAR = "unclear"

EXTRACTION_PROMPT = """Look closely at the image and report exactly what you can observe about \
the single main object in it and its immediate surroundings.

Describe only what is visible in THIS image. Do not reference any other image, any description, \
or what this kind of object usually looks like elsewhere — report only what you can actually see \
here.

For every field below, provide:
- "value": a short, literal, lowercase phrase for what you observe, or exactly "{unclear}" if you \
cannot determine it from this image.
- "visibility": "clear" if you can confidently observe it, "partial" if it is partially visible, \
occluded, or ambiguous, "not_visible" if it is not visible in this image at all.

Rules:
- Ignore compression artifacts, digital noise, rendering glitches, or watermarks entirely — never \
report them as real content.
- Object-name fields (ctx.above.object, ctx.support.object, ctx.adjacent[i].object, \
room.notable_appliance) must be a single bare noun with no adjectives — "doorway", not "open \
doorway"; "picture", not "black framed picture". Put color/material in the matching separate \
field instead of folding it into the noun.
- If a relation genuinely doesn't apply to this scene (e.g. there is no adjacent object at all), \
set its value to "{unclear}" and its visibility to "not_visible" — do not invent one.

Return ONLY strict JSON, no other text, with exactly these keys and no others, each an object \
with "value" and "visibility":
{schema_json}
"""

VISIBILITY_VALUES = ("clear", "partial", "not_visible")
_VISIBILITY_FACTOR = {"clear": 1.0, "partial": 0.5, "not_visible": 0.0}


class ExtractionParseError(RuntimeError):
    pass


def response_schema(max_adjacent: int) -> dict:
    """JSON schema requested from the model: one object per slot key, each with a raw text
    `value` (or the literal "unclear") and a self-reported `visibility`. Uses
    `schema.queryable_slot_keys` — obj.category and Tier-C slots are excluded (see that
    function's docstring: every candidate is guaranteed to be the target's category, and Tier-C
    slots have no consumer anywhere in compare/select/adjudicate).
    """
    slot_keys = schema.queryable_slot_keys(max_adjacent)
    return {
        "type": "object",
        "properties": {
            key: {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "description": f'The observed value, or "{UNCLEAR}".'},
                    "visibility": {"type": "string", "enum": list(VISIBILITY_VALUES)},
                },
                "required": ["value", "visibility"],
            }
            for key in slot_keys
        },
        "required": slot_keys,
    }


def _extract_json(text: str) -> dict:
    """Strip markdown code fences if present, then parse. Raises ExtractionParseError on failure
    so the caller can degrade per spec §7 (skip extraction -> all slots unknown -> adjudicate).
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as e:
        raise ExtractionParseError(f"Model did not return valid JSON: {e}\n---\n{text}") from e


def _slot_value_from_field(slot_key: str, field: dict, logprob_conf: float | None = None) -> SlotValue:
    """`logprob_conf=None` means "no real per-token logprob signal available" — confidence is
    then the model's self-reported `visibility` alone, NOT multiplied by a placeholder. Spec
    §5.1's mean-token-logprob signal is a real v2 addition (needs slicing result.logprobs to the
    exact token span of each field's "value" string, which depends on the prompt's exact JSON
    layout — worth doing once we've validated the visibility-only signal isn't already sufficient,
    per the same "test rather than assume" spirit as the ablation ladder). Multiplying in a fake
    neutral 0.5 here instead of skipping it would silently fail every slot below tau_obs (0.80
    default) — including "clear" ones — with no visible error; caught while wiring this up.
    """
    raw = field.get("value")
    visibility = field.get("visibility", "not_visible")
    vis_factor = _VISIBILITY_FACTOR.get(visibility, 0.0)

    if visibility == "not_visible" or raw is None or raw.strip().lower() == UNCLEAR:
        return SlotValue(raw=raw, canon=None, confidence=0.0, certainty="unknown", provenance=None)

    value_type = schema.spec_for(slot_key).type
    canon_value = canon.normalize(raw, value_type)
    confidence = vis_factor if logprob_conf is None else vis_factor * logprob_conf
    if canon_value is None:
        return SlotValue(raw=raw, canon=None, confidence=confidence, certainty="unknown", provenance=None)
    return SlotValue(raw=raw, canon=canon_value, confidence=confidence, certainty="resolved", provenance=None)


def _build_frame(image_hash: str, parsed: dict, max_adjacent: int) -> ObservationFrame:
    frame = ObservationFrame(image_hash=image_hash)
    for slot_key in schema.queryable_slot_keys(max_adjacent):
        field = parsed.get(slot_key, {"value": None, "visibility": "not_visible"})
        frame.slots[slot_key] = _slot_value_from_field(slot_key, field)
    return frame


def _majority_vote(frames: list[ObservationFrame], max_adjacent: int) -> ObservationFrame:
    """§5.1 self-consistency: per-slot majority over k samples; slot confidence becomes the
    agreement fraction. Only meaningful for candidate #1 given the time budget — see questioner.py
    for the caller-side "only on candidate #1" policy; this function itself is agnostic to that.
    """
    assert frames, "majority_vote requires at least one sample"
    merged = ObservationFrame(image_hash=frames[0].image_hash)
    k = len(frames)
    for slot_key in schema.queryable_slot_keys(max_adjacent):
        votes = [f.slots[slot_key] for f in frames]
        resolved_votes = [v for v in votes if v.canon is not None]
        if not resolved_votes:
            merged.slots[slot_key] = SlotValue(certainty="unknown", confidence=0.0)
            continue
        tally: dict[str, int] = {}
        for v in resolved_votes:
            tally[v.canon] = tally.get(v.canon, 0) + 1
        best_value, best_count = max(tally.items(), key=lambda kv: kv[1])
        agreement = best_count / k
        example = next(v for v in resolved_votes if v.canon == best_value)
        merged.slots[slot_key] = SlotValue(
            raw=example.raw, canon=best_value, confidence=agreement, certainty="resolved", provenance=None,
        )
    return merged


def _build_extraction_prompt(max_adjacent: int) -> str:
    schema_json = json.dumps(
        {k: {"value": UNCLEAR, "visibility": "not_visible"} for k in schema.queryable_slot_keys(max_adjacent)},
        indent=2,
    )
    return EXTRACTION_PROMPT.format(unclear=UNCLEAR, schema_json=schema_json)


def extract(
    image,
    llm_client: LLMClient,
    *,
    max_adjacent: int,
    self_consistency_k: int = 1,
    self_consistency_temperature: float = 0.7,
) -> ObservationFrame:
    image_hash = _image_hash(image)
    prompt = _build_extraction_prompt(max_adjacent)
    k = max(1, self_consistency_k)
    frames = []
    for i in range(k):
        temp = 0.0 if k == 1 else self_consistency_temperature
        try:
            result = llm_client.call(prompt, image, temperature=temp, want_logprobs=True)
        except LLMCallFailed:
            if frames:
                break  # degrade to majority over whatever samples we did get
            raise
        parsed = _extract_json(result.text)
        frames.append(_build_frame(image_hash, parsed, max_adjacent))

    if len(frames) == 1:
        return frames[0]
    return _majority_vote(frames, max_adjacent)
