"""Candidate image -> ObservationFrame via the Qwen VLM (spec §5.1).

The prompt itself (`EXTRACTION_PROMPT`) is the first thing we're writing together next — see the
TODO below. Everything else here (the requested JSON shape, confidence combination, self-
consistency voting) is fully wired so that dropping in the prompt text is the only remaining step.

Hard requirements the prompt must satisfy (spec §5.1, non-negotiable regardless of wording):
  - strict JSON matching `response_schema(max_adjacent)` below;
  - an explicit "unclear" option on every field;
  - a per-field self-reported `visibility` in {clear, partial, not_visible};
  - description-blind: no mention of "target", no reference to the description at all — this
    call must not be biased toward confirming what the description says (that bias is exactly
    what compare.py relies on *not* existing);
  - explicit instruction to ignore digital artifacts/distortions and never report them as content.
"""

import json
import math

from . import canon, schema
from .llm import LLMClient, LLMCallFailed, _image_hash
from .state import ObservationFrame, SlotValue

# TODO(together): write this. Must satisfy every bullet in the module docstring above.
EXTRACTION_PROMPT = None

VISIBILITY_VALUES = ("clear", "partial", "not_visible")
_VISIBILITY_FACTOR = {"clear": 1.0, "partial": 0.5, "not_visible": 0.0}
UNCLEAR = "unclear"


class ExtractionParseError(RuntimeError):
    pass


def response_schema(max_adjacent: int) -> dict:
    """JSON schema requested from the model: one object per slot key, each with a raw text
    `value` (or the literal "unclear") and a self-reported `visibility`.
    """
    slot_keys = schema.all_slot_keys(max_adjacent)
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


def _logprob_confidence(token_logprobs: list[float] | None) -> float:
    """Mean token logprob of the value tokens, mapped through a sigmoid (spec §5.1). Falls back
    to a neutral 0.5 when the server didn't return logprobs (e.g. `want_logprobs=False`, or a
    backend that doesn't support it) rather than silently pretending to be confident.
    """
    if not token_logprobs:
        return 0.5
    mean_lp = sum(token_logprobs) / len(token_logprobs)
    return 1.0 / (1.0 + math.exp(-mean_lp))


def _slot_value_from_field(slot_key: str, field: dict, logprob_conf: float) -> SlotValue:
    raw = field.get("value")
    visibility = field.get("visibility", "not_visible")
    vis_factor = _VISIBILITY_FACTOR.get(visibility, 0.0)

    if visibility == "not_visible" or raw is None or raw.strip().lower() == UNCLEAR:
        return SlotValue(raw=raw, canon=None, confidence=0.0, certainty="unknown", provenance=None)

    value_type = schema.spec_for(slot_key).type
    canon_value = canon.normalize(raw, value_type)
    confidence = logprob_conf * vis_factor
    if canon_value is None:
        return SlotValue(raw=raw, canon=None, confidence=confidence, certainty="unknown", provenance=None)
    return SlotValue(raw=raw, canon=canon_value, confidence=confidence, certainty="resolved", provenance=None)


def _build_frame(image_hash: str, parsed: dict, per_slot_logprob_conf: dict, max_adjacent: int) -> ObservationFrame:
    frame = ObservationFrame(image_hash=image_hash)
    for slot_key in schema.all_slot_keys(max_adjacent):
        field = parsed.get(slot_key, {"value": None, "visibility": "not_visible"})
        frame.slots[slot_key] = _slot_value_from_field(slot_key, field, per_slot_logprob_conf.get(slot_key, 0.5))
    return frame


def _majority_vote(frames: list[ObservationFrame], max_adjacent: int) -> ObservationFrame:
    """§5.1 self-consistency: per-slot majority over k samples; slot confidence becomes the
    agreement fraction. Only meaningful for candidate #1 given the time budget — see questioner.py
    for the caller-side "only on candidate #1" policy; this function itself is agnostic to that.
    """
    assert frames, "majority_vote requires at least one sample"
    merged = ObservationFrame(image_hash=frames[0].image_hash)
    k = len(frames)
    for slot_key in schema.all_slot_keys(max_adjacent):
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


def extract(
    image,
    llm_client: LLMClient,
    *,
    max_adjacent: int,
    self_consistency_k: int = 1,
    self_consistency_temperature: float = 0.7,
) -> ObservationFrame:
    if EXTRACTION_PROMPT is None:
        raise NotImplementedError(
            "extract.EXTRACTION_PROMPT is not written yet — see the module docstring TODO."
        )

    image_hash = _image_hash(image)
    k = max(1, self_consistency_k)
    frames = []
    for i in range(k):
        temp = 0.0 if k == 1 else self_consistency_temperature
        try:
            result = llm_client.call(EXTRACTION_PROMPT, image, temperature=temp, want_logprobs=True)
        except LLMCallFailed:
            if frames:
                break  # degrade to majority over whatever samples we did get
            raise
        parsed = _extract_json(result.text)
        per_slot_conf = {
            key: _logprob_confidence(None)  # TODO(together): slice result.logprobs per value-token span once the prompt's token layout is fixed
            for key in schema.all_slot_keys(max_adjacent)
        }
        frames.append(_build_frame(image_hash, parsed, per_slot_conf, max_adjacent))

    if len(frames) == 1:
        return frames[0]
    return _majority_vote(frames, max_adjacent)
