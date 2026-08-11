"""context_parser (SPEC.md §10): description -> (target_category, target_phrase, other_objects,
checklist).

Text-only call -- no image. Runs once per episode, at episode start.

Prompt text below is used verbatim -- do not paraphrase, shorten, or "clean up" the rule lists,
same policy as self_check.py and zone_gen.py.

`other_objects` exists because the model's own prompt examples were teaching the wrong behavior:
"cabinet with brass handles" -> Target and "bed with a blue blanket" -> "on" share the exact same
surface form ("with X"), so a model generalizing from examples learns "with X -> Target: it has
X" unless something forces it to separate the two cases before writing the checklist. Field order
is generation order (same pattern as self_check's evidence-before-verdict, zone_gen's
note-before-key): other_objects must be committed to BEFORE checklist, so a relation can't get
silently folded into a Target-shaped sentence after the fact.

That alone was not enough. Confirmed against a live server, 6 description types x 30 episodes:
other_objects came back correct (object/cue/key all right) in nearly every case, but the model's
own checklist still frequently failed to carry those same entries through -- either dropping them
entirely or padding "Target" with repetitive filler instead. The failure rate tracked description
richness almost exactly: ~0% on category/color-only descriptions (nothing relational to get
wrong), 10-17% on single-relation descriptions, 90-93% on the two richest types that pack a color
clause and a context clause (and sometimes a feature clause) into one sentence. So checklist's
non-"Target" entries are no longer asked of the model at all -- they are built mechanically in
code from other_objects (`_merge_other_objects_into_checklist`), which is the one field confirmed
reliable. The model's checklist output is now used for "Target" only, the one thing it already
gets right at near-0% failure.

RULE 1 was atomic decomposition (split every Target attribute into its own child) until
2026-08-05 -- reversed, see SPEC.md SS2. Real sweeps showed the split routinely severing a
shape/material word from the color it modified (e.g. "it is mirrored" / "it is silver" as two
disconnected children), and the dominant real failure mode is self_check false-rejecting genuine
matches, not the ambiguity-from-occlusion atomicity was meant to prevent. Target attributes are
now written as one combined assertion.

Depth cues ("in front of", "behind") had no worked example before 2026-08-05, and the model
guessed a 2D screen direction for them -- confirmed on a live run: "desk positioned in front of a
window" produced key="below" for the window, which is physically nonsensical (windows are
wall-height, not below desk-height) and, because the checklist persists and grows for the whole
episode (SPEC.md SS2), poisoned every candidate in that episode with an unsatisfiable assertion
(self_check correctly says "no" every time, since there truly is no window below the desk in any
photo) -- one bad entry from episode start fails the whole episode regardless of which candidate
is the real match. Fixed by mapping depth cues to "next to" (SS REGION KEYS below), the same
escape hatch already used for "no side given" -- this project's region keys are 2D-screen-only, so
a depth relation has no honest directional answer among them.
"""

import json
import re
from dataclasses import dataclass, field

from llm import LLMClient
from templates import CHECKLIST_KEYS

CONTEXT_PARSER_PROMPT = """Parse an object description into a target object and a checklist.

The description always describes one main target object, sometimes with other
objects around it.

=== PROCEDURE ===

1. Identify the target object.
2. List every OTHER object the description mentions. For each, note the wording
   that links it to the target, then choose its region key.
3. Write the target's OWN attributes as the checklist's "Target" entries.

Do not repeat other_objects in the checklist -- their entries are added
automatically from other_objects. Never describe an other_objects entry under
"Target" either. "Target" holds only the target's own attributes and its
integral parts.

=== OUTPUT FIELDS ===

target_category — the bare category noun phrase, no attributes.
                  Used to locate the object in other images where attributes differ.

target_phrase   — the target with attributes that belong to the target ITSELF.
                  Do NOT include clauses about other objects.
                  "navy blue cabinet under a white sink" -> "navy blue cabinet"

other_objects   — every object other than the target that the description
                  mentions. For each: the object itself, the wording that
                  links it to the target ("cue"), and the region key that
                  wording implies.

checklist       — the target's OWN attributes, under "Target", only. Leave out
                  any key other than "Target" -- that content comes from
                  other_objects, not from here.

=== REGION KEYS ===

left, right, above, below,
left-top, right-top, left-bottom, right-bottom,
on, next to, Target

Relations are from the TARGET's point of view: describe where the OTHER object
sits relative to the target. Reverse the description's wording when needed:
  "cabinet UNDER a countertop"  -> the countertop is ABOVE the cabinet -> "above"
  "bed BESIDE a nightstand"     -> no side given                       -> "next to"
  "vase ON a wooden cabinet"    -> the cabinet is BELOW the vase       -> "below"

"left"/"right" (and the four corner keys) are always screen-left/screen-right as
the photo is viewed -- the same convention a person points with while looking at
the image. Never mirror them as if the target object itself were a person facing
the camera. This is the only convention used anywhere in this pipeline (self_check,
zone_gen, the oracle questions) -- do not reinterpret it per description.

DEPTH cues -- "in front of", "behind" -- describe distance from the camera, not a
2D screen direction, and this project's region keys are 2D-screen-only (no "behind"
key exists). Map both to "next to", the same escape hatch used for "beside"/no-side
cues, rather than guessing a screen direction:
  "desk in front of a window"   -> no 2D side given                    -> "next to"
  "shelf behind a chair"        -> no 2D side given                    -> "next to"

Use "next to" only when the description gives no side, including for the DEPTH
case above.

=== "ON" HAS TWO DIRECTIONS -- ASK WHICH OBJECT IS HOLDING THE OTHER ===

The region key "on" means ONE thing: the other object rests on the TARGET's own
top surface. It never means the target rests on something else.

So the word "on" in a description does not pick the key. Ask which of the two
objects is holding the other up:

  the OTHER object rests on the target -> "on"
    "bed with a blue blanket"        the blanket sits on the bed      -> "on"
    "table with books on it"         the books sit on the table       -> "on"

  the TARGET rests on the other object -> "below"
    "vase on a wooden cabinet"       the cabinet holds up the vase    -> "below"
    "armchair on a red rug"          the rug is under the armchair    -> "below"
    "blanket on a bed"               the bed holds up the blanket     -> "below"

  the target is MOUNTED on a wall or other vertical surface -> "next to"
    A wall behind the target is neither above nor below it -- the same depth
    problem as "behind"/"in front of", so it takes the same escape hatch.
    "clock hanging on a beige wall"  the wall is behind the clock     -> "next to"
    "grab bar on the tiled wall"     the wall is behind the bar       -> "next to"

Note how the same object flips key with the roles: a blanket is "on" when the bed
is the target, and the bed is "below" when the blanket is the target.

=== PARTS vs SEPARATE OBJECTS ===

Test: could you carry it into another room and leave the target unchanged?

  no  -> integral part -> "Target"
         handles, drawers, legs, doors, frame, upholstery
  yes -> separate object -> a region key
         blankets, pillows, plants, books, dishes, mirrors, sinks, tables

Both appear as "with X" in descriptions. The wording does not decide it.

  "cabinet with brass handles"   -> handles are part of the cabinet -> Target
  "bed with a blue blanket"      -> a blanket is a separate object  -> "on"

=== NEVER PUT A RELATION IN A TARGET ASSERTION ===

A "Target" assertion must not contain a spatial word: next to, beside, under,
beneath, below, above, on, on top of, behind, in front of, near.

  wrong:  Target: ["the bed is next to a nightstand"]
  right:  next to: ["a nightstand"]

  wrong:  Target: ["the bed is beneath a round mirror"]
  right:  above: ["a round mirror"]

=== ASSERTION STYLE ===

An assertion is a short phrase, NOT a sentence. It does not restate the region
or the target — those are added later from the region key.

  good:  "a white farmhouse sink"
  good:  "it is navy blue"                          (under Target)
  bad:   "There is a white sink above the cabinet."  (restates the region)
  bad:   "The cabinet is navy blue."                 (restates the target)

=== RULES ===

1. ONE COMBINED ASSERTION. Do not split the target's attributes into separate
   facts -- state everything the description says about the target's own
   attributes in a single assertion.
   "navy blue with brass handles" -> "it is navy blue and has brass handles"

2. NO INVENTION. Add nothing the description does not state.

3. If the target itself has no attributes to state, leave "Target" out of the
   checklist -- the checklist can be empty even when other_objects is not.

=== EXAMPLES ===

"Kitchen lower cabinet"
{"target_category": "kitchen lower cabinet",
 "target_phrase": "kitchen lower cabinet",
 "other_objects": [],
 "checklist": {}}

"Navy blue kitchen lower cabinet with brass handles"
{"target_category": "kitchen lower cabinet",
 "target_phrase": "navy blue kitchen lower cabinet",
 "other_objects": [],
 "checklist": {"Target": ["it is navy blue and has brass handles"]}}

"Kitchen lower cabinet situated beneath a white countertop"
{"target_category": "kitchen lower cabinet",
 "target_phrase": "kitchen lower cabinet",
 "other_objects": [{"object": "a white countertop", "cue": "beneath", "key": "above"}],
 "checklist": {}}

"Navy blue kitchen lower cabinet under a white farmhouse sink"
{"target_category": "kitchen lower cabinet",
 "target_phrase": "navy blue kitchen lower cabinet",
 "other_objects": [{"object": "a white farmhouse sink", "cue": "under", "key": "above"}],
 "checklist": {"Target": ["it is navy blue"]}}

"White bed with a blue blanket next to a nightstand"
{"target_category": "bed",
 "target_phrase": "white bed",
 "other_objects": [{"object": "a blue blanket", "cue": "with", "key": "on"},
                    {"object": "a nightstand", "cue": "next to", "key": "next to"}],
 "checklist": {"Target": ["it is white"]}}

"Green display cabinet next to open shelving"
{"target_category": "display cabinet",
 "target_phrase": "green display cabinet",
 "other_objects": [{"object": "open shelving", "cue": "next to", "key": "next to"}],
 "checklist": {"Target": ["it is green"]}}

"Desk positioned in front of a window"
{"target_category": "desk",
 "target_phrase": "desk",
 "other_objects": [{"object": "a window", "cue": "in front of", "key": "next to"}],
 "checklist": {}}

"Dark gray slatted heater beneath a round mirror"
{"target_category": "heater",
 "target_phrase": "dark gray slatted heater",
 "other_objects": [{"object": "a round mirror", "cue": "beneath", "key": "above"}],
 "checklist": {"Target": ["it is dark gray and slatted"]}}

"Large beige carpet under a wooden coffee table"
{"target_category": "carpet",
 "target_phrase": "large beige carpet",
 "other_objects": [{"object": "a wooden coffee table", "cue": "under", "key": "above"}],
 "checklist": {"Target": ["it is large and beige"]}}

"Gray couch with pillows under three framed artworks"
{"target_category": "couch",
 "target_phrase": "gray couch",
 "other_objects": [{"object": "pillows", "cue": "with", "key": "on"},
                    {"object": "three framed artworks", "cue": "under", "key": "above"}],
 "checklist": {"Target": ["it is gray"]}}

"Terracotta vase on a wooden cabinet"
{"target_category": "vase",
 "target_phrase": "terracotta vase",
 "other_objects": [{"object": "a wooden cabinet", "cue": "on", "key": "below"}],
 "checklist": {"Target": ["it is terracotta"]}}

"Green armchair on a large red rug"
{"target_category": "armchair",
 "target_phrase": "green armchair",
 "other_objects": [{"object": "a large red rug", "cue": "on", "key": "below"}],
 "checklist": {"Target": ["it is green"]}}

"Black clock hanging on a beige wall"
{"target_category": "clock",
 "target_phrase": "black clock",
 "other_objects": [{"object": "a beige wall", "cue": "hanging on", "key": "next to"}],
 "checklist": {"Target": ["it is black"]}}"""

# maxItems=8 on both arrays below: confirmed against a live vllm==0.15.0 server that without it,
# the model can enter a token-repetition loop (the same assertion string appended dozens of
# times) that a strict json_schema grammar does nothing to stop -- an unbounded array is a valid
# completion at every step, so it runs until max_tokens truncates the response mid-string and it
# fails to parse entirely. No real description needs more than a handful of items per key.
#
# maxLength=200 on every string field in this module: maxItems alone was NOT sufficient --
# reproduced on checklist_update.py's identically-shaped schema during the full 167-episode real
# sweep, where the model instead degenerated WITHIN a single string (repeating the same clause
# over and over) rather than across array items, still truncating mid-string since maxItems never
# bounds a string's own length. Applied here too, defensively, before the same failure mode shows
# up in this module's own strings under the larger sample.
_CHECKLIST_VALUE_SCHEMA = {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 8}

# other_objects' own "key" is never "Target" -- these are explicitly objects OTHER than the
# target, so "Target" is not a meaningful choice for where they'd file in the checklist.
_RELATION_KEYS_NO_TARGET = tuple(k for k in CHECKLIST_KEYS if k != "Target")

_OTHER_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "object": {"type": "string", "maxLength": 200, "description": "The other object mentioned in the description."},
        "cue": {"type": "string", "maxLength": 200, "description": "The exact wording linking it to the target."},
        "key": {"type": "string", "enum": list(_RELATION_KEYS_NO_TARGET)},
    },
    "required": ["object", "cue", "key"],
}

# checklist keys are pinned to the 11-key enum (templates.CHECKLIST_KEYS) via additionalProperties:
# False + an explicit property per key -- there is no per-key "required", since only the region
# keys the description actually mentions should appear (rule 3: category-only -> checklist == {}).
CONTEXT_PARSER_SCHEMA = {
    "type": "object",
    "properties": {
        "target_category": {"type": "string", "maxLength": 200, "description": "Bare category noun phrase, no attributes."},
        "target_phrase": {"type": "string", "maxLength": 200, "description": "Target with its own attributes; no clauses about other objects."},
        "other_objects": {
            "type": "array",
            "items": _OTHER_OBJECT_SCHEMA,
            "maxItems": 8,
        },
        "checklist": {
            "type": "object",
            "properties": {k: _CHECKLIST_VALUE_SCHEMA for k in CHECKLIST_KEYS},
            "additionalProperties": False,
        },
    },
    "required": ["target_category", "target_phrase", "other_objects", "checklist"],
}


class ContextParserError(Exception):
    """Raised when the response can't be turned into a usable ParsedContext -- malformed JSON or
    a missing required field."""


@dataclass
class ParsedContext:
    target_category: str  # bare noun phrase -- feeds zone_gen.locate (SPEC.md §5-1)
    target_phrase: str  # target + its own attributes -- feeds templates.region_for/question_for
    other_objects: list  # [{"object": ..., "cue": ..., "key": ...}, ...], as returned
    checklist: dict  # {region_key: [assertion, ...]}; only keys from templates.CHECKLIST_KEYS
    validation_problems: list = field(default_factory=list)  # non-empty if still flagged post-retry
    retried: bool = False  # True iff the first attempt was flagged and a retry was made


def build_prompt(description: str) -> tuple:
    prompt = f"{CONTEXT_PARSER_PROMPT}\n\n{description}"
    return prompt, CONTEXT_PARSER_SCHEMA


# Word-boundary matched against a lowercased Target assertion. "on" alone would false-positive
# inside ordinary words (e.g. "wooden"), hence \b on both sides via the regex build below.
_SPATIAL_WORDS = [
    "next to", "beside", "under", "beneath", "below", "above",
    "on top of", "on", "behind", "in front of", "near",
]


def _target_assertions_with_spatial_words(checklist: dict) -> list:
    """Detects the one remaining failure mode this module's prompt redesign targets: a relation
    reworded into a Target-shaped sentence instead of listed in other_objects. (The other original
    failure mode -- an other_objects entry never making it into the checklist -- is now prevented
    by construction: _merge_other_objects_into_checklist builds those entries in code, it never
    trusts the model to restate them.)"""
    hits = []
    for assertion in checklist.get("Target", []):
        lowered = assertion.lower()
        for word in _SPATIAL_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                hits.append((assertion, word))
                break
    return hits


def _support_surfaces_misfiled_under_on(description: str, other_objects: list) -> list:
    """Detects the §11 failure mode: the description says the TARGET rests on X, and X was filed
    under `on` -- which `templates.region_for` renders as "on top of the {target}", i.e. the
    target's own top surface. `"Terracotta vase on a wooden cabinet"` then asks for a wooden cabinet
    on top of the vase, which no photo can satisfy, and because it lands in the INITIAL checklist it
    fails every candidate in the episode including the real match. Measured on
    `full_sweep_qwen36_v2`: 26 of 41 runs whose description has a "target on X" phrase filed X under
    `on`, and 17 of those failed.

    The test is precise because of how these descriptions are built: the target is the head noun the
    description opens with, so a literal "on <X>" phrase attaches to the target and says the target
    rests on X. X therefore belongs under `below` (a support surface) or `next to` (a wall the target
    is mounted on) -- never `on`. The inverse wording, which IS `on`, never produces this phrase:
    "bed with a blue blanket" and "table with books on it" contain no "on <the other object>".

    Detection only, never reclassification -- same policy as the spatial-word check above.
    """
    hits = []
    lowered = " ".join(description.lower().split())
    for obj in other_objects:
        if not isinstance(obj, dict) or obj.get("key") != "on":
            continue
        noun = " ".join(str(obj.get("object", "")).lower().split())
        noun = re.sub(r"^(?:a|an|the)\s+", "", noun)
        if not noun:
            continue
        if re.search(rf"\bon\s+(?:a|an|the)?\s*{re.escape(noun)}", lowered):
            hits.append(obj["object"])
    return hits


def _validate(checklist: dict, description: str = "", other_objects: list | None = None) -> list:
    """`description`/`other_objects` are optional so the older single-argument call in
    scripts/run_atomicity_experiment.py keeps working -- it only exercises the spatial-word check."""
    problems = []
    for assertion, word in _target_assertions_with_spatial_words(checklist):
        problems.append(f"Target assertion {assertion!r} contains spatial word {word!r}")
    for object_text in _support_surfaces_misfiled_under_on(description, other_objects or []):
        problems.append(
            f"other_objects entry {object_text!r} filed under 'on', but the description says the "
            f"target rests on it -- belongs under 'below' (or 'next to' if it is a wall)"
        )
    return problems


def _merge_other_objects_into_checklist(other_objects: list, model_checklist: dict) -> dict:
    """checklist's non-"Target" entries are built here, in code, from other_objects -- never
    trusted from the model's own checklist output. Confirmed against a live server, 6 description
    types x 30 episodes: other_objects came back correct in nearly every case while the model's
    own checklist failed to carry those same entries through 35% of the time overall (90%+ on the
    two richest description types). "Target" is the one part of checklist still taken from the
    model, since it's the one part confirmed reliable (~0% failure on category/color-only
    descriptions, where there's nothing relational to get wrong).

    Dedup is exact-match after case/whitespace normalization (same style as checklist_update.py)
    -- a cheap safety net against two other_objects entries coincidentally naming the same thing
    under the same key, not expected to fire often.
    """
    merged = {}
    target = model_checklist.get("Target") or []
    if target:
        merged["Target"] = list(target)

    for obj in other_objects:
        bucket = merged.setdefault(obj["key"], [])
        normalized_existing = {re.sub(r"\s+", " ", a.strip().lower()) for a in bucket}
        normalized_candidate = re.sub(r"\s+", " ", obj["object"].strip().lower())
        if normalized_candidate not in normalized_existing:
            bucket.append(obj["object"])

    return merged


def _call_and_parse(llm_client: LLMClient, description: str, *, use_cache: bool | None) -> tuple:
    prompt, schema = build_prompt(description)
    result = llm_client.call(prompt, image=None, response_schema=schema, use_cache=use_cache)
    try:
        parsed = json.loads(result.text)
        other_objects = parsed["other_objects"]
        checklist = _merge_other_objects_into_checklist(other_objects, parsed["checklist"])
        return parsed["target_category"], parsed["target_phrase"], other_objects, checklist
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ContextParserError(f"context_parser: malformed response for description={description!r}: {result.text!r}") from e


def parse_context(llm_client: LLMClient, description: str) -> ParsedContext:
    # None = inherit the client's own caching setting, so a measurement run (VLM_USE_CACHE=0)
    # actually queries the model here too. The retry below still forces False explicitly.
    target_category, target_phrase, other_objects, checklist = _call_and_parse(llm_client, description, use_cache=None)
    problems = _validate(checklist, description, other_objects)
    retried = False

    if problems:
        # One retry, bypassing the cache -- a cache hit would just replay the same flagged
        # response verbatim. temperature=0.0 (llm.py) makes this a defense against
        # serving-time nondeterminism rather than a guaranteed fix, not a reclassification step.
        retried = True
        target_category, target_phrase, other_objects, checklist = _call_and_parse(llm_client, description, use_cache=False)
        problems = _validate(checklist, description, other_objects)
        if problems:
            print(f"[WARN] context_parser: validation still failing after retry for description={description!r}: {problems}")

    return ParsedContext(
        target_category=target_category,
        target_phrase=target_phrase,
        other_objects=other_objects,
        checklist=checklist,
        validation_problems=problems,
        retried=retried,
    )
