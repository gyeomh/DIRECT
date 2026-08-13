"""self_check (SPEC.md §7): image + (region, assertion) -> (evidence, verdict).

Polarity decided: (a) yes/no, framed as "does the image CONTRADICT the assertion" rather than
"does the image align with the assertion" -- answering "no" (contradiction) requires positive
visual evidence of falseness; "cannot confirm" is scored "yes" (not contradicted), never "no".
Polarity (b) (contradicts/consistent/cant_tell) is removed; this was the empirical decision from
the (now superseded) two-polarity comparison.

One call per statement, never batched. The prompt text below is used verbatim -- do not
paraphrase, shorten, or "clean up" the rule lists; the specificity is deliberate and is the whole
point of this design. SPEC.md §7 describes the design decisions (input/output format, region
derivation, prefix layout, experiment metric) without duplicating this prompt text, to avoid the
two ever drifting apart -- this module is the single source of truth for the literal wording.
"""

import json
from dataclasses import dataclass

from color_family import reconcile as reconcile_color
from llm import LLMClient

SELF_CHECK_PROMPT = """You verify a single claim against an image of an indoor scene.

The claim comes in two fields:
  region    — which part of the scene the claim is about
  assertion — what is being claimed about that region

Your job is NOT to judge whether the claim is a complete description.
Your job is to judge whether the image CONTRADICTS the claim.

=== CORE RULE ===

Answer "no" ONLY when the image gives positive visual evidence that the assertion
is false. If you simply cannot confirm the assertion, answer "yes".

Not confirmed is not the same as contradicted. When in doubt, answer "yes".

=== THESE ARE NOT CONTRADICTIONS — answer "yes" ===

1. INCOMPLETENESS. The assertion mentions less than what is in the region.
   Claim: "a window with plants"
   Image: window, plants, a chair, a towel rail
   -> yes. The claim never said those were the only things.

2. NAMING VARIANCE. Same object, different reasonable word.
   "wooden table" / butcher-block console      -> yes
   "cabinet" / cupboard                        -> yes
   "sofa" / couch                              -> yes
   "shelving" / open shelves                   -> yes

3. COLOR VARIANCE. Be generous here. Colors are named loosely, and lighting,
   white balance, and shadow shift them further. Read the stated color as a broad
   family, never as an exact swatch.
   All pale neutrals are ONE family — white, off-white, cream, ivory, beige,
   tan, taupe, sand, greige, light grey, and light natural wood all match each
   other.                                                          -> yes
   "beige" / wood, oak, pine, tan, cream, light brown              -> yes
   "wooden" / oak, pine, walnut, natural wood, beige or tan tones   -> yes
   "navy" / dark blue, deep blue, near-black blue                  -> yes
   "grey" / light grey, greige, silver, charcoal                   -> yes
   "black" / very dark grey, near-black brown, dark charcoal       -> yes
   Answer "no" on color ONLY when the two colors are unmistakably different
   hues — green vs red, green vs yellow, blue vs orange, red vs white.
   A difference of shade, tint, saturation, or warmth inside one family is
   never a contradiction. If you would need to compare paint chips to tell the
   two colors apart, answer "yes".

4. APPROXIMATE POSITION. Spatial terms are coarse.
   "left" covers the whole left portion — upper-left, lower-left, near or far,
   foreground or background. Same for the other directions.
   "on" means resting on the object's top surface.
   If the object is present anywhere plausibly in that region -> yes.
   "left"/"right" are always screen-left/screen-right as the image is viewed —
   the same convention a person points with while looking at the photo. Never
   mirror them as if the object in the region were a person facing the camera.

5. MULTIPLE OBJECTS IN THE REGION. If the region holds several objects and ANY
   ONE of them matches the assertion -> yes. The claim need not describe the
   most prominent one.

6. VAGUE QUANTITIES. "multiple items", "some plants", "a few books" match any
   count of two or more. Do not count precisely.

7. OCCLUSION OR CROPPING. The region is cut off by the frame, hidden behind
   something, too small, or too dark to read -> yes. You cannot contradict what
   you cannot see.

8. HEDGED WORDING. "appears to be", "looks like", "possibly" — treat as a weak
   claim. Only contradict if the image clearly shows otherwise.

9. SPEAKER FRAMING. Ignore phrases like "I can see", "In this image", "There is".
   Judge only the content that follows.

10. RELATED OBJECT, SIMILAR SHAPE, MATCHING COLOR. The claim names one specific
    object, and the region holds a different but closely related object — same
    general category, similar silhouette — AND the color the claim states matches
    what you see (matching per rule 3: same family is enough). Do not require the
    exact object name; a shared shape/silhouette plus matching color is enough.
    Claim: "a black music note decoration" / Image: a black music staff/sheet
    decoration, similar shape                                        -> yes
    Claim: "a brown wooden shelf" / Image: a brown wooden bookshelf   -> yes
    Claim: "a round mirror" / Image: a round clock, similar shape but
    a clock is not a mirror-family object                            -> no (rule 2 below)
    This only covers related objects with matching color and shape — it does not
    relax rule 1 (wrong attribute) or rule 2 (a genuinely different, unrelated
    object type still contradicts).

=== THESE ARE CONTRADICTIONS — answer "no" ===

1. WRONG ATTRIBUTE, CLEARLY VISIBLE. The object is plainly visible and the stated
   attribute is plainly different. For color this means a different hue family
   per rule 3 above — a different shade or tint of the same family is not enough.
   Claim: "the cabinet is navy blue" / Image: the cabinet is clearly white -> no
   Claim: "the white counter" / Image: the counter is beige            -> yes

2. WRONG OBJECT TYPE. The region clearly holds something of a different kind, and
   nothing matching the assertion is anywhere in that region. Does not apply when
   rule 10 above applies (related object, same shape, matching color).
   Claim: "a wooden table on the left" / Image: left side is a refrigerator and
   bare floor, no table of any kind -> no

3. REGION VISIBLY EMPTY. The region is fully visible and contains nothing, while
   the assertion says something is there.
   Claim: "there are items on the cabinet" / Image: the cabinet top is fully
   visible and completely bare -> no

4. WRONG SCENE TYPE. The assertion describes a room or setting incompatible with
   the image.
   Claim: "on the left of the bed there is a bathtub" / Image: a kitchen -> no

5. REGION CLAIMED EMPTY BUT IS NOT. The assertion says the region is empty or
   holds nothing, and the region is fully visible with clear objects in it.

=== OUTPUT ===

Write evidence first, then the verdict.

evidence — under 15 words. Name what you actually looked at in the region and
           what you saw there. Not a restatement of the claim.
verdict  — "yes" or "no".

{"evidence": "...", "verdict": "yes" | "no"}"""

# Field order matches generation order (SPEC.md §7 item 4: "evidence first — field order is
# generation order, so it acts as a short CoT"). Guided decoding pins verdict to the enum;
# evidence is free text.
SELF_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string", "maxLength": 200, "description": "Under 15 words. What was actually seen in the region."},
        "verdict": {"type": "string", "enum": ["yes", "no"]},
    },
    "required": ["evidence", "verdict"],
}


def build_prompt(region: str, assertion: str) -> tuple[str, dict]:
    # Order is [image] -> [fixed system prompt] -> [variable region/assertion] (SPEC.md §7 item
    # 5): the image goes first via llm.py's existing image-first message construction; within
    # the text block, SELF_CHECK_PROMPT is a byte-identical prefix across every single call
    # (not just same-image calls), with only this region/assertion suffix varying.
    variable_part = f"region: {region}\nassertion: {assertion}"
    prompt = f"{SELF_CHECK_PROMPT}\n\n{variable_part}"
    return prompt, SELF_CHECK_SCHEMA


@dataclass
class SelfCheckResult:
    evidence: str
    verdict: str  # "yes" | "no" | "PARSE_ERROR"
    color_reconciled: bool = False  # a "no" overridden as within-family color variance
    raw_verdict: str = ""  # the model's own verdict, kept when it was overridden


def self_check(llm_client: LLMClient, image, region: str, assertion: str) -> SelfCheckResult:
    prompt, schema = build_prompt(region, assertion)
    result = llm_client.call(prompt, image, response_schema=schema)
    try:
        parsed = json.loads(result.text)
        evidence, verdict = parsed["evidence"], parsed["verdict"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return SelfCheckResult(evidence="", verdict="PARSE_ERROR")

    # The one judgment not left to the model. Rule 3 above says the pale neutrals are one family;
    # the model reads colors correctly and then ignores that rule anyway, through three separate
    # rewrites of it (color_family.py's docstring has the measurements: 3/8 before and after each).
    # Its evidence names what it saw, so the naming is taken from there and the family call is made
    # in code. One-directional by construction: only "no" can be overturned, never "yes", so this
    # cannot introduce a false "no" -- it can only fail to reject a distractor on color.
    if verdict == "no" and reconcile_color(assertion, evidence):
        return SelfCheckResult(evidence=evidence, verdict="yes", color_reconciled=True, raw_verdict="no")
    return SelfCheckResult(evidence=evidence, verdict=verdict)


def is_failure(verdict: str) -> bool:
    """"no" is a contradiction (the real failure mode this experiment measures: the false-no
    rate, since ground truth is always "yes" here). A parse failure is also worth counting as a
    failure -- it's not a clean pass either, and silently treating it as "yes" would hide it.
    """
    return verdict in ("no", "PARSE_ERROR")
