# SPEC.md — Simple Spatial Graph + Checklist

Method specification. Read `ENV.md` first for the harness constraints.

Model: `Qwen/Qwen3-VL-30B-A3B-Instruct` for **all four modules and the local stand-in oracle**.
Instruct, not Thinking.

---

## 1. Modules

Four VLM-backed modules. No non-VLM components except the geometry fallback in §5.

| Module | Input | Output |
|---|---|---|
| `context_parser` | description | `target_category`, `target_phrase`, initial checklist (§10) |
| `self_check` | image + (region, assertion) | evidence + verdict |
| `zone_gen` | image + target | bbox, then relation set |
| `checklist_update` | current checklist + this round's (relation, answer) pairs | updated checklist (§12) |

`region_for`/`question_for` (§6) are shared, non-VLM string-template helpers used by several
modules — not counted as a fifth module, since they make no VLM call of their own.

---

## 2. Checklist

Two-level structure. Parent node = relation key. Children = **atomic** assertions.

```
Target:
  - it is navy blue
  - there are multiple items on it
left:
  - wooden tiles hung on a black wall
next to:
  - open shelving
```

**Children are stored as assertions only, never full sentences.** The region a child belongs to
is always derivable from its parent key plus the target phrase (`templates.region_for(parent_key,
target_phrase)` — §6), so a child never restates it: not "On the left of the cabinet there are wooden
tiles on a black wall", just "wooden tiles hung on a black wall"; not "The cabinet is black", just
"it is navy blue" (a pronoun is fine — `region` alone tells `self_check` what "it" refers to).
This guarantees the checklist path and the oracle-answer path produce identical region strings
for the same relation, since both go through the same `region_for` call.

**Atomic decomposition is required** regardless of this change. `self_check` verifies assertions
one at a time, so a compound assertion ("navy blue with multiple items on it") makes the verdict
ambiguous when one clause holds and the other is occluded. Split at parse time (`context_parser`)
and at update time (`checklist_update`).

Assertion style, not fragments — each child must be a standalone claim that can be checked against
an image on its own, once paired with its region.

The checklist **persists for the whole episode and only grows**. It resets when the episode ends.

---

## 3. Relation vocabulary

Closed enum, enforced with guided decoding:

```
left, right, above, below, left-top, right-top, left-bottom, right-bottom, on
```

Deduplication is key matching, so drift in this vocabulary silently causes repeat questions or missed
zones. `zone_gen` must emit only these values.

`next to` and any other non-directional relation produced by `context_parser` goes into the checklist
but is **excluded from dedup** — the description does not say which side, so it cannot be matched
against a directional key. Do not infer a direction for it from a candidate image: the description is
authoritative about the target, and a direction guessed from a distractor would contaminate it.

---

## 4. Episode flow

```
on episode start:
    target_category, target_phrase, checklist = context_parser(description)   # §10

for each candidate image:

    # Step 2 — check in advance. Only re-verifies EXISTING checklist assertions -- no new oracle
    # question is asked here, so a Step 2 failure has nothing new to feed checklist_update with.
    for each (parent_key, assertion) in checklist:     # one self_check call per assertion
        region = region_for(parent_key, target_phrase)   # §6 — same function both call sites use
        if self_check(image, region, assertion) fails:  # fails means verdict == "no"
            conclude mismatch                    # stop immediately, checklist unchanged
    # if the checklist is empty (category-only description), skip this step

    # Step 3 — zones
    bbox        = zone_gen.locate(image, target_category)     # §5-1: bare noun, not target_phrase
    relations   = zone_gen.zones(boxed_image, target_category)
    relations  -= relations already covered by checklist parent nodes

    # Step 4/5 — ask and verify. Record every answer as it arrives, BEFORE checking it, and stop
    # asking as soon as one fails self_check -- but do not discard that failing answer (§12, and
    # the design note right after this pseudocode).
    round_answers = []
    mismatch_found = false
    questions = [("Target", target-appearance question)] if this is the episode's first question else []
    questions += [(r, question_for(r, target_phrase)) for r in relations]
    for relation, question in questions:
        answer = ask(question)
        round_answers.append((relation, answer))
        if self_check(image, region_for(relation, target_phrase), answer) fails:
            mismatch_found = true
            break                                 # stop asking further questions this round

    # Step 6/7 — checklist_update always runs on whatever was actually asked this round, whether
    # this candidate matched or not (round_answers may be empty if Step 2 already broke earlier).
    if round_answers:
        checklist = checklist_update(checklist, round_answers)   # §12

    conclude mismatch if mismatch_found else conclude match
```

**Consequence worth knowing:** because covered relations are removed in Step 3, **each direction is
asked at most once per episode.** After candidate #1 asks `left`, every later candidate verifies
`left` through the checklist instead. Episode question count is roughly
`1 (target) + number of distinct relations encountered`.

**Every oracle answer feeds `checklist_update`, including the one that just failed.** The local
oracle stand-in only ever looks at the true target image (`oracle_stub.py`), matching the real
harness (ENV.md) — the oracle describes the actual target regardless of which candidate is on
screen. So the answer that triggers a mismatch is still a true fact about the target, and it is
the single most discriminative one available (it just eliminated a candidate). Discarding it, as
an earlier draft of this flow did, throws that signal away for every later candidate. `checklist =
checklist_update(...)` therefore runs after every candidate round that asked at least one new
question — match or mismatch — never only on the winning candidate.

---

## 5. zone_gen

Two VLM calls with a deterministic drawing step between them: **locate** (5-1) grounds the target
and returns a box; box-drawing (no VLM) burns that box onto a copy of the image; **zones** (5-2)
looks at the boxed image and picks which surrounding relations are worth asking about. Prompt text
for both calls is kept verbatim in `zone_gen.py` (`LOCATE_PROMPT`, `ZONES_SYSTEM_PROMPT`), not
duplicated here, same policy as `self_check` (§7) — the two must never drift apart.

**5-1 — locate.** Input: original image + `target_category`, the **category noun only**
(`"kitchen lower cabinet"`), never the full description — grounding with attributes ("navy blue …")
fails or returns a bad box on exactly the distractors that differ in that attribute.

Output: `{"boxes": [{"bbox_2d": [x0, y0, x1, y1], "label": "...", "note": "..."}]}`, guided
decoding requires exactly 4 integers per `bbox_2d`. Record `n_boxes` (how many boxes of the target
category came back) — more than one is an ambiguity signal to log; the candidate and the oracle may
be locked onto different instances.

Coordinate convention is assumed relative `[0, 1000]` (matching Qwen3-VL's documented behavior), but
this is **not yet verified against real model output** — `smart_resize` can shift the reference
frame the model actually used. Verify empirically tomorrow on the first 20-30 images
(`scripts/verify_zone_gen.py`) by drawing the box on the original and inspecting it; correct
`zone_gen._bbox_to_pixels` if the real convention differs.

**Box drawing (deterministic, no VLM).** Produces a separate image; the original is kept and used
everywhere else (a different image, a different hash, a separate prefix-cache entry — see
Caching below). Largest box only, even when several come back. Red outline, no fill, no label text
anywhere on the image. Line width = 0.5% of the image's short side, minimum 2px. Coordinates are
converted back to original pixel dimensions before drawing (§5-1's `[0, 1000]` frame scaled by the
original image's width/height, not any resized frame the model may have seen internally).

**5-2 — zones.** Input: boxed image + `target: {target_category}` as an appended field (same
call-signature pattern as `self_check`'s region/assertion suffix — §7). The checklist is **not**
passed in; dedup against it happens outside this module (post-processing, below) so the output
stays a pure function of the (boxed) image and target, and can be cached by image hash.

Region key vocabulary (§3): `left, right, above, below, left-top, right-top, left-bottom,
right-bottom, on`. `on` means the target's own top surface, not a directional area — the prompt
states explicitly that `on` must be included when objects rest on top of the target, since that is
not visible from the bounding box layout alone; this is a VLM call, not geometry, because 2D margins
cannot tell whether something in a direction is adjacent to the target or far behind it in 3D, and
the oracle's answer depends on that.

Output: `{"scene": "...", "regions": [{"note": "...", "key": "..."}]}`. Field order is generation
order — `scene` before `regions`, `note` before `key` within each item — both act as a short CoT
before the key is committed, same pattern as `self_check`'s evidence-before-verdict. Guided decoding
pins `key` to the region enum and caps `regions` at `MAX_REGIONS` via the schema's `maxItems`, in
addition to the prompt's own stated limit.

**Post-processing (outside the VLM call), in this order:**

1. Dedup the VLM-returned keys, keeping first occurrence, preserving order.
2. If that leaves an empty set (the VLM returned nothing), fall back to bbox margins (`x0`,
   `1000-x1`, `y0`, `1000-y1`) — any direction with room inside the frame, guaranteeing at least
   one. The fallback only ever produces the four base directions, never corner keys or `on`.
3. Remove keys already present as checklist parent nodes. An empty result **at this step** is
   legitimate and means "conclude match" — it must never trigger the fallback; the fallback is only
   for an empty VLM response, not for an empty post-checklist-dedup result.

**Geometry cross-check is logging only.** If the box touches an image edge (per the same
`[0, 1000]`-frame margin check used for the fallback) and the VLM still returned the direction on
that side, record it but keep the region — box placement may be slightly off, and whether to start
filtering these is a decision for later, from the logged frequency.

**Config:** `MAX_REGIONS = 5`.

**Caching.** The boxed image and the original are different images with different hashes and
separate prefix-cache entries. Both the 5-1 and the 5-2 result are cached per candidate against the
**original** image hash, so neither is recomputed across the repeated `ask_or_conclude` calls for
the same candidate: `locate` is memoized on `(original_image_hash, target_category)` directly;
`zones` is memoized on `(boxed_image_hash, target_category)`, which is equivalent in practice since
box-drawing is a deterministic function of the (cached) `locate` result — the same original image
and target always produce the same boxed image.

**Verify tomorrow, 20-30 training images** (`scripts/verify_zone_gen.py`, built now, not run for
real yet). Automatically measured: box count distribution, region key/count distribution, how often
the geometry fallback fires, how often a directional key whose edge the box touches still comes back
from the VLM (the edge-touch leak rate). Needs a human reading the dumped gallery (boxed images +
returned region lists): is the box on the correct object; is the "one box per run" rule respected —
this is the item that matters most, since a box around a single cabinet door makes the adjacent door
read as `left` and every directional question becomes meaningless; does `scene` correctly name the
edges the target touches. None of this is auto-classified.

---

## 6. Questions and region strings — shared templates (`templates.py`)

`region_for` (used by `self_check`, §7) and `question_for` (used to ask the oracle) are both
**shared, per-key template tables** in `templates.py`, not independently-formatted strings. This
replaced an earlier generic formula, `"{relation} of the {target}"`, that was only grammatical for
`left`/`right` — it produced "above of the cabinet" for `above` and worse for the corner keys.
Using one shared table for both paths is what guarantees the region `self_check` is told about
always matches what the oracle was actually asked about; independent formatting in each path risks
drifting apart the moment either one is edited.

Both tables key on the same 9 zone_gen relations (`left, right, above, below, left-top, right-top,
left-bottom, right-bottom, on`) and take `target_phrase` (not `target_category` — §10) as `{t}`:

```
REGION_TEMPLATES = {
    "left":         "left of the {t}",
    "right":        "right of the {t}",
    "above":        "above the {t}",
    "below":        "below the {t}",
    "left-top":     "above and to the left of the {t}",
    "right-top":    "above and to the right of the {t}",
    "left-bottom":  "below and to the left of the {t}",
    "right-bottom": "below and to the right of the {t}",
    "on":           "on top of the {t}",
    "next to":      "next to the {t}",
    "Target":       "the {t} itself",
}

QUESTION_TEMPLATES = {
    "left":         "What is on the left of the {t}?",
    "right":        "What is on the right of the {t}?",
    "above":        "What is above the {t}?",
    "below":        "What is below the {t}?",
    "left-top":     "What is above and to the left of the {t}?",
    "right-top":    "What is above and to the right of the {t}?",
    "left-bottom":  "What is below and to the left of the {t}?",
    "right-bottom": "What is below and to the right of the {t}?",
    "on":           "What is on top of the {t}?",
}
```

Every `QUESTION_TEMPLATES` entry has `" Can you describe the shape and color?"` appended.
`REGION_TEMPLATES` has two extra entries, `next to` and `Target`, that `QUESTION_TEMPLATES` does
not: those two region keys only ever arrive from `context_parser` (§10), never from `zone_gen`, so
no live question is ever generated for them — `question_for` raises `KeyError` if called with
either. `REGION_TEMPLATES` is the full 11-key checklist enum (`templates.CHECKLIST_KEYS`);
`QUESTION_TEMPLATES` is the 9-key subset that matches `zone_gen.REGION_KEYS` exactly. Tests assert
both correspondences directly — a key added to one enum without a matching template is a bug this
project cares about catching immediately, not discovering as a mid-episode crash.

**First question of the episode** (mandatory, before any relation question, and not part of either
template table above — it has no corresponding `zone_gen` relation):

```
Can you describe the {TARGET}'s location and visual appearance (e.g., color, shape, size).
```

Its answer is verified by `self_check` like any other — `region = region_for("Target",
target_phrase)` = `"the {target_phrase} itself"` — then stored under the `Target` node. This is
the only path that verifies the target object itself, so the atomic split of its answer matters.

**Relation questions:** `question_for(relation, target_phrase)`, for each relation `zone_gen`
returns. Never re-ask a relation already resolved this episode — the oracle is deterministic, so
it returns the identical answer for a wasted step.

---

## 7. self_check

Image + two fields → (`evidence`, `verdict`). **One call per statement**; never batch several
statements into one call.

**Decided: polarity (a), framed as contradiction-detection, not alignment.** "Does the image
CONTRADICT the assertion?" — `verdict = "no"` only when the image gives positive visual evidence
the assertion is false; if the assertion simply cannot be confirmed, `verdict = "yes"`. "Not
confirmed" is never treated as "contradicted". This supersedes the earlier two-polarity choice
(alignment-framed yes/no vs. `contradicts`/`consistent`/`cant_tell`) — see the experiment below.

**Input, never a pre-joined sentence:**

```
[image]

region: left of the kitchen lower cabinet
assertion: a large wooden table with a plant on it
```

Both the checklist path and the oracle-answer path go through this same signature, and both
derive `region` with the same shared function — `templates.region_for(relation_or_parent_key,
target_phrase)`, per its full template table (§6) — so they produce identical region strings for
the same relation:

- **Oracle answer** → `region = region_for(relation, target_phrase)`, `assertion` = the raw oracle
  answer, unmodified (no cleanup — the prompt's own incompleteness/naming-variance/speaker-framing
  rules handle that). Deterministic string assembly, no VLM call. The mandatory first/target
  question uses the sentinel `relation = "Target"`.
- **Checklist item** → `region = region_for(parent_key, target_phrase)`, `assertion` = the stored
  child text — already atomic, already region-stripped (§2).

**Prompt text:** kept verbatim in `self_check.py` (`SELF_CHECK_PROMPT`), not duplicated here so
the two can never drift. It defines the core rule (contradiction requires positive evidence),
nine "not a contradiction" cases (incompleteness, naming variance, color-family variance,
approximate position, multiple objects per region, vague quantities, occlusion/cropping, hedged
wording, speaker framing), and five "is a contradiction" cases (wrong attribute when clearly
visible, wrong object type, region visibly empty, wrong scene type, region claimed empty but is
not — this last one added to close the reverse-empty gap that "region visibly empty" doesn't
cover: an assertion claiming a region is empty when it plainly is not). Do not paraphrase, shorten,
or trim that text when touching this module — the specificity is deliberate.

**Prompt layout:** `[image] → [fixed system prompt] → [variable region/assertion]`. The image and
the entire fixed system-prompt text never change between calls; only the region/assertion suffix
does — making the shared portion a byte-identical prefix across every single `self_check` call
(not just same-image ones), which is what makes prefix caching (§8) actually pay off.

**Output:**

```json
{"evidence": "...", "verdict": "yes" | "no"}
```

`evidence` first — field order is generation order, so it acts as a short chain-of-thought before
the verdict: name what was actually looked at in the region and what was seen there, under 15
words, not a restatement of the claim. Guided decoding pins `verdict` to the `yes`/`no` enum.

**Experiment measures one thing, not a comparison.** Ground truth is `yes` for every case — the
assertion is checked against the exact image it was derived from (the local oracle stand-in only
ever sees the target image), so a truthful answer can never actually be contradicted by it. The
number that matters is the **false-`"no"` rate**: report it overall and per question type
(target-appearance vs. each of `left`/`right`/`above`), and dump every `no` — image path, region,
assertion, evidence, verdict — for manual inspection. Still do not auto-classify failures; reading
the dump is a human step.

**Deferred, not reopened by this decision:** logprobs on the verdict token, and a follow-up
question when the model is unsure — construction of that follow-up question is still unresolved.
Do not implement until the base (single-call) path has been run for real.

---

## 8. Performance

`self_check` is called once per checklist statement per candidate, and the checklist grows through the
episode — roughly 40–50 `self_check` calls per episode, plus `zone_gen`, per-answer checks, and
updates. Order 60–70 VLM calls per episode against a 600 s budget.

**Prefix caching is what makes this fit.** All `self_check` calls for one candidate share the same
image, so place the **image at the very front of the prompt**, identically across calls, and enable
prefix caching in vllm. The image prefill is then paid once per candidate instead of once per
statement. Without this the per-statement design does not fit in 600 s.

Other serving notes: prefer the official FP8 quant; `--limit-mm-per-prompt.video 0`;
`--max-model-len 8192`–`16384` (default 262K is unnecessary, and the freed memory goes to KV cache);
`guided_json` for `zone_gen` and `self_check` outputs. **`max_pixels` is the highest-leverage latency
knob** — sweep it early.

Run the stand-in oracle on a separate GPU or server; on the same device its inference competes with
the modules and the timing measurements will not reflect the real setup.

---

## 9. Build order

Do not build the full loop first.

1. **ENV verification** — confirm `ENV.md` against the installed repo, apply `patches/`, get upstream
   running end to end with `MockOracle`.
2. **`llm.py`** — one vllm client for all calls. Guided JSON, disk cache keyed by
   `sha256(model | prompt | image_hash)`, timing accumulation. Confirm prefix caching actually
   engages. Sweep `max_pixels`. Measure per-call latency.
3. **Each module in isolation**, with a small script per module that runs it on a handful of images
   and prints the output for inspection. Specifically: settle the `self_check` polarity here, and
   confirm `zone_gen` never emits an out-of-enum relation.
4. **Episode loop.**
5. **Evaluation**, logging error direction (`false_match` / `false_mismatch` / `truncated`), split by
   description type.

Step 3 matters most — polarity errors and enum drift are visible per module and nearly invisible once
buried in the loop.

---

## 10. context_parser

Input: the description string. Text-only call — **no image**. Runs once per episode, at episode
start. Prompt text is kept verbatim in `context_parser.py` (`CONTEXT_PARSER_PROMPT`), same policy
as `self_check` (§7) and `zone_gen` (§5).

**The core failure this module's design defends against, found against the real model:** its own
first-draft prompt taught the wrong behavior by example. `"cabinet with brass handles"` →
`Target: "it has brass handles"` and `"bed with a blue blanket"` → `on: "a blue blanket"` share
the identical surface form (`"with X"`), so a model generalizing from examples learns `"with X" →
Target: "it has X"` regardless of the rule text saying otherwise — a rule loses to an example that
contradicts it. Confirmed on the real model: `"White bed with a blue blanket next to a nightstand"`
came back as `{"Target": ["it is white", "it has a blue blanket", "it is next to a nightstand"]}`
— every relational fact reworded into a `Target`-shaped sentence and dumped under `Target`, and
`"the bed is next to a nightstand"` reads as perfectly natural English, so nothing about the
sentence's fluency signals that it's misfiled. This is a **silent** failure: the JSON is valid,
parses fine, and is semantically not obviously wrong — invisible unless someone reads the output
by hand.

**First fix attempt: `other_objects`, a field the model must fill in before `checklist`.** Field
order is generation order — same pattern as `self_check`'s evidence-before-verdict and `zone_gen`'s
note-before-key — so every object other than the target gets named, with the wording that links
it to the target (`cue`) and the region key that wording implies, *before* the model is allowed to
write the checklist. This alone was not enough. Confirmed against a live server, 6 description
types × 30 episodes: `other_objects` itself came back correct (object/cue/key all right) in nearly
every case, but the model's own `checklist` still frequently failed to carry those same entries
through — sometimes dropping them entirely, sometimes padding `Target` with repetitive filler
instead. The failure rate tracked description richness almost exactly:

| description type | flag rate |
|---|---|
| category, color (no relation to get wrong) | 0% |
| context, color_feature (one relation) | 10–17% |
| color_context, color_context_feature (color + context clauses combined) | 90–93% |

**Second, working fix: stop asking the model to write the relational entries twice.**
`checklist`'s non-`"Target"` entries are now built **mechanically in code**
(`_merge_other_objects_into_checklist`) directly from `other_objects` — never taken from the
model's own `checklist` output for any key other than `"Target"`. The model's `checklist` field is
now used for `"Target"` only, the one part already confirmed reliable (~0% failure on
category/color-only descriptions, where there's nothing relational to get wrong). Any non-`Target`
key the model writes anyway is discarded, not merged in — trusting it selectively would just
reintroduce the same inconsistency in a different shape.

**Output, four fields, in this order:**

- `target_category` — the bare category noun phrase, no attributes. Feeds `zone_gen.locate` (§5-1)
  — grounding must use this, never `target_phrase`, since attributes make grounding fail on exactly
  the distractors that differ in that attribute.
- `target_phrase` — the target with attributes that belong to the target itself only, no clauses
  about other objects. Feeds `region_for`/`question_for` (§6) as `{t}` everywhere else in the
  pipeline.
- `other_objects` — `[{"object": ..., "cue": ..., "key": ...}, ...]`, every object other than the
  target the description mentions. `key` is enum-pinned to the 10-key relation vocabulary
  (`templates.CHECKLIST_KEYS` minus `"Target"` — an object other than the target is never itself
  the `Target` node). Field order within each item (`object`, then `cue`, then `key`) mirrors the
  outer field order: name the object and the literal linking wording before committing to a key.
  This is the field the merge trusts.
- `checklist` — the model writes `"Target"` only (its own attributes and integral parts); every
  other key is filled in afterward from `other_objects`, not asked of the model at all. `next to`
  is used only when the description gives no side; a description with no target-only attributes
  produces an empty (or `Target`-only) model checklist, which is fine — the checklist can still end
  up with real content once the merge adds `other_objects`' entries.

**`=== PARTS vs SEPARATE OBJECTS ===`** replaces the old "parts go under Target" rule with a test
that doesn't depend on wording: *could you carry it into another room and leave the target
unchanged?* No → integral part → `Target` (handles, drawers, legs, doors, frame, upholstery). Yes
→ separate object → a region key, i.e. `other_objects` (blankets, pillows, plants, books, dishes,
mirrors, sinks, tables). Both can appear as `"with X"` — the wording never decides it, only the
test does.

**`=== NEVER PUT A RELATION IN A TARGET ASSERTION ===`** is a direct, checkable constraint: a
`Target` assertion must not contain a spatial word (`next to, beside, under, beneath, below,
above, on, on top of, behind, in front of, near`). This is the one failure mode left uncovered by
the code-level merge (the merge fixes `other_objects` never reaching the checklist; it does
nothing about a relation *also* getting reworded into a `Target` sentence), and is exactly what the
validator below still checks for.

Assertions follow the same atomic, region-stripped style as everywhere else in the checklist (§2):
a short phrase, not a sentence, that does not restate the region or the target. No invention —
nothing the description doesn't state.

**Validator: detection only, never reclassification, and now single-purpose.** One check remains
(the "does every `other_objects` entry reach the checklist" check is gone — the merge makes that
true by construction, so checking it would never fire): does any `Target` assertion contain a
spatial word? If it does, `parse_context` retries **once**, bypassing the disk cache (a cache hit
would just replay the identical flagged response — `llm.py`'s cache key has no temperature
component, and `temperature=0.0` here, so this is a defense against serving-time nondeterminism,
not a guaranteed fix — confirmed on the real model that the retry recovers **0** of the cases it's
triggered on, at temperature 0). If the retry is still flagged, `parse_context` logs a `[WARN]`
line and **returns the result anyway** — `ParsedContext.validation_problems` carries whatever was
found, so a caller can inspect or count it, but nothing here silently rewrites the model's output.

**maxItems=8** on `checklist`'s per-key arrays and on `other_objects` itself: confirmed against a
live server that an unbounded array lets the model enter a token-repetition loop (the same
assertion string appended dozens of times) that a strict `json_schema` grammar does nothing to
stop, since an unbounded array is a valid completion at every generation step — it runs until
`max_tokens` truncates the response mid-string and the whole thing fails to parse (3/11 hand-picked
descriptions crashed this way before the cap was added).

---

## 11. Open items

- Logprob tie-break and follow-up question construction (§7) — deferred.
- `max_pixels` value.
- Whether the edge-touch leak logged by `zone_gen`'s geometry cross-check (§5-2) is frequent enough
  to warrant filtering — decide from the logged frequency once more real numbers exist (6/25 in
  the first real sample; see the real-server section below).
- **Resolved.** `context_parser`'s relation-vs-`Target` filing problem (§10): the `other_objects`
  field alone (prompt-level fix) cut nothing on its own — confirmed at 6 types × 30 episodes, 35%
  overall flag rate, 90%+ on the two richest description types. Moving checklist's non-`Target`
  entries out of the model's hands entirely (built in code from `other_objects` instead) dropped
  that to **0.6% (1/180)** on the same real sweep, and the one remaining flag is the expected false-
  positive class of the word-list validator (`"a knight on horseback"` — the target's own subject
  matter, correctly kept in `Target`, incorrectly flagged for containing "on") rather than a real
  filing error. No further action planned unless this class of false positive turns out to be
  more common than this one sample suggests.
- `checklist_update` misfiled one case in five hand-built examples: the mandatory first/target
  question's own answer landed under `on` instead of `Target` (rule 2, "keep the asked key," not
  followed for that one case). The other four (atomic splitting, empty-region wording, duplicate
  skipping, hedge preservation) all matched the spec exactly. Isolated so far — re-test with more
  cases before deciding whether this needs a fix.

**Resolved by this decision:** `self_check` polarity is (a), contradiction-framed yes/no (§7). The
`Target` node needs no special handling in Step 2 beyond `region_for`'s own `"Target"` sentinel —
same call shape as any relation node, just a different region string.

**`zone_gen` built (§5):** `locate` (5-1), deterministic box-drawing, and `zones` (5-2) are
implemented, tested against `FakeVLM`, and cached per-candidate against the original image hash.

**`region_for`/`question_for` fixed and shared (§6):** the old generic `"{relation} of the
{target}"` formula (ungrammatical for everything but `left`/`right`) is replaced by the
per-key `REGION_TEMPLATES`/`QUESTION_TEMPLATES` tables in `templates.py`, used by both the
checklist path and the question path so they can never diverge.

**`context_parser` built (§10):** `target_category`/`target_phrase`/`checklist` parsing is
implemented and tested against `FakeVLM`.

**`checklist_update` built, all four modules now exist (§12):** the last module is implemented and
tested against `FakeVLM`. Two decisions made along the way:

- **Every oracle answer from a round feeds `checklist_update`, including the one that triggered a
  mismatch** (§4's "every oracle answer feeds `checklist_update`" note) — the oracle stand-in only
  ever looks at the true target, so a failing answer is still a true, and highly discriminative,
  fact about it. An earlier draft of the episode flow discarded it; that was a bug, not a design
  choice, and is now fixed in the pseudocode.
- **`self_check`'s contradiction list gained a fifth case** (§7): "region claimed empty but is
  not" — the assertion says a region is empty/holds nothing while the region is fully visible with
  clear objects in it. The original four cases only covered the reverse (assertion claims
  something is there; region is bare); this closes a miss (not a false death), found while
  reasoning about `checklist_update`'s empty-region rule (rule 7: an oracle answer describing an
  empty region is stored verbatim as `"nothing visible"`, which needs its own contradiction check
  on later candidates).

**Episode loop built (§13), build order step 4 done:** `DirectionMethodQuestioner` wires all four
modules together and passed a full dry run — 167/167 training episodes against `FakeVLM`, zero
crashes, zero invalid actions, zero ENV.md §6 bug-3 discards. Under `FakeVLM` every candidate is
judged "match" on the first try (the fake self_check backend can never actually detect a
mismatch), so every one of the 167 episodes terminates after exactly one candidate and one
question — expected and consistent with "the accuracy number is meaningless" (§ build order step
4), not a sign anything is broken. Multi-candidate continuation, relation-queue draining, budget
stops, and checklist growth across candidates are exercised instead by `tests/test_questioner.py`,
where a scripted backend controls exact verdict sequences.

**First real-server verification pass (GPU 1, `vllm==0.15.0`, `Qwen/Qwen3-VL-30B-A3B-Instruct`,
port 8002 — 8000/8001 were already in use by unrelated processes on this shared machine):**

- **`guided_json` fix, confirmed and applied.** `extra_body={"guided_json": schema}` is silently
  ignored by this vllm version — the server logs `WARNING ... fields were present in the request
  but ignored: {'guided_json'}` on every call, no exception. Short prompts happened to get valid
  JSON anyway (the model complied without being forced to); `self_check`'s long prompt did not —
  the model mimicked the prompt's own "evidence — ... / verdict — ..." field-description
  formatting instead of emitting JSON, so 100% of calls came back as `PARSE_ERROR`.
  `extra_body={"response_format": {"type": "json_schema", "json_schema": {"name": "response",
  "schema": schema, "strict": True}}}` **is** honored (no "ignored" warning, valid JSON on both
  short and long prompts) and is now what `llm.py`'s `VLLMBackend.generate()` sends.
- **`zone_gen`'s bbox `[0, 1000]` coordinate convention is confirmed correct** on the real model —
  drawn boxes land squarely on the target at the right scale across every image checked by hand,
  including the "one box per run" cases that matter most (a double-vanity cabinet, a 4-drawer
  dresser, a glass-front display cabinet all box as one unit, not per-door/per-drawer). `verify_zone_gen.py`'s
  real numbers on 25 images: box count `{1: 24, 2: 1}`, region count `{3: 1, 4: 10, 5: 14}`,
  0/25 fallback fires, 6/25 edge-touch leaks (a direction whose edge the box touches still came
  back from the VLM) — logged per §5-2's design, not yet acted on.
- **`self_check` false-`"no"` rate: 4.2% (5/120)** once `guided_json` was fixed (was 100%
  `PARSE_ERROR` before the fix, 0% after on the same statements). The 5 real failures, read by
  hand: two color-adjacent misses (oracle said "white frame," `self_check` saw "silver frame," on
  a mirror, twice), one shape miss ("curved top edge" vs. a straight one), one **left/right flip**
  (oracle: "on the left," `self_check`: "on the right of the monitor" — the most concerning of the
  five, since left/right is a hard binary, not a coarse judgment call), and one `above`/`on`
  wording clash (oracle said "above the carpet" meaning "resting on it"; `self_check` read `above`
  literally and called it a contradiction — suggests `above` may not read naturally for floor-level
  targets like carpets/rugs, worth a design note if carpet/rug episodes are common).
- **`context_parser`'s checklist arrays needed a `maxItems` cap.** Without one, 3/11 hand-picked
  descriptions caused the model to enter a token-repetition loop inside one checklist array (the
  same assertion string appended dozens of times) — a strict `json_schema` grammar does nothing to
  stop this, since an unbounded array is a valid completion at every step, so generation ran until
  `max_tokens` truncated the response mid-string and it failed to parse. `maxItems: 8` on every
  checklist value array (both `context_parser.py` and `checklist_update.py`, same schema shape) fixed
  the crash. It did not fix the underlying content problem — see Open Items (§11) for the
  relation-vs-`Target` key-filing issue this surfaced.
- **Prefix caching confirmed engaging**: `check_prefix_caching()` on a real image, two suffixes —
  first call TTFT 0.47s, second (same image, different suffix) TTFT 0.11s, about 4x faster,
  consistent with the image's KV cache being reused.

**Full real-VLM sweep (`scripts/run_full_sweep.py`): all 167 episodes × all 6 description types,
1002 episode-runs, port 8002.** ThreadPoolExecutor (12 workers), one fresh `LLMClient` + `QAEnv`
per task, resumable (a task whose output JSON already exists is skipped). Rich per-run JSON logs
(`context_parser` result, checklist before/after per candidate, every `self_check` interaction
with region/assertion-or-question/answer/evidence/verdict, `zone_gen` bbox/scene/zone_list,
resolved image paths) saved to `/data/gyeom/coin_challenge/direction_method_logs/full_sweep_v1/`,
browsable via `scripts/serve_viewer.py` (a local zero-dependency stdlib HTTP server + single-page
viewer at `viewer/index.html`).

Final result after the fixes below: **1002/1002 runs completed, zero crashes, zero invalid
actions** (944 `terminated`, 58 `discarded_env_bug` per ENV.md §6 bug 3). Full-success rate
519/1002 (51.8%), fairly even across description types (45.5%–55.7%; `category`-only lowest, as
expected — least disambiguating information).

Two real bugs surfaced by running at this scale (both fixed, both with regression tests):

- **`LLMClient._store_cache`'s tmp-file suffix was keyed on `os.getpid()` alone**, which every
  thread in one process shares. Two threads racing on the same cache key (identical prompt+image
  — realistic once many episode-runs share a thread pool) collided on one tmp path; the loser's
  `tmp_path.replace()` raised `FileNotFoundError` once the winner had already renamed it away.
  Fixed by keying the suffix on `threading.get_ident()` too.
- **`maxItems` alone was not sufficient to stop the token-repetition-loop crash** (§10/§11's
  earlier fix). 9/1002 runs (0.9%) crashed the same way but one level deeper: the model degenerated
  by repeating the same clause *within a single string element* (`checklist_update`'s `additions`,
  mainly `color_context`/`color_feature`/`color_context_feature`), not across array items — still
  running until `max_tokens` truncated the response mid-string, since `maxItems` never bounds a
  string's own length. Fixed with `maxLength: 200` on every free-text string field across all four
  modules (context_parser, checklist_update, self_check, zone_gen), plus a `maxItems` cap on
  zone_gen's `LOCATE_SCHEMA` "boxes" array, which had no bound at all. Retrying the 9 crashed runs
  after the fix still reproduced 6/9 identically at first — `LLMClient`'s cache key deliberately
  excludes `response_schema` (this doc's own formula in §9), so a prompt whose bad response was
  already cached returns the same truncated text regardless of the new schema. Purged the 9
  oversized/unparseable cache entries by content (text over 500 chars that still fails
  `json.loads`) and reran; all 1002 runs then completed clean.

---

## 12. checklist_update

Text-only call, no image. **One call per candidate-image judgement** — every `(relation, answer)`
pair gathered for that candidate in one round goes into a single call, never one call per pair.
Prompt text is kept verbatim in `checklist_update.py` (`CHECKLIST_UPDATE_PROMPT`), same policy as
the other three modules.

**Input** (two labeled sections, not a single blob):

```
current checklist:
{key}: {assertion}
...

new answers:
{key}: {raw oracle answer}
...
```

The VLM sees the *entire* current checklist (one `{key}: {assertion}` line per existing entry,
however many that is) plus every new answer gathered this round, and is asked to extract only what
should be **added** — it never re-emits the existing checklist, and it is not the thing that
merges. Assertion style, atomicity, framing-stripping, and duplicate-skipping rules mirror
`context_parser`'s (§10) rules 1/3/4/8 wording, adapted for update rather than initial extraction;
rule 2 ("keep the asked key") is specific to this module — file an assertion under the key the
*question* asked about even if the answer's wording suggests a different position, since the
oracle is answering the question that was actually asked, not re-describing the whole scene from
scratch. Rule 7 (empty regions) fixes the canonical wording: an oracle answer describing an empty
region becomes exactly the single assertion `"nothing visible"` under that key — this is also why
`self_check` needed the fifth contradiction case above.

**Output:** `{"additions": {key: [str]}}`, keys pinned to the same 11-key enum as `context_parser`
(§10) via `additionalProperties: false` plus an explicit property per key. Same empty-list
normalization as `context_parser`: an entry with an empty assertion list is dropped.

**Merge happens in code, never in the VLM:**

1. Append `additions[key]` to the existing list under `key`.
2. Never modify, reword, reorder, or delete an existing assertion — the merge only ever appends.
3. Safety-net dedup on exact match, after normalizing case, surrounding whitespace, and internal
   whitespace runs. The prompt's own rule 8 is the primary defence against duplicates; this only
   catches the literal repeats that rule misses (checked against both the pre-existing assertions
   for that key and any new assertions already added earlier in the same merge).
4. Assert the post-merge checklist is a superset of the pre-merge one — concretely, every
   pre-existing per-key assertion list must survive as an exact, unreordered prefix of the
   post-merge list for that key. This should be unreachable given rules 1-2 above; it exists as a
   runtime guard against a future bug in the merge code, not a real failure mode today.

---

## 13. Episode loop (`questioner.py`, `DirectionMethodQuestioner`)

Wires the four modules together into `QuestionerInterface.ask_or_conclude`. No new prompts — every
VLM call goes through `context_parser`, `self_check`, `zone_gen`, or `checklist_update`.

**State.** Built once in `__init__` from `info["target_description"]` via `context_parser`:
`target_category`, `target_phrase`, `checklist`, plus whether the mandatory first question has
been asked, and step/time counters. `info` itself is **never stored** — only
`target_description` is read out of it — so there is no path back to `info["task_image"]`
(ENV.md §5's leaked ground-truth-image key) from this object, regardless of its exact name. A
test greps the whole package for that literal key name and fails if found, per ENV.md §5's own
instruction. Reset only by constructing a new questioner (matches ENV.md §4: the questioner is
built once per episode and never reset between candidates — this persistence is what makes the
accumulating checklist work).

**Per-candidate cache, keyed by image hash.** `ask_or_conclude` is called repeatedly for the same
candidate (ENV.md §4): once per question, then once more to receive the answer, then again for
the next question or the conclusion. Each call hashes the incoming image and compares to the
previous hash to detect a transition. On a transition: a fresh per-candidate cache is built
(bbox, boxed image, zone list, the pending question queue, answers received so far); nothing is
recomputed within a candidate.

**Control flow, per call:**

1. Hard-stop check (see Budget) — absolute cutoff, checked before anything else.
2. If a question is pending from last call, process its answer: `self_check(region_for(relation,
   target_phrase), answer)`. A failure concludes mismatch immediately.
3. If this is a new candidate: run every existing checklist statement through `self_check` (Step
   2) — one call per statement, first failure concludes mismatch immediately, skipping `zone_gen`
   entirely. An empty checklist skips this phase. If it passes (or was empty) and the budget still
   allows asking, run `zone_gen` (`locate` then `zones`), dedup the returned relations against
   checklist parent keys, and build the question queue — the mandatory target-appearance question
   goes first if it hasn't been asked yet this episode.
4. Ask the next queued question if the budget allows; a `no` verdict on a later answer or an
   emptied queue with pending relations left unaddressed both conclude mismatch. A genuinely empty
   queue concludes match.
5. On conclusion, run `checklist_update` once with every `(relation, answer)` pair collected for
   this candidate this round.

**Ordering decision: `checklist_update` runs synchronously, before returning the conclusion** —
not lazily at the start of the next candidate. In this harness the two are equally slow in total
wall-clock terms either way, since the call blocks regardless of which single step absorbs its
latency; synchronous is strictly simpler, with no real latency cost, and avoids carrying "pending
update" state across the candidate boundary. The one exception is the hard-stop path (below),
which skips it entirely, honoring "zero further calls" literally even if the current candidate
already has some not-yet-merged answers.

**Every oracle answer feeds `checklist_update`, including the one that just failed self_check** —
see the design note in §4. This is true for every mismatch conclusion except the hard-stop one.

**Budget.** The questioner is never told how many candidates its episode has (range 1-7, mean 3.16
across `episodes_train.jsonl`) — `ASSUMED_MAX_CANDIDATES = 7` is used as a fixed, conservative
stand-in. Two independent checks, both folded into one `_can_ask_more_questions()` gate:

- **Soft stop, 0.60 × 600s:** past this point, stop asking NEW oracle questions — for the current
  candidate (mid-queue) and every future one. Step 2 (already-necessary, no oracle round-trip)
  still runs; if it passes with nothing left to check, the candidate concludes mismatch rather
  than spending time on `zone_gen` for relations that would never get asked anyway.
- **Step reserve:** asking one more question must still leave at least one step for this
  candidate's own conclusion, plus one for each candidate still assumed to come after it
  (`ASSUMED_MAX_CANDIDATES - candidates_seen`) — a margin against the harness's hard 60-step cap
  (ENV.md §3), framed as "reserve one step per remaining candidate."

**Hard stop, 0.85 × 600s:** an absolute cutoff, checked first on every single call regardless of
phase — zero further VLM calls, immediate conclusion, no `checklist_update`. Any answers gathered
for the current candidate before this point but not yet merged are dropped.

**On budget exhaustion mid-candidate (soft, hard, or step-reserve), always conclude mismatch** —
matches are rarer than mismatches given the candidate counts above, and `n_successes` (not
calibration) is the metric, so guessing the majority class under uncertainty is the better bet.

**Never return both `question` and `conclusion` as `None`.** Upstream's own `_validate_action` has
an operator-precedence bug that lets exactly this case through silently (ENV.md §4/§5) — the
harness will not catch it. Every return statement in `ask_or_conclude` funnels through one of two
helpers (`_ask` / `_conclude`), each of which sets the other field to `None` unconditionally and
asserts the invariant, so the failure mode is structurally hard to reintroduce by accident.
`tests/test_questioner.py` checks this invariant on every single call across every scenario, not
just once.

**Logging.** Per candidate: questions asked, `self_check` calls made, verdicts (in call order),
conclusion, reasoning, elapsed time. Per episode: candidate count, question/`self_check` call
totals, whether soft/hard stop fired, and how many conclusions were budget-forced rather than
genuine. This is what tomorrow's real-server diagnosis runs read.

**Dry run.** `scripts/run_full_loop_dry_run.py` drives `DirectionMethodQuestioner` against the
real `env.QAEnv` for all 167 training episodes, against `FakeVLM`, replicating `eval_model.py`'s
own workaround for ENV.md §6 bug 3 (discard, don't crash, if a conclusion should have advanced the
candidate but didn't). Result: 167/167 constructed and completed cleanly, zero crashes, zero
invalid actions, zero discards. See the note in §11 for why every episode resolves in exactly one
candidate under `FakeVLM`, and why that's expected rather than a red flag.
