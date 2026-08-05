# CoIN Challenge 2026 — Questioner Implementation Spec

Target repo: `https://github.com/e-zorzi/coin_challenge`
Deliverable: a `QuestionerInterface` subclass that maximizes correct conclusions while asking few questions.

This spec is written to be implemented directly. Where a value is a tuning knob it is marked
`[CONFIG]` and must live in `config.yaml`, never hardcoded.

---

## 0. Ground truth about the environment

These facts were read from `env.py`, `Questioner.py`, and `eval_model.py`. Do not re-derive them;
do not contradict them. If the upstream repo changes, re-verify before editing this spec.

### 0.1 Episode structure

- One episode = one **fixed target image** + one **description** (`info["target_description"]`) +
  an ordered list of **candidate images** (6 in the published example).
- `reset()` opens the target image once. It never changes during the episode.
- The observation is always `distractors[current_distractor_idx]`.
- The index advances **only** when your conclusion equals the ground-truth `match` label.
- A wrong conclusion **terminates the episode immediately**. So does a correct conclusion on the
  final candidate.
- `match` is a per-candidate label reused across all six description types. It therefore means
  **"this image shows the same object instance as the target"**, not "this image satisfies the
  description". With `--description-type category` several candidates satisfy the description and
  are still `False`.

### 0.2 The oracle

- Sees **only the target image**. It does not see the candidate you are looking at, does not see the
  description, and has no notion of which object is "the target".
- Prompt wrapper: `"You are a faithful assistant. Answer correnctly to the following question based
  on the above image: {QUESTION}. Be concise (under 15 words)."`
- Default model `gemini-3-flash` at `temperature=1e-6`. **Effectively deterministic**: re-asking an
  identical question returns an identical answer. Never re-ask to "average out" noise.
- Answers are capped under 15 words.
- `ask_oracle` is wrapped in `@retry(stop_max_attempt_number=5, wait_fixed=80000)`. One flaky call
  costs 80 s of episode wall clock; four failures cost 320 s.

### 0.3 Budgets (shared across the whole episode, not per image)

| Budget | Value | Enforcement |
|---|---|---|
| Steps | `max_steps = 60` | `_is_truncated()` |
| Wall clock | `_MAX_TIME_ALLOWED = 600` s from `reset()` | `_is_truncated()` |

Every action is one step, question or conclusion. Truncation is not penalized directly — the loop
simply exits and remaining candidates are never judged, freezing `n_successes`. **Truncation is an
accuracy loss.**

### 0.4 Reward (for reference; the leaderboard is accuracy-weighted)

`+10` correct conclusion, `-10` wrong conclusion, `-1` per question. Nineteen questions are cheaper
than one flipped conclusion. Optimize accuracy first, question count second.

### 0.5 Interface contract

```python
class QuestionerInterface(ABC):
    def __init__(self, info, *args)          # info["target_description"] is the description
    def ask_or_conclude(self, observation)    # -> dict(question=..., conclusion=..., reasoning=...)
    def add_answer(self, answer)              # harness calls this after each question
    def reset_questions(self); def reset_time()
```

- `observation = {"image": np.ndarray (H,W,3) uint8, "answer": str | None}`
- Return exactly one of `question` / `conclusion` as non-`None`. `conclusion` is `0` or `1`.
  **Never return both `None`** — `_validate_action` has a precedence bug that lets it through, and
  the env then computes `reward = None`.
- Always compare with `is not None`, never truthiness: `conclusion=0` is falsy.
- `eval_model.py` reads `questioner.n_questions` and `questioner.time_required` for logging. You must
  maintain both. `env.n_questions` is never incremented upstream and is always 0 — ignore it.
- `eval_model.py` instantiates the questioner **once per episode** and never calls
  `reset_questions()` between candidates. State persists across candidates by design.

### 0.6 Integrity constraints — non-negotiable

- `info` returned by `reset()` contains `info["task_image"]`, which **is the target image PIL
  object**. This is an upstream leak. **The questioner must never read this key.** Add an assertion
  in `__init__` that pops/discards it, and a unit test that fails if any module references it.
- Do not condition on candidate position or on episode index. In the published example the
  `match=True` candidate is last; the organizers state the held-out set is regenerated. Using the
  base rate over labels is fine (see §5.4); using the *index* is not.
- `n_questions` is self-reported to the logger. Report it exactly.

### 0.7 Known upstream issues to handle locally

Keep upstream files unmodified; put fixes in `patches/` with a `README` explaining each.

1. `eval_model.py` prints `info['task_description']`, but `_get_info()` returns
   `target_description`. Raises `KeyError`. Patch to `target_description`.
2. `eval_model.py` loads `f"QA_eval/episodes_{run_type}.jsonl"`; the repo ships
   `episodes_train.jsonl` at the root. Patch the path or symlink.
3. Known upstream bug: after a conclusion the observation sometimes does not switch on the training
   set, and the eval loop `break`s, discarding the episode. Training metrics are therefore computed
   over a subset. Log how many episodes are discarded and report accuracy with that denominator
   stated explicitly. Never tune on fewer than `[CONFIG] min_eval_episodes: 100`.

---

## 1. Design principles

These are the reasons behind the structure below. Follow them when resolving anything this spec
leaves ambiguous.

1. **The persistent object is the target belief, not the per-image structure.** Everything learned
   while judging candidate #1 remains true for #2–#6. Build one `TargetBelief` per episode; rebuild
   only the `ObservationFrame` per image.
2. **Keep the symbolic layer out of the decision path.** The slot frame exists to *propose and rank
   questions*. The final match/no-match call is made by a VLM looking at the actual pixels plus the
   belief rendered as text. A slot-extraction error must cost a wasted question, not a wrong
   conclusion.
3. **Evidence is asymmetric.** One decisive conflict proves `False`. Confirmations never prove
   identity — candidates are near-duplicates. So: stop on first decisive conflict; conclude `True`
   only when no *discriminative* slot remains unresolved.
4. **Ask wh-questions, not yes/no.** A wh-question returns the value (populating the belief for all
   later candidates) and avoids the acquiescence bias of a flash-tier oracle. A false "yes" is the
   single most expensive failure mode.
5. **Front-load.** Questions asked on candidate #1 amortize across the whole episode. Spending 4
   steps early and 0 later beats 2 steps on each of 6.
6. **Attributes you cannot read in the candidate are worthless**, however well the oracle answers
   them. `UNCLEAR` is a first-class value and disqualifies a slot from the question pool.

---

## 2. Repository layout

```
coin_agent/
  config.yaml
  agent/
    __init__.py
    schema.py        # slot ontology, canonical vocabularies, tiers
    state.py         # TargetBelief, ObservationFrame, SlotValue
    canon.py         # value normalization + confusability classes
    extract.py       # candidate-image -> ObservationFrame (VLM, JSON, logprobs)
    parse.py         # oracle answer -> SlotValue (hedge detection)
    compare.py       # conflict detection, decisiveness rules
    select.py        # question ranking + wh-templating
    adjudicate.py    # final VLM call on pixels + belief text
    budget.py        # step/time controller
    priors.py        # discriminativeness table lookup
    questioner.py    # GraphQuestioner(QuestionerInterface)  <-- entry point
    llm.py           # vllm client wrapper w/ logprobs, caching, timing
  scripts/
    analyze_episodes.py
    build_priors.py
    run_eval.py           # thin wrapper over upstream eval_model.py
    ablate.py
  patches/
  tests/
  artifacts/
    priors.json
    cache/
```

Local VLM assumption: an OpenAI-compatible vllm server. Model id from `env: LOCAL_VLM_MODEL_ID`
(default a Qwen3-VL checkpoint). All calls go through `agent/llm.py`; no module talks to the server
directly.

---

## 3. Slot schema (`agent/schema.py`)

A **fixed typed frame**, not an open-ended scene graph. Descriptions in this benchmark never exceed
~2 hops ("brass handles below a white countertop"), so depth is capped at 2 and the slot set is
closed. This makes comparison a dict lookup instead of graph matching, which is the part of graph
pipelines that breaks.

### 3.1 Slots

| Key | Type | Tier | Notes |
|---|---|---|---|
| `obj.category` | str | A | Always known from the description |
| `obj.color_primary` | COLOR | A | Dominant color of the target object |
| `obj.color_secondary` | COLOR | B | |
| `obj.material` | MATERIAL | A | painted / bare_wood / metal / laminate / glass / stone / fabric |
| `obj.hardware_type` | HARDWARE | B | knob / bar_pull / recessed / none |
| `obj.hardware_finish` | FINISH | A | brass / chrome / black / nickel / wood / none |
| `obj.state` | STATE | C | open / closed / on / off — transient |
| `obj.style` | str | C | free text, e.g. shaker, flat-panel |
| `obj.count` | int | B | how many of this object type are visible |
| `ctx.above.object` | str | A | e.g. countertop |
| `ctx.above.material` | MATERIAL | A | strongest single discriminator in kitchen scenes |
| `ctx.above.color` | COLOR | A | |
| `ctx.support.object` | str | A | what the object rests on / is set into |
| `ctx.adjacent[i].object` | str | B | up to `[CONFIG] max_adjacent: 3` |
| `ctx.adjacent[i].color` | COLOR | B | |
| `ctx.contains[i]` | str | C | movable contents |
| `room.type` | ROOM | A | kitchen / bedroom / bathroom / living_room / office |
| `room.floor_material` | MATERIAL | A | very stable across viewpoints |
| `room.floor_color` | COLOR | B | lighting-sensitive |
| `room.wall_color` | COLOR | B | lighting-sensitive |
| `room.window_present` | bool | B | |
| `room.notable_appliance` | str | B | e.g. range, dishwasher |

### 3.2 Tiers

- **Tier A — decisive-eligible.** A conflict here can terminate deliberation with `False`. Chosen for
  viewpoint- and lighting-stability.
- **Tier B — evidence only.** Accumulates toward the adjudicator; never decisive on its own.
- **Tier C — never decisive, never queried.** Transient or movable (contents, open/closed state,
  small props like plants and towels). These generate false conflicts across re-photographs.

Tier membership is data, not code — it lives in `schema.py` as a dict so `ablate.py` can vary it.

### 3.3 Canonical vocabularies (`agent/canon.py`)

Every slot with an enum type has:
- a canonical value list,
- a synonym map (`"navy" -> NAVY`, `"dark blue" -> NAVY`, `"butcher block" -> BARE_WOOD`),
- an **equivalence/confusability structure**.

The confusability structure is what prevents the single largest source of false conflicts. Define
three relations over canonical values:

- `SAME`: identical after normalization.
- `NEAR`: plausibly the same thing under different lighting/compression/viewpoint.
  Examples: `NAVY ~ DARK_BLUE ~ BLUE`; `WHITE ~ OFF_WHITE ~ CREAM`; `GREY ~ LIGHT_GREY ~ WHITE`;
  `BARE_WOOD ~ OAK ~ BUTCHER_BLOCK`.
- `FAR`: mutually exclusive. Examples: `NAVY / WHITE`; `PAINTED / BARE_WOOD`; `BRASS / CHROME`.

Only `FAR` pairs may produce a decisive conflict. `NEAR` pairs yield `WEAK_CONFLICT`, which feeds the
adjudicator but never terminates on its own. Store the relation as an explicit table in
`canon.py`; do not infer it from string distance.

---

## 4. State (`agent/state.py`)

```python
Provenance = Literal["description", "oracle", "prior"]
Certainty  = Literal["resolved", "hedged", "unknown"]

@dataclass
class SlotValue:
    raw: str | None            # verbatim source text
    canon: str | None          # canonical value, None if unresolved
    confidence: float          # [0,1]
    certainty: Certainty
    provenance: Provenance     # TargetBelief only
    source_question: str | None

@dataclass
class TargetBelief:
    description: str
    slots: dict[str, SlotValue]        # keyed by schema key
    asked: list[tuple[str, str]]       # (slot_key, question) — never re-ask a slot
    def render_text(self) -> str: ...  # for the adjudicator prompt
```

`ObservationFrame` is the same shape minus `provenance`, plus `image_hash` for caching.

**Initialization.** At `__init__`, parse the description into slots with
`provenance="description"`, `certainty="resolved"`, `confidence=1.0`. The description is
authoritative — the organizers' own prompt states the initial description can always be trusted.
Parse with one text-only LLM call using the same JSON schema, plus a regex fast path for the
`category` type (which is a bare noun phrase).

**Monotonicity.** A slot filled from `description` or `oracle` is never overwritten. The target image
does not change, so the belief only accumulates. Assert this invariant.

---

## 5. Per-candidate decision procedure (`agent/questioner.py`)

`ask_or_conclude(observation)` executes the following. It is called repeatedly for the same candidate
until a conclusion is returned, so it must be **idempotent with respect to the image**: cache the
`ObservationFrame` by `image_hash` and do not re-extract on later calls for the same candidate.

```
1. hash the image. if new candidate:
       frame = extract(image)                       # 1 VLM call
       increment candidate counter; recompute budget
2. if observation["answer"] is not None:
       slot = the slot of the last asked question
       belief.slots[slot] = parse(answer, slot)      # 1 cheap text call or regex
3. verdict = compare(frame, belief)
4. if verdict.decisive_conflict:
       return conclude(False, reasoning=verdict.explain())
5. pool = select.candidates(frame, belief, priors)
   if pool is empty:
       return conclude(True, ...)                    # nothing discriminative left to check
6. if budget.may_ask():
       q, slot = select.top(pool)
       record (slot, q); n_questions += 1
       return ask(q)
7. # budget exhausted, still undecided
   return conclude(adjudicate(image, belief, frame), ...)
```

### 5.1 Extraction (`agent/extract.py`)

One call per candidate. Request strict JSON matching the slot schema, with an explicit
`"unclear"` option for every field, and a per-field self-reported `visibility` in
`{clear, partial, not_visible}`.

Confidence per slot = combination of:
- mean token logprob of the value tokens (vllm `logprobs=1`), mapped through a sigmoid; and
- the `visibility` field (`not_visible` -> confidence 0, certainty `unknown`).

`[CONFIG] extract_self_consistency_k: 1`. If raised above 1, sample at `temperature=0.7` and take
per-slot majority; slot confidence becomes the agreement fraction. **Default 1** — the 600 s budget
does not accommodate K>1 on every candidate. If used at all, use it only on candidate #1.

Prompt requirements: no mention of a "target"; no reference to the description (extraction must be
description-blind so it cannot be biased toward confirming it); explicitly instruct that digital
artifacts and distortions are to be ignored and never reported.

### 5.2 Comparison (`agent/compare.py`)

For each slot present in both `frame` and `belief`, emit one of
`MATCH | WEAK_CONFLICT | CONFLICT | UNKNOWN | INCOMPARABLE`.

```
CONFLICT  requires ALL of:
    slot in TIER_A
    belief.certainty == "resolved"           (hedged oracle answers never decide)
    frame.confidence >= [CONFIG] tau_obs: 0.80
    canon relation(frame.canon, belief.canon) == FAR
```

Anything else that disagrees is `WEAK_CONFLICT`. A single `CONFLICT` is decisive.
`[CONFIG] weak_conflicts_for_decisive: 3` — that many independent Tier-B weak conflicts may also be
treated as decisive; default it **off** (`null`) and enable only if the ablation shows it helps.

`compare()` returns a structured object with an `explain()` string for `reasoning`. The reasoning
field is logged, so make it readable: which slot, which two values, which source.

### 5.3 Question selection (`agent/select.py`)

Candidate pool = slots where **all** hold:
- slot in Tier A or Tier B (never C);
- `belief.slots[slot].certainty == "unknown"` (never ask twice — the oracle is deterministic);
- `frame.slots[slot].certainty == "resolved"` and `confidence >= tau_obs` — you must be able to read
  it in the candidate for the answer to be usable;
- slot not implied by the description.

Score:

```
score(s) = frame.confidence[s]
         * disc(s, frame.canon[s], category)      # discriminativeness, §6.2
         * tier_weight[s]                          # [CONFIG] A: 1.0, B: 0.6
         * stability[s]                            # [CONFIG] per-slot, lighting sensitivity
```

`disc` is the estimated probability that a plausible distractor of the same category does **not**
share this value. Anything the description already states has `disc ≈ 0` by construction — the
distractors were selected to be plausible given the description — hence the "not implied" filter.

**Templating.** One fixed wh-template per slot. Rules, all testable:
- The object is named with the noun phrase from the description ("the navy blue kitchen lower
  cabinet"), never "it", never "the target", never "the object in your image".
- Open wh-form, not yes/no: "What material is the surface directly above the {np}?" not "Is the
  surface above the {np} white?"
- Under 20 words, so the under-15-word answer has room to be a value rather than a hedge.
- Never reference the candidate image, image quality, or artifacts.
- Bundling: `[CONFIG] allow_bundle: true` permits combining **at most 2** slots in one question when
  both concern the same region (`obj.color_primary` + `obj.hardware_finish`). Do not bundle across
  regions — a 15-word answer cannot carry it, and partial answers create hedged slots.

**Front-loading.** On candidate #1, override ranking to prefer maximum *coverage*: pick the highest-
scoring slot from each distinct region (`obj`, `ctx.above`, `room`) in turn. Room-level slots reject
far distractors for the rest of the episode at zero marginal cost.

### 5.4 Adjudication and defaults (`agent/adjudicate.py`)

One VLM call: the candidate image + `belief.render_text()` + the description + the Q/A history as
prose. **The slot frame is not passed** — principle 2. Ask for `<motivation>` (<60 words, no double
quotes) and `<score>` in `{0,1,2}`, matching the organizers' format; map `2 -> True`,
`0 -> False`, `1 -> False` (a residual "unsure" resolves to the base rate).

Fallbacks, in order:
1. Adjudicator parse failure -> retry once with a stricter format instruction.
2. Second failure, or time budget in the hard-stop zone -> conclude `False`.

`False` is the correct default: with one matching candidate per episode the prior on any given
candidate is roughly 1/6. Log every fallback; a high fallback rate is a bug, not a strategy.

### 5.5 Budget controller (`agent/budget.py`)

The questioner does not know how many candidates the episode holds. Assume
`[CONFIG] assumed_n_candidates: 8` (conservative; the example has 6).

```
steps_used, elapsed tracked internally (time.monotonic at __init__)
remaining_candidates_est = max(1, assumed_n - candidates_seen)
reserve = remaining_candidates_est                 # 1 conclusion step each
askable = max(0, 60 - steps_used - reserve)

per_candidate_cap = floor(askable / remaining_candidates_est) * front_load[candidates_seen]
    front_load: [CONFIG] {0: 2.0, 1: 1.0, default: 0.8}

ambiguity_allowance = count of Tier-A slots not implied by the description
per_candidate_cap = min(per_candidate_cap, ambiguity_allowance, [CONFIG] hard_cap_per_image: 6)

may_ask() is False if:
    questions_this_candidate >= per_candidate_cap
    or elapsed > [CONFIG] soft_time_frac: 0.60 * 600
    or steps_used >= 60 - reserve
hard stop (adjudicate immediately, no further calls beyond one):
    elapsed > [CONFIG] hard_time_frac: 0.85 * 600
```

The `ambiguity_allowance` term is what makes the policy adapt across description types:
`category` leaves ~6 Tier-A slots unspecified, `color_context_feature` leaves ~2.

Log per episode: `steps_used`, `elapsed`, `questions_per_candidate`, whether soft/hard stop fired.
If the truncation rate exceeds `[CONFIG] max_truncation_rate: 0.02`, the timing profile is wrong and
accuracy numbers are confounded — treat that as a blocking bug.

---

## 6. Offline artifacts

### 6.1 `scripts/analyze_episodes.py`

Reads `episodes_train.jsonl` and reports, to stdout and `artifacts/episode_stats.json`:
- distribution of candidates per episode;
- number of `match=True` per episode (verify whether it is always exactly 1);
- **position** of the `match=True` candidate — reported for awareness only, with a printed warning
  that no model component may consume it (§0.6);
- category frequency; description length per type;
- for each description type, how many Tier-A slots the description already specifies (this
  calibrates `ambiguity_allowance`).

### 6.2 `scripts/build_priors.py` -> `artifacts/priors.json`

For each category and each slot, estimate

```
disc(s, v, category) = 1 - P(a co-episode candidate shares value v | category)
```

Procedure: run `extract()` over all candidate images in the training episodes (cache aggressively —
this is a one-time cost, run it offline, not inside an episode). For each episode and slot, compute
the fraction of candidate pairs sharing the canonical value. Aggregate by category with Laplace
smoothing, `[CONFIG] prior_alpha: 1.0`. Default `disc = 0.5` for unseen (category, slot) pairs.

This table is the main reason the pipeline should beat a plain prompted baseline. Build it early.

---

## 7. LLM client (`agent/llm.py`)

- OpenAI-compatible chat completions against vllm; images as base64 data URLs.
- Every call returns `(text, logprobs, latency_s)` and accumulates into `time_required`.
- **Disk cache** keyed by `sha256(model_id | prompt | image_hash | temperature)` under
  `artifacts/cache/`. Needed for repeatable ablations and to avoid re-paying extraction cost.
- **Oracle cache** in the test harness only, keyed by `(episode_id, question)`. The oracle is
  deterministic at `temperature=1e-6`, so this is faithful and saves substantial API spend during
  development. Provide it as a `CachedOracle(OracleInterface)` wrapper — do not modify `Oracle.py`.
- Timeouts: `[CONFIG] llm_timeout_s: 20`, at most `[CONFIG] llm_retries: 1`. Never let a client-side
  retry storm consume the episode clock; on failure, degrade (skip extraction -> treat all slots
  unknown -> adjudicate).

---

## 8. Ablation ladder (`scripts/ablate.py`)

Implement and evaluate in this order. Each rung must be a flag, not a branch of the codebase.
Do not build rung *n+1* before rung *n* is measured.

| Rung | Flag | Description |
|---|---|---|
| A0 | `--baseline-prompted` | Single VLM call per candidate: description + image + Q/A history, organizers' example prompt. No belief state, no slots. **Floor.** |
| A1 | `--no-slots` | A0 + persistent `TargetBelief` (text only) across candidates, wh-questions only |
| A2 | `--no-priors` | A1 + typed slot extraction used only for question ranking and conflict detection, `disc` fixed at 0.5 |
| A3 | *(default)* | A2 + empirical `disc` table from §6.2 |
| A4 | `--weak-conflict-decisive` | A3 + Tier-B weak-conflict accumulation |

Report per rung and per description type: conclusion accuracy, mean questions per episode, mean
episode wall clock, truncation rate, and the confusion split (false-`True` vs false-`False`).

Expectation to test, not assume: **A1 may capture most of the gain.** If A2/A3 do not beat A1 by
more than run-to-run variance, the slot machinery is over-engineering and should be cut from the
submission.

---

## 9. Tests (`tests/`)

Unit, no network:
- `canon`: synonym normalization; `SAME`/`NEAR`/`FAR` relation table is symmetric and total over
  each vocabulary; `NAVY/WHITE` is `FAR`, `NAVY/BLUE` is `NEAR`.
- `compare`: a Tier-B `FAR` disagreement is never decisive; a hedged belief slot is never decisive; a
  low-confidence observation slot is never decisive.
- `select`: never returns a slot already in `belief.asked`; never returns a Tier-C slot; never
  returns a slot implied by the description; templates contain no banned substrings
  (`"target"`, `"image I"`, `"artifact"`, `"?"`-less strings).
- `budget`: reserves one step per remaining candidate; caps drop after candidate #1; hard stop fires.
- `state`: monotonicity invariant; `render_text()` never emits the description verbatim twice.
- **`integrity`: grep the whole `agent/` package for `task_image` and fail if found.**

Integration:
- `MockOracle` from `env.py` (returns a fixed affirmative string) — asserts the loop terminates, emits
  valid actions, and never returns both fields `None`.
- Scripted-oracle fixture with hand-written answers for 3 synthetic episodes covering: decisive
  conflict on candidate #1; true match requiring 3 questions; hedged answer forcing adjudication.

---

## 10. Implementation order

1. §0 verification script: assert every ground-truth fact in §0 against the installed repo. Fail
   loudly if upstream has changed.
2. `patches/` for §0.7 issues; get upstream `eval_model.py` running end-to-end with `MockOracle`.
3. `llm.py` with caching + timing. Measure single-call latency for extraction and adjudication on
   the real hardware. **If one candidate's decision costs more than 15 s, redesign before
   continuing** — the 600 s budget is the binding constraint.
4. `analyze_episodes.py`. Read the output before writing the policy.
5. Rung A0, evaluated on `[CONFIG] min_eval_episodes` episodes per description type.
6. Rung A1.
7. `schema.py` / `canon.py` / `extract.py` / `compare.py` / `select.py`; rung A2.
8. `build_priors.py`; rung A3.
9. Ablation report.

---

## 11. Submission note (out of scope for this build, decide before the deadline)

The challenge asks for **trained weights on HuggingFace** plus a technical report, and the
organizers' own model is pitched as substantially smaller and faster than modular pipelines. A
multi-call pipeline is admissible but fights both the 600 s limit and the stated values of the
benchmark.

The intended endgame: use this pipeline as a **teacher** to generate and filter question-asking
traces (alongside the 28k traces the organizers release), then SFT a single small VLM to emit
`<motivation><score><question>` in one forward pass. That collapses per-decision latency to a single
call while retaining the question-selection quality this pipeline is designed to produce. Keep trace
logging in a shape suitable for SFT from day one: `(description, image, qa_history) -> (motivation,
score, question)`, one row per `ask_or_conclude` call, with the eventual episode outcome attached for
filtering.
