# coin_agent architecture

Implements `graph_method/coin_questioner_spec.md`'s `GraphQuestioner`: a questioner that tracks a
persistent belief about the target object across all candidates in an episode, uses a typed slot
frame only to *rank questions and detect conflicts* (never to make the final call directly), and
asks wh-questions that front-load onto candidate #1.

## Data flow (per `ask_or_conclude` call)

```
observation {image, answer}
      │
      ├─ new candidate image? ──yes──> extract() [Qwen VLM]  ──> ObservationFrame (cached by image hash)
      │                                                             │
      ├─ answer present? ──> parse_oracle_answer() ──> belief.set_slot()
      │                                                             │
      ▼                                                             ▼
                         compare(frame, belief) ──> decisive_conflict? ──yes──> conclude(False)
                                   │no
                                   ▼
                    select.candidate_pool() ──> empty? ──yes──> hedged slot stuck in there?
                                   │no                              │yes: adjudicate() instead
                                   │                                │no: conclude(True) — elimination
                                   ▼
                    budget.may_ask()? ──yes──> select.top() ──> ask(question)
                                   │no
                                   ▼
                          adjudicate() [Qwen VLM] ──> conclude(bool)
```

`TargetBelief` (the target's accumulated knowledge) persists across all candidates in the
episode; `ObservationFrame` (one candidate's readable slots) is rebuilt only when the image hash
changes. Belief slots are monotonic — once resolved from the description or an oracle answer,
never overwritten, only ever added to.

## Modules (`agent/`)

| Module | Responsibility |
|---|---|
| `schema.py` | Slot ontology: 22 slots (10 indexed `ctx.adjacent[i]`/`ctx.contains[i]` expand to more), each with a type, a Tier (A/B/C), and a region (used for front-load coverage + question bundling). |
| `canon.py` | Synonym normalization (`"navy blue"` → `navy`) and `SAME`/`NEAR`/`FAR` confusability tables. Only `FAR` can produce a decisive conflict. |
| `state.py` | `SlotValue`, `TargetBelief`, `ObservationFrame`. Enforces the monotonicity invariant. |
| `parse.py` | Text → `SlotValue`: oracle-answer hedge detection (`"maybe navy"` → hedged, never decisive) and the initial description parse — one text-only Qwen call seeding everything `target_description` states (plus `obj.category` from `info["category"]`, no LLM needed for that one) into the belief before any candidate image is seen. |
| `compare.py` | Per-slot `MATCH / WEAK_CONFLICT / CONFLICT / UNKNOWN / INCOMPARABLE` classification. A `CONFLICT` requires Tier A + resolved belief + confident observation + `FAR` relation. |
| `select.py` | Which unresolved slot is worth asking about (score = frame confidence × discriminativeness × tier weight × stability), front-loads region coverage on candidate #1, bundles ≤2 same-region slots, and renders the oracle-facing wh-question. |
| `budget.py` | `BudgetController` — per-candidate question cap derived from steps/time remaining, reserving one conclusion step per estimated remaining candidate; soft/hard time stops. |
| `priors.py` | Runtime lookup for `disc(slot, value, category)` from `artifacts/priors.json` (built offline by `scripts/build_priors.py`). |
| `extract.py` | Candidate image → `ObservationFrame` via one Qwen VLM call (optionally self-consistency-voted on candidate #1). Confidence is the model's self-reported `visibility` alone for now — real per-token logprob slicing is a documented v2 addition, not faked in. |
| `adjudicate.py` | Final match/no-match call: image + belief text + description + Q/A history (never the slot frame) → `<motivation><score>`. Unlike `extract.py`, not description-blind — this call's whole job is comparing the candidate against the target concept. |
| `llm.py` | Cached, timed, retrying wrapper around `utils.ClientBasedLLM` (the vllm/Qwen client). Everything else calls the VLM only through here. |
| `questioner.py` | `GraphQuestioner(QuestionerInterface)` — wires all of the above into the loop above. The only place allowed to touch (and discard) `info["task_image"]`. |

## Ground-truth notes not in the original spec

- `env.py:135` sets `info["category"]` on every `reset()` regardless of `--description-type` — `obj.category` is seeded from this directly rather than guessed from the description text.
- Empirically confirmed via `scripts/analyze_episodes.py`: in all 167 training episodes, the `match=True` candidate is the last one in the distractor list. Never condition on this (per spec §0.6) — the held-out set is regenerated.
- Empirically confirmed via a quick script over `episodes_train.jsonl`: every one of the 528 training distractors shares its filename's category prefix with the episode's target category — 0 mismatches. So `obj.category` never needs re-extraction per candidate either; `schema.queryable_slot_keys()` excludes it (and all Tier-C slots, which have zero consumers anywhere in compare/select/adjudicate) from both `extract.py`'s and `parse.py`'s VLM prompts.

## What's stubbed vs. real

All three prompts (`parse.DESCRIPTION_PARSE_PROMPT`, `extract.EXTRACTION_PROMPT`,
`adjudicate.ADJUDICATION_PROMPT`) are written, wired in, and **live-tested end to end** against a
real model: `Qwen/Qwen3-VL-30B-A3B-Instruct` served via vllm, pinned to a single GPU. See
`config.yaml`'s `vllm.model_id` comment for the exact vllm/torch combo that made this work on a
driver capped at CUDA 12.8 (`vllm==0.15.0`, which pins `torch==2.9.1` — its default PyPI build is
`+cu128`; anything from `vllm>=~0.2x` defaults to a CUDA-13-only build and will not run here).

- **Real, tested now (90 passing tests):** every module in `agent/`. Integration tests use
  monkeypatched `extract`/`adjudicate` (no vllm server in CI), but `test_extract.py` /
  `test_adjudicate.py` / `test_parse.py` exercise the real prompt text and parsing logic directly,
  and several of their regression tests came directly from bugs the live run below surfaced.
- **Live run against the real model:** 14 full episodes through `env.QAEnv` + `GraphQuestioner`,
  using our own `ClientBasedLLM` (same server) as a stand-in oracle (no Gemini key configured).
  Zero crashes, all actions valid. The one 1-candidate episode tested (trivially always the match)
  went 6/6 correct across every description type; a harder 5-candidate episode never went fully
  correct in 6 attempts — see the open question below.
- **Five real bugs caught and fixed during this live run** (all have regression tests):
  1. `canon.py`'s color synonym table only listed specific "modifier + color" phrases
     (`"light blue"`, `"dark grey"`) — every other combination (`"light green"`, `"multicolored"`)
     silently failed to normalize, discarding genuinely correct model output. Added a
     modifier-stripping fallback instead of trying to enumerate every combination.
  2. `llm.py` crashed trying to JSON-cache logprobs (`want_logprobs=True` returns openai SDK
     `TopLogprob` objects, not plain dicts) — every single extraction call would have failed the
     moment a live server was used. Also: the crash left a truncated cache file that would have
     kept failing forever on retry (a `_load_cached` cache hit on a corrupt file). Fixed both:
     serialize logprobs to plain dicts, and write the cache file atomically (temp file + rename).
  3. **The most consequential one:** `compare.py` treated free-text slot mismatches (e.g.
     `ctx.above.object`) as `FAR` (decisive), reasoning that otherwise these Tier-A slots could
     never be decisive. Live-tested this exact assumption and it backfired immediately: `extract()`
     correctly said `"picture"` (bare noun) while the oracle's verbose answer to the same question
     ("A painting of a girl on a swing, flowers, and butterflies") produced a differently-worded
     belief value — flipping a genuinely correct match into a wrong conclusion on the very first
     test episode. Reverted to `NEAR` (never decisive) for free-text slots — tested, not assumed,
     per spec §8's own philosophy. Also improved `parse._clean_free_text` to reduce verbose oracle
     answers to their head noun, independently of whether that reduction is decisive.
  4. `extract.py` confidence used to multiply the visibility factor by a placeholder logprob
     value of 0.5, so even a `"clear"` observation computed confidence `0.5 × 1.0 = 0.5` — below
     the default `tau_obs=0.80` — silently failing every slot's confidence check with no visible
     error. Fixed: confidence is the self-reported visibility factor alone until real per-token
     logprob slicing is a documented v2 addition, not faked in.
  5. `adjudicate.qa_history_text` printed a bundled question's Q&A pair twice (`select.py` bundles
     ≤2 same-region slots into one question, recorded once per slot in `belief.asked` with
     identical question text) — deduped by question text.
- **Open design question, not yet resolved:** on the harder 5-candidate episode, the questioner
  repeatedly concluded `True` via `select.candidate_pool()` being empty ("no discriminative slots
  remain, elimination") on candidates that likely weren't the actual match. Root cause hypothesis:
  an empty pool can mean either "everything checkable has been confirmed" (the spec's intended
  reading) *or* "this candidate's photo just wasn't informative enough for the remaining unresolved
  slots" (low-confidence extraction, not confirmation) — and these are currently indistinguishable.
  This is the same shape of bug as the hedged-slot fix in `questioner.py` (empty-pool-for-the-wrong-
  reason), but for low observation confidence rather than hedged oracle answers. Not fixed yet —
  needs a design decision on how to detect "this candidate just wasn't informative" vs. "genuinely
  confirmed," discussed with the user before changing.
- **CLI stubs (logic depends on the above, now unblocked):** `scripts/build_priors.py`,
  `scripts/run_eval.py`, `scripts/ablate.py`.
- **Runtime-only patches, upstream files untouched:** `patches/` — see `patches/README.md`.
