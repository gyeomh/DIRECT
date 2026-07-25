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
| `parse.py` | Text → `SlotValue`: oracle-answer hedge detection (`"maybe navy"` → hedged, never decisive) and the initial description parse (currently only `obj.category`, seeded from `info["category"]` — see below). |
| `compare.py` | Per-slot `MATCH / WEAK_CONFLICT / CONFLICT / UNKNOWN / INCOMPARABLE` classification. A `CONFLICT` requires Tier A + resolved belief + confident observation + `FAR` relation. |
| `select.py` | Which unresolved slot is worth asking about (score = frame confidence × discriminativeness × tier weight × stability), front-loads region coverage on candidate #1, bundles ≤2 same-region slots, and renders the oracle-facing wh-question. |
| `budget.py` | `BudgetController` — per-candidate question cap derived from steps/time remaining, reserving one conclusion step per estimated remaining candidate; soft/hard time stops. |
| `priors.py` | Runtime lookup for `disc(slot, value, category)` from `artifacts/priors.json` (built offline by `scripts/build_priors.py`). |
| `extract.py` | Candidate image → `ObservationFrame` via one Qwen VLM call (optionally self-consistency-voted on candidate #1). **`EXTRACTION_PROMPT` not written yet.** |
| `adjudicate.py` | Final match/no-match call: image + belief text + description + Q/A history (never the slot frame) → `<motivation><score>`. **`ADJUDICATION_PROMPT` not written yet.** |
| `llm.py` | Cached, timed, retrying wrapper around `utils.ClientBasedLLM` (the vllm/Qwen client). Everything else calls the VLM only through here. |
| `questioner.py` | `GraphQuestioner(QuestionerInterface)` — wires all of the above into the loop above. The only place allowed to touch (and discard) `info["task_image"]`. |

## Ground-truth notes not in the original spec

- `env.py:135` sets `info["category"]` on every `reset()` regardless of `--description-type` — `obj.category` is seeded from this directly rather than guessed from the description text.
- Empirically confirmed via `scripts/analyze_episodes.py`: in all 167 training episodes, the `match=True` candidate is the last one in the distractor list. Never condition on this (per spec §0.6) — the held-out set is regenerated.

## What's stubbed vs. real

- **Real, tested now (53 passing tests):** `schema`, `canon`, `state`, `compare`, `select`, `budget`, `parse` (oracle-answer path), `llm`, `questioner` orchestration (tested via monkeypatched `extract`/`adjudicate`).
- **Structurally wired, waiting on a prompt:** `extract.EXTRACTION_PROMPT`, `adjudicate.ADJUDICATION_PROMPT`, `parse.DESCRIPTION_PARSE_PROMPT` — see the conversation for current priority (extraction first).
- **CLI stubs (logic depends on the above):** `scripts/build_priors.py`, `scripts/run_eval.py`, `scripts/ablate.py`.
- **Runtime-only patches, upstream files untouched:** `patches/` — see `patches/README.md`.
