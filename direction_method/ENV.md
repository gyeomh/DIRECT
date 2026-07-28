# ENV.md — CoIN Challenge environment constraints

Fixed facts about the harness. These are not design choices; do not change code to work around them,
and do not edit upstream files. If upstream changes, re-verify this document first.

Upstream: `https://github.com/e-zorzi/coin_challenge`

---

## 1. Episode structure

- **Episode** = one fixed target image + one description + an ordered list of candidate images
  (6 in the published example).
- The target image is opened once in `reset()` and **never changes during the episode**.
- Candidates are judged **one at a time, in order**. The index advances only when the conclusion
  matches the ground-truth label.
- **A wrong conclusion terminates the episode immediately.** So does a correct conclusion on the
  final candidate. Per-candidate accuracy compounds across the episode.
- `match` means **"this image shows the same object instance as the target"**, not "this image
  satisfies the description". The same label is reused across all six description types.

## 2. Oracle

- Sees **only the target image**. Not the candidate, not the description, and it has no notion of
  which object is "the target". Questions must name the object self-containedly — never "it", never
  "the target object".
- Answers are capped **under 15 words**.
- Runs at `temperature=1e-6` → effectively **deterministic**. Re-asking the same question in the same
  wording returns the same answer, costing a step for zero information.
- **The oracle is not ours.** `eval_model.py` constructs it; the official run uses a hosted model.
  Any local oracle is a development stand-in.
- `ask_oracle` retries 5× with `wait_fixed=80000` — one flaky call costs 80 s of episode wall clock.

## 3. Budgets — episode-wide, shared across all candidates

| Budget | Value |
|---|---|
| Steps | `max_steps = 60` |
| Wall clock | `_MAX_TIME_ALLOWED = 600` s, from `reset()` |

Every action is one step, question or conclusion. Truncation is silent: the loop exits and the
remaining candidates are never judged, freezing the success count. **Truncation is an accuracy loss.**

Reward, for reference: `+10` correct conclusion, `-10` wrong, `-1` per question.

## 4. Interface contract

```python
class QuestionerInterface(ABC):
    def __init__(self, info, *args)
    def ask_or_conclude(self, observation) -> dict(question=..., conclusion=..., reasoning=...)
    def add_answer(self, answer)
    def reset_questions(self); def reset_time()
```

- `info["target_description"]` holds the description.
- `observation = {"image": np.ndarray (H,W,3) uint8, "answer": str | None}`
- Return **exactly one** of `question` / `conclusion` as non-`None`.
  **Never return both as `None`** — upstream `_validate_action` has an operator-precedence bug that
  lets it pass, and the env then computes `reward = None`.
- `conclusion` is `0` or `1`. Always test with `is not None`, never truthiness — `0` is falsy.
- Maintain `self.n_questions` and `self.time_required` yourself. `env.n_questions` is never
  incremented upstream and is always 0. `eval_model.py` reads the questioner's values for logging.

### Two call-pattern facts that drive the design

1. **The questioner is constructed once per episode and is never reset between candidates.**
   `eval_model.py` does not call `reset_questions()` in the candidate loop. Instance state persists
   across all candidates by design — this is what makes the accumulating checklist work.
2. **`ask_or_conclude` is called repeatedly for the same candidate** — once per question, then once
   more for the conclusion. Detect candidate transitions by **hashing the image array** and comparing
   to the previous hash. Cache per-candidate work (bbox, zone set) against that hash so it is not
   recomputed on every call.

## 5. Integrity — non-negotiable

- `info["task_image"]` **is the target image object**. This is an upstream leak. **Never read this
  key.** Discard it in `__init__`. Add a test that greps the package for `task_image` and fails if
  found.
- Do not condition on candidate position or episode index.
- Report `n_questions` honestly; it is self-reported to the logger.

## 6. Known upstream bugs

Keep upstream files unmodified. Put fixes in `patches/` with a short README.

1. `eval_model.py` prints `info['task_description']`, but `_get_info()` returns
   `target_description` → `KeyError`.
2. `eval_model.py` loads `QA_eval/episodes_{run_type}.jsonl`; the repo ships `episodes_train.jsonl`
   at the root.
3. After a conclusion the observation sometimes fails to switch on the training set and the eval loop
   `break`s, discarding the episode. Log the discard count and state the denominator when reporting
   accuracy.

## 7. Verify before building anything

```python
# Does the matching candidate equal the target image, and is there exactly one per episode?
for ep in episodes:
    m = [d for d in ep["distractors"] if d["match"]]
    print(len(m), [d["path"] == ep["path"] for d in m])
```

The method assumes the matching candidate is the target image itself. Confirm this holds before
implementing. Also report, for awareness only, the position of the matching candidate — **no
component may consume it.**

Then get upstream running end to end with `MockOracle` before writing any method code.
