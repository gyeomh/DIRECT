# Upstream patches — direction_method

ENV.md §6: keep `eval_model.py` / `env.py` / `Questioner.py` unmodified on disk. Every fix lives
here, applied at *runtime* by whatever script drives the eval loop — never by editing the repo
root files. (Independent of `coin_agent/patches/`, which patches the same three issues for the
other method in this repo — duplicated here so `direction_method/` stays self-contained.)

## 1. `info['task_description']` KeyError

`eval_model.py:161`:

```python
print(f"Task is: {info['task_description']}")
```

`env.py:_get_info()` returns the key as `target_description`, not `task_description` → `KeyError`
on the very first episode. `apply_patches.load_patched_eval_model_source()` substitutes this one
line in the source text in memory before it's `exec`'d.

## 2. Missing `QA_eval/episodes_{run_type}.jsonl`

`eval_model.py:112` hardcodes `f"QA_eval/episodes_{run_type}.jsonl"` (`run_type` is hardcoded to
`"train"` a few lines up) → always looks for `QA_eval/episodes_train.jsonl`. The repo ships
`episodes_train.jsonl` at the repo root instead. Fix: a symlink, not a code change —
`apply_patches.ensure_episodes_symlink()` creates `QA_eval/episodes_train.jsonl -> ../episodes_train.jsonl`
idempotently.

## 3. Conclusion doesn't always switch the observation on the training set

`eval_model.py:180-188`: after a correct conclusion, the env is supposed to advance
`current_distractor_idx` and hand back a new candidate image. On some training-set episodes this
doesn't happen — the loop detects `obs["image"] == old_obs["image"]` unexpectedly and `break`s,
discarding the episode. Not patchable (it's inside `env.py`'s switching logic, which stays
untouched, and the organizers state it's already fixed on the held-out set). Any eval driver must
**count** episodes discarded this way and report accuracy with that denominator stated explicitly
— never silently drop it from the count.

## 4. `MockOracle.ask()` is missing `self` — not in ENV.md, found while doing step 1

```python
class MockOracle:
    def __init__(self):
        pass

    def ask(*, prompt="", images=[]):
        return "Yes that is true [Mock answer]"
```

`ask` has no `self` parameter. Calling it the normal way — `MockOracle().ask(prompt=..., images=...)`
— raises `TypeError: ask() takes 0 positional arguments but 1 positional argument ... were given`,
because Python still binds the instance as an implicit positional argument on a bound-method call,
and the signature (all keyword-only, no leading positional slot) has nowhere to put it. This means
upstream's own `MockOracle` cannot be used as shipped — required fixing to satisfy ENV.md §7's
"get upstream running end to end with MockOracle" instruction.

Fix: `apply_patches.WorkingMockOracle` — a drop-in replacement with the same behavior (`.ask()`
always returns the same fixed affirmative string, ignoring its arguments) and a correct signature.
Not a subclass of the broken `MockOracle` (subclassing wouldn't fix the missing `self` on the
inherited method); a standalone class satisfying the same duck-typed interface `env.py` expects
(an object with `.ask(*, prompt, images)`).

## 5. `results/` directory doesn't exist — not a bug, a missing precondition

`eval_model.py`'s final step writes `results/<QuestionerClass>_<type>_<run>_<start>_<end>.gzip.json`
via `gzip.GzipFile(path, "w")`, which doesn't create parent directories. The repo ships no
`results/` directory, so a full run crashes with `FileNotFoundError` at the very last step —
after every episode has already completed — unless the directory is created first. Not
version-controllable as an empty dir (git doesn't track empty directories), so anything driving
`eval_model.py` end to end must `mkdir -p results` before running. `scripts/verify_env.py` and
`scripts/run_self_check_experiment.py` (once it drives the full harness rather than calling the
oracle stand-in directly) both need this.

## 6. `ANSWER_PROMPT` typo — kept as ENV.md's word choice, not "fixed" upstream

`env.py:14`:

```python
ANSWER_PROMPT = "You are a faithful assistant. Answer correnctly to the following question based on the above image: {QUESTION}. Be concise (under 15 words)."
```

`correnctly` should be `correctly`. `apply_patches.fix_answer_prompt_typo()` monkeypatches
`env.ANSWER_PROMPT` in memory (module-global reassignment after import — `_get_observation()`
reads the name from `env`'s module namespace at call time, so this is sufficient; no need for the
exec-a-patched-source trick issues 1–2 use, since this isn't inside a function body executed once
at script-run time). **Nothing else about the prompt text changes.**

**The official harness does not apply this patch and therefore keeps the typo.** Any local run
using this fix is not byte-for-byte identical to what the real evaluation sends the real oracle.
This matters if a method's self_check/context_parser prompts are ever tuned against logged
oracle answers — those answers were produced against the *typo'd* prompt in a real run, the
*fixed* one in ours. In practice this one word is unlikely to change answer quality, but the
discrepancy is worth remembering rather than silently forgetting which prompt produced which log.

## Not a bug: swapping in a questioner

Same mechanism as `coin_agent/patches/`: `apply_patches.substitute_questioner_import()` rewrites
`from Questioner import YourQuestioner` to point at whichever questioner class this method
provides. Used in `scripts/verify_env.py` (step 1) to run the real, patched `eval_model.py`
`__main__` block end to end with a throwaway `TrivialQuestioner`. Will be used again once a real
questioner exists.
