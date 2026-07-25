# Upstream patches

Spec §0.7: keep `eval_model.py` / `env.py` / `Questioner.py` unmodified on disk. Every fix for a
known upstream issue lives here instead, applied at *runtime* by `scripts/run_eval.py` — never by
editing the files in the repo root.

## 1. `info['task_description']` KeyError (`001_task_description_keyerror.diff`)

`eval_model.py:161` does:

```python
print(f"Task is: {info['task_description']}")
```

but `env.py:_get_info()` returns the key as `target_description`, not `task_description`. This
raises `KeyError` the first time it executes. The diff is informational (what the one-line fix
is); `apply_patches.load_patched_eval_model_source()` applies the same substitution to the source
text in memory before it's exec'd by `run_eval.py` — `eval_model.py` on disk is never touched.

## 2. Missing `QA_eval/episodes_{run_type}.jsonl` (symlink)

`eval_model.py:112` hardcodes `f"QA_eval/episodes_{run_type}.jsonl"` (`run_type` is hardcoded to
`"train"` a few lines up), i.e. it always looks for `QA_eval/episodes_train.jsonl`. The repo ships
`episodes_train.jsonl` at the repo root instead, so this path doesn't exist out of the box.

Fix: a symlink, not a code change — `QA_eval/episodes_train.jsonl -> ../episodes_train.jsonl`.
`apply_patches.ensure_episodes_symlink()` creates it (idempotently) if missing; also created once
manually under `<repo_root>/QA_eval/`.

## 3. Conclusion doesn't always switch the observation on the training set

`eval_model.py:180-188`: after a correct conclusion, the env is *supposed* to advance
`current_distractor_idx` and hand back a new candidate image. On (some of) the training set this
sometimes doesn't happen — the loop detects `obs["image"] == old_obs["image"]` unexpectedly and
`break`s, discarding the whole episode with nothing logged. The spec's own words: "the organizers
state this is fixed on the held-out set" — so this is a training-set-only artifact, not something
to chase in our own code.

This is not a bug we can patch away (it's inside `env.py`'s image-switching logic, which we're
also not supposed to touch, and the organizers say it's already fixed downstream). Instead:
`scripts/run_eval.py` must **count** every episode discarded this way and report accuracy with
that denominator stated explicitly (e.g. "82.4% over 143/167 episodes, 24 discarded by the
known observation-switch bug"). Never silently drop the denominator. Per spec §0.7.3, never tune
on fewer than `config.yaml: eval.min_eval_episodes` (100) episodes after discards.

If the discard rate ever exceeds what's expected, that's a signal something *else* is wrong
(e.g. our own questioner producing a malformed action) — don't assume it's always this bug without
checking `_reasonings`/`_actions` logs for the discarded episode first.

## Not a bug: swapping in `GraphQuestioner`

`eval_model.py` imports `YourQuestioner` from `Questioner.py` and instantiates it directly. To run
our questioner through the unmodified eval loop, `apply_patches.substitute_questioner_import()`
rewrites that one import line (in memory only) to `from agent.questioner import GraphQuestioner as
YourQuestioner`. This isn't fixing anything upstream — it's how `scripts/run_eval.py` plugs our
implementation into their harness — but it's grouped here since it uses the same
patch-source-in-memory mechanism as fixes 1–2.

