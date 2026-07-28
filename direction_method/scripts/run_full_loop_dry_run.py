"""Full episode-loop dry run (SPEC.md build order step 4), against FakeVLM: drives
DirectionMethodQuestioner against the real env.QAEnv for all 167 training episodes.

Accuracy is meaningless here -- FakeVLM's canned responses don't reflect real image/description
content -- the point is exercising the full plumbing (env.reset/step, all four modules, budget
logic, logging) end to end without a live model, and confirming zero crashes and zero invalid
actions (ENV.md §4/§5: exactly one of question/conclusion must be non-None on every single call).

Uses one shared LLMClient for both the local oracle stand-in and the questioner itself -- fine for
this dry run (no real timing to protect), but SPEC.md §8 notes a real run should put the oracle on
a separate GPU/server so its inference doesn't compete with the modules being timed.

Replicates eval_model.py's own workaround for ENV.md §6 bug 3 (observation sometimes fails to
switch after a correct conclusion): such an episode is discarded, not counted as a crash.

Run: `VLM_BACKEND=fake python scripts/run_full_loop_dry_run.py`
"""

import os
import sys
import time
from pathlib import Path

DIRECTION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DIRECTION_ROOT.parent
for p in (DIRECTION_ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# env.py opens each episode's image via its stored path ("images/...") relative to the CWD, not
# relative to the jsonl file or the repo root -- match eval_model.py's own assumption that it is
# run from the repo root.
os.chdir(REPO_ROOT)

import numpy as np

from env import QAEnv
from llm import LLMClient
from oracle_stub import LocalOracleStandIn
from patches.apply_patches import fix_answer_prompt_typo
from questioner import DirectionMethodQuestioner

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
# Richest description variant -- exercises the most checklist/relation code paths (category-only
# descriptions barely touch context_parser's checklist-splitting logic at all).
TASK_TYPE = "color_context_feature"
N_EPISODES = 167
MAX_LOOP_ITERS = 100  # generous margin over env.max_steps (60); a real episode never needs this many


def main() -> None:
    fix_answer_prompt_typo()

    llm_client = LLMClient(MODEL_ID, cache_dir=DIRECTION_ROOT / "artifacts" / "cache")
    oracle = LocalOracleStandIn(llm_client)
    env = QAEnv(oracle, REPO_ROOT / "episodes_train.jsonl", task_type=TASK_TYPE)

    n_constructed = 0
    n_construction_crashes = 0
    n_loop_crashes = 0
    n_discarded_env_bug = 0
    n_truncated = 0
    n_completed_cleanly = 0
    invalid_actions = 0

    start = time.time()
    for episode_idx in range(N_EPISODES):
        old_obs, info = env.reset(options={"episode_idx": episode_idx})
        episode_id = env.current_episode_data["id"]

        try:
            questioner = DirectionMethodQuestioner(info, llm_client=llm_client)
        except Exception as e:  # noqa: BLE001 -- deliberately broad: this is a crash-detection harness
            n_construction_crashes += 1
            print(f"[CRASH] episode {episode_idx} (id={episode_id}) construction: {e!r}")
            continue
        n_constructed += 1

        outcome = None
        for step in range(MAX_LOOP_ITERS):
            try:
                action = questioner.ask_or_conclude(old_obs)
            except Exception as e:  # noqa: BLE001
                n_loop_crashes += 1
                print(f"[CRASH] episode {episode_idx} (id={episode_id}) step {step}: {e!r}")
                outcome = "crash"
                break

            if (action["question"] is None) == (action["conclusion"] is None):
                invalid_actions += 1
                print(f"[INVALID ACTION] episode {episode_idx} step {step}: {action}")
                outcome = "invalid_action"
                break

            obs, reward, terminated, truncated, info = env.step(action)

            # ENV.md §6 bug 3 replica (matches eval_model.py's own workaround).
            if action["conclusion"] is not None and np.all(obs["image"] == old_obs["image"]) and not terminated and not truncated:
                n_discarded_env_bug += 1
                outcome = "discarded_env_bug"
                break

            if action["question"] is not None:
                questioner.add_answer(obs["answer"] or "")

            old_obs = obs
            if terminated or truncated:
                if truncated and not terminated:
                    n_truncated += 1
                outcome = "truncated" if (truncated and not terminated) else "terminated"
                break
        else:
            outcome = "exceeded_max_loop_iters"
            print(f"[WARNING] episode {episode_idx} (id={episode_id}) never terminated within {MAX_LOOP_ITERS} calls")

        if outcome in ("terminated", "truncated"):
            n_completed_cleanly += 1

        print(
            f"episode {episode_idx:3d} (id={episode_id}): outcome={outcome:22s} "
            f"n_successes={env.n_successes} n_questions={questioner.n_questions} "
            f"candidates_seen={questioner.candidates_seen} "
            f"soft_stop={questioner.episode_log['soft_stop_fired']} hard_stop={questioner.episode_log['hard_stop_fired']}"
        )

    total_time = time.time() - start
    print("=" * 70)
    print(f"{N_EPISODES} episodes attempted in {total_time:.1f}s")
    print(f"constructed cleanly: {n_constructed}  (construction crashes: {n_construction_crashes})")
    print(f"completed cleanly (terminated or truncated): {n_completed_cleanly}")
    print(f"loop crashes: {n_loop_crashes}")
    print(f"invalid actions (both/neither None): {invalid_actions}")
    print(f"discarded (ENV.md §6 bug 3): {n_discarded_env_bug}")
    print(f"env-level truncations (hit 60 steps or 600s): {n_truncated}")
    print("\nAccuracy is meaningless under FakeVLM -- this run only validates plumbing.")

    assert n_construction_crashes == 0, "questioner construction crashed on at least one episode"
    assert n_loop_crashes == 0, "ask_or_conclude crashed on at least one episode"
    assert invalid_actions == 0, "an action returned both/neither of question and conclusion as None"


if __name__ == "__main__":
    main()
