"""Agent-as-VLM verification run: drives the real env.QAEnv + DirectionMethodQuestioner over a
hand-picked sample of training episodes, with every VLM call (oracle included) answered by Claude
directly reading the prompt and image via agent_bridge.AgentBridgeBackend -- no vllm server, no
FakeVLM. This is a genuine content check (does the design produce sensible checklists/verdicts on
real images), not a plumbing smoke test (that's scripts/run_full_loop_dry_run.py, against FakeVLM).

Sample: 5 short (1-candidate) + 5 long (7-candidate) episodes from episodes_train.jsonl, chosen to
stress both ends of the per-episode call-count range without attempting all 167.

Run in the background; a Monitor watches its stdout for "AGENT_REQUEST_READY <id>" lines. For
each one: read queue/requests/<id>.json + queue/images/<id>.png (if present), answer, write
queue/responses/<id>.json as {"text": "..."}.
"""

import json
import os
import sys
from pathlib import Path

DIRECTION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DIRECTION_ROOT.parent
for p in (DIRECTION_ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.chdir(REPO_ROOT)  # env.py opens each episode's image relative to CWD, not to the jsonl path

import numpy as np

from agent_bridge import AgentBridgeBackend
from env import QAEnv
from llm import LLMClient
from oracle_stub import LocalOracleStandIn
from patches.apply_patches import fix_answer_prompt_typo
from questioner import DirectionMethodQuestioner

MODEL_ID = "claude (agent bridge)"
TASK_TYPE = "color_context_feature"
MAX_LOOP_ITERS = 100

# (episode_idx, label) -- 5 short (1 candidate) + 5 long (7 candidates), by category for variety.
_DEFAULT_TARGET_EPISODES = [
    (0, "short"),   # cabinet
    (2, "short"),   # blanket
    (6, "short"),   # shelf
    (9, "short"),   # tv
    (10, "short"),  # cloth
    (83, "long"),   # sink
    (99, "long"),   # bathroom cabinet
    (107, "long"),  # heater
    (108, "long"),  # drawer
    (138, "long"),  # kitchen cabinet
]

# Override for a partial/targeted run, e.g. AGENT_VERIFY_EPISODES="83:long,99:long"
if os.environ.get("AGENT_VERIFY_EPISODES"):
    TARGET_EPISODES = [
        (int(pair.split(":")[0]), pair.split(":")[1]) for pair in os.environ["AGENT_VERIFY_EPISODES"].split(",")
    ]
else:
    TARGET_EPISODES = _DEFAULT_TARGET_EPISODES

QUEUE_DIR = DIRECTION_ROOT / "agent_bridge_queue"
RESULTS_PATH = DIRECTION_ROOT / "artifacts" / "agent_verification" / os.environ.get("AGENT_VERIFY_RESULTS_NAME", "results.json")


def main() -> None:
    fix_answer_prompt_typo()

    llm_client = LLMClient(MODEL_ID, backend="fake", cache_dir=DIRECTION_ROOT / "artifacts" / "agent_verification" / "cache")
    bridge = AgentBridgeBackend(QUEUE_DIR)
    llm_client._backend.generate = bridge.generate  # noqa: SLF001 -- deliberate, see agent_bridge.py

    oracle = LocalOracleStandIn(llm_client)
    env = QAEnv(oracle, REPO_ROOT / "episodes_train.jsonl", task_type=TASK_TYPE)

    results = []
    for episode_idx, label in TARGET_EPISODES:
        old_obs, info = env.reset(options={"episode_idx": episode_idx})
        episode_id = env.current_episode_data["id"]
        print(f"\n===== EPISODE {episode_idx} ({label}, id={episode_id}, category={env.current_episode_data['category']!r}) =====", flush=True)
        print(f"description: {info['target_description']!r}", flush=True)

        questioner = DirectionMethodQuestioner(info, llm_client=llm_client)

        outcome = None
        for step in range(MAX_LOOP_ITERS):
            action = questioner.ask_or_conclude(old_obs)
            assert (action["question"] is None) != (action["conclusion"] is None), f"INVALID ACTION: {action}"

            obs, reward, terminated, truncated, info = env.step(action)

            if action["conclusion"] is not None and np.all(obs["image"] == old_obs["image"]) and not terminated and not truncated:
                outcome = "discarded_env_bug"
                break

            if action["question"] is not None:
                questioner.add_answer(obs["answer"] or "")

            old_obs = obs
            if terminated or truncated:
                outcome = "truncated" if (truncated and not terminated) else "terminated"
                break
        else:
            outcome = "exceeded_max_loop_iters"

        result = {
            "episode_idx": episode_idx,
            "episode_id": episode_id,
            "label": label,
            "category": env.current_episode_data["category"],
            "description": info["target_description"] if "target_description" in info else None,
            "outcome": outcome,
            "n_successes": env.n_successes,
            "n_candidates_total": len(env.current_episode_data["distractors"]),
            "n_questions": questioner.n_questions,
            "candidates_seen": questioner.candidates_seen,
            "episode_log": questioner.episode_log,
            "final_checklist": questioner.checklist,
            "candidate_logs": [
                {
                    "questions_asked": c.questions_asked,
                    "self_check_calls": c.self_check_calls,
                    "verdicts": c.verdicts,
                    "conclusion": c.conclusion,
                    "reasoning": c.reasoning,
                    "elapsed_s": round(c.elapsed_s, 2),
                }
                for c in questioner.candidate_logs
            ],
        }
        results.append(result)
        print(f"EPISODE_DONE {episode_idx} outcome={outcome} n_successes={env.n_successes}/{result['n_candidates_total']}", flush=True)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nALL_EPISODES_DONE. Results written to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
