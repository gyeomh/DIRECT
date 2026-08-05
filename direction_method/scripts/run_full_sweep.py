"""Full real-VLM sweep: every training episode x every description type, against the live
vllm server, with rich per-episode-run logs for a later webpage trace viewer.

167 episodes x 6 description types (episodes_train.jsonl's task keys: category, color, context,
color_feature, color_context, color_context_feature) = 1002 total episode-runs. Each run drives
DirectionMethodQuestioner against the real env.QAEnv exactly like run_full_loop_dry_run.py, but
against the real vllm backend (VLM_PORT=8002 on GPU 1, per this session's server) instead of
FakeVLM, and with a full log record written to disk per run instead of just a print line.

Concurrency: one (episode_idx, task_type) pair per submitted task, run via ThreadPoolExecutor.
Each task builds its OWN LLMClient and QAEnv instance -- no shared mutable state across threads
except the on-disk response cache (safe: LLMClient._store_cache does write-then-rename). vllm's
continuous batching serves many concurrent chat completions efficiently on one GPU; SWEEP_WORKERS
controls how many run at once (default 12 -- adjust down if the server starts queuing/timing out
under load, up if GPU utilization has headroom).

Resumable: a task whose output JSON already exists on disk is skipped, so a killed/crashed run can
just be restarted with the same command and it picks up where it left off. Delete the specific
episode/task_type JSON files (or the whole log dir) to force a re-run.

Image handling: the target image and every candidate (distractor) image are already persistent
files under images/ (a symlink to /data/gyeom/coin_challenge/images/) -- the log records their
resolved, symlink-independent absolute paths rather than copying them. The one image that does NOT
already exist as a file anywhere is zone_gen's red-boxed candidate image (synthesized at runtime,
one per candidate) -- that one is saved to LOG_ROOT/images/.

Run: `VLM_PORT=8002 python scripts/run_full_sweep.py`
Override worker count: `SWEEP_WORKERS=16 VLM_PORT=8002 python scripts/run_full_sweep.py`
Smoke-test on a slice first: `SWEEP_EPISODES=3 SWEEP_TASK_TYPES=category,color VLM_PORT=8002 python scripts/run_full_sweep.py`
"""

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

DIRECTION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DIRECTION_ROOT.parent
for p in (DIRECTION_ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# env.py opens each episode's image via its stored path ("images/...") relative to the CWD --
# match eval_model.py's own assumption that it is run from the repo root.
os.chdir(REPO_ROOT)

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from env import QAEnv
from llm import LLMClient, image_hash
from oracle_stub import LocalOracleStandIn
from patches.apply_patches import fix_answer_prompt_typo
from questioner import DirectionMethodQuestioner

MODEL_ID = os.environ.get("SWEEP_MODEL_ID", "Qwen/Qwen3-VL-30B-A3B-Instruct")
# Thinking-capable models (e.g. Qwen3.6) need enable_thinking forced off per call, see llm.py's
# LLMClient(disable_thinking=...) -- irrelevant/no-op for Instruct-only models like Qwen3-VL.
DISABLE_THINKING = os.environ.get("SWEEP_DISABLE_THINKING", "0") == "1"
ALL_TASK_TYPES = ["category", "color", "context", "color_feature", "color_context", "color_context_feature"]
MAX_LOOP_ITERS = 100  # generous margin over env.max_steps (60)
CALL_TIMEOUT_S = float(os.environ.get("SWEEP_CALL_TIMEOUT_S", "60.0"))  # generous: concurrent workers share one GPU, individual calls slow under load

N_EPISODES = int(os.environ.get("SWEEP_EPISODES", "167"))
TASK_TYPES = os.environ.get("SWEEP_TASK_TYPES", ",".join(ALL_TASK_TYPES)).split(",")
N_WORKERS = int(os.environ.get("SWEEP_WORKERS", "12"))
LOG_ROOT = Path(os.environ.get("SWEEP_LOG_ROOT", "/data/gyeom/coin_challenge/direction_method_logs/full_sweep_v1"))
EPISODES_DIR = LOG_ROOT / "episodes"
IMAGES_DIR = LOG_ROOT / "images"
CACHE_DIR = DIRECTION_ROOT / "artifacts" / "cache"

_manifest_lock = threading.Lock()
_progress_lock = threading.Lock()
_progress = {"done": 0, "total": 0, "start": 0.0}


def _record_path(episode_idx: int, task_type: str) -> Path:
    return EPISODES_DIR / f"{episode_idx:03d}_{task_type}.json"


def _resolved_path(rel_path: str) -> str:
    return str(Path(rel_path).resolve())


def _save_boxed_image(image: np.ndarray, episode_idx: int, task_type: str, candidate_idx: int) -> str:
    out_dir = IMAGES_DIR / f"{episode_idx:03d}_{task_type}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"candidate_{candidate_idx}_boxed.png"
    Image.fromarray(image).save(out_path)
    return str(out_path.resolve())


def run_one(episode_idx: int, task_type: str) -> dict:
    """Runs one (episode_idx, task_type) episode-run to completion, writes its JSON log, and
    returns a small summary dict for the manifest. Never raises -- any exception at any stage
    (construction, loop, env) is caught and recorded as the outcome instead.
    """
    out_path = _record_path(episode_idx, task_type)
    llm_client = LLMClient(MODEL_ID, cache_dir=CACHE_DIR, timeout_s=CALL_TIMEOUT_S, disable_thinking=DISABLE_THINKING)
    oracle = LocalOracleStandIn(llm_client)
    env = QAEnv(oracle, REPO_ROOT / "episodes_train.jsonl", task_type=task_type)

    run_start = time.time()
    image_path_by_hash = {}
    candidate_raw_hashes = []
    outcome = None
    error = None
    questioner = None

    old_obs, info = env.reset(options={"episode_idx": episode_idx})
    episode_id = env.current_episode_data["id"]
    target_description = info["target_description"]
    category = info["category"]
    image_path_by_hash[image_hash(old_obs["image"])] = _resolved_path(
        env.distractors[env.current_distractor_idx]["path"]
    )
    candidate_raw_hashes.append(image_hash(old_obs["image"]))

    try:
        questioner = DirectionMethodQuestioner(info, llm_client=llm_client)
    except Exception as e:  # noqa: BLE001 -- crash-detection harness, must never abort the sweep
        error = f"{e!r}\n{traceback.format_exc()}"
        outcome = "construction_crash"

    if questioner is not None:
        last_hash = candidate_raw_hashes[0]
        for step in range(MAX_LOOP_ITERS):
            try:
                action = questioner.ask_or_conclude(old_obs)
            except Exception as e:  # noqa: BLE001
                error = f"{e!r}\n{traceback.format_exc()}"
                outcome = "loop_crash"
                break

            if (action["question"] is None) == (action["conclusion"] is None):
                error = f"invalid action: {action}"
                outcome = "invalid_action"
                break

            obs, reward, terminated, truncated, info = env.step(action)
            image_path_by_hash[image_hash(obs["image"])] = _resolved_path(
                env.distractors[env.current_distractor_idx]["path"]
            )

            # ENV.md §6 bug 3 (matches eval_model.py's own workaround): observation sometimes
            # fails to switch after a correct conclusion -- discard rather than count as a crash.
            if action["conclusion"] is not None and np.all(obs["image"] == old_obs["image"]) and not terminated and not truncated:
                outcome = "discarded_env_bug"
                break

            if action["question"] is not None:
                questioner.add_answer(obs["answer"] or "")

            h = image_hash(obs["image"])
            if h != last_hash:
                candidate_raw_hashes.append(h)
                last_hash = h

            old_obs = obs
            if terminated or truncated:
                outcome = "truncated" if (truncated and not terminated) else "terminated"
                break
        else:
            outcome = "exceeded_max_loop_iters"

    wall_clock_s = time.time() - run_start

    candidates = []
    if questioner is not None:
        for i, clog in enumerate(questioner.candidate_logs):
            boxed_path = None
            if clog.boxed_image is not None:
                boxed_path = _save_boxed_image(clog.boxed_image, episode_idx, task_type, i)
            candidates.append({
                "index": i,
                "raw_image_path": image_path_by_hash.get(candidate_raw_hashes[i]) if i < len(candidate_raw_hashes) else None,
                "boxed_image_path": boxed_path,
                "bbox_2d": list(clog.bbox_2d) if clog.bbox_2d is not None else None,
                "zone_list": clog.zone_list,
                "scene": clog.scene,
                "checklist_before": clog.checklist_before,
                "checklist_after": clog.checklist_after,
                "questions_asked": clog.questions_asked,
                "self_check_calls": clog.self_check_calls,
                "verdicts": clog.verdicts,
                "conclusion": clog.conclusion,
                "reasoning": clog.reasoning,
                "elapsed_s": clog.elapsed_s,
                "interactions": clog.interactions,
            })

    record = {
        "episode_idx": episode_idx,
        "episode_id": episode_id,
        "task_type": task_type,
        "target_description": target_description,
        "category": category,
        "target_image_path": _resolved_path(env.current_episode_data["path"]),
        "outcome": outcome,
        "error": error,
        "n_distractors": len(env.distractors),
        "n_successes": env.n_successes,
        "full_success": outcome == "terminated" and env.n_successes == len(env.distractors),
        "n_questions": questioner.n_questions if questioner is not None else None,
        "candidates_seen": questioner.candidates_seen if questioner is not None else None,
        "time_required_s": questioner.time_required if questioner is not None else None,
        "wall_clock_s": wall_clock_s,
        "soft_stop_fired": questioner.episode_log["soft_stop_fired"] if questioner is not None else None,
        "hard_stop_fired": questioner.episode_log["hard_stop_fired"] if questioner is not None else None,
        "budget_forced_conclusions": questioner.episode_log["budget_forced_conclusions"] if questioner is not None else None,
        "context_parser": {
            "target_category": questioner.context_parser_result.target_category,
            "target_phrase": questioner.context_parser_result.target_phrase,
            "other_objects": questioner.context_parser_result.other_objects,
            "validation_problems": questioner.context_parser_result.validation_problems,
            "retried": questioner.context_parser_result.retried,
        } if questioner is not None else None,
        "initial_checklist": questioner.initial_checklist if questioner is not None else None,
        "candidates": candidates,
    }

    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(record, f, indent=2)
    tmp_path.replace(out_path)

    return {
        "episode_idx": episode_idx,
        "task_type": task_type,
        "outcome": outcome,
        "full_success": record["full_success"],
        "n_successes": record["n_successes"],
        "n_distractors": record["n_distractors"],
        "wall_clock_s": wall_clock_s,
    }


def _print_progress(summary: dict) -> None:
    with _progress_lock:
        _progress["done"] += 1
        done, total = _progress["done"], _progress["total"]
        elapsed = time.time() - _progress["start"]
        rate = done / elapsed if elapsed > 0 else 0.0
        eta_s = (total - done) / rate if rate > 0 else float("inf")
    print(
        f"[{done:4d}/{total}] ep={summary['episode_idx']:3d} type={summary['task_type']:24s} "
        f"outcome={summary['outcome']!s:22s} success={summary['n_successes']}/{summary['n_distractors']} "
        f"({summary['wall_clock_s']:.1f}s)  elapsed={elapsed / 60:.1f}m eta={eta_s / 60:.1f}m",
        flush=True,
    )


def main() -> None:
    fix_answer_prompt_typo()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    tasks = []
    for episode_idx in range(N_EPISODES):
        for task_type in TASK_TYPES:
            if _record_path(episode_idx, task_type).exists():
                continue
            tasks.append((episode_idx, task_type))

    n_skipped = N_EPISODES * len(TASK_TYPES) - len(tasks)
    print(f"log root: {LOG_ROOT}")
    print(f"{N_EPISODES} episodes x {len(TASK_TYPES)} task types = {N_EPISODES * len(TASK_TYPES)} total runs")
    print(f"{n_skipped} already done (resuming), {len(tasks)} to run now, {N_WORKERS} workers")

    _progress["total"] = len(tasks)
    _progress["start"] = time.time()
    summaries = []

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(run_one, ep, tt): (ep, tt) for ep, tt in tasks}
        for fut in as_completed(futures):
            ep, tt = futures[fut]
            try:
                summary = fut.result()
            except Exception as e:  # noqa: BLE001 -- run_one is designed not to raise, but be safe
                summary = {
                    "episode_idx": ep, "task_type": tt, "outcome": "driver_crash",
                    "full_success": False, "n_successes": 0, "n_distractors": 0, "wall_clock_s": 0.0,
                }
                print(f"[DRIVER CRASH] episode {ep} type {tt}: {e!r}")
            summaries.append(summary)
            _print_progress(summary)

    total_time = time.time() - _progress["start"]
    outcome_counts = {}
    n_full_success = 0
    for s in summaries:
        outcome_counts[s["outcome"]] = outcome_counts.get(s["outcome"], 0) + 1
        if s["full_success"]:
            n_full_success += 1

    manifest = {
        "log_root": str(LOG_ROOT),
        "n_episodes": N_EPISODES,
        "task_types": TASK_TYPES,
        "n_runs_this_invocation": len(tasks),
        "n_skipped_already_done": n_skipped,
        "n_workers": N_WORKERS,
        "total_time_s": total_time,
        "outcome_counts": outcome_counts,
        "n_full_success_this_invocation": n_full_success,
    }
    with _manifest_lock:
        manifest_path = LOG_ROOT / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print("=" * 70)
    print(f"{len(tasks)} runs completed in {total_time / 60:.1f} minutes")
    print(f"outcome counts: {outcome_counts}")
    print(f"full successes this invocation: {n_full_success}/{len(tasks)}")
    print(f"manifest written to {manifest_path}")
    print("dumped to", LOG_ROOT)


if __name__ == "__main__":
    main()
