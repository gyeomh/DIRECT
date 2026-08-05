"""Emit the challenge's summary-JSON format from eval_model.py's gzip logs.

The format (field names and the scoring block) comes from a run seen outside this repo -- it is
NOT produced by upstream `eval_model.py`, which only writes the per-episode gzip log, and its
scoring is NOT `env.py`'s reward (+10 correct / -10 wrong / -1 per question). Reconstructed from a
worked example and verified against it arithmetically:

    accuracy    = successful_episodes / evaluated_episodes      43/167  = 0.25748...
    total_score = sum over SUCCESSFUL episodes of (10 - questions_in_that_episode)
                                                                 430-1  = 429
    mean_score  = total_score / evaluated_episodes              429/167 = 2.5688...
    failure     = 0 (a failed episode contributes nothing, and is not penalized)

The consequence worth stating plainly: questions are a direct, linear deduction from the score of
an episode you already won. `env.py`'s own reward says the same thing at -1/question, but this
scoring has no -10 for a wrong answer, so guessing costs nothing while asking always costs.

`successful_episodes` counts COMPLETE successes -- every candidate in the episode concluded
correctly -- which is `n_successes == len(distractors)` for that episode id.

Denominator note (ENV.md §6 bug 3): episodes the harness discards never reach the log. They are
counted here in `evaluated_episodes` and score 0, since that is what the summary format implies;
`discarded_episodes` is reported alongside so the number is never silently absorbed.
"""

import argparse
import glob
import gzip
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_episode_candidate_counts(episodes_path: Path) -> dict:
    with open(episodes_path) as f:
        return {e["id"]: len(e["distractors"]) for e in (json.loads(l) for l in f if l.strip())}


def summarize(log_path: Path, episodes_path: Path, *, questioner: str, oracle_backend: str,
              oracle_model_id: str, total_episodes: int) -> dict:
    counts = load_episode_candidate_counts(episodes_path)
    with gzip.GzipFile(log_path) as f:
        d = json.load(f)

    ids, succ, quest = d["id"], d["n_successes"], d["n_questions"]
    complete = [(i, q) for i, s, q in zip(ids, succ, quest) if s == counts[i]]

    total_score = sum(10 - q for _, q in complete)

    # env.py's own reward, the ONLY scoring formula defined anywhere in the upstream repo
    # (_compute_reward: +10 per correct conclusion, -10 per wrong one, -1 per question). The
    # README states the objective in words -- "maximize the number of correct conclusions while
    # asking as few questions as possible" -- but gives no formula, so both are reported and
    # neither is presented as "the" official number.
    #
    # An episode ends on its FIRST wrong conclusion (ENV.md §1), so an episode that did not fully
    # succeed contains exactly one wrong conclusion, unless it ran out of steps/time instead.
    # Truncation is not distinguishable from the gzip log, so this treats every incomplete episode
    # as having taken the -10; that is the pessimistic reading and is flagged as such.
    complete_ids = {i for i, _ in complete}
    env_reward = 0
    for i, s, q in zip(ids, succ, quest):
        env_reward += 10 * s - q
        if i not in complete_ids:
            env_reward -= 10
    description_type = os.path.basename(log_path).split("_train_")[0].split("Questioner_")[-1]

    return {
        "episodes_file": str(episodes_path),
        "description_type": description_type,
        "questioner": questioner,
        "oracle_backend": oracle_backend,
        "oracle_model_id": oracle_model_id,
        "evaluated_episodes": total_episodes,
        "successful_episodes": len(complete),
        "accuracy": len(complete) / total_episodes,
        "total_score": total_score,
        "mean_score": total_score / total_episodes,
        "total_questions": sum(quest),
        "scoring": {"complete_success": "10 - number_of_questions", "failure": 0},
        # Secondary, from env.py's _compute_reward -- the repo's only numeric scoring. Reported
        # alongside because the README defines the objective in words but no formula at all.
        "_env_reward_total": env_reward,
        "_env_reward_mean": env_reward / total_episodes,
        "_env_reward_scoring": "+10 per correct conclusion, -10 per wrong one, -1 per question",
        # Not part of the reference format -- kept so the discarded-episode denominator (ENV.md §6
        # bug 3) is never lost, and so accuracy over completed episodes stays visible next to it.
        "_logged_episodes": len(ids),
        "_discarded_episodes": total_episodes - len(ids),
        "_accuracy_over_logged": len(complete) / len(ids) if ids else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-glob", default=str(REPO_ROOT / "results" / "*_train_0_167.gzip.json"))
    ap.add_argument("--episodes", default=str(REPO_ROOT / "episodes_train.jsonl"))
    ap.add_argument("--questioner", default="direction_method")
    ap.add_argument("--oracle-backend", default="vllm")
    ap.add_argument("--oracle-model-id", required=True)
    ap.add_argument("--total-episodes", type=int, default=167)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "summary"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for p in sorted(glob.glob(args.results_glob)):
        s = summarize(
            Path(p), Path(args.episodes),
            questioner=args.questioner, oracle_backend=args.oracle_backend,
            oracle_model_id=args.oracle_model_id, total_episodes=args.total_episodes,
        )
        summaries.append(s)
        out = out_dir / f"{s['description_type']}.json"
        out.write_text(json.dumps(s, indent=2))
        print(json.dumps(s, indent=2))

    if summaries:
        ev = sum(s["evaluated_episodes"] for s in summaries)
        sc = sum(s["total_score"] for s in summaries)
        su = sum(s["successful_episodes"] for s in summaries)
        overall = {
            "description_type": "ALL",
            "questioner": args.questioner,
            "oracle_model_id": args.oracle_model_id,
            "evaluated_episodes": ev,
            "successful_episodes": su,
            "accuracy": su / ev,
            "total_score": sc,
            "mean_score": sc / ev,
            "total_questions": sum(s["total_questions"] for s in summaries),
            "scoring": {"complete_success": "10 - number_of_questions", "failure": 0},
            "_env_reward_total": sum(s["_env_reward_total"] for s in summaries),
            "_env_reward_scoring": "+10 per correct conclusion, -10 per wrong one, -1 per question",
        }
        (out_dir / "ALL.json").write_text(json.dumps(overall, indent=2))
        print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
