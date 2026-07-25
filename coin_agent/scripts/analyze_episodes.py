#!/usr/bin/env python3
"""Reads episodes_train.jsonl and reports the stats spec §6.1 asks for. Fully runnable now — no
VLM/network required. Read this output before writing the policy (spec §10 step 4).
"""

import json
import sys
from collections import Counter
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from agent import canon, schema  # noqa: E402

REPO_ROOT = _PACKAGE_ROOT.parent
JSONL_PATH = REPO_ROOT / "episodes_train.jsonl"
OUT_PATH = _PACKAGE_ROOT / "artifacts" / "episode_stats.json"

TASK_TYPES = ["category", "color", "context", "color_context_feature", "color_feature", "color_context"]
MAX_ADJACENT = 3  # matches config.yaml schema.max_adjacent; kept as a literal here to stay dependency-free


def load_episodes(path: Path = JSONL_PATH) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def tier_a_slots_mentioned(description: str) -> int:
    """Heuristic count of Tier-A slots this description text already specifies. Real slot
    parsing needs an LLM call we haven't written the prompt for yet (parse.DESCRIPTION_PARSE_PROMPT)
    — this uses canon.find_in_text's substring search as a stand-in, which will under-count
    anything phrased outside the current synonym tables. `obj.category` is counted unconditionally
    since it's always known from `info["category"]` regardless of description type (env.py:135).
    """
    count = 1  # obj.category
    for slot_key in schema.all_slot_keys(MAX_ADJACENT):
        if slot_key == "obj.category":
            continue
        spec = schema.spec_for(slot_key)
        if spec.tier != schema.TIER_A:
            continue
        if spec.type in schema.VOCAB and canon.find_in_text(description, spec.type) is not None:
            count += 1
    return count


def summarize(values: list[float]) -> dict:
    if not values:
        return {}
    return {"mean": round(sum(values) / len(values), 2), "min": min(values), "max": max(values)}


def main() -> None:
    episodes = load_episodes()
    n = len(episodes)

    candidates_per_episode = Counter(len(ep["distractors"]) for ep in episodes)
    match_true_counts = Counter(sum(1 for d in ep["distractors"] if d["match"]) for ep in episodes)

    positions = [i for ep in episodes for i, d in enumerate(ep["distractors"]) if d["match"]]
    position_counts = Counter(positions)

    category_freq = Counter(ep["category"] for ep in episodes)

    desc_len_by_type = {t: [] for t in TASK_TYPES}
    tier_a_mentioned_by_type = {t: [] for t in TASK_TYPES}
    for ep in episodes:
        for t in TASK_TYPES:
            text = ep["tasks"].get(t, "")
            desc_len_by_type[t].append(len(text.split()))
            tier_a_mentioned_by_type[t].append(tier_a_slots_mentioned(text))

    n_tier_a = sum(1 for k in schema.all_slot_keys(MAX_ADJACENT) if schema.spec_for(k).tier == schema.TIER_A)

    report = {
        "n_episodes": n,
        "n_tier_a_slots_total": n_tier_a,
        "candidates_per_episode_distribution": dict(sorted(candidates_per_episode.items())),
        "match_true_count_distribution": dict(sorted(match_true_counts.items())),
        "match_true_position_distribution_AWARENESS_ONLY": dict(sorted(position_counts.items())),
        "category_frequency": dict(category_freq.most_common()),
        "description_length_words": {t: summarize(v) for t, v in desc_len_by_type.items()},
        "tier_a_slots_mentioned_heuristic": {t: summarize(v) for t, v in tier_a_mentioned_by_type.items()},
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Analyzed {n} episodes -> {OUT_PATH}")
    print()
    print("Candidates per episode:", report["candidates_per_episode_distribution"])
    print("match=True count per episode (expect exactly {1: n}):", report["match_true_count_distribution"])
    print()
    print("[WARNING] match=True POSITION distribution is reported for awareness only (spec §6.1).")
    print("No model component may condition on candidate position or episode index (spec §0.6).")
    print("Position distribution:", report["match_true_position_distribution_AWARENESS_ONLY"])
    print()
    print(f"Categories ({len(category_freq)} distinct), top 10:", category_freq.most_common(10))
    print()
    print("Description length (words) by type:")
    for t, s in report["description_length_words"].items():
        print(f"  {t}: {s}")
    print()
    print(f"Tier-A slots mentioned in description, heuristic (of {n_tier_a} total) — calibrates budget.ambiguity_allowance:")
    for t, s in report["tier_a_slots_mentioned_heuristic"].items():
        print(f"  {t}: {s}")


if __name__ == "__main__":
    main()
