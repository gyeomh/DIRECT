#!/usr/bin/env python3
"""Offline data work (no model needed): candidates-per-episode distribution, category
distribution, and a per-description-type attribute-count heuristic. Fully runnable now.
"""

import json
import re
from collections import Counter
from pathlib import Path

DIRECTION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DIRECTION_ROOT.parent
EPISODES_PATH = REPO_ROOT / "episodes_train.jsonl"
OUT_PATH = DIRECTION_ROOT / "artifacts" / "episode_stats.json"

DESCRIPTION_TYPES = ["category", "color", "context", "color_feature", "color_context", "color_context_feature"]

# No model available -- this is a coarse, explicitly-labeled heuristic, not a verified attribute
# count (that would need context_parser, which doesn't exist yet). Counts recognizable
# color/material words plus this method's own relation vocabulary (SPEC.md §3) as a proxy for
# "how many distinct, checkable facts this description packs in." Real numbers will differ once
# context_parser actually runs; this is only meant to calibrate expectations before then.
_ATTRIBUTE_KEYWORDS = {
    # colors
    "white", "black", "navy", "blue", "red", "green", "yellow", "orange", "purple", "pink",
    "brown", "tan", "beige", "cream", "gray", "grey", "gold", "silver", "multicolor", "multicolored",
    # materials
    "wood", "wooden", "metal", "glass", "stone", "marble", "granite", "tile", "fabric", "leather",
    "brass", "chrome", "steel", "ceramic",
    # SPEC.md §3's own relation vocabulary (as words that appear in descriptions, not the enum itself)
    "on", "above", "below", "under", "next", "beside", "near", "behind", "left", "right", "top", "bottom",
}


def load_episodes(path: Path = EPISODES_PATH) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def attribute_count_heuristic(text: str) -> int:
    words = re.findall(r"[a-z]+", text.lower())
    return sum(1 for w in words if w in _ATTRIBUTE_KEYWORDS)


def summarize(values: list[float]) -> dict:
    if not values:
        return {}
    return {"mean": round(sum(values) / len(values), 2), "min": min(values), "max": max(values)}


def main() -> None:
    episodes = load_episodes()
    n = len(episodes)

    candidates_dist = Counter(len(ep["distractors"]) for ep in episodes)
    category_dist = Counter(ep["category"] for ep in episodes)

    desc_len_by_type = {t: [] for t in DESCRIPTION_TYPES}
    attr_count_by_type = {t: [] for t in DESCRIPTION_TYPES}
    for ep in episodes:
        for t in DESCRIPTION_TYPES:
            text = ep["tasks"][t]
            desc_len_by_type[t].append(len(text.split()))
            attr_count_by_type[t].append(attribute_count_heuristic(text))

    report = {
        "n_episodes": n,
        "candidates_per_episode_distribution": dict(sorted(candidates_dist.items())),
        "category_distribution": dict(category_dist.most_common()),
        "n_distinct_categories": len(category_dist),
        "description_length_words_by_type": {t: summarize(v) for t, v in desc_len_by_type.items()},
        "attribute_count_heuristic_by_type": {t: summarize(v) for t, v in attr_count_by_type.items()},
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Analyzed {n} episodes -> {OUT_PATH}\n")
    print("Candidates per episode:", report["candidates_per_episode_distribution"])
    print(f"\nCategories ({report['n_distinct_categories']} distinct), top 10:", category_dist.most_common(10))
    print("\nDescription length (words) by type:")
    for t, s in report["description_length_words_by_type"].items():
        print(f"  {t}: {s}")
    print("\nAttribute-count heuristic (keyword-based, NOT verified against a real parser) by type:")
    for t, s in report["attribute_count_heuristic_by_type"].items():
        print(f"  {t}: {s}")


if __name__ == "__main__":
    main()
