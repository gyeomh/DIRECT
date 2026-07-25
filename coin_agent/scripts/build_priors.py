#!/usr/bin/env python3
"""Builds artifacts/priors.json (spec §6.2). STUB: aggregation math is implemented; the actual
run needs extract.EXTRACTION_PROMPT written first (this calls extract() over every candidate
image in the training set — a one-time offline cost, cached to artifacts/cache/ per llm.py).

    disc(s, v, category) = 1 - P(a co-episode candidate shares value v | category)

Procedure (spec §6.2): run extract() over all candidate images in the training episodes; for
each episode and slot, compute the fraction of candidate pairs sharing the canonical value;
aggregate by category with Laplace smoothing (config.yaml: priors.prior_alpha).
"""

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent
for p in (_PACKAGE_ROOT, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from agent import schema  # noqa: E402
from agent.extract import extract  # noqa: E402
from agent.llm import LLMClient  # noqa: E402
from agent.priors import PriorsTable  # noqa: E402
from agent.questioner import load_config  # noqa: E402

JSONL_PATH = _REPO_ROOT / "episodes_train.jsonl"


def load_episodes(path: Path = JSONL_PATH) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def build(config: dict, max_adjacent: int, llm_client: LLMClient) -> PriorsTable:
    """For each (category, slot), fraction of within-episode candidate pairs sharing a value,
    aggregated with Laplace smoothing. Extraction is cached (llm.py), so re-running this after a
    partial failure is cheap — already-extracted candidates come straight from disk cache.
    """
    episodes = load_episodes()
    alpha = config["priors"]["prior_alpha"]

    # {(category, slot_key): {value: co_occurrence_count}}, plus total pair counts per (category, slot)
    shares: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))
    pair_totals: dict[tuple, int] = defaultdict(int)

    for ep in episodes:
        category = ep["category"]
        image_paths = [ep["path"]] + [d["path"] for d in ep["distractors"]]
        frames = []
        for path in image_paths:
            image = np.array(Image.open(_REPO_ROOT / path))
            frames.append(extract(image, llm_client, max_adjacent=max_adjacent))

        for slot_key in schema.all_slot_keys(max_adjacent):
            for fa, fb in combinations(frames, 2):
                va, vb = fa.get(slot_key), fb.get(slot_key)
                if va.canon is None or vb.canon is None:
                    continue
                pair_totals[(category, slot_key)] += 1
                if va.canon == vb.canon:
                    shares[(category, slot_key)][va.canon] += 1

    table_dict: dict = defaultdict(lambda: defaultdict(dict))
    for (category, slot_key), total in pair_totals.items():
        for value, shared_count in shares[(category, slot_key)].items():
            p_share = (shared_count + alpha) / (total + 2 * alpha)  # Laplace smoothing
            table_dict[category][slot_key][value] = round(1.0 - p_share, 4)

    return PriorsTable({k: dict(v) for k, v in table_dict.items()}, default_disc=config["priors"]["default_disc"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(_PACKAGE_ROOT / "artifacts" / "priors.json"))
    args = parser.parse_args()

    config = load_config()
    llm_client = LLMClient(
        config["vllm"]["model_id"],
        port=config["vllm"]["port"],
        temperature=config["vllm"]["temperature"],
        cache_dir=_PACKAGE_ROOT / config["llm"]["cache_dir"],
        timeout_s=config["llm"]["llm_timeout_s"],
        retries=config["llm"]["llm_retries"],
    )
    table = build(config, config["schema"]["max_adjacent"], llm_client)
    table.save(args.out)
    print(f"Wrote priors to {args.out}")


if __name__ == "__main__":
    main()
