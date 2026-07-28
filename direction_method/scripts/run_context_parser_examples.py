"""context_parser, run in isolation on a handful of descriptions and printed for inspection
(SPEC.md build order step 3). Text-only -- no images involved. Includes SPEC.md §10's own worked
examples (so a real run can be checked against the expected checklist shape by eye) plus a few
raw descriptions pulled from episodes_train.jsonl.

Run: `python scripts/run_context_parser_examples.py` (or `VLM_BACKEND=fake ...` for a dry run).
"""

import json
import random
import sys
from pathlib import Path

DIRECTION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DIRECTION_ROOT.parent
for p in (DIRECTION_ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from context_parser import parse_context
from llm import LLMClient

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
N_FROM_EPISODES = 5
SAMPLE_SEED = 0

# SPEC.md §10's own worked examples -- expected checklists are in the prompt/spec, not re-stated
# here, so a real run's output can be eyeballed against them directly.
SPEC_EXAMPLES = [
    "Kitchen lower cabinet",
    "Navy blue kitchen lower cabinet with brass handles",
    "Kitchen lower cabinet situated beneath a white countertop",
    "Navy blue kitchen lower cabinet under a white farmhouse sink",
    "White bed with a blue blanket next to a nightstand",
    "Green display cabinet next to open shelving",
]


def load_episode_descriptions(path: Path, n: int, seed: int) -> list:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    sample = random.Random(seed).sample(episodes, min(n, len(episodes)))
    # color_context_feature is the richest description variant -- most likely to exercise
    # multiple checklist keys at once.
    return [ep["tasks"]["color_context_feature"] for ep in sample]


def main() -> None:
    llm_client = LLMClient(MODEL_ID, cache_dir=DIRECTION_ROOT / "artifacts" / "cache")
    descriptions = SPEC_EXAMPLES + load_episode_descriptions(
        REPO_ROOT / "episodes_train.jsonl", N_FROM_EPISODES, SAMPLE_SEED
    )

    print(f"context_parser: {len(descriptions)} descriptions, backend={llm_client.backend_name}, model={MODEL_ID}")
    print("=" * 70)
    n_errors = 0
    for description in descriptions:
        print(f"\n> {description!r}")
        try:
            result = parse_context(llm_client, description)
        except Exception as e:  # noqa: BLE001 -- one bad example must not stop the rest
            n_errors += 1
            print(f"  ERROR: {e}")
            continue
        print(f"  target_category: {result.target_category!r}")
        print(f"  target_phrase:   {result.target_phrase!r}")
        print(f"  checklist:       {result.checklist}")

    if n_errors:
        print(f"\n{n_errors}/{len(descriptions)} descriptions raised an error -- see above.")


if __name__ == "__main__":
    main()
