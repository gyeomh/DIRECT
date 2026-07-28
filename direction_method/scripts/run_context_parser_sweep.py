"""context_parser sweep (SPEC.md §11): 6 description types x N episodes against the real
backend, reporting the post-retry validation flag rate -- the improvement metric for the
other_objects / PARTS-vs-SEPARATE-OBJECTS / NEVER-RELATION-IN-TARGET prompt redesign (§10),
which replaced the earlier design where relational facts kept getting reworded into Target-shaped
sentences and dumped under "Target" instead of their own region key.

Run: `python scripts/run_context_parser_sweep.py` (needs a real vllm server; set VLM_PORT if not
on the default 8000. `VLM_BACKEND=fake` for a dry run).
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

DIRECTION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DIRECTION_ROOT.parent
for p in (DIRECTION_ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from context_parser import parse_context
from llm import LLMClient

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
N_EPISODES = 30
SAMPLE_SEED = 0
DESCRIPTION_TYPES = ["category", "color", "context", "color_feature", "color_context", "color_context_feature"]


def load_episodes(path: Path) -> list:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def main() -> None:
    llm_client = LLMClient(MODEL_ID, cache_dir=DIRECTION_ROOT / "artifacts" / "cache")
    episodes = load_episodes(REPO_ROOT / "episodes_train.jsonl")
    sample = random.Random(SAMPLE_SEED).sample(episodes, min(N_EPISODES, len(episodes)))
    n_per_type = len(sample)

    n_total = 0
    n_retried = Counter()
    n_flagged_final = Counter()
    dumps = []

    for description_type in DESCRIPTION_TYPES:
        for ep in sample:
            description = ep["tasks"][description_type]
            n_total += 1
            try:
                result = parse_context(llm_client, description)
            except Exception as e:  # noqa: BLE001 -- a hard parse failure counts as flagged too
                n_flagged_final[description_type] += 1
                dumps.append({
                    "episode_id": ep["id"], "description_type": description_type,
                    "description": description, "error": str(e),
                })
                continue
            if result.retried:
                n_retried[description_type] += 1
            if result.validation_problems:
                n_flagged_final[description_type] += 1
                dumps.append({
                    "episode_id": ep["id"],
                    "description_type": description_type,
                    "description": description,
                    "other_objects": result.other_objects,
                    "checklist": result.checklist,
                    "validation_problems": result.validation_problems,
                })

    print("=" * 70)
    print(f"context_parser sweep: {len(DESCRIPTION_TYPES)} types x {n_per_type} episodes = {n_total} calls")
    print(f"backend: {llm_client.backend_name}  model: {MODEL_ID}")
    print("=" * 70)
    total_retried = sum(n_retried.values())
    total_flagged = sum(n_flagged_final.values())
    print(f"\nOVERALL retry rate (first attempt flagged): {total_retried}/{n_total} = {total_retried / n_total:.1%}")
    print(f"OVERALL post-retry flag rate: {total_flagged}/{n_total} = {total_flagged / n_total:.1%}")
    print("\nPer description type:")
    for t in DESCRIPTION_TYPES:
        print(f"  {t:22s} retried={n_retried[t]}/{n_per_type}  flagged(post-retry)={n_flagged_final[t]}/{n_per_type}")

    out_path = DIRECTION_ROOT / "artifacts" / "context_parser_sweep_flags.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dumps, indent=2))
    print(f"\nAll {len(dumps)} flagged/failed cases dumped to {out_path} for manual inspection.")
    print("No automatic reclassification. Reading the dump is a human step.")


if __name__ == "__main__":
    main()
