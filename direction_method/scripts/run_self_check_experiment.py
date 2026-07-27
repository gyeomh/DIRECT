"""SPEC.md §7 self_check polarity experiment. Built now, meant to run unattended tomorrow with a
live vllm server: `python scripts/run_self_check_experiment.py`.

Does NOT run automatically as part of any test suite, and does NOT auto-classify failures or
pick a polarity -- it reports the false-failure rate per polarity, dumps every failure to
artifacts/self_check_failures.json for manual inspection, and reports the oracle answer
word-length distribution. Picking a polarity is SPEC's own explicit instruction to make on the
data, by hand, after reading the dumped failures.

Procedure: for 25-30 sampled training episodes, ask the local oracle stand-in -- against the
TARGET image, which is what the real oracle also sees -- the mandatory first question (SPEC §6)
plus the left/right/above relation questions, using the category-type description as "the
target noun" (SPEC §5-1: ground with the category noun only). Ground truth for every one of
these self_check calls is "pass": the statement (the oracle's own answer) is being checked
against the exact same image it was derived from, so a truthful oracle answering honestly can
never actually contradict it. Any failure is attributable to self_check, not to the oracle or a
real image mismatch -- which is exactly what makes this a clean measurement of each polarity's
false-failure rate.
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

import numpy as np
from PIL import Image

from llm import LLMClient
from oracle_stub import LocalOracleStandIn
from patches.apply_patches import fix_answer_prompt_typo
from self_check import POLARITIES, is_failure, self_check

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
N_EPISODES = 30
SAMPLE_SEED = 0
RELATIONS = ["left", "right", "above"]

FIRST_QUESTION_TEMPLATE = "Can you describe the {TARGET}'s location and visual appearance (e.g., color, shape, size)."
RELATION_QUESTION_TEMPLATE = "What is on the {RELATION} of the {TARGET}? Can you describe the shape and color?"


def load_episodes(path: Path) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def questions_for(target_noun: str) -> list[str]:
    return [FIRST_QUESTION_TEMPLATE.format(TARGET=target_noun)] + [
        RELATION_QUESTION_TEMPLATE.format(RELATION=r, TARGET=target_noun) for r in RELATIONS
    ]


def main() -> None:
    fix_answer_prompt_typo()

    episodes = load_episodes(REPO_ROOT / "episodes_train.jsonl")
    sample = random.Random(SAMPLE_SEED).sample(episodes, min(N_EPISODES, len(episodes)))

    llm_client = LLMClient(MODEL_ID, cache_dir=DIRECTION_ROOT / "artifacts" / "cache")
    oracle = LocalOracleStandIn(llm_client)

    verdict_counts = {p: Counter() for p in POLARITIES}
    failures = {p: [] for p in POLARITIES}
    word_lengths = []
    n_statements = 0

    for ep in sample:
        target_image = np.array(Image.open(REPO_ROOT / ep["path"]))
        target_noun = ep["tasks"]["category"]

        for question in questions_for(target_noun):
            answer = oracle.ask_question(question, target_image)
            word_lengths.append(len(answer.split()))
            n_statements += 1

            for polarity in POLARITIES:
                verdict = self_check(llm_client, target_image, answer, polarity)
                verdict_counts[polarity][verdict] += 1
                if is_failure(polarity, verdict):
                    failures[polarity].append({
                        "episode_id": ep["id"],
                        "image_path": ep["path"],
                        "target_noun": target_noun,
                        "question": question,
                        "answer": answer,
                        "verdict": verdict,
                    })

    print("=" * 70)
    print(f"self_check polarity experiment: {len(sample)} episodes x {len(RELATIONS) + 1} questions = {n_statements} statements")
    print(f"backend: {llm_client.backend_name}  model: {MODEL_ID}")
    print("=" * 70)

    for polarity in POLARITIES:
        n_fail = len(failures[polarity])
        print(f"\n[{polarity}]")
        print(f"  verdict distribution: {dict(verdict_counts[polarity])}")
        rate = n_fail / n_statements if n_statements else 0.0
        print(f"  false-failure rate (ground truth is always 'pass'): {n_fail}/{n_statements} = {rate:.1%}")

    out_dir = DIRECTION_ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    failures_path = out_dir / "self_check_failures.json"
    with open(failures_path, "w") as f:
        json.dump(failures, f, indent=2)
    print(f"\nAll failures dumped to {failures_path} for manual inspection.")

    wl = Counter(word_lengths)
    mean_wl = sum(word_lengths) / len(word_lengths) if word_lengths else 0.0
    print(f"\noracle answer word-length distribution: {dict(sorted(wl.items()))}")
    print(f"mean word length: {mean_wl:.2f}  (ENV.md: answers are requested to stay under 15 words -- not enforced in code)")

    print("\nNo polarity has been auto-selected. Read artifacts/self_check_failures.json and pick by hand.")


if __name__ == "__main__":
    main()
