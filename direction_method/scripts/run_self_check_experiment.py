"""self_check experiment, per the decided design (SPEC.md §7). Built now, meant to run
unattended tomorrow with a live vllm server: `python scripts/run_self_check_experiment.py`.

Polarity is settled (yes/no, "does the image contradict the assertion") -- this is no longer a
comparison between candidate framings. It measures ONE thing: the false-"no" rate, since ground
truth is always "yes" here (the assertion is checked against the exact image it was derived
from, via the local oracle stand-in, which only ever sees the target image). Reports the rate
overall and broken out per question type (target-appearance vs each of left/right/above), and
dumps every "no" (and every parse failure) with image path, region, assertion, evidence, and
verdict for manual inspection. Does not auto-classify failures.
"""

import json
import random
import sys
from collections import Counter, defaultdict
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
from self_check import is_failure, region_for, self_check

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
N_EPISODES = 30
SAMPLE_SEED = 0
RELATIONS = ["left", "right", "above"]
TARGET_QUESTION_TYPE = "target_appearance"  # the mandatory first question (SPEC.md §6)

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


def questions_for(target_noun: str) -> list[tuple[str, str, str]]:
    """Returns (question_type, relation_key, question_text) triples. relation_key is what
    region_for() expects: "Target" for the mandatory first question, else the relation itself.
    """
    out = [(TARGET_QUESTION_TYPE, "Target", FIRST_QUESTION_TEMPLATE.format(TARGET=target_noun))]
    for r in RELATIONS:
        out.append((r, r, RELATION_QUESTION_TEMPLATE.format(RELATION=r, TARGET=target_noun)))
    return out


def main() -> None:
    fix_answer_prompt_typo()

    episodes = load_episodes(REPO_ROOT / "episodes_train.jsonl")
    sample = random.Random(SAMPLE_SEED).sample(episodes, min(N_EPISODES, len(episodes)))

    llm_client = LLMClient(MODEL_ID, cache_dir=DIRECTION_ROOT / "artifacts" / "cache")
    oracle = LocalOracleStandIn(llm_client)

    verdict_counts_by_type = defaultdict(Counter)
    n_statements_by_type = Counter()
    failures = []
    word_lengths = []

    for ep in sample:
        target_image = np.array(Image.open(REPO_ROOT / ep["path"]))
        target_noun = ep["tasks"]["category"]  # SPEC §5-1: ground with the category noun only

        for question_type, relation_key, question in questions_for(target_noun):
            answer = oracle.ask_question(question, target_image)
            word_lengths.append(len(answer.split()))

            region = region_for(relation_key, target_noun)
            result = self_check(llm_client, target_image, region, answer)

            n_statements_by_type[question_type] += 1
            verdict_counts_by_type[question_type][result.verdict] += 1

            if is_failure(result.verdict):
                failures.append({
                    "episode_id": ep["id"],
                    "image_path": ep["path"],
                    "question_type": question_type,
                    "question": question,
                    "region": region,
                    "assertion": answer,
                    "evidence": result.evidence,
                    "verdict": result.verdict,
                })

    n_total = sum(n_statements_by_type.values())
    n_fail_total = len(failures)

    print("=" * 70)
    print(f"self_check experiment: {len(sample)} episodes x {len(RELATIONS) + 1} questions = {n_total} statements")
    print(f"backend: {llm_client.backend_name}  model: {MODEL_ID}")
    print("=" * 70)

    print(f"\nOVERALL false-'no' rate (ground truth is always 'yes'): {n_fail_total}/{n_total} = {n_fail_total/n_total:.1%}" if n_total else "no data")

    print("\nPer question type:")
    for qtype in [TARGET_QUESTION_TYPE, *RELATIONS]:
        n_type = n_statements_by_type[qtype]
        n_fail_type = sum(1 for f in failures if f["question_type"] == qtype)
        rate = n_fail_type / n_type if n_type else 0.0
        print(f"  {qtype:20s} verdicts={dict(verdict_counts_by_type[qtype])}  false-no rate={n_fail_type}/{n_type} = {rate:.1%}")

    out_dir = DIRECTION_ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    failures_path = out_dir / "self_check_failures.json"
    with open(failures_path, "w") as f:
        json.dump(failures, f, indent=2)
    print(f"\nAll {n_fail_total} failures dumped to {failures_path} for manual inspection.")

    wl = Counter(word_lengths)
    mean_wl = sum(word_lengths) / len(word_lengths) if word_lengths else 0.0
    print(f"\noracle answer word-length distribution: {dict(sorted(wl.items()))}")
    print(f"mean word length: {mean_wl:.2f}  (ENV.md: answers are requested to stay under 15 words -- not enforced in code)")

    print("\nNo automatic classification of failures. Read artifacts/self_check_failures.json by hand.")


if __name__ == "__main__":
    main()
