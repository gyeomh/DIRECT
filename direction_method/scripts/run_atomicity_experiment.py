"""Experiment: is context_parser.py's RULE 1 (ATOMIC -- "one fact per assertion, split
compounds") actually helping, or is the model just bad at doing the split? Concern raised after
reviewing sweep output: atomic decomposition looked unreliable on multi-attribute descriptions.

Builds a second prompt variant, identical to CONTEXT_PARSER_PROMPT except RULE 1 is replaced with
"keep the target's attributes as ONE combined assertion, do not split" (and the few-shot examples
are updated to match -- teaching by example matters more than the rule text, per this module's own
other_objects lesson). Same schema, same merge logic (_merge_other_objects_into_checklist is
imported straight from context_parser.py, unchanged) -- only the Target-splitting instruction
differs.

Runs BOTH variants on the identical sample run_context_parser_sweep.py uses (6 description types
x 30 episodes, seed 0) against the real vllm server, so this is directly comparable to that
sweep's own numbers. Reports the validator flag rate for each (sanity only -- the validator
checks relation-in-Target / other_objects-in-checklist, not atomicity, so it isn't expected to
move much) and, more to the point, prints every case where the atomic variant actually split
Target into 2+ assertions side-by-side with the non-atomic variant's single combined assertion,
for manual read-through -- that's where any splitting-quality difference would actually show up.

Run: `VLM_PORT=8002 python scripts/run_atomicity_experiment.py`
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

from context_parser import (  # noqa: E402
    CONTEXT_PARSER_PROMPT,
    CONTEXT_PARSER_SCHEMA,
    ContextParserError,
    _merge_other_objects_into_checklist,
    _validate,
    parse_context,
)
from llm import LLMClient  # noqa: E402

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
N_EPISODES = 30
SAMPLE_SEED = 0
DESCRIPTION_TYPES = ["category", "color", "context", "color_feature", "color_context", "color_context_feature"]

# Same prompt, RULE 1 replaced, and the three multi-attribute examples updated so the combined
# assertion is what the few-shot examples actually demonstrate.
NON_ATOMIC_PROMPT = (
    CONTEXT_PARSER_PROMPT
    .replace(
        "1. ATOMIC. One fact per assertion. Split compounds.\n"
        '   "navy blue with brass handles" -> "it is navy blue", "it has brass handles"',
        "1. ONE COMBINED ASSERTION. Do not split the target's attributes into separate\n"
        "   facts -- state everything the description says about the target's own\n"
        '   attributes in a single assertion.\n'
        '   "navy blue with brass handles" -> "it is navy blue and has brass handles"',
    )
    .replace(
        '"checklist": {"Target": ["it is navy blue", "it has brass handles"]}}',
        '"checklist": {"Target": ["it is navy blue and has brass handles"]}}',
    )
    .replace(
        '"checklist": {"Target": ["it is dark gray", "it is slatted"]}}',
        '"checklist": {"Target": ["it is dark gray and slatted"]}}',
    )
    .replace(
        '"checklist": {"Target": ["it is large", "it is beige"]}}',
        '"checklist": {"Target": ["it is large and beige"]}}',
    )
)
assert NON_ATOMIC_PROMPT != CONTEXT_PARSER_PROMPT
assert "ONE COMBINED ASSERTION" in NON_ATOMIC_PROMPT
assert "it is navy blue and has brass handles" in NON_ATOMIC_PROMPT


def call_non_atomic(llm_client: LLMClient, description: str) -> dict:
    prompt = f"{NON_ATOMIC_PROMPT}\n\n{description}"
    result = llm_client.call(prompt, image=None, response_schema=CONTEXT_PARSER_SCHEMA, use_cache=True)
    parsed = json.loads(result.text)  # let a malformed response raise -- same as context_parser's own
    checklist = _merge_other_objects_into_checklist(parsed["other_objects"], parsed["checklist"])
    return {
        "target_category": parsed["target_category"],
        "target_phrase": parsed["target_phrase"],
        "other_objects": parsed["other_objects"],
        "checklist": checklist,
    }


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

    n_total = 0
    n_flagged = {"atomic": Counter(), "non_atomic": Counter()}
    n_errors = {"atomic": Counter(), "non_atomic": Counter()}
    target_assertion_counts = {"atomic": [], "non_atomic": []}
    divergent_cases = []  # atomic split Target into 2+ assertions

    for description_type in DESCRIPTION_TYPES:
        for ep in sample:
            description = ep["tasks"][description_type]
            n_total += 1

            try:
                atomic_result = parse_context(llm_client, description)
                atomic_target = atomic_result.checklist.get("Target", [])
                if atomic_result.validation_problems:
                    n_flagged["atomic"][description_type] += 1
            except ContextParserError:
                n_errors["atomic"][description_type] += 1
                atomic_target = None

            try:
                non_atomic_parsed = call_non_atomic(llm_client, description)
                non_atomic_target = non_atomic_parsed["checklist"].get("Target", [])
                if _validate(non_atomic_parsed["checklist"]):
                    n_flagged["non_atomic"][description_type] += 1
            except (json.JSONDecodeError, KeyError, TypeError):
                n_errors["non_atomic"][description_type] += 1
                non_atomic_target = None

            if atomic_target is not None:
                target_assertion_counts["atomic"].append(len(atomic_target))
            if non_atomic_target is not None:
                target_assertion_counts["non_atomic"].append(len(non_atomic_target))

            if atomic_target is not None and len(atomic_target) >= 2:
                divergent_cases.append({
                    "episode_id": ep["id"],
                    "description_type": description_type,
                    "description": description,
                    "atomic_target": atomic_target,
                    "non_atomic_target": non_atomic_target,
                })

    print("=" * 70)
    print(f"atomicity experiment: {len(DESCRIPTION_TYPES)} types x {len(sample)} episodes = {n_total} calls per variant")
    print("=" * 70)

    for variant in ("atomic", "non_atomic"):
        total_flagged = sum(n_flagged[variant].values())
        total_errors = sum(n_errors[variant].values())
        counts = target_assertion_counts[variant]
        avg_assertions = sum(counts) / len(counts) if counts else 0.0
        print(f"\n[{variant}]")
        print(f"  validator flag rate: {total_flagged}/{n_total} = {total_flagged / n_total:.1%}")
        print(f"  parse errors: {total_errors}/{n_total} = {total_errors / n_total:.1%}")
        print(f"  avg Target assertions per description (non-empty + empty): {avg_assertions:.2f}")

    print(f"\n{len(divergent_cases)} cases where the atomic variant split Target into 2+ assertions:")
    print("-" * 70)
    for case in divergent_cases:
        print(f"\n[{case['description_type']}] {case['description']!r}")
        print(f"  atomic:     {case['atomic_target']}")
        print(f"  non_atomic: {case['non_atomic_target']}")

    out_path = DIRECTION_ROOT / "artifacts" / "atomicity_experiment_divergent_cases.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(divergent_cases, indent=2))
    print(f"\nAll {len(divergent_cases)} divergent cases dumped to {out_path}.")


if __name__ == "__main__":
    main()
