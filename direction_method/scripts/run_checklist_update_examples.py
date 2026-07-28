"""checklist_update, run in isolation on a handful of hand-built cases and printed for inspection
(SPEC.md build order step 3). Text-only -- no images involved.

Run: `python scripts/run_checklist_update_examples.py` (or `VLM_BACKEND=fake ...` for a dry run).
"""

import sys
from pathlib import Path

DIRECTION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DIRECTION_ROOT.parent
for p in (DIRECTION_ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from checklist_update import checklist_update
from llm import LLMClient

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# Each case: (label, starting checklist, round_answers). Chosen to exercise the prompt's own rule
# set: atomic splitting (rule 1), empty regions (rule 7), duplicate skipping against an existing
# checklist entry worded differently (rule 8), and framing/hedge handling (rules 4-5).
CASES = [
    (
        "empty checklist, first-question answer",
        {},
        [("Target", "I can see a navy blue kitchen cabinet with brass handles.")],
    ),
    (
        "compound answer needing atomic split",
        {"Target": ["it is navy blue"]},
        [("left", "There is a large wooden table with a plant on it.")],
    ),
    (
        "empty region answer",
        {"Target": ["it is navy blue"]},
        [("on", "There is nothing on top of it.")],
    ),
    (
        "duplicate of an existing checklist entry, different wording",
        {"next to": ["a nightstand"]},
        [("left", "It appears that there is a nightstand.")],
    ),
    (
        "hedge kept as-is",
        {},
        [("above", "It appears that there is possibly a floor lamp.")],
    ),
]


def main() -> None:
    llm_client = LLMClient(MODEL_ID, cache_dir=DIRECTION_ROOT / "artifacts" / "cache")
    print(f"checklist_update: {len(CASES)} cases, backend={llm_client.backend_name}, model={MODEL_ID}")
    print("=" * 70)
    for label, checklist, round_answers in CASES:
        updated = checklist_update(llm_client, checklist, round_answers)
        print(f"\n> {label}")
        print(f"  before: {checklist}")
        print(f"  round_answers: {round_answers}")
        print(f"  after:  {updated}")


if __name__ == "__main__":
    main()
