#!/usr/bin/env python3
"""Ablation ladder (spec §8). STUB: flag -> config-override wiring is complete; the actual
per-rung evaluation loop is TODO until run_eval.py has real accuracy numbers to report (which
in turn needs extract.py / adjudicate.py's prompts). Do not build rung n+1 before rung n is
measured (spec §8) — that applies to filling in this file's TODOs too, not just the pipeline.
"""

import argparse
import copy
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from agent.questioner import load_config  # noqa: E402

DESCRIPTION_TYPES = ["category", "color", "context", "color_context_feature", "color_feature", "color_context"]

# Each rung's override is applied ON TOP OF the previous rung's, matching the spec's ladder
# (A1 = A0 + belief; A2 = A1 + typed extraction; A3 = A2 + empirical priors; A4 = A3 + weak-conflict).
RUNGS = {
    "A0": {"flag": "--baseline-prompted", "overrides": {"use_belief": False, "use_slots": False, "use_priors": False}},
    "A1": {"flag": "--no-slots", "overrides": {"use_belief": True, "use_slots": False, "use_priors": False}},
    "A2": {"flag": "--no-priors", "overrides": {"use_belief": True, "use_slots": True, "use_priors": False}},
    "A3": {"flag": None, "overrides": {"use_belief": True, "use_slots": True, "use_priors": True}},  # default
    "A4": {
        "flag": "--weak-conflict-decisive",
        "overrides": {"use_belief": True, "use_slots": True, "use_priors": True, "weak_conflicts_for_decisive": 3},
    },
}


def config_for_rung(rung: str, base_config: dict | None = None) -> dict:
    config = copy.deepcopy(base_config or load_config())
    overrides = RUNGS[rung]["overrides"]
    if "weak_conflicts_for_decisive" in overrides:
        config["comparison"]["weak_conflicts_for_decisive"] = overrides["weak_conflicts_for_decisive"]
    # TODO(next): `use_belief` / `use_slots` / `use_priors` aren't real GraphQuestioner switches
    # yet — questioner.py currently always runs the full A3-shaped pipeline. Wiring A0/A1/A2 means
    # adding those as constructor flags that short-circuit extract()/select() the way each rung
    # describes, not just toggling config values that nothing reads yet.
    config["_rung"] = rung
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start_idx", type=int)
    parser.add_argument("end_idx", type=int)
    parser.add_argument("--rungs", nargs="+", default=list(RUNGS.keys()), choices=list(RUNGS.keys()))
    parser.add_argument(
        "--description-type", default="all", choices=["all", *DESCRIPTION_TYPES],
        help="Passed through to run_eval.py per rung.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    base_config = load_config()

    report = {}
    for rung in args.rungs:
        config = config_for_rung(rung, base_config)
        print(f"=== Rung {rung} ({RUNGS[rung]['flag'] or 'default'}) ===")
        # TODO(next): once run_eval.py returns real per-episode results, call it here once per
        # description type (or 'all', looping per spec's own --description-type semantics),
        # passing `config`, and record: conclusion accuracy, mean questions/episode, mean wall
        # clock, truncation rate, and the false-True vs false-False confusion split (spec §8).
        report[rung] = {"config": config, "results": "NOT YET IMPLEMENTED — see TODO above"}

    print()
    print("Expectation to test, not assume (spec §8): A1 may capture most of the gain. If")
    print("A2/A3 don't beat A1 by more than run-to-run variance, the slot machinery is")
    print("over-engineering and should be cut from the submission.")


if __name__ == "__main__":
    main()
