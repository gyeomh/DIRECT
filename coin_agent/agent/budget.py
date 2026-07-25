"""Step/time controller (spec §5.5).

The questioner never learns how many candidates the episode holds, so this assumes a
conservative `assumed_n_candidates` and reserves one conclusion step per remaining candidate.
Ground-truth constants (`MAX_STEPS`, `MAX_TIME_S`) come from `env.py` / spec §0.3 — re-verify
against upstream `env.py` if it changes; see `scripts/verify_ground_truth.py`.
"""

import math
import time

from . import schema
from .state import TargetBelief

MAX_STEPS = 60
MAX_TIME_S = 600.0


class BudgetController:
    def __init__(self, config: dict, max_adjacent: int, *, _time_fn=time.monotonic):
        self._time_fn = _time_fn
        self.assumed_n_candidates = config["assumed_n_candidates"]
        self.hard_cap_per_image = config["hard_cap_per_image"]
        self.soft_time_frac = config["soft_time_frac"]
        self.hard_time_frac = config["hard_time_frac"]
        self.front_load = config["front_load"]["weights"] if "weights" in config.get("front_load", {}) else config["front_load"]
        self.max_adjacent = max_adjacent

        self.start_time = self._time_fn()
        self.steps_used = 0
        self.candidates_seen = 0
        self.questions_this_candidate = 0
        self.per_candidate_cap = 0
        self.soft_stop_fired = False
        self.hard_stop_fired = False

    @property
    def elapsed(self) -> float:
        return self._time_fn() - self.start_time

    def _remaining_candidates_est(self) -> int:
        return max(1, self.assumed_n_candidates - self.candidates_seen)

    def _reserve(self) -> int:
        return self._remaining_candidates_est()

    def _front_load_weight(self) -> float:
        key = self.candidates_seen
        if key in self.front_load:
            return self.front_load[key]
        if str(key) in self.front_load:
            return self.front_load[str(key)]
        return self.front_load.get("default", 0.8)

    def _ambiguity_allowance(self, belief: TargetBelief) -> int:
        """Count of Tier-A slots not already implied by the description (spec §5.5 — this is
        what makes the cap adapt across description types: 'category' leaves ~6 Tier-A slots
        unspecified, 'color_context_feature' leaves ~2).
        """
        count = 0
        for slot_key in schema.all_slot_keys(self.max_adjacent):
            if schema.spec_for(slot_key).tier != schema.TIER_A:
                continue
            value = belief.get(slot_key)
            implied_by_description = value.provenance == "description" and value.canon is not None
            if not implied_by_description:
                count += 1
        return count

    def start_candidate(self, belief: TargetBelief) -> None:
        """Call once when a new candidate's ObservationFrame is created (not on repeated
        ask_or_conclude() calls for the same candidate — questioner.py must guard that).
        """
        self.questions_this_candidate = 0
        askable = max(0, MAX_STEPS - self.steps_used - self._reserve())
        cap = math.floor(askable / self._remaining_candidates_est()) * self._front_load_weight()
        cap = min(cap, self._ambiguity_allowance(belief), self.hard_cap_per_image)
        self.per_candidate_cap = max(0, int(cap))

    def may_ask(self) -> bool:
        if self.questions_this_candidate >= self.per_candidate_cap:
            return False
        if self.elapsed > self.soft_time_frac * MAX_TIME_S:
            self.soft_stop_fired = True
            return False
        if self.steps_used >= MAX_STEPS - self._reserve():
            return False
        return True

    def hard_stop(self) -> bool:
        fired = self.elapsed > self.hard_time_frac * MAX_TIME_S
        if fired:
            self.hard_stop_fired = True
        return fired

    def record_question(self) -> None:
        self.steps_used += 1
        self.questions_this_candidate += 1

    def record_conclusion(self) -> None:
        self.steps_used += 1

    def advance_candidate(self) -> None:
        self.candidates_seen += 1

    def snapshot(self) -> dict:
        """Per-episode logging payload (spec §5.5: "Log per episode: steps_used, elapsed,
        questions_per_candidate, whether soft/hard stop fired").
        """
        return {
            "steps_used": self.steps_used,
            "elapsed": self.elapsed,
            "candidates_seen": self.candidates_seen,
            "questions_this_candidate": self.questions_this_candidate,
            "soft_stop_fired": self.soft_stop_fired,
            "hard_stop_fired": self.hard_stop_fired,
        }
