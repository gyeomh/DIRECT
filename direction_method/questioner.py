"""Episode loop (SPEC.md §13, build order step 4): the QuestionerInterface implementation that
wires context_parser, self_check, zone_gen, and checklist_update together. No new prompts here --
every VLM call goes through context_parser, self_check, or zone_gen (checklist_update makes none;
see its own module docstring).

ENV.md §4's two call-pattern facts drive this design:
  1. The questioner is constructed once per episode and never reset between candidates -- instance
     state persists across all candidates (this is what makes the accumulating checklist work).
  2. ask_or_conclude is called repeatedly for the SAME candidate (once per question, then once
     more for the conclusion). Candidate transitions are detected by hashing the image array.

ENV.md §5: one of `info`'s keys is a leaked reference to the true target image, and must never be
read. This class never stores `info` itself -- only the one field it actually needs
(`target_description`) is read in __init__, so there is no path from this object back to that key
at all, regardless of its name.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DIRECTION_ROOT = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _DIRECTION_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Questioner import QuestionerInterface  # noqa: E402 -- repo root, path inserted above
from checklist_update import checklist_update as run_checklist_update  # noqa: E402
from context_parser import parse_context  # noqa: E402
from llm import LLMClient, image_hash  # noqa: E402
from self_check import is_failure, self_check  # noqa: E402
from templates import assertion_for_answer, question_for, region_for  # noqa: E402
from zone_gen import ZoneGenError, locate, resolve_relations, zones  # noqa: E402

FIRST_QUESTION_TEMPLATE = "Can you describe the {TARGET}'s location and visual appearance (e.g., color, shape, size)."

# Default model for every module. Qwen3.6 replaced Qwen3-VL-30B-A3B-Instruct as the project default
# on 2026-08-05: same MoE footprint class, and it won every single description type in the head-to-
# head sweep (SPEC.md §13's model-swap section -- 711/1002 there, 759/1002 on the later v2 run, vs.
# the old model's 612/1002). Two things travel with it and must not be dropped:
#   - thinking has to be forced off per call (it defaults to emitting a reasoning block, and §8's
#     ~60-70 calls/episode against a 600s budget has no room for one),
#   - it needs vllm>=0.19.0; the old 0.15.0 pin does not know its architecture class.
# Overridable by env for one-off comparisons without editing this file -- eval_model.py constructs
# the questioner as `YourQuestioner(info)`, so env is the only channel available to that path.
DEFAULT_MODEL_ID = os.environ.get("VLM_MODEL_ID", "Qwen/Qwen3.6-35B-A3B-FP8")
DEFAULT_DISABLE_THINKING = os.environ.get("VLM_DISABLE_THINKING", "1") == "1"

# ENV.md §3 -- the harness's own hard caps, shared across ALL candidates in an episode.
ENV_MAX_STEPS = 60
TOTAL_TIME_BUDGET_S = 600.0

# Self-imposed, ahead of the harness's own hard caps, so the episode concludes cleanly instead of
# being truncated. Fractions and ASSUMED_MAX_CANDIDATES per the build spec; observed candidate
# count per episode ranges 1-7 (mean 3.16) across episodes_train.jsonl -- 7 is the conservative
# assumption since the questioner is never told the true count for the episode it's in.
SOFT_STOP_FRACTION = 0.60
HARD_STOP_FRACTION = 0.85
ASSUMED_MAX_CANDIDATES = 7


@dataclass
class CandidateLog:
    """One per candidate, for tomorrow's diagnosis runs. Finalized once the candidate concludes.

    `interactions` is the rich, ordered trace a visual viewer reads: one entry per checklist
    check (Step 2) or relation question+answer+check (Step 4/5), each carrying the actual region/
    assertion/evidence/verdict text, not just the verdict. Aggregate fields (questions_asked,
    self_check_calls, verdicts) are kept alongside for cheap counting without walking interactions.
    """

    questions_asked: int = 0
    self_check_calls: int = 0
    verdicts: list = field(default_factory=list)  # in call order, across both Step 2 and Step 4/5
    interactions: list = field(default_factory=list)  # [{"type": ..., ...}], in call order
    conclusion: object = None  # True (match) / False (mismatch), set once concluded
    reasoning: str = ""
    started_at: float = 0.0
    elapsed_s: float = 0.0
    bbox_2d: object = None  # zone_gen's box for this candidate, in the model's native frame
    zone_list: list = field(default_factory=list)  # resolved relations after dedup (§5-2)
    scene: str = ""  # zone_gen's own "scene" description of where the target sits in frame
    boxed_image: object = None  # np.ndarray -- the driver saves this to disk, questioner does not
    checklist_before: dict = field(default_factory=dict)  # self.checklist snapshot at candidate start
    checklist_after: dict = field(default_factory=dict)  # self.checklist snapshot after concluding


@dataclass
class _CandidateState:
    """Per-candidate cache (ENV.md §4 fact 2). Rebuilt wholesale on every image-hash transition;
    never recomputed within a candidate."""

    bbox_2d: object = None
    boxed_image: object = None
    zone_list: list = field(default_factory=list)
    question_queue: list = field(default_factory=list)  # [(relation, question_text), ...]
    round_answers: list = field(default_factory=list)  # [(relation, answer), ...] this candidate
    awaiting_relation: object = None  # relation whose answer we're waiting for, or None
    awaiting_question: object = None  # the question text asked for awaiting_relation, for logging
    checklist_phase_done: bool = False
    log: CandidateLog = field(default_factory=CandidateLog)


class DirectionMethodQuestioner(QuestionerInterface):
    def __init__(self, info, model_id: str = DEFAULT_MODEL_ID, llm_client=None):
        target_description = info["target_description"]
        # info is intentionally never stored (see module docstring / ENV.md §5) -- nothing below
        # this line reads `info` again.

        self.llm_client = llm_client or LLMClient(model_id, disable_thinking=DEFAULT_DISABLE_THINKING)
        parsed = parse_context(self.llm_client, target_description)
        self.context_parser_result = parsed  # full ParsedContext -- target_category/target_phrase/
        # other_objects/validation_problems/retried, for episode-level logging
        self.target_category = parsed.target_category
        self.target_phrase = parsed.target_phrase
        self.checklist = parsed.checklist
        self.initial_checklist = dict(parsed.checklist)  # snapshot before any candidate mutates it

        self.first_question_asked = False
        self.step_count = 0
        self.candidates_seen = 0
        self._episode_start = time.time()
        self._current_image_hash = None
        self._candidate = None

        # QuestionerInterface / eval_model.py bookkeeping: the base class's add_answer() appends
        # to self.answers; eval_model.py reads n_questions/time_required as plain attributes.
        self.questions = []
        self.answers = []
        self.n_questions = 0
        self.time_required = 0.0

        self.candidate_logs = []
        self.episode_log = {
            "n_candidates": 0,
            "total_questions": 0,
            "total_self_check_calls": 0,
            "soft_stop_fired": False,
            "hard_stop_fired": False,
            "budget_forced_conclusions": 0,
        }

    def reset_time(self):
        self.time_required = 0.0

    # --- budget -------------------------------------------------------------------------------

    def _elapsed(self) -> float:
        return time.time() - self._episode_start

    def _hard_stop_reached(self) -> bool:
        reached = self._elapsed() >= HARD_STOP_FRACTION * TOTAL_TIME_BUDGET_S
        if reached:
            self.episode_log["hard_stop_fired"] = True
        return reached

    def _can_ask_more_questions(self) -> bool:
        """Gates every point where we would either start Step 3 (zone_gen) or pop the next
        question from an already-built queue. False for either of two reasons:

        - soft time-stop (0.60 * 600s): stop asking NEW questions from here on; still allow
          Step 2 (already-necessary, cheap, no oracle round-trip) to run for future candidates.
        - step reserve: asking one more question must still leave at least one step for THIS
          candidate's own conclusion, plus one for each candidate still assumed to come after it
          (ASSUMED_MAX_CANDIDATES - candidates_seen). Framed as a "reserve one step per remaining
          candidate" margin against the harness's hard 60-step cap (ENV.md §3).
        """
        if self._elapsed() >= SOFT_STOP_FRACTION * TOTAL_TIME_BUDGET_S:
            self.episode_log["soft_stop_fired"] = True
            return False
        steps_left_after_this_question = ENV_MAX_STEPS - (self.step_count + 1)
        future_candidates = max(0, ASSUMED_MAX_CANDIDATES - self.candidates_seen)
        reserve_needed = 1 + future_candidates  # this candidate's conclusion + future ones
        return steps_left_after_this_question >= reserve_needed

    # --- return-value helpers ------------------------------------------------------------------
    # Every return statement in ask_or_conclude funnels through exactly one of these two, so
    # "both None" / "both non-None" (ENV.md §4/§5's one silently-mishandled failure mode) is
    # structurally unreachable rather than something each call site has to get right on its own.

    def _ask(self, candidate: _CandidateState, relation: str, question_text: str, reasoning: str) -> dict:
        candidate.awaiting_relation = relation
        candidate.awaiting_question = question_text
        if relation == "Target":
            self.first_question_asked = True
        candidate.log.questions_asked += 1
        self.n_questions += 1
        self.episode_log["total_questions"] += 1
        action = dict(question=question_text, conclusion=None, reasoning=reasoning)
        assert (action["question"] is None) != (action["conclusion"] is None)
        return action

    def _conclude(
        self, candidate: _CandidateState, match: bool, reasoning: str, *, run_update: bool = True, budget_forced: bool = False
    ) -> dict:
        # checklist_update is a pure code function now (no LLM call -- see checklist_update.py's
        # module docstring), so the earlier "run synchronously vs. lazily" latency tradeoff no
        # longer applies; it just always runs here, before returning the conclusion.
        if run_update and candidate.round_answers:
            self.checklist = run_checklist_update(self.checklist, candidate.round_answers)

        candidate.log.conclusion = match
        candidate.log.reasoning = reasoning
        candidate.log.elapsed_s = time.time() - candidate.log.started_at
        candidate.log.checklist_after = dict(self.checklist)
        self.candidate_logs.append(candidate.log)
        if budget_forced:
            self.episode_log["budget_forced_conclusions"] += 1

        action = dict(question=None, conclusion=1 if match else 0, reasoning=reasoning)
        assert (action["question"] is None) != (action["conclusion"] is None)
        return action

    # --- main state machine --------------------------------------------------------------------

    def ask_or_conclude(self, observation: dict) -> dict:
        # `time_required` accumulates wall-clock spent inside this method, matching what upstream's
        # own QuestionerLocalVLM.ask_or_conclude times (start_time/end_time around its work).
        # eval_model.py reads it as a plain attribute for logging, so leaving it at 0 would report
        # a self-measured number dishonestly (ENV.md §4/§5). Distinct from the budget clock, which
        # runs on _episode_start and measures elapsed time, not time consumed.
        _t0 = time.time()
        try:
            return self._ask_or_conclude(observation)
        finally:
            self.time_required += time.time() - _t0

    def _ask_or_conclude(self, observation: dict) -> dict:
        self.step_count += 1
        image = observation["image"]
        answer = observation["answer"]

        img_hash = image_hash(image)
        is_new_candidate = img_hash != self._current_image_hash
        if is_new_candidate:
            self._current_image_hash = img_hash
            self.candidates_seen += 1
            self._candidate = _CandidateState()
            self._candidate.log.started_at = time.time()
            self._candidate.log.checklist_before = dict(self.checklist)
            self.episode_log["n_candidates"] += 1
        candidate = self._candidate

        # Absolute cutoff -- zero further VLM calls, whatever phase we're in. Any not-yet-merged
        # round_answers for this candidate are dropped along with everything else (run_update=False).
        if self._hard_stop_reached():
            return self._conclude(candidate, False, "hard budget stop", run_update=False, budget_forced=True)

        # Receive the answer to whatever question was asked last call, if any.
        if candidate.awaiting_relation is not None:
            relation = candidate.awaiting_relation
            question_text = candidate.awaiting_question
            candidate.awaiting_relation = None
            candidate.awaiting_question = None
            # A bare "nothing" becomes the canonical emptiness assertion before it reaches either
            # consumer (templates.assertion_for_answer): self_check cannot judge "nothing", and
            # checklist_update files answers verbatim, so an un-normalized one would be re-checked
            # against every later candidate too. Everything else passes through untouched.
            assertion = assertion_for_answer(answer)
            candidate.round_answers.append((relation, assertion))
            region = region_for(relation, self.target_phrase)
            result = self_check(self.llm_client, image, region, assertion)
            candidate.log.self_check_calls += 1
            candidate.log.verdicts.append(result.verdict)
            candidate.log.interactions.append({
                "type": "relation_answer_check",
                "relation": relation,
                "question": question_text,
                "answer": answer,
                # what self_check was actually given -- equal to `answer` unless it was normalized
                "assertion": assertion,
                "region": region,
                "verdict": result.verdict,
                "evidence": result.evidence,
            })
            self.episode_log["total_self_check_calls"] += 1
            if is_failure(result.verdict):
                # SPEC.md §13 / B1: this answer is still true about the target regardless of the
                # mismatch, and it is included in round_answers (appended above) -- checklist_update
                # below sees it like any other.
                return self._conclude(candidate, False, f"self_check failed on answer for relation {relation!r}")

        if is_new_candidate:
            # Step 2 -- re-verify the EXISTING checklist against this candidate. No new oracle
            # question is asked here, so there is nothing new for checklist_update to see yet.
            for parent_key, assertions in self.checklist.items():
                for assertion in assertions:
                    region = region_for(parent_key, self.target_phrase)
                    result = self_check(self.llm_client, image, region, assertion)
                    candidate.log.self_check_calls += 1
                    candidate.log.verdicts.append(result.verdict)
                    candidate.log.interactions.append({
                        "type": "checklist_check",
                        "parent_key": parent_key,
                        "assertion": assertion,
                        "region": region,
                        "verdict": result.verdict,
                        "evidence": result.evidence,
                    })
                    self.episode_log["total_self_check_calls"] += 1
                    if is_failure(result.verdict):
                        return self._conclude(candidate, False, f"checklist self_check failed on {parent_key!r}")
            candidate.checklist_phase_done = True

            if not self._can_ask_more_questions():
                # Skip zone_gen entirely -- we already know we won't ask anything it returns.
                return self._conclude(candidate, False, "budget exhausted before zone_gen", budget_forced=True)

            # Step 3 -- zones.
            try:
                loc = locate(self.llm_client, image, self.target_category)
                zr = zones(self.llm_client, loc.boxed_image, self.target_category)
            except ZoneGenError:
                return self._conclude(candidate, False, "zone_gen error, conservative mismatch")
            resolved = resolve_relations(zr, loc.bbox_2d, existing_parent_keys=set(self.checklist.keys()))
            candidate.bbox_2d = loc.bbox_2d
            candidate.boxed_image = loc.boxed_image
            candidate.zone_list = resolved.relations
            candidate.log.bbox_2d = loc.bbox_2d
            candidate.log.boxed_image = loc.boxed_image
            candidate.log.zone_list = resolved.relations
            candidate.log.scene = zr.scene
            candidate.question_queue = [(r, question_for(r, self.target_phrase)) for r in resolved.relations]
            if not self.first_question_asked:
                candidate.question_queue.insert(
                    0, ("Target", FIRST_QUESTION_TEMPLATE.format(TARGET=self.target_phrase))
                )

        # Step 4/5 -- ask the next queued question, if the budget still allows it.
        if candidate.question_queue:
            if self._can_ask_more_questions():
                relation, question_text = candidate.question_queue.pop(0)
                return self._ask(candidate, relation, question_text, f"asking about {relation!r}")
            return self._conclude(candidate, False, "budget exhausted mid-candidate", budget_forced=True)

        # Queue genuinely empty (drained, or there was nothing to ask) -- every relation checked
        # out clean.
        return self._conclude(candidate, True, "all relations verified")
