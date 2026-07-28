import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for Questioner.py

import numpy as np
import pytest

import zone_gen as zg
from llm import LLMClient
from questioner import (
    ASSUMED_MAX_CANDIDATES,
    ENV_MAX_STEPS,
    HARD_STOP_FRACTION,
    SOFT_STOP_FRACTION,
    TOTAL_TIME_BUDGET_S,
    DirectionMethodQuestioner,
)

IMG_A = np.zeros((12, 12, 3), dtype=np.uint8)
IMG_B = np.full((12, 12, 3), 80, dtype=np.uint8)
IMG_C = np.full((12, 12, 3), 160, dtype=np.uint8)


@pytest.fixture(autouse=True)
def _clear_zone_gen_module_caches():
    # locate()/zones() memoize per (image_hash, target_category) at module scope (zone_gen.py) --
    # tests reusing IMG_A/IMG_B/IMG_C and the default target_category would otherwise silently
    # read another test's cached response instead of this test's own scripted one.
    zg._locate_cache.clear()
    zg._zones_cache.clear()
    yield
    zg._locate_cache.clear()
    zg._zones_cache.clear()


class ScriptedBackend:
    """Dispatches on each module's distinctive, never-substituted prompt text, so a single fake
    backend can drive the full pipeline (context_parser once in __init__, then self_check/
    zone_gen/checklist_update repeatedly) without caring about call order.
    """

    def __init__(
        self,
        *,
        target_category="kitchen lower cabinet",
        target_phrase="navy blue kitchen lower cabinet",
        checklist=None,
        other_objects=None,
        self_check_verdicts=None,
        locate_boxes=None,
        zones_regions=None,
        checklist_additions=None,
    ):
        self.target_category = target_category
        self.target_phrase = target_phrase
        # context_parser.py merges non-"Target" checklist keys from other_objects only (§10) --
        # to seed an existing non-Target checklist entry, pass other_objects, not a raw checklist
        # dict with that key; checklist itself is only ever used here to seed "Target".
        self.checklist = checklist if checklist is not None else {}
        self.other_objects = other_objects if other_objects is not None else []
        # Popped in call order; None left in the queue after exhaustion defaults to "yes".
        self.self_check_verdicts = list(self_check_verdicts) if self_check_verdicts is not None else []
        self.locate_boxes = (
            locate_boxes if locate_boxes is not None else [{"bbox_2d": [200, 200, 800, 800], "label": "t", "note": "n"}]
        )
        self.zones_regions = zones_regions if zones_regions is not None else [{"note": "n", "key": "left"}]
        self.checklist_additions = checklist_additions if checklist_additions is not None else {}
        self.calls = []  # module name per call, in order

    def generate(self, prompt, image, response_schema, timeout_s):
        if "Parse an object description into a target object and a checklist." in prompt:
            self.calls.append("context_parser")
            return json.dumps({
                "target_category": self.target_category,
                "target_phrase": self.target_phrase,
                "other_objects": self.other_objects,
                "checklist": self.checklist,
            })
        if "You verify a single claim against an image" in prompt:
            self.calls.append("self_check")
            verdict = self.self_check_verdicts.pop(0) if self.self_check_verdicts else "yes"
            return json.dumps({"evidence": "e", "verdict": verdict})
        if "=== BUILT-IN RUNS ===" in prompt:
            self.calls.append("locate")
            return json.dumps({"boxes": self.locate_boxes})
        if "A red box marks the TARGET object." in prompt:
            self.calls.append("zones")
            return json.dumps({"scene": "s", "regions": self.zones_regions})
        if "You maintain a checklist of facts about a scene." in prompt:
            self.calls.append("checklist_update")
            return json.dumps({"additions": self.checklist_additions})
        raise AssertionError(f"unscripted prompt: {prompt[:100]!r}")


def make_questioner(tmp_path, **scripted_kwargs):
    scripted = ScriptedBackend(**scripted_kwargs)
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    client._backend.generate = scripted.generate
    info = {"target_description": "Navy blue kitchen lower cabinet"}
    q = DirectionMethodQuestioner(info, llm_client=client)
    return q, scripted


def call(q, image, answer=None):
    """Every call in this suite goes through here so the ENV.md §4/§5 "never both None, never
    neither" invariant is checked on literally every ask_or_conclude return value, not just in
    one dedicated test."""
    action = q.ask_or_conclude({"image": image, "answer": answer})
    assert (action["question"] is None) != (action["conclusion"] is None), f"invalid action: {action}"
    assert "reasoning" in action
    return action


# --- construction / ENV.md compliance -------------------------------------------------------


def test_info_dict_itself_is_never_stored(tmp_path):
    sentinel = object()
    scripted = ScriptedBackend(checklist={})
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    client._backend.generate = scripted.generate
    info = {"target_description": "Kitchen lower cabinet", "task_image": sentinel}

    q = DirectionMethodQuestioner(info, llm_client=client)

    assert not hasattr(q, "info")
    assert not any(v is sentinel for v in vars(q).values())


def test_no_task_image_reference_anywhere_in_the_package():
    forbidden = "task" + "_image"  # split so this test file's own occurrence doesn't self-flag
    this_file = Path(__file__).resolve()
    package_root = this_file.parents[1]  # direction_method/
    hits = []
    for py_file in package_root.rglob("*.py"):
        if py_file.resolve() == this_file:
            continue
        if forbidden in py_file.read_text():
            hits.append(str(py_file.relative_to(package_root)))
    assert hits == [], f"forbidden leaked-target-image key referenced in: {hits}"


def test_budget_constants_match_the_build_spec():
    assert ENV_MAX_STEPS == 60
    assert TOTAL_TIME_BUDGET_S == 600.0
    assert SOFT_STOP_FRACTION == 0.60
    assert HARD_STOP_FRACTION == 0.85
    assert ASSUMED_MAX_CANDIDATES == 7


def test_context_parser_runs_once_in_init(tmp_path):
    q, scripted = make_questioner(tmp_path, checklist={})
    assert scripted.calls == ["context_parser"]
    assert q.target_category == "kitchen lower cabinet"
    assert q.target_phrase == "navy blue kitchen lower cabinet"
    assert q.checklist == {}


# --- happy path: empty checklist, one relation, all pass --------------------------------------


def test_happy_path_first_question_then_one_relation_then_match(tmp_path):
    q, scripted = make_questioner(
        tmp_path, checklist={}, zones_regions=[{"note": "n", "key": "left"}],
        checklist_additions={"Target": ["it is navy blue"]},
    )

    a1 = call(q, IMG_A, answer=None)  # new candidate: empty checklist -> straight to zone_gen
    assert a1["question"] is not None
    assert "Target" in a1["reasoning"] or "location and visual appearance" in a1["question"]

    a2 = call(q, IMG_A, answer="a navy blue cabinet in the corner")
    assert a2["question"] is not None  # the "left" relation question

    a3 = call(q, IMG_A, answer="a wooden table on the left")
    assert a3["conclusion"] == 1  # match: queue drained, nothing failed

    assert q.first_question_asked is True
    assert q.checklist == {"Target": ["it is navy blue"]}  # merged via checklist_update
    assert scripted.calls.count("locate") == 1
    assert scripted.calls.count("zones") == 1
    assert scripted.calls.count("checklist_update") == 1


def test_first_question_not_repeated_for_a_later_candidate(tmp_path):
    q, scripted = make_questioner(tmp_path, checklist={}, zones_regions=[])
    call(q, IMG_A, answer=None)
    call(q, IMG_A, answer="a navy blue cabinet")  # concludes: no other relations queued
    scripted.calls.clear()

    a = call(q, IMG_B, answer=None)  # new candidate; first question already asked this episode
    assert "location and visual appearance" not in (a["question"] or "")


# --- Step 2: existing checklist verification ---------------------------------------------------


def test_step2_failure_concludes_mismatch_and_skips_zone_gen(tmp_path):
    q, scripted = make_questioner(
        tmp_path,
        checklist={"Target": ["it is navy blue"]},
        self_check_verdicts=["no"],  # the one checklist assertion fails
    )
    a = call(q, IMG_A, answer=None)
    assert a["conclusion"] == 0
    assert "locate" not in scripted.calls
    assert "zones" not in scripted.calls


def test_step2_skipped_entirely_when_checklist_empty(tmp_path):
    q, scripted = make_questioner(tmp_path, checklist={}, zones_regions=[])
    call(q, IMG_A, answer=None)
    assert "self_check" not in scripted.calls  # no checklist statements to verify
    assert "locate" in scripted.calls  # went straight to zone_gen


def test_step2_passes_then_proceeds_to_zone_gen(tmp_path):
    q, scripted = make_questioner(
        tmp_path, checklist={"Target": ["it is navy blue"]}, self_check_verdicts=["yes"], zones_regions=[]
    )
    scripted.calls.clear()  # drop the constructor's own context_parser call
    call(q, IMG_A, answer=None)
    assert scripted.calls[0] == "self_check"
    assert "locate" in scripted.calls


# --- Step 4/5: relation question failure --------------------------------------------------------


def test_relation_answer_failure_concludes_mismatch_and_updates_checklist_with_failing_answer(tmp_path):
    q, scripted = make_questioner(
        tmp_path,
        checklist={},
        zones_regions=[{"note": "n", "key": "left"}],
        self_check_verdicts=["yes", "no"],  # Target passes, "left" fails
        checklist_additions={"Target": ["it is navy blue"], "left": ["a refrigerator"]},
    )
    call(q, IMG_A, answer=None)  # asks Target
    call(q, IMG_A, answer="a navy blue cabinet")  # Target passes, asks "left"
    a3 = call(q, IMG_A, answer="a refrigerator, not a table")  # left fails
    assert a3["conclusion"] == 0

    # B1: checklist_update still ran, including the failing round -- both answers this round
    # (the passing Target one AND the failing left one) were fed into it.
    assert scripted.calls[-1] == "checklist_update"
    assert q.checklist == {"Target": ["it is navy blue"], "left": ["a refrigerator"]}


def test_all_relations_pass_concludes_match(tmp_path):
    q, scripted = make_questioner(
        tmp_path, checklist={}, zones_regions=[{"note": "n", "key": "left"}, {"note": "n2", "key": "on"}],
        self_check_verdicts=["yes", "yes", "yes"],
    )
    call(q, IMG_A, answer=None)  # Target
    call(q, IMG_A, answer="a")  # left
    call(q, IMG_A, answer="b")  # on
    a4 = call(q, IMG_A, answer="c")
    assert a4["conclusion"] == 1


# --- dedup against checklist parent keys --------------------------------------------------------


def test_relation_already_a_checklist_parent_key_is_not_asked_again(tmp_path):
    q, scripted = make_questioner(
        tmp_path,
        other_objects=[{"object": "wooden tiles", "cue": "on", "key": "left"}],
        self_check_verdicts=["yes"],  # Step 2's one assertion passes
        zones_regions=[{"note": "n", "key": "left"}, {"note": "n2", "key": "above"}],
    )
    a1 = call(q, IMG_A, answer=None)
    # "left" is already a checklist parent key -> deduped out; only "above" (then queued after
    # the mandatory Target question) should ever be asked.
    assert "above" in a1["question"] or "location and visual appearance" in a1["question"]
    a2 = call(q, IMG_A, answer="x")
    assert "above" in a2["question"]


# --- per-candidate caching: zone_gen not recomputed within a candidate -------------------------


def test_zone_gen_runs_exactly_once_per_candidate(tmp_path):
    q, scripted = make_questioner(
        tmp_path, checklist={}, zones_regions=[{"note": "n", "key": "left"}, {"note": "n2", "key": "on"}]
    )
    call(q, IMG_A, answer=None)
    call(q, IMG_A, answer="a")
    call(q, IMG_A, answer="b")
    call(q, IMG_A, answer="c")  # concludes
    assert scripted.calls.count("locate") == 1
    assert scripted.calls.count("zones") == 1


def test_checklist_growth_is_reverified_on_the_next_candidate(tmp_path):
    q, scripted = make_questioner(
        tmp_path, checklist={}, zones_regions=[], checklist_additions={"Target": ["it is navy blue"]},
    )
    # zones_regions=[] still lets the bbox-margin fallback queue a few directional relations
    # (zone_gen.py's own designed behavior) -- drain the candidate however many calls that takes.
    answer = None
    for _ in range(10):
        a = call(q, IMG_A, answer=answer)
        if a["conclusion"] is not None:
            break
        answer = "a navy blue cabinet"
    else:
        raise AssertionError("candidate never concluded")
    assert q.checklist == {"Target": ["it is navy blue"]}
    scripted.calls.clear()
    scripted.self_check_verdicts = ["yes"] * 10

    call(q, IMG_B, answer=None)  # new candidate -- Step 2 must check the newly grown checklist
    assert scripted.calls[0] == "self_check"


# --- zone_gen error handling ---------------------------------------------------------------------


def test_zone_gen_error_is_a_conservative_mismatch_not_a_crash(tmp_path):
    q, scripted = make_questioner(tmp_path, checklist={}, locate_boxes=[])  # empty -> ZoneGenError
    a = call(q, IMG_A, answer=None)
    assert a["conclusion"] == 0


# --- budget: hard stop ---------------------------------------------------------------------------


def test_hard_stop_immediate_mismatch_zero_calls(tmp_path):
    q, scripted = make_questioner(tmp_path, checklist={})
    scripted.calls.clear()
    q._episode_start = time.time() - (HARD_STOP_FRACTION * TOTAL_TIME_BUDGET_S + 1)

    a = call(q, IMG_A, answer=None)
    assert a["conclusion"] == 0
    assert scripted.calls == []
    assert q.episode_log["hard_stop_fired"] is True


def test_hard_stop_mid_candidate_drops_unmerged_round_answers(tmp_path):
    q, scripted = make_questioner(
        tmp_path, checklist={}, zones_regions=[{"note": "n", "key": "left"}], self_check_verdicts=["yes"]
    )
    call(q, IMG_A, answer=None)  # asks Target
    q._episode_start = time.time() - (HARD_STOP_FRACTION * TOTAL_TIME_BUDGET_S + 1)
    scripted.calls.clear()

    a = call(q, IMG_A, answer="a navy blue cabinet")  # would normally self_check this answer
    assert a["conclusion"] == 0
    assert scripted.calls == []  # zero further calls, including no checklist_update
    assert q.checklist == {}  # nothing merged


# --- budget: soft stop / step reserve -------------------------------------------------------------


def test_can_ask_more_questions_false_when_soft_time_stop_reached(tmp_path):
    q, _ = make_questioner(tmp_path, checklist={})
    q._episode_start = time.time() - (SOFT_STOP_FRACTION * TOTAL_TIME_BUDGET_S + 1)
    assert q._can_ask_more_questions() is False
    assert q.episode_log["soft_stop_fired"] is True


def test_can_ask_more_questions_false_when_step_reserve_exhausted(tmp_path):
    q, _ = make_questioner(tmp_path, checklist={})
    # far under the time thresholds, but few steps left relative to assumed future candidates
    q.step_count = ENV_MAX_STEPS - 5
    q.candidates_seen = 1
    assert q._can_ask_more_questions() is False


def test_can_ask_more_questions_true_when_comfortably_within_budget(tmp_path):
    q, _ = make_questioner(tmp_path, checklist={})
    q.step_count = 2
    q.candidates_seen = 1
    assert q._can_ask_more_questions() is True


def test_soft_stop_before_zone_gen_skips_zone_gen_entirely(tmp_path):
    q, scripted = make_questioner(tmp_path, checklist={})
    q._episode_start = time.time() - (SOFT_STOP_FRACTION * TOTAL_TIME_BUDGET_S + 1)
    scripted.calls.clear()

    a = call(q, IMG_A, answer=None)
    assert a["conclusion"] == 0
    assert "locate" not in scripted.calls
    assert "zones" not in scripted.calls


def test_soft_stop_mid_queue_concludes_mismatch_but_still_updates_checklist(tmp_path):
    q, scripted = make_questioner(
        tmp_path,
        checklist={},
        zones_regions=[{"note": "n", "key": "left"}, {"note": "n2", "key": "on"}],
        self_check_verdicts=["yes", "yes"],
        checklist_additions={"Target": ["it is navy blue"]},
    )
    call(q, IMG_A, answer=None)  # Target
    call(q, IMG_A, answer="a navy blue cabinet")  # Target passes, asks "left"
    q._episode_start = time.time() - (SOFT_STOP_FRACTION * TOTAL_TIME_BUDGET_S + 1)

    a = call(q, IMG_A, answer="a wooden table")  # "left" passes, but budget stops "on" from being asked
    assert a["conclusion"] == 0
    assert q.checklist == {"Target": ["it is navy blue"]}  # checklist_update still ran


# --- multiple candidates end to end ----------------------------------------------------------


def test_three_candidates_in_one_episode(tmp_path):
    q, scripted = make_questioner(
        tmp_path, checklist={}, zones_regions=[], checklist_additions={"Target": ["it is navy blue"]},
    )
    for image, answer_seq in [
        (IMG_A, ["a navy blue cabinet"]),
        (IMG_B, ["a white cabinet, not navy"]),
        (IMG_C, ["a navy blue cabinet"]),
    ]:
        call(q, image, answer=None)
        scripted.self_check_verdicts = ["no"] if "not navy" in answer_seq[0] else ["yes"]
        call(q, image, answer=answer_seq[0])

    assert q.candidates_seen == 3
    assert q.episode_log["n_candidates"] == 3


# --- randomized-ish stress: invariant holds across many varied runs ----------------------------


def test_invariant_holds_across_many_varied_scenarios(tmp_path):
    scenarios = [
        dict(checklist={}, zones_regions=[], self_check_verdicts=[]),
        dict(checklist={"Target": ["it is navy blue"]}, self_check_verdicts=["no"]),
        dict(checklist={}, zones_regions=[{"note": "n", "key": k} for k in ["left", "right", "above", "below", "on"]]),
        dict(checklist={}, locate_boxes=[]),
    ]
    for i, kwargs in enumerate(scenarios):
        q, scripted = make_questioner(tmp_path / f"s{i}", **kwargs)
        image = IMG_A
        answer = None
        for _ in range(10):
            a = call(q, image, answer=answer)
            if a["conclusion"] is not None:
                break
            answer = "some answer"
        else:
            raise AssertionError("scenario did not conclude within 10 calls")
