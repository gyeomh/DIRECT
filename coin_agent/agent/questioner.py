"""GraphQuestioner(QuestionerInterface) — entry point wiring the §5 decision procedure together.

Integrity (spec §0.6): `info["task_image"]` is the target image PIL object — an upstream leak.
The line below is the *only* place in this package allowed to mention that key, and it only
discards it; `tests/test_integrity.py` greps the whole `agent/` package to enforce this.
"""

import sys
from pathlib import Path

import yaml

from . import adjudicate, budget as budget_mod, compare, extract, parse, select
from .llm import LLMClient, _image_hash as hash_image
from .priors import PriorsTable
from .state import ObservationFrame

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # coin_agent/
_REPO_ROOT = _PACKAGE_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Questioner import QuestionerInterface, _validate_observation  # noqa: E402

DEFAULT_CONFIG_PATH = _PACKAGE_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG_PATH) as f:
        return yaml.safe_load(f)


class GraphQuestioner(QuestionerInterface):
    def __init__(self, info, model_id: str | None = None, config_path: str | Path | None = None):
        info = dict(info)
        info.pop("task_image", None)  # integrity: sanctioned discard (spec §0.6) — never read, only discarded
        super().__init__(info)

        self.config = load_config(config_path)
        self.max_adjacent = self.config["schema"]["max_adjacent"]

        # Ground-truth note (env.py:135, not in the original spec): info["category"] is set on
        # every reset() regardless of --description-type — use it instead of guessing obj.category
        # from the description text. See parse.py's module docstring for the full rationale.
        self.belief = parse.parse_description(self.target_description, info.get("category"))

        model_id = model_id or self.config["vllm"]["model_id"]
        self.llm_client = LLMClient(
            model_id,
            port=self.config["vllm"]["port"],
            temperature=self.config["vllm"]["temperature"],
            cache_dir=_PACKAGE_ROOT / self.config["llm"]["cache_dir"],
            timeout_s=self.config["llm"]["llm_timeout_s"],
            retries=self.config["llm"]["llm_retries"],
        )
        self.priors = PriorsTable.load(
            _PACKAGE_ROOT / "artifacts" / "priors.json",
            default_disc=self.config["priors"]["default_disc"],
        )
        budget_config = {**self.config["budget"], "front_load": self.config["front_load"]}
        self.budget = budget_mod.BudgetController(budget_config, self.max_adjacent)

        # Base QuestionerInterface.add_answer()/reset_questions() expect these; eval_model.py
        # never calls reset_questions() between candidates (spec §0.5), so seed them here once.
        self.questions: list[str] = []
        self.answers: list[str] = []
        self.reasonings: list[str] = []
        self.time_required = 0.0
        self.n_questions = 0

        self._current_image_hash: str | None = None
        self._current_frame: ObservationFrame | None = None
        self._last_asked_slots: list[str] = []
        self._extraction_failed: bool = False

    def reset_time(self) -> None:
        self.time_required = 0.0
        self.budget.start_time = self.budget._time_fn()

    def add_answer(self, answer: str) -> None:
        self.answers.append(answer)

    def _conclude(self, conclusion: int, reasoning: str) -> dict:
        return dict(question=None, conclusion=conclusion, reasoning=reasoning)

    def _category(self) -> str:
        return self.belief.get("obj.category").canon or ""

    def ask_or_conclude(self, observation: dict) -> dict:
        """Idempotent w.r.t. the image (spec §5): the harness may call this repeatedly for the
        same candidate until a conclusion is returned, so extraction is cached by image hash.
        """
        _validate_observation(observation)
        image = observation["image"]
        image_hash = hash_image(image)
        is_new_candidate = image_hash != self._current_image_hash

        if is_new_candidate:
            self._current_image_hash = image_hash
            k = self.config["extraction"]["extract_self_consistency_k"] if self.budget.candidates_seen == 0 else 1
            try:
                self._current_frame = extract.extract(
                    image, self.llm_client, max_adjacent=self.max_adjacent, self_consistency_k=k,
                )
                self._extraction_failed = False
            except (extract.ExtractionParseError, NotImplementedError):
                # Degrade per spec §7: skip extraction -> treat all slots unknown -> adjudicate.
                # This must go straight to adjudicate() below, NOT fall into the "pool is empty"
                # branch — an all-unknown frame always makes candidate_pool() empty too, which
                # would otherwise read as "nothing left to check, conclude True", a different
                # condition with a different (much more confident) meaning.
                self._current_frame = ObservationFrame(image_hash=image_hash)
                self._extraction_failed = True
            self.budget.start_candidate(self.belief)
            self._last_asked_slots = []

        if observation["answer"] is not None and self._last_asked_slots:
            last_question = self.questions[-1] if self.questions else ""
            for slot_key in self._last_asked_slots:
                value = parse.parse_oracle_answer(observation["answer"], slot_key, question=last_question)
                self.belief.set_slot(slot_key, value)
            self._last_asked_slots = []  # answer consumed; don't reapply on a later idempotent call

        if self._extraction_failed:
            conclusion_bool, motivation, fallback = adjudicate.adjudicate(
                image, self.belief, self.llm_client, skip_call=self.budget.hard_stop(),
            )
            self.budget.record_conclusion()
            self.budget.advance_candidate()
            self.reasonings.append(f"[extraction failed] {motivation}")
            return self._conclude(1 if conclusion_bool else 0, motivation)

        verdict = compare.compare(
            self._current_frame,
            self.belief,
            tau_obs=self.config["comparison"]["tau_obs"],
            weak_conflicts_for_decisive=self.config["comparison"]["weak_conflicts_for_decisive"],
        )
        if verdict.decisive_conflict:
            self.budget.record_conclusion()
            self.budget.advance_candidate()
            return self._conclude(0, verdict.explain())

        pool = select.candidate_pool(
            self._current_frame, self.belief, max_adjacent=self.max_adjacent, tau_obs=self.config["comparison"]["tau_obs"],
        )
        if not pool:
            # An empty pool can mean two different things: genuinely nothing discriminative left
            # (conclude True), or the only discriminative slot(s) came back hedged — permanently
            # excluded from the pool, never decisive, but NOT the same as confirmed evidence.
            # Route the latter to adjudicate() instead of a blind elimination conclusion.
            if select.has_hedged_discriminative_slot(self.belief, self.max_adjacent):
                conclusion_bool, motivation, fallback = adjudicate.adjudicate(
                    image, self.belief, self.llm_client, skip_call=self.budget.hard_stop(),
                )
                self.budget.record_conclusion()
                self.budget.advance_candidate()
                self.reasonings.append(f"[hedged slot forced adjudication] {motivation}")
                return self._conclude(1 if conclusion_bool else 0, motivation)
            self.budget.record_conclusion()
            self.budget.advance_candidate()
            return self._conclude(1, "No discriminative slots remain unresolved; concluding match by elimination.")

        if self.budget.may_ask() and not self.budget.hard_stop():
            question, slots = select.top(
                pool, self._current_frame, self.belief, self._category(), self.priors,
                candidates_seen=self.budget.candidates_seen, max_adjacent=self.max_adjacent,
                tier_weight=self.config["selection"]["tier_weight"], stability=self.config["selection"]["stability"],
                allow_bundle=self.config["selection"]["allow_bundle"], max_bundle_slots=self.config["selection"]["max_bundle_slots"],
            )
            for slot_key in slots:
                self.belief.record_question(slot_key, question)
            self._last_asked_slots = slots
            self.budget.record_question()
            self.n_questions += 1
            self.questions.append(question)
            reasoning = f"Asking about: {', '.join(slots)}."
            self.reasonings.append(reasoning)
            return dict(question=question, conclusion=None, reasoning=reasoning)

        skip_call = self.budget.hard_stop()
        conclusion_bool, motivation, fallback = adjudicate.adjudicate(image, self.belief, self.llm_client, skip_call=skip_call)
        self.budget.record_conclusion()
        self.budget.advance_candidate()
        self.reasonings.append(motivation)
        return self._conclude(1 if conclusion_bool else 0, motivation)
