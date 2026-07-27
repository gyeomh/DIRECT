"""Throwaway questioner for ENV.md step-1 verification ONLY. Not method code — asks one
placeholder question per candidate (to exercise the oracle round-trip through env.py's
ask_oracle -> MockOracle path), then always concludes "match".

Since the matching candidate is always last (verified in scripts/verify_env.py's §7 check),
"always conclude match" reliably exercises both of ENV.md §1's termination conditions: a wrong
conclusion terminates immediately on every non-final candidate, and a correct conclusion
terminates on the final one (including trivially for every 1-candidate episode).
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for Questioner.py

from Questioner import QuestionerInterface  # noqa: E402


class TrivialQuestioner(QuestionerInterface):
    def __init__(self, info):
        super().__init__(info)
        self.questions = []
        self.answers = []
        self.time_required = 0.0
        self.n_questions = 0
        self._current_hash = None
        self._asked_this_candidate = False

    def ask_or_conclude(self, observation):
        img_hash = hashlib.sha256(observation["image"].tobytes()).hexdigest()
        if img_hash != self._current_hash:
            self._current_hash = img_hash
            self._asked_this_candidate = False

        if not self._asked_this_candidate:
            self._asked_this_candidate = True
            self.n_questions += 1
            q = "What color is the main object in this image?"
            self.questions.append(q)
            return dict(question=q, conclusion=None, reasoning="verification stub: placeholder question")

        return dict(question=None, conclusion=1, reasoning="verification stub: always concludes match")
