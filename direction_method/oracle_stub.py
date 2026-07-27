"""Local oracle stand-in (ENV.md §2: "The oracle is not ours... any local oracle is a
development stand-in"). Sees only the target image, exactly like the real oracle — callers must
never pass it a candidate image.

Single source of truth for the prompt text: `env.ANSWER_PROMPT`, after
`patches.apply_patches.fix_answer_prompt_typo()` has corrected it in memory. This class doesn't
carry its own copy of the template, so it can't drift from whichever version of the constant
`env` currently holds — call `fix_answer_prompt_typo()` once before constructing this if the
typo-fixed wording is wanted (see patches/README.md #6 for why the official harness disagrees).
"""

import sys
from pathlib import Path

_DIRECTION_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _DIRECTION_ROOT.parent
for p in (_DIRECTION_ROOT, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from Oracle import OracleInterface  # noqa: E402

from llm import LLMClient  # noqa: E402


class LocalOracleStandIn(OracleInterface):
    """Satisfies OracleInterface, so it can be passed directly as QAEnv's `client` argument
    (env.py's own _get_observation() formats ANSWER_PROMPT and calls .ask(prompt=..., images=[target]))
    -- or used directly via ask_question(), bypassing QAEnv's step loop entirely, for module-in-
    isolation testing (SPEC.md §9 step 3), which is how the self_check experiment uses it.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def ask(self, *, prompt: str = "", images=None):
        image = images[0] if images else None
        return self.llm_client.call(prompt, image).text

    def ask_question(self, question: str, target_image) -> str:
        import env  # imported lazily so fix_answer_prompt_typo() (called by the caller first) is visible

        prompt = env.ANSWER_PROMPT.format(QUESTION=question)
        return self.ask(prompt=prompt, images=[target_image])
