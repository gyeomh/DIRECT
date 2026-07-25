"""Final match/no-match call: pixels + belief text, never the slot frame (spec §5.4, principle §1.2).

The slot frame is deliberately withheld here — it exists only to propose and rank questions.
The actual conclusion comes from a VLM looking at the real candidate image plus the belief
rendered as prose, so a slot-extraction error costs a wasted question, never a wrong conclusion.

`ADJUDICATION_PROMPT` is the second prompt we're writing together (after extract.py's). For
reference, `Questioner.py:QUESTIONER_EXAMPLE_PROMPT` is the organizers' own single-call baseline
and already uses the same `<motivation>/<score>/<question>` output format this module parses —
worth reusing its phrasing conventions rather than inventing a new format from scratch.
"""

import re

from .llm import LLMClient, LLMCallFailed
from .state import TargetBelief

# TODO(together): write this. Must ask for exactly <motivation>...</motivation><score>0|1|2</score>,
# motivation under 60 words with no double quotes, and must pass belief.render_text() + the
# description + qa_history_text — but NOT the slot frame (see module docstring).
ADJUDICATION_PROMPT = None

_RESPONSE_RE = re.compile(
    r"<motivation>(?P<motivation>.*?)</motivation>\s*<score>(?P<score>[012])</score>",
    re.DOTALL,
)

_STRICT_RETRY_SUFFIX = (
    "\n\nYour previous reply did not match the required format. Reply with EXACTLY: "
    "<motivation>...</motivation><score>0</score> (or 1, or 2). Nothing else."
)


class AdjudicationParseError(RuntimeError):
    pass


def parse_response(text: str) -> tuple[str, int]:
    m = _RESPONSE_RE.search(text)
    if not m:
        raise AdjudicationParseError(f"Could not find <motivation>/<score> tags in: {text!r}")
    return m.group("motivation").strip(), int(m.group("score"))


def score_to_conclusion(score: int) -> bool:
    """2 -> True, 0 -> False, 1 (residual 'unsure') -> False — with one matching candidate per
    episode the base rate on any given candidate is roughly 1/6, so False is the better default
    than a coin flip (spec §5.4).
    """
    return score == 2


def qa_history_text(belief: TargetBelief) -> str:
    if not belief.asked:
        return "(no questions asked yet)"
    lines = []
    for slot_key, question in belief.asked:
        value = belief.get(slot_key)
        answer = value.raw if value.raw is not None else "(no answer recorded)"
        lines.append(f"Q: {question}\nA: {answer}")
    return "\n".join(lines)


def adjudicate(
    image,
    belief: TargetBelief,
    llm_client: LLMClient,
    *,
    skip_call: bool = False,
) -> tuple[bool, str, str | None]:
    """Returns (conclusion, motivation, fallback_used). `fallback_used` is None on a clean parse —
    log it whenever it isn't; spec §5.4: "a high fallback rate is a bug, not a strategy."
    """
    if skip_call:
        return False, "Hard time budget exceeded before adjudication could run.", "hard_time_budget"

    if ADJUDICATION_PROMPT is None:
        raise NotImplementedError(
            "adjudicate.ADJUDICATION_PROMPT is not written yet — see the module docstring TODO."
        )

    base_prompt = ADJUDICATION_PROMPT.format(
        description=belief.description,
        belief_text=belief.render_text(),
        qa_history=qa_history_text(belief),
    )

    for attempt_prompt, fallback_tag in ((base_prompt, None), (base_prompt + _STRICT_RETRY_SUFFIX, "retry_parse")):
        try:
            result = llm_client.call(attempt_prompt, image, temperature=0.0)
        except LLMCallFailed:
            continue
        try:
            motivation, score = parse_response(result.text)
            return score_to_conclusion(score), motivation, fallback_tag
        except AdjudicationParseError:
            continue

    return False, "Adjudicator failed to produce a parseable response twice; defaulting to no-match.", "double_parse_failure"
