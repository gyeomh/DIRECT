"""Final match/no-match call: pixels + belief text, never the slot frame (spec §5.4, principle §1.2).

The slot frame is deliberately withheld here — it exists only to propose and rank questions.
The actual conclusion comes from a VLM looking at the real candidate image plus the belief
rendered as prose, so a slot-extraction error costs a wasted question, never a wrong conclusion.

Unlike extract.py, this prompt is NOT description-blind — the whole point of this call is to
compare the candidate against the target concept, so it names "the target" freely. Reuses the
organizers' own `<motivation>/<score>` format (`Questioner.py:QUESTIONER_EXAMPLE_PROMPT`) minus
the `<question>` tag — this is a terminal call (budget already decided no more questions get
asked), so there's nothing to ask.
"""

import re

from .llm import LLMClient, LLMCallFailed
from .state import TargetBelief

ADJUDICATION_PROMPT = """You are deciding whether the object shown in this image is the same \
specific object as a target object described below, which you have not seen directly.

Target description: "{description}"

Additional confirmed facts about the target (from the description, or from questions already \
asked and answered by an oracle who has seen the target directly):
{belief_text}

Questions already asked and their answers:
{qa_history}

Look at the image above. Compare what you actually see in it against the description and the \
confirmed facts. Candidates are often near-duplicates that differ only in a few details (color, \
material, a nearby object, the room) — a single concrete, confirmed mismatch on any of these \
means this is NOT the same object, even if everything else matches. Ignore compression \
artifacts, digital noise, or rendering glitches entirely — never treat them as a real difference.

Provide your reasoning, then a score:
- 2 if you are confident this is the same specific object as the target.
- 0 if you are confident this is NOT the same object (something concretely conflicts).
- 1 if you are genuinely unsure either way.

Strictly follow this output format: <motivation>your reasoning here, under 60 words, do NOT use \
double quotes</motivation><score>0, 1, or 2</score>
"""

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
    """A bundled question (select.py bundles ≤2 same-region slots into one question) is recorded
    once per slot in `belief.asked` — same question text, different slot_key — since one oracle
    answer resolves both. Dedupe by question text so a bundled Q&A doesn't print twice.
    """
    if not belief.asked:
        return "(no questions asked yet)"
    seen_questions: set[str] = set()
    lines = []
    for slot_key, question in belief.asked:
        if question in seen_questions:
            continue
        seen_questions.add(question)
        value = belief.get(slot_key)
        answer = value.raw if value.raw is not None else "(no answer recorded)"
        lines.append(f"Q: {question}\nA: {answer}")
    return "\n".join(lines)


def _build_adjudication_prompt(belief: TargetBelief) -> str:
    return ADJUDICATION_PROMPT.format(
        description=belief.description,
        belief_text=belief.render_text(),
        qa_history=qa_history_text(belief),
    )


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

    base_prompt = _build_adjudication_prompt(belief)

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
