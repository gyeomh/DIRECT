"""self_check (SPEC.md §7): single statement + image -> verdict. One call per statement, never
batched. Two candidate polarities, deliberately NOT chosen here — see
scripts/run_self_check_experiment.py, which measures the false-failure rate of each empirically,
per SPEC's own instruction: "Pick on the data."
"""

import json

from llm import LLMClient

POLARITY_YES_NO = "yes_no"
POLARITY_CONTRADICTS = "contradicts_consistent"
POLARITIES = (POLARITY_YES_NO, POLARITY_CONTRADICTS)

# --- polarity (a): "does the answer align with the image?" ------------------------------------
YES_NO_PROMPT = """Look at the image. Does the following statement accurately describe something \
that is visible in the image?

Statement: "{STATEMENT}"

Answer with exactly one word: "yes" if the statement aligns with what is shown in the image, \
"no" if it does not."""

YES_NO_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["verdict"],
}

# --- polarity (b): "is anything in this description inconsistent with the image?" -------------
CONTRADICTS_PROMPT = """Look at the image and the following statement.

Statement: "{STATEMENT}"

Does anything in the statement directly and clearly contradict what is visible in the image? \
Answer "contradicts" only if something in the statement is definitely wrong given what the image \
shows. Answer "cant_tell" if the image does not show enough to judge one way or the other \
(for example, the statement mentions something outside the frame, or something too small or \
occluded to confirm). Otherwise, if nothing in the statement conflicts with the image, answer \
"consistent"."""

CONTRADICTS_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["contradicts", "consistent", "cant_tell"]}},
    "required": ["verdict"],
}

_TEMPLATES = {
    POLARITY_YES_NO: (YES_NO_PROMPT, YES_NO_SCHEMA),
    POLARITY_CONTRADICTS: (CONTRADICTS_PROMPT, CONTRADICTS_SCHEMA),
}


def build_prompt(polarity: str, statement: str) -> tuple[str, dict]:
    if polarity not in _TEMPLATES:
        raise ValueError(f"Unknown polarity: {polarity!r}, expected one of {POLARITIES}")
    template, schema = _TEMPLATES[polarity]
    return template.format(STATEMENT=statement), schema


def self_check(llm_client: LLMClient, image, statement: str, polarity: str) -> str:
    """Returns the raw verdict string ("yes"/"no", or "contradicts"/"consistent"/"cant_tell"),
    or "PARSE_ERROR" if the model's response didn't parse as the requested schema — a parse
    failure is itself worth counting, not silently swallowed (see is_failure below).
    """
    prompt, schema = build_prompt(polarity, statement)
    result = llm_client.call(prompt, image, response_schema=schema)
    try:
        return json.loads(result.text)["verdict"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return "PARSE_ERROR"


def is_failure(polarity: str, verdict: str) -> bool:
    """SPEC.md §7: for polarity (b), only "contradicts" fails -- "cant_tell" counts as neither
    pass nor fail. For this experiment's false-failure-rate metric specifically (ground truth is
    always "pass"), "neither" is scored as not-a-failure: cant_tell doesn't wrongly terminate an
    episode the way an actual "contradicts"/"no" would.
    """
    if verdict == "PARSE_ERROR":
        return True
    if polarity == POLARITY_YES_NO:
        return verdict == "no"
    if polarity == POLARITY_CONTRADICTS:
        return verdict == "contradicts"
    raise ValueError(f"Unknown polarity: {polarity!r}")
