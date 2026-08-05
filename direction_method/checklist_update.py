"""checklist_update (SPEC.md §12): the last of the four modules. current checklist + new
(relation, answer) pairs from one candidate image -> updated checklist.

[CORRECTED -- 2026-08-05] The 51.5% figure below traced back to full_sweep_v1, which was
FakeVLMBackend placeholder output, not real model output -- root cause and full writeup in
SPEC.md SS13. That number described the OLD (now-removed) LLM-classification design, which no
longer exists to re-measure -- there is nothing left to misfile a key, by construction, since
`relation` is now taken directly from the code, never asked of a model. A real rerun
(full_sweep_real, 2026-08-05, empty cache, real vllm backend, 1002/1002 runs, 612/1002 full
success) confirms zero "fake" contamination in this design's output, but doesn't re-derive a
misfile rate -- there's no comparable statistic anymore, the design eliminated the failure mode
rather than reducing its rate.

No LLM call. Confirmed against the real full 167-episode x 6-description-type sweep: the
previous LLM-driven design (ask the model to extract assertions from each answer AND choose
which of the 11 checklist keys to file them under) misfiled content under the wrong key in 51.5%
of candidates where the checklist grew at all -- e.g. the oracle's answer to "what is on the left
of the target?" (a mirror) landed under "left-bottom" instead of "left", despite the prompt's own
rule 2 ("KEEP THE ASKED KEY... even if the answer mentions a different position") saying not to.
The model was routinely re-deriving its own key from the answer's content instead of keeping the
key it was actually asked about.

This is the exact same lesson as context_parser's other_objects fix (§10): once the correct key
is already known in code, never ask the model to reclassify it. Every (relation, answer) pair in
round_answers already carries its own correct key -- `relation` is the exact region key the
question was generated for (templates.question_for/region_for), fixed by zone_gen's resolved
relations before the question was ever asked, not something inferrable-and-therefore-guessable
from the answer text. So there is nothing left to classify: the answer is filed verbatim, in code,
under `relation`. No atomic splitting, no framing-strip, no rephrasing -- the checklist entry is
the oracle's own sentence.
"""

import re


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class ChecklistUpdateError(Exception):
    """Raised if a merge would violate the append-only invariant (should be unreachable given the
    code below; kept as a runtime guard, not a real failure mode)."""


def merge_checklist(checklist: dict, additions: dict) -> dict:
    """Merge in code. Returns a NEW dict -- `checklist` itself is never mutated, so the caller
    keeps a clean pre-merge reference for the superset check below.

    - append `additions[key]` to the existing list under `key`
    - never modify, reword, reorder, or delete an existing assertion
    - exact-match dedup after normalizing case/whitespace -- checked against both the
      pre-existing assertions AND assertions already added earlier in this same merge
    """
    merged = {key: list(assertions) for key, assertions in checklist.items()}

    for key, new_assertions in additions.items():
        existing = merged.setdefault(key, [])
        seen_normalized = {_normalize(a) for a in existing}
        for assertion in new_assertions:
            normalized = _normalize(assertion)
            if normalized in seen_normalized:
                continue
            existing.append(assertion)
            seen_normalized.add(normalized)

    _assert_superset(checklist, merged)
    return merged


def _assert_superset(pre: dict, post: dict) -> None:
    # Append-only implies every pre-existing assertion list survives as an exact, unreordered
    # prefix of the post-merge list for that key.
    for key, assertions in pre.items():
        post_assertions = post.get(key, [])
        if post_assertions[: len(assertions)] != assertions:
            raise ChecklistUpdateError(f"merge violated the append-only invariant for key {key!r}")


def checklist_update(checklist: dict, round_answers: list) -> dict:
    if not round_answers:
        return checklist  # nothing new was asked this round
    additions = {}
    for relation, answer in round_answers:
        additions.setdefault(relation, []).append(answer)
    return merge_checklist(checklist, additions)
