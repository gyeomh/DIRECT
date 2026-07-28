"""Shared region/question string templates.

Used by both the checklist path (self_check's `region` field, via `region_for`) and the question
path (what is actually asked to the oracle, via `question_for`). A single shared table is
required: the old formula `"{relation} of the {target}"` was only grammatical for `left`/`right`
("above of the cabinet" is wrong), and letting the two paths format independently risks them
drifting apart -- self_check's region string must always describe the same thing the oracle was
actually asked about.

`next to` and `Target` only ever arrive from `context_parser` (never from `zone_gen`), so they have
no entry in `QUESTION_TEMPLATES` -- no question is ever generated for them via this table; the
mandatory first/target-appearance question is a separate, fixed string (SPEC.md §6).
"""

REGION_TEMPLATES = {
    "left": "left of the {t}",
    "right": "right of the {t}",
    "above": "above the {t}",
    "below": "below the {t}",
    "left-top": "above and to the left of the {t}",
    "right-top": "above and to the right of the {t}",
    "left-bottom": "below and to the left of the {t}",
    "right-bottom": "below and to the right of the {t}",
    "on": "on top of the {t}",
    "next to": "next to the {t}",
    "Target": "the {t} itself",
}

QUESTION_TEMPLATES = {
    "left": "What is on the left of the {t}?",
    "right": "What is on the right of the {t}?",
    "above": "What is above the {t}?",
    "below": "What is below the {t}?",
    "left-top": "What is above and to the left of the {t}?",
    "right-top": "What is above and to the right of the {t}?",
    "left-bottom": "What is below and to the left of the {t}?",
    "right-bottom": "What is below and to the right of the {t}?",
    "on": "What is on top of the {t}?",
}

QUESTION_SUFFIX = " Can you describe the shape and color?"

# The full 11-key checklist enum (context_parser's region keys, §10) -- every key REGION_TEMPLATES
# knows about. zone_gen's own REGION_KEYS (§5) is the 9-key subset that excludes "next to"/"Target",
# since those two can never come from a zone_gen call.
CHECKLIST_KEYS = tuple(REGION_TEMPLATES.keys())


def region_for(relation: str, target_phrase: str) -> str:
    """Deterministic string assembly, no VLM call. `relation` is a checklist parent key (any of
    the 11 `CHECKLIST_KEYS`) or a zone_gen relation key (a 9-key subset of the same vocabulary) --
    both vocabularies are covered by the same `REGION_TEMPLATES` table, so the checklist path and
    the oracle-answer path always produce identical region strings for the same relation.
    """
    return REGION_TEMPLATES[relation].format(t=target_phrase)


def question_for(relation: str, target_phrase: str) -> str:
    """Only defined for zone_gen's 9 relation keys -- `next to`/`Target` raise KeyError, since a
    live question is never generated for them (SPEC.md §6)."""
    return QUESTION_TEMPLATES[relation].format(t=target_phrase) + QUESTION_SUFFIX
