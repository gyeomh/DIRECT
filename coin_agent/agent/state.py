"""TargetBelief / ObservationFrame / SlotValue (spec §4).

The persistent object across an episode is the target belief, not any per-image structure —
everything learned while judging candidate #1 remains true for #2-#6 (spec §1.1). Only the
ObservationFrame is rebuilt per candidate image.
"""

from dataclasses import dataclass, field
from typing import Literal

Provenance = Literal["description", "oracle", "prior"]
Certainty = Literal["resolved", "hedged", "unknown"]


@dataclass
class SlotValue:
    raw: str | None = None            # verbatim source text
    canon: str | None = None          # canonical value, None if unresolved
    confidence: float = 0.0           # [0, 1]
    certainty: Certainty = "unknown"
    provenance: Provenance | None = None   # TargetBelief only; None on ObservationFrame
    source_question: str | None = None

    @staticmethod
    def unknown() -> "SlotValue":
        return SlotValue(certainty="unknown", confidence=0.0)


class MonotonicityViolation(RuntimeError):
    pass


@dataclass
class TargetBelief:
    """Accumulates everything known about the target object across the whole episode."""

    description: str
    # Short noun phrase used to name the object in oracle-facing questions (spec §5.3: "the
    # object is named with the noun phrase from the description ... never 'it' / 'the target'").
    # Filled once at __init__ time by the same description-parse step that seeds `slots` — see
    # `parse.extract_noun_phrase`. Falls back to `description` itself until that's wired in.
    noun_phrase: str = ""
    slots: dict[str, SlotValue] = field(default_factory=dict)
    asked: list[tuple[str, str]] = field(default_factory=list)  # (slot_key, question) — never re-ask a slot

    def __post_init__(self):
        if not self.noun_phrase:
            self.noun_phrase = self.description

    def get(self, slot_key: str) -> SlotValue:
        return self.slots.get(slot_key, SlotValue.unknown())

    def set_slot(self, slot_key: str, value: SlotValue) -> None:
        """Fill a slot. Monotonic: a slot already resolved from description/oracle is never
        overwritten (spec §4 "Monotonicity" — the target image never changes, so belief only
        accumulates). Re-deriving the identical canonical value is a no-op, not a violation.
        """
        existing = self.slots.get(slot_key)
        if existing is not None and existing.provenance in ("description", "oracle") and existing.canon is not None:
            if value.canon != existing.canon:
                raise MonotonicityViolation(
                    f"Refusing to overwrite resolved slot '{slot_key}' "
                    f"({existing.canon!r} from {existing.provenance} -> {value.canon!r})"
                )
            return
        self.slots[slot_key] = value

    def has_asked(self, slot_key: str) -> bool:
        return any(k == slot_key for k, _ in self.asked)

    def record_question(self, slot_key: str, question: str) -> None:
        self.asked.append((slot_key, question))

    def render_text(self) -> str:
        """Render the belief as prose for the adjudicator prompt. Never repeats the raw
        description verbatim (the description is already passed separately) — only the slots
        derived *from* it plus anything learned via oracle questions.
        """
        lines = []
        for slot_key, value in sorted(self.slots.items()):
            if value.canon is None:
                continue
            tag = "hedged" if value.certainty == "hedged" else value.provenance
            lines.append(f"- {slot_key}: {value.canon} (source: {tag})")
        if not lines:
            return "(no slots resolved yet)"
        return "\n".join(lines)


@dataclass
class ObservationFrame:
    """Same shape as TargetBelief.slots, minus provenance, plus an image identity for caching."""

    image_hash: str
    slots: dict[str, SlotValue] = field(default_factory=dict)

    def get(self, slot_key: str) -> SlotValue:
        return self.slots.get(slot_key, SlotValue.unknown())


def new_belief(description: str) -> TargetBelief:
    return TargetBelief(description=description)
