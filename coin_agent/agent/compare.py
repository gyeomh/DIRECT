"""Conflict detection between an ObservationFrame and the TargetBelief (spec §5.2).

Evidence is asymmetric (spec §1.3): one decisive CONFLICT proves False; confirmations never prove
identity (candidates are near-duplicates by construction). So this module's only job is to find
the strongest disagreement, not to accumulate positive evidence — that's the adjudicator's job.
"""

from dataclasses import dataclass, field

from . import canon, schema
from .state import ObservationFrame, TargetBelief

MATCH = "MATCH"
WEAK_CONFLICT = "WEAK_CONFLICT"
CONFLICT = "CONFLICT"
UNKNOWN = "UNKNOWN"
INCOMPARABLE = "INCOMPARABLE"


@dataclass
class SlotVerdict:
    slot_key: str
    verdict: str
    frame_value: str | None
    belief_value: str | None


@dataclass
class CompareResult:
    per_slot: dict[str, SlotVerdict] = field(default_factory=dict)
    decisive_conflict: bool = False
    decisive_slot: str | None = None
    weak_conflict_slots: list[str] = field(default_factory=list)

    def explain(self) -> str:
        """Readable reasoning string for the `reasoning` field (spec §5.2: "logged, so make it
        readable: which slot, which two values, which source").
        """
        if self.decisive_conflict:
            v = self.per_slot[self.decisive_slot]
            return (
                f"CONFLICT on '{v.slot_key}': candidate shows {v.frame_value!r}, "
                f"target belief says {v.belief_value!r} -> not a match."
            )
        if self.weak_conflict_slots:
            parts = [
                f"{k}={self.per_slot[k].frame_value!r} vs {self.per_slot[k].belief_value!r}"
                for k in self.weak_conflict_slots
            ]
            return "Weak conflicts (non-decisive): " + "; ".join(parts)
        return "No conflicts detected across comparable slots."


def _slot_type(slot_key: str) -> str:
    return schema.spec_for(slot_key).type


def compare_slot(slot_key: str, frame_value, belief_value, tau_obs: float) -> str:
    if belief_value.canon is None or frame_value.canon is None:
        return UNKNOWN if (belief_value.certainty == "unknown" or frame_value.certainty == "unknown") else INCOMPARABLE

    value_type = _slot_type(slot_key)
    if value_type not in schema.VOCAB:
        # Free-text slots (ctx.above.object, ctx.support.object, obj.style, ...) have no synonym
        # table, so there's no NEAR cluster to fall back on: two different noun strings are
        # treated as FAR outright. This matters because ctx.above.object / ctx.support.object are
        # Tier A specifically for their stability (§3.2) — if mismatches here only ever produced
        # NEAR, those slots could never be decisive despite being curated as decisive-eligible.
        # The risk this trades in is false conflicts from paraphrase ("counter" vs "countertop");
        # test rather than assume (ablate.py) once real extractions are available.
        rel = canon.SAME if frame_value.canon == belief_value.canon else canon.FAR
    else:
        rel = canon.relation(value_type, frame_value.canon, belief_value.canon)

    if rel == canon.SAME:
        return MATCH

    is_tier_a = schema.spec_for(slot_key).tier == schema.TIER_A
    hedged_belief = belief_value.certainty != "resolved"
    low_conf_obs = frame_value.confidence < tau_obs

    if rel == canon.FAR and is_tier_a and not hedged_belief and not low_conf_obs:
        return CONFLICT
    return WEAK_CONFLICT


def compare(
    frame: ObservationFrame,
    belief: TargetBelief,
    *,
    tau_obs: float = 0.80,
    weak_conflicts_for_decisive: int | None = None,
) -> CompareResult:
    """§5.2. `weak_conflicts_for_decisive` implements the (off-by-default) rule: that many
    independent Tier-B WEAK_CONFLICTs may also be treated as decisive.
    """
    result = CompareResult()
    shared_slots = set(frame.slots.keys()) & set(belief.slots.keys())

    for slot_key in sorted(shared_slots):
        frame_value = frame.get(slot_key)
        belief_value = belief.get(slot_key)
        verdict = compare_slot(slot_key, frame_value, belief_value, tau_obs)
        result.per_slot[slot_key] = SlotVerdict(slot_key, verdict, frame_value.canon, belief_value.canon)

        if verdict == CONFLICT and not result.decisive_conflict:
            result.decisive_conflict = True
            result.decisive_slot = slot_key
        elif verdict == WEAK_CONFLICT:
            result.weak_conflict_slots.append(slot_key)

    if not result.decisive_conflict and weak_conflicts_for_decisive is not None:
        tier_b_weak = [
            k for k in result.weak_conflict_slots
            if schema.spec_for(k).tier == schema.TIER_B
        ]
        if len(tier_b_weak) >= weak_conflicts_for_decisive:
            result.decisive_conflict = True
            result.decisive_slot = tier_b_weak[0]

    return result
