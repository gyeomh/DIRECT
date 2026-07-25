"""Runtime lookup for the empirical discriminativeness table (spec §6.2).

`disc(s, v, category)` estimates P(a plausible distractor of `category` does *not* share value
`v` on slot `s`). This module only *reads* `artifacts/priors.json`; building it (running
extract() over every training candidate, offline, once) is `scripts/build_priors.py`.
"""

import json
from pathlib import Path

DEFAULT_DISC = 0.5


class PriorsTable:
    def __init__(self, table: dict | None = None, default_disc: float = DEFAULT_DISC):
        # table shape: {category: {slot_key: {canonical_value: disc_float}}}
        self._table = table or {}
        self.default_disc = default_disc

    @classmethod
    def load(cls, path: str | Path, default_disc: float = DEFAULT_DISC) -> "PriorsTable":
        path = Path(path)
        if not path.exists():
            return cls({}, default_disc=default_disc)
        with open(path) as f:
            return cls(json.load(f), default_disc=default_disc)

    @classmethod
    def empty(cls, default_disc: float = DEFAULT_DISC) -> "PriorsTable":
        """Used before `build_priors.py` has ever run — rung A2 (`--no-priors`) pins every
        lookup to `default_disc` regardless of what's in the file.
        """
        return cls({}, default_disc=default_disc)

    def disc(self, slot_key: str, canonical_value: str | None, category: str) -> float:
        if canonical_value is None:
            return self.default_disc
        return (
            self._table.get(category, {})
            .get(slot_key, {})
            .get(canonical_value, self.default_disc)
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._table, f, indent=2, sort_keys=True)
