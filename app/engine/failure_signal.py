"""Learning from what failed, not only from what survived.

Smart Mode weights a gene by the scores of the Vault strategies it
appears in. That is learning from winners, and it leaves one thing
invisible: the difference between a gene nobody has tried and a gene
that has been drawn four thousand times and has never once survived
screening. Both look identical to Smart Mode - absent from the Vault -
and they are not remotely the same fact.

This is the other half. Phase 0 draws thousands of genomes per campaign
and throws almost all of them away; counting what went in and what came
out costs two integers per gene and answers the question directly.

What it deliberately is not is a clustering algorithm over failed
genomes. Storing thousands of rejected genomes per campaign to run
dimensionality reduction over them would be a great deal of machinery to
arrive at the same advice two counters already give: this gene keeps
being tried and keeps not working.

A penalty never reaches zero. "Has failed so far" is not "cannot work" -
a gene that is useless in a trending market may be the one that matters
when the regime turns, and a search that has permanently deleted it from
its own vocabulary cannot find out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

FILENAME = "gene_outcomes.json"

# How many draws a gene needs before its survival rate is worth acting
# on. Below this, a run of bad luck looks exactly like a bad gene.
MIN_DRAWS_TO_JUDGE = 200

# The strongest the penalty gets. A gene that never survives is drawn a
# quarter as often as one that always does - discouraged, never silenced.
PENALTY_FLOOR = 0.25


@dataclass
class GeneOutcomes:
    drawn: dict[str, int] = field(default_factory=dict)
    survived: dict[str, int] = field(default_factory=dict)
    campaigns: int = 0
    updated_at: str = ""

    def survival_rate(self, key: str) -> float | None:
        drawn = self.drawn.get(key, 0)
        if drawn < MIN_DRAWS_TO_JUDGE:
            return None
        return self.survived.get(key, 0) / drawn

    def to_dict(self) -> dict:
        return {
            "drawn": self.drawn,
            "survived": self.survived,
            "campaigns": self.campaigns,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GeneOutcomes":
        return cls(
            drawn={str(k): int(v) for k, v in (data.get("drawn") or {}).items()},
            survived={str(k): int(v) for k, v in (data.get("survived") or {}).items()},
            campaigns=int(data.get("campaigns", 0)),
            updated_at=str(data.get("updated_at", "")),
        )


def gene_keys(genome) -> list[str]:
    """Every gene a genome uses, as `layer::name`."""
    keys = []
    for layer in ("signal", "bias", "execution", "management"):
        choice = getattr(genome, layer, None)
        if choice is not None:
            keys.append(f"{layer}::{choice.name}")
    for choice in getattr(genome, "filters", None) or []:
        keys.append(f"filter::{choice.name}")
    return keys


def record_screening(outcomes: GeneOutcomes, drawn_genomes: list, surviving_genomes: list) -> GeneOutcomes:
    """Counts one screening round into the running totals."""
    for genome in drawn_genomes:
        for key in gene_keys(genome):
            outcomes.drawn[key] = outcomes.drawn.get(key, 0) + 1
    for genome in surviving_genomes:
        for key in gene_keys(genome):
            outcomes.survived[key] = outcomes.survived.get(key, 0) + 1
    outcomes.updated_at = datetime.now(timezone.utc).isoformat()
    return outcomes


def penalties(outcomes: GeneOutcomes) -> dict[str, float]:
    """A multiplier per gene, from how often it survives screening.

    Scaled against the best-performing gene rather than against an
    absolute rate: screening is a hard gate and a 2% survival rate may be
    the best anything achieves on a given market. What matters is which
    genes are worse than their peers, not that all of them look small.
    """
    rates = {key: rate for key in outcomes.drawn if (rate := outcomes.survival_rate(key)) is not None}
    if not rates:
        return {}

    best = max(rates.values())
    if best <= 0:
        # Nothing survived anything: that is a statement about the market
        # or the costs, not about any one gene.
        return {}

    return {key: PENALTY_FLOOR + (1 - PENALTY_FLOOR) * min(1.0, rate / best) for key, rate in rates.items()}


def load(data_dir: Path) -> GeneOutcomes:
    path = data_dir / FILENAME
    if not path.exists():
        return GeneOutcomes()
    try:
        return GeneOutcomes.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        # Corrupt counters are not worth failing a campaign over; the
        # search simply goes back to drawing uniformly.
        return GeneOutcomes()


def save(data_dir: Path, outcomes: GeneOutcomes) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / FILENAME).write_text(json.dumps(outcomes.to_dict(), indent=2), encoding="utf-8")


def reset(data_dir: Path) -> None:
    path = data_dir / FILENAME
    if path.exists():
        path.unlink()
