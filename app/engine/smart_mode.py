"""Smart Mode — learning which genes have worked — tasks.md 12.3.

Every campaign starts by drawing genomes uniformly at random. After a few
campaigns the Vault already knows something that draw ignores: which genes
keep turning up in strategies that survived Phase 4, and which never do.

Smart Mode turns that into sampling weights. A gene that appears in
high-scoring Vault strategies is drawn more often; a gene that has never
produced one is drawn less often — but never zero, because "has not worked
yet" is not "cannot work", and a gene pool that silently collapses onto
last month's winners stops searching.

The weights are advisory. The Gene Pool toggles (REQ-ENGINE-1.3) still
decide what may be drawn at all; Smart Mode only changes how often.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.genome.genes import ALL_GENES
from app.models import Strategy

# Weight bounds. The floor keeps every enabled gene reachable; the ceiling
# stops one lucky campaign from dominating the next.
MIN_WEIGHT = 0.25
MAX_WEIGHT = 3.0
# Below this many Vault strategies the sample says more about luck than
# about genes, so weights stay uniform.
MIN_STRATEGIES_TO_LEARN = 20
FILENAME = "gene_fitness_v2.json"


@dataclass
class GeneFitness:
    weights: dict[str, float] = field(default_factory=dict)  # "{layer}::{name}" -> weight
    strategies_learned_from: int = 0
    computed_at: str = ""
    reason: str = ""

    def weight_for(self, layer: str, name: str) -> float:
        return self.weights.get(f"{layer}::{name}", 1.0)

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "strategies_learned_from": self.strategies_learned_from,
            "computed_at": self.computed_at,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GeneFitness":
        return cls(
            weights={str(k): float(v) for k, v in (data.get("weights") or {}).items()},
            strategies_learned_from=int(data.get("strategies_learned_from", 0)),
            computed_at=str(data.get("computed_at", "")),
            reason=str(data.get("reason", "")),
        )


def _genes_of(strategy: Strategy) -> list[str]:
    try:
        genome = json.loads(strategy.genome_json)
    except (json.JSONDecodeError, TypeError):
        return []

    keys = []
    for layer in ("signal", "bias", "execution", "management"):
        entry = genome.get(layer)
        if entry:
            keys.append(f"{layer}::{entry['name']}")
    for entry in genome.get("filters") or []:
        keys.append(f"filter::{entry['name']}")
    return keys


def compute_gene_fitness(strategies: list[Strategy]) -> GeneFitness:
    """Sampling weights from the scores of the strategies each gene is in."""
    now = datetime.now(timezone.utc).isoformat()
    usable = [s for s in strategies if s.deleted_at is None and s.score is not None]

    if len(usable) < MIN_STRATEGIES_TO_LEARN:
        return GeneFitness(
            weights={},
            strategies_learned_from=len(usable),
            computed_at=now,
            reason=(
                f"{len(usable)} scored strategies is too few to learn from "
                f"(needs {MIN_STRATEGIES_TO_LEARN}) — sampling stays uniform"
            ),
        )

    average_score = sum(s.score for s in usable) / len(usable)
    totals: dict[str, list[float]] = {}
    for strategy in usable:
        for key in _genes_of(strategy):
            totals.setdefault(key, []).append(float(strategy.score))

    weights: dict[str, float] = {}
    for (layer, name) in ALL_GENES:
        key = f"{layer}::{name}"
        scores = totals.get(key)
        if not scores:
            # Never produced a Vault strategy: drawn less often, never
            # excluded — the search must stay able to rediscover it.
            weights[key] = MIN_WEIGHT
            continue
        # Relative to the Vault's own average, so the scale is meaningful
        # whatever the score distribution happens to be.
        ratio = (sum(scores) / len(scores)) / average_score if average_score else 1.0
        # Genes seen in more strategies move further from 1.0 than genes
        # seen once: evidence, not a single lucky draw.
        confidence = min(1.0, len(scores) / 10)
        weights[key] = round(max(MIN_WEIGHT, min(MAX_WEIGHT, 1.0 + (ratio - 1.0) * confidence * 2)), 3)

    return GeneFitness(
        weights=weights,
        strategies_learned_from=len(usable),
        computed_at=now,
        reason=f"learned from {len(usable)} scored strategies",
    )


def save(fitness: GeneFitness, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / FILENAME
    path.write_text(json.dumps(fitness.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load(data_dir: Path) -> GeneFitness:
    path = data_dir / FILENAME
    if not path.exists():
        return GeneFitness(reason="Smart Mode has not learned anything yet")
    try:
        return GeneFitness.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError, ValueError):
        return GeneFitness(reason="stored gene fitness could not be read — sampling stays uniform")


def clear(data_dir: Path) -> bool:
    """Settings → Smart Mode's reset action. Returns whether a file was removed."""
    path = data_dir / FILENAME
    if path.exists():
        path.unlink()
        return True
    return False
