"""Genome model — REQ-ENGINE-1.1.

A genome selects one gene per required layer (Signal, Execution,
Management) plus optional Bias (neutral if absent) and 0-2 Filters, drawn
only from genes enabled in the campaign's Gene Pool (REQ-ENGINE-1.3).
"""

from __future__ import annotations

import copy
import random
import uuid
from dataclasses import dataclass, field

from app.genome.gene_pool import GenePoolConfig
from app.genome.genes import BIAS_GENES, EXECUTION_GENES, FILTER_GENES, MANAGEMENT_GENES, SIGNAL_GENES

MAX_FILTERS_PER_GENOME = 2


@dataclass
class GeneChoice:
    name: str
    params: dict = field(default_factory=dict)


@dataclass
class Genome:
    signal: GeneChoice
    execution: GeneChoice
    # Optional: the backtester always applies a baseline ATR stop/target
    # (see app/engine/backtest.py DEFAULT_STOP/TARGET_ATR_MULT) so a genome
    # can be evaluated with no Management gene. REQ-ENGINE-1.4's gate is
    # enforced pool-wide at Phase 3 (see app/engine/phases.py), not here.
    management: GeneChoice | None = None
    bias: GeneChoice | None = None
    filters: list[GeneChoice] = field(default_factory=list)
    genome_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict:
        return {
            "genome_id": self.genome_id,
            "signal": {"name": self.signal.name, "params": self.signal.params},
            "bias": {"name": self.bias.name, "params": self.bias.params} if self.bias else None,
            "filters": [{"name": f.name, "params": f.params} for f in self.filters],
            "execution": {"name": self.execution.name, "params": self.execution.params},
            "management": {"name": self.management.name, "params": self.management.params} if self.management else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        return cls(
            signal=GeneChoice(data["signal"]["name"], data["signal"]["params"]),
            bias=GeneChoice(data["bias"]["name"], data["bias"]["params"]) if data.get("bias") else None,
            filters=[GeneChoice(f["name"], f["params"]) for f in data.get("filters", [])],
            execution=GeneChoice(data["execution"]["name"], data["execution"]["params"]),
            management=GeneChoice(data["management"]["name"], data["management"]["params"]) if data.get("management") else None,
            genome_id=data.get("genome_id", uuid.uuid4().hex[:12]),
        )


def _random_choice(pool: GenePoolConfig, layer: str, catalog: dict) -> GeneChoice | None:
    enabled = pool.enabled_names(layer)  # type: ignore[arg-type]
    if not enabled:
        return None
    name = random.choice(enabled)
    return GeneChoice(name, copy.deepcopy(catalog[name].default_params))


def random_genome(pool: GenePoolConfig) -> Genome:
    """Draws a random genome from the enabled subset of each layer. Signal
    and Execution are required layers and raise if nothing is enabled for
    them. Management is optional at the genome level — REQ-ENGINE-1.4's
    "all Management genes disabled" gate is a pool-wide Phase 3 check (see
    app/engine/phases.py), not a per-genome generation failure."""
    signal = _random_choice(pool, "signal", SIGNAL_GENES)
    execution = _random_choice(pool, "execution", EXECUTION_GENES)
    if signal is None:
        raise ValueError("No SIGNAL genes enabled — cannot generate a genome")
    if execution is None:
        raise ValueError("No EXECUTION genes enabled — cannot generate a genome")

    management = _random_choice(pool, "management", MANAGEMENT_GENES) if pool.enabled_names("management") else None
    bias = _random_choice(pool, "bias", BIAS_GENES) if pool.enabled_names("bias") and random.random() > 0.2 else None

    filter_pool = pool.enabled_names("filter")
    num_filters = random.randint(0, min(MAX_FILTERS_PER_GENOME, len(filter_pool)))
    filter_names = random.sample(filter_pool, num_filters) if filter_pool else []
    filters = [GeneChoice(name, copy.deepcopy(FILTER_GENES[name].default_params)) for name in filter_names]

    return Genome(signal=signal, execution=execution, management=management, bias=bias, filters=filters)


def random_population(pool: GenePoolConfig, size: int) -> list[Genome]:
    return [random_genome(pool) for _ in range(size)]


def crossover(parent_a: Genome, parent_b: Genome) -> Genome:
    """Single-point layer crossover: each layer is independently inherited
    from one parent or the other — REQ-ENGINE-3.5 (Phase 3 Hybridization)."""
    def pick(a, b):
        return copy.deepcopy(random.choice([a, b]))

    child_filters = pick(parent_a.filters, parent_b.filters)
    return Genome(
        signal=pick(parent_a.signal, parent_b.signal),
        bias=pick(parent_a.bias, parent_b.bias),
        filters=child_filters,
        execution=pick(parent_a.execution, parent_b.execution),
        management=pick(parent_a.management, parent_b.management),
    )


def mutate(genome: Genome, pool: GenePoolConfig, mutation_rate: float = 0.1) -> Genome:
    """Randomly re-rolls each layer with probability `mutation_rate`,
    respecting the Gene Pool's enabled genes."""
    mutated = copy.deepcopy(genome)
    if random.random() < mutation_rate:
        choice = _random_choice(pool, "signal", SIGNAL_GENES)
        if choice:
            mutated.signal = choice
    if random.random() < mutation_rate:
        mutated.bias = (
            _random_choice(pool, "bias", BIAS_GENES)
            if pool.enabled_names("bias") and random.random() > 0.2
            else None
        )
    if random.random() < mutation_rate:
        choice = _random_choice(pool, "execution", EXECUTION_GENES)
        if choice:
            mutated.execution = choice
    if random.random() < mutation_rate:
        choice = _random_choice(pool, "management", MANAGEMENT_GENES)
        if choice:
            mutated.management = choice
    if random.random() < mutation_rate:
        filter_pool = pool.enabled_names("filter")
        num_filters = random.randint(0, min(MAX_FILTERS_PER_GENOME, len(filter_pool))) if filter_pool else 0
        names = random.sample(filter_pool, num_filters) if filter_pool else []
        mutated.filters = [GeneChoice(n, copy.deepcopy(FILTER_GENES[n].default_params)) for n in names]
    return mutated
