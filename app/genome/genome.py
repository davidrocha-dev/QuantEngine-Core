"""Genome model — REQ-ENGINE-1.1.

A genome selects one gene per required layer (Signal, Execution,
Management) plus optional Bias (neutral if absent) and 0-2 Filters, drawn
only from genes enabled in the campaign's Gene Pool (REQ-ENGINE-1.3).
"""

from __future__ import annotations

import copy
import json
import random
import uuid
from dataclasses import dataclass, field

from app.genome.gene_pool import GenePoolConfig
from app.genome.genes import (
    BIAS_GENES,
    EXECUTION_GENES,
    FILTER_GENES,
    MANAGEMENT_GENES,
    SIGNAL_GENES,
    jitter_params,
    sample_params,
)

MAX_FILTERS_PER_GENOME = 2

# Stop/target distance in ATRs — the documented baseline (backtest.py) is
# the centre of each range, which is what a genome built from defaults uses.
DEFAULT_STOP_ATR_MULT = 1.5
DEFAULT_TARGET_ATR_MULT = 3.0
STOP_ATR_MULT_RANGE = (0.8, 3.0)
TARGET_ATR_MULT_RANGE = (1.0, 6.0)


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
    # Risk geometry is part of the strategy, not a constant: the same signal
    # with a 1.0x ATR stop and a 4.0x ATR target is a different strategy than
    # with 2.5x/3.0x. These were hardcoded in the backtester, which meant
    # every candidate in every campaign traded the same reward:risk.
    stop_atr_mult: float = DEFAULT_STOP_ATR_MULT
    target_atr_mult: float = DEFAULT_TARGET_ATR_MULT
    genome_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict:
        return {
            "genome_id": self.genome_id,
            "signal": {"name": self.signal.name, "params": self.signal.params},
            "bias": {"name": self.bias.name, "params": self.bias.params} if self.bias else None,
            "filters": [{"name": f.name, "params": f.params} for f in self.filters],
            "execution": {"name": self.execution.name, "params": self.execution.params},
            "management": {"name": self.management.name, "params": self.management.params} if self.management else None,
            "stop_atr_mult": self.stop_atr_mult,
            "target_atr_mult": self.target_atr_mult,
        }

    def fingerprint(self) -> str:
        """Content identity (ignores `genome_id`): two genomes with the same
        genes and params trade identically, so they are the same strategy."""
        content = self.to_dict()
        content.pop("genome_id")
        return json.dumps(content, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        return cls(
            signal=GeneChoice(data["signal"]["name"], data["signal"]["params"]),
            bias=GeneChoice(data["bias"]["name"], data["bias"]["params"]) if data.get("bias") else None,
            filters=[GeneChoice(f["name"], f["params"]) for f in data.get("filters", [])],
            execution=GeneChoice(data["execution"]["name"], data["execution"]["params"]),
            management=GeneChoice(data["management"]["name"], data["management"]["params"]) if data.get("management") else None,
            stop_atr_mult=data.get("stop_atr_mult", DEFAULT_STOP_ATR_MULT),
            target_atr_mult=data.get("target_atr_mult", DEFAULT_TARGET_ATR_MULT),
            genome_id=data.get("genome_id", uuid.uuid4().hex[:12]),
        )


def _random_choice(
    pool: GenePoolConfig, layer: str, catalog: dict, weights: dict[str, float] | None = None
) -> GeneChoice | None:
    enabled = pool.enabled_names(layer)  # type: ignore[arg-type]
    if not enabled:
        return None
    if weights:
        # Smart Mode (app/engine/smart_mode.py) biases the draw towards
        # genes that have produced Vault strategies. It never excludes an
        # enabled gene — the Gene Pool decides what may be drawn, this only
        # decides how often.
        name = random.choices(enabled, weights=[max(weights.get(f"{layer}::{n}", 1.0), 0.01) for n in enabled])[0]
    else:
        name = random.choice(enabled)
    # Parameters are drawn from the gene's search space, not pinned to the
    # documented defaults — see genes.py `param_space`.
    return GeneChoice(name, sample_params(catalog[name]))


def random_genome(pool: GenePoolConfig, weights: dict[str, float] | None = None) -> Genome:
    """Draws a random genome from the enabled subset of each layer. Signal
    and Execution are required layers and raise if nothing is enabled for
    them. Management is optional at the genome level — REQ-ENGINE-1.4's
    "all Management genes disabled" gate is a pool-wide Phase 3 check (see
    app/engine/phases.py), not a per-genome generation failure."""
    signal = _random_choice(pool, "signal", SIGNAL_GENES, weights)
    execution = _random_choice(pool, "execution", EXECUTION_GENES, weights)
    if signal is None:
        raise ValueError("No SIGNAL genes enabled — cannot generate a genome")
    if execution is None:
        raise ValueError("No EXECUTION genes enabled — cannot generate a genome")

    management = _random_choice(pool, "management", MANAGEMENT_GENES, weights) if pool.enabled_names("management") else None
    bias = (
        _random_choice(pool, "bias", BIAS_GENES, weights)
        if pool.enabled_names("bias") and random.random() > 0.2
        else None
    )

    filter_pool = pool.enabled_names("filter")
    num_filters = random.randint(0, min(MAX_FILTERS_PER_GENOME, len(filter_pool)))
    filter_names = _sample_filters(filter_pool, num_filters, weights)
    filters = [GeneChoice(name, sample_params(FILTER_GENES[name])) for name in filter_names]

    return Genome(
        signal=signal,
        execution=execution,
        management=management,
        bias=bias,
        filters=filters,
        stop_atr_mult=round(random.uniform(*STOP_ATR_MULT_RANGE), 2),
        target_atr_mult=round(random.uniform(*TARGET_ATR_MULT_RANGE), 2),
    )


def _sample_filters(filter_pool: list[str], count: int, weights: dict[str, float] | None) -> list[str]:
    """Distinct filters, drawn with Smart Mode's weights when it has any."""
    if not filter_pool or count <= 0:
        return []
    if not weights:
        return random.sample(filter_pool, count)

    remaining, chosen = list(filter_pool), []
    for _ in range(count):
        picked = random.choices(
            remaining, weights=[max(weights.get(f"filter::{n}", 1.0), 0.01) for n in remaining]
        )[0]
        chosen.append(picked)
        remaining.remove(picked)
    return chosen


def random_population(pool: GenePoolConfig, size: int, weights: dict[str, float] | None = None) -> list[Genome]:
    return [random_genome(pool, weights) for _ in range(size)]


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
        stop_atr_mult=pick(parent_a.stop_atr_mult, parent_b.stop_atr_mult),
        target_atr_mult=pick(parent_a.target_atr_mult, parent_b.target_atr_mult),
    )


def _jitter_choice(choice: GeneChoice | None, catalog: dict, rate: float) -> GeneChoice | None:
    if choice is None or choice.name not in catalog:
        return choice
    return GeneChoice(choice.name, jitter_params(catalog[choice.name], choice.params, rate))


def mutate(genome: Genome, pool: GenePoolConfig, mutation_rate: float = 0.1) -> Genome:
    """Randomly re-rolls each layer with probability `mutation_rate`,
    respecting the Gene Pool's enabled genes, and — at the same rate —
    perturbs the surviving genes' parameters and the genome's risk geometry.
    Re-rolling genes alone only ever searches which genes to combine; the
    parameter jitter is what lets a promising combination be refined."""
    mutated = copy.deepcopy(genome)

    # Parameter-level mutation first, so a layer that is re-rolled below
    # simply overwrites it with a freshly sampled gene.
    mutated.signal = _jitter_choice(mutated.signal, SIGNAL_GENES, mutation_rate)
    mutated.bias = _jitter_choice(mutated.bias, BIAS_GENES, mutation_rate)
    mutated.execution = _jitter_choice(mutated.execution, EXECUTION_GENES, mutation_rate)
    mutated.management = _jitter_choice(mutated.management, MANAGEMENT_GENES, mutation_rate)
    mutated.filters = [_jitter_choice(f, FILTER_GENES, mutation_rate) for f in mutated.filters]

    if random.random() < mutation_rate:
        mutated.stop_atr_mult = round(
            min(max(mutated.stop_atr_mult + random.uniform(-0.5, 0.5), STOP_ATR_MULT_RANGE[0]), STOP_ATR_MULT_RANGE[1]), 2
        )
    if random.random() < mutation_rate:
        mutated.target_atr_mult = round(
            min(max(mutated.target_atr_mult + random.uniform(-1.0, 1.0), TARGET_ATR_MULT_RANGE[0]), TARGET_ATR_MULT_RANGE[1]), 2
        )
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
        # Drawn from the search space like every other re-rolled gene, not
        # pinned to the documented defaults.
        mutated.filters = [GeneChoice(n, sample_params(FILTER_GENES[n])) for n in names]
    return mutated
