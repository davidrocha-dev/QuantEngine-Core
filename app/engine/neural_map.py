"""Neural Map — REQ-MAP-1: every Vault strategy projected onto a 2D UMAP
space, density-clustered, and each cluster auto-labeled by its dominant
archetype. This is the "richer, population-aware alternative" referenced in
app/engine/scoring.py's classify_archetype docstring — that heuristic still
decides each *individual* strategy's archetype (used everywhere else in the
app); this module only decides which strategies group together visually and
what to call the group.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import HDBSCAN
from umap import UMAP

from app.genome.genes import BIAS_GENES, EXECUTION_GENES, FILTER_GENES, MANAGEMENT_GENES, SIGNAL_GENES
from app.genome.genome import Genome
from app.models import Strategy

# UMAP needs at least n_neighbors+1 samples to learn any neighborhood
# structure at all; below that a "map" is just noise pretending to be
# signal. design.md: "state plainly that clusters need a minimum
# population to be meaningful, rather than rendering a mostly-blank canvas."
MIN_STRATEGIES_FOR_MAP = 5

_SIGNAL_NAMES = sorted(SIGNAL_GENES)
_EXECUTION_NAMES = sorted(EXECUTION_GENES)
_MANAGEMENT_NAMES = sorted(MANAGEMENT_GENES)
_BIAS_NAMES = sorted(BIAS_GENES)
_FILTER_NAMES = sorted(FILTER_GENES)


@dataclass
class NeuralMapNode:
    id: str
    x: float
    y: float
    cluster_id: int
    archetype: str
    score: int | None
    cagr_pct: float | None
    max_dd_pct: float | None
    symbol: str
    timeframe: str
    brightness: float  # 0..1 — this node's score relative to its cluster's own range


@dataclass
class NeuralMapCluster:
    id: int
    archetype: str
    strategy_count: int
    best_id: str | None
    worst_id: str | None
    avg_return_pct: float | None
    avg_drawdown_pct: float | None
    profitable_pct: float
    top_symbol: str | None
    top_timeframe: str | None


@dataclass
class NeuralMapResult:
    nodes: list[NeuralMapNode] = field(default_factory=list)
    clusters: list[NeuralMapCluster] = field(default_factory=list)


def _one_hot(name: str | None, vocabulary: list[str]) -> list[float]:
    return [1.0 if name == candidate else 0.0 for candidate in vocabulary]


def _feature_vector(genome: Genome) -> list[float]:
    # Gene identity dominates the feature space on purpose — two strategies
    # built from the same Signal/Execution/Management genes are "the same
    # kind of strategy" almost by definition, which is what a similarity
    # map should group together. Filters are a small multi-hot addendum
    # (a genome can carry 0-2 of them, order doesn't matter).
    filters = {f.name for f in genome.filters}
    return [
        *_one_hot(genome.signal.name, _SIGNAL_NAMES),
        *_one_hot(genome.execution.name, _EXECUTION_NAMES),
        *_one_hot(genome.management.name if genome.management else None, _MANAGEMENT_NAMES),
        *_one_hot(genome.bias.name if genome.bias else None, _BIAS_NAMES),
        *[1.0 if name in filters else 0.0 for name in _FILTER_NAMES],
    ]


def _dominant_archetype(archetypes: list[str]) -> str:
    return Counter(archetypes).most_common(1)[0][0]


def _mode(values: list[str | None]) -> str | None:
    present = [v for v in values if v]
    if not present:
        return None
    return Counter(present).most_common(1)[0][0]


def compute_neural_map(strategies: list[Strategy]) -> NeuralMapResult | None:
    """Returns None when there isn't enough Vault population yet for a
    UMAP projection to mean anything (REQ-MAP-1.1) — the caller renders an
    explicit "not enough strategies yet" state instead of an empty canvas."""
    if len(strategies) < MIN_STRATEGIES_FOR_MAP:
        return None

    genomes: list[Genome] = []
    valid_strategies: list[Strategy] = []
    for row in strategies:
        try:
            genomes.append(Genome.from_dict(json.loads(row.genome_json)))
            valid_strategies.append(row)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    if len(valid_strategies) < MIN_STRATEGIES_FOR_MAP:
        return None

    features = np.array([_feature_vector(g) for g in genomes])
    n_neighbors = max(2, min(15, len(valid_strategies) - 1))
    coords = UMAP(n_neighbors=n_neighbors, min_dist=0.3, n_components=2, random_state=42).fit_transform(features)

    min_cluster_size = max(2, min(5, len(valid_strategies) // 3))
    labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(coords)
    # HDBSCAN's -1 ("noise") is one bucket, not "no cluster" — split each
    # noise point into its own singleton cluster so a click on any node
    # always resolves to a well-defined cluster popup (REQ-MAP-1.4/1.5).
    next_singleton_id = (labels.max() + 1) if len(labels) else 0
    resolved_labels = list(labels)
    for i, label in enumerate(resolved_labels):
        if label == -1:
            resolved_labels[i] = next_singleton_id
            next_singleton_id += 1

    by_cluster: dict[int, list[int]] = {}
    for idx, label in enumerate(resolved_labels):
        by_cluster.setdefault(int(label), []).append(idx)

    nodes: list[NeuralMapNode] = []
    clusters: list[NeuralMapCluster] = []
    for cluster_id, indices in by_cluster.items():
        members = [valid_strategies[i] for i in indices]
        scores = [m.score for m in members if m.score is not None]
        score_min, score_max = (min(scores), max(scores)) if scores else (0, 0)
        # A cluster whose members all share one score - and every cluster
        # of one, which is most of them until a Vault has grown - has no
        # range to be relative to. That is not a brightness of zero, it is
        # an undefined ratio, and reading it as zero rendered every node
        # at the dim end of its hue: a Neural Map that was simply blank.
        # A lone member is the best in its own cluster, so it reads as it.
        degenerate = score_max == score_min
        score_span = score_max - score_min or 1

        for i, member in zip(indices, members):
            score = member.score if member.score is not None else score_min
            nodes.append(
                NeuralMapNode(
                    id=member.id,
                    x=float(coords[i][0]),
                    y=float(coords[i][1]),
                    cluster_id=cluster_id,
                    archetype=member.archetype or "MULTI_SIGNAL",
                    score=member.score,
                    cagr_pct=member.cagr_pct,
                    max_dd_pct=member.max_dd_pct,
                    symbol=member.asset,
                    timeframe=member.timeframe,
                    brightness=1.0 if degenerate else (score - score_min) / score_span,
                )
            )

        returns = [m.return_pct for m in members if m.return_pct is not None]
        drawdowns = [m.max_dd_pct for m in members if m.max_dd_pct is not None]
        ranked = sorted(members, key=lambda m: m.score if m.score is not None else -1)
        clusters.append(
            NeuralMapCluster(
                id=cluster_id,
                archetype=_dominant_archetype([m.archetype or "MULTI_SIGNAL" for m in members]),
                strategy_count=len(members),
                best_id=ranked[-1].id if ranked else None,
                worst_id=ranked[0].id if ranked else None,
                avg_return_pct=sum(returns) / len(returns) if returns else None,
                avg_drawdown_pct=sum(drawdowns) / len(drawdowns) if drawdowns else None,
                # Profitable means it made money, not that it graded well.
                # This counted GOOD grades - a composite score of 57+,
                # which blends robustness, persistence, sample size and
                # drawdown - and showed the result under a label reading
                # "Profitable". /analytics computes the same-named field
                # from return_pct > 0 (main.py), so the app had two
                # different answers behind one word.
                profitable_pct=(
                    100.0 * sum(1 for m in members if (m.return_pct or 0) > 0) / len(members)
                ),
                top_symbol=_mode([m.asset for m in members]),
                top_timeframe=_mode([m.timeframe for m in members]),
            )
        )

    return NeuralMapResult(nodes=nodes, clusters=clusters)
