"""Per-campaign Gene Pool — REQ-ENGINE-1.3 / REQ-SET-2.

Settings → Gene Pool toggles each of the 46 genes on/off per campaign. This
module holds that enabled/disabled state and exposes the enabled subset per
layer to the random genome generator (genome.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.genome.genes import ALL_GENES, Layer


@dataclass
class GenePoolConfig:
    # Keyed by (layer, name) — see genes.ALL_GENES for why a bare name isn't unique.
    enabled: dict[tuple[Layer, str], bool] = field(
        default_factory=lambda: {key: True for key in ALL_GENES}
    )

    def is_enabled(self, layer: Layer, name: str) -> bool:
        return self.enabled.get((layer, name), False)

    def set_enabled(self, layer: Layer, name: str, value: bool) -> None:
        if (layer, name) not in ALL_GENES:
            raise ValueError(f"Unknown gene: {layer}/{name}")
        self.enabled[(layer, name)] = value

    def enabled_names(self, layer: Layer) -> list[str]:
        return [name for (gene_layer, name), on in self.enabled.items() if gene_layer == layer and on]

    def layer_counts(self) -> dict[Layer, dict[str, int]]:
        """Per-layer enabled/total counts, for the Settings → Gene Pool UI (REQ-SET-2.1)."""
        counts: dict[Layer, dict[str, int]] = {}
        for (layer, _name), on in self.enabled.items():
            bucket = counts.setdefault(layer, {"enabled": 0, "total": 0})
            bucket["total"] += 1
            if on:
                bucket["enabled"] += 1
        return counts

    def all_management_disabled(self) -> bool:
        """REQ-ENGINE-1.4: Phase 3 must fail when every Management gene is off."""
        return len(self.enabled_names("management")) == 0

    def to_json(self) -> str:
        return json.dumps({f"{layer}::{name}": on for (layer, name), on in self.enabled.items()})

    @classmethod
    def from_json(cls, raw: str) -> "GenePoolConfig":
        data = json.loads(raw)
        enabled = {}
        for key, on in data.items():
            layer, name = key.split("::", 1)
            enabled[(layer, name)] = on
        config = cls(enabled={})
        # Start from all-enabled defaults, then apply the saved overrides —
        # keeps forward compatibility if new genes are added later.
        config.enabled = {k: True for k in ALL_GENES}
        config.enabled.update(enabled)
        return config

    def save(self, path: Path) -> None:
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> "GenePoolConfig":
        if not path.exists():
            return cls()
        return cls.from_json(path.read_text())
