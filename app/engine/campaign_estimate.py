"""CPU-based campaign duration estimate — REQ-DASH-1.5.

Pure function so the Dashboard's "before you confirm launch" estimate can be
unit-tested without spinning up a real campaign.
"""

from __future__ import annotations

MIN_MINUTES = 15
MAX_MINUTES = 60

# Cores at/below this saturate the estimate at MAX_MINUTES; at/above this it
# saturates at MIN_MINUTES. Linear interpolation between the two.
_SLOW_CORES = 2
_FAST_CORES = 16


def estimate_campaign_minutes(cpu_cores: int) -> tuple[int, int]:
    """Return an inclusive (min, max) minute range for the given core count.

    More cores → shorter estimate, clamped to the [15, 60] band from
    REQ-DASH-1.5. The range narrows as cores increase since more parallelism
    also reduces variance between runs.
    """
    cores = max(1, cpu_cores)
    if cores <= _SLOW_CORES:
        return (MAX_MINUTES // 2, MAX_MINUTES)
    if cores >= _FAST_CORES:
        return (MIN_MINUTES, MIN_MINUTES * 2)

    span = _FAST_CORES - _SLOW_CORES
    progress = (cores - _SLOW_CORES) / span
    high = round(MAX_MINUTES - progress * (MAX_MINUTES - MIN_MINUTES * 2))
    low = round((MAX_MINUTES // 2) - progress * (MAX_MINUTES // 2 - MIN_MINUTES))
    return (low, high)
