"""Cooperative pause and skip for a running campaign — REQ-BREED-1.2.

The PAUSE and SKIP buttons existed and did nothing: the endpoints behind
them answered "not supported" while the campaign carried on. An honest
refusal is better than a silent no-op, but a button that cannot do its
job is worse than either.

The reason it was hard is that a campaign is a genetic search running
across a process pool, and stopping one mid-generation would throw away
the generation's work and leave the checkpoint describing a population
that no longer exists. So this does not interrupt anything. It raises a
flag, and the phase loop reads it at the one moment where stopping is
free: immediately after a generation has been evaluated and checkpointed.

A pause therefore costs at most one generation of waiting, and resumes
from a checkpoint that was always going to be written anyway.
"""

from __future__ import annotations

import threading


class CampaignPaused(Exception):
    """Raised inside the phase loop when a pause has been requested.

    An exception rather than a return value because it has to unwind
    through several layers of phase runners, and every one of them would
    otherwise need to understand and forward a "stopped early" result.
    """


class CampaignControl:
    """Thread-safe flags set by the HTTP layer, read by the engine.

    The campaign runs on its own thread; the request that pauses it
    arrives on another. Both flags are one-shot: they describe a request
    that has been made, and reading one consumes it, so a pause cannot
    linger and stop the next campaign too.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pause = False
        self._skip = False

    def request_pause(self) -> None:
        with self._lock:
            self._pause = True

    def request_skip(self) -> None:
        with self._lock:
            self._skip = True

    def clear(self) -> None:
        with self._lock:
            self._pause = self._skip = False

    @property
    def pause_requested(self) -> bool:
        with self._lock:
            return self._pause

    def take_skip(self) -> bool:
        """Consumes a skip request. Skipping applies to the current phase
        only: a single click must not skip every phase that follows."""
        with self._lock:
            requested, self._skip = self._skip, False
            return requested

    def checkpoint(self) -> None:
        """Called by the phase loop at a generation boundary.

        Raises `CampaignPaused` if a pause is pending. The pause flag is
        not consumed here - `run_campaign` clears it once the campaign has
        actually stopped, so a pause that arrives during teardown cannot
        be lost.
        """
        if self.pause_requested:
            raise CampaignPaused()
