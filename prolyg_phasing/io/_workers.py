"""The worker-process count for the per-locus process pools, from the live box."""

from __future__ import annotations

import math
import os

# Memory one worker process can use at a deep locus, with headroom.
_WORKER_MEMORY_BYTES = 1 << 30

# Share of the available memory that the workers may use together.
_WORKER_MEMORY_SHARE = 0.5

# Share of the usable cores that is left free for other work.
_FREE_CORE_SHARE = 0.125


def worker_count() -> int:
    """Worker processes for a per-locus pool (flank search, extraction).

    The usable cores (``os.sched_getaffinity``) less `_FREE_CORE_SHARE` of
    them, and no more than `_WORKER_MEMORY_SHARE` of ``MemAvailable`` in
    `_WORKER_MEMORY_BYTES` units. At least one.
    """
    cores = len(os.sched_getaffinity(0))
    by_cores = cores - max(1, math.ceil(cores * _FREE_CORE_SHARE))
    with open("/proc/meminfo") as f:
        available_kb = next(int(line.split()[1]) for line in f
                            if line.startswith("MemAvailable:"))
    by_memory = int(available_kb * 1024 * _WORKER_MEMORY_SHARE // _WORKER_MEMORY_BYTES)
    return max(1, min(by_cores, by_memory))
