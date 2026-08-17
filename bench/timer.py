"""The trustworthy timer.

Every timed measurement in this project goes through here so the four disciplines the
brief demands are applied uniformly and cannot be forgotten:

  1. torch.cuda.synchronize() on both sides of anything timed (CUDA is async).
  2. a warm-up pass before timing (don't time autotune / allocation / compile).
  3. >= 3 repeats, spread reported (see bench.stats.summarize).
  4. prefill timed separately from decode (different bottlenecks) — callers time the
     two phases with separate benchmark() calls.

Two independent clocks are exposed and cross-checked in the gauntlet (G4):
  - time_event: torch.cuda.Event, device-side, the primary clock for GPU work.
  - time_wall:  perf_counter with a trailing sync, host-side.
"""
from __future__ import annotations

import time
from typing import Callable

import torch

from .stats import summarize


def sync() -> None:
    """Block until all queued CUDA work on the current device has completed."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_event(fn: Callable[[], object]) -> float:
    """Time one call with CUDA events. Returns milliseconds (device-side)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    sync()
    start.record()
    fn()
    end.record()
    sync()
    return start.elapsed_time(end)  # ms


def time_wall(fn: Callable[[], object], do_sync: bool = True) -> float:
    """Time one call with perf_counter. Returns milliseconds (host-side).

    do_sync=False deliberately omits the trailing synchronize — used ONLY by the
    gauntlet's G2 to demonstrate the async-launch lie. Real measurements keep it True.
    """
    sync()  # ensure we start from a quiescent device
    t0 = time.perf_counter()
    fn()
    if do_sync:
        sync()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e3  # ms


def benchmark(
    fn: Callable[[], object],
    warmup: int = 2,
    repeats: int = 5,
    clock: str = "event",
) -> dict:
    """Warm up, then time `fn` `repeats` times and return a spread summary.

    clock: "event" (CUDA events, default) or "wall" (synced perf_counter).
    Returns the dict from bench.stats.summarize, in milliseconds, plus metadata.
    """
    if clock not in ("event", "wall"):
        raise ValueError("clock must be 'event' or 'wall'")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    for _ in range(max(0, warmup)):
        fn()
    sync()

    tick = time_event if clock == "event" else time_wall
    samples = [tick(fn) for _ in range(repeats)]

    out = summarize(samples)
    out["unit"] = "ms"
    out["clock"] = clock
    out["warmup"] = warmup
    return out
