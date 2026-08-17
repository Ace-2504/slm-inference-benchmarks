"""Spread statistics for timing samples.

Every timed configuration in this project reports a *spread*, never a single number
(brief: "run each configuration at least three times and report the spread").
"""
from __future__ import annotations

import math
from typing import Sequence


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 100]. Matches numpy's default."""
    if not sorted_vals:
        raise ValueError("percentile of empty sequence")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[int(pos)]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize(values: Sequence[float]) -> dict:
    """Return mean/std(sample)/min/median/max/p95/cv/n for a list of timings.

    cv = coefficient of variation = std / mean (dimensionless); the reproducibility
    knob the gauntlet's G7 checks.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        raise ValueError("summarize of empty sequence")
    mean = sum(vals) / n
    # sample standard deviation (n-1); 0 for a single sample
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    s = sorted(vals)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": s[0],
        "median": _percentile(s, 50),
        "max": s[-1],
        "p95": _percentile(s, 95),
        "cv": (std / mean) if mean else float("nan"),
        "raw": vals,
    }
