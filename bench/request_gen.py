"""Request generator for the serving experiments (3 & 4).

Poisson arrivals at a chosen rate, with output lengths drawn from a heavy-tailed
distribution — because uniform output lengths hide the entire continuous-batching effect
(the brief). A few long requests among many short ones is exactly what makes static
batching stall with idle slots.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Request:
    id: int
    arrival_s: float      # when it arrives at the server
    prompt_len: int       # prompt tokens (for TTFT / prefill accounting)
    output_len: int       # tokens it will generate before finishing


def make_requests(n: int, rate_per_s: float, seed: int = 0,
                  out_len_median: int = 64, out_len_sigma: float = 0.9,
                  prompt_len_range: tuple[int, int] = (16, 128)) -> list[Request]:
    """n requests, Poisson arrivals at rate_per_s, lognormal output lengths.

    lognormal(median, sigma) gives a right-skewed spread: most requests short, a few very
    long. out_len_sigma ~0.9 => heavy tail (the long-request stragglers static batching
    waits on). Returns requests sorted by arrival time.
    """
    rng = np.random.default_rng(seed)
    # Poisson process: inter-arrival ~ Exponential(1/rate)
    gaps = rng.exponential(1.0 / rate_per_s, size=n)
    arrivals = np.cumsum(gaps)
    out_lens = rng.lognormal(mean=np.log(out_len_median), sigma=out_len_sigma, size=n)
    out_lens = np.clip(out_lens.round().astype(int), 4, 2048)
    prompt_lens = rng.integers(prompt_len_range[0], prompt_len_range[1] + 1, size=n)
    return [
        Request(id=i, arrival_s=float(arrivals[i]),
                prompt_len=int(prompt_lens[i]), output_len=int(out_lens[i]))
        for i in range(n)
    ]


def length_stats(reqs: list[Request]) -> dict:
    lens = np.array([r.output_len for r in reqs])
    return {
        "n": len(reqs), "min": int(lens.min()), "median": int(np.median(lens)),
        "p95": int(np.percentile(lens, 95)), "max": int(lens.max()),
        "mean": float(lens.mean()), "cv": float(lens.std() / lens.mean()),
    }
