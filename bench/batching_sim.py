"""Discrete-event simulator for Experiment 3 (static vs continuous batching), driven by
REAL measured decode-step times step_time(active) so the only thing that differs between
the two policies is the SCHEDULING — which is exactly the variable the experiment isolates.

We implement BOTH sides ourselves (the brief calls the naive static side mandatory and the
continuous side "the better exercise"). The engine steps token-by-token; each engine
iteration advances wall-clock by the measured cost of one decode step at the current batch
width. Prefill compute is approximated as one decode step per request at admission (decode
dominates); this is stated in the report.

  static:     take up to B arrived requests, run them as one fixed-width-B batch, and admit
              NOBODY until every sequence in the batch has finished. Short requests sit
              completed in their slot while the longest one runs on.
  continuous: keep up to B slots full — the instant a sequence finishes, evict it and admit
              a waiting request into that slot at the next step.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimResult:
    policy: str
    batch_size: int
    makespan_s: float
    throughput_req_s: float
    throughput_tok_s: float
    latency_median_s: float
    latency_p95_s: float
    ttft_median_s: float
    util_series: list = field(default_factory=list)   # (time_s, active/B)
    queue_series: list = field(default_factory=list)  # (time_s, waiting)
    latencies: list = field(default_factory=list)
    ttfts: list = field(default_factory=list)


def _pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    pos = (q / 100) * (len(s) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _summarize(policy, B, total_tokens, completions, ttfts, latencies, util, queue):
    makespan = max((c for c in completions), default=0.0)
    n = len(latencies)
    return SimResult(
        policy=policy, batch_size=B, makespan_s=makespan,
        throughput_req_s=n / makespan if makespan else 0.0,
        throughput_tok_s=total_tokens / makespan if makespan else 0.0,
        latency_median_s=_pct(latencies, 50), latency_p95_s=_pct(latencies, 95),
        ttft_median_s=_pct(ttfts, 50),
        util_series=util, queue_series=queue, latencies=latencies, ttfts=ttfts,
    )


def simulate_continuous(reqs, B, step_time_fn):
    reqs = sorted(reqs, key=lambda r: r.arrival_s)
    i = 0                       # next request to arrive
    active = []                 # list of dicts: {req, remaining, admit_t, first_t}
    t = 0.0
    completions, ttfts, latencies = [], [], []
    util, queue = [], []
    done = 0
    N = len(reqs)

    while done < N:
        # admit arrived requests into free slots
        while i < N and reqs[i].arrival_s <= t and len(active) < B:
            active.append({"req": reqs[i], "remaining": reqs[i].output_len,
                           "admit_t": t, "first_t": None})
            i += 1
        if not active:
            # jump to next arrival
            if i < N:
                t = max(t, reqs[i].arrival_s)
                continue
            break

        dt = step_time_fn(len(active))
        t += dt
        waiting = sum(1 for j in range(i, N) if reqs[j].arrival_s <= t)
        util.append((t, len(active) / B))
        queue.append((t, waiting))

        finished = []
        for slot in active:
            if slot["first_t"] is None:
                slot["first_t"] = t
                ttfts.append(t - slot["req"].arrival_s)
            slot["remaining"] -= 1
            if slot["remaining"] <= 0:
                completions.append(t)
                latencies.append(t - slot["req"].arrival_s)
                finished.append(slot)
        for slot in finished:
            active.remove(slot)
            done += 1

    total_tokens = sum(r.output_len for r in reqs)
    return _summarize("continuous", B, total_tokens, completions, ttfts, latencies, util, queue)


def simulate_static(reqs, B, step_time_fn):
    reqs = sorted(reqs, key=lambda r: r.arrival_s)
    i = 0
    t = 0.0
    completions, ttfts, latencies = [], [], []
    util, queue = [], []
    N = len(reqs)

    while i < N:
        # wait until at least one request is available; take up to B arrived
        if reqs[i].arrival_s > t:
            t = reqs[i].arrival_s
        batch = []
        while i < N and reqs[i].arrival_s <= t and len(batch) < B:
            batch.append(reqs[i])
            i += 1
        # run this batch to completion at FIXED width B; no admission until all finish
        max_len = max(r.output_len for r in batch)
        for r in batch:
            ttfts.append((t + step_time_fn(B)) - r.arrival_s)  # first token after 1 step
        for s in range(1, max_len + 1):
            t += step_time_fn(B)
            active = sum(1 for r in batch if r.output_len >= s)
            waiting = sum(1 for j in range(i, N) if reqs[j].arrival_s <= t)
            util.append((t, active / B))
            queue.append((t, waiting))
            for r in batch:
                if r.output_len == s:
                    completions.append(t)
                    latencies.append(t - r.arrival_s)

    total_tokens = sum(r.output_len for r in reqs)
    return _summarize("static", B, total_tokens, completions, ttfts, latencies, util, queue)
