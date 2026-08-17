"""Experiment 3, part 2 — run BOTH batching policies over the same request stream using
the measured step times, at 3 arrival rates x 2 batch sizes, then plot. Local, no GPU.

Writes results/exp3.json and:
  utilisation_over_time.png  — slot utilisation, static vs continuous (the headline plot)
  queue_depth_over_time.png  — waiting requests over time
  throughput_vs_rate.png     — completed req/s, static vs continuous, per batch size
  latency_vs_rate.png        — median & p95 latency, static vs continuous
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.request_gen import make_requests, length_stats
from bench.batching_sim import simulate_static, simulate_continuous

ROOT = Path(__file__).resolve().parents[1]
ST = json.loads((ROOT / "results" / "exp3_steptimes.json").read_text())
OUT = Path(__file__).resolve().parent
GPU = ST["gpu_info"].get("name", "GPU")

WIDTHS = [int(b) for b in ST["batch_widths"]]
STEP_MS = {int(k): v for k, v in ST["step_ms"].items()}

N_REQ = 200
ARRIVAL_RATES = [20.0, 40.0, 80.0]   # requests / s
BATCH_SIZES = [8, 32]
C_STATIC, C_CONT = "#b2182b", "#2166ac"


def step_time_s(active):
    """Linear interpolation of measured step time (ms) -> seconds, for any active count."""
    active = max(1, int(round(active)))
    xs = WIDTHS
    if active <= xs[0]:
        ms = STEP_MS[xs[0]]
    elif active >= xs[-1]:
        # extrapolate from the last segment
        x0, x1 = xs[-2], xs[-1]
        ms = STEP_MS[x1] + (STEP_MS[x1] - STEP_MS[x0]) / (x1 - x0) * (active - x1)
    else:
        i = next(j for j in range(len(xs) - 1) if xs[j] <= active <= xs[j + 1])
        x0, x1 = xs[i], xs[i + 1]
        f = (active - x0) / (x1 - x0)
        ms = STEP_MS[x0] + f * (STEP_MS[x1] - STEP_MS[x0])
    return ms / 1e3


def run_all():
    results = {"gpu": GPU, "model": ST["model"], "n_requests": N_REQ,
               "arrival_rates": ARRIVAL_RATES, "batch_sizes": BATCH_SIZES, "runs": {}}
    for B in BATCH_SIZES:
        for rate in ARRIVAL_RATES:
            reqs = make_requests(N_REQ, rate, seed=42)
            st = simulate_static(reqs, B, step_time_s)
            co = simulate_continuous(reqs, B, step_time_s)
            key = f"B{B}_r{int(rate)}"
            results["runs"][key] = {
                "batch_size": B, "arrival_rate": rate,
                "length_stats": length_stats(reqs),
                "static": _metrics(st), "continuous": _metrics(co),
            }
    (ROOT / "results" / "exp3.json").write_text(json.dumps(results, indent=2))
    return results


def _metrics(r):
    return {
        "throughput_req_s": r.throughput_req_s, "throughput_tok_s": r.throughput_tok_s,
        "latency_median_s": r.latency_median_s, "latency_p95_s": r.latency_p95_s,
        "ttft_median_s": r.ttft_median_s, "makespan_s": r.makespan_s,
        "util_series": r.util_series, "queue_series": r.queue_series,
    }


def plot_timeseries(results, B, rate):
    key = f"B{B}_r{int(rate)}"
    run = results["runs"][key]
    for metric, fname, ylabel, title in [
        ("util_series", "utilisation_over_time.png", "slot utilisation (active / B)",
         f"Exp 3 — slot utilisation over time (B={B}, {rate:.0f} req/s, {GPU})"),
        ("queue_series", "queue_depth_over_time.png", "queue depth (waiting requests)",
         f"Exp 3 — queue depth over time (B={B}, {rate:.0f} req/s, {GPU})"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for pol, color in [("static", C_STATIC), ("continuous", C_CONT)]:
            series = run[pol][metric]
            if series:
                xs, ys = zip(*series)
                ax.plot(xs, ys, color=color, lw=1.4, label=pol, alpha=0.85)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=130)
        plt.close(fig)


def plot_vs_rate(results, B):
    rates = ARRIVAL_RATES
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for pol, color in [("static", C_STATIC), ("continuous", C_CONT)]:
        ys = [results["runs"][f"B{B}_r{int(r)}"][pol]["throughput_req_s"] for r in rates]
        ax.plot(rates, ys, "o-", color=color, lw=2, label=pol)
    ax.set_xlabel("arrival rate (requests / s)")
    ax.set_ylabel("throughput (completed requests / s)")
    ax.set_title(f"Exp 3 — throughput vs arrival rate (B={B}, {GPU})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "throughput_vs_rate.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for pol, color, ls in [("static", C_STATIC, "-"), ("continuous", C_CONT, "-")]:
        med = [results["runs"][f"B{B}_r{int(r)}"][pol]["latency_median_s"] for r in rates]
        p95 = [results["runs"][f"B{B}_r{int(r)}"][pol]["latency_p95_s"] for r in rates]
        ax.plot(rates, med, "o-", color=color, lw=2, label=f"{pol} median")
        ax.plot(rates, p95, "o--", color=color, lw=1.5, alpha=0.7, label=f"{pol} p95")
    ax.set_xlabel("arrival rate (requests / s)")
    ax.set_ylabel("latency (s)")
    ax.set_title(f"Exp 3 — latency vs arrival rate (B={B}, {GPU})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "latency_vs_rate.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    res = run_all()
    plot_timeseries(res, B=32, rate=80.0)
    plot_vs_rate(res, B=32)
    print("=== Exp 3 (batching) ===")
    for B in BATCH_SIZES:
        for rate in ARRIVAL_RATES:
            r = res["runs"][f"B{B}_r{int(rate)}"]
            s, c = r["static"], r["continuous"]
            print(f"B={B:>2} rate={rate:>4.0f}/s | static {s['throughput_req_s']:6.1f} req/s "
                  f"p95={s['latency_p95_s']:6.2f}s || continuous {c['throughput_req_s']:6.1f} req/s "
                  f"p95={c['latency_p95_s']:6.2f}s")
    print("wrote results/exp3.json + plots")
