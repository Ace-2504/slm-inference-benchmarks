"""Plot Experiment 1 from results/exp1.json (run locally; no GPU needed).

Produces, all axis-labelled with units:
  per_step_time.png       — per-step decode time vs step index, cached vs uncached
                            (batch 1 and the largest batch, side by side)
  speedup_vs_batch.png    — cache speedup (cached tok/s / uncached tok/s) vs batch size
  throughput_vs_batch.png — cached & uncached tok/s vs batch size
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "results" / "exp1.json").read_text())
OUT = Path(__file__).resolve().parent
GPU = RES["gpu_info"].get("name", "GPU")
BATCHES = [int(b) for b in RES["batch_sizes"]]

C_CACHED, C_UNCACHED = "#2166ac", "#b2182b"  # CVD-safe blue / red


def _per_step():
    show = [BATCHES[0], BATCHES[-1]]
    fig, axes = plt.subplots(1, len(show), figsize=(11, 4.2), sharey=False)
    if len(show) == 1:
        axes = [axes]
    for ax, bs in zip(axes, show):
        r = RES["per_batch"][str(bs)]
        c, u = r["cached_per_step_ms"], r["uncached_per_step_ms"]
        ax.plot(range(1, len(c) + 1), c, color=C_CACHED, label="cached", lw=1.8)
        ax.plot(range(1, len(u) + 1), u, color=C_UNCACHED, label="uncached", lw=1.8)
        ax.set_title(f"batch = {bs}")
        ax.set_xlabel("decode step index")
        ax.set_ylabel("per-step time (ms)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"Exp 1 — per-step decode time, cached vs uncached ({GPU})")
    fig.tight_layout()
    fig.savefig(OUT / "per_step_time.png", dpi=130)
    plt.close(fig)


def _speedup():
    sp = [RES["per_batch"][str(b)]["cache_speedup"] for b in BATCHES]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(BATCHES, sp, "o-", color=C_CACHED, lw=2)
    for x, y in zip(BATCHES, sp):
        ax.annotate(f"{y:.1f}x", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="parity (1x)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCHES)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("batch size")
    ax.set_ylabel("cache speedup  (cached tok/s ÷ uncached tok/s)")
    ax.set_title(f"Exp 1 — what the KV cache is worth vs batch size ({GPU})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "speedup_vs_batch.png", dpi=130)
    plt.close(fig)


def _throughput():
    c = [RES["per_batch"][str(b)]["cached_tok_s"] for b in BATCHES]
    u = [RES["per_batch"][str(b)]["uncached_tok_s"] for b in BATCHES]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(BATCHES, c, "o-", color=C_CACHED, lw=2, label="cached")
    ax.plot(BATCHES, u, "s-", color=C_UNCACHED, lw=2, label="uncached")
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCHES)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("batch size")
    ax.set_ylabel("throughput (tokens / s)")
    ax.set_title(f"Exp 1 — throughput vs batch size ({GPU})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "throughput_vs_batch.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import matplotlib.ticker  # noqa: F401  (used via matplotlib.ticker above)
    _per_step()
    _speedup()
    _throughput()
    print("wrote per_step_time.png, speedup_vs_batch.png, throughput_vs_batch.png to", OUT)
