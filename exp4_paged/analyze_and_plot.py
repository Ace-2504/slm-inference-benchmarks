"""Experiment 4, part 2 — wasted KV memory, concurrency, and the block-size sweep.

Uses the REAL measured KV-per-token footprint and naive concurrency from
results/exp4_naive.json, and a realistic (heavy-tailed) sequence-length distribution, to
compare naive contiguous allocation (reserve max_len per sequence) against paged
allocation (fixed blocks, allocated as tokens are generated):

  * wasted KV = reserved-but-never-written, as a % — measured against the length
    distribution, not a fixed length.
  * concurrency both ways from the same KV budget.
  * block-size sweep: small blocks waste less but cost more bookkeeping (more blocks per
    sequence) — the curve.

Local, no GPU. Writes results/exp4.json + plots.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bench.request_gen import make_requests

ROOT = Path(__file__).resolve().parents[1]
NAIVE = json.loads((ROOT / "results" / "exp4_naive.json").read_text())
OUT = Path(__file__).resolve().parent
GPU = NAIVE["gpu_info"].get("name", "GPU")

KV_PER_TOKEN = NAIVE["kv_bytes_per_token"]
MAX_LEN = NAIVE["max_len"]
KV_BUDGET = NAIVE["kv_budget_bytes"]
BLOCK_SIZES = [1, 4, 8, 16, 32, 64, 128, 256]
C_NAIVE, C_PAGED = "#b2182b", "#2166ac"


def seq_lengths(n=5000, seed=7):
    reqs = make_requests(n, rate_per_s=50, seed=seed)
    return np.array([min(r.prompt_len + r.output_len, MAX_LEN) for r in reqs])


def analyze():
    lens = seq_lengths()
    mean_len = float(lens.mean())

    naive_bytes_per_seq = KV_PER_TOKEN * MAX_LEN
    naive_conc = KV_BUDGET // naive_bytes_per_seq
    naive_waste = 1.0 - mean_len / MAX_LEN  # reserved max_len, used mean_len

    sweep = []
    for blk in BLOCK_SIZES:
        alloc = np.ceil(lens / blk) * blk           # tokens actually reserved (paged)
        mean_alloc = float(alloc.mean())
        waste = float((alloc - lens).sum() / alloc.sum())  # internal fragmentation
        bytes_per_seq = KV_PER_TOKEN * mean_alloc
        conc = KV_BUDGET / bytes_per_seq
        blocks_per_seq = float(np.ceil(lens / blk).mean())  # bookkeeping cost proxy
        sweep.append({
            "block_size": blk, "mean_alloc_tokens": mean_alloc,
            "waste_frac": waste, "concurrency": conc, "blocks_per_seq": blocks_per_seq,
        })

    res = {
        "gpu": GPU, "model": NAIVE["model"], "max_len": MAX_LEN,
        "kv_bytes_per_token": KV_PER_TOKEN, "kv_budget_bytes": KV_BUDGET,
        "length_dist": {"mean": mean_len, "median": float(np.median(lens)),
                        "p95": float(np.percentile(lens, 95)), "max": int(lens.max())},
        "naive": {
            "bytes_per_seq": naive_bytes_per_seq,
            "concurrency": int(naive_conc),
            "measured_concurrency_oom": NAIVE["measured_concurrency_naive"],
            "waste_frac": naive_waste,
        },
        "paged_block_sweep": sweep,
    }
    (ROOT / "results" / "exp4.json").write_text(json.dumps(res, indent=2))
    return res


def plot(res):
    sweep = res["paged_block_sweep"]
    blks = [s["block_size"] for s in sweep]

    # 1) wasted memory: naive vs paged(block)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.axhline(res["naive"]["waste_frac"] * 100, color=C_NAIVE, ls="--", lw=2,
               label=f"naive (reserve max_len={MAX_LEN}): {res['naive']['waste_frac']*100:.1f}%")
    ax.plot(blks, [s["waste_frac"] * 100 for s in sweep], "o-", color=C_PAGED, lw=2,
            label="paged (block internal fragmentation)")
    ax.set_xscale("log", base=2); ax.set_xticks(blks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("block size (tokens)")
    ax.set_ylabel("wasted KV memory (%)")
    ax.set_title(f"Exp 4 — wasted KV: naive vs paged ({GPU}, {res['model'].split('/')[-1]})")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "wasted_memory.png", dpi=130); plt.close(fig)

    # 2) concurrency: naive vs paged(block)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.axhline(res["naive"]["concurrency"], color=C_NAIVE, ls="--", lw=2,
               label=f"naive: {res['naive']['concurrency']} seqs")
    ax.plot(blks, [s["concurrency"] for s in sweep], "o-", color=C_PAGED, lw=2,
            label="paged")
    ax.set_xscale("log", base=2); ax.set_xticks(blks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("block size (tokens)")
    ax.set_ylabel("concurrent sequences that fit")
    ax.set_title(f"Exp 4 — concurrency from the same KV budget ({GPU})")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "concurrency.png", dpi=130); plt.close(fig)

    # 3) block-size sweep: waste vs bookkeeping (blocks/seq)
    fig, ax1 = plt.subplots(figsize=(7, 4.4))
    ax1.plot(blks, [s["waste_frac"] * 100 for s in sweep], "o-", color=C_PAGED, lw=2)
    ax1.set_xscale("log", base=2); ax1.set_xticks(blks)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel("block size (tokens)")
    ax1.set_ylabel("wasted KV memory (%)", color=C_PAGED)
    ax1.tick_params(axis="y", labelcolor=C_PAGED)
    ax2 = ax1.twinx()
    ax2.plot(blks, [s["blocks_per_seq"] for s in sweep], "s--", color="#666", lw=1.6)
    ax2.set_ylabel("blocks per sequence (bookkeeping)", color="#666")
    ax2.tick_params(axis="y", labelcolor="#666")
    ax1.set_title(f"Exp 4 — block-size trade-off: waste vs bookkeeping ({GPU})")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "block_size_sweep.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    import matplotlib.ticker  # noqa
    res = analyze()
    print(f"=== Exp 4 (paged) — {res['model']} on {GPU} ===")
    print(f"KV = {KV_PER_TOKEN/1024:.1f} KB/token; budget {KV_BUDGET/1024**3:.2f} GB; "
          f"seq len mean={res['length_dist']['mean']:.0f} p95={res['length_dist']['p95']:.0f} "
          f"(max_len={MAX_LEN})")
    print(f"naive: {res['naive']['concurrency']} seqs (OOM-confirmed "
          f"{res['naive']['measured_concurrency_oom']}), waste {res['naive']['waste_frac']*100:.1f}%")
    for s in res["paged_block_sweep"]:
        print(f"  paged block={s['block_size']:>3}: {s['concurrency']:>6.0f} seqs, "
              f"waste {s['waste_frac']*100:>5.1f}%, {s['blocks_per_seq']:.1f} blocks/seq")
    plot(res)
    print("wrote results/exp4.json + plots")
