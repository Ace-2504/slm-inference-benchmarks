"""Plot Experiment 2 from results/exp2.json (local; no GPU).

  speedup_vs_k.png       — measured speedup vs k, one line per prompt type, with the
                           predicted-speedup curve (from measured c) dashed for overlay
  acceptance_vs_k.png    — acceptance rate vs k, one line per prompt type
  tokens_per_pass_vs_k.png — tokens committed per target forward pass vs k
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "results" / "exp2.json").read_text())
OUT = Path(__file__).resolve().parent
GPU = RES["gpu_info"].get("name", "GPU")
KS = RES["k_values"]
C = RES["diagnosis"]["c"]
COLORS = {"copy": "#1b9e77", "code": "#d95f02", "explain": "#7570b3", "creative": "#e7298a"}


def _line(metric, ylabel, fname, title, add_predicted=False, hline=None):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for name, pr in RES["per_prompt"].items():
        ys = [pr["by_k"][str(k)][metric] for k in KS]
        ax.plot(KS, ys, "o-", color=COLORS.get(name, "gray"), lw=2, label=name)
        if add_predicted:
            yp = [pr["by_k"][str(k)]["predicted_speedup"] for k in KS]
            ax.plot(KS, yp, "--", color=COLORS.get(name, "gray"), lw=1, alpha=0.6)
    if hline is not None:
        ax.axhline(hline, color="gray", ls="--", lw=1, label=f"parity ({hline}x)")
    ax.set_xlabel("k  (draft tokens proposed per round)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(KS)
    ax.grid(True, alpha=0.3)
    if add_predicted:
        ax.plot([], [], "k--", alpha=0.6, label="predicted (from c)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    _line("speedup", "speedup  (spec tok/s ÷ target-alone tok/s)", "speedup_vs_k.png",
          f"Exp 2 — speculative speedup vs k  (c={C:.2f}, {GPU})",
          add_predicted=True, hline=1.0)
    _line("acceptance_rate", "acceptance rate  (drafted tokens kept)", "acceptance_vs_k.png",
          f"Exp 2 — acceptance rate vs k  ({GPU})")
    _line("tokens_per_target_pass", "tokens committed / target forward pass",
          "tokens_per_pass_vs_k.png", f"Exp 2 — tokens per target pass vs k  ({GPU})")
    print("wrote speedup_vs_k.png, acceptance_vs_k.png, tokens_per_pass_vs_k.png to", OUT)
