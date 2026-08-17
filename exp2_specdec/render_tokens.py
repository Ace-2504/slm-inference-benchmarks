"""Render Exp-2's per-token producer tags as a coloured HTML page (one block per prompt
type, at k=TAG_K). Colours: draft-accepted = green, target-correction = red,
bonus (free token) = blue. This is the brief's "output coloured token by token, showing
which tokens came from the draft and were accepted, and which the target produced."
"""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "results" / "exp2.json").read_text())
OUT = Path(__file__).resolve().parent / "colored_tokens.html"

TAG_COLOR = {"draft": "#1b7837", "target": "#b2182b", "bonus": "#2166ac"}
TAG_LABEL = {"draft": "draft-accepted", "target": "target-correction", "bonus": "bonus (free)"}


def block(name, entry):
    toks = entry.get("tagged_tokens", [])
    spans = []
    for t in toks:
        color = TAG_COLOR.get(t["tag"], "#333")
        txt = html.escape(t["tok"]).replace("\n", "⏎<br>")
        spans.append(f'<span style="background:{color}22;border-bottom:2px solid {color};'
                     f'padding:1px 2px;border-radius:3px" title="{t["tag"]}">{txt}</span>')
    acc = entry["acceptance_rate"] * 100
    return (f'<h3>{html.escape(name)} '
            f'<small style="font-weight:normal;color:#666">— k={entry["k"]}, '
            f'acceptance {acc:.0f}%, speedup {entry["speedup"]:.2f}x, '
            f'exact={entry["match"]}</small></h3>'
            f'<p style="line-height:2;font-family:ui-monospace,monospace;font-size:14px">'
            f'{"".join(spans)}</p>')


def main():
    legend = " &nbsp; ".join(
        f'<span style="border-bottom:3px solid {c}">{TAG_LABEL[t]}</span>'
        for t, c in TAG_COLOR.items())
    blocks = []
    for name, pr in RES["per_prompt"].items():
        e = pr["by_k"].get(str(4)) or next(iter(pr["by_k"].values()))
        if "tagged_tokens" in e:
            blocks.append(block(name, e))
    doc = (f'<!doctype html><meta charset="utf-8"><title>Exp 2 — coloured tokens</title>'
           f'<body style="max-width:820px;margin:2rem auto;font-family:system-ui;color:#222">'
           f'<h2>Experiment 2 — who produced each token</h2>'
           f'<p style="color:#555">Legend: {legend}</p>{"".join(blocks)}</body>')
    OUT.write_text(doc, encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
