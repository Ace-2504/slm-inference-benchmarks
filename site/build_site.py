"""Extract the real measured results into site/data.js so the demo page is rich and
honest even when the live GPU endpoint is offline (sliders explore the recorded sweep;
live buttons light up only if an endpoint is reachable). Run from repo root:
    python site/build_site.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = lambda n: json.loads((ROOT / "results" / n).read_text())

e1 = R("exp1.json")
e2 = R("exp2.json")
e3 = R("exp3.json")
e4 = R("exp4.json")

gpu = e1["gpu_info"].get("name", "GPU")

# ---- Exp 1 ----
batches = []
per_step = {}
for bs in e1["batch_sizes"]:
    r = e1["per_batch"][str(bs)]
    batches.append({
        "bs": bs, "cached_tps": round(r["cached_tok_s"], 1),
        "uncached_tps": round(r["uncached_tok_s"], 1),
        "speedup": round(r["cache_speedup"], 2), "match": r["match"],
    })
    per_step[str(bs)] = {
        "cached": [round(x, 3) for x in r["cached_per_step_ms"]],
        "uncached": [round(x, 3) for x in r["uncached_per_step_ms"]],
    }

exp1 = {
    "model": e1["model"], "prompt_len": e1["prompt_len"], "gen": e1["max_new_tokens"],
    "tp": {"cached": e1["token_positions"]["cached_total"],
           "uncached": e1["token_positions"]["uncached_total"],
           "ratio": round(e1["token_positions"]["ratio_uncached_over_cached"], 1)},
    "batches": batches, "per_step": per_step,
    "sample": {"cached": e2 and e1["sample_text"]["cached"],
               "uncached": e1["sample_text"]["uncached"],
               "match": e1["sample_text"]["match"],
               "prompt_tail": e1["sample_text"]["prompt_tail"]},
}

# ---- Exp 2 ----
d = e2["diagnosis"]
prompts = {}
for name, pr in e2["per_prompt"].items():
    by_k = []
    tagged = None
    text = None
    for k in e2["k_values"]:
        ent = pr["by_k"][str(k)]
        by_k.append({
            "k": k, "accept": round(ent["acceptance_rate"], 3),
            "tok_per_pass": round(ent["tokens_per_target_pass"], 2),
            "spec_tps": round(ent["spec_tok_s"], 1),
            "speedup": round(ent["speedup"], 2),
            "pred": round(ent["predicted_speedup"], 2), "match": ent["match"],
        })
        if "tagged_tokens" in ent:
            tagged = ent["tagged_tokens"]
            text = ent.get("text")
    prompts[name] = {"target_tps": round(pr["target_alone_tok_s"], 1),
                     "by_k": by_k, "tagged": tagged, "text": text}

exp2 = {
    "target": e2["target"], "draft": e2["draft"], "c": round(d["c"], 3),
    "target_step_ms": round(d["target_step_ms"], 1),
    "draft_step_ms": round(d["draft_step_ms"], 1),
    "target_layers": d["target_layers"], "draft_layers": d["draft_layers"],
    "prompts": prompts, "tag_k": 4,
}

# ---- Exp 3 / 4 (for the summary strip) ----
r3 = e3["runs"]["B32_r80"]
exp3 = {"cont_req_s": round(r3["continuous"]["throughput_req_s"], 1),
        "static_req_s": round(r3["static"]["throughput_req_s"], 1),
        "cont_p95": round(r3["continuous"]["latency_p95_s"], 1),
        "static_p95": round(r3["static"]["latency_p95_s"], 1)}
paged = max(s["concurrency"] for s in e4["paged_block_sweep"])
exp4 = {"kv_kb": round(e4["kv_bytes_per_token"] / 1024, 1),
        "naive_waste": round(e4["naive"]["waste_frac"] * 100),
        "naive_conc": e4["naive"]["concurrency"], "paged_conc": round(paged),
        "ratio": round(paged / e4["naive"]["concurrency"])}

data = {"gpu": gpu, "exp1": exp1, "exp2": exp2, "exp3": exp3, "exp4": exp4}
out = ROOT / "site" / "data.js"
out.write_text("window.DATA = " + json.dumps(data) + ";\n", encoding="utf-8")
print("wrote", out, f"({out.stat().st_size//1024} KB)")
