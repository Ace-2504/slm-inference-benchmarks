"""Experiment 2 — With and without speculative decoding, on Modal L4.

Target = Qwen2.5-7B-Instruct, draft = Qwen2.5-0.5B-Instruct (same tokenizer family),
greedy. For four prompt types (copy / code / explain / creative) we generate once with
the target alone and once speculatively, sweeping k = 1..8, and report:
  - tok/s both ways + speedup
  - acceptance rate (fraction of drafted tokens the target kept)
  - target forward passes each way + tokens committed per target pass
  - per-token producer tags (draft-accepted / target-correction / bonus) for colouring
  - exactness: is the speculative output identical to target-alone? if not, where, and
    the top-2 logit gap there (the bf16 near-tie check)

Plus the diagnosis the brief demands: per-step time of each model alone, c = draft/target
step time, layer counts, and predicted speedup = tokens_per_target_pass / (1 + k*c).

    python -m modal run exp2_specdec/run_exp2.py     # (run from repo root)
Writes results/exp2.json.
"""
import json
import os
from pathlib import Path

import modal

from bench.modal_env import IMAGE, GPU, HF_CACHE, HF_CACHE_DIR

app = modal.App("a4-exp2-specdec")

TARGET_ID = "Qwen/Qwen2.5-7B-Instruct"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = int(os.environ.get("EXP2_TOKENS", "128"))
K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]
TAG_K = 4  # which k to save per-token tags for (the coloured-output example)

PROMPTS = {
    "copy": "Repeat the following passage exactly, word for word:\n\n"
            "\"The mitochondrion is the powerhouse of the cell. It generates most of the "
            "cell's supply of adenosine triphosphate, used as a source of chemical energy.\"",
    "code": "Write a Python function `merge_sort(arr)` that sorts a list using the merge "
            "sort algorithm. Include a short docstring.",
    "explain": "Explain how a transformer's attention mechanism works, in simple terms, "
               "to someone who knows basic linear algebra.",
    "creative": "Write a short, vivid story (a few sentences) about a lighthouse keeper "
                "who discovers the light has started speaking to him.",
}


@app.function(image=IMAGE, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE},
              timeout=3600, memory=32768)
def run(smoke: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bench.gpu_info import get_gpu_info
    from bench.speculative import speculative_generate, target_alone_greedy
    from bench.timer import benchmark
    from bench.token_utils import token_equal

    torch.manual_seed(0)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(TARGET_ID)
    target = AutoModelForCausalLM.from_pretrained(TARGET_ID, dtype=torch.bfloat16).to(dev).eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.bfloat16).to(dev).eval()

    def chat_ids(user_msg):
        msgs = [{"role": "user", "content": user_msg}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt")["input_ids"].to(dev)

    # ---- c diagnosis: single decode-step time of each model alone ----
    def step_time(model, ids):
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        cur = ids.shape[1]
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)

        def one():
            model(input_ids=nxt, past_key_values=past, use_cache=True,
                  cache_position=torch.tensor([cur], device=dev), logits_to_keep=1)
        return benchmark(one, warmup=5, repeats=20, clock="event")["median"]

    probe_ids = chat_ids(PROMPTS["explain"])
    t_step = step_time(target, probe_ids)
    d_step = step_time(draft, probe_ids)
    c = d_step / t_step
    diag = {
        "target_step_ms": t_step,
        "draft_step_ms": d_step,
        "c": c,
        "target_layers": target.config.num_hidden_layers,
        "draft_layers": draft.config.num_hidden_layers,
        "layer_ratio_draft_over_target": draft.config.num_hidden_layers / target.config.num_hidden_layers,
    }

    prompts = {"explain": PROMPTS["explain"]} if smoke else PROMPTS
    ks = [4] if smoke else K_VALUES

    results = {
        "target": TARGET_ID, "draft": DRAFT_ID, "dtype": "torch.bfloat16",
        "max_new_tokens": MAX_NEW_TOKENS, "k_values": ks,
        "gpu_info": get_gpu_info(), "diagnosis": diag, "per_prompt": {},
    }

    for name, msg in prompts.items():
        ids = chat_ids(msg)
        # target alone
        ta_ms = benchmark(lambda: target_alone_greedy(target, ids, MAX_NEW_TOKENS),
                          warmup=1, repeats=2, clock="event")
        ta_tokens = target_alone_greedy(target, ids, MAX_NEW_TOKENS)
        ta_tps = MAX_NEW_TOKENS / (ta_ms["median"] / 1e3)

        by_k = {}
        for k in ks:
            spec = speculative_generate(target, draft, ids, MAX_NEW_TOKENS, k)
            spec_ms = benchmark(
                lambda: speculative_generate(target, draft, ids, MAX_NEW_TOKENS, k),
                warmup=1, repeats=2, clock="event")
            spec_tps = MAX_NEW_TOKENS / (spec_ms["median"] / 1e3)
            eq = token_equal(spec["tokens"], ta_tokens)

            gap = None
            if not eq["match"] and eq["first_divergence"] is not None:
                # top-2 logit gap of the TARGET at the divergence position (near-tie check)
                d = eq["first_divergence"]
                with torch.inference_mode():
                    full = torch.cat([ids, torch.tensor([ta_tokens[:d]], device=dev).long()], dim=1) \
                        if d > 0 else ids
                    lg = target(input_ids=full, use_cache=False, logits_to_keep=1).logits[0, -1]
                    top2 = torch.topk(lg.float(), 2).values
                    gap = float(top2[0] - top2[1])

            entry = {
                "k": k, "acceptance_rate": spec["acceptance_rate"],
                "target_passes": spec["target_passes"],
                "tokens_per_target_pass": spec["tokens_per_target_pass"],
                "spec_tok_s": spec_tps, "target_alone_tok_s": ta_tps,
                "speedup": spec_tps / ta_tps,
                "predicted_speedup": spec["tokens_per_target_pass"] / (1 + k * c),
                "match": eq["match"], "first_divergence": eq["first_divergence"],
                "divergence_logit_gap": gap,
            }
            if k == TAG_K:
                entry["tagged_tokens"] = [
                    {"tok": tok.decode([t]), "tag": g}
                    for t, g in zip(spec["tokens"], spec["tags"])
                ]
                entry["text"] = tok.decode(spec["tokens"])
            by_k[str(k)] = entry

        results["per_prompt"][name] = {
            "prompt": msg, "target_alone_tok_s": ta_tps,
            "target_alone_passes": MAX_NEW_TOKENS, "by_k": by_k,
            "target_alone_text": tok.decode(ta_tokens),
        }

    return results


@app.local_entrypoint()
def main(smoke: bool = False):
    res = run.remote(smoke=smoke)
    out = Path(__file__).resolve().parents[1] / "results" / \
        ("exp2_smoke.json" if smoke else "exp2.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))

    d = res["diagnosis"]
    print(f"\n=== Experiment 2 (speculative decoding) on {res['gpu_info'].get('name')} ===")
    print(f"target={res['target']} ({d['target_layers']}L)  draft={res['draft']} ({d['draft_layers']}L)")
    print(f"step times: target={d['target_step_ms']:.2f}ms draft={d['draft_step_ms']:.2f}ms "
          f"-> c={d['c']:.3f}  (layer ratio {d['layer_ratio_draft_over_target']:.3f})")
    for name, pr in res["per_prompt"].items():
        print(f"\n[{name}]  target-alone {pr['target_alone_tok_s']:.1f} tok/s")
        print(f"  {'k':>2} {'accept':>7} {'tok/pass':>9} {'spec tok/s':>11} "
              f"{'speedup':>8} {'pred':>6} {'exact':>6}")
        for k in res["k_values"]:
            e = pr["by_k"][str(k)]
            print(f"  {k:>2} {e['acceptance_rate']*100:>6.0f}% {e['tokens_per_target_pass']:>9.2f} "
                  f"{e['spec_tok_s']:>11.1f} {e['speedup']:>7.2f}x {e['predicted_speedup']:>5.2f}x "
                  f"{str(e['match']):>6}")
    print(f"\nwrote {out}")
