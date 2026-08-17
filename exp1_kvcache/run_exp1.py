"""Experiment 1 — With and without the KV cache, on Modal L4.

Two decode loops over the SAME gpt2 (124M) model, greedy, at batch sizes 1/2/4/8/16:
  cached   — prefill once, one token per step over cached KV
  uncached — cache off, re-pass the whole growing sequence every step

For each batch we report generated text (both) + token-equality, total wall-clock +
tok/s (spread over repeats), per-step time vs step index, and the token positions each
loop computed. The headline is the speedup-vs-batch curve: the uncached loop does ~60x
more arithmetic yet at batch 1 finishes at nearly the same time (roofline), and the gap
only opens as the batch grows.

    python -m modal run exp1_kvcache/run_exp1.py     # (run from repo root)
Writes results/exp1.json.
"""
import json
from pathlib import Path

import modal

from bench.modal_env import IMAGE, GPU, HF_CACHE, HF_CACHE_DIR

app = modal.App("a4-exp1-kvcache")

MODEL_ID = "gpt2"
PROMPT_TARGET_TOKENS = 512
MAX_NEW_TOKENS = 64
BATCH_SIZES = [1, 2, 4, 8, 16]
REPEATS = 3

# A coherent seed paragraph, repeated up to PROMPT_TARGET_TOKENS so the generated text
# is real (not gibberish) while the prompt is long enough to make the arithmetic gap big.
SEED_TEXT = (
    "The history of computing is a history of moving work closer to where the data "
    "already lives. Every generation of engineers rediscovers that the cost of a "
    "computation is rarely the arithmetic itself but the waiting: waiting for memory, "
    "waiting for the network, waiting for a lock. A processor that never stalls is a "
    "myth, and the art of systems work is deciding which stalls are worth paying for. "
)


@app.function(image=IMAGE, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE}, timeout=1800)
def run():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bench.decode_loops import greedy_cached, greedy_uncached
    from bench.gpu_info import get_gpu_info
    from bench.stats import summarize
    from bench.timer import benchmark, time_event, sync
    from bench.token_utils import token_equal, token_positions

    torch.manual_seed(0)
    dev = "cuda"
    # bf16 = the realistic serving dtype. It matters for the roofline story: with tensor
    # cores the uncached loop's ~60x extra arithmetic is genuinely near-free at batch 1
    # (so batch-1 speedup ~1), and the gap only opens as the batch saturates the ALUs. In
    # fp32 (no tensor cores) that extra arithmetic is NOT free and inflates the batch-1
    # speedup, muting the very finding Exp 1 is about.
    DTYPE = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(dev).eval()

    # Build a >=PROMPT_TARGET_TOKENS prompt, then trim to exactly that length.
    text = SEED_TEXT
    while len(tok(text)["input_ids"]) < PROMPT_TARGET_TOKENS:
        text += SEED_TEXT
    prompt_ids = tok(text, return_tensors="pt")["input_ids"][:, :PROMPT_TARGET_TOKENS].to(dev)
    prompt_len = prompt_ids.shape[1]

    tp = token_positions(prompt_len, MAX_NEW_TOKENS)
    results = {
        "model": MODEL_ID,
        "dtype": str(DTYPE),
        "prompt_len": prompt_len,
        "max_new_tokens": MAX_NEW_TOKENS,
        "batch_sizes": BATCH_SIZES,
        "repeats": REPEATS,
        "token_positions": {
            "cached_total": tp["cached_total"],
            "uncached_total": tp["uncached_total"],
            "ratio_uncached_over_cached": tp["ratio_uncached_over_cached"],
        },
        "gpu_info": get_gpu_info(),
        "per_batch": {},
    }

    for bs in BATCH_SIZES:
        ids = prompt_ids.repeat(bs, 1)

        # correctness + per-step timing (one recorded run each)
        gen_c, steps_c = greedy_cached(model, ids, MAX_NEW_TOKENS, record_steps=True)
        gen_u, steps_u = greedy_uncached(model, ids, MAX_NEW_TOKENS, record_steps=True)
        eq = token_equal(gen_c[0].tolist(), gen_u[0].tolist())

        # prefill time (cached side), isolated
        def _prefill():
            model(input_ids=ids, use_cache=True)
        prefill = benchmark(_prefill, warmup=2, repeats=REPEATS, clock="event")

        # throughput: whole decode loop, no per-step recording, spread over repeats
        cached_ms = benchmark(
            lambda: greedy_cached(model, ids, MAX_NEW_TOKENS, record_steps=False),
            warmup=1, repeats=REPEATS, clock="event")
        uncached_ms = benchmark(
            lambda: greedy_uncached(model, ids, MAX_NEW_TOKENS, record_steps=False),
            warmup=1, repeats=REPEATS, clock="event")

        gen_tokens = bs * MAX_NEW_TOKENS
        cached_tps = gen_tokens / (cached_ms["median"] / 1e3)
        uncached_tps = gen_tokens / (uncached_ms["median"] / 1e3)

        results["per_batch"][str(bs)] = {
            "batch_size": bs,
            "match": eq["match"],
            "first_divergence": eq["first_divergence"],
            "cached_total_ms": cached_ms,
            "uncached_total_ms": uncached_ms,
            "cached_tok_s": cached_tps,
            "uncached_tok_s": uncached_tps,
            "cache_speedup": cached_tps / uncached_tps,
            "prefill_ms": prefill,
            "cached_per_step_ms": steps_c,
            "uncached_per_step_ms": steps_u,
        }

    # readable generated text at batch 1 (both loops), for the report / demo
    gen_c1, _ = greedy_cached(model, prompt_ids, MAX_NEW_TOKENS)
    gen_u1, _ = greedy_uncached(model, prompt_ids, MAX_NEW_TOKENS)
    results["sample_text"] = {
        "prompt_tail": tok.decode(prompt_ids[0, -40:]),
        "cached": tok.decode(gen_c1[0]),
        "uncached": tok.decode(gen_u1[0]),
        "match": token_equal(gen_c1[0].tolist(), gen_u1[0].tolist())["match"],
    }
    return results


@app.local_entrypoint()
def main():
    res = run.remote()
    out = Path(__file__).resolve().parents[1] / "results" / "exp1.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))

    print(f"\n=== Experiment 1 (KV cache) on {res['gpu_info'].get('name')} ===")
    print(f"model={res['model']} prompt={res['prompt_len']}tok gen={res['max_new_tokens']} "
          f"| token positions: cached={res['token_positions']['cached_total']} "
          f"uncached={res['token_positions']['uncached_total']} "
          f"({res['token_positions']['ratio_uncached_over_cached']:.1f}x)")
    print(f"{'batch':>6} {'cached tok/s':>13} {'uncached tok/s':>15} {'speedup':>8} {'match':>6}")
    for bs in BATCH_SIZES:
        r = res["per_batch"][str(bs)]
        print(f"{bs:>6} {r['cached_tok_s']:>13.1f} {r['uncached_tok_s']:>15.1f} "
              f"{r['cache_speedup']:>7.2f}x {str(r['match']):>6}")
    print(f"\nsample match (batch1): {res['sample_text']['match']}")
    print(f"wrote {out}")
