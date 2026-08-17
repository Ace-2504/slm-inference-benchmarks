"""Experiment 2 diagnosis follow-up — can we bring c down?

The naive draft step (~25 ms for 0.5B) is far above its ~3 ms memory-bound ideal because
it is kernel-launch-bound: 24 layers x many tiny kernels, the GPU idling between launches.
CUDA graphs collapse those launches into one replay. This probe measures the draft's
single-token decode step (a) naively and (b) under torch.compile(mode="reduce-overhead")
over a StaticCache, and recomputes c against the (bandwidth-bound, unchanged) 7B target.

    python -m modal run exp2_specdec/probe_c_fix.py     # writes results/exp2_cfix.json
"""
import json
from pathlib import Path

import modal

from bench.modal_env import IMAGE, GPU, HF_CACHE, HF_CACHE_DIR

app = modal.App("a4-exp2-cfix")

TARGET_ID = "Qwen/Qwen2.5-7B-Instruct"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CTX = 128


@app.function(image=IMAGE, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE},
              timeout=1800, memory=32768)
def run():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

    from bench.gpu_info import get_gpu_info
    from bench.timer import benchmark

    torch.manual_seed(0)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(TARGET_ID)
    target = AutoModelForCausalLM.from_pretrained(TARGET_ID, dtype=torch.bfloat16).to(dev).eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.bfloat16).to(dev).eval()
    ids = torch.randint(0, draft.config.vocab_size, (1, CTX), device=dev)

    # --- naive single-token decode step time (DynamicCache) ---
    def naive_step(model):
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)

        def one():
            model(input_ids=nxt, past_key_values=past, use_cache=True,
                  cache_position=torch.tensor([CTX], device=dev), logits_to_keep=1)
        return benchmark(one, warmup=5, repeats=20, clock="event")["median"]

    t_naive = naive_step(target)
    d_naive = naive_step(draft)

    # --- compiled draft step: manual CUDA-graph capture over a StaticCache ---
    # A single fixed-shape decode step captured into a CUDA graph replays all of the
    # draft's ~240 tiny kernel launches as one graph launch — eliminating the launch
    # overhead that makes the draft slow. (StaticCache gives fixed buffers so capture is
    # valid; the position is held fixed since this measures step time, not real decoding.)
    d_compiled = None
    err = None
    try:
        max_len = CTX + 64
        static = StaticCache(config=draft.config, max_batch_size=1, max_cache_len=max_len,
                             device=dev, dtype=torch.bfloat16)
        inp = torch.zeros((1, 1), dtype=torch.long, device=dev)
        pos = torch.tensor([CTX], device=dev)
        with torch.inference_mode():
            draft(input_ids=ids, past_key_values=static, use_cache=True,
                  cache_position=torch.arange(CTX, device=dev))
            # warm up on a side stream before capture
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    draft(input_ids=inp, past_key_values=static, use_cache=True, cache_position=pos)
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                draft(input_ids=inp, past_key_values=static, use_cache=True, cache_position=pos)

        d_compiled = benchmark(lambda: g.replay(), warmup=5, repeats=30, clock="event")["median"]
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    out = {
        "target": TARGET_ID, "draft": DRAFT_ID, "gpu_info": get_gpu_info(),
        "target_step_ms": t_naive, "draft_step_naive_ms": d_naive,
        "draft_step_compiled_ms": d_compiled,
        "c_naive": d_naive / t_naive,
        "c_compiled": (d_compiled / t_naive) if d_compiled else None,
        "compile_error": err,
    }
    return out


@app.local_entrypoint()
def main():
    r = run.remote()
    out = Path(__file__).resolve().parents[1] / "results" / "exp2_cfix.json"
    out.write_text(json.dumps(r, indent=2))
    print(f"\n=== Exp 2 c-reduction on {r['gpu_info'].get('name')} ===")
    print(f"target step      : {r['target_step_ms']:.2f} ms")
    print(f"draft step naive : {r['draft_step_naive_ms']:.2f} ms  -> c = {r['c_naive']:.3f}")
    if r["draft_step_compiled_ms"]:
        print(f"draft compiled   : {r['draft_step_compiled_ms']:.2f} ms  -> c = {r['c_compiled']:.3f}")
    else:
        print(f"compiled draft   : FAILED ({r['compile_error']})")
    print(f"wrote {out}")
