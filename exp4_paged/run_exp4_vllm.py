"""Experiment 4, part 3 — the PAGED side, confirmed with the real vLLM engine on L4.

vLLM manages KV in fixed-size blocks from a shared pool. We ask the running engine how
many GPU KV blocks it allocated and the block size, which gives the real paged KV
capacity (blocks x block_size tokens) and the concurrency it supports at a realistic
mean sequence length — the real-engine counterpart to the arithmetic in analyze_and_plot.py.
We also measure serving throughput at high concurrency.

    python -m modal run exp4_paged/run_exp4_vllm.py     # writes results/exp4_vllm.json
"""
import json
from pathlib import Path

import modal

VLLM_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    # vLLM 0.11 uses the transformers 4.x tokenizer API (all_special_tokens_extended),
    # which transformers 5.x removed — pin to 4.x so the two agree.
    .pip_install("vllm==0.11.0", "transformers==4.57.1")
    .add_local_python_source("bench")
)
HF_CACHE = modal.Volume.from_name("a4-hf-cache", create_if_missing=True)
HF_CACHE_DIR = "/root/.cache/huggingface"

app = modal.App("a4-exp4-vllm")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GMU = 0.85
MAX_LEN = 2048


@app.function(image=VLLM_IMAGE, gpu="L4", volumes={HF_CACHE_DIR: HF_CACHE},
              timeout=1800, memory=32768)
def run():
    import time
    import numpy as np
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL_ID, dtype="bfloat16", gpu_memory_utilization=GMU,
              max_model_len=MAX_LEN, enforce_eager=True)

    # dig the real paged KV capacity out of the engine (attribute path varies by version)
    num_blocks = block_size = None
    for obj_path in ("llm_engine.cache_config", "llm_engine.vllm_config.cache_config"):
        obj = llm
        try:
            for a in obj_path.split("."):
                obj = getattr(obj, a)
            num_blocks = getattr(obj, "num_gpu_blocks", None) or num_blocks
            block_size = getattr(obj, "block_size", None) or block_size
        except AttributeError:
            continue
    kv_tokens_capacity = (num_blocks * block_size) if (num_blocks and block_size) else None

    # heavy-tailed output lengths -> realistic concurrency + throughput
    rng = np.random.default_rng(0)
    n = 256
    out_lens = np.clip(rng.lognormal(np.log(64), 0.9, n).round().astype(int), 4, 512)
    prompts = [f"Write about topic number {i} in a few sentences." for i in range(n)]
    sps = [SamplingParams(temperature=0.0, max_tokens=int(L)) for L in out_lens]

    t0 = time.time()
    outs = llm.generate(prompts, sps)
    dt = time.time() - t0
    total_out = sum(len(o.outputs[0].token_ids) for o in outs)
    mean_len = float(out_lens.mean())

    paged_concurrency = (kv_tokens_capacity / mean_len) if kv_tokens_capacity else None

    return {
        "model": MODEL_ID, "gpu_memory_utilization": GMU, "max_len": MAX_LEN,
        "num_gpu_blocks": num_blocks, "block_size": block_size,
        "kv_tokens_capacity": kv_tokens_capacity,
        "mean_output_len": mean_len,
        "paged_concurrency_at_mean": paged_concurrency,
        "throughput_tok_s": total_out / dt, "throughput_req_s": n / dt,
        "n_requests": n, "total_output_tokens": total_out, "wall_s": dt,
    }


@app.local_entrypoint()
def main():
    r = run.remote()
    out = Path(__file__).resolve().parents[1] / "results" / "exp4_vllm.json"
    out.write_text(json.dumps(r, indent=2))
    print(f"\n=== Exp 4 paged (vLLM real engine) — {r['model']} on L4 ===")
    print(f"GPU KV blocks = {r['num_gpu_blocks']} x block_size {r['block_size']} "
          f"= {r['kv_tokens_capacity']} KV tokens")
    print(f"paged concurrency at mean_len {r['mean_output_len']:.0f} = "
          f"{r['paged_concurrency_at_mean']:.0f} seqs")
    print(f"throughput: {r['throughput_tok_s']:.0f} tok/s, {r['throughput_req_s']:.1f} req/s "
          f"({r['n_requests']} reqs in {r['wall_s']:.1f}s)")
    print(f"wrote {out}")
