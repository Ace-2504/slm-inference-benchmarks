"""Experiment 4, part 1 — the NAIVE contiguous KV allocator (which we implement) on L4.

Naive serving reserves one contiguous KV block per sequence, sized for the server's max
sequence length, at admission time. We compute the concurrency this allows from the
model's real KV-per-token footprint, then CONFIRM it by allocating max-length KV blocks
until we hit a CUDA out-of-memory error.

KV bytes/token = 2 * layers * kv_heads * head_dim * bytes_per_element
  (kv_heads, not attention heads — Qwen2.5-0.5B uses grouped-query attention, so this is
   much smaller than a naive layers*heads*head_dim guess; getting it wrong is the #1 error.)

    python -m modal run exp4_paged/run_exp4_naive.py     # writes results/exp4_naive.json
"""
import json
from pathlib import Path

import modal

from bench.modal_env import IMAGE, GPU, HF_CACHE, HF_CACHE_DIR

app = modal.App("a4-exp4-naive")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LEN = 2048  # the server's max sequence length -> what naive reserves per sequence


@app.function(image=IMAGE, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE}, timeout=1200)
def run():
    import torch
    from transformers import AutoModelForCausalLM

    from bench.gpu_info import get_gpu_info

    dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    kv_bytes_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2

    torch.cuda.synchronize()
    free_after_model, total = torch.cuda.mem_get_info()

    # leave a safety margin for activations/fragmentation, matching a real server
    budget = int(free_after_model * 0.90)
    bytes_per_seq = kv_bytes_per_token * MAX_LEN
    predicted = budget // bytes_per_seq

    # confirm: allocate one max-length KV block per sequence until OOM
    blocks = []
    measured = 0
    elems_per_seq = bytes_per_seq // 2  # bf16 = 2 bytes/elem
    try:
        while True:
            blocks.append(torch.empty(elems_per_seq, dtype=torch.bfloat16, device=dev))
            torch.cuda.synchronize()
            measured += 1
            if measured > predicted * 3 + 50:  # safety stop
                break
    except torch.cuda.OutOfMemoryError:
        pass
    del blocks
    torch.cuda.empty_cache()

    return {
        "model": MODEL_ID, "dtype": "torch.bfloat16", "max_len": MAX_LEN,
        "config": {
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "head_dim": head_dim,
        },
        "kv_bytes_per_token": kv_bytes_per_token,
        "bytes_per_seq_naive": bytes_per_seq,
        "gpu_total_bytes": total,
        "free_after_model_bytes": free_after_model,
        "kv_budget_bytes": budget,
        "predicted_concurrency_naive": int(predicted),
        "measured_concurrency_naive": measured,
        "gpu_info": get_gpu_info(),
    }


@app.local_entrypoint()
def main():
    r = run.remote()
    out = Path(__file__).resolve().parents[1] / "results" / "exp4_naive.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(r, indent=2))
    print(f"\n=== Exp 4 (naive KV) on {r['gpu_info'].get('name')} — {r['model']} ===")
    c = r["config"]
    print(f"layers={c['num_hidden_layers']} kv_heads={c['num_key_value_heads']} "
          f"head_dim={c['head_dim']}  -> KV = {r['kv_bytes_per_token']} bytes/token "
          f"= {r['kv_bytes_per_token']/1024:.1f} KB/token")
    print(f"naive reserves {r['bytes_per_seq_naive']/1024**2:.1f} MB/seq (max_len={r['max_len']})")
    print(f"KV budget {r['kv_budget_bytes']/1024**3:.2f} GB -> "
          f"predicted {r['predicted_concurrency_naive']} seqs, "
          f"measured (OOM) {r['measured_concurrency_naive']} seqs")
    print(f"wrote {out}")
