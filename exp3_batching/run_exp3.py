"""Experiment 3, part 1 — measure the real decode-step time vs batch width on Modal L4.

Continuous vs static batching differ only in SCHEDULING, so we measure the one physical
input the schedule can't change — how long a single decode step takes for a batch of b
active sequences — and feed it into the discrete-event simulator (bench.batching_sim),
which runs BOTH policies over the same request stream. This isolates the scheduling
variable cleanly and yields the utilisation / queue-depth plots.

    python -m modal run exp3_batching/run_exp3.py     # writes results/exp3_steptimes.json
"""
import json
from pathlib import Path

import modal

from bench.modal_env import IMAGE, GPU, HF_CACHE, HF_CACHE_DIR

app = modal.App("a4-exp3-steptimes")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BATCH_WIDTHS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
CONTEXT_LEN = 128  # representative prefix length for the decode-step measurement


@app.function(image=IMAGE, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE}, timeout=1200)
def measure():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bench.gpu_info import get_gpu_info
    from bench.timer import benchmark

    torch.manual_seed(0)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(dev).eval()

    # clock warm-up burst (bring the GPU off idle before the first measurement, so the
    # first width isn't inflated by clock ramp — the lesson from gauntlet G8)
    warm = torch.randint(0, model.config.vocab_size, (16, CONTEXT_LEN), device=dev)
    for _ in range(12):
        model(input_ids=warm, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()

    def step_time(b):
        ids = torch.randint(0, model.config.vocab_size, (b, CONTEXT_LEN), device=dev)
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        cur = CONTEXT_LEN

        def one():
            model(input_ids=nxt, past_key_values=past, use_cache=True,
                  cache_position=torch.tensor([cur], device=dev), logits_to_keep=1)
        return benchmark(one, warmup=5, repeats=15, clock="event")["median"]

    step_ms = {}
    for b in BATCH_WIDTHS:
        step_ms[b] = step_time(b)

    cfg = model.config
    kv_bytes_per_token = (2 * cfg.num_hidden_layers * cfg.num_key_value_heads
                          * (cfg.hidden_size // cfg.num_attention_heads) * 2)  # bf16=2B
    return {
        "model": MODEL_ID, "dtype": "torch.bfloat16", "context_len": CONTEXT_LEN,
        "batch_widths": BATCH_WIDTHS, "step_ms": step_ms,
        "config": {
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "head_dim": cfg.hidden_size // cfg.num_attention_heads,
            "kv_bytes_per_token": kv_bytes_per_token,
        },
        "gpu_info": get_gpu_info(),
    }


@app.local_entrypoint()
def main():
    res = measure.remote()
    out = Path(__file__).resolve().parents[1] / "results" / "exp3_steptimes.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\n=== Exp 3 step times on {res['gpu_info'].get('name')} ({res['model']}) ===")
    print(f"KV bytes/token = {res['config']['kv_bytes_per_token']} "
          f"(layers={res['config']['num_hidden_layers']}, "
          f"kv_heads={res['config']['num_key_value_heads']}, "
          f"head_dim={res['config']['head_dim']})")
    for b in res["batch_widths"]:
        print(f"  batch {b:>3}: {res['step_ms'][str(b)] if str(b) in res['step_ms'] else res['step_ms'][b]:.3f} ms/step")
    print(f"wrote {out}")
