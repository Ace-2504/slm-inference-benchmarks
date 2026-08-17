"""Live-demo backend on Modal L4 — the GPU endpoints the Vercel site calls.

Two FastAPI endpoints, both reusing the exact harness code the experiments use:
  GET /exp1?batch=N&prompt=...  -> cached vs uncached decode: text + tok/s both modes + match
  GET /exp2?k=K&prompt=...      -> target-alone vs speculative: coloured tokens (who produced
                                   each) + acceptance rate + both throughputs

Deploy (keeps a scale-to-zero endpoint warm between requests):
    python -m modal deploy site/demo_backend.py
The printed URLs go into site/index.html (BASE).
"""
import modal

from bench.modal_env import GPU, HF_CACHE, HF_CACHE_DIR

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "numpy", "accelerate", "fastapi[standard]")
    .add_local_python_source("bench")
)

app = modal.App("a4-live-demo")

EXP1_MODEL = "gpt2"
TARGET_ID = "Qwen/Qwen2.5-7B-Instruct"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@app.cls(image=IMAGE, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE},
         scaledown_window=300, timeout=600, memory=32768)
class Demo:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.dev = "cuda"
        self.tok1 = AutoTokenizer.from_pretrained(EXP1_MODEL)
        self.m1 = AutoModelForCausalLM.from_pretrained(
            EXP1_MODEL, dtype=torch.bfloat16).to(self.dev).eval()
        self.tokq = AutoTokenizer.from_pretrained(TARGET_ID)
        self.target = AutoModelForCausalLM.from_pretrained(
            TARGET_ID, dtype=torch.bfloat16).to(self.dev).eval()
        self.draft = AutoModelForCausalLM.from_pretrained(
            DRAFT_ID, dtype=torch.bfloat16).to(self.dev).eval()

    @modal.fastapi_endpoint()
    def exp1(self, prompt: str = "The future of computing is", batch: int = 1,
             max_new_tokens: int = 48):
        import torch
        from bench.decode_loops import greedy_cached, greedy_uncached
        from bench.timer import benchmark
        from bench.token_utils import token_equal

        batch = max(1, min(int(batch), 16))
        ids = self.tok1(prompt, return_tensors="pt")["input_ids"].to(self.dev)
        ids = ids.repeat(batch, 1)
        gen_c, _ = greedy_cached(self.m1, ids, max_new_tokens)
        gen_u, _ = greedy_uncached(self.m1, ids, max_new_tokens)
        cm = benchmark(lambda: greedy_cached(self.m1, ids, max_new_tokens),
                       warmup=1, repeats=2)["median"]
        um = benchmark(lambda: greedy_uncached(self.m1, ids, max_new_tokens),
                       warmup=1, repeats=2)["median"]
        n = batch * max_new_tokens
        return {
            "batch": batch,
            "cached_text": self.tok1.decode(gen_c[0]),
            "uncached_text": self.tok1.decode(gen_u[0]),
            "cached_tok_s": n / (cm / 1e3), "uncached_tok_s": n / (um / 1e3),
            "speedup": (n / (cm / 1e3)) / (n / (um / 1e3)),
            "match": token_equal(gen_c[0].tolist(), gen_u[0].tolist())["match"],
        }

    @modal.fastapi_endpoint()
    def exp2(self, prompt: str = "Explain attention in simple terms.", k: int = 4,
             max_new_tokens: int = 64):
        from bench.speculative import speculative_generate, target_alone_greedy
        from bench.timer import benchmark

        k = max(1, min(int(k), 8))
        msgs = [{"role": "user", "content": prompt}]
        text = self.tokq.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tokq(text, return_tensors="pt")["input_ids"].to(self.dev)

        spec = speculative_generate(self.target, self.draft, ids, max_new_tokens, k)
        ta = target_alone_greedy(self.target, ids, max_new_tokens)
        sm = benchmark(lambda: speculative_generate(self.target, self.draft, ids, max_new_tokens, k),
                       warmup=0, repeats=1)["median"]
        tm = benchmark(lambda: target_alone_greedy(self.target, ids, max_new_tokens),
                       warmup=0, repeats=1)["median"]
        return {
            "k": k,
            "tagged": [{"tok": self.tokq.decode([t]), "tag": g}
                       for t, g in zip(spec["tokens"], spec["tags"])],
            "acceptance_rate": spec["acceptance_rate"],
            "spec_tok_s": max_new_tokens / (sm / 1e3),
            "target_tok_s": max_new_tokens / (tm / 1e3),
            "speedup": (max_new_tokens / (sm / 1e3)) / (max_new_tokens / (tm / 1e3)),
        }
