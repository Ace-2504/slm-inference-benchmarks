# Inference Optimization Benchmarks

Measuring what each of four serving optimizations is actually worth — on a real GPU (**NVIDIA L4**,
Modal), with timings that survive scrutiny. One inference pipeline, four controlled experiments, each
toggling exactly one thing. **The rule across all four: both sides must emit identical tokens before any
speedup is claimed.**

**🔴 Live demo:** https://harman-inference-lab.vercel.app — runs experiments 1 & 2 on a Modal L4 when you
press a button (scale-to-zero, so the first request wakes the GPU). Exp 1 has a batch-size control and shows
both modes' text + tok/s; Exp 2 streams the answer with each token coloured by who produced it.

| # | Optimization | Headline finding |
|---|--------------|------------------|
| 1 | **KV cache** | At batch 1 the uncached loop does **60× more arithmetic yet finishes in the same time** (memory-bound); the cache only wins as the batch fills the ALUs (→ 9× at batch 16). |
| 2 | **Speculative decoding** | Naive it's a **slowdown** — the 0.5B draft is kernel-launch-bound (`c≈0.43`); the diagnosis, not the mechanism, is the work. Exact modulo bf16 near-ties (gap 0.0). |
| 3 | **Continuous batching** | Static isn't slow, it's **idle** — the batch drains to near-empty while one long request finishes; continuous holds slots full for **~3× throughput**. |
| 4 | **PagedAttention** | Reserving max-length per sequence **wastes 91.8%** of KV; paging fits **~12× more sequences** from the same budget. |

## How the timings are trusted

Before collecting a single number, the harness passes an **8-test validation gauntlet**
([`bench/GAUNTLET.md`](bench/GAUNTLET.md)) — a measurement instrument is trusted by calibration against
ground truths it can't fake. It proves an un-synchronised matmul under-reports its time by ~330×, that the
CUDA-event and `perf_counter` clocks agree, and that measured FP32 throughput never exceeds the card's
roofline. Every timed measurement uses `cuda.synchronize` + warm-up + ≥3 repeats with spread, and times
prefill separately from decode.

## Structure

```
bench/           timing harness (timer, stats, gpu_info, token_utils), gauntlet,
                 decode loops, speculative decoder, batching sim, request generator
exp1_kvcache/    cached vs uncached loops + batch sweep (Modal L4) + plots
exp2_specdec/    speculative decoder + k-sweep + 4-prompt table + coloured tokens
exp3_batching/   step-time measurement + static/continuous discrete-event sim + plots
exp4_paged/      naive contiguous allocator (admit-until-OOM) + paged arithmetic + plots
results/         one JSON per experiment (raw enough to redraw every plot)
site/            deployed live demo (experiments 1 & 2)
report.pdf       ≤ 2 pages
```

## Reproduce, in order

Prereqs: a Modal account (`modal` CLI configured), Python 3.12 with `torch transformers matplotlib
numpy fpdf2 modal`, and (Windows) `PYTHONUTF8=1`. All GPU work runs on Modal L4; models are pulled into a
persistent Modal volume on first run. Run everything from the repo root.

```bash
# 0. validate the timing harness (local GPU or any CUDA box)
python bench/gauntlet.py

# 1. KV cache — cached vs uncached, batch sweep 1..16
python -m modal run exp1_kvcache/run_exp1.py
python exp1_kvcache/plot_exp1.py

# 2. speculative decoding — Qwen2.5-7B target + 0.5B draft, k sweep, 4 prompts
python -m modal run exp2_specdec/run_exp2.py
python exp2_specdec/plot_exp2.py
python exp2_specdec/render_tokens.py

# 3. continuous batching — measure step times on L4, then simulate both policies
python -m modal run exp3_batching/run_exp3.py
python exp3_batching/simulate_and_plot.py

# 4. PagedAttention — naive allocator + admit-until-OOM, then paged arithmetic
python -m modal run exp4_paged/run_exp4_naive.py
python exp4_paged/analyze_and_plot.py

# report (<=2 pages)
python report/make_report.py

# live-demo site: bake the real results into the page, then deploy the static frontend
python site/build_site.py           # -> site/data.js (embeds exp1/exp2 recorded data)
# (deploy site/ to Vercel; the page runs exp 1 & 2 live if the Modal endpoint is up,
#  otherwise it shows the recorded results and the sliders explore the measured sweep)
```

Local plotting/sim scripts need the repo root on `PYTHONPATH` (`PYTHONPATH=. python exp3_batching/...`).

## Hardware & cost

Every number is on an **NVIDIA L4 (24 GB)** via Modal scale-to-zero, models in **bf16**. Per-experiment
spend is logged in [`costs.md`](costs.md) (≈ $2.6 total for the whole assignment).
