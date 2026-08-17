# PLAN.md — Assignment 4: Inference Optimization Benchmarks

> **What this file is:** the end-to-end plan I (Claude) follow for this assignment. It is the map;
> [`story.md`](story.md) is the running log of what actually happened. When reality and this plan
> disagree, I update this file and record *why* in `story.md`.

**Assignment in one line:** Build **one** inference pipeline + a trustworthy timing harness, then run
**four controlled experiments** that each toggle exactly one serving optimization on/off, measure what
each is worth on real hardware, and prove both sides emit **identical tokens** before claiming any speedup.

---

## 0. The task and the outcome (plain words)

**Task.** Training gave us a model that *can* answer. Inference decides whether anyone can *afford* to ask.
Session 7 taught four tricks that sit between "works" and "serves". This assignment makes me measure each
one myself instead of trusting a number in a paper.

I build **one pipeline** and push four experiments through it. Each experiment is the *same model, same
prompt, same decoding settings*, with exactly **one** thing switched off:

| # | OFF (naive) | ON (optimized) | The question it answers |
|---|-------------|----------------|--------------------------|
| 1 | Recompute whole prefix every step | **KV cache** | When is recompute actually free? (roofline) |
| 2 | Big model decodes alone | **Speculative decoding** (draft proposes, target verifies) | Is a 15× smaller draft actually a win? |
| 3 | Static batching (batch finishes together) | **Continuous batching** (refill finished slots) | Static loses by being *idle*, not slow |
| 4 | One contiguous KV block per sequence | **PagedAttention** (shared block pool) | How much memory is reserved-but-never-written |

**The rule that runs through all four:** the two sides must produce the **same tokens**. If the output
changes, I'm not measuring a speedup — I'm measuring two different programs. Check and report it *every time*.

**Outcome (what "done" looks like):**
- A repo with `bench/ exp1_kvcache/ exp2_specdec/ exp3_batching/ exp4_paged/ results/ site/ report.pdf README.md`.
- Every number: synchronized, warmed-up, repeated ≥3× with its spread, GPU named.
- Each experiment: proof both sides match, headline numbers, the required plots (all axis-labelled with units).
- A **live Vercel demo** (Modal GPU backend) running experiments 1 & 2 on button-press.
- A **≤2-page report.pdf**: one 4-experiment table, four best plots, one expected-vs-measured sentence each,
  and one closing "measurement that changed how I think about serving a model."
- Spend logged per experiment; every surprising result *investigated*, not reported as-is.

**Grading lens (what they scrutinize):** (a) do timings survive scrutiny; (b) did I check token-equality
before claiming a speedup; (c) every plot labelled with units; (d) did I *investigate* surprises.
**They said Experiment 2 is where to spend the most time** — the obvious measurement leads to the wrong
conclusion, and the diagnosis is the real work.

---

## 1. Golden rules (apply to every experiment)

1. **Change one thing.** Same model, prompt, decoding settings on both sides. One toggle.
2. **Token-equality gate.** Diff both sides token-by-token; report match, or the exact first divergence index.
3. **Trustworthy timing (non-negotiable):**
   - `torch.cuda.synchronize()` on both sides of anything timed (CUDA is async).
   - One **warm-up** pass first (don't time autotuning / compile).
   - **≥3 repeats**, report the spread (mean ± std or min/median/max), never a single number.
   - Time **prefill** and **decode** separately — different bottlenecks.
4. **Report hardware for every number.** "A speedup without a GPU name is not a result."
5. **Log spend per experiment** (Modal GPU-seconds → $).
6. **Investigate surprises.** A weird number is a lead, not a finding.

---

## 2. Hardware & model decisions

| Where | Hardware | Used for |
|-------|----------|----------|
| Local | **RTX 3060, 12 GB** | Exp-1 dev + cheap iteration; harness unit tests. 7B will **not** fit here. |
| Cloud | **Modal L4 24 GB (or A10G), scale-to-zero** | Exp-2/3/4, and the live-demo endpoints. Brief's recommended, affordable. |

**Models (defaults — confirm in Prerequisites):**
- **Exp 1 (KV cache):** the **125M model from Session 2** — cleanest roofline, matches the lab's calibration
  table (61× extra token-positions, batch-1 near-parity). Fine-tuned model from Assignment 1 is the alt.
- **Exp 2 (speculative):** **Qwen2.5-7B-Instruct** (target) + **Qwen2.5-0.5B-Instruct** (draft) — same
  tokenizer family, the pair the lab used. Must run on Modal (7B > 12 GB local).
- **Exp 3/4 (batching / paging):** **Qwen2.5-0.5B-Instruct** on Modal L4 — small enough to make many-sequence
  effects visible, and it uses **grouped-query attention** (few kv_heads) which is exactly the trap Exp-4's
  KV-byte arithmetic is teaching. vLLM serves the "ON" side for 3 & 4.

**Budget:** target modest (< ~$15 GPU). Modal scale-to-zero + short holds. Track in `costs.md` (local).

---

## 3. Phases

### Phase 0 — Foundations & scaffold  ✅ (this session)
- Read `agent-prompt.md`; write this plan; init `story.md`; scaffold repo; private git repo.
- **Prerequisites handed to Harman** (Modal auth, model access, venv). *(see §5)*
- **Gate:** repo structure matches the brief's "What to submit"; plan + story committed.

### Phase 1 — The timing harness (`bench/`)  ← build this *before* collecting any number
- `bench/timer.py`: `benchmark()` doing sync + warmup + N-repeat + spread; CUDA-Event + wall clocks; separate prefill/decode.
- `bench/gpu_info.py`: capture GPU name, VRAM, clocks, temp, theoretical FP32 peak, driver, torch/CUDA versions → stamped into every results JSON.
- `bench/stats.py`: mean/std/min/median/max/p95/coefficient-of-variation.
- `bench/token_utils.py`: token-position accounting (cached vs uncached arithmetic) + token-equality → (match | first-divergence index + logit-gap context).
- `bench/plotting.py`, `bench/request_gen.py`: added later (Exp-1 plots; Exp-3/4 arrival generator) — not needed for the gauntlet.
- **Deliverable:** `bench/` + a paragraph draft for the report on *how I convinced myself the timings are real*.
- **Gate — the Phase-1 validation gauntlet** (`bench/gauntlet.py`, documented in [`bench/GAUNTLET.md`](bench/GAUNTLET.md)).
  A measurement instrument is trusted by calibration against ground truths it cannot fake, not by looking right.
  Must pass **before** any experiment number is collected:

  | ID | Test | Ground truth known a priori | Hard-fail if |
  |----|------|-----------------------------|--------------|
  | G1 | Deterministic unit tests (CPU) | token-position formula, equality-diff index, stats math have closed-form answers | measured ≠ golden |
  | G2 | Sync-lie detection | timing a big matmul *without* `cuda.synchronize` under-reports vs *with* | sync ≈ no-sync (our sync is a no-op) |
  | G3 | Warm-up outlier | call #1 carries autotune/alloc/cuBLAS-init overhead | first call *faster* than steady state |
  | G4 | Two-clock cross-check | `cuda.Event` and post-sync `perf_counter` must agree | disagree beyond tolerance |
  | G5 | Roofline upper bound | achieved FP32 TFLOP/s **cannot exceed the GPU's spec peak** (TF32 off) | measured > theoretical peak |
  | G6 | Compute linearity | size-2N matmul ≈ 8× size-N (O(n³)) | ratio outside tolerance band |
  | G7 | Reproducibility / spread | same config → small coefficient of variation | CV above threshold |
  | G8 | Prefill/decode split (real model) | prefill scales with prompt length; per-decode-step ≈ flat and ≪ prefill | timers cross-contaminate (step-1 outlier) |

  Runner prints PASS/FAIL + measured numbers, writes `results/phase1_gauntlet.json`, exits non-zero on any hard-fail.
  The gauntlet **is** the report's "how I trust the timings" paragraph, with receipts. Re-runnable on the Modal L4.

### Phase 2 — Experiment 1: KV cache (`exp1_kvcache/`)
- Two decode loops over the **same** model:
  - **cached:** keep KV, feed one token/step.
  - **uncached:** cache disabled, re-pass the whole growing sequence every step.
- **The main event — sweep batch = 1, 2, 4, 8, 16.** Plot speedup vs batch size.
- Report per loop: generated text (both) + match ✓; total wall-clock + tok/s; **per-step time vs step index**
  (both loops, one chart); **token positions computed** (the arithmetic each loop did).
- **Expected finding:** at batch 1 the uncached loop does ~60× more arithmetic yet finishes *near the same
  time* (memory-bound, arithmetic units idle); the gap only opens as batch grows and arithmetic becomes the wall.
- **Sanity trap:** a *large* batch-1 speedup ⇒ bug in the cached loop. Check before celebrating.
- **Deliverable:** `results/exp1.json` (raw enough to redraw every plot) + the two plots.

### Phase 3 — Experiment 2: Speculative decoding (`exp2_specdec/`)  ← spend the most time here
- Target alone vs draft-proposes-k / target-verifies-in-one-pass. **Greedy**, same answer both ways.
- Report: tok/s both ways + speedup; **acceptance rate**; **tokens committed per target pass** + total target
  passes each way; **token-coloured output** (draft-accepted vs target-correction vs free bonus token); sweep
  **k = 1..8**; identical-output check (or exact first divergence).
- **Four prompt types in one table:** copy a passage · write code · explain a concept · write something creative.
  Mechanism identical, payoff differs — explaining *why* is the point.
- **The trap (expect a slowdown first):** measure each model's per-step time; compute **c = draft-step /
  target-step**; predict `speedup = tokens_per_target_pass / (1 + k·c)`. If c is large, no acceptance rate
  saves you. Compare step times to **layer counts, not params** — if the ratio tracks layers, the draft is
  **kernel-launch-overhead-bound** (idling the GPU between tiny kernels), not compute/memory-bound.
  Fix with **CUDA graphs / `torch.compile(mode="reduce-overhead")` over a static cache** on the draft; the
  target won't move (it's genuinely bandwidth-bound). Report measured c, step times, and what I did to lower c.
- **Exactness:** bf16 may flip a single near-tie token. Before calling it a bug, check the **top-2 logit gap**
  at that position vs the median gap. A ~0.0 gap = tie resolved by float reduction order, not the model.
- **Deliverable:** `results/exp2.json` + k-sweep plots + prompt-type table + coloured-token render.

### Phase 4 — Experiment 3: Continuous batching (`exp3_batching/`)
- **Static (naive, I write it):** collect B requests, run as one batch, admit **no one** until *every*
  sequence finishes. Short requests sit completed in their slot while the longest runs on.
- **Continuous (real):** evict a finished sequence, admit a waiting one into that slot next step. Implement
  myself (better exercise) and/or compare to vLLM.
- Report for both, at **3 arrival rates × ≥2 batch sizes**: throughput (req/s **and** tok/s); latency
  (median, p95, **TTFT**); **slot-utilisation over time** (fraction doing useful work); **queue depth over time**.
- **Finding to look for:** static loses by being **idle** — utilisation plot drains to near-nothing while one
  long sequence finishes; continuous holds slots full. Quantify the throughput cost at each arrival rate.
- **Deliverable:** `results/exp3.json` + utilisation + queue-depth plots.

### Phase 5 — Experiment 4: PagedAttention (`exp4_paged/`)  ← about memory; time follows memory
- **Naive (I write it):** one contiguous KV block per sequence, sized for the server's **max** seq length,
  reserved at admission.
- **Paged (vLLM):** fixed-size blocks from a shared pool, allocated as tokens are generated, block table maps
  logical→physical.
- **Arithmetic, explicit in the report:** `KV_bytes/token = 2 · layers · kv_heads · head_dim · bytes`.
  **State kv_heads** (GQA ⇒ ≪ attention heads; getting it wrong is the #1 error). Predict memory, then confirm.
- Report: concurrent sequences that fit **both ways** (compute from real KV footprint, then admit-until-OOM to
  confirm); **wasted KV %** (reserved-but-never-written, measured against a realistic length distribution, not
  fixed length); throughput at that concurrency (more resident sequences ⇒ larger effective batch ⇒ speed);
  **block-size sweep** (small blocks waste less, cost more bookkeeping — show the curve).
- **Deliverable:** `results/exp4.json` + block-size curve + memory-fit table.

### Phase 6 — Live demo, report, packaging (`site/`, `report.pdf`, `README.md`)
- **Modal endpoints** for Exp-1 (batch-size control → text + tok/s both modes) and Exp-2 (stream tokens
  coloured by producer + acceptance rate + both throughputs). Exp 3 & 4 stay **offline** (holding a big GPU
  under sustained load for a visitor isn't reasonable to deploy).
- **Vercel site** wired to those endpoints; phone-responsive; button-press live runs.
- **report.pdf (≤2 pages):** one table (all 4 experiments + headline numbers), four best plots, per-experiment
  one-sentence expected vs one-sentence measured, closing single measurement that changed my thinking. Plus the
  "how I trust the timings" paragraph and hardware/spend.
- **README.md:** reproduce every number, in order.
- **results/**: one JSON per experiment, raw enough to redraw every plot.
- **Gate:** fresh-clone reproduce check; all plots labelled; token-equality reported in each section.

---

## 4. Repo structure (the submission)

```
bench/           harness: timing, gpu_info, request generator, plotting, token-equality
exp1_kvcache/    both decode loops, batch sweep, plots
exp2_specdec/    speculative loop, per-token tags, k sweep, 4-prompt table, c-diagnosis
exp3_batching/   static (mine) + continuous side, load results
exp4_paged/      naive allocator (mine) + paged side, memory & block-size results
results/         one JSON per experiment (raw enough to redraw every plot)
site/            deployed live demo (exp 1 & 2)
report.pdf       ≤ 2 pages
README.md        reproduce every number, in order
```
Process/meta files (`plan.md`, `story.md`, `agent-prompt.md`, `costs.md`) live in the repo locally but may be
gitignored from the minimal public submission later — decided at Phase 6, mirroring Assignment 3.

---

## 5. Prerequisites from Harman (blockers before Phase 1 build)

See the message accompanying this plan. In short: Modal account + CLI/venv, model access (Qwen pair + the
125M/A1 model), confirm GPU choice (L4 vs A10G) and spend ceiling, and Vercel for the demo (Phase 6).

## 6. Decisions — CONFIRMED (2026-08-17)

- Exp-1 model: **125M (Session 2)**. ✅
- Exp-3/4 model: **Qwen2.5-0.5B-Instruct** (GQA — feeds Exp-4's kv_heads arithmetic). ✅
- Continuous-batching "ON" side: **implement myself + vLLM cross-check**. ✅
- Repo name: **`slm-inference-benchmarks`** (private, on GitHub). ✅

**Still pending from Harman (hard blockers for Phase 1+):** Modal account/CLI + venv choice;
GPU (L4 vs A10G) + spend ceiling; Vercel confirmation (Phase 6 only).
