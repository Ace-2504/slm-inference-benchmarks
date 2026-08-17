# The Phase-1 Validation Gauntlet

> **Why this exists.** The brief's graders look first at *"whether your timings survive scrutiny:
> synchronised, warmed up, repeated, and reported with their spread."* A timing harness is a
> **measurement instrument** — you don't trust it because the code looks right, you trust it because it
> is **calibrated against ground truths it cannot fake.** This gauntlet is that calibration. It must
> pass **before** a single experiment number is collected, and it doubles as the report's required
> paragraph on *how I convinced myself the timings are real* — with receipts.

Run it:

```bash
python bench/gauntlet.py
```

Output: a PASS/FAIL table with the measured numbers, and `results/phase1_gauntlet.json` (raw, stamped
with GPU name / clocks / temp / driver / torch+CUDA versions). The process exits non-zero if any
**hard** test fails.

---

## The two failure classes we defend against

1. **The timer lies.** CUDA is asynchronous — an untimed kernel launch returns in microseconds while the
   work takes milliseconds on the GPU. Without `torch.cuda.synchronize()` we would "measure" the launch,
   not the compute. This is the single most common way GPU benchmarks report fiction.
2. **The discipline is misapplied.** Warm-up skipped (timing autotune/compile), prefill leaking into the
   decode timer, too few repeats, or noisy/throttling clocks.

Trust is earned in three layers: **(A)** conventional golden-value unit tests for the deterministic parts,
**(B)** calibration/sanity tests that check the timer against known physics/arithmetic, **(C)** environmental
controls recorded alongside every number.

---

## The tests

### G1 — Deterministic unit tests (CPU, golden values)
Parts of the harness are pure logic with a *known correct answer*, so we test them like any code.
- **Token-position accounting** — the "arithmetic each loop did" the brief asks for. Closed form:
  - cached loop: `prompt_len + gen_len`
  - uncached loop: `Σ_{t=1..gen_len}(prompt_len + t − 1) = gen_len·prompt_len + gen_len(gen_len−1)/2`
  - Sanity anchor from the brief's lab (prompt 587, gen 64): cached `= 651`, uncached `= 39 584`,
    ratio `≈ 60.8×` (the brief says "61× more token positions"). We assert exactly these.
- **Token-equality checker** — identical sequences → `(match, None)`; sequences differing at a known
  index `k` → `(mismatch, k)`.
- **Stats math** — mean / std / median / p95 asserted against hand-computed values.

**Hard-fail:** any measured value ≠ its golden value.

### G2 — Sync-lie detection (GPU)
Time a large matmul with `perf_counter` **without** a trailing `synchronize`, then **with** it. The synced
time must be dramatically larger — that gap *is* the asynchronous work the un-synced timer would have
hidden. Proves our synchronization is real and necessary.

**Hard-fail:** synced time is not meaningfully larger than un-synced (our sync is a no-op).

### G3 — Warm-up outlier (GPU)
Run the same matmul N times individually. Call #1 carries one-time costs — cuBLAS handle creation, workspace
allocation, algorithm selection. It must be an outlier; steady-state is what we report.

**Hard-fail:** the first call is *faster* than the steady-state median (impossible if warm-up is real →
means we mis-measured). We also report the observed first-call overhead.

### G4 — Two-clock cross-check (GPU)
Measure the same op two independent ways: `torch.cuda.Event.elapsed_time` (device-side) and post-sync
`perf_counter` (host-side). Two independent instruments must agree within tolerance — they cannot lie the
same way by accident.

**Hard-fail:** the two clocks disagree beyond tolerance (default 25%).

### G5 — Roofline upper bound (GPU)  ← the strongest test
For a matmul we know the exact FLOP count (`2·m·n·k`). Divide by measured time → achieved FP32 TFLOP/s.
This **cannot exceed the GPU's theoretical FP32 peak** (`SMs · cores-per-SM · 2 · clock`). We disable TF32
so the FP32 matmul really runs on FP32 CUDA cores, making the comparison honest. A measured number above
hardware peak is *proof* the timer is lying (a missed sync). We also report achieved-as-%-of-peak.

**Hard-fail:** achieved TFLOP/s > theoretical peak.

### G6 — Compute linearity (GPU)
A size-`2N` square matmul does `8×` the FLOPs of size-`N` (O(n³)). If the instrument is linear, the measured
time ratio tracks it. A ratio far from 8× means overhead is dominating or the clock is unstable.

**Hard-fail:** measured ratio outside a tolerance band (default 4×–12×).

### G7 — Reproducibility / spread (GPU)
Run one fixed config many times; the coefficient of variation (std/mean) must be small. High variance means
a noisy environment (thermal throttling, background load, clock boosting) and tells us to lock clocks or add
repeats. This is *why* we always report spread, never a single number.

**Hard-fail:** CV above threshold (default 15% on the local dev box; tighter on the L4).

### G8 — Prefill/decode split (real model)
Load a small real model and time the two phases separately, the same way the experiments will. Prefill (one
forward over the whole prompt) must scale up with prompt length; each decode step (one token with the KV
cache) must be roughly flat across steps and far cheaper than prefill. If the decode timer is accidentally
including prefill, step 1 shows up as a huge outlier — this test catches that cross-contamination.

**Hard-fail:** per-decode-step time is not ≪ prefill, or step 1 is a gross outlier vs later steps.
**Skips gracefully** (marked SKIPPED, not failed) if the model can't be loaded offline.

---

## Environmental controls (layer C — recorded, not asserted)
Stamped into `results/phase1_gauntlet.json` so every downstream number is traceable:
GPU name, total VRAM, compute capability, SM count, **max & current SM clock**, **temperature**, driver
version, torch + CUDA versions, and the theoretical FP32 peak used by G5. Experiments use greedy + fixed
seed so token counts are deterministic and tok/s is apples-to-apples.

## Scope note
The gauntlet validates the **instrument** and runs on the local **RTX 3060** for fast iteration. It is
**re-runnable on the Modal L4** (same command) so the numbers we actually report are validated on the same
hardware that produced them. Tolerances for G7 may be tightened on the L4 (a quieter, dedicated GPU).
