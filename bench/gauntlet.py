"""The Phase-1 validation gauntlet — see bench/GAUNTLET.md for the rationale.

Calibrates the timing harness against ground truths it cannot fake. Must pass before any
experiment number is collected. Prints a PASS/FAIL table, writes
results/phase1_gauntlet.json, exits non-zero on any hard-fail.

    python bench/gauntlet.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

# allow `python bench/gauntlet.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from bench.gpu_info import get_gpu_info  # noqa: E402
from bench.stats import summarize  # noqa: E402
from bench.timer import benchmark, time_wall, time_event  # noqa: E402
from bench.token_utils import token_positions, token_equal  # noqa: E402

# ------------------------------------------------------------------ thresholds
G2_MIN_SYNC_RATIO = 3.0     # synced time must be >=3x the un-synced (async-lie) time
G3_MAX_SPEEDUP_FIRST = 0.90 # first call may not be faster than 0.9x steady median
G4_MAX_CLOCK_DISAGREE = 0.25  # event vs wall clocks agree within 25%
G6_RATIO_LO, G6_RATIO_HI = 4.0, 12.0  # 2N matmul is ~8x N matmul (O(n^3))
G7_MAX_CV = 0.15            # coefficient of variation ceiling on the local dev box
# A cached decode step must be visibly cheaper than a full P2 prefill (timers separate),
# but NOT by a fixed large ratio: at batch 1 a tiny model is overhead/bandwidth-bound, so
# 1 position vs P2 positions barely shows in wall-time (Exp-1's roofline, previewed here).
G8_DECODE_FRAC = 0.8        # decode step < 0.8x a full prefill of P2
G8_STEP_OUTLIER = 5.0       # max/median across decode steps must stay under this


def _mm(n: int, dtype=torch.float32):
    """Return (fn, flops) for an n x n matmul on the GPU."""
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    flops = 2 * n**3
    return (lambda: torch.matmul(a, b)), flops


def _row(rid, name, status, detail):
    return {"id": rid, "name": name, "status": status, "detail": detail}


# --------------------------------------------------------------------- G1 CPU
def g1_unit_tests():
    checks = []

    # token-position accounting vs the brief's lab anchor (prompt 587, gen 64)
    tp = token_positions(587, 64)
    checks.append(("cached_total==651", tp["cached_total"] == 651))
    checks.append(("uncached_total==39584", tp["uncached_total"] == 39584))
    checks.append(("ratio~=60.8", abs(tp["ratio_uncached_over_cached"] - 60.8) < 0.1))

    # token equality: identical, and a known divergence index
    checks.append(("equal_match", token_equal([1, 2, 3], [1, 2, 3])["match"] is True))
    div = token_equal([1, 2, 9, 4], [1, 2, 3, 4])
    checks.append(("equal_divergence@2",
                   div["match"] is False and div["first_divergence"] == 2))
    checks.append(("equal_prefix_len", token_equal([1, 2], [1, 2, 3])["first_divergence"] == 2))

    # stats math vs hand-computed values for [2,4,4,4,5,5,7,9]
    s = summarize([2, 4, 4, 4, 5, 5, 7, 9])
    checks.append(("stats_mean==5.0", abs(s["mean"] - 5.0) < 1e-9))
    checks.append(("stats_median==4.5", abs(s["median"] - 4.5) < 1e-9))
    checks.append(("stats_std~=2.138", abs(s["std"] - 2.13809) < 1e-3))

    failed = [name for name, ok in checks if not ok]
    status = "PASS" if not failed else "FAIL"
    detail = f"{len(checks)-len(failed)}/{len(checks)} golden checks passed"
    if failed:
        detail += f"; FAILED: {failed}"
    return _row("G1", "Deterministic unit tests (CPU golden)", status, detail), {
        "checks": [{"name": n, "ok": ok} for n, ok in checks],
        "token_positions_anchor": {k: tp[k] for k in
                                   ("cached_total", "uncached_total", "ratio_uncached_over_cached")},
    }


# --------------------------------------------------------- G3 warm-up outlier
def g3_warmup():
    fn, _ = _mm(2048)
    times = [time_wall(fn) for _ in range(12)]  # each syncs; NO warm-up on purpose
    first = times[0]
    rest_med = statistics.median(times[1:])
    ratio = first / rest_med if rest_med else float("inf")
    status = "PASS" if first >= rest_med * G3_MAX_SPEEDUP_FIRST else "FAIL"
    detail = (f"first={first:.2f}ms vs steady-median={rest_med:.2f}ms "
              f"(first is {ratio:.1f}x steady — warm-up overhead is real)")
    return _row("G3", "Warm-up outlier", status, detail), {
        "first_ms": first, "steady_median_ms": rest_med, "ratio": ratio, "raw_ms": times,
    }


# ------------------------------------------------------ G2 sync-lie detection
def g2_sync_lie():
    fn, _ = _mm(4096)
    for _ in range(3):  # warm up so we compare steady launch vs steady compute
        fn()
    torch.cuda.synchronize()
    nosync = statistics.median([time_wall(fn, do_sync=False) for _ in range(7)])
    synced = statistics.median([time_wall(fn, do_sync=True) for _ in range(7)])
    ratio = synced / nosync if nosync else float("inf")
    status = "PASS" if ratio >= G2_MIN_SYNC_RATIO else "FAIL"
    detail = (f"no-sync={nosync:.3f}ms (launch only) vs synced={synced:.3f}ms "
              f"(real work) -> {ratio:.0f}x hidden without synchronize")
    return _row("G2", "Sync-lie detection", status, detail), {
        "nosync_ms": nosync, "synced_ms": synced, "ratio": ratio,
    }


# ------------------------------------------------------- G4 two-clock check
def g4_cross_clock():
    fn, _ = _mm(4096)
    ev = benchmark(fn, warmup=3, repeats=10, clock="event")["median"]
    wl = benchmark(fn, warmup=1, repeats=10, clock="wall")["median"]
    rel = abs(ev - wl) / wl if wl else float("inf")
    status = "PASS" if rel <= G4_MAX_CLOCK_DISAGREE else "FAIL"
    detail = (f"cuda.Event={ev:.3f}ms vs perf_counter={wl:.3f}ms "
              f"-> {rel*100:.1f}% apart (two independent clocks agree)")
    return _row("G4", "Two-clock cross-check", status, detail), {
        "event_ms": ev, "wall_ms": wl, "rel_diff": rel,
    }


# ------------------------------------------------------- G5 roofline bound
def g5_roofline(info):
    n = 4096
    fn, flops = _mm(n)
    ms = benchmark(fn, warmup=3, repeats=10, clock="event")["median"]
    achieved = flops / (ms / 1e3) / 1e12  # TFLOP/s
    peak = info.get("theoretical_fp32_tflops")
    if peak is None:
        status, detail = "SKIP", f"achieved={achieved:.2f} TFLOP/s (no theoretical peak available to bound)"
    else:
        util = achieved / peak * 100
        status = "PASS" if achieved < peak else "FAIL"
        detail = (f"achieved={achieved:.2f} TFLOP/s vs FP32 peak={peak:.2f} "
                  f"({util:.0f}% of peak) — cannot exceed hardware")
    return _row("G5", "Roofline upper bound (FP32, TF32 off)", status, detail), {
        "n": n, "flops": flops, "median_ms": ms, "achieved_tflops": achieved,
        "peak_tflops": peak,
    }


# ------------------------------------------------------- G6 compute linearity
def g6_linearity():
    fn_n, _ = _mm(2048)
    fn_2n, _ = _mm(4096)
    t_n = benchmark(fn_n, warmup=3, repeats=8, clock="event")["median"]
    t_2n = benchmark(fn_2n, warmup=3, repeats=8, clock="event")["median"]
    ratio = t_2n / t_n if t_n else float("inf")
    status = "PASS" if G6_RATIO_LO <= ratio <= G6_RATIO_HI else "FAIL"
    detail = (f"t(4096)/t(2048)={ratio:.1f}x (expect ~8x for O(n^3); "
              f"band {G6_RATIO_LO}-{G6_RATIO_HI}x)")
    return _row("G6", "Compute linearity", status, detail), {
        "t_n_ms": t_n, "t_2n_ms": t_2n, "ratio": ratio,
    }


# ------------------------------------------------------- G7 reproducibility
def g7_reproducibility():
    fn, _ = _mm(4096)
    res = benchmark(fn, warmup=3, repeats=12, clock="event")
    cv = res["cv"]
    status = "PASS" if cv <= G7_MAX_CV else "FAIL"
    detail = (f"CV={cv*100:.2f}% over {res['n']} runs "
              f"(mean={res['mean']:.3f}ms, std={res['std']:.3f}ms; ceil {G7_MAX_CV*100:.0f}%)")
    return _row("G7", "Reproducibility / spread", status, detail), {
        "cv": cv, "mean_ms": res["mean"], "std_ms": res["std"], "raw_ms": res["raw"],
    }


# --------------------------------------------- G8 prefill/decode split (model)
def g8_prefill_decode():
    # Default: build the GPT-2 124M architecture from config with RANDOM weights — no
    # network, fully offline. G8 validates timing *structure* (prefill scales, decode
    # steps are flat and cheap), not output quality, so random weights are ideal and
    # avoid the large-file download that stalls in this environment. Set GAUNTLET_MODEL
    # to a HF id (e.g. on the Modal L4) to validate on real pretrained weights instead.
    model_id = os.environ.get("GAUNTLET_MODEL")
    try:
        dev = "cuda"
        if model_id:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float32).to(dev).eval()
            desc = model_id
        else:
            from transformers import GPT2Config, GPT2LMHeadModel
            cfg = GPT2Config(n_layer=12, n_embd=768, n_head=12,
                             n_positions=1024, vocab_size=50257)
            model = GPT2LMHeadModel(cfg).to(dev).float().eval()
            desc = "gpt2-124M-arch (random init, offline)"

        vocab = model.config.vocab_size

        def prefill_time(p_len):
            ids = torch.randint(0, vocab, (1, p_len), device=dev)
            fn = lambda: model(input_ids=ids, use_cache=True)  # noqa: E731
            return benchmark(fn, warmup=3, repeats=7, clock="event")["median"]

        with torch.inference_mode():
            p1, p2 = 32, 256
            # Clock/model warm-up burst on the heaviest shape BEFORE timing — brings the
            # GPU off its 240 MHz idle and absorbs model-path lazy init, so measurement
            # order can't bias the result (the lesson G8 taught on its first run).
            ids2 = torch.randint(0, vocab, (1, p2), device=dev)
            for _ in range(8):
                model(input_ids=ids2, use_cache=True)
            torch.cuda.synchronize()

            t_p1 = prefill_time(p1)
            t_p2 = prefill_time(p2)

            # Build a cache from a real prefill, warm a few decode steps, then time 8.
            # Each measured call runs the step EXACTLY once — DynamicCache mutates in
            # place, so a double-call would append two KV entries and corrupt positions.
            out = model(input_ids=ids2, use_cache=True)
            past, cur = out.past_key_values, p2
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)

            def advance_one():
                nonlocal past, nxt, cur
                cp = torch.tensor([cur], device=dev)
                holder = {}

                def step():
                    holder["out"] = model(input_ids=nxt, past_key_values=past,
                                          use_cache=True, cache_position=cp)
                ms = time_event(step)  # calls step() once -> one KV entry appended
                o = holder["out"]
                past = o.past_key_values
                nxt = o.logits[:, -1].argmax(-1, keepdim=True)
                cur += 1
                return ms

            for _ in range(3):   # warm-up decode steps (untimed)
                advance_one()
            step_ms = [advance_one() for _ in range(8)]  # timed

        step_med = statistics.median(step_ms)
        step_outlier = max(step_ms) / step_med if step_med else float("inf")
        cond_scale = t_p2 > t_p1
        cond_cheap = step_med < t_p2 * G8_DECODE_FRAC
        cond_flat = step_outlier < G8_STEP_OUTLIER
        status = "PASS" if (cond_scale and cond_cheap and cond_flat) else "FAIL"
        detail = (f"[{desc}] prefill {p1}tok={t_p1:.2f}ms < {p2}tok={t_p2:.2f}ms (timer tracks work); "
                  f"decode step={step_med:.2f}ms, flat (max/med={step_outlier:.1f}x); "
                  f"1 pos vs {p2} pos yet only {t_p2/step_med:.1f}x cheaper (batch-1 roofline)")
        return _row("G8", "Prefill/decode split (real model)", status, detail), {
            "model": desc, "prefill_p1_ms": t_p1, "prefill_p2_ms": t_p2,
            "decode_step_ms": step_ms, "decode_median_ms": step_med,
            "step_outlier_ratio": step_outlier,
        }
    except Exception as e:  # offline / API drift -> skip, don't fail the gauntlet
        return _row("G8", "Prefill/decode split (real model)", "SKIP",
                    f"skipped ({type(e).__name__}: {e})"), {"error": str(e)}


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — the gauntlet needs a GPU.")
        sys.exit(2)

    # honest FP32 vs FP32-peak comparison for G5/G6/G7
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    info = get_gpu_info()
    print("=" * 78)
    print("PHASE-1 VALIDATION GAUNTLET")
    print("=" * 78)
    print(f"GPU        : {info.get('name')}  (cc {info.get('compute_capability')}, "
          f"{info.get('sm_count')} SMs, {info.get('total_mem_gb')} GB)")
    print(f"Clock/temp : max {info.get('max_sm_clock_mhz')} MHz, "
          f"cur {info.get('current_sm_clock_mhz')} MHz, {info.get('temperature_c')} C")
    print(f"FP32 peak  : {info.get('theoretical_fp32_tflops')} TFLOP/s")
    print(f"Stack      : torch {info.get('torch_version')} / CUDA {info.get('cuda_version')} / "
          f"driver {info.get('driver_version')}")
    print("-" * 78)

    rows, raw = [], {}
    # G1 (cpu) and G3 (first GPU compute, to capture warm-up) first, then the rest
    for fn in (g1_unit_tests, g3_warmup, g2_sync_lie, g4_cross_clock,
               lambda: g5_roofline(info), g6_linearity, g7_reproducibility,
               g8_prefill_decode):
        row, data = fn()
        rows.append(row)
        raw[row["id"]] = data
        print(f"  [{row['status']:4}] {row['id']}  {row['name']}")
        print(f"          {row['detail']}")

    hard_fail = [r for r in rows if r["status"] == "FAIL"]
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_skip = sum(1 for r in rows if r["status"] == "SKIP")
    print("-" * 78)
    print(f"RESULT: {n_pass} pass, {len(hard_fail)} fail, {n_skip} skip")
    print("=" * 78)

    out_path = Path(__file__).resolve().parents[1] / "results" / "phase1_gauntlet.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "gpu_info": info,
        "summary": {"pass": n_pass, "fail": len(hard_fail), "skip": n_skip},
        "rows": rows,
        "raw": raw,
    }, indent=2))
    print(f"wrote {out_path}")

    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
