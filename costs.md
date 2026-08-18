# Spend log

All GPU work on **Modal L4** (~$1.10/hr ≈ $0.018/min, scale-to-zero). Phase-1 gauntlet ran on the
local RTX 3060 (free). Estimates from container wall-time; see the Modal dashboard for exact billing.

| Item | GPU-min (approx) | Cost (approx) |
|------|------------------|---------------|
| Phase 1 gauntlet | local 3060 | $0.00 |
| Modal connectivity probe | 1 | $0.02 |
| Exp 1 (KV cache) — image build + fp32 + bf16 + one stopped run | ~12 | $0.22 |
| Exp 2 smoke (7B download + run) | ~5 | $0.09 |
| Exp 2 full sweep (first, heavy, stopped at ~65 min) | ~65 | $1.19 |
| Exp 2 c-reduction probes (×2) | ~10 | $0.18 |
| Exp 2 full sweep (lighter re-run) | ~15 | $0.28 |
| Exp 3 (step-time measurement ×2) | ~6 | $0.11 |
| Exp 4 (naive allocator + OOM) | ~2 | $0.04 |
| Live demo (deploy build + test requests) | ~3 | $0.06 |
| **Total** | | **≈ $2.2** |

Well within the ~$15 ceiling. The single biggest line is the first Exp-2 sweep, which I stopped after
realizing `repeats=2` timing on every high-k config with a launch-bound draft was ~4× more work than
needed; the lighter re-run (time once per config, trust the gauntlet for the timer) is ~4× cheaper.
