# Inference Optimization Benchmarks

Measuring what each of four serving optimizations is actually worth — on real GPUs, with timings that
survive scrutiny. One inference pipeline, four controlled experiments, each toggling exactly one thing:

| # | Optimization | Question |
|---|--------------|----------|
| 1 | **KV cache** | When is recomputing the whole prefix actually free? (roofline) |
| 2 | **Speculative decoding** | Is a 15× smaller draft model actually a speedup? |
| 3 | **Continuous batching** | Does static batching lose by being *idle* rather than slow? |
| 4 | **PagedAttention** | How much KV memory is reserved but never written? |

**The rule across all four:** both sides must emit **identical tokens** before any speedup is claimed.

> 🚧 **Status:** planning stage. Structure scaffolded; experiments not yet built. See
> [`plan.md`](plan.md) for the end-to-end plan and [`story.md`](story.md) for the running log.

## Structure

```
bench/           timing harness, request generator, plotting, token-equality check
exp1_kvcache/    cached vs uncached decode loops, batch-size sweep
exp2_specdec/    speculative decoding, k-sweep, per-token tags, 4-prompt table
exp3_batching/   static (naive) vs continuous batching, load results
exp4_paged/      naive contiguous KV vs PagedAttention, memory & block-size results
results/         one JSON per experiment (raw enough to redraw every plot)
site/            deployed live demo (experiments 1 & 2)
report.pdf       ≤ 2 pages
```

## Reproduce

Full reproduce-in-order instructions land here as each phase completes.
