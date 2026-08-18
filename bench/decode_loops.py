"""The two decode loops at the heart of Experiment 1 — the same model, greedy, one with
the KV cache and one without. Shared by the experiment runner and the live demo so both
sides are literally the same code path (brief: "change one thing").

greedy_cached:   prefill once, then feed ONE token per step attending to cached KV.
greedy_uncached: no cache — re-pass the whole growing sequence through the model every
                 step (the naive side we implement to measure what the cache is worth).

Both are batched and greedy, so for the same input they MUST emit identical tokens; the
caller checks that with bench.token_utils.token_equal before reporting any speedup.
"""
from __future__ import annotations

import torch

from .timer import time_event


@torch.inference_mode()
def greedy_cached(model, input_ids, max_new_tokens, record_steps=False):
    """Generate greedily using past_key_values. Returns (generated_ids, per_step_ms).

    per_step_ms holds the decode-step times (one per generated token after the first)
    when record_steps=True, else []. Prefill is timed separately by the caller.
    """
    dev = input_ids.device
    # logits_to_keep=1: greedy needs only the last position's logits. This does NOT
    # change the attention arithmetic (the roofline story) — it just avoids materializing
    # a huge (batch, seq, vocab) logits tensor every step, which OOM-thrashes at batch 16.
    out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    cur = input_ids.shape[1]
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    gen = [nxt]
    per_step = []

    for _ in range(max_new_tokens - 1):
        cp = torch.tensor([cur], device=dev)
        holder = {}

        def step():
            holder["o"] = model(input_ids=nxt, past_key_values=past,
                                use_cache=True, cache_position=cp, logits_to_keep=1)

        if record_steps:
            per_step.append(time_event(step))  # calls step() exactly once
        else:
            step()
        o = holder["o"]
        past = o.past_key_values
        nxt = o.logits[:, -1].argmax(-1, keepdim=True)
        gen.append(nxt)
        cur += 1

    return torch.cat(gen, dim=1), per_step


@torch.inference_mode()
def greedy_uncached(model, input_ids, max_new_tokens, record_steps=False):
    """Generate greedily WITHOUT a cache: every step re-processes the full sequence.

    Returns (generated_ids, per_step_ms). This is the naive baseline — at step t it
    computes over (prompt_len + t - 1) positions, so it performs far more arithmetic
    than the cached loop (quantified by bench.token_utils.token_positions).
    """
    seq = input_ids
    gen = []
    per_step = []

    for _ in range(max_new_tokens):
        holder = {}

        def step():
            # logits_to_keep=1 on BOTH loops so the ONLY difference is the KV cache itself
            # (greedy needs only the last logit). Without it, the uncached loop also pays a
            # full (batch, seq, vocab) lm_head every step — extra work the cached loop skips,
            # which both inflates the measured speedup and OOMs at high batch/context.
            holder["o"] = model(input_ids=seq, use_cache=False, logits_to_keep=1)

        if record_steps:
            per_step.append(time_event(step))
        else:
            step()
        o = holder["o"]
        nxt = o.logits[:, -1].argmax(-1, keepdim=True)
        gen.append(nxt)
        seq = torch.cat([seq, nxt], dim=1)

    return torch.cat(gen, dim=1), per_step
