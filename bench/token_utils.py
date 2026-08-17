"""Token-position accounting and token-equality — the deterministic, golden-testable
core of the harness (gauntlet G1).

token_positions:  the "arithmetic each decode loop performed" the brief asks Exp-1 to
report — how many token positions the model actually pushed through, cached vs uncached.

token_equal:  the token-by-token equality gate that must pass before any speedup is
claimed. Returns the first divergence index (and, if logits are supplied, the top-2
logit gap there — the bf16 near-tie diagnostic Exp-2 needs).
"""
from __future__ import annotations

from typing import Optional, Sequence


def token_positions(prompt_len: int, gen_len: int) -> dict:
    """Token positions computed by each decode loop generating `gen_len` tokens after a
    `prompt_len`-token prompt.

    cached:   prefill processes `prompt_len` positions, then each of `gen_len` decode
              steps processes exactly 1 (the new token, attending to cached KV).
              total = prompt_len + gen_len
    uncached: no cache — to produce the t-th new token the model re-processes the whole
              current sequence of length (prompt_len + t - 1).
              total = sum_{t=1..gen_len}(prompt_len + t - 1)
                    = gen_len*prompt_len + gen_len*(gen_len-1)/2
    """
    if prompt_len < 0 or gen_len < 0:
        raise ValueError("lengths must be non-negative")
    cached_total = prompt_len + gen_len
    cached_per_step = [prompt_len] + [1] * gen_len  # [prefill, then one per decode step]
    uncached_per_step = [prompt_len + t - 1 for t in range(1, gen_len + 1)]
    uncached_total = sum(uncached_per_step)
    ratio = (uncached_total / cached_total) if cached_total else float("nan")
    return {
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "cached_total": cached_total,
        "uncached_total": uncached_total,
        "ratio_uncached_over_cached": ratio,
        "cached_per_step": cached_per_step,
        "uncached_per_step": uncached_per_step,
    }


def token_equal(
    a: Sequence[int],
    b: Sequence[int],
    logits_at_divergence: Optional[Sequence[float]] = None,
) -> dict:
    """Compare two token id sequences.

    Returns match=True iff identical. On mismatch, first_divergence is the index of the
    first differing position (or the shorter length if one is a prefix of the other).
    If `logits_at_divergence` (the loser side's logit vector at that step) is supplied,
    also reports the top-2 gap there — a ~0 gap flags a near-tie resolved by float
    reduction order rather than a real disagreement (Exp-2's exactness check).
    """
    n = min(len(a), len(b))
    first = None
    for i in range(n):
        if a[i] != b[i]:
            first = i
            break
    if first is None and len(a) != len(b):
        first = n  # identical up to the shorter length, then one ran longer

    result = {
        "match": first is None,
        "first_divergence": first,
        "len_a": len(a),
        "len_b": len(b),
    }
    if first is not None and logits_at_divergence is not None:
        srt = sorted(logits_at_divergence, reverse=True)
        result["top2_logit_gap"] = (srt[0] - srt[1]) if len(srt) >= 2 else float("inf")
    return result
