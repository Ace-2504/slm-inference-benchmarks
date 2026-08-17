"""Greedy speculative decoding — the draft proposes k tokens, the target verifies them
in ONE forward pass, and the two together are exact: in exact arithmetic the committed
sequence equals plain target-alone greedy decoding (Exp-2 checks this and, on a bf16
near-tie, reports the first divergence and its top-2 logit gap).

Cache bookkeeping (the fiddly, viva-relevant part):
  * Both models keep a DynamicCache. After prefill we crop both to prompt_len-1 and carry
    the last prompt token as `pending`, so EVERY round has p>=1 committed-but-uncached
    tokens and the indexing is uniform (no first-round special case).
  * Each round the target verify pass feeds `pending + draft_toks`. Its logit at feed
    index (p-1+j) predicts draft token j; index (p-1+k) is the free bonus token.
  * We accept the longest matching prefix A, crop both caches back to cache_len+p+A
    (discarding the KV of rejected drafts), and carry the target's token at position A
    (a correction if A<k, the bonus if A==k) as the next `pending`. Exactly ONE target
    forward pass per round.
"""
from __future__ import annotations

import torch


@torch.inference_mode()
def _prefill(model, input_ids):
    out = model(input_ids=input_ids, use_cache=True)
    return out.past_key_values


@torch.inference_mode()
def speculative_generate(target, draft, input_ids, max_new_tokens, k):
    """Return a dict with the committed tokens, per-token producer tags, and the counters
    Exp-2 reports (acceptance rate, verify passes, tokens/target-pass)."""
    dev = input_ids.device
    prompt_len = input_ids.shape[1]

    t_cache = _prefill(target, input_ids)
    d_cache = _prefill(draft, input_ids)
    # Reprocess the last prompt token inside the loop -> uniform p>=1 indexing.
    t_cache.crop(prompt_len - 1)
    d_cache.crop(prompt_len - 1)
    cache_len = prompt_len - 1
    pending = [int(input_ids[0, -1])]

    committed: list[int] = []
    tags: list[str] = []
    accepted_total = 0
    proposed_total = 0
    verify_passes = 0

    def feed(model, cache, toks, start):
        t = torch.tensor([toks], device=dev)
        cp = torch.arange(start, start + len(toks), device=dev)
        out = model(input_ids=t, past_key_values=cache, use_cache=True, cache_position=cp)
        return out.logits  # (1, len(toks), V)

    while len(committed) < max_new_tokens:
        p = len(pending)

        # 1) draft proposes k tokens greedily, extending the draft cache
        d_logits = feed(draft, d_cache, pending, cache_len)  # process pending
        logit = d_logits[:, -1]
        draft_toks = []
        for i in range(k):
            nxt = int(logit.argmax(-1))
            draft_toks.append(nxt)
            logit = feed(draft, d_cache, [nxt], cache_len + p + i)[:, -1]
        proposed_total += k

        # 2) target verifies pending+drafts in ONE pass
        t_logits = feed(target, t_cache, pending + draft_toks, cache_len)  # (1, p+k, V)
        verify_passes += 1
        t_pred = [int(t_logits[:, p - 1 + j].argmax(-1)) for j in range(k + 1)]

        # 3) accept the longest matching prefix
        A = 0
        while A < k and draft_toks[A] == t_pred[A]:
            A += 1
        accepted_total += A

        for j in range(A):
            committed.append(draft_toks[j]); tags.append("draft")
        correction = t_pred[A]  # correction if A<k, bonus if A==k
        committed.append(correction)
        tags.append("bonus" if A == k else "target")

        # 4) crop both caches to the accepted length; carry correction as next pending
        keep = cache_len + p + A
        t_cache.crop(keep)
        d_cache.crop(keep)
        cache_len = keep
        pending = [correction]

    committed = committed[:max_new_tokens]
    tags = tags[:max_new_tokens]
    return {
        "tokens": committed,
        "tags": tags,
        "k": k,
        "proposed": proposed_total,
        "accepted": accepted_total,
        "acceptance_rate": (accepted_total / proposed_total) if proposed_total else 0.0,
        "target_passes": verify_passes,
        "tokens_per_target_pass": len(committed) / verify_passes if verify_passes else 0.0,
    }


@torch.inference_mode()
def target_alone_greedy(target, input_ids, max_new_tokens):
    """Plain greedy decoding with the KV cache — the exactness reference and the
    'target passes = one per token' baseline."""
    dev = input_ids.device
    out = target(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    cur = input_ids.shape[1]
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    toks = [int(nxt)]
    for _ in range(max_new_tokens - 1):
        cp = torch.tensor([cur], device=dev)
        out = target(input_ids=nxt, past_key_values=past, use_cache=True, cache_position=cp)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        toks.append(int(nxt))
        cur += 1
    return toks[:max_new_tokens]
