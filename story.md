# story.md — the running log of this assignment

> **SELF-MAINTENANCE DIRECTIVE (permanent):** Claude MUST append to this file **every time** an
> experiment is run or an implementation/decision takes place — successes *and* missteps, whether
> the change left us **better or worse off** than before. Each entry records: what we did, why,
> the outcome, and the net effect on our position. This is the project's memory of *how* we got
> here, so a mentor (or a future session) can follow the reasoning, including dead ends. Keep it
> honest: record what was thrown away and what regressed, not only what worked.

**Project:** Assignment 4 — Inference optimization benchmarks. Build one inference pipeline + a
trustworthy timing harness, then measure what four serving tricks are each worth: (1) KV cache,
(2) speculative decoding, (3) continuous batching, (4) PagedAttention. Golden rule across all four:
both sides must emit **identical tokens** before any speedup is claimed.
**Started:** 2026-08-17. All entries below are 2026-08-17 unless noted.

---

## Phase 0 — Foundations

**E1 · Read the brief.** Parsed `agent-prompt.md`. Four controlled experiments, each the same model /
prompt / decoding with exactly one toggle flipped (cache, spec-decode, continuous batching, paging).
Non-negotiables: token-equality check every time; trustworthy timing (`cuda.synchronize`, warm-up,
≥3 repeats + spread, prefill timed apart from decode); report the GPU for every number; log spend;
**investigate** surprises rather than reporting them raw. Deliverables: `bench/` + four `expN_*` dirs +
`results/` JSON + `site/` (live Vercel demo of exp 1 & 2 on a Modal GPU) + `report.pdf` (≤2pp) + README.
The brief flags **Experiment 2 (speculative decoding)** as where to spend the most time — the obvious
measurement usually gives the wrong conclusion and the diagnosis (the `c = draft/target step-time` story,
CUDA-graphs fix, bf16 near-tie exactness) is the real work. → *Position: oriented.*

**E2 · Environment recon.** Checked local tooling before promising anything (Assignment-3 lesson: verify,
don't trust). Found: git configured (`Ace-2504` / harman2504sandhu@gmail.com); **gh authenticated** as
Ace-2504 with `repo` scope (private-repo creation available); Python 3.12.10; local GPU **RTX 3060, 12 GB**.
**Gap:** `modal` CLI not on PATH — needs the project venv / reinstall before any cloud GPU work. Hardware
note that shapes the plan: **Qwen2.5-7B (Exp 2) will not fit in 12 GB**, so Exp 2/3/4 and the live demo
target **Modal L4/A10G**; the 3060 is for Exp-1 dev and harness iteration only. → *Better: real constraints
known up front; Modal auth surfaced as a prerequisite instead of a Phase-3 surprise.*

**E3 · Plan + story + repo scaffold.** Wrote `plan.md` (task-in-plain-words, golden rules, hardware/model
decisions, Phases 0–6 mapped to the brief's exact deliverables, submission structure, open decisions).
Recommended models: **125M (Session 2)** for Exp 1 (cleanest roofline, matches the lab calibration table),
**Qwen2.5-7B-Instruct + Qwen2.5-0.5B-Instruct** for Exp 2 (the lab pair), **Qwen2.5-0.5B-Instruct** for
Exp 3/4 (small + GQA, which is exactly the `kv_heads` trap Exp-4's KV-byte arithmetic teaches). Initialized
this `story.md`. Set up the repo as a **private git repo** with the brief's directory structure. Handed
Harman the prerequisites list. **Not executing any experiment phase yet — planning stage only** (per
Harman's instruction). → *Better: scaffolded and de-risked before spend, same discipline as Assignment 3.*

---

## Current status (as of last entry)

- **Oriented** to the brief; four experiments + golden rules understood (E1).
- **Environment mapped** (E2): git + gh ready; local RTX 3060 12 GB; **Modal CLI missing** (prereq);
  7B needs cloud.
- **Planned** (E3): `plan.md` written end-to-end; models proposed; repo scaffolded; private git repo set up.
- **Next (pending Harman's go-ahead + prerequisites):** Phase 1 — build and self-test the `bench/` timing
  harness *before* collecting a single number. **No experiment code has been executed yet.**
