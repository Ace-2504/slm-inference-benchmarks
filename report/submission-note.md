Dr. Sreedath Panat, thank you for another assignment that taught me more than I expected.

The brief was simple to state and hard to do honestly: build one inference pipeline and measure what four
serving tricks are each worth — the **KV cache**, **speculative decoding**, **continuous batching**, and
**PagedAttention** — with one rule running through all of it: both sides of every comparison must produce the
exact same tokens before I trust any speedup.

Before I measured anything, I spent time making the stopwatch trustworthy. I wrote an 8-test "gauntlet" that
calibrates the timer against things it cannot fake, and it earned its keep immediately: it showed that timing
a GPU matmul without `cuda.synchronize()` under-reports the time by **334×**, and it caught two real bugs in
my own test code. That set the tone — measure the instrument before you trust the number.

Then the four experiments, and each one surprised me.

The **KV cache** was the biggest lesson. At a batch of 1, the version *without* the cache does **60× more
arithmetic and finishes in the same wall-clock time**. That looked wrong until I understood the roofline: a
single stream barely keeps the GPU busy, so the extra math is nearly free. The cache only pulls ahead as you
batch more users together — about **10× at batch 16**.

**Speculative decoding** was the honest trap. A small 0.5B "draft" model guesses a few tokens ahead and the
big 7B "target" checks the guesses in one pass. My first measurement showed it running *slower*. The reason
was not the acceptance rate — it was that my draft takes 25 ms per step when it should take about 3; it is too
small to keep the GPU busy and wastes time launching tiny kernels (**c = 0.43**). The real fix is CUDA
graphs, which I tried two ways — both crashed with device-side errors on my torch/transformers versions, so I
reported the honest c rather than a number I could not reproduce.

**Continuous batching** taught me that static batching does not lose by being slow — it loses by being
*idle*: short requests finish and sit in their slot while one long one runs on. Refilling those slots the
moment they free up gave about **3× the throughput**.

**PagedAttention** was about memory: reserving a full max-length block per request wastes **91.8%** of it
against real (short) answers; handing out small blocks on demand fits about **19× more sequences**, which I
confirmed on the real vLLM engine (15,220 versus 801).

My biggest bottleneck was an honest one. I first measured the KV cache with a tiny prompt in my live demo and
got a small speedup, which did not match the report's 512-token measurement. Chasing it down, I found no bug —
just that the cache's payoff scales with how long the context is — and I fixed the demo to measure the same
thing as the report.

If I had to keep one sentence: serving a model well is not about making any one step fast, it is about never
letting the expensive, bandwidth-limited GPU sit idle.

I would love your thoughts, sir — especially on whether the speculative-decoding `c` could be brought down on
a different stack. Everything is live at https://harman-inference-lab.vercel.app and the code + report are in
the repo. Thank you for reading.
