"""bench — the trustworthy timing harness for the inference-optimization experiments.

Public surface:
    from bench.timer import benchmark, sync, time_wall, time_event
    from bench.stats import summarize
    from bench.gpu_info import get_gpu_info, theoretical_fp32_tflops
    from bench.token_utils import token_positions, token_equal
"""
