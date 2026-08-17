"""Hardware provenance — stamped into every results JSON so no number is orphaned from
the GPU that produced it (brief: "a speedup without a GPU name is not a result").

Also computes the theoretical FP32 peak used as the roofline upper bound in gauntlet G5.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Optional

import torch

# FP32 CUDA cores per SM, keyed by compute capability (major, minor).
# Ampere GA10x (RTX 30xx, sm_86) and Ada (L4, sm_89) pack 128; server Ampere/Hopper and
# Volta/Turing differ. Conservative default of 64 if unknown.
_CORES_PER_SM = {
    (6, 0): 64, (6, 1): 128, (6, 2): 128,
    (7, 0): 64, (7, 2): 64, (7, 5): 64,
    (8, 0): 64, (8, 6): 128, (8, 7): 128, (8, 9): 128,
    (9, 0): 128,
}


def _nvidia_smi(query: str) -> Optional[str]:
    """Return the first row of `nvidia-smi --query-gpu=<query>` (csv, no header/units)."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode("utf-8", "ignore")
        first = out.strip().splitlines()[0]
        return first.strip()
    except Exception:
        return None


def get_gpu_info(device: int = 0) -> dict:
    """Full hardware + software provenance for the active CUDA device."""
    if not torch.cuda.is_available():
        return {"cuda_available": False, "torch_version": torch.__version__}

    props = torch.cuda.get_device_properties(device)
    cc = (props.major, props.minor)
    max_sm_clock = _nvidia_smi("clocks.max.sm")      # MHz
    cur_sm_clock = _nvidia_smi("clocks.current.sm")  # MHz
    info = {
        "cuda_available": True,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_mem_gb": round(props.total_memory / 1024**3, 2),
        "sm_count": props.multi_processor_count,
        "cores_per_sm": _CORES_PER_SM.get(cc),
        "max_sm_clock_mhz": _to_float(max_sm_clock),
        "current_sm_clock_mhz": _to_float(cur_sm_clock),
        "temperature_c": _to_float(_nvidia_smi("temperature.gpu")),
        "driver_version": _nvidia_smi("driver_version"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "theoretical_fp32_tflops": theoretical_fp32_tflops(device),
    }
    return info


def _to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def theoretical_fp32_tflops(device: int = 0) -> Optional[float]:
    """Peak FP32 throughput = SMs * cores_per_SM * 2 (FMA) * clock_hz.

    Uses the max SM clock from nvidia-smi. Returns None if we can't determine
    cores-per-SM or the clock (in which case G5 reports without a hard peak bound).
    """
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(device)
    cores = _CORES_PER_SM.get((props.major, props.minor))
    clock_mhz = _to_float(_nvidia_smi("clocks.max.sm"))
    if cores is None or clock_mhz is None:
        return None
    flops = props.multi_processor_count * cores * 2 * (clock_mhz * 1e6)
    return round(flops / 1e12, 3)
