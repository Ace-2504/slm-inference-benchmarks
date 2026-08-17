"""Shared Modal image + GPU choice for every experiment, so all numbers come off the
same hardware and software stack (brief: "report the hardware for every number").

The bench package is copied into the container as local Python source, so remote
functions run the *same* harness code we validated with the gauntlet.
"""
import modal

GPU = "L4"  # confirmed available; matches the brief's reference calibration hardware

# vLLM is added only where needed (exp 3/4). Base stack covers exp 1/2 + the harness.
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "numpy", "accelerate")
    .add_local_python_source("bench")
)

# Persist HF downloads between runs so we pay the model pull once.
HF_CACHE = modal.Volume.from_name("a4-hf-cache", create_if_missing=True)
HF_CACHE_DIR = "/root/.cache/huggingface"
