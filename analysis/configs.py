"""Common configs for KV cache/memory analysis.

This module centralizes model specs, GPU capacities, and precision byte sizes
used across the analysis scripts.
"""

from dataclasses import dataclass
from typing import Dict
import os


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_layers: int
    hidden_size: int  # d_model


# NOTE: Specs are approximate and representative of common configurations.
# Update if you have the exact configs for your local checkpoints.
MODEL_CONFIGS: Dict[str, ModelSpec] = {
    # LLaMA 3 8B (approx): 32 layers, 4096 hidden
    "llama3-8b": ModelSpec(name="llama3-8b", num_layers=32, hidden_size=4096),
    # LLaMA 2/3 70B (approx): 80 layers, 8192 hidden
    "llama3-70b": ModelSpec(name="llama3-70b", num_layers=80, hidden_size=8192),
}


# Bytes per scalar element for different precisions
PRECISION_BYTES: Dict[str, int] = {
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
}


# Total VRAM capacities in bytes
GPU_CAPACITIES: Dict[str, int] = {
    "V100-32GB": 32 * 1024**3,
    "H100-80GB": 80 * 1024**3,
    "H200-140GB": 140 * 1024**3,
}


# Default data paths
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inputs", "musique_s.json")


def bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


