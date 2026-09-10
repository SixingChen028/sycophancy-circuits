"""Mechanisms of sycophantic agreement in language models.

Code for the paper "Tracing mechanisms of sycophantic agreement in language
models". See the README for the figure-to-command map.
"""

from .models import MODELS, ModelSpec, get_model_spec
from .conditions import CONDITIONS, get_condition
from .runtime import load_model, run_dir, run_path

__all__ = [
    "MODELS", "ModelSpec", "get_model_spec",
    "CONDITIONS", "get_condition",
    "load_model", "run_dir", "run_path",
]
