"""Shared figure style and data access.

Colors are the ones used in the paper. Every figure script writes into
figures/output/<model>/ so that the same script run against a different model
never overwrites another model's panels.
"""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from syco.models import get_model_spec          # noqa: E402
from syco.runtime import load_npz, run_path     # noqa: E402

CORRECT = "#4586b0"
SYCOPHANTIC = "#c9574b"
OTHER = "#7f8c8d"
LABEL_SWAP = "#3daebb"
DECEPTIVE = "#f18a3b"
EARLY = "#fcb13d"
LATE = "#e86363"
HIGHLIGHT = "#6c3d8f"
BACKGROUND = "#c8c8c8"
PROBE_COLORS = ["#2471a3", "#c0392b", "#85c1e9", "#f1948a"]

__all__ = [
    "CORRECT", "SYCOPHANTIC", "OTHER", "LABEL_SWAP", "DECEPTIVE",
    "EARLY", "LATE", "HIGHLIGHT", "BACKGROUND", "PROBE_COLORS",
    "apply_style", "figure_parser", "output_dir", "save", "result", "panel",
    "get_model_spec", "load_npz", "run_path", "np", "plt",
]


def apply_style():
    matplotlib.rcParams.update({
        "font.size": 12,
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.spines.right": False,
        "axes.spines.top": False,
        "pdf.fonttype": 42,   # editable text in the PDF, not outlines
        "ps.fonttype": 42,
    })


def figure_parser(description):
    import argparse

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="llama")
    return parser


def output_dir(model_key):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output",
                        model_key)
    os.makedirs(path, exist_ok=True)
    return path


def save(fig, model_key, name):
    path = os.path.join(output_dir(model_key), name)
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    fig.savefig(f"{path}.png", bbox_inches="tight", dpi=200)
    print(f"Saved -> {path}.pdf")
    plt.close(fig)


def result(model_key, condition, filename):
    """Load one analysis output, with a clear error if it is missing."""
    return load_npz(run_path(model_key, condition, filename, create=False))


def panel(name):
    """Decorator: skip a panel whose inputs have not been produced yet.

    Figures are usually built while a replication is still in progress, so a
    missing input should cost one panel, not the whole run.
    """
    def decorate(fn):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except SystemExit as exc:
                print(f"Skipping {name}: {exc}")
                return None
        return wrapper
    return decorate
