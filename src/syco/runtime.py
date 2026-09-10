"""Model loading, batched inference and run-directory layout."""

import argparse
import os

import numpy as np
import torch

from .models import get_model_spec, verify_answer_ids, verify_geometry

__all__ = [
    "RUNS_DIR", "run_dir", "run_path",
    "load_model", "base_parser", "batches", "collect_scores",
    "save_npz", "load_npz",
]

# Every artifact lives under runs/<model>/<condition>/. Nothing else is written
# outside this tree, so the whole output of a replication is one directory.
RUNS_DIR = os.environ.get(
    "SYCO_RUNS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs"),
)


def run_dir(model_key, condition, create=True):
    path = os.path.abspath(os.path.join(RUNS_DIR, model_key, condition))
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def run_path(model_key, condition, filename, create=True):
    return os.path.join(run_dir(model_key, condition, create), filename)


def base_parser(description):
    """Arguments shared by every GPU experiment."""
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="llama",
                   help="Model key from syco.models.MODELS")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Prompts per forward pass")
    p.add_argument("--limit", type=int, default=None,
                   help="Use only the first N examples (for smoke tests)")
    return p


def load_model(model_key, verify=True):
    """Load a model and tokenizer, and check the registry against the config.

    Left padding is required: every readout takes the logits at position -1,
    which is only the last real token when padding is on the left.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    spec = get_model_spec(model_key)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {spec.hf_id} on {device}...", flush=True)

    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    if verify:
        verify_answer_ids(spec, tok)
        verify_geometry(spec, model)

    print(f"Loaded. {spec.n_layers} layers x {spec.n_heads} heads, "
          f"d_head={spec.d_head}, critical layer {spec.critical_layer}.", flush=True)
    return spec, tok, model, device


def batches(items, batch_size):
    """Yield (start_index, batch) pairs."""
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def _encode(tok, batch, device):
    return tok(batch, return_tensors="pt", padding=True,
               truncation=True, max_length=2048).to(device)


def collect_scores(model, tok, prompts, readout, device, batch_size, label=""):
    """Run prompts and return the readout's (N, K) score array.

    `logits_to_keep=1` asks for the final position only, which is all any
    readout uses; it cuts the logits tensor from (batch, seq, vocab) to
    (batch, 1, vocab) and is a pure memory saving.
    """
    out = []
    for start, batch in batches(prompts, batch_size):
        enc = _encode(tok, batch, device)
        with torch.no_grad():
            result = model(**enc, logits_to_keep=1)
        out.append(readout.scores(result.logits[:, -1, :], start, len(batch)))
        if label:
            print(f"  {label} [{start + len(batch)}/{len(prompts)}]", flush=True)
    return np.concatenate(out, axis=0)


def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)
    print(f"Saved -> {path}", flush=True)


def load_npz(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"Missing input: {path}\nRun the experiment that produces it first "
            f"(see docs/paper_map.md for the order)."
        )
    return np.load(path, allow_pickle=True)
