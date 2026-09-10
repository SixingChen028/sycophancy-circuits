#!/usr/bin/env python3
"""Collect head write-in vectors and reduce them with PCA (paper Appendix B.2).

The write-in is the vector a head adds to the last token's residual stream
through its OV circuit -- the same quantity the patching analysis intervenes on.
Probing it asks what answer a head's output encodes, as opposed to how much it
matters causally.

Two head populations are collected, both defined by the patching sweeps:

  opinion heads    top early heads from the plain-to-deceptive sweep
  retrieval heads  top late heads from the answer-retrieval sweeps, scored by
                   the mean of the label-swap and content-swap effects, so that
                   a head counts as a retrieval head only if it tracks the
                   correct answer under both kinds of relabelling

PCA is fit per head on the pooled source and target write-ins, after mean
centering, so both runs are expressed in the same basis and their probe
accuracies are comparable. The reduction is needed because the write-in is
hidden_size-dimensional while there are only a few hundred prompts.

Requires: head_patching.npz for mmlu, mmlu_label_swap and mmlu_content_swap.
Output: runs/<model>/<condition>/writeins.npz
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from syco.conditions import get_condition
from syco.heads import format_heads, opinion_population, retrieval_population
from syco.hooks import cache_o_proj_inputs, head_writein
from syco.runtime import (base_parser, batches, load_model, load_npz, run_path,
                          save_npz)


def collect(model, tok, prompts, spec, heads, device, batch_size, label):
    """Write-in vectors per head: {(layer, head): (N, hidden_size) float32}."""
    layers = sorted({l for l, _ in heads})
    n = len(prompts)
    out = {h: np.zeros((n, model.config.hidden_size), dtype=np.float32)
           for h in heads}

    for start, batch in batches(prompts, batch_size):
        store = {}
        with cache_o_proj_inputs(model, layers, store):
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                model(**enc, logits_to_keep=1)
        with torch.no_grad():
            for layer, head in heads:
                vec = head_writein(model, spec, layer, head, store[layer][0])
                out[(layer, head)][start:start + len(batch)] = vec.cpu().numpy()
        print(f"  {label} [{start + len(batch)}/{n}]", flush=True)
    return out


def fit_pca(source_vectors, target_vectors, n_components):
    """Fit PCA on the pooled runs, then project each run into it.

    Returns (source_projection, target_projection, variance_ratio).
    """
    pooled = np.concatenate([source_vectors, target_vectors], axis=0)
    mean = pooled.mean(axis=0)
    centered = (pooled - mean).astype(np.float32)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    components = components[:n_components]
    total = (singular ** 2).sum()
    ratio = (singular[:n_components] ** 2) / total
    return (
        ((source_vectors - mean) @ components.T).astype(np.float32),
        ((target_vectors - mean) @ components.T).astype(np.float32),
        ratio.astype(np.float32),
    )


def main():
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--condition", default="mmlu")
    parser.add_argument("--k", type=int, default=10, help="Heads per population")
    parser.add_argument("--n_components", type=int, default=10,
                        help="PCA components kept; the probes use 10")
    parser.set_defaults(batch_size=32)
    args = parser.parse_args()

    condition = get_condition(args.condition)
    spec, tok, model, device = load_model(args.model)
    pair = condition.build(spec, tok, args.model, limit=args.limit)

    opinion = opinion_population(args.model, spec, k=args.k,
                                 condition=condition.name)
    print(f"\nOpinion heads:   {format_heads(opinion)}", flush=True)

    try:
        retrieval = retrieval_population(args.model, spec, args.k)
        print(f"Retrieval heads: {format_heads(retrieval)}", flush=True)
    except SystemExit:
        # The retrieval population needs the two relabelling sweeps. They are
        # only defined for MMLU, so conditions like pushback probe opinion
        # heads alone (paper, Figure 6B).
        retrieval = []
        print("Retrieval heads: skipped (relabelling sweeps not found)", flush=True)

    heads = opinion + retrieval
    print(f"\nCollecting {condition.source_label} write-ins...", flush=True)
    source = collect(model, tok, pair.source, spec, heads, device,
                     args.batch_size, condition.source_label)
    print(f"\nCollecting {condition.target_label} write-ins...", flush=True)
    target = collect(model, tok, pair.target, spec, heads, device,
                     args.batch_size, condition.target_label)

    print("\nFitting PCA...", flush=True)
    arrays = {}
    for name, population in (("opinion", opinion), ("retrieval", retrieval)):
        if not population:
            continue
        n, k = len(pair), args.n_components
        src = np.zeros((len(population), n, k), dtype=np.float32)
        tgt = np.zeros((len(population), n, k), dtype=np.float32)
        var = np.zeros((len(population), k), dtype=np.float32)
        for i, head in enumerate(population):
            src[i], tgt[i], var[i] = fit_pca(source[head], target[head], k)
            print(f"  {name} L{head[0]}H{head[1]}: "
                  f"PC1={var[i][0]:.3f}  cumulative={var[i].sum():.3f}", flush=True)
        arrays[f"{name}_heads"] = np.array(population)
        arrays[f"{name}_source_pca"] = src
        arrays[f"{name}_target_pca"] = tgt
        arrays[f"{name}_variance_ratio"] = var

    save_npz(run_path(args.model, condition.name, "writeins.npz"),
             correct_idx=pair.correct_idx, wrong_idx=pair.wrong_idx, **arrays)


if __name__ == "__main__":
    main()
