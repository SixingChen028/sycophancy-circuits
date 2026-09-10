#!/usr/bin/env python3
"""Cumulative patching of the opinion heads (paper Appendix D.1, Figure S2).

Takes the top early heads from the per-head sweep and patches them from the
source run into the target run together, adding one head at a time in order of
individual causal effect. Reports the sycophancy rate after each addition.

The unpatched target run is sycophantic by construction (every example was
screened for that), so the curve starts at 100% and falls as the heads carrying
the opinion are restored to their no-opinion values.

Requires: head_patching.npz for the same model and condition.
Output: runs/<model>/<condition>/head_patching_cumulative.npz
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from syco.conditions import get_condition
from syco.heads import format_heads, top_heads
from syco.hooks import cache_o_proj_inputs, patch_head_outputs
from syco.metrics import accuracy_summary
from syco.runtime import (base_parser, batches, collect_scores, load_model,
                          load_npz, run_path, save_npz)


def cache_source(model, tok, prompts, layers, device, batch_size):
    store = {}
    with cache_o_proj_inputs(model, layers, store):
        for start, batch in batches(prompts, batch_size):
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                model(**enc, logits_to_keep=1)
            print(f"  source cache [{start + len(batch)}/{len(prompts)}]", flush=True)
    return {l: torch.cat(store[l], dim=0) for l in layers}


def run_patched(model, tok, prompts, spec, readout, heads, cache, device, batch_size):
    scores = []
    for start, batch in batches(prompts, batch_size):
        values = {l: cache[l][start:start + len(batch)].to(device)
                  for l in {layer for layer, _ in heads}}
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to(device)
        with patch_head_outputs(model, spec, heads, values):
            with torch.no_grad():
                out = model(**enc, logits_to_keep=1)
        scores.append(readout.scores(out.logits[:, -1, :], start, len(batch)))
    return np.concatenate(scores, axis=0)


def main():
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--condition", default="mmlu")
    parser.add_argument("--k", type=int, default=10,
                        help="How many top early heads to patch cumulatively. "
                             "The paper uses 10; Mistral spreads the effect "
                             "over more heads and uses 20 (Figure S7).")
    parser.set_defaults(batch_size=64)
    args = parser.parse_args()

    condition = get_condition(args.condition)
    spec, tok, model, device = load_model(args.model)
    pair = condition.build(spec, tok, args.model, limit=args.limit)
    readout = pair.readout
    if not readout.supports_accuracy:
        raise SystemExit("Cumulative patching reports a sycophancy rate, which "
                         "needs the four-way multiple-choice readout.")

    sweep = load_npz(run_path(args.model, condition.name, "head_patching.npz",
                              create=False))
    heads = top_heads(sweep["norm_logit_diff"], spec, band="early", k=args.k)
    print(f"\nOpinion heads by individual effect: {format_heads(heads)}", flush=True)

    layers = sorted({l for l, _ in heads})
    print("\nCaching source run...", flush=True)
    cache = cache_source(model, tok, pair.source, layers, device, args.batch_size)

    baseline = collect_scores(model, tok, pair.target, readout, device,
                              args.batch_size, "target baseline")
    base_summary = accuracy_summary(baseline, readout.correct_col, readout.wrong_col)
    print(f"\nUnpatched target: sycophancy {base_summary['sycophancy']:.1%}, "
          f"accuracy {base_summary['accuracy']:.1%}", flush=True)

    syco_rates, accuracies = [], []
    for k in range(1, len(heads) + 1):
        scores = run_patched(model, tok, pair.target, spec, readout, heads[:k],
                             cache, device, args.batch_size)
        summary = accuracy_summary(scores, readout.correct_col, readout.wrong_col)
        syco_rates.append(summary["sycophancy"])
        accuracies.append(summary["accuracy"])
        print(f"  +{format_heads([heads[k - 1]])}  ({k} patched)  "
              f"sycophancy {summary['sycophancy']:.1%}  "
              f"accuracy {summary['accuracy']:.1%}", flush=True)

    save_npz(run_path(args.model, condition.name, "head_patching_cumulative.npz"),
             heads=np.array(heads),
             sycophancy_rate=np.array(syco_rates, dtype=np.float32),
             accuracy=np.array(accuracies, dtype=np.float32),
             baseline_sycophancy=base_summary["sycophancy"],
             baseline_accuracy=base_summary["accuracy"])


if __name__ == "__main__":
    main()
