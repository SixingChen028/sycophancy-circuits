#!/usr/bin/env python3
"""Per-head activation patching, source run into target run (paper Section 3.2).

For every (layer, head), replace that head's contribution to the last token's
residual stream in the target run with the value it took in the source run, and
measure the normalized logit difference. A head with a high value mediates the
answer shift between the two runs.

This is the sweep behind Figures 4A/4B (MMLU, both patching directions),
5A/5B/5C (the generalization conditions) and 6A (content-free pushback). Which
of those it produces depends entirely on --condition.

The sweep is n_layers x n_heads x ceil(N/batch) forward passes and takes a few
GPU-hours. It checkpoints after every layer, so an interrupted or timed-out run
can be resubmitted unchanged and resumes where it stopped.

Output: runs/<model>/<condition>/head_patching.npz
  norm_logit_diff      (n_layers, n_heads)      mean over examples
  per_example          (n_layers, n_heads, N)
  source_ld, target_ld (N,)                     unpatched baselines
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from syco.conditions import get_condition
from syco.heads import format_heads, rank_heads
from syco.hooks import cache_o_proj_inputs, patch_head_outputs
from syco.metrics import aggregate, logit_difference, normalized_logit_difference
from syco.runtime import (base_parser, batches, collect_scores, load_model,
                          run_path, save_npz)


def cache_source_writeins(model, tok, prompts, spec, readout, device, batch_size):
    """Last-token o_proj input at every layer, plus the source logit difference."""
    store = {}
    scores = []
    layers = range(spec.n_layers)
    with cache_o_proj_inputs(model, layers, store):
        for start, batch in batches(prompts, batch_size):
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                out = model(**enc, logits_to_keep=1)
            scores.append(readout.scores(out.logits[:, -1, :], start, len(batch)))
            print(f"  source cache [{start + len(batch)}/{len(prompts)}]", flush=True)

    cache = {l: torch.cat(store[l], dim=0) for l in layers}
    width = cache[0].shape[1]
    expected = spec.n_heads * spec.d_head
    if width != expected:
        raise RuntimeError(
            f"o_proj input width {width} != n_heads*d_head ({expected}). "
            f"The head geometry in syco.models is wrong for this model."
        )
    return cache, np.concatenate(scores, axis=0)


def patch_one_head(model, tok, prompts, spec, readout, cache_layer, layer, head,
                   target_ld, source_ld, device, batch_size):
    """Normalized logit difference per example for a single patched head."""
    patched = np.zeros(len(prompts), dtype=np.float32)
    for start, batch in batches(prompts, batch_size):
        values = {layer: cache_layer[start:start + len(batch)].to(device)}
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to(device)
        with patch_head_outputs(model, spec, [(layer, head)], values):
            with torch.no_grad():
                out = model(**enc, logits_to_keep=1)
        scores = readout.scores(out.logits[:, -1, :], start, len(batch))
        sl = slice(start, start + len(batch))
        patched[sl] = logit_difference(
            scores, readout.correct_col[sl], readout.wrong_col[sl]
        )
    return normalized_logit_difference(patched, source_ld, target_ld)


def main():
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--condition", default="mmlu",
                        help="Condition name from syco.conditions.CONDITIONS")
    parser.add_argument("--cache_batch_size", type=int, default=64,
                        help="Batch size for the two unpatched baseline passes")
    parser.set_defaults(batch_size=128)
    args = parser.parse_args()

    condition = get_condition(args.condition)
    spec, tok, model, device = load_model(args.model)
    pair = condition.build(spec, tok, args.model, limit=args.limit)
    n = len(pair)
    readout = pair.readout
    print(f"\nCondition {condition.name}: {condition.source_label} -> "
          f"{condition.target_label}, N={n}", flush=True)

    out_file = run_path(args.model, condition.name, "head_patching.npz")
    ckpt_file = run_path(args.model, condition.name, "head_patching.ckpt.npz")

    print("\nCaching source run...", flush=True)
    cache, source_scores = cache_source_writeins(
        model, tok, pair.source, spec, readout, device, args.cache_batch_size
    )
    source_ld = logit_difference(source_scores, readout.correct_col, readout.wrong_col)

    print("\nTarget baseline...", flush=True)
    target_scores = collect_scores(model, tok, pair.target, readout, device,
                                   args.cache_batch_size, "target")
    target_ld = logit_difference(target_scores, readout.correct_col, readout.wrong_col)
    print(f"source_ld = {source_ld.mean():+.3f}   target_ld = {target_ld.mean():+.3f}",
          flush=True)

    shape = (spec.n_layers, spec.n_heads)
    mean_effect = np.zeros(shape, dtype=np.float32)
    per_example = np.full(shape + (n,), np.nan, dtype=np.float32)
    done = set()

    if os.path.exists(ckpt_file):
        ck = np.load(ckpt_file)
        if ck["per_example"].shape == per_example.shape:
            mean_effect = ck["norm_logit_diff"]
            per_example = ck["per_example"]
            done = {int(x) for x in ck["done_layers"]}
            print(f"Resuming; {len(done)} layers already done.", flush=True)
        else:
            print("Checkpoint shape does not match this run; starting fresh.",
                  flush=True)

    for layer in range(spec.n_layers):
        if layer in done:
            continue
        for head in range(spec.n_heads):
            effect = patch_one_head(
                model, tok, pair.target, spec, readout, cache[layer], layer, head,
                target_ld, source_ld, device, args.batch_size
            )
            per_example[layer, head] = effect
            mean_effect[layer, head] = aggregate(effect)
        best = np.argsort(mean_effect[layer])[::-1][:3]
        print(f"Layer {layer:2d}  best: "
              + "  ".join(f"H{h}={mean_effect[layer, h]:+.4f}" for h in best),
              flush=True)
        done.add(layer)
        np.savez(ckpt_file, norm_logit_diff=mean_effect, per_example=per_example,
                 done_layers=np.array(sorted(done)))

    print("\nTop 20 heads overall:")
    for layer, head, score in rank_heads(mean_effect, spec, "all")[:20]:
        band = "early" if layer <= spec.critical_layer else "late"
        print(f"  L{layer:2d}H{head:2d}  {score:+.4f}  ({band})")
    print("\nTop 10 early (opinion candidates): "
          + format_heads([(l, h) for l, h, _ in rank_heads(mean_effect, spec, "early")[:10]]))
    print("Top 10 late  (retrieval candidates): "
          + format_heads([(l, h) for l, h, _ in rank_heads(mean_effect, spec, "late")[:10]]))

    save_npz(out_file,
             norm_logit_diff=mean_effect,
             per_example=per_example,
             source_ld=source_ld,
             target_ld=target_ld,
             critical_layer=spec.critical_layer)
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)


if __name__ == "__main__":
    main()
