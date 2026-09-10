#!/usr/bin/env python3
"""Layer-by-layer residual-stream patching (paper Section 3.1, Figure 3B).

Patches the last token's residual stream at one layer from the source run into
the target run, and measures the normalized logit difference. This localizes the
shift to a depth without yet saying which component carries it.

Running this for --condition mmlu and --condition mmlu_label_swap gives the two
curves of Figure 3B. Their separation is the paper's central claim about depth:
the plain-to-label-swap curve, which isolates answer retrieval, only rises after
the critical layer, while the plain-to-deceptive curve begins rising several
layers earlier, so the opinion is already present in the residual stream before
retrieval begins.

Output: runs/<model>/<condition>/residual_patching.npz
  norm_logit_diff  (n_layers,)     mean over examples
  per_example      (n_layers, N)
  source_ld, target_ld (N,)
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from syco.conditions import get_condition
from syco.hooks import cache_residual_stream, patch_residual_stream
from syco.metrics import aggregate, logit_difference, normalized_logit_difference
from syco.runtime import (base_parser, batches, collect_scores, load_model,
                          run_path, save_npz)


def cache_source(model, tok, prompts, spec, readout, device, batch_size):
    store = {}
    scores = []
    with cache_residual_stream(model, range(spec.n_layers), store):
        for start, batch in batches(prompts, batch_size):
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                out = model(**enc, logits_to_keep=1)
            scores.append(readout.scores(out.logits[:, -1, :], start, len(batch)))
            print(f"  source cache [{start + len(batch)}/{len(prompts)}]", flush=True)
    cache = {l: torch.cat(store[l], dim=0) for l in range(spec.n_layers)}
    return cache, np.concatenate(scores, axis=0)


def main():
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--condition", default="mmlu")
    parser.set_defaults(batch_size=32)
    args = parser.parse_args()

    condition = get_condition(args.condition)
    spec, tok, model, device = load_model(args.model)
    pair = condition.build(spec, tok, args.model, limit=args.limit)
    readout = pair.readout
    n = len(pair)
    print(f"\nCondition {condition.name}: {condition.source_label} -> "
          f"{condition.target_label}, N={n}", flush=True)

    print("\nCaching source run...", flush=True)
    cache, source_scores = cache_source(model, tok, pair.source, spec, readout,
                                        device, args.batch_size)
    source_ld = logit_difference(source_scores, readout.correct_col, readout.wrong_col)

    print("\nTarget baseline...", flush=True)
    target_scores = collect_scores(model, tok, pair.target, readout, device,
                                   args.batch_size, "target")
    target_ld = logit_difference(target_scores, readout.correct_col, readout.wrong_col)
    print(f"source_ld = {source_ld.mean():+.3f}   target_ld = {target_ld.mean():+.3f}",
          flush=True)

    per_example = np.zeros((spec.n_layers, n), dtype=np.float32)
    mean_effect = np.zeros(spec.n_layers, dtype=np.float32)

    for layer in range(spec.n_layers):
        patched = np.zeros(n, dtype=np.float32)
        for start, batch in batches(pair.target, args.batch_size):
            source_hidden = cache[layer][start:start + len(batch)].to(device)
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(device)
            with patch_residual_stream(model, layer, source_hidden):
                with torch.no_grad():
                    out = model(**enc, logits_to_keep=1)
            scores = readout.scores(out.logits[:, -1, :], start, len(batch))
            sl = slice(start, start + len(batch))
            patched[sl] = logit_difference(
                scores, readout.correct_col[sl], readout.wrong_col[sl]
            )
        per_example[layer] = normalized_logit_difference(patched, source_ld, target_ld)
        mean_effect[layer] = aggregate(per_example[layer])
        print(f"Layer {layer:2d}  D = {mean_effect[layer]:+.4f}", flush=True)

    save_npz(run_path(args.model, condition.name, "residual_patching.npz"),
             norm_logit_diff=mean_effect,
             per_example=per_example,
             source_ld=source_ld,
             target_ld=target_ld,
             critical_layer=spec.critical_layer)


if __name__ == "__main__":
    main()
