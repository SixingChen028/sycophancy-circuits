#!/usr/bin/env python3
"""Zero-ablation of the opinion heads (paper Section 3.2 and Appendix B.3).

Ablating a head means setting its contribution to the residual stream to zero,
at the same point where the patching analysis intervenes on it. For each ablation
set this reports three rates:

  plain accuracy      correct answers in the source run    (collateral damage)
  deceptive accuracy  correct answers in the target run    (sycophancy removed)
  sycophancy rate     answers matching the stated opinion

The script does two passes, matching the paper:

  1. Ablate each candidate head on its own (Figure S3), and rank the candidates
     by deceptive accuracy + plain accuracy.
  2. Ablate them cumulatively in that order (Figure 4F).

Why rank by ablation effect and not by patching effect
------------------------------------------------------
The two measure different things (paper, Appendix B.3). Attention heads are
polysemantic. Patching swaps only the opinion-dependent part of what a head
writes, because the source and target runs differ only in whether an opinion is
stated, so it isolates the opinion signal and says nothing about the head's other
functions. Ablation zeroes the head entirely, taking those other functions with
it. Ranking by individual ablation effect therefore selects for heads whose
removal is *selective* -- which is the property that matters when the question
is whether intervening can suppress sycophancy at acceptable cost.

Requires: head_patching.npz for the same model and condition.
Output: runs/<model>/<condition>/head_ablation.npz
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from syco.conditions import get_condition
from syco.heads import format_heads, top_heads
from syco.hooks import ablate_heads
from syco.metrics import accuracy_summary
from syco.runtime import (base_parser, batches, load_model, load_npz, run_path,
                          save_npz)

import torch


def run_ablated(model, tok, prompts, spec, readout, heads, device, batch_size,
                last_token_only):
    scores = []
    for start, batch in batches(prompts, batch_size):
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to(device)
        with ablate_heads(model, spec, heads, last_token_only=last_token_only):
            with torch.no_grad():
                out = model(**enc, logits_to_keep=1)
        scores.append(readout.scores(out.logits[:, -1, :], start, len(batch)))
    return np.concatenate(scores, axis=0)


def evaluate(model, tok, pair, spec, heads, device, batch_size, last_token_only):
    readout = pair.readout
    source = run_ablated(model, tok, pair.source, spec, readout, heads, device,
                         batch_size, last_token_only)
    target = run_ablated(model, tok, pair.target, spec, readout, heads, device,
                         batch_size, last_token_only)
    src = accuracy_summary(source, readout.correct_col, readout.wrong_col)
    tgt = accuracy_summary(target, readout.correct_col, readout.wrong_col)
    return {
        "plain_accuracy": src["accuracy"],
        "deceptive_accuracy": tgt["accuracy"],
        "sycophancy_rate": tgt["sycophancy"],
    }


def main():
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--condition", default="mmlu")
    parser.add_argument("--k", type=int, default=10,
                        help="Size of the candidate pool, taken from the top of "
                             "the early-head patching ranking")
    parser.add_argument("--last_token_only", action="store_true",
                        help="Ablate only at the final position. The paper "
                             "ablates at every position, the default here.")
    parser.set_defaults(batch_size=64)
    args = parser.parse_args()

    condition = get_condition(args.condition)
    spec, tok, model, device = load_model(args.model)
    pair = condition.build(spec, tok, args.model, limit=args.limit)
    if not pair.readout.supports_accuracy:
        raise SystemExit("Ablation reports answer rates, which need the "
                         "four-way multiple-choice readout.")

    sweep = load_npz(run_path(args.model, condition.name, "head_patching.npz",
                              create=False))
    candidates = top_heads(sweep["norm_logit_diff"], spec, band="early", k=args.k)
    print(f"\nCandidates: {format_heads(candidates)}", flush=True)

    unablated = evaluate(model, tok, pair, spec, [], device, args.batch_size,
                         args.last_token_only)
    print(f"\nUnablated: plain {unablated['plain_accuracy']:.1%}  "
          f"deceptive {unablated['deceptive_accuracy']:.1%}  "
          f"sycophancy {unablated['sycophancy_rate']:.1%}", flush=True)

    print("\n-- Each head ablated on its own --", flush=True)
    individual = []
    for head in candidates:
        result = evaluate(model, tok, pair, spec, [head], device, args.batch_size,
                          args.last_token_only)
        score = result["deceptive_accuracy"] + result["plain_accuracy"]
        individual.append((score, head, result))
        print(f"  L{head[0]}H{head[1]}  deceptive {result['deceptive_accuracy']:.1%}  "
              f"plain {result['plain_accuracy']:.1%}  score {score:.3f}", flush=True)

    individual.sort(key=lambda e: e[0], reverse=True)
    order = [head for _, head, _ in individual]
    print(f"\nRe-ranked order: {format_heads(order)}", flush=True)

    print("\n-- Cumulative ablation --", flush=True)
    cumulative = []
    for k in range(1, len(order) + 1):
        result = evaluate(model, tok, pair, spec, order[:k], device,
                          args.batch_size, args.last_token_only)
        cumulative.append(result)
        print(f"  {k:2d} heads  plain {result['plain_accuracy']:.1%}  "
              f"deceptive {result['deceptive_accuracy']:.1%}  "
              f"sycophancy {result['sycophancy_rate']:.1%}", flush=True)

    def column(rows, key):
        return np.array([r[key] for r in rows], dtype=np.float32)

    save_npz(run_path(args.model, condition.name, "head_ablation.npz"),
             candidate_heads=np.array(candidates),
             ablation_order=np.array(order),
             individual_plain_accuracy=column([r for _, _, r in individual],
                                              "plain_accuracy"),
             individual_deceptive_accuracy=column([r for _, _, r in individual],
                                                  "deceptive_accuracy"),
             individual_sycophancy_rate=column([r for _, _, r in individual],
                                               "sycophancy_rate"),
             cumulative_plain_accuracy=column(cumulative, "plain_accuracy"),
             cumulative_deceptive_accuracy=column(cumulative, "deceptive_accuracy"),
             cumulative_sycophancy_rate=column(cumulative, "sycophancy_rate"),
             unablated_plain_accuracy=unablated["plain_accuracy"],
             unablated_deceptive_accuracy=unablated["deceptive_accuracy"],
             unablated_sycophancy_rate=unablated["sycophancy_rate"],
             last_token_only=args.last_token_only)


if __name__ == "__main__":
    main()
