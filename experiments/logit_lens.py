#!/usr/bin/env python3
"""Logit lens over the last token's residual stream (paper Section 3.1, Fig 3A).

At each layer, applies the model's final norm and unembedding to the last
token's residual stream, then takes the softmax restricted to the four answer
letters. This shows when the model's answer becomes readable in the residual
stream, in the plain run and in a run with a stated opinion.

Both curves start near chance (0.25) and diverge only after the critical layer,
which is the first evidence that answer retrieval happens late. The logit lens
is blind to information not yet in an output-ready form, which is why the
patching experiments are needed to locate the opinion itself.

Multiple choice only: it reads a fixed four-way restricted softmax, so it does
not apply to the free-form TriviaQA condition.

Output: runs/<model>/<condition>/logit_lens.npz
  p_correct_source, p_syco_source, p_other_source  (n_layers, N)
  p_correct_target, p_syco_target, p_other_target  (n_layers, N)
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from syco.conditions import get_condition
from syco.hooks import cache_residual_stream
from syco.runtime import base_parser, batches, load_model, run_path, save_npz


def collect(model, tok, prompts, spec, correct_idx, wrong_idx, device, batch_size,
            label):
    """Restricted answer probabilities at every layer.

    Returns three (n_layers, N) arrays: the probability of the correct answer,
    of the sycophantic answer, and the mean of the two remaining answers.
    """
    n = len(prompts)
    p_correct = np.zeros((spec.n_layers, n), dtype=np.float32)
    p_syco = np.zeros((spec.n_layers, n), dtype=np.float32)
    p_other = np.zeros((spec.n_layers, n), dtype=np.float32)
    answer_ids = list(spec.answer_ids)

    for start, batch in batches(prompts, batch_size):
        store = {}
        with cache_residual_stream(model, range(spec.n_layers), store):
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                model(**enc)

        norm_device = next(model.model.norm.parameters()).device
        sl = slice(start, start + len(batch))
        rows = np.arange(len(batch))
        with torch.no_grad():
            for layer in range(spec.n_layers):
                hidden = store[layer][0].to(norm_device)
                logits = model.lm_head(model.model.norm(hidden)).float()
                probs = torch.softmax(logits[:, answer_ids], dim=1).cpu().numpy()
                pc = probs[rows, correct_idx[sl]]
                pw = probs[rows, wrong_idx[sl]]
                p_correct[layer, sl] = pc
                p_syco[layer, sl] = pw
                # The other two answers, averaged.
                p_other[layer, sl] = (1.0 - pc - pw) / 2.0
        print(f"  {label} [{start + len(batch)}/{n}]", flush=True)

    return p_correct, p_syco, p_other


def main():
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--condition", default="mmlu")
    parser.set_defaults(batch_size=16)
    args = parser.parse_args()

    condition = get_condition(args.condition)
    if condition.name == "triviaqa":
        raise SystemExit(
            "The logit lens here reads a four-way restricted softmax and does "
            "not apply to the free-form TriviaQA condition."
        )

    spec, tok, model, device = load_model(args.model)
    pair = condition.build(spec, tok, args.model, limit=args.limit)
    print(f"\nCondition {condition.name}, N={len(pair)}", flush=True)

    print(f"\n{condition.source_label} run...", flush=True)
    src = collect(model, tok, pair.source, spec, pair.correct_idx, pair.wrong_idx,
                  device, args.batch_size, condition.source_label)
    print(f"\n{condition.target_label} run...", flush=True)
    tgt = collect(model, tok, pair.target, spec, pair.correct_idx, pair.wrong_idx,
                  device, args.batch_size, condition.target_label)

    save_npz(run_path(args.model, condition.name, "logit_lens.npz"),
             p_correct_source=src[0], p_syco_source=src[1], p_other_source=src[2],
             p_correct_target=tgt[0], p_syco_target=tgt[1], p_other_target=tgt[2],
             critical_layer=spec.critical_layer)


if __name__ == "__main__":
    main()
