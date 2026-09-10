#!/usr/bin/env python3
"""Cross-question head patching on TriviaQA (paper Appendix D.4, Figure S5).

Does an opinion head carry a REFERENCE to the user's preferred answer, or the
answer CONTENT itself? Patching a head from one question's run into a DIFFERENT
question's run separates them: content should stay meaningful in the new
context, a reference should not. See `syco.crossquestion` for the argument and
the size caveat that governs how these numbers may be quoted.

The paper's figure needs only the deceptive arm:

    python experiments/patch_cross_question.py --model llama --source_arm dec

An optional control arm, in which the source states no opinion at all, isolates
what the source's opinion contributed from generic cross-context disruption. It
is a diagnostic rather than the headline. If you run it, use the SAME seed --
the arms are comparable only under the same pairing:

    python experiments/patch_cross_question.py --model llama --source_arm plain --seed 0

Why TriviaQA and not MMLU: in MMLU the source's stated answer is almost never an
option in the target question, so no logit registers it. TriviaQA reads out over
the whole vocabulary, so the source's answer is always measurable however
unrelated the two questions are. The model produces incoherent text under this
patch, which is expected -- the measurement is one token's logit, not an answer.

Each arm is a full sweep over every head, checkpointed per layer; resubmit the
identical command to resume after a timeout.

Output: runs/<model>/triviaqa/cross_question_<arm>.npz
  patched_logits (n_layers, n_heads, 3, pairs)   read-outs [c_t, w_t, w_s]
  patched_logZ   (n_layers, n_heads, pairs)      full-vocabulary logsumexp
  base_logits    (3, pairs) and base_logZ (pairs)
  t_idx, s_idx, ct_ids, wt_ids, ws_ids, target_qids, source_qids
"""

import contextlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from syco import prompts as P
from syco.crossquestion import MAX_GUESS_WORDS, READOUTS, make_pairs
from syco.data import first_token_id, load_screened
from syco.hooks import cache_o_proj_inputs, patch_head_outputs
from syco.runtime import base_parser, batches, load_model, run_path, save_npz


def usable_examples(tok, model_key, limit, max_guess_words):
    """Screened questions, minus those a first-token read-out cannot measure."""
    path = run_path(model_key, "triviaqa", "screened_examples.json", create=False)
    records = load_screened(path, limit=limit)

    examples, correct_ids, wrong_ids = [], [], []
    n_garbled = n_collision = 0
    for record in records:
        if len(record["wrong_guess"].split()) > max_guess_words:
            n_garbled += 1
            continue
        correct = first_token_id(tok, record["plain_raw_answer"])
        wrong = first_token_id(tok, record["wrong_guess"])
        if correct == wrong:
            n_collision += 1
            continue
        examples.append(record)
        correct_ids.append(correct)
        wrong_ids.append(wrong)

    print(f"{len(records)} screened; dropped {n_garbled} for a long wrong answer "
          f"and {n_collision} for a shared first token -> {len(examples)} usable",
          flush=True)
    return examples, np.array(correct_ids), np.array(wrong_ids)


def collect(model, tok, prompts, token_ids, device, batch_size, patch=None,
            label=""):
    """Read-out logits and the full-vocabulary logsumexp for each prompt.

    `patch` is None or (spec, layer, head, source_cache); the source cache is
    indexed to match the prompt list position by position.

    The logsumexp is stored so that log-probabilities can be recovered later
    without re-running: the source's answer sits far enough down the
    distribution that raw logits are not interpretable on their own.
    """
    n = len(prompts)
    logits = np.zeros((len(READOUTS), n), dtype=np.float32)
    log_z = np.zeros(n, dtype=np.float32)

    for start, batch in batches(prompts, batch_size):
        window = slice(start, start + len(batch))
        if patch is None:
            intervention = contextlib.nullcontext()
        else:
            spec, layer, head, cache = patch
            intervention = patch_head_outputs(
                model, spec, [(layer, head)], {layer: cache[window].to(device)}
            )

        encoded = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to(device)
        with intervention:
            with torch.no_grad():
                out = model(**encoded, logits_to_keep=1)

        final = out.logits[:, -1, :].float()
        rows = torch.arange(len(batch), device=final.device)
        for k in range(len(READOUTS)):
            ids = torch.as_tensor(token_ids[k][window], dtype=torch.long,
                                  device=final.device)
            logits[k, window] = final[rows, ids].cpu().numpy()
        log_z[window] = torch.logsumexp(final, dim=-1).cpu().numpy()
        if label:
            print(f"  {label} [{start + len(batch)}/{n}]", flush=True)

    return logits, log_z


def cache_sources(model, tok, prompts, spec, device, batch_size):
    store = {}
    with cache_o_proj_inputs(model, range(spec.n_layers), store):
        for start, batch in batches(prompts, batch_size):
            encoded = tok(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                model(**encoded, logits_to_keep=1)
            print(f"  source cache [{start + len(batch)}/{len(prompts)}]",
                  flush=True)

    cache = {l: torch.cat(store[l], dim=0) for l in range(spec.n_layers)}
    width = cache[0].shape[1]
    if width != spec.n_heads * spec.d_head:
        raise RuntimeError(
            f"o_proj input width {width} != n_heads*d_head "
            f"({spec.n_heads * spec.d_head}). The head geometry is wrong."
        )
    return cache


def main():
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--source_arm", required=True, choices=["dec", "plain"],
                        help="dec: the source states its own opinion. "
                             "plain: the source states none (the control arm).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Pairing seed. MUST match across the two arms.")
    parser.add_argument("--cache_batch_size", type=int, default=64,
                        help="Batch size for the source-activation pass only")
    parser.add_argument("--max_guess_words", type=int, default=MAX_GUESS_WORDS,
                        help="Drop questions whose stated wrong answer is longer "
                             "than this, on top of the screening judge")
    parser.set_defaults(batch_size=128)
    args = parser.parse_args()

    out_file = run_path(args.model, "triviaqa",
                        f"cross_question_{args.source_arm}.npz")
    ckpt_file = run_path(args.model, "triviaqa",
                         f"cross_question_{args.source_arm}.ckpt.npz")

    spec, tok, model, device = load_model(args.model)
    examples, correct_ids, wrong_ids = usable_examples(
        tok, args.model, args.limit, args.max_guess_words
    )
    targets, sources = make_pairs(correct_ids, wrong_ids, args.seed)
    n_pairs = len(targets)

    # Read-out token ids, aligned position by position with the target list.
    ct_ids = correct_ids[targets]
    wt_ids = wrong_ids[targets]
    ws_ids = wrong_ids[sources]
    token_ids = (ct_ids, wt_ids, ws_ids)

    target_prompts = [
        P.triviaqa_deceptive(tok, examples[i]["question"],
                             examples[i]["wrong_guess"]) for i in targets
    ]
    if args.source_arm == "dec":
        source_prompts = [
            P.triviaqa_deceptive(tok, examples[j]["question"],
                                 examples[j]["wrong_guess"]) for j in sources
        ]
    else:
        source_prompts = [P.triviaqa_plain(tok, examples[j]["question"])
                          for j in sources]

    print(f"\nArm '{args.source_arm}': {n_pairs} pairs, seed {args.seed}",
          flush=True)

    # The baseline runs at --batch_size, NOT --cache_batch_size, so that it is
    # batched identically to the patched runs. The measurement is
    # (patched - baseline) on raw logits, so if the two used different batch
    # boundaries, bfloat16 nondeterminism would leak straight into the result.
    print("\nUnpatched deceptive-target baseline...", flush=True)
    base_logits, base_log_z = collect(model, tok, target_prompts, token_ids,
                                      device, args.batch_size, label="baseline")
    for k, name in enumerate(READOUTS):
        print(f"  baseline {name}: logit={base_logits[k].mean():+.3f}  "
              f"log-prob={(base_logits[k] - base_log_z).mean():+.3f}", flush=True)

    print(f"\nCaching source activations ({args.source_arm})...", flush=True)
    cache = cache_sources(model, tok, source_prompts, spec, device,
                          args.cache_batch_size)

    shape = (spec.n_layers, spec.n_heads)
    patched_logits = np.full(shape + (len(READOUTS), n_pairs), np.nan,
                             dtype=np.float32)
    patched_log_z = np.full(shape + (n_pairs,), np.nan, dtype=np.float32)
    done = set()

    if os.path.exists(ckpt_file):
        checkpoint = np.load(ckpt_file)
        if checkpoint["patched_logits"].shape == patched_logits.shape:
            patched_logits = checkpoint["patched_logits"]
            patched_log_z = checkpoint["patched_logZ"]
            done = {int(x) for x in checkpoint["done_layers"]}
            print(f"Resuming; {len(done)} layers already done.", flush=True)
        else:
            print("Checkpoint shape does not match this run; starting fresh.",
                  flush=True)

    for layer in range(spec.n_layers):
        if layer in done:
            continue
        for head in range(spec.n_heads):
            logits, log_z = collect(
                model, tok, target_prompts, token_ids, device, args.batch_size,
                patch=(spec, layer, head, cache[layer])
            )
            patched_logits[layer, head] = logits
            patched_log_z[layer, head] = log_z
        delta = patched_logits[layer, :, :, :] - base_logits[None]
        best = np.argsort(delta[:, 2, :].mean(axis=1))[::-1][:3]
        print(f"Layer {layer:2d}  best by d_ws: "
              + "  ".join(f"H{h}={delta[h, 2, :].mean():+.4f}" for h in best),
              flush=True)
        done.add(layer)
        np.savez(ckpt_file, patched_logits=patched_logits,
                 patched_logZ=patched_log_z, done_layers=np.array(sorted(done)))

    save_npz(out_file,
             patched_logits=patched_logits,
             patched_logZ=patched_log_z,
             base_logits=base_logits,
             base_logZ=base_log_z,
             t_idx=targets, s_idx=sources,
             ct_ids=ct_ids, wt_ids=wt_ids, ws_ids=ws_ids,
             target_qids=np.array([examples[i]["question_id"] for i in targets]),
             source_qids=np.array([examples[j]["question_id"] for j in sources]),
             source_arm=args.source_arm, seed=args.seed)
    print("\nRun figures/figS5_cross_question.py to summarize and plot this.",
          flush=True)

    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)


if __name__ == "__main__":
    main()
