#!/usr/bin/env python3
"""Are the opinion heads just induction heads? (paper Appendix D.3, Figure S4)

Opinion heads are recruited by a cue that names a target answer, so one worry is
that they are generic induction heads -- heads that attend to the token following
an earlier occurrence of the current token and copy it forward. Such a head would
show the same causal signature whenever the opinion and the answer share a token.

Both standard criteria of Olsson et al. (2022) are computed for every head on the
same repeated-random-token sequences, [BOS, t_1..t_R, t_1..t_R]:

  Prefix matching  At query position R+i (token t_i), the induction target is
                   position i+1 (token t_{i+1}, what followed t_i last time).
                   The score is the attention weight on that target, averaged
                   over query positions and sequences. Null value 0.

  Copying          Whether the head actually copies what it attends to, measured
                   by direct logit attribution: take the head's write-in at that
                   position and project it onto the unembedding row of t_{i+1}.

A head counts as an induction head only if it exceeds the 95th percentile of both
scores. The paper finds no opinion head meets both.

Why copying is measured by direct logit attribution
---------------------------------------------------
The eigenvalue-based copying score of Olsson et al. is the fraction of
eigenvalues of W_U W_OV W_E^T with positive real part. For a model with tied
embeddings that reduces to a congruence of W_OV, so by Sylvester's law of inertia
it reports the inertia of sym(W_OV) rather than copying, forcing near-definite
heads to exactly 1.0 or 0.0. Direct logit attribution never forms that product,
is defined for tied and untied models alike, and is comparable across them.

Two matched controls remove the offset that comes from a token merely having
appeared in the sequence:

  control  DLA to t_{j+1} for a random j != i -- also in the sequence, same
           familiarity, but not the induction target. Cancels in the difference,
           which is the primary score.
  z-score  DLA to the true target, standardized against vocabulary tokens absent
           from the sequence. Scale-free, so comparable across models.

Uses no dataset: the sequences are random tokens. Runs in minutes.
Output: runs/<model>/induction/induction_heads.npz
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from syco.heads import format_heads, rank_heads
from syco.hooks import cache_o_proj_inputs, layer_modules
from syco.models import get_model_spec, verify_geometry
from syco.runtime import base_parser, load_npz, run_path, save_npz


def repeated_random_sequences(tok, n_seqs, length, rng):
    """[BOS, t_1..t_R, t_1..t_R] with tokens drawn from mid-vocabulary.

    The mid-vocabulary range avoids the special and byte-level tokens clustered
    at both ends, which behave unlike ordinary tokens.
    """
    low, high = 1000, tok.vocab_size - 1000
    seqs = np.empty((n_seqs, 2 * length + 1), dtype=np.int64)
    for i in range(n_seqs):
        tokens = rng.integers(low, high, size=length)
        seqs[i, 0] = tok.bos_token_id
        seqs[i, 1:length + 1] = tokens
        seqs[i, length + 1:] = tokens
    return seqs, (low, high)


def final_norm_gain(model):
    """The final norm's learned gain, for folding into the unembedding.

    The RMS divisor is one positive scalar per position, so it scales every
    logit at that position equally and cannot change which token a head favours.
    Gemma applies (1 + weight); Llama and Mistral apply weight directly.
    """
    weight = model.model.norm.weight.detach().float()
    gain = (1.0 + weight) if "Gemma" in type(model.model.norm).__name__ else weight
    tolerance = 1e-3 * gain.abs().median()
    if (gain < -tolerance).any():
        raise RuntimeError(
            "Final-norm gain has meaningfully negative entries; folding it into "
            "the unembedding would flip logit signs. Inspect before proceeding."
        )
    return gain


def prefix_matching(model, spec, input_ids, length, device):
    """Attention weight on the induction target, per (layer, head).

    Raw attention weights are only exposed under the eager attention
    implementation; sdpa and flash-attention do not return them.
    """
    with torch.no_grad():
        out = model(input_ids, output_attentions=True)
    scores = np.zeros((spec.n_layers, spec.n_heads), dtype=np.float64)
    queries = torch.arange(1, length, device=device) + length   # R+1 .. 2R-1
    targets = torch.arange(1, length, device=device) + 1        # 2 .. R
    positions = torch.arange(len(queries), device=device)
    for layer in range(spec.n_layers):
        attn = out.attentions[layer].float()
        selected = attn[:, :, queries, :][:, :, positions, targets]
        scores[layer] = selected.mean(dim=(0, 2)).cpu().numpy()
    del out
    return scores


def copying(model, spec, input_ids, seqs, length, unembed, rng, n_control_vocab,
            vocab_range, device):
    """Direct logit attribution to the induction target, with both controls."""
    store = {}
    with cache_o_proj_inputs_all_positions(model, range(spec.n_layers), store):
        with torch.no_grad():
            model(input_ids)

    n_seqs = input_ids.shape[0]
    query_offsets = np.arange(1, length)
    target_tokens = seqs[:, query_offsets + 2]

    # Matched control: t_{j+1} for a random j != i, drawn per (sequence, query).
    control_tokens = np.empty_like(target_tokens)
    for b in range(n_seqs):
        for k, i in enumerate(query_offsets):
            j = i
            while j == i:
                j = rng.integers(1, length)
            control_tokens[b, k] = seqs[b, j + 2]

    # Null pool: vocabulary tokens that do not appear anywhere in the batch.
    low, high = vocab_range
    present = set(seqs.ravel().tolist())
    sampled = rng.integers(low, high, size=n_control_vocab * 3)
    pool = np.array([t for t in sampled if t not in present][:n_control_vocab])

    u_target = unembed[torch.as_tensor(target_tokens, device=unembed.device)]
    u_control = unembed[torch.as_tensor(control_tokens, device=unembed.device)]
    u_pool = unembed[torch.as_tensor(pool, device=unembed.device)]

    queries = torch.as_tensor(query_offsets + length, device=device)
    correct = np.zeros((spec.n_layers, spec.n_heads), dtype=np.float64)
    control = np.zeros((spec.n_layers, spec.n_heads), dtype=np.float64)
    zscore = np.zeros((spec.n_layers, spec.n_heads), dtype=np.float64)

    with torch.no_grad():
        for layer in range(spec.n_layers):
            activations = store[layer][0][:, queries, :].to(unembed.device)
            weight = layer_modules(model)[layer].self_attn.o_proj.weight.detach().float()
            weight = weight.to(unembed.device)
            for head in range(spec.n_heads):
                sl = spec.head_slice(head)
                writein = activations[:, :, sl] @ weight[:, sl].T
                dla_target = (writein * u_target).sum(-1)
                dla_control = (writein * u_control).sum(-1)
                pool_dla = writein @ u_pool.T
                mean = pool_dla.mean(-1)
                std = pool_dla.std(-1).clamp_min(1e-6)
                correct[layer, head] = dla_target.mean().item()
                control[layer, head] = dla_control.mean().item()
                zscore[layer, head] = ((dla_target - mean) / std).mean().item()
    return correct, control, zscore


from contextlib import contextmanager


@contextmanager
def cache_o_proj_inputs_all_positions(model, layers, store):
    """Like syco.hooks.cache_o_proj_inputs but keeps every position.

    Copying is measured at the repeated-sequence query positions, not at the
    last token, so the full sequence is needed here.
    """
    handles = []
    for layer in layers:
        def make_hook(idx):
            def hook(module, args):
                store.setdefault(idx, []).append(args[0].detach().float())
            return hook

        handles.append(
            layer_modules(model)[layer].self_attn.o_proj.register_forward_pre_hook(
                make_hook(layer)
            )
        )
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def main():
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--length", type=int, default=50,
                        help="R, the length of the repeated random block")
    parser.add_argument("--n_seqs", type=int, default=10, help="Sequences per batch")
    parser.add_argument("--n_batches", type=int, default=3)
    parser.add_argument("--n_control_vocab", type=int, default=512,
                        help="Off-sequence tokens forming the z-score null")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare_condition", default="mmlu",
                        help="Condition whose top early heads are reported "
                             "against the two criteria")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    spec = get_model_spec(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {spec.hf_id} on {device} "
          f"(eager attention, required for attention weights)...", flush=True)
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    verify_geometry(spec, model)

    unembed = model.lm_head.weight.detach().float()
    unembed = unembed * final_norm_gain(model).to(unembed.device)

    prefix = np.zeros((spec.n_layers, spec.n_heads), dtype=np.float64)
    correct = np.zeros_like(prefix)
    control = np.zeros_like(prefix)
    zscore = np.zeros_like(prefix)

    for batch in range(args.n_batches):
        print(f"\nBatch {batch + 1}/{args.n_batches}...", flush=True)
        rng = np.random.default_rng(args.seed + batch)
        seqs, vocab_range = repeated_random_sequences(tok, args.n_seqs,
                                                      args.length, rng)
        input_ids = torch.as_tensor(seqs, dtype=torch.long, device=device)

        prefix += prefix_matching(model, spec, input_ids, args.length, device)
        c, k, z = copying(model, spec, input_ids, seqs, args.length, unembed,
                          np.random.default_rng(args.seed + 12345 + batch),
                          args.n_control_vocab, vocab_range, device)
        correct += c
        control += k
        zscore += z
        if device == "cuda":
            torch.cuda.empty_cache()

    prefix /= args.n_batches
    correct /= args.n_batches
    control /= args.n_batches
    zscore /= args.n_batches
    copy_score = correct - control

    prefix_threshold = np.percentile(prefix, 95)
    copy_threshold = np.percentile(copy_score, 95)
    induction = [
        (l, h)
        for l in range(spec.n_layers) for h in range(spec.n_heads)
        if prefix[l, h] > prefix_threshold and copy_score[l, h] > copy_threshold
    ]
    print(f"\n95th percentile: prefix {prefix_threshold:.4f}, "
          f"copying {copy_threshold:.4f}")
    print(f"Heads meeting both criteria (n={len(induction)}): "
          f"{format_heads(induction)}")

    sweep_path = run_path(args.model, args.compare_condition, "head_patching.npz",
                          create=False)
    if os.path.exists(sweep_path):
        sweep = load_npz(sweep_path)
        opinion = [(l, h) for l, h, _ in
                   rank_heads(sweep["norm_logit_diff"], spec, "early")[:10]]
        overlap = [h for h in opinion if h in induction]
        print(f"\nOpinion heads ({args.compare_condition}):")
        for layer, head in opinion:
            flag = "  <- induction head" if (layer, head) in induction else ""
            print(f"  L{layer:2d}H{head:2d}  prefix={prefix[layer, head]:.4f}  "
                  f"copying={copy_score[layer, head]:+.4f}  "
                  f"z={zscore[layer, head]:+.2f}{flag}")
        print(f"\n{len(overlap)} of {len(opinion)} opinion heads meet both criteria.")
    else:
        opinion = []
        print(f"\n(No head_patching.npz for {args.compare_condition}; "
              f"skipping the opinion-head comparison.)")

    save_npz(run_path(args.model, "induction", "induction_heads.npz"),
             prefix_matching_score=prefix,
             copying_score=copy_score,
             copying_zscore=zscore,
             dla_correct=correct,
             dla_control=control,
             prefix_threshold=prefix_threshold,
             copy_threshold=copy_threshold,
             induction_heads=np.array(induction, dtype=np.int64).reshape(-1, 2),
             opinion_heads=np.array(opinion, dtype=np.int64).reshape(-1, 2),
             length=args.length,
             n_seqs=args.n_seqs * args.n_batches)


if __name__ == "__main__":
    main()
