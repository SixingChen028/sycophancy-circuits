"""Selecting head populations from a patching sweep.

The critical layer is the last layer at which the median plain-to-label-swap
normalized logit difference remains below 0.1 (paper, Appendix B.1). It splits
every head into one of two bands (paper, Section 3.2):

  early heads  layers 0 .. critical_layer. These show a causal effect in the
               plain-to-deceptive patching but not in the plain-to-label-swap
               patching, so their effect is specific to the presence of a stated
               opinion. These are the OPINION heads.

  late heads   layers critical_layer+1 .. last. These show a causal effect in
               both conditions, so they perform generic answer selection
               whatever the answer turns out to be. These are the ANSWER
               RETRIEVAL heads.
"""

import numpy as np

__all__ = ["band_mask", "rank_heads", "top_heads", "format_heads",
           "opinion_population", "retrieval_population", "complement"]


def band_mask(spec, band):
    """Boolean (n_layers, n_heads) mask selecting one side of the critical layer."""
    mask = np.zeros((spec.n_layers, spec.n_heads), dtype=bool)
    if band == "early":
        mask[: spec.critical_layer + 1] = True
    elif band == "late":
        mask[spec.critical_layer + 1:] = True
    elif band == "all":
        mask[:] = True
    else:
        raise ValueError(f"band must be 'early', 'late' or 'all', got {band!r}")
    return mask


def rank_heads(scores, spec, band="all"):
    """All (layer, head) pairs in `band`, ordered by descending score."""
    mask = band_mask(spec, band)
    entries = [
        (float(scores[l, h]), l, h)
        for l in range(spec.n_layers)
        for h in range(spec.n_heads)
        if mask[l, h]
    ]
    entries.sort(key=lambda e: e[0], reverse=True)
    return [(l, h, s) for s, l, h in entries]


def top_heads(scores, spec, band="early", k=10):
    """The k highest-scoring heads in `band`, as (layer, head) pairs."""
    return [(l, h) for l, h, _ in rank_heads(scores, spec, band)[:k]]


def format_heads(heads):
    return ", ".join(f"L{l}H{h}" for l, h in heads)


# ── The paper's two head populations ─────────────────────────────────────────
#
# Defined once here so every analysis that needs them agrees. Both read the
# per-head sweeps, so the sweeps must exist first.


def opinion_population(model_key, spec, k=10, condition="mmlu"):
    """Top-k early heads from the plain-to-deceptive sweep."""
    from .runtime import load_npz, run_path

    scores = load_npz(
        run_path(model_key, condition, "head_patching.npz", create=False)
    )["norm_logit_diff"]
    return top_heads(scores, spec, band="early", k=k)


def retrieval_population(model_key, spec, k=10):
    """Top-k late heads, scored across both relabelling conditions.

    A head counts as an answer retrieval head only if it tracks the correct
    answer under BOTH kinds of relabelling, so the two sweeps are averaged
    rather than either being used alone.
    """
    from .runtime import load_npz, run_path

    scores = [
        load_npz(run_path(model_key, condition, "head_patching.npz",
                          create=False))["norm_logit_diff"]
        for condition in ("mmlu_label_swap", "mmlu_content_swap")
    ]
    return top_heads(np.mean(scores, axis=0), spec, band="late", k=k)


def complement(population, spec, band):
    """Every head in `band` that is not in `population`.

    The other heads in the same layers are the within-band control: they hold
    depth fixed, so a difference between them and the population cannot be
    explained by layer alone.
    """
    selected = set(map(tuple, population))
    mask = band_mask(spec, band)
    return [(l, h)
            for l in range(spec.n_layers) for h in range(spec.n_heads)
            if mask[l, h] and (l, h) not in selected]
