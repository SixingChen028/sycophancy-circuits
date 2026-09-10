"""The paper's outcome measures.

Normalized logit difference (paper, Section 3.1)
------------------------------------------------
For a question, let a_plain be the answer label the model produces under the
source prompt and a_target the label it produces under the target prompt (the
label named by the opinion in the deceptive run, or the label attached to the
correct content in the label-swapped run). With

    delta      = logit(a_plain) - logit(a_target)
    delta_src  = delta in the source run
    delta_tgt  = delta in the target run
    delta_p(l) = delta after patching component l from source into target

the normalized logit difference is

    D(l) = (delta_p(l) - delta_tgt) / (delta_src - delta_tgt)

D = 0 means the patch produces no recovery toward the source run's answer, so
that component does not mediate the shift. D = 1 means the patch fully restores
it, so that component fully mediates it.

The per-example denominator can be zero when the two runs happen to give the
same logit difference; those examples are excluded from the mean rather than
producing infinities, which is why the aggregate uses nanmean.
"""

import warnings

import numpy as np

__all__ = [
    "logit_difference",
    "normalized_logit_difference",
    "aggregate",
    "accuracy_summary",
]


def logit_difference(scores, correct_col, wrong_col):
    """Per-example logit(correct) - logit(wrong) from a (N, K) score array."""
    rows = np.arange(len(scores))
    return scores[rows, correct_col] - scores[rows, wrong_col]


def normalized_logit_difference(patched_ld, source_ld, target_ld):
    """Per-example normalized logit difference; NaN where the denominator is 0."""
    denom = np.asarray(source_ld, dtype=np.float64) - np.asarray(target_ld, dtype=np.float64)
    denom = np.where(denom != 0, denom, np.nan)
    return ((np.asarray(patched_ld, dtype=np.float64) - np.asarray(target_ld, dtype=np.float64))
            / denom).astype(np.float32)


def aggregate(per_example):
    """Mean over examples, ignoring the excluded zero-denominator cases.

    Returns NaN if every example was excluded, rather than warning: that is a
    legitimate outcome (the two runs gave identical logit differences
    throughout), and the caller sees it as NaN in the saved array.
    """
    values = np.asarray(per_example, dtype=np.float64)
    if not np.isfinite(values).any():
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(values))


def accuracy_summary(scores, correct_col, wrong_col):
    """Answer rates for a multiple-choice run.

    Returns plain-language rates used throughout the ablation analyses:
      accuracy   fraction answering the correct option
      sycophancy fraction answering the stated wrong option
    """
    chosen = np.argmax(scores, axis=1)
    return {
        "accuracy": float((chosen == correct_col).mean()),
        "sycophancy": float((chosen == wrong_col).mean()),
        "correct_per_example": (chosen == correct_col),
        "sycophantic_per_example": (chosen == wrong_col),
    }
