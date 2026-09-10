"""Cross-question patching: does an opinion head carry a reference or content?

The question (paper, Appendix D.4)
----------------------------------
Opinion heads encode the stated opinion in a form that is not yet output-ready.
Two accounts fit that:

  reference  the head deposits a reference identifying the user's preferred
             answer, which a later retrieval step resolves into the answer
  content    the head already carries the answer, which a later step merely
             puts into output-ready form

The logit lens may fail to decode either, so depth separation alone cannot
separate them. What does separate them is whether the head's output means
anything OUTSIDE the context it came from. A reference is defined relative to
its own prompt: transplanted elsewhere, the context needed to resolve it is
gone. Content encodes the answer itself and should survive the move.

The design
----------
Pair TriviaQA questions by a random derangement, so no question is paired with
itself. Run both under the deceptive prompt, each with its own stated wrong
answer, and call one the source and the other the target. For each head, replace
its output at the target's final token position with the source's, and measure
the change in log-probability of the first token of the SOURCE question's stated
wrong answer, patched against unpatched.

    content    predicts a large increase -- the answer travels
    reference  predicts little change -- the transplanted vector is
               uninformative in a context that never mentioned that answer

The paper finds answer retrieval heads produce a substantial increase and
opinion heads about one fifth of it, which supports the reference account.

Read-outs
---------
Three first-token read-outs are stored per (layer, head, pair), all relative to
the unpatched deceptive target. The paper's figure uses only the third; the
other two say what the patch did to the target's own answers, which is what
distinguishes "the source's answer moved in" from "the target's answer moved
out":

    c_t  the target's correct answer      rises -> the patch removed the opinion
    w_t  the target's own stated opinion  falls -> the patch displaced its reference
    w_s  the SOURCE's stated opinion      rises -> content moved across questions

Size caveat
-----------
Quote this with the magnitude attached. In Llama the source question's answer
has a median probability around 1.5e-07 in the target run, and the largest
effect moves it to about 2.5e-07. These are directions written into the residual
stream, not the model approaching that answer.

The optional control arm
------------------------
The paper's measure needs only the deceptive arm. This module also supports a
second arm in which the source states no opinion at all, which isolates what the
source's OPINION contributed from generic cross-context disruption. Treat it as
a diagnostic, not as the headline: it is a sound scalar summary but a poor
per-head map, because at heads where the two arms differ mostly in how much they
restore the target's correct answer, the contrast inverts the sign of what
actually happened. Both arms must use the same --seed or the pairings differ;
`load_arms` refuses to proceed if they disagree.
"""

import os

import numpy as np

__all__ = [
    "READOUTS", "TARGET_CORRECT", "TARGET_OPINION",
    "SOURCE_OPINION", "MAX_GUESS_WORDS",
    "make_pairs", "load_arm", "load_arms", "delta_log_prob", "per_head_effect",
    "population_summary", "bootstrap_ci",
]

READOUTS = ("c_t", "w_t", "w_s")
TARGET_CORRECT, TARGET_OPINION, SOURCE_OPINION = 0, 1, 2

# Beam search occasionally returns a garbled multi-clause string instead of a
# short answer. A first-token logit comparison is exactly where that corrupts
# results silently, so this analysis applies a length filter on top of whatever
# the LLM judge already removed during screening.
MAX_GUESS_WORDS = 6


def make_pairs(correct_ids, wrong_ids, seed):
    """Pair each question with a different one, as a random derangement.

    Pairs are then dropped where the source's stated answer collides with either
    of the target's two read-out tokens: the three read-outs have to be distinct
    or they contaminate each other. c_t and w_t are already guaranteed distinct
    by screening, so only w_s has to be checked.
    """
    n = len(correct_ids)
    rng = np.random.default_rng(seed)
    while True:
        permutation = rng.permutation(n)
        if not np.any(permutation == np.arange(n)):
            break

    targets, sources = [], []
    for i in range(n):
        j = permutation[i]
        if wrong_ids[j] == correct_ids[i] or wrong_ids[j] == wrong_ids[i]:
            continue
        targets.append(i)
        sources.append(j)

    dropped = n - len(targets)
    print(f"Paired {n} questions -> {len(targets)} usable pairs "
          f"({dropped} dropped for read-out token collision)", flush=True)
    return np.array(targets), np.array(sources)


def load_arm(model_key, arm, condition="triviaqa"):
    """Load one arm, or explain how to produce it."""
    from .runtime import load_npz, run_path

    path = run_path(model_key, condition, f"cross_question_{arm}.npz",
                    create=False)
    if not os.path.exists(path):
        raise SystemExit(
            f"Missing {path}\nRun experiments/patch_cross_question.py "
            f"--model {model_key} --source_arm {arm} first."
        )
    return load_npz(path)


def load_arms(model_key, condition="triviaqa", require_control=False):
    """Load the deceptive arm, and the control arm if it exists.

    Returns (dec, plain_or_None). The paper's measure needs only `dec`, so a
    missing control arm is not an error unless `require_control` is set.
    """
    dec = load_arm(model_key, "dec", condition)
    try:
        plain = load_arm(model_key, "plain", condition)
    except SystemExit:
        if require_control:
            raise
        return dec, None

    if not (np.array_equal(dec["t_idx"], plain["t_idx"])
            and np.array_equal(dec["s_idx"], plain["s_idx"])):
        raise SystemExit(
            "The two arms use different pairings. Re-run both with the same "
            "--seed; the contrast between them is otherwise meaningless."
        )

    # The target prompts are identical in both arms, so their unpatched
    # baselines must agree. If they do not, the arms saw different data.
    drift = float(np.abs(dec["base_logits"] - plain["base_logits"]).max())
    if drift > 1e-3:
        print(f"WARNING: the arms' baselines differ by up to {drift:.4f} on "
              f"identical target prompts (expected ~0). Check that both ran "
              f"against the same screening file.\n", flush=True)
    return dec, plain


def delta_log_prob(arm):
    """Patched minus unpatched log-probabilities, (n_layers, n_heads, 3, pairs).

    Log-probabilities rather than logits: the stored logsumexp makes the
    normalization recoverable, and the source's answer sits so far down the
    distribution that an unnormalized comparison would not be interpretable.
    """
    patched = arm["patched_logits"] - arm["patched_logZ"][:, :, None, :]
    base = arm["base_logits"] - arm["base_logZ"][None, :]
    return patched - base[None, None]


def per_head_effect(delta, heads, readout=SOURCE_OPINION):
    """One number per head: its mean effect over question pairs."""
    if len(heads) == 0:
        return np.array([])
    layers = np.array([l for l, _ in heads])
    indices = np.array([h for _, h in heads])
    return delta[layers, indices, readout, :].mean(axis=1)


def bootstrap_ci(values, n_resamples=10000, seed=0):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_resamples, len(values)))
    return tuple(np.percentile(values[draws].mean(axis=1), [2.5, 97.5]))


def population_summary(dec_delta, heads, plain_delta=None, seed=0):
    """Summarize one head population.

    The dec-arm effects use the standard error ACROSS heads: head-to-head spread
    dominates pair noise here, so that is the honest error bar for a claim about
    a population. The control contrast, when the second arm was run, is
    bootstrapped over pairs, since it is a per-pair difference.
    """
    if len(heads) == 0:
        return None
    layers = np.array([l for l, _ in heads])
    indices = np.array([h for _, h in heads])

    summary = {"n_heads": len(heads)}
    for k, name in enumerate(READOUTS):
        per_head = dec_delta[layers, indices, k, :].mean(axis=1)
        summary[name] = float(per_head.mean())
        summary[f"{name}_sem"] = (
            float(per_head.std(ddof=1) / np.sqrt(len(per_head)))
            if len(per_head) > 1 else float("nan")
        )

    # The fraction of pairs moving in the predicted direction, which separates a
    # consistent push from a few large outliers.
    summary["pairs_positive"] = float(
        (dec_delta[layers, indices, SOURCE_OPINION, :].mean(axis=0) > 0).mean()
    )

    if plain_delta is not None:
        per_pair = (dec_delta[layers, indices, SOURCE_OPINION, :]
                    - plain_delta[layers, indices, SOURCE_OPINION, :]).mean(axis=0)
        summary["transfer"] = float(per_pair.mean())
        summary["transfer_ci"] = bootstrap_ci(per_pair, seed=seed)
    return summary
