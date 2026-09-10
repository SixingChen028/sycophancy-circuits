"""Loading screened examples and choosing the sycophantic answer.

Screening (paper, Appendix A.1) writes one record per candidate question with a
`pass_all` flag; every analysis works on the subset where that is true. Records
are always sorted by `orig_idx` so that example order -- and therefore the
per-example axis of every saved array -- is identical across experiments and
comparable across models.

Selecting the sycophantic answer
--------------------------------
A question can be flipped by more than one incorrect option. Following the
paper, the one kept is the option producing the strongest sycophantic shift:
the option w minimizing logit(correct) - logit(w) in the run that states w.
"""

import json
import os

__all__ = ["load_screened", "strongest_shift", "first_token_id"]


def load_screened(path, limit=None):
    """Read a screening file and return the passing records in a stable order."""
    if not os.path.exists(path):
        raise SystemExit(
            f"Missing screening file: {path}\n"
            f"Run the matching experiments/screen_*.py first."
        )
    with open(path) as f:
        records = json.load(f)
    passing = [r for r in records if r.get("pass_all")]
    passing.sort(key=lambda r: r["orig_idx"])
    if limit:
        passing = passing[:limit]
    if not passing:
        raise SystemExit(f"No examples passed screening in {path}")
    print(f"Loaded {len(passing)} screened examples from {path}", flush=True)
    return passing


def strongest_shift(example, field):
    """Index of the wrong option producing the largest shift toward itself.

    `field` names the per-option logit dictionary to read, e.g. "logits_dec"
    for the deceptive run or "logits_label_swap" for the label-swapped run.
    Keys are option indices stored as strings.
    """
    correct = example["correct_answer"]
    logits = example[field]
    return min(
        (int(w) for w in logits),
        key=lambda w: logits[str(w)][correct] - logits[str(w)][w],
    )


def first_token_id(tok, text):
    """First token of `text` as it appears after "Answer:", i.e. with a space.

    TriviaQA answers are free-form, so the comparison is between the first
    tokens of the correct and the stated wrong answer (paper, Appendix A.3).
    """
    return tok(" " + text, add_special_tokens=False)["input_ids"][0]
