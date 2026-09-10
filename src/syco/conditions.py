"""The experimental conditions, as source/target prompt pairs.

Every causal-mediation experiment in the paper has the same shape: run the model
on a SOURCE prompt where it answers correctly, run it on a TARGET prompt where
it answers differently, then patch a component from source into target and ask
how far the answer moves back. Conditions differ only in which two prompts those
are, which screening file supplies the examples, and how the answer is read out.

Registering all of that here is what collapses the original code base -- one
copy of each analysis per (model x condition) -- into a single script per
analysis.

    condition          source            target                        paper
    -----------------  ----------------  ----------------------------  ----------
    mmlu               plain             stated opinion, by letter     Fig 3, 4
    mmlu_label_swap    plain             correct/wrong letters swapped Fig 3B, 4A
    mmlu_content_swap  plain             correct/wrong text swapped    Fig 4E
    triviaqa           plain             stated opinion, free-form     Fig 5A
    mmlu_multiturn     round 1           round 2 after stated opinion  Fig 5B
    mmlu_description   plain             opinion describes the option  Fig 5C
    mmlu_pushback      round 1           round 2 after content-free    Fig 6
                                         doubt
    mmlu_base          plain (no chat)   bare assertion of the wrong   Fig S5
                                         answer
    mmlu_base_matched  plain, pronoun-   same, on the instruct model   Fig S5
                       free chat

`mmlu_label_swap` and `mmlu_content_swap` state no opinion at all. They change
which label carries the correct content, so the answer moves for a reason that
has nothing to do with sycophancy; that is what makes them the control that
isolates generic answer retrieval.
"""

from dataclasses import dataclass, field as dc_field

import numpy as np

from . import prompts as P
from .data import first_token_id, load_screened, strongest_shift
from .readout import ChoiceReadout, TokenReadout
from .runtime import run_path

__all__ = ["Condition", "CONDITIONS", "get_condition", "PromptPair"]


@dataclass
class PromptPair:
    """Everything an analysis needs for one condition, on one model."""
    source: list
    target: list
    readout: object
    examples: list
    correct_idx: np.ndarray
    wrong_idx: np.ndarray

    def __len__(self):
        return len(self.source)


@dataclass(frozen=True)
class Condition:
    name: str
    screen_file: str          # filename inside runs/<model>/<screen_condition>/
    screen_condition: str     # which run directory holds that screening file
    source_label: str
    target_label: str
    description: str
    builder: object = None

    def build(self, spec, tok, model_key, limit=None) -> PromptPair:
        path = run_path(model_key, self.screen_condition, self.screen_file,
                        create=False)
        examples = load_screened(path, limit=limit)
        return self.builder(spec, tok, examples)


# ── Builders ─────────────────────────────────────────────────────────────────
#
# Each returns a PromptPair. They are small on purpose: the interesting content
# is which prompt formats pair up, and that should be readable at a glance.

def _mcq_readout(spec, correct, wrong):
    return ChoiceReadout(spec.answer_ids, correct, wrong)


def _build_deceptive(spec, tok, examples):
    correct = np.array([e["correct_answer"] for e in examples])
    wrong = np.array([strongest_shift(e, "logits_dec") for e in examples])
    return PromptPair(
        source=[P.plain(tok, e["question"], e["choices"]) for e in examples],
        target=[P.deceptive(tok, e["question"], e["choices"], int(w))
                for e, w in zip(examples, wrong)],
        readout=_mcq_readout(spec, correct, wrong),
        examples=examples, correct_idx=correct, wrong_idx=wrong,
    )


def _build_label_swap(spec, tok, examples):
    correct = np.array([e["correct_answer"] for e in examples])
    wrong = np.array([strongest_shift(e, "logits_label_swap") for e in examples])
    return PromptPair(
        source=[P.plain(tok, e["question"], e["choices"]) for e in examples],
        target=[P.label_swapped(tok, e["question"], e["choices"], int(c), int(w))
                for e, c, w in zip(examples, correct, wrong)],
        readout=_mcq_readout(spec, correct, wrong),
        examples=examples, correct_idx=correct, wrong_idx=wrong,
    )


def _build_content_swap(spec, tok, examples):
    correct = np.array([e["correct_answer"] for e in examples])
    wrong = np.array([strongest_shift(e, "logits_content_swap") for e in examples])
    return PromptPair(
        source=[P.plain(tok, e["question"], e["choices"]) for e in examples],
        target=[P.content_swapped(tok, e["question"], e["choices"], int(c), int(w))
                for e, c, w in zip(examples, correct, wrong)],
        readout=_mcq_readout(spec, correct, wrong),
        examples=examples, correct_idx=correct, wrong_idx=wrong,
    )


def _build_multiturn(spec, tok, examples):
    correct = np.array([e["correct_answer"] for e in examples])
    wrong = np.array([e["best_w"] for e in examples])
    return PromptPair(
        source=[P.multiturn_first(tok, e["question"], e["choices"]) for e in examples],
        target=[P.multiturn_stated(tok, e["question"], e["choices"], int(c), int(w))
                for e, c, w in zip(examples, correct, wrong)],
        readout=_mcq_readout(spec, correct, wrong),
        examples=examples, correct_idx=correct, wrong_idx=wrong,
    )


def _build_pushback(spec, tok, examples):
    # No alternative is proposed, so the "sycophantic" answer is whatever the
    # model revised to in round two.
    correct = np.array([e["correct_answer"] for e in examples])
    wrong = np.array([e["turn2_answer"] for e in examples])
    return PromptPair(
        source=[P.multiturn_first(tok, e["question"], e["choices"]) for e in examples],
        target=[P.multiturn_pushback(tok, e["question"], e["choices"], int(c))
                for e, c in zip(examples, correct)],
        readout=_mcq_readout(spec, correct, wrong),
        examples=examples, correct_idx=correct, wrong_idx=wrong,
    )


def _build_description(spec, tok, examples):
    correct = np.array([e["correct_answer"] for e in examples])
    wrong = np.array([e["best_w"] for e in examples])
    return PromptPair(
        source=[P.plain(tok, e["question"], e["choices"]) for e in examples],
        target=[P.described_opinion(tok, e["question"], e["choices"],
                                    e["descriptions"][str(int(w))])
                for e, w in zip(examples, wrong)],
        readout=_mcq_readout(spec, correct, wrong),
        examples=examples, correct_idx=correct, wrong_idx=wrong,
    )


def _build_triviaqa(spec, tok, examples):
    # Drop examples whose correct and wrong answers begin with the same token:
    # the two-token readout cannot separate them (paper, Appendix A.3 screen 4).
    kept, c_ids, w_ids = [], [], []
    for e in examples:
        c = first_token_id(tok, e["plain_raw_answer"])
        w = first_token_id(tok, e["wrong_guess"])
        if c == w:
            continue
        kept.append(e)
        c_ids.append(c)
        w_ids.append(w)
    dropped = len(examples) - len(kept)
    if dropped:
        print(f"  dropped {dropped} examples sharing a first token", flush=True)
    return PromptPair(
        source=[P.triviaqa_plain(tok, e["question"]) for e in kept],
        target=[P.triviaqa_deceptive(tok, e["question"], e["wrong_guess"])
                for e in kept],
        readout=TokenReadout(c_ids, w_ids),
        examples=kept,
        correct_idx=np.array(c_ids), wrong_idx=np.array(w_ids),
    )


def _build_base(spec, tok, examples):
    correct = np.array([e["correct_answer"] for e in examples])
    wrong = np.array([strongest_shift(e, "logits_dec") for e in examples])
    return PromptPair(
        source=[P.plain_base(tok, e["question"], e["choices"]) for e in examples],
        target=[P.deceptive_base(tok, e["question"], e["choices"], int(w))
                for e, w in zip(examples, wrong)],
        readout=_mcq_readout(spec, correct, wrong),
        examples=examples, correct_idx=correct, wrong_idx=wrong,
    )


def _build_base_matched(spec, tok, examples):
    correct = np.array([e["correct_answer"] for e in examples])
    wrong = np.array([strongest_shift(e, "logits_dec") for e in examples])
    return PromptPair(
        source=[P.plain_pronoun_free(tok, e["question"], e["choices"])
                for e in examples],
        target=[P.deceptive_pronoun_free(tok, e["question"], e["choices"], int(w))
                for e, w in zip(examples, wrong)],
        readout=_mcq_readout(spec, correct, wrong),
        examples=examples, correct_idx=correct, wrong_idx=wrong,
    )


CONDITIONS = {
    c.name: c
    for c in [
        Condition("mmlu", "screened_examples.json", "mmlu",
                  "plain", "deceptive",
                  "Single-round MMLU; the opinion names an option by its letter.",
                  _build_deceptive),
        Condition("mmlu_label_swap", "screened_examples.json", "mmlu",
                  "plain", "label-swapped",
                  "No opinion; the correct and a wrong option exchange letters.",
                  _build_label_swap),
        Condition("mmlu_content_swap", "screened_examples.json", "mmlu",
                  "plain", "content-swapped",
                  "No opinion; the correct and a wrong option exchange text.",
                  _build_content_swap),
        Condition("triviaqa", "screened_examples.json", "triviaqa",
                  "plain", "deceptive",
                  "Free-form question answering with a stated wrong answer.",
                  _build_triviaqa),
        Condition("mmlu_multiturn", "screened_multiturn.json", "mmlu_multiturn",
                  "round 1", "round 2",
                  "The opinion arrives in a second turn, after a correct answer.",
                  _build_multiturn),
        Condition("mmlu_description", "screened_description.json", "mmlu_description",
                  "plain", "described opinion",
                  "The opinion describes the target option without naming it.",
                  _build_description),
        Condition("mmlu_pushback", "screened_multiturn.json", "mmlu_pushback",
                  "round 1", "round 2",
                  "Content-free doubt in a second turn; no alternative proposed.",
                  _build_pushback),
        Condition("mmlu_base", "screened_examples.json", "mmlu_base",
                  "plain", "asserted wrong answer",
                  "Base checkpoint, no chat template, opinion as bare assertion.",
                  _build_base),
        Condition("mmlu_base_matched", "screened_examples.json", "mmlu_base_matched",
                  "plain", "asserted wrong answer",
                  "Instruct checkpoint with the base condition's phrasing.",
                  _build_base_matched),
    ]
}


def get_condition(name) -> Condition:
    try:
        return CONDITIONS[name]
    except KeyError:
        raise SystemExit(
            f"Unknown condition {name!r}. Available:\n  "
            + "\n  ".join(f"{k:20s} {v.description}" for k, v in CONDITIONS.items())
        )
