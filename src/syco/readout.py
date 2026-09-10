"""How an answer is read out of the final-token logits.

Two settings appear in the paper and they differ only here:

  ChoiceReadout  Multiple choice. The answer is one of four fixed letter tokens,
                 identical for every example, so the readout is a 4-way
                 restricted slice of the vocabulary and argmax over it gives the
                 model's answer (paper, Section 2.1).

  TokenReadout   Free-form TriviaQA. There is no fixed option set, so each
                 example carries its own pair of first tokens -- the first token
                 of the correct answer and of the stated wrong answer -- and the
                 comparison is between those two (paper, Appendix A.3). Examples
                 whose two answers share a first token carry no signal and are
                 dropped upstream, in `syco.data`.

Both expose the same interface, which is what lets one patching implementation
serve both. `scores()` returns a (batch, K) array whose columns are addressed by
`correct_col` / `wrong_col`, so the logit difference is always

    scores[i, correct_col[i]] - scores[i, wrong_col[i]]
"""

import numpy as np

__all__ = ["ChoiceReadout", "TokenReadout"]


class ChoiceReadout:
    """Restricted 4-way readout over the bare answer letters A/B/C/D."""

    supports_accuracy = True
    n_columns = 4

    def __init__(self, answer_ids, correct_idx, wrong_idx):
        self.answer_ids = list(answer_ids)
        self.correct_col = np.asarray(correct_idx, dtype=np.int64)
        self.wrong_col = np.asarray(wrong_idx, dtype=np.int64)

    def scores(self, logits_last, start, count):
        """logits_last: (batch, vocab) tensor -> (batch, 4) float32 ndarray."""
        return logits_last[:, self.answer_ids].float().cpu().numpy()


class TokenReadout:
    """Per-example two-token readout for free-form answers.

    Column 0 is the example's correct-answer first token, column 1 its stated
    wrong-answer first token.
    """

    supports_accuracy = False
    n_columns = 2

    def __init__(self, correct_token_ids, wrong_token_ids):
        self.correct_token_ids = np.asarray(correct_token_ids, dtype=np.int64)
        self.wrong_token_ids = np.asarray(wrong_token_ids, dtype=np.int64)
        n = len(self.correct_token_ids)
        self.correct_col = np.zeros(n, dtype=np.int64)
        self.wrong_col = np.ones(n, dtype=np.int64)

    def scores(self, logits_last, start, count):
        import torch

        sl = slice(start, start + count)
        device = logits_last.device
        c = torch.as_tensor(self.correct_token_ids[sl], device=device)
        w = torch.as_tensor(self.wrong_token_ids[sl], device=device)
        rows = torch.arange(count, device=device)
        out = torch.stack(
            [logits_last[rows, c], logits_last[rows, w]], dim=1
        )
        return out.float().cpu().numpy()
