#!/usr/bin/env python3
"""Answer probing on head write-ins (paper Appendix B.2; Figures 4D, 4E, 6B).

Fits a four-way multinomial logistic regression from each head's write-in
(reduced to its top PCA components) to an answer label, separately for four
combinations of run and label:

    source run -> correct answer      target run -> correct answer
    source run -> sycophantic answer  target run -> sycophantic answer

What the pattern means (paper, Section 3.2):

  Opinion heads strongly encode the sycophantic answer in the target run, but
  not the correct answer there, and neither answer in the source run. They carry
  the stated opinion specifically.

  Retrieval heads strongly encode the correct answer in the source run and the
  sycophantic answer in the target run. They encode whichever answer is
  ultimately output, and get redirected when an opinion is present.

Permutation baseline
--------------------
The four answer letters are not perfectly balanced across questions, so a probe
can beat chance without carrying any answer information. Accuracy is therefore
compared against a baseline obtained by shuffling the target labels and refitting
the entire cross-validation procedure, repeated --n_permutations times.

CPU only. Requires writeins.npz.
Output: runs/<model>/<condition>/answer_probes.npz
"""

import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from syco.runtime import load_npz, run_path, save_npz


def make_probe():
    """L2 logistic regression, lbfgs, at most 1000 iterations (paper B.2)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )


def probe_accuracy(features, labels, n_folds, seed=0):
    """Mean accuracy over stratified k-fold cross-validation."""
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return float(cross_val_score(make_probe(), features, labels, cv=folds).mean())


def permutation_baseline(features, labels, n_folds, n_permutations, rng):
    """Refit the whole procedure on shuffled labels and average."""
    scores = []
    for i in range(n_permutations):
        shuffled = rng.permutation(labels)
        scores.append(probe_accuracy(features, shuffled, n_folds, seed=i))
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="llama")
    parser.add_argument("--condition", default="mmlu")
    parser.add_argument("--n_components", type=int, default=10)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_permutations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = load_npz(run_path(args.model, args.condition, "writeins.npz",
                             create=False))
    correct = data["correct_idx"]
    wrong = data["wrong_idx"]
    rng = np.random.default_rng(args.seed)
    k = args.n_components

    results = {}
    for population in ("opinion", "retrieval"):
        key = f"{population}_heads"
        if key not in data:
            continue
        heads = data[key]
        projections = {
            "source": data[f"{population}_source_pca"],
            "target": data[f"{population}_target_pca"],
        }
        accuracy = np.zeros((len(heads), 4), dtype=np.float32)
        baseline = np.zeros((len(heads), 4), dtype=np.float32)

        print(f"\n== {population} heads ==", flush=True)
        for i, (layer, head) in enumerate(heads):
            row = []
            for j, (run, target_name) in enumerate(
                [(r, t) for r in ("source", "target") for t in ("correct", "syco")]
            ):
                features = projections[run][i][:, :k]
                labels = correct if target_name == "correct" else wrong
                accuracy[i, j] = probe_accuracy(features, labels, args.n_folds,
                                                args.seed)
                baseline[i, j] = permutation_baseline(
                    features, labels, args.n_folds, args.n_permutations, rng
                )
                row.append(f"{run}/{target_name} {accuracy[i, j]:.1%}"
                           f" (base {baseline[i, j]:.1%})")
            print(f"  L{layer}H{head}: " + "  ".join(row), flush=True)

        results[f"{population}_heads"] = heads
        results[f"{population}_accuracy"] = accuracy
        results[f"{population}_baseline"] = baseline

    if not results:
        raise SystemExit("No head populations found in writeins.npz")

    save_npz(run_path(args.model, args.condition, "answer_probes.npz"),
             probe_columns=np.array(["source_correct", "source_syco",
                                     "target_correct", "target_syco"]),
             **results)


if __name__ == "__main__":
    main()
