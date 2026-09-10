#!/usr/bin/env python3
"""Figure 6B: what do the pushback-only heads encode?

Content-free pushback names no alternative answer, so the heads it recruits are
unlikely to carry a sycophantic answer. The alternative is that they suppress the
original correct answer instead. If so, they should encode the correct answer
preferentially in the second round, once the pushback has occurred -- which is
what the probes show.

The heads plotted are those in the top ten under pushback but NOT in the top ten
under the affirmation setting, so the panel is about the mechanism pushback adds
rather than the one it shares.

Inputs: answer_probes.npz (mmlu_pushback), head_patching.npz (mmlu_pushback, mmlu)
"""

from style import (PROBE_COLORS, apply_style, figure_parser, get_model_spec,
                   np, plt, result, save)


def main():
    args = figure_parser(__doc__.splitlines()[0]).parse_args()
    apply_style()
    spec = get_model_spec(args.model)

    pushback = result(args.model, "mmlu_pushback",
                      "head_patching.npz")["norm_logit_diff"]
    affirmation = result(args.model, "mmlu", "head_patching.npz")["norm_logit_diff"]
    split = spec.critical_layer + 1

    # Heads recruited by pushback but not by affirmation.
    top_pushback = set(np.argsort(pushback[:split].ravel())[-10:].tolist())
    top_affirmation = set(np.argsort(affirmation[:split].ravel())[-10:].tolist())
    pushback_only = sorted(top_pushback - top_affirmation,
                           key=lambda i: -pushback[:split].ravel()[i])
    heads = [(i // spec.n_heads, i % spec.n_heads) for i in pushback_only]
    print(f"Pushback-only heads: " + ", ".join(f"L{l}H{h}" for l, h in heads))

    probes = result(args.model, "mmlu_pushback", "answer_probes.npz")
    probe_heads = [tuple(h) for h in probes["opinion_heads"]]
    rows = [probe_heads.index(h) for h in heads if h in probe_heads]
    if not rows:
        raise SystemExit(
            "None of the pushback-only heads appear in answer_probes.npz. "
            "Re-run experiments/collect_writeins.py and probe_answers.py for "
            "--condition mmlu_pushback."
        )

    accuracy = probes["opinion_accuracy"][rows] * 100
    baseline = probes["opinion_baseline"][rows] * 100
    labels = ["Round1 correct", "Round1 syco", "Round2 correct", "Round2 syco"]

    x = np.arange(len(rows))
    width = 0.2
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for j, (label, color) in enumerate(zip(labels, PROBE_COLORS)):
        offset = (j - 1.5) * width
        ax.bar(x + offset, accuracy[:, j], width, color=color, alpha=0.9,
               label=label)
        for i in x:
            ax.plot([i + offset - width / 2, i + offset + width / 2],
                    [baseline[i, j]] * 2, "k-", lw=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{heads[i][0]}H{heads[i][1]}" for i in range(len(rows))],
                       fontsize=9)
    ax.set_xlabel("Head")
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_title("Answer probing on pushback-only heads")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, args.model, "fig6b_pushback_probes")


if __name__ == "__main__":
    main()
