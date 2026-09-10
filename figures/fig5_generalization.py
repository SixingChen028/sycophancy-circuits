#!/usr/bin/env python3
"""Figures 5 and 6A: do the same early heads carry the opinion in other formats?

Each panel plots every early head's normalized logit difference in one prompt
format against its value in the original single-round, label-based MMLU setting.
Points in the top ten by effect size in both formats are highlighted, and points
in the top ten of only one are marked separately.

  Figure 5A  free-form TriviaQA
  Figure 5B  multi-round, opinion stated in a second turn
  Figure 5C  description-based opinion
  Figure 6A  content-free pushback

The first three correlate strongly with the reference format: the same heads
carry the opinion however it is phrased, and whether or not it shares a token
with the answer. Pushback correlates far less, which is the evidence that
content-free doubt recruits a different mechanism.

Inputs: head_patching.npz for mmlu and for each condition compared.
"""

from scipy import stats

from style import (BACKGROUND, DECEPTIVE, HIGHLIGHT, apply_style,  # noqa: F401
                   figure_parser, get_model_spec, np, plt, result, save)

PANELS = [
    ("triviaqa", "Early heads: TriviaQA vs. MMLU", "TriviaQA norm. logit diff.",
     "fig5a_triviaqa"),
    ("mmlu_multiturn", "Early heads: multi- vs. single-round",
     "Multi-round MMLU norm. logit diff.", "fig5b_multiturn"),
    ("mmlu_description", "Early heads: description vs. label",
     "Description-opinion MMLU norm. logit diff.", "fig5c_description"),
    ("mmlu_pushback", "Early heads: pushback vs. affirmation",
     "Pushback MMLU norm. logit diff.", "fig6a_pushback"),
]

ONLY_A = "#c0392b"   # top-10 in the reference format only
ONLY_B = "#27ae60"   # top-10 in the compared format only


def panel(model_key, spec, condition, title, ylabel, reference):
    other = result(model_key, condition, "head_patching.npz")["norm_logit_diff"]
    split = spec.critical_layer + 1
    x = reference[:split].ravel()
    y = other[:split].ravel()
    r, _ = stats.pearsonr(x, y)

    top_x = set(np.argsort(x)[-10:].tolist())
    top_y = set(np.argsort(y)[-10:].tolist())
    both = sorted(top_x & top_y)
    only_x = sorted(top_x - top_y)
    only_y = sorted(top_y - top_x)
    rest = [i for i in range(len(x)) if i not in top_x | top_y]

    fig, ax = plt.subplots(figsize=(4, 4))
    lo = min(x.min(), y.min()) - 0.02
    hi = max(x.max(), y.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.8, alpha=0.6)

    groups = [
        (rest, BACKGROUND, None, 18),
        (only_x, ONLY_A, f"MMLU only ({len(only_x)})", 42),
        (only_y, ONLY_B, f"This format only ({len(only_y)})", 42),
        (both, HIGHLIGHT, f"Top-10 in both ({len(both)})", 42),
    ]
    for indices, color, label, size in groups:
        if not indices:
            continue
        ax.scatter(x[indices], y[indices], s=size, color=color, alpha=0.9,
                   edgecolors="none" if label is None else "black",
                   linewidths=0.5, label=label, zorder=2 if label is None else 3)

    ax.text(0.04, 0.95, f"$r = {r:.2f}$", transform=ax.transAxes, va="top",
            fontsize=11)
    ax.set_xlabel("MMLU norm. logit diff.")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    print(f"  {condition}: r = {r:.2f}, {len(both)}/10 heads shared")
    return fig


def main():
    args = figure_parser(__doc__.splitlines()[0]).parse_args()
    apply_style()
    spec = get_model_spec(args.model)
    reference = result(args.model, "mmlu", "head_patching.npz")["norm_logit_diff"]

    for condition, title, ylabel, name in PANELS:
        try:
            fig = panel(args.model, spec, condition, title, ylabel, reference)
        except SystemExit as exc:
            print(f"  skipping {condition}: {exc}")
            continue
        save(fig, args.model, name)


if __name__ == "__main__":
    main()
