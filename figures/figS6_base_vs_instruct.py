#!/usr/bin/env python3
"""Figure S6: opinion and answer retrieval heads before and after instruction tuning.

Plots each head's normalized logit difference in the instruction-tuned model
against the base checkpoint, separately for early heads (from plain-to-deceptive
patching) and late heads (from plain-to-label-swap patching). The two checkpoints
share an architecture and a tokenizer, so layer and head indices are directly
comparable.

Both populations are already present in the base model, so the mechanism is not
installed from scratch by post-training. They differ in how much post-training
changes them: late heads lie along the identity line, while early heads are
amplified, so post-training leaves answer retrieval largely untouched while
strengthening the components that carry a stated opinion.

Requires head_patching.npz for mmlu_base (base checkpoint) and mmlu_base_matched
(instruct checkpoint, same pronoun-free phrasing), plus mmlu_label_swap under
each, so the comparison holds the prompt phrasing fixed.

Run with --model llama-base for the base results and --instruct_model llama for
the instruction-tuned ones.
"""

from scipy import stats

from style import (BACKGROUND, EARLY, HIGHLIGHT, LATE, apply_style,
                   figure_parser, get_model_spec, np, plt, result, save)


def panel(base, instruct, spec, band, title, color):
    split = spec.critical_layer + 1
    rows = slice(0, split) if band == "early" else slice(split, spec.n_layers)
    x = base[rows].ravel()
    y = instruct[rows].ravel()
    r, _ = stats.pearsonr(x, y)
    slope = np.polyfit(x, y, 1)[0]

    top_x = set(np.argsort(x)[-10:].tolist())
    top_y = set(np.argsort(y)[-10:].tolist())
    both = sorted(top_x & top_y)
    rest = [i for i in range(len(x)) if i not in top_x | top_y]

    fig, ax = plt.subplots(figsize=(4, 4))
    lo = min(x.min(), y.min()) - 0.02
    hi = max(x.max(), y.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.8, alpha=0.6)
    ax.scatter(x[rest], y[rest], s=18, color=BACKGROUND, alpha=0.8,
               edgecolors="none")
    ax.scatter(x[sorted(top_x - top_y)], y[sorted(top_x - top_y)], s=42,
               color=color, edgecolors="black", linewidths=0.5,
               label="Base only")
    ax.scatter(x[sorted(top_y - top_x)], y[sorted(top_y - top_x)], s=42,
               color="#3daebb", edgecolors="black", linewidths=0.5,
               label="Instruct only")
    ax.scatter(x[both], y[both], s=42, color=HIGHLIGHT, edgecolors="black",
               linewidths=0.5, label=f"Top-10 in both ({len(both)})")

    ax.text(0.04, 0.95, f"$r = {r:.2f}$", transform=ax.transAxes, va="top",
            fontsize=11)
    ax.set_xlabel("Base model norm. logit diff.")
    ax.set_ylabel("Instruct model norm. logit diff.")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    print(f"  {band}: r = {r:.2f}, slope = {slope:.2f}, "
          f"{len(both)}/10 heads shared")
    return fig


def main():
    parser = figure_parser(__doc__.splitlines()[0])
    parser.add_argument("--instruct_model", default="llama",
                        help="Model key holding the instruction-tuned results")
    parser.set_defaults(model="llama-base")
    args = parser.parse_args()
    apply_style()
    spec = get_model_spec(args.model)

    pairs = [
        ("early", "mmlu_base", "mmlu_base_matched",
         "Early heads: instruct vs. base", EARLY),
        ("late", "mmlu_label_swap", "mmlu_label_swap",
         "Late heads: instruct vs. base", LATE),
    ]
    for band, base_condition, instruct_condition, title, color in pairs:
        try:
            base = result(args.model, base_condition,
                          "head_patching.npz")["norm_logit_diff"]
            instruct = result(args.instruct_model, instruct_condition,
                              "head_patching.npz")["norm_logit_diff"]
        except SystemExit as exc:
            print(f"Skipping {band} heads: {exc}")
            continue
        save(panel(base, instruct, spec, band, title, color), args.model,
             f"figS6_{band}_heads")


if __name__ == "__main__":
    main()
