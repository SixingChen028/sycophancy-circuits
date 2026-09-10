#!/usr/bin/env python3
"""Figure 3: opinion registration happens before answer retrieval.

A  Logit lens. Restricted softmax probability of the correct answer, the
   sycophantic answer, and the mean of the remaining two, at each layer, in the
   plain and deceptive runs. Both start near chance and diverge only after the
   critical layer.

B  Residual-stream patching. Normalized logit difference layer by layer, from
   the plain run into the label-swapped run and into the deceptive run. The
   plain-to-label-swap curve rises only after the critical layer, while the
   plain-to-deceptive curve begins rising several layers earlier -- the gap that
   separates opinion registration from answer retrieval.

Panel B is plotted over the half of examples with the largest gap between the
source and target logit differences. That gap is the denominator of the
normalized logit difference, so examples where it is small give a ratio
dominated by noise.

Inputs: logit_lens.npz (mmlu), residual_patching.npz (mmlu, mmlu_label_swap)
"""

from style import (CORRECT, DECEPTIVE, LABEL_SWAP, OTHER, SYCOPHANTIC,
                   apply_style, figure_parser, get_model_spec, np, panel, plt,
                   result, save)


@panel("logit lens")
def panel_a(model_key, spec):
    data = result(model_key, "mmlu", "logit_lens.npz")
    layers = np.arange(spec.n_layers)
    runs = [
        ("Plain", "source"),
        ("Deceptive", "target"),
    ]
    lines = [("Correct", CORRECT, "p_correct"),
             ("Sycophantic", SYCOPHANTIC, "p_syco"),
             ("Mean other", OTHER, "p_other")]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4), sharey=True)
    for ax, (title, suffix) in zip(axes, runs):
        for i, (label, color, key) in enumerate(lines):
            values = data[f"{key}_{suffix}"]
            ax.plot(layers, values.mean(axis=1), color=color, lw=2.2,
                    label=label, zorder=5 - i)
        ax.axhline(0.25, color="gray", lw=1, ls="--", alpha=0.8, zorder=-1)
        ax.axvline(spec.critical_layer, color="gray", lw=1, ls="--", alpha=0.8,
                   zorder=-1)
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_xlim(0, spec.n_layers - 1)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Restricted softmax probability")
    axes[0].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    return fig


@panel("residual stream patching")
def panel_b(model_key, spec, top_fraction):
    series = [
        ("mmlu_label_swap", "Plain-to-label-swap", LABEL_SWAP),
        ("mmlu", "Plain-to-deceptive", DECEPTIVE),
    ]
    layers = np.arange(spec.n_layers)
    fig, ax = plt.subplots(figsize=(5, 4))
    for condition, label, color in series:
        data = result(model_key, condition, "residual_patching.npz")
        per_example = data["per_example"]
        gap = data["source_ld"] - data["target_ld"]
        keep = np.argsort(gap)[-int(len(gap) * top_fraction):]
        selected = per_example[:, keep]
        ax.fill_between(layers,
                        np.nanpercentile(selected, 25, axis=1),
                        np.nanpercentile(selected, 75, axis=1),
                        alpha=0.10, color=color, linewidth=0)
        ax.plot(layers, np.nanmean(selected, axis=1), color=color, lw=2.2,
                label=label)

    for y in (0, 1):
        ax.axhline(y, color="gray", lw=1, ls="--", alpha=0.8, zorder=-1)
    ax.axvline(spec.critical_layer, color="gray", lw=1, ls="--", alpha=0.8,
               zorder=-1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized logit difference")
    ax.set_title("Residual stream patching")
    ax.set_xlim(0, spec.n_layers - 1)
    ax.set_ylim(-0.2, 1.1)
    ax.legend(frameon=False, fontsize=10, loc="center left",
              bbox_to_anchor=(0.0, 0.8))
    fig.tight_layout()
    return fig


def main():
    parser = figure_parser(__doc__.splitlines()[0])
    parser.add_argument("--top_fraction", type=float, default=0.5,
                        help="Fraction of examples, ranked by source-target gap, "
                             "used in panel B")
    args = parser.parse_args()

    apply_style()
    spec = get_model_spec(args.model)
    for fig, name in [
        (panel_a(args.model, spec), "fig3a_logit_lens"),
        (panel_b(args.model, spec, args.top_fraction), "fig3b_residual_patching"),
    ]:
        if fig is not None:
            save(fig, args.model, name)


if __name__ == "__main__":
    main()
