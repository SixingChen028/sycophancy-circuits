#!/usr/bin/env python3
"""Figure 4: two distinct populations of heads carry opinion and retrieve answers.

A  Per-head normalized logit difference, plain into label-swapped, split at the
   critical layer. The ten strongest heads in each band are outlined.
B  As A, plain into deceptive.
C  The two patching conditions plotted against each other, one point per head,
   split into early and late bands. Late heads fall near the identity line --
   their effect in one condition predicts the other -- while early heads have an
   effect only in the deceptive condition, which is what makes them specific to
   the stated opinion.
D  Probe accuracy on the opinion heads.
E  Probe accuracy on the answer retrieval heads.
F  Sycophancy, deceptive accuracy and plain accuracy as opinion heads are
   ablated cumulatively.

Inputs: head_patching.npz (mmlu, mmlu_label_swap), answer_probes.npz (mmlu),
        head_ablation.npz (mmlu)
"""

from matplotlib.patches import Rectangle
from scipy import stats

from style import (CORRECT, EARLY, LATE, PROBE_COLORS, SYCOPHANTIC,  # noqa: F401
                   apply_style, figure_parser, get_model_spec, np, panel, plt,
                   result, save)


@panel("heatmap")
def heatmap(model_key, spec, condition, title, name):
    scores = result(model_key, condition, "head_patching.npz")["norm_logit_diff"]
    split = spec.critical_layer + 1
    bands = [
        (list(range(split, spec.n_layers)), True),   # late, on top
        (list(range(0, split)), False),              # early, below
    ]
    vmax = float(scores.max())

    fig, axes = plt.subplots(
        2, 1, figsize=(3.4, 4),
        gridspec_kw={"height_ratios": [spec.n_layers - split, split]},
    )
    fig.subplots_adjust(left=0.07, right=0.88, hspace=0.12)

    for ax, (rows, is_top) in zip(axes, bands):
        block = scores[rows, :][::-1, :]
        flipped = rows[::-1]
        mesh = ax.pcolormesh(block, cmap="Reds", vmin=0, vmax=vmax)
        ax.invert_yaxis()
        # Outline the ten strongest heads within this band.
        ranked = sorted(((scores[l, h], l, h) for l in rows
                         for h in range(spec.n_heads)), reverse=True)
        for _, layer, head in ranked[:10]:
            ax.add_patch(Rectangle((head, flipped.index(layer)), 1, 1,
                                   fill=False, edgecolor="black", linewidth=0.8))
        ax.set_yticks([i + 0.5 for i in range(0, len(rows), 2)])
        ax.set_yticklabels(flipped[::2], fontsize=8)
        ax.set_xticks([h + 0.5 for h in range(0, spec.n_heads, 2)])
        ax.set_xticklabels(range(0, spec.n_heads, 2), fontsize=8)
        if is_top:
            ax.set_title(title)
            ax.set_xticklabels([])

    axes[1].set_xlabel("Head")
    cbar = fig.add_axes([1.0, 0.2, 0.02, 0.6])
    fig.colorbar(mesh, cax=cbar, label="Normalized logit difference")
    fig.text(-0.01, 0.5, "Layer", va="center", rotation="vertical")
    # No tight_layout here: the colorbar is placed with explicit coordinates,
    # which tight_layout cannot account for.
    save(fig, model_key, name)


@panel("condition scatter")
def scatter_conditions(model_key, spec):
    deceptive = result(model_key, "mmlu", "head_patching.npz")["norm_logit_diff"]
    label_swap = result(model_key, "mmlu_label_swap",
                        "head_patching.npz")["norm_logit_diff"]
    split = spec.critical_layer + 1

    fig, ax = plt.subplots(figsize=(4, 4))
    bands = [
        ("Early heads", slice(0, split), EARLY),
        ("Late heads", slice(split, spec.n_layers), LATE),
    ]
    lo = min(deceptive.min(), label_swap.min()) - 0.02
    hi = max(deceptive.max(), label_swap.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.8, alpha=0.6)

    for i, (label, rows, color) in enumerate(bands):
        x = deceptive[rows].ravel()
        y = label_swap[rows].ravel()
        r, _ = stats.pearsonr(x, y)
        ax.scatter(x, y, s=40, color=color, alpha=0.9, edgecolors="black",
                   linewidths=0.6, zorder=2 + i, label=f"{label}, $r={r:.2f}$")

    ax.set_xlabel("Plain-to-deceptive norm. logit diff.")
    ax.set_ylabel("Plain-to-label-swap norm. logit diff.")
    ax.set_title("Head normalized logit difference")
    ax.legend(fontsize=10, frameon=False, loc="upper left",
              bbox_to_anchor=(-0.02, 0.98))
    fig.tight_layout()
    save(fig, model_key, "fig4c_condition_scatter")


@panel("probe bars")
def probe_bars(model_key, population, title, name, n_heads=6):
    data = result(model_key, "mmlu", "answer_probes.npz")
    key = f"{population}_heads"
    if key not in data:
        print(f"No {population} heads in answer_probes.npz; skipping {name}.")
        return
    heads = data[key][:n_heads]
    accuracy = data[f"{population}_accuracy"][:n_heads] * 100
    baseline = data[f"{population}_baseline"][:n_heads] * 100
    labels = ["Plain\ncorrect", "Plain\nsyco", "Deceptive\ncorrect",
              "Deceptive\nsyco"]

    x = np.arange(len(heads))
    width = 0.2
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for j, (label, color) in enumerate(zip(labels, PROBE_COLORS)):
        offset = (j - 1.5) * width
        ax.bar(x + offset, accuracy[:, j], width, color=color, alpha=0.9,
               label=label)
        # Permutation baseline for this bar.
        for i in x:
            ax.plot([i + offset - width / 2, i + offset + width / 2],
                    [baseline[i, j]] * 2, "k-", lw=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}H{h}" for l, h in heads], fontsize=9)
    ax.set_xlabel("Head")
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, model_key, name)


@panel("ablation curve")
def ablation_curve(model_key):
    data = result(model_key, "mmlu", "head_ablation.npz")
    order = data["ablation_order"]
    x = np.arange(1, len(order) + 1)
    series = [
        ("Sycophancy rate", "cumulative_sycophancy_rate", SYCOPHANTIC),
        ("Deceptive accuracy", "cumulative_deceptive_accuracy", "#27ae60"),
        ("Plain accuracy", "cumulative_plain_accuracy", CORRECT),
    ]

    fig, ax = plt.subplots(figsize=(4.6, 4))
    for label, key, color in series:
        ax.plot(x, data[key] * 100, "o-", color=color, lw=2, ms=5, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"+L{l}H{h}" for l, h in order], rotation=60,
                       ha="right", fontsize=8)
    ax.set_xlabel("Heads ablated (cumulative)")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Zero ablating opinion heads")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, model_key, "fig4f_ablation")


def main():
    args = figure_parser(__doc__.splitlines()[0]).parse_args()
    apply_style()
    spec = get_model_spec(args.model)

    heatmap(args.model, spec, "mmlu_label_swap", "Plain-to-label-swap patching",
            "fig4a_label_swap_heatmap")
    heatmap(args.model, spec, "mmlu", "Plain-to-deceptive patching",
            "fig4b_deceptive_heatmap")
    scatter_conditions(args.model, spec)
    probe_bars(args.model, "opinion", "Answer probing on opinion heads",
               "fig4d_probe_opinion")
    probe_bars(args.model, "retrieval", "Answer probing on retrieval heads",
               "fig4e_probe_retrieval")
    ablation_curve(args.model)


if __name__ == "__main__":
    main()
