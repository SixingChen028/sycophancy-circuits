#!/usr/bin/env python3
"""Figure S4: the opinion heads are not induction heads.

A  Prefix-matching (induction) score for every head, by layer and head.
B  Distribution of induction scores over all heads, with the opinion heads
   marked.
C  Copying score, measured by direct logit attribution.
D  Distribution of copying scores, with the opinion heads marked.
E  The two criteria plotted jointly, with the 95th percentile of each marked.
   A head counts as an induction head only if it exceeds both thresholds; the
   opinion heads sit at the null on the copying axis and none exceeds both.

Inputs: induction_heads.npz (runs/<model>/induction/)
"""

from style import (BACKGROUND, HIGHLIGHT, apply_style, figure_parser,
                   get_model_spec, np, plt, result, save)

INDUCTION_COLOR = "#e67e22"


def heatmap(ax, values, spec, opinion, title, cmap, center_zero=False):
    limit = np.abs(values).max()
    kwargs = dict(cmap=cmap, vmin=-limit, vmax=limit) if center_zero \
        else dict(cmap=cmap, vmin=0, vmax=values.max())
    mesh = ax.pcolormesh(values, **kwargs)
    for layer, head in opinion:
        ax.add_patch(plt.Rectangle((head, layer), 1, 1, fill=False,
                                   edgecolor="#2ecc71", linewidth=1.0))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(0, spec.n_heads, 4))
    ax.set_yticks(range(0, spec.n_layers, 5))
    return mesh


def distribution(ax, values, opinion, threshold, title, xlabel):
    flat = values.ravel()
    ax.hist(flat, bins=60, color="#95a5a6", log=True)
    ax.axvline(threshold, color="black", ls=":", lw=1.2,
               label="95th percentile")
    for layer, head in opinion:
        ax.axvline(values[layer, head], color=HIGHLIGHT, lw=1.0, alpha=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count (log scale)")
    ax.legend(frameon=False, fontsize=8)


def main():
    args = figure_parser(__doc__.splitlines()[0]).parse_args()
    apply_style()
    spec = get_model_spec(args.model)
    data = result(args.model, "induction", "induction_heads.npz")

    prefix = data["prefix_matching_score"]
    copying = data["copying_score"]
    opinion = [tuple(h) for h in data["opinion_heads"]]
    induction = [tuple(h) for h in data["induction_heads"]]
    prefix_threshold = float(data["prefix_threshold"])
    copy_threshold = float(data["copy_threshold"])

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    mesh = heatmap(axes[0, 0], prefix, spec, opinion, "Induction score", "Reds")
    fig.colorbar(mesh, ax=axes[0, 0])
    distribution(axes[0, 1], prefix, opinion, prefix_threshold,
                 "Distribution of induction score", "Induction score")
    mesh = heatmap(axes[1, 0], copying, spec, opinion, "Copying score", "RdBu_r",
                   center_zero=True)
    fig.colorbar(mesh, ax=axes[1, 0])
    distribution(axes[1, 1], copying, opinion, copy_threshold,
                 "Distribution of copying score", "Copying score")
    fig.tight_layout()
    save(fig, args.model, "figS4_induction_scores")

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(prefix.ravel(), copying.ravel(), s=14, color=BACKGROUND,
               alpha=0.7, edgecolors="none", label="All other heads")
    if induction:
        ax.scatter([prefix[l, h] for l, h in induction],
                   [copying[l, h] for l, h in induction],
                   s=55, facecolors="none", edgecolors=INDUCTION_COLOR,
                   linewidths=1.4, label=f"Induction heads ({len(induction)})")
    ax.scatter([prefix[l, h] for l, h in opinion],
               [copying[l, h] for l, h in opinion],
               s=55, color=HIGHLIGHT, edgecolors="black", linewidths=0.6,
               label=f"Opinion heads ({len(opinion)})", zorder=4)
    ax.axvline(prefix_threshold, color="gray", ls=":", lw=1)
    ax.axhline(copy_threshold, color="gray", ls=":", lw=1)
    ax.set_xlabel("Induction score (null = 0)")
    ax.set_ylabel("Copying score (null = 0)")
    ax.set_title("Induction-head criteria", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, args.model, "figS4_induction_criteria")

    overlap = [h for h in opinion if h in induction]
    print(f"{len(overlap)} of {len(opinion)} opinion heads meet both criteria.")


if __name__ == "__main__":
    main()
