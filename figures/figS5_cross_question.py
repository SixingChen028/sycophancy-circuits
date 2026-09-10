#!/usr/bin/env python3
"""Figure S5: cross-question patching separates reference from content.

Change in log-probability of the first token of the SOURCE question's stated
wrong answer, when a head's output at the final token position is patched from
one question's deceptive run into an unrelated question's deceptive run.

Answer retrieval heads raise the source's answer substantially, consistent with
carrying transferable answer content. Opinion heads produce much smaller effects
-- about one fifth -- which is what a context-dependent reference should do:
transplanted into a prompt that never mentioned that answer, most of its meaning
is gone. See `syco.crossquestion` for the full argument and the size caveat.

The paper's panel needs only the deceptive arm. If the optional control arm was
also run (a source stating no opinion), a second figure contrasts the two and
the report adds the opinion-specific contrast; both are diagnostics, not the
headline.

Produces the same panel for Figure S12 (Mistral) and S18 (Gemma) with
--model mistral / --model gemma.

Inputs: cross_question_dec.npz (triviaqa), optionally cross_question_plain.npz,
        plus the head_patching.npz sweeps that define the populations.
"""

import sys

from style import (apply_style, figure_parser, get_model_spec, np, plt, save)

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/src")

from syco.crossquestion import (READOUTS, SOURCE_OPINION, delta_log_prob,
                                load_arms, per_head_effect, population_summary)
from syco.heads import (complement, format_heads, opinion_population,
                        retrieval_population, top_heads)
from syco.runtime import load_npz, run_path

OPINION_COLOR = "#2a78d6"
RETRIEVAL_COLOR = "#eb6834"
OTHER_COLOR = "#898781"


def populations(model_key, spec, k, head_source):
    """The two head populations, and the other heads in each of their bands."""
    if head_source == "mmlu":
        opinion = opinion_population(model_key, spec, k=k, condition="mmlu")
        retrieval = retrieval_population(model_key, spec, k=k)
    else:
        scores = load_npz(run_path(model_key, "triviaqa", "head_patching.npz",
                                   create=False))["norm_logit_diff"]
        opinion = top_heads(scores, spec, band="early", k=k)
        retrieval = top_heads(scores, spec, band="late", k=k)
    selected = opinion + retrieval
    return {
        "opinion": opinion,
        "opinion_other": complement(selected, spec, "early"),
        "retrieval": retrieval,
        "retrieval_other": complement(selected, spec, "late"),
    }


def mean_and_sem_over_heads(delta, heads):
    """Mean across heads, with the standard error ACROSS heads.

    Head-to-head spread dominates question-pair noise, so the head-level error
    is the honest one for a claim about a population of heads.
    """
    values = per_head_effect(delta, heads)
    if len(values) == 0:
        return float("nan"), float("nan")
    sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return float(values.mean()), float(sem)


def report(dec_delta, plain_delta, groups, spec):
    print(f"\n{dec_delta.shape[3]} pairs, {spec.n_layers}x{spec.n_heads} heads, "
          f"critical layer {spec.critical_layer}")
    print(f"opinion heads   : {format_heads(groups['opinion'])}")
    print(f"retrieval heads : {format_heads(groups['retrieval'])}")

    print("\nPer-head change in log-probability "
          "(deceptive arm; w_s is the paper's measure):")
    print(f"{'head':>9}  " + "  ".join(f"{n:>9}" for n in READOUTS))
    for name in ("opinion", "retrieval"):
        print(f"  -- {name} heads --")
        for layer, head in groups[name]:
            row = "  ".join(
                f"{dec_delta[layer, head, k, :].mean():+9.4f}" for k in range(3)
            )
            print(f"  L{layer:2d}H{head:2d}  {row}")

    print("\nPopulation summary:")
    summaries = {}
    for name in ("opinion", "opinion_other", "retrieval", "retrieval_other"):
        summary = population_summary(dec_delta, groups[name], plain_delta)
        if summary is None:
            continue
        summaries[name] = summary
        line = (f"  {name:16s} n={summary['n_heads']:4d}  "
                f"d_ct={summary['c_t']:+.3f}  d_wt={summary['w_t']:+.3f}  "
                f"d_ws={summary['w_s']:+.3f} (SEM {summary['w_s_sem']:.3f})  "
                f"pairs positive {summary['pairs_positive']:.0%}")
        if "transfer" in summary:
            low, high = summary["transfer_ci"]
            line += f"  control-contrast={summary['transfer']:+.3f} [{low:+.3f}, {high:+.3f}]"
        print(line)

    interpret(summaries, plain_delta is not None)
    return summaries


def interpret(summaries, has_control):
    """State what the numbers do and do not support."""
    if "opinion" not in summaries or "retrieval" not in summaries:
        return
    opinion, retrieval = summaries["opinion"], summaries["retrieval"]

    print("\nReading:")
    if retrieval["w_s"] <= 0:
        print("  Answer retrieval heads do not raise the source's answer, so the\n"
              "  assay detected no content transfer at all and nothing can be\n"
              "  concluded about the opinion heads.")
        return

    ratio = opinion["w_s"] / retrieval["w_s"]
    print(f"  Answer retrieval heads raise the source question's answer by "
          f"{retrieval['w_s']:+.3f}\n"
          f"  log-probability; opinion heads by {opinion['w_s']:+.3f}, "
          f"which is {ratio:.0%} of it.")
    if ratio < 0.5:
        print("  Most of the opinion heads' effect disappears when their output is\n"
              "  moved to an unrelated prompt, which is what a context-dependent\n"
              "  reference should do. This supports the reference account.")
    else:
        print("  The opinion heads' effect largely survives the move to an unrelated\n"
              "  prompt, which is what transferable content would do. This does NOT\n"
              "  support the reference account for this model -- report it as such.")

    if has_control and "transfer" in opinion:
        low, high = opinion["transfer_ci"]
        print(f"\n  Control arm (diagnostic): the opinion-specific contrast is "
              f"{opinion['transfer']:+.3f}\n  [{low:+.3f}, {high:+.3f}]. ", end="")
        if low <= 0 <= high:
            print("It contains zero.")
        elif high < 0:
            # A negative contrast is a real effect, not an absence of one.
            print("It lies entirely BELOW zero, which is not a\n"
                  "  null: a source stating no opinion raises the source's answer MORE\n"
                  "  than one that states it. Inspect the arms separately.")
        else:
            print("It excludes zero from above.")

    print("\n  Size caveat: these are directions written into the residual stream.\n"
          "  The source question's answer remains orders of magnitude away from\n"
          "  being produced -- quote the absolute probability alongside these.")


def panel_per_head(dec_delta, groups, model_key, k, sort_by_value):
    """The paper's panel: one bar per head, deceptive arm only."""
    fig, ax = plt.subplots(figsize=(9, 4))
    position = 0
    ticks, labels = [], []
    n_pairs = dec_delta.shape[3]

    for key, color in (("opinion", OPINION_COLOR), ("retrieval", RETRIEVAL_COLOR)):
        heads = groups[key]
        # Order within the population, then take the top k of it, so no head
        # outside the defined population can appear.
        if sort_by_value:
            heads = sorted(heads,
                           key=lambda lh: -dec_delta[lh[0], lh[1],
                                                     SOURCE_OPINION, :].mean())
        for layer, head in heads[:k]:
            values = dec_delta[layer, head, SOURCE_OPINION, :]
            ax.bar(position, values.mean(), width=0.75, color=color,
                   yerr=values.std(ddof=1) / np.sqrt(n_pairs),
                   error_kw=dict(lw=0.8, ecolor="#0b0b0b"))
            ticks.append(position)
            labels.append(f"L{layer}H{head}")
            position += 1
        position += 1.4

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_xlim(-1, position - 1.8)
    ax.set_ylabel("$\\Delta$ log-probability of\nthe source's opinion")
    ax.bar(np.nan, np.nan, color=OPINION_COLOR, label="Opinion heads")
    ax.bar(np.nan, np.nan, color=RETRIEVAL_COLOR, label="Answer retrieval heads")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout()
    save(fig, model_key, "figS5_cross_question")


def panel_control(dec_delta, plain_delta, groups, model_key):
    """Diagnostic, not a paper panel: the two arms side by side.

    Shown rather than differenced, because the per-head contrast between the
    arms inverts the sign of what happened at heads where they differ mostly in
    how much they restore the target's correct answer.
    """
    order = [("Opinion\nheads", "opinion", OPINION_COLOR),
             ("Other heads\nsame layers", "opinion_other", OTHER_COLOR),
             ("Retrieval\nheads", "retrieval", RETRIEVAL_COLOR),
             ("Other heads\nsame layers", "retrieval_other", OTHER_COLOR)]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    panels = [(dec_delta, "Source states an opinion (the paper's condition)"),
              (plain_delta, "Source states none (control)")]
    for ax, (delta, title) in zip(axes, panels):
        for i, (label, key, color) in enumerate(order):
            mean, sem = mean_and_sem_over_heads(delta, groups[key])
            ax.bar(i, mean, width=0.62, color=color, yerr=sem, capsize=3,
                   error_kw=dict(lw=1, ecolor="#0b0b0b"))
            ax.annotate(f"{mean:+.2f}", (i, mean + np.sign(mean) * sem),
                        ha="center", fontsize=8,
                        va="bottom" if mean >= 0 else "top",
                        xytext=(0, 3 if mean >= 0 else -3),
                        textcoords="offset points")
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([o[0] for o in order], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.margins(y=0.22)
    axes[0].set_ylabel("$\\Delta$ log-probability of the\n"
                       "source question's stated answer")
    fig.tight_layout()
    save(fig, model_key, "figS5_cross_question_control")


def main():
    parser = figure_parser(__doc__.splitlines()[0])
    parser.add_argument("--head_source", default="mmlu",
                        choices=["mmlu", "triviaqa"],
                        help="Which sweeps define the head populations")
    parser.add_argument("--k", type=int, default=10,
                        help="Heads per population")
    parser.add_argument("--plot_k", type=int, default=8,
                        help="Heads shown per population in the figure")
    parser.add_argument("--rank_order", action="store_true",
                        help="Order bars by patching rank instead of by the "
                             "plotted value (the paper's panel is sorted by "
                             "value)")
    args = parser.parse_args()

    apply_style()
    spec = get_model_spec(args.model)
    dec, plain = load_arms(args.model)
    dec_delta = delta_log_prob(dec)
    plain_delta = delta_log_prob(plain) if plain is not None else None
    groups = populations(args.model, spec, args.k, args.head_source)

    report(dec_delta, plain_delta, groups, spec)
    panel_per_head(dec_delta, groups, args.model, args.plot_k,
                   sort_by_value=not args.rank_order)
    if plain_delta is not None:
        panel_control(dec_delta, plain_delta, groups, args.model)


if __name__ == "__main__":
    main()
