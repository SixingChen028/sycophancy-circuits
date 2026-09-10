# Paper figure map

Every figure in the paper, the command that produces it, and what has to exist
first. `M` stands for a model key (`llama`, `mistral`, `gemma`, `llama-base`).
Main-paper figures report Llama-3.1-8B-Instruct.

## Main figures

### Figure 3 — Opinion registration happens before answer retrieval

```bash
python experiments/logit_lens.py            --model M --condition mmlu
python experiments/patch_residual_stream.py --model M --condition mmlu
python experiments/patch_residual_stream.py --model M --condition mmlu_label_swap
python figures/fig3_depth.py                --model M
```

Panel A is the logit lens; panel B is residual-stream patching in the two
conditions. The separation between the two curves in B is the paper's depth
claim.

### Figure 4 — Two populations of heads

```bash
python experiments/patch_heads.py           --model M --condition mmlu
python experiments/patch_heads.py           --model M --condition mmlu_label_swap
python experiments/patch_heads.py           --model M --condition mmlu_content_swap
python experiments/collect_writeins.py      --model M --condition mmlu
python experiments/probe_answers.py         --model M --condition mmlu   # CPU
python experiments/ablate_heads.py          --model M --condition mmlu
python figures/fig4_heads.py                --model M
```

A and B are the two patching heatmaps, C the scatter between them, D and E the
answer probes, F the cumulative ablation. The content-swap sweep is needed only
for panel E: retrieval heads are scored by the mean effect across both
relabelling conditions, so a head counts as a retrieval head only if it tracks
the correct answer under both.

### Figure 5 — Generalization across prompt formats

```bash
# 5A, free-form TriviaQA: see the TriviaQA pipeline in docs/replication.md
python experiments/patch_heads.py --model M --condition triviaqa

# 5B, multi-round
python experiments/screen_multiturn.py --model M --mode stated
python experiments/patch_heads.py      --model M --condition mmlu_multiturn

# 5C, description-based opinion
python scripts/generate_descriptions.py                       # once, model-agnostic
python experiments/screen_description.py --model M
python experiments/patch_heads.py        --model M --condition mmlu_description

python figures/fig5_generalization.py --model M
```

The same script also draws Figure 6A, since it is the same comparison applied to
the pushback condition.

### Figure 6 — Pushback recruits a distinct set of heads

```bash
python experiments/screen_multiturn.py  --model M --mode pushback
python experiments/patch_heads.py       --model M --condition mmlu_pushback
python experiments/collect_writeins.py  --model M --condition mmlu_pushback
python experiments/probe_answers.py     --model M --condition mmlu_pushback
python figures/fig5_generalization.py   --model M   # panel A
python figures/fig6_pushback_probes.py  --model M   # panel B
```

## Supplementary figures

| Figure | Command | Notes |
|---|---|---|
| S1 — usable examples and sycophancy rates | `python figures/figS1_screening_summary.py --models llama mistral gemma` | Reads the screening files directly; no extra experiment |
| S2 — cumulative patching | `python experiments/patch_heads_cumulative.py --model M --condition mmlu` | Use `--k 20` for Mistral, whose effect spreads over more heads |
| S3 — each opinion head ablated alone | Included in `head_ablation.npz` from `ablate_heads.py` | Plotted by `fig4_heads.py` |
| S4 — opinion heads are not induction heads | `python experiments/induction_heads.py --model M` then `python figures/figS4_induction.py --model M` | Needs no dataset; runs in minutes |
| S5 — cross-question patching (reference vs content) | See below | |
| S6 — base vs instruction-tuned | See below | |
| S7–S12 — Mistral | The same commands with `--model mistral` | S12 is that model's cross-question panel |
| S13–S18 — Gemma | The same commands with `--model gemma` | S18 is that model's cross-question panel; Gemma has no pushback panel (only 12 usable examples) |

### Figure S5 — cross-question patching, reference versus content

Pairs each TriviaQA question with a different one and patches a head between
them, measuring the change in log-probability of the *source* question's stated
answer. Answer retrieval heads move it substantially; opinion heads about one
fifth as much, which is what a context-dependent reference should do.

```bash
python experiments/patch_cross_question.py --model M --source_arm dec --seed 0
python figures/figS5_cross_question.py --model M
```

The paper's panel needs only that one arm. An optional control arm
(`--source_arm plain`, same seed) isolates what the source's opinion contributed
from generic cross-context disruption; the figure script adds a second,
non-paper panel when it is present. `--head_source triviaqa` re-derives the head
populations from the TriviaQA sweep instead of the paper's MMLU definitions.

Read the size caveat the script prints before quoting any number from it: these
are directions in the residual stream, not the model approaching that answer.

The same commands give Figure S12 (Mistral) and Figure S18 (Gemma).

### Figure S6 — before and after instruction tuning

The base checkpoint has no chat template and no assistant persona, so the
opinion is stated as a bare assertion of fact rather than as a belief. The
instruct model is re-run with that same phrasing so the comparison holds
everything but the weights fixed.

```bash
python experiments/screen_mmlu.py  --model llama-base --format base
python experiments/screen_mmlu.py  --model llama      --format pronoun_free
python experiments/patch_heads.py  --model llama-base --condition mmlu_base
python experiments/patch_heads.py  --model llama      --condition mmlu_base_matched
python experiments/patch_heads.py  --model llama-base --condition mmlu_label_swap
python figures/figS6_base_vs_instruct.py --model llama-base --instruct_model llama
```

## Tables

| Table | Source |
|---|---|
| 1 — MMLU subjects and question counts | Printed by `scripts/build_mmlu_dataset.py` |
| Screening counts (Appendix A.1, C) | Printed by each `screen_*.py`; summarized by `figures/figS1_screening_summary.py` |

## Prompt boxes

The four prompt boxes in the appendix are the system and user prompts of the two
LLM-assisted stages:

| Box | Location |
|---|---|
| 1, 2 — description generation | `SYSTEM` and `user_prompt()` in `scripts/generate_descriptions.py` |
| 3, 4 — wrong-answer validation | `SYSTEM` and `judge()` in `scripts/judge_triviaqa_answers.py` |
