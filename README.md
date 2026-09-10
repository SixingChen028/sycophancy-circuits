# Tracing Mechanisms of Sycophantic Agreement in Language Models

This repository contains code for the paper *Tracing Mechanisms of Sycophantic Agreement in Language Models*. We use causal mediation analysis to locate the components that carry a user's stated opinion in Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3 and Gemma-2-9B-it, and to show how that signal biases the model into abandoning an answer it would otherwise get right.

## Repository Structure

```
.
├── src/syco/
│   ├── models.py                     # Model registry: HF id, answer token ids, head geometry, critical layer
│   ├── prompts.py                    # Every prompt format in the paper
│   ├── conditions.py                 # Source/target prompt pairs, one per condition
│   ├── hooks.py                      # Activation caching, patching and ablation hooks
│   ├── metrics.py                    # Normalized logit difference, answer rates
│   ├── readout.py                    # 4-way (MMLU) and per-example 2-token (TriviaQA) read-outs
│   ├── heads.py                      # Early/late band split, opinion and retrieval populations
│   ├── data.py                       # Screened-example loading, sycophantic-answer selection
│   ├── crossquestion.py              # Pairing and analysis for cross-question patching
│   ├── triviaqa.py                   # Free-form answer normalization and matching
│   ├── overlap.py                    # Word-level overlap checks for described opinions
│   ├── llm.py                        # Anthropic API helper for the two dataset stages
│   └── runtime.py                    # Model loading, batching, run-directory layout
│
├── scripts/
│   ├── build_mmlu_dataset.py         # Download the 15 MMLU STEM subjects (Table 1)
│   ├── build_triviaqa_dataset.py     # Download TriviaQA rc.nocontext validation split
│   ├── generate_descriptions.py      # Generate referring descriptions (Boxes 1-2)
│   └── judge_triviaqa_answers.py     # Validate constructed wrong answers (Boxes 3-4)
│
├── experiments/
│   ├── screen_mmlu.py                # Step 1a: four screens on MMLU
│   ├── screen_multiturn.py           # Step 1b: multi-round and pushback screens
│   ├── screen_description.py         # Step 1c: description-based opinion screen
│   ├── screen_triviaqa.py            # Step 1d: four-stage free-form screen
│   ├── logit_lens.py                 # Step 2a: layer-by-layer answer probabilities
│   ├── patch_residual_stream.py      # Step 2b: residual-stream patching by layer
│   ├── patch_heads.py                # Step 3: per-head patching sweep
│   ├── patch_heads_cumulative.py     # Step 4a: cumulative patching of opinion heads
│   ├── ablate_heads.py               # Step 4b: individual and cumulative ablation
│   ├── collect_writeins.py           # Step 4c: head write-in vectors, reduced by PCA
│   ├── probe_answers.py              # Step 4d: answer probes (CPU)
│   ├── patch_cross_question.py       # Step 5: cross-question patching
│   └── induction_heads.py            # Step 6: prefix-matching and copying scores
│
├── figures/
│   ├── style.py                      # Shared palette and figure helpers
│   ├── fig3_depth.py                 # Fig. 3   Logit lens and residual-stream patching
│   ├── fig4_heads.py                 # Fig. 4   Head heatmaps, probes, ablation
│   ├── fig5_generalization.py        # Fig. 5   Cross-format correlations (and Fig. 6A)
│   ├── fig6_pushback_probes.py       # Fig. 6B  Probes on pushback-only heads
│   ├── figS1_screening_summary.py    # Fig. S1  Usable examples and sycophancy rates
│   ├── figS4_induction.py            # Fig. S4  Induction-head criteria
│   ├── figS5_cross_question.py       # Fig. S5  Cross-question patching
│   └── figS6_base_vs_instruct.py     # Fig. S6  Base vs instruction-tuned checkpoint
│
├── slurm/
│   ├── env.sh                        # Conda environment and HuggingFace cache
│   ├── run.sh                        # One runner for every experiment
│   └── submit_model.sh               # Submit the whole pipeline for one model
│
├── data/                             # Datasets, produced by scripts/ (not committed)
└── runs/                             # All experiment output (not committed)
```

## Setup

```bash
pip install -e .            # add '.[llm]' for the two dataset stages that call an API
```

Llama and Gemma are gated on HuggingFace, so authenticate before the first download:

```bash
huggingface-cli login
```

The reported runs used a single 48GB NVIDIA L40S GPU with 8 CPU cores and 48–64GB of system memory, with all models loaded in bfloat16. No individual job exceeded four hours.

## Datasets

```bash
python scripts/build_mmlu_dataset.py        # -> data/mmlu_stem.json (2,476 questions)
python scripts/build_triviaqa_dataset.py    # -> data/triviaqa.json  (9,960 questions)
```

The description-based condition additionally needs generated descriptions. These are model-agnostic and generated once, so set `ANTHROPIC_API_KEY` first:

```bash
python scripts/generate_descriptions.py --limit 50   # pilot, check the attrition rate
python scripts/generate_descriptions.py              # full run, 7,428 descriptions
```

## Reproducing Results

Every experiment takes `--model` and, where relevant, `--condition`. All output is written to `runs/<model>/<condition>/`, so one replication is one directory.

| `--model` | Checkpoint |
|-----------|------------|
| `llama` | meta-llama/Llama-3.1-8B-Instruct |
| `mistral` | mistralai/Mistral-7B-Instruct-v0.3 |
| `gemma` | google/gemma-2-9b-it |
| `llama-base` | meta-llama/Llama-3.1-8B |

| `--condition` | Source run | Target run |
|---------------|-----------|------------|
| `mmlu` | plain | opinion names an option by letter |
| `mmlu_label_swap` | plain | correct and wrong options exchange letters |
| `mmlu_content_swap` | plain | correct and wrong options exchange text |
| `triviaqa` | plain | opinion states a free-form wrong answer |
| `mmlu_multiturn` | round 1 | round 2 after a stated opinion |
| `mmlu_description` | plain | opinion describes the option, sharing no words |
| `mmlu_pushback` | round 1 | round 2 after content-free doubt |
| `mmlu_base` / `mmlu_base_matched` | plain | wrong answer asserted as fact |

The two swap conditions state no opinion. They move the answer for a reason unrelated to sycophancy, which is what makes them the control that isolates answer retrieval.

On a cluster, `bash slurm/submit_model.sh llama` submits everything below with the dependencies between stages already wired up.

### Step 1 — Screening

Selects questions the model answers correctly, resolves from the option text rather than the letter, and is flipped by a stated opinion (Appendix A).

```bash
python experiments/screen_mmlu.py       --model llama
python experiments/screen_multiturn.py  --model llama --mode stated
python experiments/screen_multiturn.py  --model llama --mode pushback
python experiments/screen_description.py --model llama
```

TriviaQA has no fixed option set, so the wrong answer is constructed from the model's own next-best guess and validated by an LLM judge (Appendix A.3):

```bash
python experiments/screen_triviaqa.py --model llama --stage plain
python experiments/screen_triviaqa.py --model llama --stage guess
python scripts/judge_triviaqa_answers.py --model llama    # needs ANTHROPIC_API_KEY
python experiments/screen_triviaqa.py --model llama --stage opinion
python experiments/screen_triviaqa.py --model llama --stage build
```

**Output:** `runs/<model>/<condition>/screened_*.json`

### Step 2 — Locating the Depth

```bash
python experiments/logit_lens.py            --model llama --condition mmlu
python experiments/patch_residual_stream.py --model llama --condition mmlu
python experiments/patch_residual_stream.py --model llama --condition mmlu_label_swap
```

**Output:** `logit_lens.npz`, `residual_patching.npz` → Figure 3

### Step 3 — Per-Head Patching

The main sweep, run once per condition. It is `n_layers × n_heads × ceil(N/batch)` forward passes, 2–4 hours per condition, and checkpoints after every layer — resubmit the identical command to resume.

```bash
for CONDITION in mmlu mmlu_label_swap mmlu_content_swap triviaqa \
                 mmlu_multiturn mmlu_description mmlu_pushback; do
    python experiments/patch_heads.py --model llama --condition $CONDITION
done
```

**Output:** `head_patching.npz` → Figures 4A, 4B, 5, 6A

### Step 4 — Ablation and Probing

```bash
python experiments/patch_heads_cumulative.py --model llama --condition mmlu
python experiments/ablate_heads.py           --model llama --condition mmlu
python experiments/collect_writeins.py       --model llama --condition mmlu
python experiments/probe_answers.py          --model llama --condition mmlu   # CPU
```

`collect_writeins.py` needs the `mmlu`, `mmlu_label_swap` and `mmlu_content_swap` sweeps: answer retrieval heads are scored across both relabelling conditions. Repeat the last two with `--condition mmlu_pushback` for Figure 6B.

**Output:** `head_ablation.npz`, `head_patching_cumulative.npz`, `writeins.npz`, `answer_probes.npz` → Figures 4D–4F, 6B, S2, S3

### Step 5 — Cross-Question Patching

Patches a head from one TriviaQA question's run into an unrelated question's run, to test whether it carries a context-dependent reference or transferable answer content (Appendix D.4).

```bash
python experiments/patch_cross_question.py --model llama --source_arm dec --seed 0
```

An optional control arm (`--source_arm plain`, same seed) isolates what the source's opinion contributed from generic cross-context disruption. The paper's figure needs only the deceptive arm.

**Output:** `cross_question_dec.npz` → Figure S5

### Step 6 — Induction-Head Criteria

Computes prefix-matching and copying scores for every head on repeated random tokens. Needs no dataset and runs in minutes.

```bash
python experiments/induction_heads.py --model llama
```

**Output:** `runs/<model>/induction/induction_heads.npz` → Figure S4

### Base vs Instruction-Tuned

The base checkpoint has no chat template and no assistant persona, so the opinion is stated as a bare assertion of fact, and the instruct model is re-run with that same phrasing (Appendix D.5).

```bash
python experiments/screen_mmlu.py  --model llama-base --format base
python experiments/screen_mmlu.py  --model llama      --format pronoun_free
python experiments/patch_heads.py  --model llama-base --condition mmlu_base
python experiments/patch_heads.py  --model llama      --condition mmlu_base_matched
python experiments/patch_heads.py  --model llama-base --condition mmlu_label_swap
```

## Figures

Each script reads the `.npz` files from `runs/` and writes PDFs and PNGs to `figures/output/<model>/`. Panels whose inputs are missing are skipped rather than aborting the run.

| Script | Figure | Description |
|--------|--------|-------------|
| `fig3_depth.py` | Fig. 3 | Logit lens and residual-stream patching by layer |
| `fig4_heads.py` | Fig. 4, S3 | Head heatmaps, condition scatter, answer probes, ablation |
| `fig5_generalization.py` | Fig. 5, 6A | Early-head correlations across prompt formats |
| `fig6_pushback_probes.py` | Fig. 6B | Answer probes on pushback-only heads |
| `figS1_screening_summary.py` | Fig. S1 | Usable examples and sycophancy rates across models |
| `figS4_induction.py` | Fig. S4 | Induction-head criteria |
| `figS5_cross_question.py` | Fig. S5 | Cross-question patching, reference vs content |
| `figS6_base_vs_instruct.py` | Fig. S6 | Opinion and retrieval heads before and after tuning |

```bash
python figures/fig3_depth.py --model llama
python figures/figS1_screening_summary.py --models llama mistral gemma
```

Mistral's Figures S7–S12 and Gemma's S13–S18 are the same scripts with `--model mistral` / `--model gemma`.

## Citation

If you use this code, please cite our paper (citation to be added upon publication).
