# Tracing mechanisms of sycophantic agreement in language models

Code for the paper. Sycophantic agreement is a model changing an otherwise
correct answer to match a user's stated position. This repository contains the
causal mediation analysis that locates the mechanism behind it.

**The result in one paragraph.** A stated opinion is written into the last
prompt token's residual stream several layers *before* the model retrieves an
answer, and it biases that retrieval. A sparse set of early attention heads
carries the opinion signal; ablating them removes most sycophantic agreement
while leaving factual accuracy largely intact. What those heads carry is a
*reference* to the user's preferred answer rather than the answer itself: moved
into an unrelated prompt, most of their effect disappears, unlike the answer
retrieval heads. The same heads carry the opinion however it is phrased — by
letter, in free-form text, in a second conversational turn, or as a description
sharing no words with the answer. Content-free pushback ("Are you sure?")
recruits a *different* set of heads, which suppress the model's original correct
answer rather than promoting a stated one.

Models studied: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Gemma-2-9B-it,
plus the Llama-3.1-8B base checkpoint for the before/after post-training
comparison.

## Install

```bash
git clone <this repository>
cd sycophancy-circuits
pip install -e .            # add '.[llm]' for the two dataset stages that call an API
```

Llama and Gemma are gated on HuggingFace; run `huggingface-cli login` first.
Every analysis runs on a single 48GB GPU with 8 CPU cores and 48–64GB of system
memory, and no individual job exceeds four hours.

## Quick start

```bash
python scripts/build_mmlu_dataset.py                          # download the questions
python experiments/screen_mmlu.py       --model llama         # select usable questions
python experiments/patch_heads.py       --model llama --condition mmlu
python experiments/patch_heads.py       --model llama --condition mmlu_label_swap
python figures/fig4_heads.py            --model llama
```

On a cluster, `bash slurm/submit_model.sh llama` submits the whole pipeline with
the dependencies between stages already wired up.

## How the code is organized

Every causal experiment in the paper has the same shape: run the model on a
**source** prompt where it answers correctly, run it on a **target** prompt where
it answers differently, patch a component from source into target, and measure
how far the answer moves back. Conditions differ only in which two prompts those
are and how the answer is read out. So each analysis is one script, and the
model and the condition are arguments:

```bash
python experiments/patch_heads.py --model gemma --condition mmlu_multiturn
```

| Path | What it holds |
|---|---|
| `src/syco/models.py` | Model registry: HF id, answer token ids, head geometry, critical layer |
| `src/syco/prompts.py` | Every prompt format in the paper |
| `src/syco/conditions.py` | The source/target prompt pairs, one per condition |
| `src/syco/crossquestion.py` | Pairing and analysis for cross-question patching |
| `src/syco/hooks.py` | Caching, patching and ablation hooks |
| `src/syco/metrics.py` | Normalized logit difference and answer rates |
| `experiments/` | One script per analysis, parameterized by model and condition |
| `scripts/` | Dataset construction (two stages call the Anthropic API) |
| `figures/` | One script per paper figure |
| `slurm/` | Cluster runner and a whole-pipeline submitter |
| `tests/` | Unit tests plus an end-to-end run on a tiny random model |

Adding a model means adding one entry to `MODELS`. Adding a condition means
adding one `Condition` with two prompt builders. Neither requires touching any
analysis.

### Output layout

Everything a run produces lands under `runs/<model>/<condition>/`, so a complete
replication is one directory and nothing is written anywhere else. Set
`SYCO_RUNS` to relocate it.

```
runs/llama/mmlu/
    screened_examples.json          screening (experiments/screen_mmlu.py)
    logit_lens.npz                  Figure 3A
    residual_patching.npz           Figure 3B
    head_patching.npz               Figures 4A, 4B
    head_patching_cumulative.npz    Figure S2
    head_ablation.npz               Figures 4F, S3
    writeins.npz, answer_probes.npz Figures 4D, 4E

runs/llama/triviaqa/
    cross_question_dec.npz          reference-vs-content sweep     Figure S5
    cross_question_plain.npz        optional no-opinion control arm
```

## Conditions

| `--condition` | Source | Target | Paper |
|---|---|---|---|
| `mmlu` | plain | opinion names an option by letter | Fig 3, 4 |
| `mmlu_label_swap` | plain | correct and wrong options exchange letters | Fig 3B, 4A |
| `mmlu_content_swap` | plain | correct and wrong options exchange text | Fig 4E |
| `triviaqa` | plain | opinion states a free-form wrong answer | Fig 5A |
| `mmlu_multiturn` | round 1 | round 2 after a stated opinion | Fig 5B |
| `mmlu_description` | plain | opinion describes the option, sharing no words | Fig 5C |
| `mmlu_pushback` | round 1 | round 2 after content-free doubt | Fig 6 |
| `mmlu_base` / `mmlu_base_matched` | plain | wrong answer asserted as fact | Fig S5 |

One analysis does not fit that table. **Cross-question patching**
(`experiments/patch_cross_question.py`, paper Appendix D.4) pairs each TriviaQA
question with a *different* one and patches between them, to ask whether an
opinion head carries a context-dependent reference or transferable answer
content. It has its own pairing and read-out, described in
`src/syco/crossquestion.py`.

The two swap conditions state **no opinion at all**. They move the answer for a
reason unrelated to sycophancy, which is what makes them the control that
isolates generic answer retrieval from opinion processing.

## Documentation

- [`docs/replication.md`](docs/replication.md) — full run order, costs, cluster notes
- [`docs/paper_map.md`](docs/paper_map.md) — every figure and table, and the command that produces it

## Tests

```bash
pytest tests
```

The suite covers the prompt formats as golden strings, the metrics, and the
intervention hooks — the last against invariants on a small real transformer,
including that patching the final layer's residual stream reproduces the source
run's logits exactly, and that per-head write-ins sum to the attention output.
`tests/test_pipeline.py` runs each experiment end to end on a tiny random model.

## Citation

```bibtex
@article{sycophancy-circuits,
  title  = {Tracing mechanisms of sycophantic agreement in language models},
  year   = {2026},
  note   = {Preprint, under review}
}
```
