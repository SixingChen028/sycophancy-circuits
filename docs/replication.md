# Replication guide

## Requirements

The reported runs used a single 48GB NVIDIA L40S GPU with 8 CPU cores and 48–64GB
of system memory, with all models loaded from their public HuggingFace
repositories in bfloat16. No individual job exceeded four hours.

Llama and Gemma are gated repositories, so run `huggingface-cli login` before the
first download. Two dataset-construction stages call the Anthropic API and need
`ANTHROPIC_API_KEY`; nothing else does.

Approximate cost per model:

| Stage | Time |
|---|---|
| MMLU screening | ~30 min |
| Residual-stream patching, per condition | ~30 min |
| Logit lens | ~20 min |
| Per-head patching sweep, per condition | 2–4 h |
| Cross-question patching, per arm | 2–4 h |
| Cumulative patching / ablation | ~1 h each |
| Write-in collection | ~20 min |
| Induction-head criteria | ~5 min |

The head sweeps dominate. They are `n_layers x n_heads x ceil(N/batch)` forward
passes, and they checkpoint after every layer: if a job times out, resubmit the
identical command and it resumes from the last completed layer.

## Order of operations

Stages on the same line are independent of each other.

```
1.  scripts/build_mmlu_dataset.py                        (once, model-agnostic)
2.  experiments/screen_mmlu.py                           (per model)
3.  logit_lens.py | patch_residual_stream.py | patch_heads.py     (per condition)
4.  patch_heads_cumulative.py | ablate_heads.py | collect_writeins.py
5.  probe_answers.py                                     (CPU, needs step 4)
6.  figures/*.py
```

Everything in step 3 depends only on screening, so those jobs can all run at
once. Steps 4 and 5 consume the per-head sweep and must wait for it.

`bash slurm/submit_model.sh <model>` submits all of this with the dependencies
already expressed as SLURM job dependencies.

## The MMLU conditions

```bash
python scripts/build_mmlu_dataset.py
python experiments/screen_mmlu.py --model llama

for CONDITION in mmlu mmlu_label_swap mmlu_content_swap; do
    python experiments/patch_heads.py --model llama --condition $CONDITION
done
python experiments/patch_residual_stream.py --model llama --condition mmlu
python experiments/patch_residual_stream.py --model llama --condition mmlu_label_swap
python experiments/logit_lens.py            --model llama --condition mmlu

python experiments/ablate_heads.py          --model llama --condition mmlu
python experiments/patch_heads_cumulative.py --model llama --condition mmlu
python experiments/collect_writeins.py      --model llama --condition mmlu
python experiments/probe_answers.py         --model llama --condition mmlu
```

The multi-round and pushback conditions reuse screens 1–3 from the single-round
screening, so run `screen_mmlu.py` first:

```bash
python experiments/screen_multiturn.py --model llama --mode stated
python experiments/screen_multiturn.py --model llama --mode pushback
python experiments/patch_heads.py --model llama --condition mmlu_multiturn
python experiments/patch_heads.py --model llama --condition mmlu_pushback
```

## The description-based condition

Descriptions are generated once and reused across models: they depend only on
the questions, and each model's screens select a different subset, so generating
per model would waste calls and break comparability.

```bash
export ANTHROPIC_API_KEY=...
python scripts/generate_descriptions.py --limit 50    # pilot, check the attrition rate
python scripts/generate_descriptions.py               # full run, ~7,400 calls
python experiments/screen_description.py --model llama
python experiments/patch_heads.py --model llama --condition mmlu_description
```

Overlap is enforced mechanically after generation rather than trusted to the
instruction: the option letter is excluded by regular expression, and shared
content words are checked at word level. A description that still overlaps is
sent back for a rewrite up to twice, then dropped.

## The TriviaQA condition

Free-form answering has no fixed option set, so the wrong answer the user states
has to be constructed from the model's own next-best guess and then validated.

```bash
python scripts/build_triviaqa_dataset.py
python experiments/screen_triviaqa.py --model llama --stage plain
python experiments/screen_triviaqa.py --model llama --stage guess
python scripts/judge_triviaqa_answers.py --model llama          # needs the API key
python experiments/screen_triviaqa.py --model llama --stage opinion
python experiments/screen_triviaqa.py --model llama --stage build
python experiments/patch_heads.py --model llama --condition triviaqa
```

The judge stage is not optional in practice. Beam search sometimes returns a
degenerate continuation that echoes the question, or a synonym of the correct
answer that the alias list does not cover; both would corrupt the sycophancy
measurement. The opinion stage will run without it and warns when it does.

The logit lens does not apply here: it reads a fixed four-way restricted softmax,
and TriviaQA answers are free-form. `patch_heads.py` handles the difference by
switching to a per-example two-token readout, comparing the first token of the
correct answer against the first token of the stated wrong answer. Questions
whose two answers share a first token carry no signal and are dropped.

## Cross-question patching (Appendix D.4)

Asks what form the opinion signal takes: a context-dependent **reference** to the
user's preferred answer, or transferable answer **content**. Patching a head from
one question's run into a *different* question's run separates them — content
should survive the move, a reference should not.

```bash
python experiments/patch_cross_question.py --model llama --source_arm dec --seed 0
python figures/figS5_cross_question.py --model llama
```

That single arm is what Figure S5 reports. It is a full head sweep over ~1,300
question pairs, so budget the same 2–4 hours as the within-question sweeps, and
it checkpoints per layer.

An optional control arm isolates what the source's *opinion* contributed from
generic cross-context disruption, since patching across questions disrupts the
target whatever the head holds:

```bash
python experiments/patch_cross_question.py --model llama --source_arm plain --seed 0
```

**If you run it, the seeds must match.** The arms are comparable only under the
same target/source pairing; `load_arms` refuses to proceed otherwise.

**Read the size caveat.** The source question's answer sits around 1e-07
probability in the target run; the largest effect moves it to about 2.5e-07.
These are directions written into the residual stream, not the model coming near
that answer. The script prints the caveat with every report.

Two things not to do with the output, both learned the hard way:

- Do not plot the control contrast as a per-head map. It is a sound scalar
  summary for the source's answer, but at heads where the two arms differ mostly
  in how much they restore the target's correct answer, the contrast inverts the
  sign of what actually happened. The control figure shows the two arms side by
  side instead.
- Do not use a restricted softmax over the three read-out tokens. The source's
  answer is not a candidate in the target's answer space, so the restricted
  probability is ~0 and the effect vanishes. It is the right measure for opinion
  removal and the wrong one for content transfer.

Finally, a significant *negative* control contrast is not a null. Reporting
"these heads carry nothing" requires the interval to contain zero; an interval
lying entirely below zero is a real effect in the other direction and needs the
arms inspected separately. The paper's claim does not rest on the contrast: it
is the ratio of the two populations' effects in the deceptive arm, about one
fifth.

## Adding a model

Add one entry to `MODELS` in `src/syco/models.py`:

```python
"my-model": ModelSpec(
    key="my-model",
    hf_id="org/my-model",
    answer_ids=(...),          # token ids of bare "A", "B", "C", "D"
    n_layers=..., n_heads=..., d_head=..., critical_layer=...,
),
```

Then run the pipeline with `--model my-model`. No analysis code changes.

Three details decide whether the numbers mean anything:

**Answer token ids** are not portable across tokenizers. They must be the ids of
a bare capital letter with no leading space, since they follow `(` in the prompt.
`verify_answer_ids` re-derives them from the tokenizer at load time and refuses
to run if the registry disagrees, so a wrong value fails loudly.

**`d_head`** is the width of one head's slice of the `o_proj` input. It is
usually `hidden_size // n_heads`, but not always: Gemma-2 sets `head_dim=256`
explicitly with `hidden_size=3584` and 16 heads, so its attention output is
4096 wide — larger than the hidden size. Deriving `d_head` by division there
would silently misalign every head's slice and produce plausible but meaningless
numbers. `verify_geometry` checks this against the loaded config.

**`critical_layer`** is the last layer at which the median plain-to-label-swap
normalized logit difference remains below 0.1, and it defines the early/late
split. Determine it by running `patch_residual_stream.py --condition
mmlu_label_swap` first and reading the curve. The threshold is not delicate: in
all three models the curve stays flat and near zero through the early band, then
rises above 0.66 within two layers, and any threshold between 0.050 and 0.131
gives the same layer for all three. A criterion needing no effect-size threshold
at all — the last layer at which no single prompt reaches 50% recovery — returns
the same layers. The logit lens agrees independently: the correct answer's
probability is still near chance at the critical layer (0.23, 0.31 and 0.21
against a chance level of 0.25) and first exceeds 0.4 exactly one layer later.

Architectures other than Llama, Mistral and Gemma-2 may also need
`syco.hooks.layer_modules` adjusted; it currently assumes `model.model.layers`.

## Reproducibility notes

Screening is deterministic: it reads argmax over logits with no sampling.
TriviaQA's wrong-answer construction uses beam search with `do_sample=False`, so
it is deterministic given the model, but the LLM judge that filters those
candidates is not, and neither is description generation. Both write their
output to a file that is then reused, so downstream stages are reproducible from
the committed artifacts even though the generation step is not bit-identical on
a rerun.

Example order is fixed: screened records are always sorted by `orig_idx`, so the
per-example axis of every saved array lines up across experiments and models.
