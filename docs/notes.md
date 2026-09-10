# Method notes

Details that do not belong in the README: things that decide whether a number
means what it appears to mean. The run order itself is in the
[README](../README.md).

## Output layout

Everything lands under `runs/<model>/<condition>/`, so one replication is one
directory and nothing is written outside it. Set `SYCO_RUNS` to relocate it.

```
runs/llama/mmlu/
    screened_examples.json          screening
    logit_lens.npz                  Figure 3A
    residual_patching.npz           Figure 3B
    head_patching.npz               Figures 4A, 4B
    head_patching_cumulative.npz    Figure S2
    head_ablation.npz               Figures 4F, S3
    writeins.npz, answer_probes.npz Figures 4D, 4E

runs/llama/triviaqa/
    cross_question_dec.npz          Figure S5
    cross_question_plain.npz        optional control arm
```

Screened records are always sorted by `orig_idx`, so the per-example axis of
every saved array lines up across experiments and is comparable across models.

## The critical layer

The critical layer is the last layer at which the median plain-to-label-swap
normalized logit difference remains below 0.1, and it defines the early/late
split: layers `0..critical_layer` are early (opinion candidates), everything
after is late (answer retrieval candidates).

| Model | Critical layer | Early band | Late band |
|-------|---------------|------------|-----------|
| Llama-3.1-8B-Instruct | 16 | L0–16 | L17–31 |
| Mistral-7B-Instruct-v0.3 | 17 | L0–17 | L18–31 |
| Gemma-2-9B-it | 27 | L0–27 | L28–41 |

The threshold is not delicate. In all three models the curve stays flat and near
zero through the early band, then rises above 0.66 within two layers, and any
threshold between 0.050 and 0.131 gives the same layer for all three. A
criterion needing no effect-size threshold at all — the last layer at which no
single prompt reaches 50% recovery — returns the same layers. The logit lens
agrees independently: the correct answer's probability is still near chance at
the critical layer (0.23, 0.31 and 0.21 against a chance level of 0.25) and
first exceeds 0.4 exactly one layer later.

Note two conventions when reading older code or drafts. `ModelSpec.critical_layer`
here is the **last early** layer, used inclusively. Some earlier scripts store a
constant named `CRITICAL_LAYER` that is the **first late** layer, used
exclusively as `range(CRITICAL_LAYER)`. The bands are identical either way;
only the number differs by one.

## Adding a model

Add one entry to `MODELS` in `src/syco/models.py` and run the pipeline with
`--model my-model`. Three fields decide whether the results mean anything.

**`answer_ids`** are the token ids of a bare capital letter with no leading
space, since they follow `(` in the prompt. They are not portable across
tokenizers — Llama and Gemma disagree completely. `verify_answer_ids` re-derives
them from the tokenizer at load time and refuses to run if the registry
disagrees, so a wrong value fails loudly rather than silently.

**`d_head`** is the width of one head's slice of the `o_proj` input. It is
usually `hidden_size // n_heads`, but not always: Gemma-2 sets `head_dim=256`
explicitly with `hidden_size=3584` and 16 heads, so its attention output is 4096
wide, larger than the hidden size. Deriving `d_head` by division there would
misalign every head's slice and produce plausible but meaningless numbers.
`verify_geometry` checks it against the loaded config.

**`critical_layer`** has to be read off the plain-to-label-swap residual-stream
patching curve, so run that condition first. See the table above.

Architectures other than Llama, Mistral and Gemma-2 may also need
`syco.hooks.layer_modules` adjusted; it currently assumes `model.model.layers`.

## Ranking heads for ablation

Cumulative ablation orders heads by their individual ablation effect, not by
their patching effect, because the two measure different things.

Attention heads are polysemantic. Patching swaps only the opinion-dependent part
of what a head writes, since the plain and deceptive runs differ solely in
whether an opinion is stated; it isolates the opinion signal and says nothing
about the head's other functions. Ablation zeroes the head entirely, taking
those other functions with it. Ranking by individual ablation effect therefore
selects for heads whose removal is *selective*, which is the property that
matters when the question is whether intervening can suppress sycophancy at
acceptable cost.

## Cross-question patching

**Read the size caveat.** In Llama the source question's answer sits around
1e-07 probability in the target run, and the largest effect moves it to about
2.5e-07. These are directions written into the residual stream, not the model
coming near that answer. The script prints the caveat with every report; quote
the magnitude alongside any of these numbers.

**If you run the control arm, the seeds must match.** The two arms are
comparable only under the same target/source pairing, and `load_arms` refuses to
proceed otherwise.

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

A significant *negative* control contrast is not a null. Reporting "these heads
carry nothing" requires the interval to contain zero; an interval lying entirely
below zero is a real effect in the other direction. The paper's claim does not
rest on the contrast in any case: it is the ratio of the two populations'
effects in the deceptive arm, about one fifth.

## Determinism

Screening is deterministic: it reads argmax over logits with no sampling.
TriviaQA's wrong-answer construction uses beam search with `do_sample=False`, so
it is deterministic given the model, but the LLM judge that filters those
candidates is not, and neither is description generation. Both write their
output to a file that is then reused, so downstream stages are reproducible from
those artifacts even though the generation step is not bit-identical on a rerun.

The head sweeps compare patched against unpatched logits, so both passes are run
at the same batch size. Different batch boundaries would let bfloat16
nondeterminism leak into the measurement.

## Known paper/code discrepancy

Section 3.4 and Figure 6A of the manuscript quote the content-free pushback as
"Are you sure? I don't think that's right." Every run actually used the reverse
clause order, `"I don't think that's right, are you sure?"`, which is the string
in `syco.prompts.PUSHBACK`. The code is kept verbatim because it is the record
of what produced the results; the manuscript is what needs correcting.
