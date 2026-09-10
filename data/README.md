# Datasets

This directory holds the question sets. Nothing here is committed; each file is
produced by a script in `scripts/`.

| File | Produced by | Contents |
|---|---|---|
| `mmlu_stem.json` | `scripts/build_mmlu_dataset.py` | 15 MMLU STEM subjects (paper Table 1) |
| `triviaqa.json` | `scripts/build_triviaqa_dataset.py` | TriviaQA `rc.nocontext` validation split |
| `descriptions.json` | `scripts/generate_descriptions.py` | Referring descriptions for the description-based condition |

All three are model-agnostic and are shared across every model in the study.
Model-specific screening output goes to `runs/<model>/<condition>/` instead.
