"""End-to-end smoke test of the experiment scripts on a tiny random model.

Runs the real `main()` of each GPU experiment against a three-layer randomly
initialized Llama and a synthetic screening file, with model loading redirected
to the tiny model. It asserts on shapes and invariants rather than values -- the
model is random, so the numbers are meaningless -- but it exercises the whole
path: condition building, prompt rendering, batching, the readouts, the patching
loops, checkpointing and the saved schema.

This is what catches integration bugs that unit tests on the pieces cannot.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

# runs/ location is read from the environment at import time.
_TMP = tempfile.mkdtemp(prefix="syco-test-")
os.environ["SYCO_RUNS"] = _TMP

sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, HERE)

from tiny_model import VOCAB, build  # noqa: E402
import syco.runtime as runtime       # noqa: E402


class StubTokenizer:
    """Enough of a HF tokenizer for the experiment scripts, with no download."""

    pad_token = "<pad>"
    pad_token_id = 0
    padding_side = "left"
    bos_token_id = 1
    vocab_size = VOCAB

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False,
                            continue_final_message=True):
        return " ".join(m["content"] for m in messages)

    def _ids(self, text):
        # Deterministic word-level hashing; the values do not matter, only that
        # different prompts give different sequences.
        return [2 + (hash(w) % (VOCAB - 3)) for w in text.split()][:24] or [2]

    def __call__(self, texts, return_tensors=None, padding=False,
                 truncation=False, max_length=None, add_special_tokens=True):
        if isinstance(texts, str):
            return {"input_ids": self._ids(texts)}
        sequences = [self._ids(t) for t in texts]
        width = max(len(s) for s in sequences)
        input_ids, mask = [], []
        for s in sequences:
            pad = width - len(s)
            input_ids.append([self.pad_token_id] * pad + s)   # left padding
            mask.append([0] * pad + [1] * len(s))
        from transformers import BatchEncoding
        return BatchEncoding({
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(mask),
        })


def synthetic_screening(n=6, seed=0):
    """A screening file with the schema the conditions registry expects."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        correct = int(rng.integers(0, 4))
        wrong_idxs = [w for w in range(4) if w != correct]

        def logits():
            return {str(w): list(rng.normal(size=4).round(3)) for w in wrong_idxs}

        records.append({
            "orig_idx": i,
            "subject": "astronomy",
            "question": f"Question number {i} about the solar system",
            "choices": [f"option {i}{c}" for c in "abcd"],
            "correct_answer": correct,
            "wrong_idxs": wrong_idxs,
            "screen1": True, "screen2": True, "screen3": True, "screen4": True,
            "pass_all": True,
            "logits_plain": list(rng.normal(size=4).round(3)),
            "logits_content_swap": logits(),
            "logits_label_swap": logits(),
            "logits_dec": logits(),
        })
    return records


def synthetic_triviaqa(n=8, seed=1):
    """A TriviaQA screening file with the fields cross-question patching reads."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        records.append({
            "orig_idx": i,
            "question_id": f"q{i}",
            "question": f"Trivia question number {i} about elements",
            "answer": f"answer{i}",
            "plain_raw_answer": f"alpha{i}",
            "wrong_guess": f"beta{i}",
            "screen1": True, "screen2": True, "screen3": True, "screen4": True,
            "pass_all": True,
        })
    return records


def setup_module():
    """Redirect model loading to the tiny model and write the screening files."""
    spec, model = build()
    tok = StubTokenizer()
    runtime.load_model = lambda model_key, verify=True: (spec, tok, model, "cpu")

    path = runtime.run_path("tiny", "mmlu", "screened_examples.json")
    with open(path, "w") as f:
        json.dump(synthetic_screening(), f)

    path = runtime.run_path("tiny", "triviaqa", "screened_examples.json")
    with open(path, "w") as f:
        json.dump(synthetic_triviaqa(), f)


def teardown_module():
    shutil.rmtree(_TMP, ignore_errors=True)


def run(module_name, argv):
    """Invoke an experiment's main() with the given command line."""
    import importlib

    module = importlib.import_module(module_name)
    module.load_model = runtime.load_model
    saved = sys.argv
    sys.argv = [module_name] + argv
    try:
        module.main()
    finally:
        sys.argv = saved


def test_head_patching_end_to_end():
    run("patch_heads", ["--model", "tiny", "--condition", "mmlu",
                        "--batch_size", "3", "--cache_batch_size", "3"])
    spec, _ = build()
    data = np.load(runtime.run_path("tiny", "mmlu", "head_patching.npz"))
    assert data["norm_logit_diff"].shape == (spec.n_layers, spec.n_heads)
    assert data["per_example"].shape == (spec.n_layers, spec.n_heads, 6)
    assert data["source_ld"].shape == (6,)
    assert np.isfinite(data["norm_logit_diff"]).all()
    # The checkpoint is removed once the sweep finishes.
    assert not os.path.exists(
        runtime.run_path("tiny", "mmlu", "head_patching.ckpt.npz"))


def test_residual_patching_end_to_end():
    run("patch_residual_stream", ["--model", "tiny", "--condition", "mmlu",
                                  "--batch_size", "3"])
    spec, _ = build()
    data = np.load(runtime.run_path("tiny", "mmlu", "residual_patching.npz"))
    assert data["norm_logit_diff"].shape == (spec.n_layers,)
    assert data["per_example"].shape == (spec.n_layers, 6)
    # Patching the final layer must fully restore the source run.
    assert np.isclose(data["norm_logit_diff"][-1], 1.0, atol=1e-3)


def test_logit_lens_end_to_end():
    run("logit_lens", ["--model", "tiny", "--condition", "mmlu",
                       "--batch_size", "3"])
    spec, _ = build()
    data = np.load(runtime.run_path("tiny", "mmlu", "logit_lens.npz"))
    assert data["p_correct_source"].shape == (spec.n_layers, 6)
    total = (data["p_correct_source"] + data["p_syco_source"]
             + 2 * data["p_other_source"])
    # The four restricted probabilities must sum to one at every layer.
    assert np.allclose(total, 1.0, atol=1e-5)


def test_ablation_end_to_end():
    run("ablate_heads", ["--model", "tiny", "--condition", "mmlu",
                         "--batch_size", "3", "--k", "3"])
    data = np.load(runtime.run_path("tiny", "mmlu", "head_ablation.npz"))
    assert data["candidate_heads"].shape == (3, 2)
    assert data["cumulative_sycophancy_rate"].shape == (3,)
    assert sorted(map(tuple, data["ablation_order"])) == \
        sorted(map(tuple, data["candidate_heads"]))
    for key in ("cumulative_plain_accuracy", "cumulative_sycophancy_rate"):
        assert ((data[key] >= 0) & (data[key] <= 1)).all()


def test_cumulative_patching_end_to_end():
    run("patch_heads_cumulative", ["--model", "tiny", "--condition", "mmlu",
                                   "--batch_size", "3", "--k", "3"])
    data = np.load(runtime.run_path("tiny", "mmlu",
                                    "head_patching_cumulative.npz"))
    assert data["heads"].shape == (3, 2)
    assert data["sycophancy_rate"].shape == (3,)


def test_writeins_end_to_end():
    # The retrieval population needs both relabelling sweeps.
    for condition in ("mmlu_label_swap", "mmlu_content_swap"):
        run("patch_heads", ["--model", "tiny", "--condition", condition,
                            "--batch_size", "3", "--cache_batch_size", "3"])
    # probe_answers.py is not covered here: it needs scikit-learn.
    run("collect_writeins", ["--model", "tiny", "--condition", "mmlu",
                             "--batch_size", "3", "--k", "2",
                             "--n_components", "3"])
    data = np.load(runtime.run_path("tiny", "mmlu", "writeins.npz"))
    assert data["opinion_heads"].shape == (2, 2)
    assert data["retrieval_heads"].shape == (2, 2)
    assert data["opinion_source_pca"].shape == (2, 6, 3)
    # PCA variance ratios are a proportion of total variance.
    assert (data["opinion_variance_ratio"] >= 0).all()
    assert data["opinion_variance_ratio"].sum(axis=1).max() <= 1.0 + 1e-5


def test_cross_question_patching_end_to_end():
    """Both arms, then the analysis that compares them."""
    from syco.crossquestion import delta_log_prob, load_arms, population_summary

    spec, _ = build()
    for arm in ("dec", "plain"):
        run("patch_cross_question", ["--model", "tiny", "--source_arm", arm,
                                     "--seed", "0", "--batch_size", "4",
                                     "--cache_batch_size", "4"])
        data = np.load(runtime.run_path("tiny", "triviaqa",
                                        f"cross_question_{arm}.npz"))
        n_pairs = data["t_idx"].shape[0]
        assert data["patched_logits"].shape == (spec.n_layers, spec.n_heads, 3,
                                                n_pairs)
        assert data["patched_logZ"].shape == (spec.n_layers, spec.n_heads,
                                              n_pairs)
        assert np.isfinite(data["patched_logits"]).all(), "unfilled sweep cells"
        # No question is ever its own source.
        assert not np.any(data["t_idx"] == data["s_idx"])
        # The checkpoint is cleaned up on success.
        assert not os.path.exists(
            runtime.run_path("tiny", "triviaqa",
                             f"cross_question_{arm}.ckpt.npz"))

    dec, plain = load_arms("tiny")
    # Both arms share the target prompts, so their baselines must be identical.
    assert np.allclose(dec["base_logits"], plain["base_logits"], atol=1e-4)
    summary = population_summary(delta_log_prob(dec), [(0, 0), (1, 1)],
                                 delta_log_prob(plain))
    assert summary["n_heads"] == 2
    assert 0.0 <= summary["pairs_positive"] <= 1.0
    low, high = summary["transfer_ci"]
    assert low <= summary["transfer"] <= high


if __name__ == "__main__":
    setup_module()
    failures = 0
    order = ["test_head_patching_end_to_end", "test_residual_patching_end_to_end",
             "test_logit_lens_end_to_end", "test_ablation_end_to_end",
             "test_cumulative_patching_end_to_end",
             "test_writeins_end_to_end",
             "test_cross_question_patching_end_to_end"]
    for name in order:
        try:
            globals()[name]()
            print(f"PASS  {name}")
        except Exception as exc:
            failures += 1
            import traceback
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    teardown_module()
    print(f"\n{failures} failures")
    sys.exit(1 if failures else 0)
