"""Metrics, head selection, example selection and TriviaQA matching."""

import numpy as np

from syco.data import strongest_shift
from syco.heads import band_mask, rank_heads, top_heads
from syco.metrics import (accuracy_summary, aggregate, logit_difference,
                          normalized_logit_difference)
from syco.models import MODELS, ModelSpec
from syco.overlap import overlap_words
from syco.triviaqa import is_correct, is_sycophantic, normalize

SPEC = ModelSpec(key="t", hf_id="t", answer_ids=(0, 1, 2, 3), n_layers=4,
                 n_heads=3, d_head=8, critical_layer=1)


def test_normalized_logit_difference_endpoints():
    source = np.array([2.0, 4.0])
    target = np.array([-2.0, 0.0])
    # Patching that recovers nothing scores 0; full recovery scores 1.
    assert np.allclose(normalized_logit_difference(target, source, target), 0.0)
    assert np.allclose(normalized_logit_difference(source, source, target), 1.0)
    midpoint = (source + target) / 2
    assert np.allclose(normalized_logit_difference(midpoint, source, target), 0.5)


def test_zero_denominator_is_excluded_rather_than_infinite():
    source = np.array([1.0, 1.0])
    target = np.array([1.0, 0.0])   # first example has no gap
    result = normalized_logit_difference(np.array([1.0, 0.5]), source, target)
    assert np.isnan(result[0])
    assert np.isclose(result[1], 0.5)
    assert np.isclose(aggregate(result), 0.5)


def test_logit_difference_indexes_per_example_columns():
    scores = np.array([[1.0, 5.0, 0.0, 0.0],
                       [3.0, 0.0, 0.0, 1.0]])
    diff = logit_difference(scores, np.array([1, 0]), np.array([0, 3]))
    assert np.allclose(diff, [4.0, 2.0])


def test_accuracy_summary_counts_correct_and_sycophantic_answers():
    scores = np.array([[0.0, 9.0, 0.0, 0.0],    # answers B
                       [9.0, 0.0, 0.0, 0.0],    # answers A
                       [0.0, 0.0, 9.0, 0.0]])   # answers C
    summary = accuracy_summary(scores, np.array([1, 1, 1]), np.array([0, 0, 2]))
    assert np.isclose(summary["accuracy"], 1 / 3)
    assert np.isclose(summary["sycophancy"], 2 / 3)


def test_band_mask_splits_at_the_critical_layer_inclusive():
    early = band_mask(SPEC, "early")
    late = band_mask(SPEC, "late")
    # The critical layer itself is early: heads there are opinion candidates.
    assert early[SPEC.critical_layer].all()
    assert late[SPEC.critical_layer + 1].all()
    assert not (early & late).any()
    assert (early | late).all()


def test_top_heads_respects_the_band():
    scores = np.zeros((SPEC.n_layers, SPEC.n_heads))
    scores[3, 0] = 10.0    # late band
    scores[0, 1] = 5.0     # early band
    assert top_heads(scores, SPEC, "early", 1) == [(0, 1)]
    assert top_heads(scores, SPEC, "late", 1) == [(3, 0)]
    assert rank_heads(scores, SPEC, "all")[0][:2] == (3, 0)


def test_strongest_shift_picks_the_most_sycophantic_option():
    example = {
        "correct_answer": 0,
        "logits_dec": {"1": [1.0, 0.5, 0, 0],    # gap  0.5
                       "2": [1.0, 0, 3.0, 0],    # gap -2.0  <- strongest
                       "3": [1.0, 0, 0, 0.9]},   # gap  0.1
    }
    assert strongest_shift(example, "logits_dec") == 2


def test_triviaqa_normalization():
    assert normalize("The Beatles") == "beatles"
    assert normalize("Paris.") == "paris"
    assert normalize("Hydrogen\nIt is the lightest element") == "hydrogen"
    assert normalize("  a  Red   Car ") == "red car"


def test_triviaqa_matching_uses_aliases():
    example = {"normalized_answer": "beatles",
               "normalized_aliases": ["the beatles", "fab four"]}
    assert is_correct("The Beatles", example)
    assert is_correct("Fab Four", example)
    assert not is_correct("Rolling Stones", example)


def test_sycophancy_requires_adopting_the_stated_answer():
    example = {"normalized_answer": "hydrogen", "normalized_aliases": []}
    assert is_sycophantic("Helium", "Helium", example)
    # Merely being wrong is confusion, not sycophancy.
    assert not is_sycophantic("Methane", "Helium", example)
    # Staying correct is not sycophancy either.
    assert not is_sycophantic("Hydrogen", "Helium", example)


def test_overlap_rejects_shared_content_words_and_the_letter():
    assert overlap_words("the green organelle in plant cells", "chloroplasts", "B") == set()
    assert "plant" in overlap_words("the plant organelle", "plant cells", "B")
    # Stopwords do not count as overlap.
    assert overlap_words("the one of them", "the cell of it", "C") == set()
    assert overlap_words("option B here", "chloroplasts", "B") == {"letter B"}


def test_registry_geometry_is_self_consistent():
    for key, spec in MODELS.items():
        assert len(spec.answer_ids) == 4, key
        assert len(set(spec.answer_ids)) == 4, key
        assert 0 <= spec.critical_layer < spec.n_layers, key
        last = spec.head_slice(spec.n_heads - 1)
        assert last.stop == spec.n_heads * spec.d_head, key


if __name__ == "__main__":
    import sys
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS  {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failures")
    sys.exit(1 if failures else 0)
