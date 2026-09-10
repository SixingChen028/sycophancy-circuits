"""Cross-question pairing and analysis."""

import numpy as np

from syco.crossquestion import (SOURCE_OPINION, bootstrap_ci, delta_log_prob,
                                make_pairs, per_head_effect, population_summary)
from syco.heads import complement
from syco.models import ModelSpec

SPEC = ModelSpec(key="t", hf_id="t", answer_ids=(0, 1, 2, 3), n_layers=4,
                 n_heads=3, d_head=8, critical_layer=1)


def test_pairing_never_pairs_a_question_with_itself():
    correct = np.arange(100, 140)
    wrong = np.arange(200, 240)
    targets, sources = make_pairs(correct, wrong, seed=0)
    assert not np.any(targets == sources)
    assert len(set(targets.tolist())) == len(targets)


def test_pairing_drops_read_out_token_collisions():
    """The three read-outs must be distinct or they contaminate each other.

    Constructed so collisions are unavoidable: every question's wrong answer is
    the next question's correct answer, so any pairing of i with i-1 must drop.
    """
    n = 12
    correct = np.arange(n)
    wrong = (np.arange(n) + 1) % n
    total_dropped = 0
    for seed in range(10):
        targets, sources = make_pairs(correct, wrong, seed)
        total_dropped += n - len(targets)
        for t, s in zip(targets, sources):
            assert wrong[s] != correct[t], "kept a pair colliding with c_t"
            assert wrong[s] != wrong[t], "kept a pair colliding with w_t"
    assert total_dropped > 0, "the collision filter was never exercised"


def test_pairing_is_reproducible_from_the_seed():
    correct, wrong = np.arange(100, 130), np.arange(200, 230)
    a = make_pairs(correct, wrong, seed=7)
    b = make_pairs(correct, wrong, seed=7)
    c = make_pairs(correct, wrong, seed=8)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not (np.array_equal(a[1], c[1]))


def _fake_arm(n_layers=4, n_heads=3, n_pairs=5, shift=0.0):
    rng = np.random.default_rng(0)
    base_logits = rng.normal(size=(3, n_pairs)).astype(np.float32)
    base_logZ = np.full(n_pairs, 5.0, dtype=np.float32)
    patched_logits = base_logits[None, None] + shift
    patched_logits = np.repeat(np.repeat(patched_logits, n_layers, 0), n_heads, 1)
    patched_logZ = np.full((n_layers, n_heads, n_pairs), 5.0, dtype=np.float32)
    return {"base_logits": base_logits, "base_logZ": base_logZ,
            "patched_logits": patched_logits.astype(np.float32),
            "patched_logZ": patched_logZ}


def test_delta_log_prob_is_zero_when_nothing_changes():
    delta = delta_log_prob(_fake_arm(shift=0.0))
    assert delta.shape == (4, 3, 3, 5)
    assert np.allclose(delta, 0.0, atol=1e-5)


def test_delta_log_prob_tracks_a_uniform_logit_shift():
    """With the normalizer held fixed, a logit shift moves log-prob one for one."""
    delta = delta_log_prob(_fake_arm(shift=0.5))
    assert np.allclose(delta, 0.5, atol=1e-5)


def test_delta_log_prob_subtracts_the_normalizer():
    """A shift in logsumexp alone must lower every read-out's log-probability."""
    arm = _fake_arm(shift=0.0)
    arm["patched_logZ"] = arm["patched_logZ"] + 1.0
    assert np.allclose(delta_log_prob(arm), -1.0, atol=1e-5)


def test_per_head_effect_averages_over_pairs():
    delta = np.zeros((4, 3, 3, 5))
    delta[2, 1, SOURCE_OPINION, :] = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert np.allclose(per_head_effect(delta, [(2, 1)]), [3.0])
    assert len(per_head_effect(delta, [])) == 0


def test_population_summary_without_a_control_arm_omits_the_contrast():
    """The paper's measure needs only the deceptive arm."""
    dec = np.zeros((4, 3, 3, 6))
    dec[1, 0, SOURCE_OPINION, :] = 0.8
    summary = population_summary(dec, [(1, 0)])
    assert np.isclose(summary["w_s"], 0.8)
    assert "transfer" not in summary


def test_population_summary_reports_transfer_as_dec_minus_plain():
    dec = np.zeros((4, 3, 3, 6))
    plain = np.zeros((4, 3, 3, 6))
    dec[1, 0, SOURCE_OPINION, :] = 0.8
    plain[1, 0, SOURCE_OPINION, :] = 0.3
    summary = population_summary(dec, [(1, 0)], plain)
    assert np.isclose(summary["w_s"], 0.8)
    assert np.isclose(summary["transfer"], 0.5)
    assert summary["pairs_positive"] == 1.0
    low, high = summary["transfer_ci"]
    assert low <= 0.5 <= high


def test_population_summary_counts_the_fraction_of_pairs_moving_up():
    dec = np.zeros((4, 3, 3, 4))
    plain = np.zeros((4, 3, 3, 4))
    dec[1, 0, SOURCE_OPINION, :] = [1.0, 1.0, -1.0, -1.0]
    summary = population_summary(dec, [(1, 0)], plain)
    assert np.isclose(summary["pairs_positive"], 0.5)


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=2.0, size=500)
    low, high = bootstrap_ci(values, n_resamples=2000, seed=3)
    assert low < values.mean() < high
    assert bootstrap_ci(values, n_resamples=2000, seed=3) == (low, high)


def test_complement_is_the_rest_of_the_band():
    population = [(0, 0), (1, 1)]
    early = complement(population, SPEC, "early")
    # Early band is layers 0..critical_layer inclusive: 2 layers x 3 heads = 6.
    assert len(early) == 6 - 2
    assert (0, 0) not in early and (1, 1) not in early
    assert all(l <= SPEC.critical_layer for l, _ in early)
    late = complement(population, SPEC, "late")
    assert all(l > SPEC.critical_layer for l, _ in late)
    assert len(late) == (SPEC.n_layers - SPEC.critical_layer - 1) * SPEC.n_heads


if __name__ == "__main__":
    import sys
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS  {name}")
            except Exception as exc:
                failures += 1
                import traceback; traceback.print_exc()
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failures")
    sys.exit(1 if failures else 0)
