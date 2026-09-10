"""The intervention hooks, checked against invariants on a tiny real model.

These are the pieces where a silent bug would produce plausible-looking but
meaningless numbers rather than an error, so each is pinned to something that
must hold exactly.
"""

import torch

from syco.hooks import (ablate_heads, cache_o_proj_inputs, cache_residual_stream,
                        head_writein, patch_head_outputs, patch_residual_stream)
from tiny_model import build, random_inputs


def logits_of(model, input_ids):
    with torch.no_grad():
        return model(input_ids).logits[:, -1, :].clone()


def test_head_slices_tile_the_o_proj_input():
    """Every head's slice together must cover the whole o_proj input."""
    spec, _ = build()
    covered = []
    for head in range(spec.n_heads):
        s = spec.head_slice(head)
        covered.extend(range(s.start, s.stop))
    assert covered == list(range(spec.n_heads * spec.d_head))


def test_cached_o_proj_width_matches_head_geometry():
    spec, model = build()
    store = {}
    with cache_o_proj_inputs(model, range(spec.n_layers), store):
        with torch.no_grad():
            model(random_inputs())
    for layer in range(spec.n_layers):
        width = torch.cat(store[layer], dim=0).shape[1]
        assert width == spec.n_heads * spec.d_head


def test_patching_a_head_with_its_own_value_is_a_no_op():
    """Patching from a run into itself must leave the logits untouched."""
    spec, model = build()
    ids = random_inputs()
    baseline = logits_of(model, ids)

    store = {}
    with cache_o_proj_inputs(model, range(spec.n_layers), store):
        with torch.no_grad():
            model(ids)
    cache = {l: torch.cat(v, dim=0) for l, v in store.items()}

    for layer in range(spec.n_layers):
        for head in range(spec.n_heads):
            with patch_head_outputs(model, spec, [(layer, head)], {layer: cache[layer]}):
                patched = logits_of(model, ids)
            assert torch.allclose(patched, baseline, atol=1e-4), \
                f"self-patch changed the output at L{layer}H{head}"


def test_patching_a_head_from_a_different_run_changes_the_output():
    """A genuine patch must actually do something, or the hook is inert."""
    spec, model = build()
    target = random_inputs(seed=0)
    source = random_inputs(seed=1)
    baseline = logits_of(model, target)

    store = {}
    with cache_o_proj_inputs(model, range(spec.n_layers), store):
        with torch.no_grad():
            model(source)
    cache = {l: torch.cat(v, dim=0) for l, v in store.items()}

    changed = 0
    for layer in range(spec.n_layers):
        for head in range(spec.n_heads):
            with patch_head_outputs(model, spec, [(layer, head)], {layer: cache[layer]}):
                patched = logits_of(model, target)
            changed += not torch.allclose(patched, baseline, atol=1e-6)
    assert changed > 0, "no head patch changed the output"


def test_hooks_are_removed_after_the_context_exits():
    spec, model = build()
    ids = random_inputs()
    baseline = logits_of(model, ids)
    source = {l: torch.zeros(ids.shape[0], spec.n_heads * spec.d_head)
              for l in range(spec.n_layers)}
    with patch_head_outputs(model, spec, [(0, 0)], source):
        pass
    with ablate_heads(model, spec, [(0, 0)]):
        pass
    assert torch.allclose(logits_of(model, ids), baseline, atol=1e-6)


def test_ablating_no_heads_is_a_no_op():
    spec, model = build()
    ids = random_inputs()
    baseline = logits_of(model, ids)
    with ablate_heads(model, spec, []):
        assert torch.allclose(logits_of(model, ids), baseline, atol=1e-6)


def test_ablation_zeroes_the_head_slice():
    """After ablation the head's slice of the o_proj input must be exactly zero."""
    spec, model = build()
    ids = random_inputs()
    store = {}
    with ablate_heads(model, spec, [(1, 2)]):
        # The cache hook is registered after the ablation hook, so it observes
        # the already-zeroed tensor.
        with cache_o_proj_inputs(model, [1], store):
            with torch.no_grad():
                model(ids)
    captured = torch.cat(store[1], dim=0)
    assert torch.count_nonzero(captured[:, spec.head_slice(2)]) == 0


def test_patching_the_last_layer_residual_reproduces_the_source_logits():
    """The final layer's last-token residual fully determines the next token.

    Replacing it with the source run's value must therefore reproduce the source
    run's logits exactly -- the strongest available check that residual patching
    writes to the right place.
    """
    spec, model = build()
    target = random_inputs(seed=0)
    source = random_inputs(seed=1)
    source_logits = logits_of(model, source)

    store = {}
    with cache_residual_stream(model, [spec.n_layers - 1], store):
        with torch.no_grad():
            model(source)
    hidden = torch.cat(store[spec.n_layers - 1], dim=0)

    with patch_residual_stream(model, spec.n_layers - 1, hidden):
        patched = logits_of(model, target)
    assert torch.allclose(patched, source_logits, atol=1e-4)


def test_writeins_sum_to_the_attention_output():
    """Head write-ins are an exact decomposition of the attention output.

    Sum over all heads must equal o_proj applied to its input (the models used
    here have no o_proj bias).
    """
    spec, model = build()
    ids = random_inputs()
    store = {}
    with cache_o_proj_inputs(model, [0], store):
        with torch.no_grad():
            model(ids)
    captured = torch.cat(store[0], dim=0)

    total = sum(head_writein(model, spec, 0, h, captured)
                for h in range(spec.n_heads))
    expected = model.model.layers[0].self_attn.o_proj(captured.to(torch.float32))
    assert torch.allclose(total, expected, atol=1e-4)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:
                failures += 1
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failures")
    sys.exit(1 if failures else 0)
