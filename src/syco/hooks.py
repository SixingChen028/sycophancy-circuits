"""Activation caching and intervention hooks.

Where the interventions attach
------------------------------
An attention layer computes

    attn_out = concat(head_0, ..., head_{H-1}) @ W_O

so the input to `o_proj` is the concatenation of every head's output, before
they are summed by the projection. Slicing that tensor gives one head's
contribution in isolation, which is exactly the OV-circuit write-in the paper
patches and ablates. Everything here therefore hooks `o_proj`'s *input*
(a forward pre-hook), except residual-stream patching, which hooks the
decoder layer's output.

Interventions are applied at the last token position only, which is the
position whose residual stream determines the next token.

Gemma-2 note: the o_proj input is wider than hidden_size there (see
`syco.models`). Because these hooks slice whatever tensor actually arrives,
they need no special case -- only `spec.d_head` has to be right.
"""

from contextlib import contextmanager

import torch

__all__ = [
    "layer_modules",
    "cache_o_proj_inputs",
    "cache_residual_stream",
    "patch_head_outputs",
    "patch_residual_stream",
    "ablate_heads",
    "head_writein",
]


def layer_modules(model):
    """The decoder layers, for the architectures used in the paper.

    Llama, Mistral and Gemma-2 all expose `model.model.layers`.
    """
    return model.model.layers


@contextmanager
def cache_o_proj_inputs(model, layers, store):
    """Record the last-token o_proj input for each layer of `layers`.

    `store` is a dict layer -> list; one float32 CPU tensor of shape
    (batch, n_heads * d_head) is appended per forward pass.
    """
    handles = []
    for layer in layers:
        def make_hook(idx):
            def hook(module, args):
                store.setdefault(idx, []).append(
                    args[0][:, -1, :].detach().float().cpu()
                )
            return hook

        handles.append(
            layer_modules(model)[layer].self_attn.o_proj.register_forward_pre_hook(
                make_hook(layer)
            )
        )
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def cache_residual_stream(model, layers, store):
    """Record the last-token residual stream after each layer of `layers`."""
    handles = []
    for layer in layers:
        def make_hook(idx):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                store.setdefault(idx, []).append(
                    hidden[:, -1, :].detach().float().cpu()
                )
            return hook

        handles.append(
            layer_modules(model)[layer].register_forward_hook(make_hook(layer))
        )
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def patch_head_outputs(model, spec, heads, source_values):
    """Replace the given heads' last-token write-ins with cached source values.

    heads          iterable of (layer, head)
    source_values  dict layer -> (batch, n_heads * d_head) tensor from the
                   source run, already on the right device
    """
    by_layer = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)

    handles = []
    for layer, layer_heads in by_layer.items():
        def make_hook(idx, hs):
            def hook(module, args):
                x = args[0].clone()
                src = source_values[idx]
                for h in hs:
                    sl = spec.head_slice(h)
                    x[:, -1, sl] = src[:, sl].to(dtype=x.dtype, device=x.device)
                return (x,)
            return hook

        handles.append(
            layer_modules(model)[layer].self_attn.o_proj.register_forward_pre_hook(
                make_hook(layer, layer_heads)
            )
        )
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def patch_residual_stream(model, layer, source_hidden):
    """Replace the last-token residual stream at `layer` with the source run's."""

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        hidden = hidden.clone()
        hidden[:, -1, :] = source_hidden.to(dtype=hidden.dtype, device=hidden.device)
        return (hidden,) + output[1:] if is_tuple else hidden

    handle = layer_modules(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def ablate_heads(model, spec, heads, last_token_only=False):
    """Zero the given heads' contribution to the residual stream.

    `last_token_only` zeroes the head only at the final position; otherwise it
    is zeroed at every position, which is the setting reported in the paper
    (a head's effect on the answer can route through earlier positions).
    """
    by_layer = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)

    handles = []
    for layer, layer_heads in by_layer.items():
        def make_hook(hs):
            def hook(module, args):
                x = args[0].clone()
                for h in hs:
                    sl = spec.head_slice(h)
                    if last_token_only:
                        x[:, -1, sl] = 0.0
                    else:
                        x[:, :, sl] = 0.0
                return (x,)
            return hook

        handles.append(
            layer_modules(model)[layer].self_attn.o_proj.register_forward_pre_hook(
                make_hook(layer_heads)
            )
        )
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def head_writein(model, spec, layer, head, o_proj_input):
    """The vector a head adds to the residual stream, via the OV circuit.

    o_proj_input is (batch, n_heads * d_head) at one position; the result is
    (batch, hidden_size). This is the quantity the answer probes are fit on
    (paper, Appendix B.2) and the same quantity the head patching intervenes on.
    """
    sl = spec.head_slice(head)
    weight = layer_modules(model)[layer].self_attn.o_proj.weight[:, sl]
    return o_proj_input[:, sl].to(weight.device).float() @ weight.float().T
