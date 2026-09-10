"""A tiny randomly-initialized Llama used to exercise the hooks on CPU.

Small enough to run in a second, and structurally identical to the real models
where it matters: the same module names, an o_proj whose input is the
concatenation of per-head outputs, and grouped-query attention.
"""

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from syco.models import ModelSpec

N_LAYERS = 3
N_HEADS = 4
D_HEAD = 8
VOCAB = 64


def build():
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=N_HEADS * D_HEAD,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=2,          # grouped-query attention, as in the real models
        head_dim=D_HEAD,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).eval()
    spec = ModelSpec(
        key="tiny", hf_id="tiny", answer_ids=(10, 11, 12, 13),
        n_layers=N_LAYERS, n_heads=N_HEADS, d_head=D_HEAD, critical_layer=1,
    )
    return spec, model


def random_inputs(batch=3, length=7, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (batch, length), generator=generator)
