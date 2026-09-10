"""Model registry.

Every model-specific fact that the analyses need lives here. The original
code base carried one near-identical copy of each script per model, differing
only in the constants below; this registry is what lets a single script run
against any of them.

Fields
------
hf_id           HuggingFace repo id.
answer_ids      Token ids of bare "A", "B", "C", "D" -- the tokens the model
                emits right after the "(" of "I believe the answer is (".
                These are NOT portable across tokenizers (Llama and Gemma
                disagree completely), so each was verified directly against
                the model's own tokenizer; `verify_answer_ids` re-checks them.
n_layers        Number of transformer blocks.
n_heads         config.num_attention_heads (query heads; KV heads may be fewer
                under GQA, which does not affect o_proj-input slicing).
d_head          Width of one head's slice of the o_proj input. Usually
                hidden_size // n_heads, but see the Gemma note below.
critical_layer  Last layer at which the median plain-to-label-swap normalized
                logit difference remains below 0.1 (paper, Appendix B.1). Heads
                at layers <= critical_layer are "early" (opinion) candidates;
                heads after it are "late" (answer retrieval) candidates.
chat            False for base checkpoints with no chat template.

Gemma-2's head geometry
-----------------------
Gemma-2-9B-it sets head_dim=256 EXPLICITLY in its config; it is not
hidden_size // n_heads (3584 // 16 = 224). The attention output is therefore
16 * 256 = 4096 wide, larger than hidden_size (3584): Gemma-2 widens the
attention output before projecting back down. Hooking the o_proj input still
works unchanged (the hook sees whatever tensor actually feeds o_proj), but
d_head must come from config.head_dim. Deriving it as hidden_size // n_heads
would silently misalign every head's slice.
"""

from dataclasses import dataclass

__all__ = ["ModelSpec", "MODELS", "get_model_spec", "verify_answer_ids", "verify_geometry"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    answer_ids: tuple
    n_layers: int
    n_heads: int
    d_head: int
    critical_layer: int
    chat: bool = True

    @property
    def label(self) -> str:
        return _LABELS.get(self.key, self.key)

    def head_slice(self, head: int) -> slice:
        """The slice of the o_proj input belonging to `head`."""
        return slice(head * self.d_head, (head + 1) * self.d_head)


_LABELS = {
    "llama": "Llama-3.1-8B-Instruct",
    "llama-base": "Llama-3.1-8B (base)",
    "mistral": "Mistral-7B-Instruct-v0.3",
    "gemma": "Gemma-2-9B-it",
}


MODELS = {
    "llama": ModelSpec(
        key="llama",
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        answer_ids=(32, 33, 34, 35),
        n_layers=32,
        n_heads=32,
        d_head=128,
        critical_layer=16,
    ),
    # Base checkpoint of the above. Same tokenizer and same architecture, so
    # layer/head indices are directly comparable (paper, Appendix D.4).
    "llama-base": ModelSpec(
        key="llama-base",
        hf_id="meta-llama/Llama-3.1-8B",
        answer_ids=(32, 33, 34, 35),
        n_layers=32,
        n_heads=32,
        d_head=128,
        critical_layer=16,
        chat=False,
    ),
    "mistral": ModelSpec(
        key="mistral",
        hf_id="mistralai/Mistral-7B-Instruct-v0.3",
        answer_ids=(1098, 1133, 1102, 1152),
        n_layers=32,
        n_heads=32,
        d_head=128,
        critical_layer=17,
    ),
    "gemma": ModelSpec(
        key="gemma",
        hf_id="google/gemma-2-9b-it",
        answer_ids=(235280, 235305, 235288, 235299),
        n_layers=42,
        n_heads=16,
        d_head=256,  # explicit config.head_dim, NOT hidden_size // n_heads
        critical_layer=27,
    ),
}


def get_model_spec(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise SystemExit(
            f"Unknown model {key!r}. Available: {', '.join(sorted(MODELS))}"
        )


def verify_answer_ids(spec: ModelSpec, tok) -> None:
    """Re-derive the answer token ids from the tokenizer and check the registry.

    The ids must be those of a bare capital letter with no leading space, since
    they follow "(" in the prompt. Raises if the registry disagrees.
    """
    derived = tuple(
        tok(letter, add_special_tokens=False)["input_ids"][-1] for letter in "ABCD"
    )
    if derived != tuple(spec.answer_ids):
        raise RuntimeError(
            f"{spec.key}: answer_ids in registry {tuple(spec.answer_ids)} disagree "
            f"with tokenizer-derived {derived}. Check the tokenizer version."
        )


def verify_geometry(spec: ModelSpec, model) -> None:
    """Check the registry against the loaded model's config.

    Guards the failure mode that motivated the Gemma note above: a wrong d_head
    misaligns every head slice, which produces plausible-looking but meaningless
    numbers rather than an error.
    """
    cfg = model.config
    if cfg.num_hidden_layers != spec.n_layers:
        raise RuntimeError(
            f"{spec.key}: n_layers {spec.n_layers} != config {cfg.num_hidden_layers}"
        )
    if cfg.num_attention_heads != spec.n_heads:
        raise RuntimeError(
            f"{spec.key}: n_heads {spec.n_heads} != config {cfg.num_attention_heads}"
        )
    d_head = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    if d_head != spec.d_head:
        raise RuntimeError(
            f"{spec.key}: d_head {spec.d_head} != config-derived {d_head}"
        )
