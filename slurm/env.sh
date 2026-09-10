#!/bin/bash
# Shared cluster environment for every job.
#
# Sourced by slurm/run.sh. Adapt the three site-specific settings below.
#
# The HuggingFace token is read BEFORE HF_HOME is redirected, because the token
# lives in the default cache location and redirecting first would hide it.

CONDA_ENV="${CONDA_ENV:-syco}"
HF_CACHE="${HF_CACHE:-$SCRATCH/hf_cache}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

HF_TOKEN=$(python -c "import huggingface_hub as h; t=h.get_token(); print(t) if t else exit(1)" 2>/dev/null)
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: no HuggingFace token found. Run 'huggingface-cli login' on the login node."
    echo "Llama and Gemma are gated repositories and cannot be downloaded without one."
    exit 1
fi
export HF_TOKEN

export HF_HOME="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false
