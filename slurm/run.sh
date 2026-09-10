#!/bin/bash
#SBATCH --job-name=syco
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#
# One runner for every experiment.
#
#   sbatch slurm/run.sh experiments/patch_heads.py --model llama --condition mmlu
#
# SBATCH directives cannot use shell substitution, so the log paths are set
# below instead. Note that SLURM copies the submitted script to a staging path
# before running it, so ${BASH_SOURCE[0]} does not point at the real script;
# $SLURM_SUBMIT_DIR, which SLURM sets to wherever sbatch was invoked, is the
# reliable way to find the repository. Submit from the repository root.

set -euo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p "$PROJECT/logs"
JOB="${SLURM_JOB_ID:-local}"
exec > "$PROJECT/logs/${SLURM_JOB_NAME:-syco}_${JOB}.out" \
     2> "$PROJECT/logs/${SLURM_JOB_NAME:-syco}_${JOB}.err"

echo "Job:  ${JOB} on ${SLURMD_NODENAME:-$(hostname)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source "$PROJECT/slurm/env.sh"
cd "$PROJECT"

SCRIPT=${1:?Usage: sbatch slurm/run.sh <script.py> [args...]}
shift
echo "Running: $SCRIPT $*"
python -u "$SCRIPT" "$@"
echo "Done."
