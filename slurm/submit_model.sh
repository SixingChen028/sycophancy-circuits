#!/bin/bash
# Submit the full pipeline for one model, with the dependencies between stages.
#
#   bash slurm/submit_model.sh llama
#
# Stages that can run concurrently are submitted without a dependency on each
# other; those that consume another stage's output wait for it. The head sweeps
# are the long jobs (a few GPU-hours each); everything else is comparatively
# quick.

set -euo pipefail
MODEL=${1:?Usage: bash slurm/submit_model.sh <model-key>}
RUN="sbatch --parsable --job-name=${MODEL} slurm/run.sh"

# 1. Screening. Everything else depends on it.
SCREEN=$($RUN experiments/screen_mmlu.py --model "$MODEL")
echo "screen_mmlu           $SCREEN"

MT=$(sbatch --parsable --dependency=afterok:$SCREEN --job-name=${MODEL} \
     slurm/run.sh experiments/screen_multiturn.py --model "$MODEL" --mode stated)
PB=$(sbatch --parsable --dependency=afterok:$SCREEN --job-name=${MODEL} \
     slurm/run.sh experiments/screen_multiturn.py --model "$MODEL" --mode pushback)
echo "screen_multiturn      $MT (stated)  $PB (pushback)"

# 2. Depth localization and the per-head sweeps, per condition.
declare -A AFTER=( [mmlu]=$SCREEN [mmlu_label_swap]=$SCREEN \
                   [mmlu_content_swap]=$SCREEN [mmlu_multiturn]=$MT \
                   [mmlu_pushback]=$PB )
declare -A SWEEP

for CONDITION in "${!AFTER[@]}"; do
    DEP=${AFTER[$CONDITION]}
    SWEEP[$CONDITION]=$(sbatch --parsable --dependency=afterok:$DEP \
        --job-name=${MODEL} --time=10:00:00 slurm/run.sh \
        experiments/patch_heads.py --model "$MODEL" --condition "$CONDITION")
    echo "patch_heads           ${SWEEP[$CONDITION]}  $CONDITION"
done

for CONDITION in mmlu mmlu_label_swap; do
    sbatch --dependency=afterok:$SCREEN --job-name=${MODEL} slurm/run.sh \
        experiments/patch_residual_stream.py --model "$MODEL" --condition "$CONDITION"
done
sbatch --dependency=afterok:$SCREEN --job-name=${MODEL} slurm/run.sh \
    experiments/logit_lens.py --model "$MODEL" --condition mmlu

# 3. Analyses that consume the sweeps.
sbatch --dependency=afterok:${SWEEP[mmlu]} --job-name=${MODEL} slurm/run.sh \
    experiments/ablate_heads.py --model "$MODEL" --condition mmlu
sbatch --dependency=afterok:${SWEEP[mmlu]} --job-name=${MODEL} slurm/run.sh \
    experiments/patch_heads_cumulative.py --model "$MODEL" --condition mmlu
sbatch --dependency=afterok:${SWEEP[mmlu]}:${SWEEP[mmlu_label_swap]}:${SWEEP[mmlu_content_swap]} \
    --job-name=${MODEL} slurm/run.sh \
    experiments/collect_writeins.py --model "$MODEL" --condition mmlu

sbatch --job-name=${MODEL} --time=01:00:00 slurm/run.sh \
    experiments/induction_heads.py --model "$MODEL"

# 4. Cross-question patching (Figure S5), if TriviaQA screening is in place.
#    The paper's panel needs only the deceptive arm. Set CROSS_QUESTION_CONTROL=1
#    to also submit the optional no-opinion control arm; both arms MUST use the
#    same seed or the contrast between them is void.
if [ -f "runs/$MODEL/triviaqa/screened_examples.json" ]; then
    ARMS="dec"
    [ -n "${CROSS_QUESTION_CONTROL:-}" ] && ARMS="dec plain"
    for ARM in $ARMS; do
        sbatch --job-name=${MODEL} --time=06:00:00 slurm/run.sh \
            experiments/patch_cross_question.py --model "$MODEL" \
            --source_arm $ARM --seed 0
    done
    echo "patch_cross_question  submitted ($ARMS, seed 0)"
else
    echo "patch_cross_question  skipped (no TriviaQA screening yet)"
fi

echo
echo "Submitted. TriviaQA and the description condition need extra setup"
echo "(dataset download and API calls); see docs/replication.md."
