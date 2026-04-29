#!/bin/bash
#SBATCH --job-name=lmer_analysis
#SBATCH --qos=pool_kotesrj_ra
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/03_analysis_%a_%j.log

# =============================================================================
# slurm/03_submit_analysis.sh
# =============================================================================
# Runs 03_lm_vs_lmer_analysis.R for Fz, Cz, Pz as a SLURM array.
# Submit after slurm/02_concat_subjects.sh completes.
#
# Usage:
#   export WORK_DIR=$HOME/rmmn
#   export N_PERM=500
#   sbatch slurm/03_submit_analysis.sh
# =============================================================================

CHANNELS=(Fz Cz Pz)
export EEG_CHANNEL="${CHANNELS[$SLURM_ARRAY_TASK_ID]}"
export WORK_DIR="${WORK_DIR:-$HOME/rmmn}"
export N_PERM="${N_PERM:-500}"

echo "Channel: $EEG_CHANNEL | N_PERM: $N_PERM"
Rscript "${WORK_DIR}/R/03_lm_vs_lmer_analysis.R"
