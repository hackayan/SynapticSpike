#!/bin/bash
#SBATCH --job-name=olife
#SBATCH --qos=pool_kotesrj_ra
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/04_olife_%j.log

# =============================================================================
# slurm/04_olife.sh
# =============================================================================
# Runs the O-LIFE correlation analysis.
# Submit after slurm/03_submit_analysis.sh completes.
# =============================================================================

export WORK_DIR="${WORK_DIR:-$HOME/rmmn}"
Rscript "${WORK_DIR}/R/04_olife_correlation.R"
