#!/bin/bash
#SBATCH --job-name=concat
#SBATCH --qos=pool_kotesrj_ra
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/02_concat_%j.log

# =============================================================================
# slurm/02_concat_subjects.sh
# =============================================================================
# Concatenates all per_subject/merged_S*.csv files into one.
# Submit after slurm/01_extract_eeg_array.sh completes.
# =============================================================================

WORK_DIR="${WORK_DIR:-$HOME/rmmn}"

python3 - "$WORK_DIR" << 'PYEOF'
import sys, glob
from pathlib import Path
import pandas as pd

work_dir = Path(sys.argv[1])
files    = sorted(glob.glob(str(work_dir / 'per_subject' / 'merged_S*.csv')))

if not files:
    raise FileNotFoundError("No per-subject CSVs found")

print(f"Concatenating {len(files)} files...")
dfs      = [pd.read_csv(f) for f in files]
all_data = pd.concat(dfs, ignore_index=True)

print(f"Total rows : {len(all_data):,}")
print(f"Subjects   : {all_data['subject'].nunique()}")
print(f"Columns    : {all_data.columns.tolist()}")

out = work_dir / 'merged_eeg_hgf_ALL.csv'
all_data.to_csv(out, index=False)
print(f"Saved: {out}")
PYEOF
