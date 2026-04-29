#!/bin/bash
#SBATCH --job-name=eeg_extract
#SBATCH --qos=pool_kotesrj_ra
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-999%20
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/01_extract_%a_%j.log

# =============================================================================
# slurm/01_extract_eeg_array.sh
# =============================================================================
# SLURM array job: extracts EEG from one .set file per array task.
# Submit after python/01_prep_hgf_output.py has been run.
#
# Usage:
#   sbatch --array=0-33 slurm/01_extract_eeg_array.sh
# =============================================================================

set -euo pipefail
WORK_DIR="${WORK_DIR:-$HOME/rmmn}"

mapfile -t SET_FILES < <(ls "${WORK_DIR}/raw_set/"*_sequence_epochs.set 2>/dev/null | sort)
N_FILES=${#SET_FILES[@]}

if [[ $SLURM_ARRAY_TASK_ID -ge $N_FILES ]]; then
    echo "Array index $SLURM_ARRAY_TASK_ID >= $N_FILES files. Skipping."
    exit 0
fi

SET_FILE="${SET_FILES[$SLURM_ARRAY_TASK_ID]}"
echo "Processing: $SET_FILE"

python3 - "$SET_FILE" "$WORK_DIR" << 'PYEOF'
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio

set_file = sys.argv[1]
work_dir = Path(sys.argv[2])

stem       = Path(set_file).stem.replace('_sequence_epochs','').lstrip('sS')
nums       = re.findall(r'\d+', stem)
subject_id = int(nums[0])
print(f"Subject ID: {subject_id}")

mat        = sio.loadmat(set_file, squeeze_me=True)
data       = mat['data']
times      = mat['times']
epoch      = mat['epoch']
chanlocs   = mat['chanlocs']

n_chan, n_times, n_trials = data.shape
chan_names = [str(chanlocs[i]['labels']) for i in range(n_chan)]
TARGET     = ['Fz', 'Cz', 'Pz']
chan_idx   = [chan_names.index(c) for c in TARGET]
bad_flags  = np.array([int(epoch[t]['badEpoch']) for t in range(n_trials)])
print(f"  bad epochs: {bad_flags.sum()}/{n_trials}")

data_sub   = data[chan_idx, :, :].transpose(2, 0, 1).reshape(-1)

eeg_df = pd.DataFrame({
    'subject'  : subject_id,
    'Trial'    : np.repeat(np.arange(1, n_trials+1), len(TARGET)*n_times),
    'channel'  : np.tile(np.repeat(TARGET, n_times), n_trials),
    'time_ms'  : np.round(np.tile(times, len(TARGET)*n_trials), 1),
    'amplitude': data_sub,
    'BadEpoch' : np.repeat(bad_flags, len(TARGET)*n_times),
})

hgf    = pd.read_csv(work_dir / 'hgf_output.csv')
merged = eeg_df.merge(hgf, on='Trial', how='left')

scores_path = work_dir / 'Survey_scores.csv'
if scores_path.exists():
    scores = pd.read_csv(scores_path)
    id_col = next((c for c in scores.columns
                   if c.strip().lower().replace(' ','_')
                   in ('roll_no','subject','id','participant','subject_id')), None)
    if id_col:
        scores = scores.rename(columns={id_col: 'subject'})
        scores['subject'] = pd.to_numeric(scores['subject'], errors='coerce')
        merged = merged.merge(scores, on='subject', how='left')

out_dir  = work_dir / 'per_subject'
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f'merged_S{subject_id:07d}.csv'
merged.to_csv(out_path, index=False)
print(f"  Saved: {out_path}  shape={merged.shape}")
PYEOF

echo "Done: task $SLURM_ARRAY_TASK_ID"
