#!/usr/bin/env python3
"""
02_extract_all_subjects.py
==========================
Extracts EEG data from all *_sequence_epochs.set files,
merges with HGF output and survey scores, saves one CSV per subject,
then concatenates into merged_eeg_hgf_ALL.csv.

Run via sbatch (see slurm/01_extract_eeg_array.sh) or directly.

Input:
    rmmn/raw_set/*_sequence_epochs.set
    rmmn/hgf_output.csv
    rmmn/Survey_scores.csv

Output:
    rmmn/per_subject/merged_S*.csv
    rmmn/merged_eeg_hgf_ALL.csv
"""
import re, glob
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio

WORK_DIR = Path.home() / 'rmmn'
files    = sorted(glob.glob(str(WORK_DIR / 'raw_set' / '*_sequence_epochs.set')))
print(f"Found {len(files)} files")

hgf    = pd.read_csv(WORK_DIR / 'hgf_output.csv')
scores = pd.read_csv(WORK_DIR / 'Survey_scores.csv')

# Find subject ID column in scores
id_col = next((c for c in scores.columns
               if c.strip().lower().replace(' ','_')
               in ('roll_no','subject','id','participant','subject_id')), None)
if id_col:
    scores = scores.rename(columns={id_col: 'subject'})
    scores['subject'] = pd.to_numeric(scores['subject'], errors='coerce')

TARGET  = ['Fz', 'Cz', 'Pz']
out_dir = WORK_DIR / 'per_subject'
out_dir.mkdir(exist_ok=True)

for i, set_file in enumerate(files):
    stem       = Path(set_file).stem.replace('_sequence_epochs','').lstrip('sS')
    nums       = re.findall(r'\d+', stem)
    subject_id = int(nums[0])
    print(f"[{i+1}/{len(files)}] Subject {subject_id}: {Path(set_file).name}")

    mat        = sio.loadmat(set_file, squeeze_me=True)
    data       = mat['data']
    times      = mat['times']
    epoch      = mat['epoch']
    chanlocs   = mat['chanlocs']

    n_chan, n_times, n_trials = data.shape
    chan_names = [str(chanlocs[j]['labels']) for j in range(n_chan)]
    chan_idx   = [chan_names.index(c) for c in TARGET]
    bad_flags  = np.array([int(epoch[t]['badEpoch']) for t in range(n_trials)])

    # Vectorised extraction (no triple loop)
    data_sub   = data[chan_idx, :, :].transpose(2, 0, 1).reshape(-1)

    eeg_df = pd.DataFrame({
        'subject'  : subject_id,
        'Trial'    : np.repeat(np.arange(1, n_trials+1), len(TARGET)*n_times),
        'channel'  : np.tile(np.repeat(TARGET, n_times), n_trials),
        'time_ms'  : np.round(np.tile(times, len(TARGET)*n_trials), 1),
        'amplitude': data_sub,
        'BadEpoch' : np.repeat(bad_flags, len(TARGET)*n_times),
    })

    # Single merge with HGF (which already contains behavioral columns)
    merged = eeg_df.merge(hgf, on='Trial', how='left')
    if id_col:
        merged = merged.merge(scores, on='subject', how='left')

    out_path = out_dir / f'merged_S{subject_id:07d}.csv'
    merged.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name}  shape={merged.shape}")

# Concatenate all subjects into one file
print("\nConcatenating all subjects...")
all_files = sorted(glob.glob(str(out_dir / 'merged_S*.csv')))
dfs       = [pd.read_csv(f) for f in all_files]
all_data  = pd.concat(dfs, ignore_index=True)
print(f"Total rows : {len(all_data):,}")
print(f"Subjects   : {all_data['subject'].nunique()}")

out = WORK_DIR / 'merged_eeg_hgf_ALL.csv'
all_data.to_csv(out, index=False)
print(f"Saved: {out}")
