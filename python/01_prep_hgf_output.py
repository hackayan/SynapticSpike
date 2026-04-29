#!/usr/bin/env python3
"""
01_prep_hgf_output.py
=====================
Prepares the pyhgf output CSV for downstream analysis.
Renames columns, adds cat_code, drops Response_Time.

Run ONCE on HPC before the pipeline:
    python3 01_prep_hgf_output.py

Input:
    rmmn/trialwise_predictions.csv  -- pyhgf output
    rmmn/behavioral_data_trialwise.csv

Output:
    rmmn/hgf_output.csv
"""
from pathlib import Path
import pandas as pd

WORK_DIR = Path.home() / 'rmmn'

print("Loading files...")
hgf = pd.read_csv(WORK_DIR / 'trialwise_predictions.csv')
beh = pd.read_csv(WORK_DIR / 'behavioral_data_trialwise.csv')
print(f"  hgf shape: {hgf.shape}")
print(f"  beh shape: {beh.shape}")

# Rename columns to match R analysis script
hgf = hgf.rename(columns={
    'epsilon_2'               : 'eps2',
    'epsilon_3'               : 'eps3',
    'abs_epsilon_2'           : 'abs_eps2',
    'abs_epsilon_3'           : 'abs_eps3',
    'hgf_predicted_p_deviant' : 'mu2',
    'trial_type_binary'       : 'u',
})

# Add cat_code (only column missing from hgf)
hgf = hgf.merge(beh[['Trial', 'cat_code']], on='Trial', how='left')

# Drop Response_Time (all NaN in passive paradigm)
if 'Response_Time' in hgf.columns:
    hgf = hgf.drop(columns=['Response_Time'])

# Verify
assert len(hgf) == 1392,                              "Expected 1392 trials"
assert list(hgf['Trial']) == list(range(1, 1393)),    "Trial numbers must be 1-1392"
assert hgf['eps2'].isna().sum() == 0,                 "eps2 has NaNs"
assert hgf['eps3'].isna().sum() == 0,                 "eps3 has NaNs"
assert hgf['trial_type'].isna().sum() == 0,           "trial_type has NaNs"
assert hgf['u'].isna().sum() == 0,                    "u has NaNs"
assert hgf['cat_code'].isna().sum() == 0,             "cat_code has NaNs"

print(f"\nAll checks passed")
print(f"  Final shape   : {hgf.shape}")
print(f"  eps2 deviant  : {hgf[hgf['trial_type']=='deviant']['eps2'].mean():.4f}  (should be ~0.9)")
print(f"  eps2 standard : {hgf[hgf['trial_type']=='standard']['eps2'].mean():.4f} (should be ~-0.1)")

out = WORK_DIR / 'hgf_output.csv'
hgf.to_csv(out, index=False)
print(f"\nSaved: {out}")
