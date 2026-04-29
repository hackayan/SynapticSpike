# Detailed Code Breakdown

This document provides a line-by-line and function-by-function explanation of every code file in the repository.

---

## 1. Python Scripts

### `python/01_prep_hgf_output.py`

This script prepares the raw output from the HGF modeling (`pyhgf`) for use in downstream R analysis.

```python
from pathlib import Path
import pandas as pd
```
- **Line 1-2**: Imports required libraries. `Path` is used for handling file paths across different operating systems. `pandas` is used for data manipulation and analysis.

```python
WORK_DIR = Path.home() / 'rmmn'
```
- **Line 4**: Defines the working directory as `~/rmmn` (the `rmmn` folder in the user's home directory). This is where all data files are expected to be.

```python
print("Loading files...")
hgf = pd.read_csv(WORK_DIR / 'trialwise_predictions.csv')
beh = pd.read_csv(WORK_DIR / 'behavioral_data_trialwise.csv')
print(f"  hgf shape: {hgf.shape}")
print(f"  beh shape: {beh.shape}")
```
- **Lines 6-11**: Prints a status message, then reads the HGF predictions (`trialwise_predictions.csv`) and the behavioral trial sequence (`behavioral_data_trialwise.csv`) into Pandas DataFrames named `hgf` and `beh`. It prints their dimensions (rows, columns).

```python
# Rename columns to match R analysis script
hgf = hgf.rename(columns={
    'epsilon_2'               : 'eps2',
    'epsilon_3'               : 'eps3',
    'abs_epsilon_2'           : 'abs_eps2',
    'abs_epsilon_3'           : 'abs_eps3',
    'hgf_predicted_p_deviant' : 'mu2',
    'trial_type_binary'       : 'u',
})
```
- **Lines 13-22**: Renames specific columns in the `hgf` DataFrame. For example, `epsilon_2` (sensory prediction error) becomes `eps2`. This is done because the R scripts downstream are explicitly coded to look for the shortened names. `u` represents the input to the HGF model.

```python
# Add cat_code (only column missing from hgf)
hgf = hgf.merge(beh[['Trial', 'cat_code']], on='Trial', how='left')
```
- **Lines 24-25**: Merges a specific column (`cat_code`, which represents categorical conditions) from the behavioral data `beh` into the `hgf` data, matching rows based on the `Trial` number. `how='left'` ensures all rows from `hgf` are kept.

```python
# Drop Response_Time (all NaN in passive paradigm)
if 'Response_Time' in hgf.columns:
    hgf = hgf.drop(columns=['Response_Time'])
```
- **Lines 27-29**: Checks if a `Response_Time` column exists. If it does, it deletes it because this is a passive listening paradigm where participants don't make responses, so this column would just be full of missing values (`NaN`).

```python
# Verify
assert len(hgf) == 1392,                              "Expected 1392 trials"
assert list(hgf['Trial']) == list(range(1, 1393)),    "Trial numbers must be 1-1392"
assert hgf['eps2'].isna().sum() == 0,                 "eps2 has NaNs"
assert hgf['eps3'].isna().sum() == 0,                 "eps3 has NaNs"
assert hgf['trial_type'].isna().sum() == 0,           "trial_type has NaNs"
assert hgf['u'].isna().sum() == 0,                    "u has NaNs"
assert hgf['cat_code'].isna().sum() == 0,             "cat_code has NaNs"
```
- **Lines 31-38**: Performs sanity checks to ensure the data is perfect before saving. It halts the script with an error if there aren't exactly 1392 trials, if trial numbers aren't sequential, or if any critical columns contain missing data (`NaN`).

```python
print(f"\nAll checks passed")
print(f"  Final shape   : {hgf.shape}")
print(f"  eps2 deviant  : {hgf[hgf['trial_type']=='deviant']['eps2'].mean():.4f}  (should be ~0.9)")
print(f"  eps2 standard : {hgf[hgf['trial_type']=='standard']['eps2'].mean():.4f} (should be ~-0.1)")
```
- **Lines 40-44**: Prints final confirmation messages, the final shape of the DataFrame, and the average `eps2` (prediction error) for deviant vs. standard trials to quickly visually verify that deviants produce higher prediction errors.

```python
out = WORK_DIR / 'hgf_output.csv'
hgf.to_csv(out, index=False)
print(f"\nSaved: {out}")
```
- **Lines 46-48**: Saves the modified, cleaned DataFrame to a new file named `hgf_output.csv` without saving the pandas row index, and prints the save location.

---

### `python/02_extract_all_subjects.py`

This script pulls single-trial EEG amplitude data from raw files and pairs it with the HGF predictions.

```python
import re, glob
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
```
- **Lines 1-5**: Imports libraries: `re` for regular expressions (parsing text), `glob` for finding files matching a pattern, `Path` for paths, `numpy` for fast numerical arrays, `pandas` for dataframes, and `scipy.io` to read MATLAB `.set` files.

```python
WORK_DIR = Path.home() / 'rmmn'
files    = sorted(glob.glob(str(WORK_DIR / 'raw_set' / '*_sequence_epochs.set')))
print(f"Found {len(files)} files")
```
- **Lines 7-9**: Sets the working directory. Finds all files in the `raw_set` folder that end in `_sequence_epochs.set`, sorts them alphabetically, and prints how many were found.

```python
hgf    = pd.read_csv(WORK_DIR / 'hgf_output.csv')
scores = pd.read_csv(WORK_DIR / 'Survey_scores.csv')
```
- **Lines 11-12**: Loads the cleaned HGF data and the clinical survey scores.

```python
# Find subject ID column in scores
id_col = next((c for c in scores.columns
               if c.strip().lower().replace(' ','_')
               in ('roll_no','subject','id','participant','subject_id')), None)
if id_col:
    scores = scores.rename(columns={id_col: 'subject'})
    scores['subject'] = pd.to_numeric(scores['subject'], errors='coerce')
```
- **Lines 14-20**: This is a robust way to find whichever column in the `scores` file identifies the participant. It converts column names to lowercase, replaces spaces with underscores, and checks if it matches common ID names. If found, it renames it standardly to `subject` and ensures the values are numbers.

```python
TARGET  = ['Fz', 'Cz', 'Pz']
out_dir = WORK_DIR / 'per_subject'
out_dir.mkdir(exist_ok=True)
```
- **Lines 22-24**: Defines the specific EEG electrodes to extract (`TARGET`). Creates a `per_subject` folder for the output, ignoring if it already exists.

```python
for i, set_file in enumerate(files):
```
- **Line 26**: Starts a loop through every EEG file found.

```python
    stem       = Path(set_file).stem.replace('_sequence_epochs','').lstrip('sS')
    nums       = re.findall(r'\d+', stem)
    subject_id = int(nums[0])
    print(f"[{i+1}/{len(files)}] Subject {subject_id}: {Path(set_file).name}")
```
- **Lines 27-30**: Extracts the subject ID from the filename. It removes standard text (e.g., `_sequence_epochs`), strips leading 's' or 'S', uses regex to find the first sequence of numbers, converts it to an integer, and prints it.

```python
    mat        = sio.loadmat(set_file, squeeze_me=True)
    data       = mat['data']
    times      = mat['times']
    epoch      = mat['epoch']
    chanlocs   = mat['chanlocs']
```
- **Lines 32-36**: Uses `scipy` to load the MATLAB formatted EEGLAB `.set` file. It extracts the raw voltage matrix (`data`), the time points (`times`), metadata about each trial (`epoch`), and metadata about channels (`chanlocs`).

```python
    n_chan, n_times, n_trials = data.shape
    chan_names = [str(chanlocs[j]['labels']) for j in range(n_chan)]
    chan_idx   = [chan_names.index(c) for c in TARGET]
    bad_flags  = np.array([int(epoch[t]['badEpoch']) for t in range(n_trials)])
```
- **Lines 38-41**: Gets the dimensions of the 3D data matrix (channels x time x trials). It extracts a list of all channel names, finds the numerical indices of the target channels (`Fz`, `Cz`, `Pz`), and creates a 1D array of 0s and 1s indicating if a trial was marked as "bad" during preprocessing.

```python
    # Vectorised extraction (no triple loop)
    data_sub   = data[chan_idx, :, :].transpose(2, 0, 1).reshape(-1)
```
- **Line 44**: Very fast, vectorized way to flatten the 3D matrix. It selects only target channels, rearranges dimensions to (trials, channels, time), and flattens it into a 1D list of voltage values.

```python
    eeg_df = pd.DataFrame({
        'subject'  : subject_id,
        'Trial'    : np.repeat(np.arange(1, n_trials+1), len(TARGET)*n_times),
        'channel'  : np.tile(np.repeat(TARGET, n_times), n_trials),
        'time_ms'  : np.round(np.tile(times, len(TARGET)*n_trials), 1),
        'amplitude': data_sub,
        'BadEpoch' : np.repeat(bad_flags, len(TARGET)*n_times),
    })
```
- **Lines 46-53**: Creates a long-format Pandas DataFrame. Because we flattened the data matrix above, we have to create matching columns for Subject, Trial, Channel, Time, and BadEpoch status. `np.repeat` and `np.tile` are used to repeat values correctly so every voltage row matches its correct metadata.

```python
    # Single merge with HGF (which already contains behavioral columns)
    merged = eeg_df.merge(hgf, on='Trial', how='left')
    if id_col:
        merged = merged.merge(scores, on='subject', how='left')
```
- **Lines 55-58**: Joins the long-format EEG data with the HGF prediction data, matching by Trial number. Then, if clinical scores were found, joins them by Subject ID.

```python
    out_path = out_dir / f'merged_S{subject_id:07d}.csv'
    merged.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name}  shape={merged.shape}")
```
- **Lines 60-62**: Saves this massive, single-subject dataframe as a CSV file in the `per_subject` folder.

```python
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
```
- **Lines 64-73**: After the loop finishes, this block finds all individual subject CSVs, reads them into a list, and uses `pd.concat` to vertically stack them into one gigantic master dataframe containing every subject, trial, channel, and timepoint. It saves this as `merged_eeg_hgf_ALL.csv`.

---

## 2. R Scripts

### `R/03_lm_vs_lmer_analysis.R`

This script runs statistical regressions at every single timepoint to see when prediction errors (`eps2`, `eps3`) correlate with EEG amplitude.

```R
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(lme4)
})
```
- **Lines 1-6**: Silently loads required R packages. `dplyr` and `readr` for data manipulation, `ggplot2` for plotting, and `lme4` for linear mixed-effects models.

```R
WORK_DIR <- Sys.getenv("WORK_DIR", unset = file.path(Sys.getenv("HOME"), "rmmn"))
N_PERM   <- as.integer(Sys.getenv("N_PERM",    unset = "500"))
CHANNEL  <- Sys.getenv("EEG_CHANNEL",           unset = "Fz")
OUT_DIR  <- file.path(WORK_DIR, "results", CHANNEL)
dir.create(file.path(OUT_DIR, "plots"), recursive=TRUE, showWarnings=FALSE)
```
- **Lines 8-12**: Sets configuration variables based on Environment Variables (which are passed from the SLURM bash scripts). If they don't exist, it uses defaults (e.g., 500 permutations, Fz channel). Creates output directories for the results.

#### `run_lm_timewise` function
```R
# ── Helper: run LM at every timepoint ────────────────────────────────────────
run_lm_timewise <- function(df, formula, effect_name) {
  df %>% group_by(time_ms) %>%
    do({
      fit <- tryCatch(lm(formula, data=.), error=function(e) NULL)
      if (is.null(fit)) return(data.frame(t_value=0, p_value=1))
      coef <- summary(fit)$coefficients
      if (effect_name %in% rownames(coef))
        data.frame(t_value=coef[effect_name,"t value"],
                   p_value=coef[effect_name,"Pr(>|t|)"])
      else data.frame(t_value=0, p_value=1)
    }) %>% ungroup()
}
```
- **Purpose**: Runs a standard Linear Regression model at every distinct time point (e.g., -100ms, -98ms, etc.).
- **Details**: It groups the large dataframe by `time_ms`. For each group (time point), it fits a standard `lm()`. `tryCatch` ensures if a model fails to fit, it doesn't crash the script but returns 0s. It extracts the t-value and p-value specifically for the predictor variable we care about (`effect_name`, e.g., "eps2") and returns them.

#### `run_lmer_timewise` function
```R
# ── Helper: run LMER at every timepoint ──────────────────────────────────────
run_lmer_timewise <- function(df, formula, effect_name) {
  df %>% group_by(time_ms) %>%
    do({
      fit <- tryCatch(
        lmer(formula, data=., REML=FALSE,
             control=lmerControl(optimizer="bobyqa",
                                 check.conv.singular=.makeCC("ignore",tol=1e-4))),
        error=function(e) NULL)
...
```
- **Purpose**: Runs a Linear Mixed-Effects Model at every timepoint.
- **Details**: Same logic as the LM function, but uses `lmer()`. It uses `REML=FALSE` (Maximum Likelihood) which is standard when comparing fixed effects. It uses the `bobyqa` optimizer to speed up model convergence and ignores "singular fit" warnings (which happen when random effect variance is near zero) so the loop doesn't crash. It calculates p-values for t-values assuming an infinite degrees-of-freedom normal distribution (`pnorm`), standard for large LMER datasets.

#### `get_clusters` function
```R
# ── Helper: find significant clusters ────────────────────────────────────────
get_clusters <- function(res_df, t_thresh=2.0) {
  res_df <- res_df %>% arrange(time_ms) %>% mutate(sig=abs(t_value)>t_thresh)
  clusters <- list(); current <- c()
  for (i in seq_len(nrow(res_df))) {
...
```
- **Purpose**: Finds consecutive time points where the statistical effect is strong (t > 2.0).
- **Details**: It identifies time points crossing the threshold. The `for` loop groups adjacent significant time points into "clusters". It calculates a `cluster_stat` for each cluster by summing the absolute t-values inside that cluster (the "cluster mass").

#### `attach_p` function
```R
# ── Helper: attach permutation p-values ──────────────────────────────────────
attach_p <- function(cl, null) {
  if (!is.null(cl) && nrow(cl)>0)
    cl$p_perm <- sapply(cl$cluster_stat, function(cs) mean(null >= cs))
  cl
}
```
- **Purpose**: Calculates final p-values for clusters.
- **Details**: Compares a real cluster's mass (`cs`) against a distribution of maximum cluster masses generated from shuffled data (`null`). The p-value is the proportion of times a shuffled dataset produced a cluster larger than the real one.

#### `plot_lm_vs_lmer` function
```R
# ── Helper: plot LM vs LMER overlay ──────────────────────────────────────────
plot_lm_vs_lmer <- function(lm_res, lmer_res, title) {
...
```
- **Purpose**: Generates the final time-series graph.
- **Details**: Uses `ggplot2` to plot t-values over time. It draws two lines (LM and LMER) to show how incorporating subject-level random effects changes the result. It adds shaded boxes highlighting standard MMN and P300 time windows, and draws dotted threshold lines at t=+2 and t=-2.

#### Data Loading and Cleaning Block
```R
cat("[1/5] Loading data...\n")
all_data <- read_csv(
  file.path(WORK_DIR, "merged_eeg_hgf_ALL.csv"), ...

# Remove subjects with >30% bad epochs
bad_pct   <- all_data %>%
  filter(channel==CHANNEL, time_ms==first(time_ms)) %>%
  group_by(subject) %>%
  summarise(bad_pct=mean(BadEpoch==1)*100, .groups="drop")
good_subs <- bad_pct %>% filter(bad_pct <= 30) %>% pull(subject)
...
df <- all_data %>%
  filter(subject %in% good_subs, BadEpoch==0, channel==CHANNEL) %>%
  mutate(Condition=factor(cat_code), u=as.integer(trial_type=="deviant"))
```
- **Details**: Loads the massive master CSV. It calculates what percentage of trials each subject had marked as "bad". It drops subjects who have more than 30% bad data. It filters the dataset to keep only good trials for the specific target channel, and converts trial types to factors and binary 1/0 integers.

```R
# Orthogonalise eps3 w.r.t. eps2 and z-score
eps3_orth      <- resid(lm(eps3 ~ eps2, data=df))
df$eps2_z      <- as.numeric(scale(df$eps2))
df$eps3_orth_z <- as.numeric(scale(eps3_orth))
```
- **Details**: Very important statistical step. `eps2` (sensory error) and `eps3` (volatility error) are highly correlated. By predicting `eps3` using `eps2` and taking the residuals (`resid`), we isolate the pure "volatility" signal that *isn't* already explained by `eps2`. It then z-scores (standardizes) both variables so their coefficients are comparable.

```R
lm_f   <- amplitude ~ u + Condition + eps2_z + eps3_orth_z
lmer_f <- amplitude ~ u + Condition + eps2_z + eps3_orth_z + (1|subject)
```
- **Details**: Defines the formulas for regression. Amplitude is predicted by trial type (`u`), condition, and prediction errors. `(1|subject)` in the LMER formula allows the baseline amplitude to vary randomly per subject.

#### Main Analysis & Permutation Loop
```R
cat("[2/5] LM eps2...\n");   res_lm_eps2   <- run_lm_timewise(df, lm_f, "eps2_z")
cat("[3/5] LMER eps2...\n"); res_lmer_eps2 <- run_lmer_timewise(df, lmer_f, "eps2_z")
cat("[4/5] LM eps3...\n");   res_lm_eps3   <- run_lm_timewise(df, lm_f, "eps3_orth_z")
cat("      LMER eps3...\n"); res_lmer_eps3 <- run_lmer_timewise(df, lmer_f, "eps3_orth_z")
```
- **Details**: Calls the helper functions to run the real analysis on the un-shuffled data.

```R
# ── Cluster permutation test ──────────────────────────────────────────────────
cat("[5/5] Permutations eps2 (n=", N_PERM, ")...\n")
set.seed(42)

# Permute at the trial level instead of randomly across all rows
max_null_eps2 <- numeric(N_PERM)
for (i in seq_len(N_PERM)) {
  if (i %% 100 == 0) cat("  perm", i, "/", N_PERM, "\n")

  # Create a mapping of trial -> shuffled eps2_z
  trial_eps2 <- df %>% select(Trial, eps2_z) %>% distinct()
  trial_eps2$eps2_z_shuffled <- sample(trial_eps2$eps2_z)

  df_p <- df %>% left_join(trial_eps2 %>% select(Trial, eps2_z_shuffled), by="Trial") %>%
    mutate(eps2_z = eps2_z_shuffled) %>%
    select(-eps2_z_shuffled)

  res_p <- run_lm_timewise(df_p, lm_f, "eps2_z")
  cl    <- get_clusters(res_p)
  max_null_eps2[i] <- if (!is.null(cl)) max(cl$cluster_stat) else 0
}
```
- **Details**: Runs the permutation test for multiple comparisons correction.
  - It loops `N_PERM` (e.g., 500) times.
  - To break the relationship between prediction errors and brain activity while preserving the structure of the data, it extracts unique Trial IDs and their corresponding `eps2_z` values, shuffles (`sample`) those values, and merges them back into the main dataframe. This ensures that every millisecond inside a single trial gets the *same* shuffled value, preserving temporal autocorrelation.
  - It runs the LM on this junk data, finds clusters, records the mass of the largest cluster, and repeats. This builds a distribution of "the largest cluster you'd expect to see purely by chance."
  - This process is repeated exactly for `eps3`.

```R
# Attach permutation p-values to clusters
...
saveRDS(all_results, file.path(OUT_DIR, "all_results.rds"))
```
- **Details**: Uses the `attach_p` function to score the real clusters against the null distributions. Saves everything as an `.rds` object (R's binary format).

```R
# ── Save plots ────────────────────────────────────────────────────────────────
...
ggsave(file.path(plot_dir, "eps2_LMvsLMER.png"), p1, width=9, height=4.5, dpi=300)
```
- **Details**: Generates the plots using `plot_lm_vs_lmer` and saves them as high-resolution PNGs.

---

### `R/04_olife_correlation.R`

This script correlates neural metrics with clinical schizotypy scores.

```R
suppressPackageStartupMessages({
  library(dplyr); library(readr); library(ggplot2); library(tidyr)
})
... (Loads Data and Filters similar to Script 3, using Fz channel) ...
```

```R
# ── Subject-level MMN amplitude (deviant - standard, 100-200ms) ───────────────
subject_mmn <- df %>%
  filter(time_ms >= 100, time_ms <= 200) %>%
  group_by(subject, trial_type) %>%
  summarise(mean_amp=mean(amplitude, na.rm=TRUE), .groups="drop") %>%
  pivot_wider(names_from=trial_type, values_from=mean_amp) %>%
  mutate(MMN_amplitude = deviant - standard)
```
- **Purpose**: Calculates the classic Mismatch Negativity (MMN) size per subject.
- **Details**: Filters to the 100-200ms time window. Calculates average amplitude for standard and deviant trials per subject. Uses `pivot_wider` to put standard and deviant averages into separate columns, then subtracts standard from deviant to get the MMN amplitude.

```R
# ── Within-subject eps2-EEG coupling ─────────────────────────────────────────
subject_eps <- df %>%
  filter(time_ms >= 100, time_ms <= 200) %>%
  group_by(subject) %>%
  summarise(
    r_eps2_amp = cor(eps2_z, amplitude, use="complete.obs"),
    r_eps3_amp = cor(eps3_orth_z, amplitude, use="complete.obs"),
    .groups="drop")
```
- **Purpose**: Calculates how strongly prediction errors "couple" with brain activity.
- **Details**: For each subject, computes the Pearson correlation (`cor`) between single-trial prediction errors (`eps2`, `eps3`) and the single-trial EEG amplitude within the 100-200ms window. A strong correlation means the brain is highly sensitive to trial-by-trial surprise.

```R
subject_data <- subject_mmn %>%
  left_join(subject_eps, by="subject") %>%
  left_join(scores, by="subject")

# ── Print correlations ────────────────────────────────────────────────────────
... (Runs cor.test to test significance and prints results) ...
```
- **Details**: Joins the calculated neural metrics with the clinical `scores` dataset and prints statistical significance tests for the relationships.

```R
# ── Plot 1 & 2: Scatter plots ...
# ── Plot 3: Grand average ERP ...
```
- **Details**: Uses `ggplot2` to create scatter plots showing the relationship between neural metrics and O-LIFE scores, plotting a linear regression line. It also creates a standard ERP "butterfly" plot showing the average brain wave for standard vs. deviant trials, shaded with standard error margins.

---

## 3. Bash Scripts (SLURM)

These scripts send instructions to the HPC job scheduler (SLURM).

### `slurm/01_extract_eeg_array.sh`
```bash
#!/bin/bash
#SBATCH --job-name=eeg_extract
#SBATCH --qos=pool_kotesrj_ra
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-999%20
#SBATCH --cpus-per-task=2
...
```
- **Purpose**: Submits an "array" of jobs.
- **Details**: The `#SBATCH` tags tell the supercomputer what resources to reserve. `--array=0-999%20` means "spawn up to 1000 identical jobs (numbered 0 to 999), but only let 20 run at the exact same time".

```bash
set -euo pipefail
WORK_DIR="${WORK_DIR:-$HOME/rmmn}"

mapfile -t SET_FILES < <(ls "${WORK_DIR}/raw_set/"*_sequence_epochs.set 2>/dev/null | sort)
N_FILES=${#SET_FILES[@]}

if [[ $SLURM_ARRAY_TASK_ID -ge $N_FILES ]]; then
    echo "Array index $SLURM_ARRAY_TASK_ID >= $N_FILES files. Skipping."
    exit 0
fi

SET_FILE="${SET_FILES[$SLURM_ARRAY_TASK_ID]}"
```
- **Details**: Finds all `.set` files and puts them in a bash array list. SLURM provides an environment variable called `SLURM_ARRAY_TASK_ID` (e.g., 0, 1, 2). It uses this ID to pick exactly one file from the list. If the ID is higher than the number of files, it safely exits.

```bash
python3 - "$SET_FILE" "$WORK_DIR" << 'PYEOF'
... (Python code identical to 02_extract_all_subjects.py, but for ONE file) ...
PYEOF
```
- **Details**: Executes inline Python code. This code is the exact same logic as `02_extract_all_subjects.py`, but it only processes the single file passed to it.

### `slurm/02_concat_subjects.sh`
```bash
#!/bin/bash
#SBATCH --job-name=concat
...
python3 - "$WORK_DIR" << 'PYEOF'
import sys, glob
from pathlib import Path
import pandas as pd
... (finds all CSVs and pd.concat them) ...
```
- **Purpose**: A small script requested after the array job finishes. It runs a fast Python snippet to `pd.concat` the individual subject files into the master CSV.

### `slurm/03_submit_analysis.sh`
```bash
#!/bin/bash
#SBATCH --array=0-2
...
CHANNELS=(Fz Cz Pz)
export EEG_CHANNEL="${CHANNELS[$SLURM_ARRAY_TASK_ID]}"
...
Rscript "${WORK_DIR}/R/03_lm_vs_lmer_analysis.R"
```
- **Purpose**: Runs the R statistical analysis.
- **Details**: Sets up an array job of 3 tasks (0 to 2). Based on the Task ID, it sets an environment variable `EEG_CHANNEL` to either Fz, Cz, or Pz. Then it runs the R script. The R script reads this environment variable to know which channel to analyze, meaning all three channels are analyzed simultaneously on the supercomputer.

### `slurm/04_olife.sh`
```bash
#!/bin/bash
...
Rscript "${WORK_DIR}/R/04_olife_correlation.R"
```
- **Purpose**: Simply runs the final R correlation script using SLURM resources to avoid running heavy computations on the login node.