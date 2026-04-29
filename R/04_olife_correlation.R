#!/usr/bin/env Rscript
# =============================================================================
# 04_olife_correlation.R
# =============================================================================
# Computes subject-level MMN amplitude and eps2-EEG coupling,
# correlates with O-LIFE schizotypy scores, produces 3 plots.
#
# Run via SLURM (see slurm/04_olife.sh) or directly:
#   Rscript 04_olife_correlation.R
#
# Input:
#   rmmn/merged_eeg_hgf_ALL.csv
#   rmmn/results/Fz/df_clean.rds  (for good subject list)
#
# Output:
#   rmmn/results/olife/MMN_vs_OLIFE.png
#   rmmn/results/olife/eps2_coupling_vs_OLIFE.png
#   rmmn/results/olife/grand_average_ERP.png
#   rmmn/results/olife/subject_level_data.csv
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(ggplot2); library(tidyr)
})

WORK_DIR <- Sys.getenv("WORK_DIR", unset = file.path(Sys.getenv("HOME"), "rmmn"))
OUT_DIR  <- file.path(WORK_DIR, "results", "olife")
dir.create(OUT_DIR, recursive=TRUE, showWarnings=FALSE)

cat("Loading data...\n")
all_data <- read_csv(
  file.path(WORK_DIR, "merged_eeg_hgf_ALL.csv"),
  col_select = c("subject","channel","time_ms","amplitude",
                 "BadEpoch","eps2","eps3","trial_type",
                 "Total_O_LIFE_Score","LSHS_Total_Score","ASI_Total_Score"),
  show_col_types = FALSE)

# Use same subjects as Fz analysis
df_fz     <- readRDS(file.path(WORK_DIR, "results/Fz/df_clean.rds"))
good_subs <- df_fz %>% distinct(subject) %>% pull(subject)

df <- all_data %>%
  filter(subject %in% good_subs, BadEpoch==0, channel=="Fz")
rm(all_data); gc()

eps3_orth      <- resid(lm(eps3 ~ eps2, data=df))
df$eps2_z      <- as.numeric(scale(df$eps2))
df$eps3_orth_z <- as.numeric(scale(eps3_orth))
cat("Subjects:", n_distinct(df$subject), "\n")

# ── Subject-level MMN amplitude (deviant - standard, 100-200ms) ───────────────
subject_mmn <- df %>%
  filter(time_ms >= 100, time_ms <= 200) %>%
  group_by(subject, trial_type) %>%
  summarise(mean_amp=mean(amplitude, na.rm=TRUE), .groups="drop") %>%
  pivot_wider(names_from=trial_type, values_from=mean_amp) %>%
  mutate(MMN_amplitude = deviant - standard)

# ── Within-subject eps2-EEG coupling ─────────────────────────────────────────
subject_eps <- df %>%
  filter(time_ms >= 100, time_ms <= 200) %>%
  group_by(subject) %>%
  summarise(
    r_eps2_amp = cor(eps2_z, amplitude, use="complete.obs"),
    r_eps3_amp = cor(eps3_orth_z, amplitude, use="complete.obs"),
    .groups="drop")

scores <- df %>%
  select(subject, Total_O_LIFE_Score, LSHS_Total_Score, ASI_Total_Score) %>%
  distinct()

subject_data <- subject_mmn %>%
  left_join(subject_eps, by="subject") %>%
  left_join(scores, by="subject")

# ── Print correlations ────────────────────────────────────────────────────────
cat("\n=== Correlations with O-LIFE ===\n")
for (v in c("MMN_amplitude","r_eps2_amp","r_eps3_amp")) {
  sub <- subject_data %>%
    filter(!is.na(Total_O_LIFE_Score), !is.na(.data[[v]]))
  ct  <- cor.test(sub[[v]], sub$Total_O_LIFE_Score)
  cat(sprintf("%-20s r=%.3f p=%.4f n=%d\n", v, ct$estimate, ct$p.value, nrow(sub)))
}

# ── Plot 1: MMN amplitude vs O-LIFE ──────────────────────────────────────────
p1 <- ggplot(subject_data %>% filter(!is.na(Total_O_LIFE_Score)),
             aes(x=MMN_amplitude, y=Total_O_LIFE_Score)) +
  geom_point(color="#7B3F9E", size=3, alpha=0.8) +
  geom_smooth(method="lm", color="#1B3A6B", se=TRUE) +
  labs(title="MMN Amplitude vs Schizotypy (O-LIFE)",
       subtitle="Fz channel, 100-200ms window",
       x="MMN Amplitude (deviant - standard, uV)",
       y="O-LIFE Total Score") +
  theme_minimal(base_size=13) +
  theme(plot.title=element_text(face="bold"))
ggsave(file.path(OUT_DIR, "MMN_vs_OLIFE.png"), p1, width=6, height=5, dpi=300)

# ── Plot 2: eps2-EEG coupling vs O-LIFE ──────────────────────────────────────
p2 <- ggplot(subject_data %>% filter(!is.na(Total_O_LIFE_Score)),
             aes(x=r_eps2_amp, y=Total_O_LIFE_Score)) +
  geom_point(color="#7B3F9E", size=3, alpha=0.8) +
  geom_smooth(method="lm", color="#1B3A6B", se=TRUE) +
  labs(title="eps2-EEG Coupling vs Schizotypy (O-LIFE)",
       subtitle="Within-subject r(eps2, amplitude), Fz 100-200ms",
       x="Within-subject r(eps2, EEG amplitude)",
       y="O-LIFE Total Score") +
  theme_minimal(base_size=13) +
  theme(plot.title=element_text(face="bold"))
ggsave(file.path(OUT_DIR, "eps2_coupling_vs_OLIFE.png"), p2, width=6, height=5, dpi=300)

# ── Plot 3: Grand average ERP deviant vs standard ─────────────────────────────
waveform <- df %>%
  group_by(trial_type, time_ms) %>%
  summarise(mean_amp=mean(amplitude, na.rm=TRUE),
            se_amp  =sd(amplitude, na.rm=TRUE)/sqrt(n()),
            .groups="drop")

p3 <- ggplot(waveform, aes(x=time_ms, y=mean_amp,
                            color=trial_type, fill=trial_type)) +
  annotate("rect", xmin=100, xmax=200, ymin=-Inf, ymax=Inf,
           fill="#D4B0F0", alpha=0.2) +
  annotate("rect", xmin=300, xmax=400, ymin=-Inf, ymax=Inf,
           fill="#B0E8C8", alpha=0.2) +
  geom_ribbon(aes(ymin=mean_amp-se_amp, ymax=mean_amp+se_amp),
              alpha=0.2, color=NA) +
  geom_line(linewidth=1.2) +
  geom_hline(yintercept=0, linetype="dashed", color="grey50") +
  scale_color_manual(values=c(deviant="#C0392B", standard="#2980B9")) +
  scale_fill_manual(values =c(deviant="#C0392B", standard="#2980B9")) +
  labs(title="Grand Average ERP: Deviant vs Standard",
       x="Time (ms)", y="Amplitude (uV)",
       color="Trial type", fill="Trial type") +
  theme_minimal(base_size=13) +
  theme(plot.title=element_text(face="bold"), legend.position="bottom")
ggsave(file.path(OUT_DIR, "grand_average_ERP.png"), p3, width=8, height=5, dpi=300)

write_csv(subject_data, file.path(OUT_DIR, "subject_level_data.csv"))
cat("\nDone! Plots saved to:", OUT_DIR, "\n")
