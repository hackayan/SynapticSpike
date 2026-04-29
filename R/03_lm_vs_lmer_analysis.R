#!/usr/bin/env Rscript
# =============================================================================
# 03_lm_vs_lmer_analysis.R
# =============================================================================
# Timewise LM vs LMER analysis with cluster permutation test.
# Runs at every timepoint (-100 to 400ms) for eps2 and eps3 predictors.
# Produces LM vs LMER overlay plots for Fz, Cz, Pz channels.
#
# Run via SLURM (see slurm/03_submit_analysis.sh) or directly:
#   EEG_CHANNEL=Fz N_PERM=500 Rscript 03_lm_vs_lmer_analysis.R
#
# Input:
#   rmmn/merged_eeg_hgf_ALL.csv
#
# Output:
#   rmmn/results/{channel}/plots/eps2_LMvsLMER.png
#   rmmn/results/{channel}/plots/eps3_LMvsLMER.png
#   rmmn/results/{channel}/all_results.rds
#   rmmn/results/{channel}/df_clean.rds
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(lme4)
})

WORK_DIR <- Sys.getenv("WORK_DIR", unset = file.path(Sys.getenv("HOME"), "rmmn"))
N_PERM   <- as.integer(Sys.getenv("N_PERM",    unset = "500"))
CHANNEL  <- Sys.getenv("EEG_CHANNEL",           unset = "Fz")
OUT_DIR  <- file.path(WORK_DIR, "results", CHANNEL)
dir.create(file.path(OUT_DIR, "plots"), recursive=TRUE, showWarnings=FALSE)

cat("============================================\n")
cat("CHANNEL  :", CHANNEL, "\n")
cat("N_PERM   :", N_PERM,  "\n")
cat("OUT_DIR  :", OUT_DIR, "\n")
cat("============================================\n\n")

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

# ── Helper: run LMER at every timepoint ──────────────────────────────────────
run_lmer_timewise <- function(df, formula, effect_name) {
  df %>% group_by(time_ms) %>%
    do({
      fit <- tryCatch(
        lmer(formula, data=., REML=FALSE,
             control=lmerControl(optimizer="bobyqa",
                                 check.conv.singular=.makeCC("ignore",tol=1e-4))),
        error=function(e) NULL)
      if (is.null(fit)) return(data.frame(t_value=0, p_value=1))
      coef <- summary(fit)$coefficients
      if (effect_name %in% rownames(coef)) {
        t_val <- coef[effect_name,"t value"]
        data.frame(t_value=t_val, p_value=2*(1-pnorm(abs(t_val))))
      } else data.frame(t_value=0, p_value=1)
    }) %>% ungroup()
}

# ── Helper: find significant clusters ────────────────────────────────────────
get_clusters <- function(res_df, t_thresh=2.0) {
  res_df <- res_df %>% arrange(time_ms) %>% mutate(sig=abs(t_value)>t_thresh)
  clusters <- list(); current <- c()
  for (i in seq_len(nrow(res_df))) {
    if (res_df$sig[i]) current <- c(current,i)
    else if (length(current)>0) {
      clusters[[length(clusters)+1]] <- current; current <- c()
    }
  }
  if (length(current)>0) clusters[[length(clusters)+1]] <- current
  if (length(clusters)==0) return(NULL)
  do.call(rbind, lapply(clusters, function(idx) {
    sub <- res_df[idx,]
    data.frame(start=min(sub$time_ms), end=max(sub$time_ms),
               cluster_stat=sum(abs(sub$t_value)))
  }))
}

# ── Helper: attach permutation p-values ──────────────────────────────────────
attach_p <- function(cl, null) {
  if (!is.null(cl) && nrow(cl)>0)
    cl$p_perm <- sapply(cl$cluster_stat, function(cs) mean(null >= cs))
  cl
}

# ── Helper: plot LM vs LMER overlay ──────────────────────────────────────────
plot_lm_vs_lmer <- function(lm_res, lmer_res, title) {
  combined <- bind_rows(
    lm_res   %>% mutate(model="LM (fixed only)"),
    lmer_res %>% mutate(model="LMER (+ random subject)")
  )
  ggplot(combined, aes(time_ms, t_value, color=model, linetype=model)) +
    annotate("rect", xmin=100, xmax=200, ymin=-Inf, ymax=Inf,
             fill="#D4B0F0", alpha=0.2) +
    annotate("rect", xmin=300, xmax=400, ymin=-Inf, ymax=Inf,
             fill="#B0E8C8", alpha=0.2) +
    geom_line(linewidth=1.1) +
    geom_hline(yintercept=0,  linetype="dashed", color="grey50") +
    geom_hline(yintercept= 2, linetype="dotted", color="grey40") +
    geom_hline(yintercept=-2, linetype="dotted", color="grey40") +
    scale_color_manual(values=c("LM (fixed only)"="#7B9EC7",
                                "LMER (+ random subject)"="#1B3A6B")) +
    scale_linetype_manual(values=c("LM (fixed only)"="dashed",
                                   "LMER (+ random subject)"="solid")) +
    labs(title=title, x="Time (ms)", y="t-value", color=NULL, linetype=NULL) +
    theme_minimal(base_size=12) +
    theme(legend.position="bottom")
}

# ── Load & clean data ─────────────────────────────────────────────────────────
cat("[1/5] Loading data...\n")
all_data <- read_csv(
  file.path(WORK_DIR, "merged_eeg_hgf_ALL.csv"),
  col_select = c("subject","Trial","channel","time_ms","amplitude",
                 "BadEpoch","eps2","eps3","trial_type","u","cat_code",
                 "Total_O_LIFE_Score","LSHS_Total_Score","ASI_Total_Score"),
  show_col_types = FALSE)

# Remove subjects with >30% bad epochs
bad_pct   <- all_data %>%
  filter(channel==CHANNEL, time_ms==first(time_ms)) %>%
  group_by(subject) %>%
  summarise(bad_pct=mean(BadEpoch==1)*100, .groups="drop")
good_subs <- bad_pct %>% filter(bad_pct <= 30) %>% pull(subject)
cat("  Subjects kept:", length(good_subs), "\n")

df <- all_data %>%
  filter(subject %in% good_subs, BadEpoch==0, channel==CHANNEL) %>%
  mutate(Condition=factor(cat_code), u=as.integer(trial_type=="deviant"))
rm(all_data); gc()

# Orthogonalise eps3 w.r.t. eps2 and z-score
eps3_orth      <- resid(lm(eps3 ~ eps2, data=df))
df$eps2_z      <- as.numeric(scale(df$eps2))
df$eps3_orth_z <- as.numeric(scale(eps3_orth))

saveRDS(df, file.path(OUT_DIR, "df_clean.rds"))
cat("  Rows:", nrow(df), "| Time points:", n_distinct(df$time_ms), "\n\n")

lm_f   <- amplitude ~ u + Condition + eps2_z + eps3_orth_z
lmer_f <- amplitude ~ u + Condition + eps2_z + eps3_orth_z + (1|subject)

# ── Run models ────────────────────────────────────────────────────────────────
cat("[2/5] LM eps2...\n");   res_lm_eps2   <- run_lm_timewise(df, lm_f, "eps2_z")
cat("[3/5] LMER eps2...\n"); res_lmer_eps2 <- run_lmer_timewise(df, lmer_f, "eps2_z")
cat("[4/5] LM eps3...\n");   res_lm_eps3   <- run_lm_timewise(df, lm_f, "eps3_orth_z")
cat("      LMER eps3...\n"); res_lmer_eps3 <- run_lmer_timewise(df, lmer_f, "eps3_orth_z")

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

cat("      Permutations eps3...\n")
max_null_eps3 <- numeric(N_PERM)
for (i in seq_len(N_PERM)) {
  if (i %% 100 == 0) cat("  perm", i, "/", N_PERM, "\n")

  # Create a mapping of trial -> shuffled eps3_orth_z
  trial_eps3 <- df %>% select(Trial, eps3_orth_z) %>% distinct()
  trial_eps3$eps3_orth_z_shuffled <- sample(trial_eps3$eps3_orth_z)

  df_p <- df %>% left_join(trial_eps3 %>% select(Trial, eps3_orth_z_shuffled), by="Trial") %>%
    mutate(eps3_orth_z = eps3_orth_z_shuffled) %>%
    select(-eps3_orth_z_shuffled)

  res_p <- run_lm_timewise(df_p, lm_f, "eps3_orth_z")
  cl    <- get_clusters(res_p)
  max_null_eps3[i] <- if (!is.null(cl)) max(cl$cluster_stat) else 0
}

# Attach permutation p-values to clusters
cl_lm_eps2   <- attach_p(get_clusters(res_lm_eps2),   max_null_eps2)
cl_lmer_eps2 <- attach_p(get_clusters(res_lmer_eps2), max_null_eps2)
cl_lm_eps3   <- attach_p(get_clusters(res_lm_eps3),   max_null_eps3)
cl_lmer_eps3 <- attach_p(get_clusters(res_lmer_eps3), max_null_eps3)

all_results <- list(
  eps2=list(LM  =list(res_time=res_lm_eps2,   clusters=cl_lm_eps2),
            LMER=list(res_time=res_lmer_eps2,  clusters=cl_lmer_eps2)),
  eps3=list(LM  =list(res_time=res_lm_eps3,   clusters=cl_lm_eps3),
            LMER=list(res_time=res_lmer_eps3,  clusters=cl_lmer_eps3))
)
saveRDS(all_results, file.path(OUT_DIR, "all_results.rds"))

# ── Save plots ────────────────────────────────────────────────────────────────
plot_dir <- file.path(OUT_DIR, "plots")

p1 <- plot_lm_vs_lmer(res_lm_eps2, res_lmer_eps2,
                       paste0(CHANNEL, " | eps2 (Sensory PE) - LM vs LMER"))
ggsave(file.path(plot_dir, "eps2_LMvsLMER.png"), p1, width=9, height=4.5, dpi=300)
cat("Saved eps2 plot\n")

p2 <- plot_lm_vs_lmer(res_lm_eps3, res_lmer_eps3,
                       paste0(CHANNEL, " | eps3 (Volatility PE) - LM vs LMER"))
ggsave(file.path(plot_dir, "eps3_LMvsLMER.png"), p2, width=9, height=4.5, dpi=300)
cat("Saved eps3 plot\n")

cat("\nDone! Plots in:", plot_dir, "\n")
