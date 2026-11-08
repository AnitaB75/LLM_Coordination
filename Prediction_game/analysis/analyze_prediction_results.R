# =============================================================================
# Name Game — PREDICTION variant: analysis
# =============================================================================
# Loads the most recent tidy prediction CSV from ../results and summarises
# whether the model expects humans to switch from J to M across rounds 9-14,
# and how that expectation scales with the amount of round-8 switching evidence.
#
# Key question: does the model UNDERESTIMATE switching? i.e. is predicted
# P(all_M) / mean prob_M low, and does it rise too weakly with stronger
# round-8 evidence (r8_0M -> r8_4M)?
#
# Run (from this analysis/ folder, or the repo root):
#     source("analyze_prediction_results.R")
# =============================================================================

library(tidyverse)

# -----------------------------------------------------------------------------
# Locate the newest prediction results CSV
# -----------------------------------------------------------------------------
# Resolve THIS script's own directory, regardless of how it is run:
#   - RStudio (Source button or pasting lines from the open editor tab)
#   - source("analyze_prediction_results.R")
#   - Rscript analyze_prediction_results.R
# This is what lets you just open the file from the folder and run it; the
# results path is anchored to the script, not to the working directory.
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)              # Rscript --file=
  m <- grep("^--file=", args, value = TRUE)
  if (length(m)) return(dirname(normalizePath(sub("^--file=", "", m[1]))))
  for (i in seq_len(sys.nframe())) {                     # source() sets ofile
    of <- sys.frame(i)$ofile
    if (!is.null(of)) return(dirname(normalizePath(of)))
  }
  if (requireNamespace("rstudioapi", quietly = TRUE) &&  # RStudio editor tab
      rstudioapi::isAvailable()) {
    p <- tryCatch(rstudioapi::getSourceEditorContext()$path, error = function(e) "")
    if (length(p) && nzchar(p)) return(dirname(normalizePath(p)))
  }
  NA_character_
}

find_results <- function() {
  # Prefer the results/ next to this script; fall back to CWD-relative dirs.
  # Scan every existing candidate for matching files and pick the newest, so a
  # repo-root CWD never matches the player study's results/ dir by mistake.
  script_dir <- get_script_dir()
  dirs <- c(if (!is.na(script_dir)) file.path(script_dir, "..", "results"),
            "../results", "results", "Prediction_game/results")
  dirs <- unique(dirs[dir.exists(dirs)])
  files <- unlist(lapply(dirs, function(d)
    list.files(d, pattern = "name_game_prediction_MN_.*\\.csv$", full.names = TRUE)))
  if (length(files) == 0)
    stop("No prediction result CSVs found. Looked in: ",
         paste(normalizePath(dirs, mustWork = FALSE), collapse = ", "),
         ". Run the experiment first (Name_Game_prediction_MN.py).")
  files[which.max(file.mtime(files))]
}

results_file <- find_results()
message("Loading: ", results_file)

condition_levels <- c("r8_0M", "r8_1M", "r8_2M", "r8_3M", "r8_4M", "r8_splitMN")

dat <- read_csv(results_file, show_col_types = FALSE) %>%
  mutate(condition = factor(condition, levels = condition_levels))

# -----------------------------------------------------------------------------
# Data-quality check: probabilities should sum to 100 within each round
# -----------------------------------------------------------------------------
bad <- dat %>% filter(prob_sum != 100)
if (nrow(bad) > 0) {
  warning(nrow(bad), " round-rows have probabilities not summing to 100. ",
          "They are retained as-is; inspect `bad` before trusting means.")
} else {
  message("QA: all rounds have probabilities summing to 100.")
}

# =============================================================================
# 1. Mean predicted letter-probabilities by condition x round
# =============================================================================
prob_summary <- dat %>%
  group_by(condition, round) %>%
  summarise(across(c(prob_J, prob_M, prob_N, prob_Other),
                   list(mean = mean, se = ~sd(.) / sqrt(n())),
                   .names = "{.col}_{.fn}"),
            n = n(), .groups = "drop")

cat("\n=== Mean predicted P(choose M) by condition x round ===\n")
prob_summary %>%
  select(condition, round, prob_M_mean) %>%
  pivot_wider(names_from = round, values_from = prob_M_mean,
              names_prefix = "r") %>%
  print(n = Inf)

# =============================================================================
# 2. Predicted group-outcome distribution by condition x round
# =============================================================================
outcome_summary <- dat %>%
  count(condition, round, group_outcome, name = "n_runs") %>%
  group_by(condition, round) %>%
  mutate(prop = n_runs / sum(n_runs)) %>%
  ungroup()

# Proportion of runs predicting full switch to M (all_M)
allM_summary <- outcome_summary %>%
  filter(group_outcome == "all_M") %>%
  select(condition, round, prop_all_M = prop)

cat("\n=== Proportion of runs predicting all_M, by condition x round ===\n")
allM_summary %>%
  pivot_wider(names_from = round, values_from = prop_all_M,
              names_prefix = "r", values_fill = 0) %>%
  print(n = Inf)

# =============================================================================
# 3. Headline metrics per condition (collapsed across rounds 9-14)
# =============================================================================
headline <- dat %>%
  group_by(condition) %>%
  summarise(
    mean_prob_M     = mean(prob_M),
    mean_prob_J     = mean(prob_J),
    pct_rounds_allM = mean(group_outcome == "all_M") * 100,
    pct_rounds_allJ = mean(group_outcome == "all_J") * 100,
    pct_rounds_mixed = mean(group_outcome == "mixed") * 100,
    .groups = "drop"
  )

cat("\n=== Headline metrics by condition (rounds 9-14 pooled) ===\n")
print(headline, n = Inf)

# =============================================================================
# Plots  (printed to the active device — not written with ggsave)
# =============================================================================
p_probM <- ggplot(prob_summary,
                  aes(x = round, y = prob_M_mean,
                      colour = condition, group = condition)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_ribbon(aes(ymin = prob_M_mean - prob_M_se,
                  ymax = prob_M_mean + prob_M_se,
                  fill = condition), alpha = 0.12, colour = NA) +
  scale_x_continuous(breaks = 9:14) +
  labs(title = "Predicted P(a human chooses M) across rounds 9-14",
       subtitle = "By round-8 evidence condition (mean +/- SE)",
       x = "Round", y = "Mean predicted P(choose M)",
       colour = "Condition", fill = "Condition") +
  theme_minimal(base_size = 13)
print(p_probM)

p_allM <- allM_summary %>%
  ggplot(aes(x = round, y = prop_all_M, colour = condition, group = condition)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = 9:14) +
  ylim(0, 1) +
  labs(title = "Predicted probability of a full switch to M (all_M)",
       subtitle = "Share of runs predicting all_M as the group outcome",
       x = "Round", y = "Proportion of runs predicting all_M",
       colour = "Condition") +
  theme_minimal(base_size = 13)
print(p_allM)

p_headline <- headline %>%
  ggplot(aes(x = condition, y = mean_prob_M, fill = condition)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.0f", mean_prob_M)), vjust = -0.4, size = 4) +
  labs(title = "Mean predicted P(choose M) by round-8 evidence",
       subtitle = "Pooled across rounds 9-14",
       x = "Round-8 evidence condition", y = "Mean predicted P(choose M)") +
  guides(fill = "none") +
  theme_minimal(base_size = 13)
print(p_headline)

cat("\n=== Analysis complete ===\n")
