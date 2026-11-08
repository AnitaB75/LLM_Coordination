# =============================================================================
# Name Game — PREDICTION variant: visualizations
# =============================================================================
# Loads the most recent tidy prediction CSV from ../results and produces four
# figures characterising how the model expects humans to switch from J to M as
# a function of the round-8 evidence it observed.
#
# Figures (printed to the active device — not written with ggsave):
#   1. Predicted P(choose M) trajectories, rounds 9-14, by condition (+/- SE)
#   2. Proportion of runs predicting a full switch (all_M), by condition
#   3. Mean predicted letter mix (J/M/N/Other) as stacked areas, faceted by condition
#   4. Pooled group-outcome composition per condition (stacked bars)
#
# Run interactively (recommended, so plots render to screen):
#     source("visualize_prediction_results.R")
# =============================================================================

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

# -----------------------------------------------------------------------------
# Locate the newest prediction results CSV
# -----------------------------------------------------------------------------
# Resolve THIS script's own directory, regardless of how it is run:
#   - RStudio (Source button or pasting lines from the open editor tab)
#   - source("visualize_prediction_results.R")
#   - Rscript visualize_prediction_results.R
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

# Ordered factor with readable labels (ascending M evidence, split last)
cond_levels <- c("r8_0M", "r8_1M", "r8_2M", "r8_3M", "r8_4M", "r8_splitMN")
cond_labels <- c("r8_0M (6J/0M)", "r8_1M (5J/1M)", "r8_2M (4J/2M)",
                 "r8_3M (3J/3M)", "r8_4M (2J/4M)", "r8_splitMN (4J/1M/1N)")

dat <- read_csv(results_file, show_col_types = FALSE) %>%
  mutate(condition = factor(condition, levels = cond_levels, labels = cond_labels))

# Colour-blind-friendly palette for the 6 conditions (Okabe-Ito)
cond_palette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7")
names(cond_palette) <- cond_labels

theme_set(theme_minimal(base_size = 13))

# =============================================================================
# Figure 1 — Predicted P(choose M) trajectories (+/- SE)
# =============================================================================
probM <- dat %>%
  group_by(condition, round) %>%
  summarise(mean = mean(prob_M),
            se   = sd(prob_M) / sqrt(n()),
            .groups = "drop")

fig1 <- ggplot(probM, aes(round, mean, colour = condition, fill = condition)) +
  geom_ribbon(aes(ymin = mean - se, ymax = mean + se), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_colour_manual(values = cond_palette) +
  scale_fill_manual(values = cond_palette) +
  scale_x_continuous(breaks = 9:14) +
  coord_cartesian(ylim = c(0, 100)) +
  labs(title = "Predicted probability a human chooses M",
       subtitle = "Mean +/- SE across 20 runs per condition, rounds 9-14",
       x = "Round", y = "Predicted P(choose M)  (0-100)",
       colour = "Round-8 evidence", fill = "Round-8 evidence")
print(fig1)

# =============================================================================
# Figure 2 — Proportion of runs predicting a full switch (all_M)
# =============================================================================
allM <- dat %>%
  group_by(condition, round) %>%
  summarise(prop_all_M = mean(group_outcome == "all_M"), .groups = "drop")

fig2 <- ggplot(allM, aes(round, prop_all_M, colour = condition)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_colour_manual(values = cond_palette) +
  scale_x_continuous(breaks = 9:14) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(title = "Predicted probability of a FULL switch to M",
       subtitle = "Share of runs naming all_M as the most likely group outcome",
       x = "Round", y = "Proportion of runs predicting all_M",
       colour = "Round-8 evidence")
print(fig2)

# =============================================================================
# Figure 3 — Mean predicted letter mix (stacked areas), faceted by condition
# =============================================================================
mix <- dat %>%
  group_by(condition, round) %>%
  summarise(J = mean(prob_J), M = mean(prob_M),
            N = mean(prob_N), Other = mean(prob_Other), .groups = "drop") %>%
  pivot_longer(c(J, M, N, Other), names_to = "letter", values_to = "prob") %>%
  mutate(letter = factor(letter, levels = c("J", "M", "N", "Other")))

letter_palette <- c(J = "#0072B2", M = "#D55E00", N = "#009E73", Other = "#BBBBBB")

fig3 <- ggplot(mix, aes(round, prob, fill = letter)) +
  geom_area(alpha = 0.9) +
  facet_wrap(~condition) +
  scale_fill_manual(values = letter_palette) +
  scale_x_continuous(breaks = c(9, 11, 13)) +
  labs(title = "Mean predicted letter mix across rounds 9-14",
       subtitle = "Stacked mean probabilities (sum to 100 within each round)",
       x = "Round", y = "Mean predicted probability", fill = "Letter")
print(fig3)

# =============================================================================
# Figure 4 — Pooled group-outcome composition per condition (stacked bars)
# =============================================================================
outcome <- dat %>%
  mutate(group_outcome = factor(group_outcome,
                                levels = c("all_J", "all_M", "all_N", "mixed"))) %>%
  count(condition, group_outcome, name = "n") %>%
  group_by(condition) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

outcome_palette <- c(all_J = "#0072B2", all_M = "#D55E00",
                     all_N = "#009E73", mixed = "#BBBBBB")

fig4 <- ggplot(outcome, aes(condition, prop, fill = group_outcome)) +
  geom_col() +
  scale_fill_manual(values = outcome_palette) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Predicted group-outcome mix by condition",
       subtitle = "Pooled across rounds 9-14 (20 runs x 6 rounds = 120 predictions each)",
       x = "Round-8 evidence condition", y = "Share of predictions",
       fill = "Group outcome") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
print(fig4)

message("Done — printed 4 figures.")
