# Path resolves from this script's own location, so the project can be moved.
.script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) normalizePath(dirname(f)) else normalizePath(getwd())
}

library(tidyverse)
library(ggplot2)
library(scales)
library(gridExtra)

setwd(.script_dir())

# Load data (combine LLM-only and human-condition runs)
df_llm   <- read_csv("results/name_game_LLM_feedback_MN_20260507_134935.csv",
                     show_col_types = FALSE)
df_human <- read_csv("results/name_game_LLM_feedback_MN_HUMAN_20260507_212952.csv",
                     show_col_types = FALSE)
df <- bind_rows(df_llm, df_human)

cat("================================================================================\n")
cat("NAME GAME COORDINATION ANALYSIS\n")
cat("================================================================================\n")
cat(sprintf("\nDataset: %d total rows across %d trials\n", nrow(df), n_distinct(df$trial)))
cat(sprintf("Partner conditions: %s\n", paste(unique(df$partner_condition), collapse = ", ")))
cat(sprintf("Rounds per trial: %d\n", max(df$round)))

# Long-form player choices (used throughout)
df_long <- df %>%
  select(trial, partner_condition, round, p1:p6) %>%
  pivot_longer(p1:p6, names_to = "player", values_to = "choice")

# Phase labels
df <- df %>% mutate(phase = if_else(round <= 7, "Rounds 1-7 (J only)", "Rounds 8-14 (J/M/N)"))
df_long <- df_long %>%
  left_join(df %>% select(trial, round, phase) %>% distinct(), by = c("trial", "round"))


# ==============================================================================
# 1. OVERALL MATCHING RATES
# ==============================================================================

cat("\n================================================================================\n")
cat("1. OVERALL MATCHING RATES\n")
cat("================================================================================\n")

cat(sprintf("\nOverall matching rate: %.1f%%\n", mean(df$matched) * 100))

condition_matches <- df %>%
  group_by(partner_condition) %>%
  summarise(
    match_rate  = mean(matched),
    matches     = sum(matched),
    total_rounds = n(),
    .groups = "drop"
  )
cat("\nMatching rate by partner condition:\n")
print(condition_matches)

round_matches <- df %>%
  group_by(round) %>%
  summarise(match_rate = mean(matched), .groups = "drop")
cat("\nMatching rate by round:\n")
print(round_matches, n = 14)


# ==============================================================================
# 2. LETTER CHOICE DISTRIBUTION
# ==============================================================================

cat("\n================================================================================\n")
cat("2. LETTER CHOICE DISTRIBUTION\n")
cat("================================================================================\n")

cat("\nOverall letter distribution:\n")
print(sort(table(df_long$choice), decreasing = TRUE))

cat("\nRounds 1-7 (J=10, others=0):\n")
print(sort(table(df_long$choice[df_long$round <= 7]), decreasing = TRUE))

cat("\nRounds 8-14 (J=10, M=15, N=15):\n")
print(sort(table(df_long$choice[df_long$round >= 8]), decreasing = TRUE))


# ==============================================================================
# 3. SWITCHING BEHAVIOR AT ROUND 8
# ==============================================================================

cat("\n================================================================================\n")
cat("3. SWITCHING BEHAVIOR AT ROUND 8\n")
cat("================================================================================\n")

round8 <- df %>% filter(round == 8)
cat(sprintf("\nRound 8 letter choices across %d trials:\n", nrow(round8)))

round8 %>%
  rowwise() %>%
  mutate(
    choice_dist = list(table(c(p1, p2, p3, p4, p5, p6))),
    summary = paste(names(choice_dist), "=", as.integer(choice_dist), collapse = ", ")
  ) %>%
  ungroup() %>%
  select(trial, partner_condition, summary, matched) %>%
  mutate(result = if_else(matched, "MATCHED", "NO MATCH")) %>%
  { cat(paste(sprintf("  Trial %d (%s): %s [%s]",
                      .$trial, .$partner_condition, .$summary, .$result),
             collapse = "\n"), "\n") }


# ==============================================================================
# 4. CONVERGENCE ANALYSIS
# ==============================================================================

cat("\n================================================================================\n")
cat("4. CONVERGENCE ANALYSIS\n")
cat("================================================================================\n")

convergence_times <- df %>%
  arrange(trial, round) %>%
  group_by(trial) %>%
  mutate(
    mismatch_start = if_else(!matched & lag(matched, default = TRUE), round, NA_real_),
    converge_round = if_else(matched & !lag(matched, default = TRUE), round, NA_real_)
  ) %>%
  summarise(
    mismatch_starts = list(na.omit(mismatch_start)),
    converge_rounds = list(na.omit(converge_round)),
    .groups = "drop"
  ) %>%
  rowwise() %>%
  mutate(
    n_pairs = min(length(mismatch_starts), length(converge_rounds)),
    conv_times = list(
      if (n_pairs > 0)
        converge_rounds[seq_len(n_pairs)] - mismatch_starts[seq_len(n_pairs)]
      else
        integer(0)
    )
  ) %>%
  pull(conv_times) %>%
  unlist()

if (length(convergence_times) > 0) {
  cat(sprintf("\nAverage rounds to reconverge after mismatch: %.2f\n",
              mean(convergence_times)))
  cat("Convergence time distribution:\n")
  print(sort(table(convergence_times)))
} else {
  cat("\nNo convergence events detected (all rounds matched or never reconverged)\n")
}


# ==============================================================================
# 5. POINTS EARNED ANALYSIS
# ==============================================================================

cat("\n================================================================================\n")
cat("5. POINTS EARNED ANALYSIS\n")
cat("================================================================================\n")

final_scores <- df %>%
  filter(round == 14) %>%
  select(trial, partner_condition, p1_total:p6_total) %>%
  rowwise() %>%
  mutate(avg_score = mean(c_across(p1_total:p6_total))) %>%
  ungroup()

cat("\nFinal scores by trial:\n")
final_scores %>%
  { cat(paste(sprintf("  Trial %d (%s): %.0f points",
                      .$trial, .$partner_condition, .$avg_score),
             collapse = "\n"), "\n") }

cat("\nAverage final score by partner condition:\n")
final_scores %>%
  group_by(partner_condition) %>%
  summarise(
    avg_score = mean(c_across(p1_total:p6_total), na.rm = TRUE),
    .groups = "drop"
  ) %>%
  print()

max_J_only  <- 14 * 10
max_optimal <- 7 * 10 + 7 * 15
cat(sprintf("\nMaximum possible scores:\n"))
cat(sprintf("  J only (all 14 rounds): %d points\n", max_J_only))
cat(sprintf("  Optimal (J in 1-7, M/N in 8-14): %d points\n", max_optimal))


# ==============================================================================
# 6. REASONING ANALYSIS
# ==============================================================================

cat("\n================================================================================\n")
cat("6. REASONING PATTERNS\n")
cat("================================================================================\n")

reasoning_cols <- paste0("p", 1:6, "_reasoning")
total_possible <- nrow(df) * 6
total_reasoning <- df %>%
  select(all_of(reasoning_cols)) %>%
  summarise(across(everything(), ~ sum(!is.na(.) & . != ""))) %>%
  rowSums()

cat(sprintf("\nTotal reasoning entries: %d out of %d possible (%.1f%%)\n",
            total_reasoning, total_possible, total_reasoning / total_possible * 100))

cat("\nSample reasoning from Round 8 (decision point for M/N):\n")
df %>%
  filter(round == 8) %>%
  slice_head(n = 2) %>%
  rowwise() %>%
  group_walk(function(row, key) {
    cat(sprintf("\nTrial %d (%s):\n", row$trial, row$partner_condition))
    for (i in 1:6) {
      r <- row[[paste0("p", i, "_reasoning")]]
      if (!is.na(r) && nchar(trimws(r)) > 0) {
        cat(sprintf("  Player %d chose %s: %s...\n",
                    i, row[[paste0("p", i)]], substr(r, 1, 150)))
        break
      }
    }
  })


# ==============================================================================
# 7. VISUALIZATION
# ==============================================================================

cat("\n================================================================================\n")
cat("7. GENERATING VISUALIZATIONS\n")
cat("================================================================================\n")

letter_by_round <- df_long %>%
  group_by(round, choice) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(round) %>%
  mutate(proportion = n / sum(n)) %>%
  filter(choice %in% c("J", "M", "N"))

letter_colors <- c("J" = "#1f77b4", "M" = "#ff7f0e", "N" = "#2ca02c")
phase_line <- geom_vline(xintercept = 7.5, color = "red",
                         linetype = "dashed", alpha = 0.6)
x_scale <- scale_x_continuous(breaks = 1:14)

# Plot 1: Matching rate by round
p1 <- ggplot(round_matches, aes(x = round, y = match_rate)) +
  geom_col(fill = "steelblue") +
  phase_line +
  x_scale +
  scale_y_continuous(labels = percent, limits = c(0, 1.1)) +
  labs(title = "Matching Rate by Round",
       x = "Round", y = "Match Rate") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 13))

# Plot 2: Letter choice proportions over time
p2 <- ggplot(letter_by_round,
             aes(x = round, y = proportion, color = choice, group = choice)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2) +
  phase_line +
  x_scale +
  scale_color_manual(values = letter_colors,
                     labels = c("J" = "J (10 pts)",
                                "M" = "M (15 pts R8+)",
                                "N" = "N (15 pts R8+)")) +
  scale_y_continuous(labels = percent, limits = c(0, 1)) +
  labs(title = "Letter Choice Proportions Over Rounds",
       x = "Round", y = "Proportion", color = "Letter") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 13),
        legend.position = "bottom")

# Plot 3: Matching rate by condition
p3 <- ggplot(condition_matches,
             aes(x = reorder(partner_condition, match_rate), y = match_rate)) +
  geom_col(fill = "coral") +
  scale_y_continuous(labels = percent, limits = c(0, 1.1)) +
  labs(title = "Matching Rate by Partner Condition",
       x = "Partner Condition", y = "Match Rate") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 13),
        axis.text.x = element_text(angle = 30, hjust = 1))

# Plot 4: Final score distribution by condition
score_long <- final_scores %>%
  pivot_longer(p1_total:p6_total, names_to = "player", values_to = "score")

p4 <- ggplot(score_long, aes(x = partner_condition, y = score)) +
  geom_boxplot(fill = "lightblue", outlier.shape = 21) +
  geom_hline(yintercept = max_optimal, color = "green",
             linetype = "dashed", alpha = 0.7) +
  geom_hline(yintercept = max_J_only, color = "orange",
             linetype = "dashed", alpha = 0.7) +
  annotate("text", x = 0.6, y = max_optimal + 3, label = "Optimal (175)",
           color = "darkgreen", size = 3, hjust = 0) +
  annotate("text", x = 0.6, y = max_J_only + 3, label = "J-only (140)",
           color = "darkorange", size = 3, hjust = 0) +
  labs(title = "Final Score Distribution by Condition",
       x = "Partner Condition", y = "Final Score") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 13),
        axis.text.x = element_text(angle = 30, hjust = 1))

print(arrangeGrob(p1, p2, p3, p4, ncol = 2,
                  top = "Name Game Coordination Analysis"))

# Plot 5: Switching behavior (J vs M/N) by condition over all 14 rounds
panel_theme <- theme_minimal() +
  theme(
    plot.title       = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle    = element_text(size = 11, hjust = 0.5),
    strip.text       = element_text(face = "bold", size = 13),
    strip.background = element_rect(fill = "lightgray", color = "black"),
    legend.position  = "bottom",
    legend.title     = element_text(face = "bold", size = 12),
    legend.text      = element_text(size = 11),
    axis.title       = element_text(face = "bold", size = 12),
    axis.text        = element_text(size = 10),
    panel.grid.minor = element_blank(),
    panel.border     = element_rect(color = "black", fill = NA, linewidth = 0.5),
    plot.background  = element_rect(color = "#4472C4", linewidth = 2, fill = NA)
  )

switching <- df_long %>%
  mutate(letter_group = case_when(
    choice == "J" ~ "J",
    choice %in% c("M", "N") ~ "M",
    TRUE ~ NA_character_
  )) %>%
  filter(!is.na(letter_group)) %>%
  group_by(partner_condition, round, letter_group) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(partner_condition, round) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup()

p5 <- ggplot(switching,
             aes(x = round, y = proportion,
                 color = letter_group, group = letter_group)) +
  geom_line(linewidth = 1.5) +
  geom_point(size = 3) +
  facet_wrap(~ partner_condition, ncol = 2) +
  geom_vline(xintercept = 7.5, color = "red", linetype = "dashed", linewidth = 1, alpha = 0.6) +
  scale_x_continuous(breaks = 1:14) +
  scale_y_continuous(labels = percent, limits = c(0, 1),
                     breaks = seq(0, 1, 0.25)) +
  scale_color_manual(
    values = c("J" = "#4472C4", "M" = "#ED7D31"),
    labels = c("J" = "J (10 pts)", "M" = "M (15 pts in R8+)")
  ) +
  labs(
    title    = "Letter Choice Proportions by Round and Partner Condition",
    subtitle = "Vertical line indicates shift from rounds 1-7 (J=10) to rounds 8-14 (J=10, M=15, N=15)",
    x        = "Round",
    y        = "Proportion of Letter Choice",
    color    = "Letter"
  ) +
  panel_theme

print(p5)


# ==============================================================================
# 8. SUMMARY STATISTICS
# ==============================================================================

cat("\n================================================================================\n")
cat("8. SUMMARY STATISTICS\n")
cat("================================================================================\n")

rounds_1_7  <- df %>% filter(round <= 7)
rounds_8_14 <- df %>% filter(round >= 8)
most_common <- names(sort(table(df_long$choice), decreasing = TRUE))[1]

cat(sprintf("
  Total trials:              %d
  Total rounds:              %d
  Overall match rate:        %.1f%%
  Match rate rounds 1-7:     %.1f%%
  Match rate rounds 8-14:    %.1f%%
  Most common letter:        %s
  Average final score:       %.1f
  Optimal score achieved:    %s
",
  n_distinct(df$trial),
  nrow(df),
  mean(df$matched) * 100,
  mean(rounds_1_7$matched) * 100,
  mean(rounds_8_14$matched) * 100,
  most_common,
  mean(final_scores$avg_score),
  if_else(any(final_scores$avg_score >= max_optimal), "Yes", "No")
))

cat("================================================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("================================================================================\n")
