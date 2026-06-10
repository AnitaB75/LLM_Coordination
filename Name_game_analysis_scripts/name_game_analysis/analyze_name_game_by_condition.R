# =============================================================================
# NAME GAME COORDINATION ANALYSIS - BY PARTNER CONDITION
# Letter choices over rounds for each partner condition
# =============================================================================

# Load required libraries
library(tidyverse)
library(ggplot2)
library(gridExtra)
library(scales)

# Set working directory to this analysis folder
setwd("/Users/calypsocalcott/LLM_Coordination/name_game_analysis")

# Load the data
df_llm   <- read_csv("name_game_LLM_feedback_MN_20260507_134935.csv")
df_human <- read_csv("name_game_LLM_feedback_MN_HUMAN_20260507_212952.csv")
df <- bind_rows(df_llm, df_human)

# =============================================================================
# LETTER CHOICE BY CONDITION AND ROUND
# =============================================================================

# Create long-form dataset
df_long <- df %>%
  dplyr::select(trial, partner_condition, round, p1, p2, p3, p4, p5, p6) %>%
  pivot_longer(cols = p1:p6, names_to = "player", values_to = "choice")

# Calculate proportions of each letter by round and condition
letter_proportions <- df_long %>%
  group_by(partner_condition, round, choice) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(partner_condition, round) %>%
  mutate(
    total = sum(count),
    proportion = count / total
  ) %>%
  ungroup()

# =============================================================================
# VISUALIZATION 1: STACKED BAR CHART
# =============================================================================

p1 <- ggplot(letter_proportions,
             aes(x = round, y = proportion, fill = choice)) +
  geom_bar(stat = "identity", position = "stack") +
  facet_wrap(~partner_condition, ncol = 1) +
  geom_vline(xintercept = 7.5, color = "red", linetype = "dashed", alpha = 0.5) +
  labs(title = "Letter Choice Distribution by Partner Condition",
       subtitle = "Stacked proportions showing all player choices per round",
       x = "Round",
       y = "Proportion of Choices",
       fill = "Letter") +
  scale_fill_manual(values = c("J" = "#1f77b4", "M" = "#ff7f0e", "N" = "#2ca02c")) +
  scale_y_continuous(labels = percent) +
  scale_x_continuous(breaks = 1:14) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 16),
        plot.subtitle = element_text(size = 12),
        strip.text = element_text(face = "bold", size = 12),
        legend.position = "bottom")

ggsave("/Users/calypsocalcott/LLM_Coordination/name_game_analysis/letter_choice_by_condition_stacked.png",
       p1, width = 12, height = 10, dpi = 300)

cat("Saved stacked bar chart to: /Users/calypsocalcott/LLM_Coordination/name_game_analysis/letter_choice_by_condition_stacked.png\n")

# =============================================================================
# VISUALIZATION 2: LINE CHART
# =============================================================================

p2 <- ggplot(letter_proportions,
             aes(x = round, y = proportion, color = choice, group = choice)) +
  geom_line(size = 1.2) +
  geom_point(size = 2.5) +
  facet_wrap(~partner_condition, ncol = 1) +
  geom_vline(xintercept = 7.5, color = "red", linetype = "dashed", alpha = 0.5) +
  labs(title = "Letter Choice Trends by Partner Condition",
       subtitle = "Proportion of each letter choice over rounds",
       x = "Round",
       y = "Proportion of Choices",
       color = "Letter") +
  scale_color_manual(values = c("J" = "#1f77b4", "M" = "#ff7f0e", "N" = "#2ca02c")) +
  scale_y_continuous(labels = percent, limits = c(0, 1)) +
  scale_x_continuous(breaks = 1:14) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 16),
        plot.subtitle = element_text(size = 12),
        strip.text = element_text(face = "bold", size = 12),
        legend.position = "bottom")

ggsave("/Users/calypsocalcott/LLM_Coordination/name_game_analysis/letter_choice_by_condition_lines.png",
       p2, width = 12, height = 10, dpi = 300)

cat("Saved line chart to: /Users/calypsocalcott/LLM_Coordination/name_game_analysis/letter_choice_by_condition_lines.png\n")

# =============================================================================
# VISUALIZATION 3: SIDE-BY-SIDE COMPARISON
# =============================================================================

p3 <- ggplot(letter_proportions,
             aes(x = round, y = proportion, fill = choice)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~partner_condition, ncol = 3) +
  geom_vline(xintercept = 7.5, color = "red", linetype = "dashed", alpha = 0.5) +
  labs(title = "Letter Choice Distribution Across Partner Conditions",
       subtitle = "Side-by-side comparison of letter choices per round",
       x = "Round",
       y = "Proportion of Choices",
       fill = "Letter") +
  scale_fill_manual(values = c("J" = "#1f77b4", "M" = "#ff7f0e", "N" = "#2ca02c")) +
  scale_y_continuous(labels = percent) +
  scale_x_continuous(breaks = seq(2, 14, by = 2)) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 16),
        plot.subtitle = element_text(size = 12),
        strip.text = element_text(face = "bold", size = 12),
        legend.position = "bottom")

ggsave("/Users/calypsocalcott/LLM_Coordination/name_game_analysis/letter_choice_by_condition_sidebyside.png",
       p3, width = 15, height = 6, dpi = 300)

cat("Saved side-by-side comparison to: /Users/calypsocalcott/LLM_Coordination/name_game_analysis/letter_choice_by_condition_sidebyside.png\n")

# =============================================================================
# DETAILED BREAKDOWN TABLE
# =============================================================================

cat("\n================================================================================\n")
cat("DETAILED LETTER CHOICES BY CONDITION AND ROUND\n")
cat("================================================================================\n")

for (condition in unique(df$partner_condition)) {
  cat(sprintf("\n--- %s ---\n", toupper(condition)))

  condition_data <- df %>% filter(partner_condition == condition)

  for (r in 1:14) {
    round_data <- condition_data %>% filter(round == r)

    if (nrow(round_data) > 0) {
      choices <- unlist(round_data %>% dplyr::select(p1, p2, p3, p4, p5, p6))
      choice_table <- table(choices)
      matched <- all(round_data$matched)

      cat(sprintf("Round %2d: ", r))
      cat(paste(names(choice_table), "=", choice_table, collapse = ", "))
      cat(sprintf(" [%s]\n", ifelse(matched, "MATCHED", "NO MATCH")))
    }
  }
}

# =============================================================================
# SUMMARY STATISTICS BY CONDITION
# =============================================================================

cat("\n================================================================================\n")
cat("SUMMARY STATISTICS BY CONDITION\n")
cat("================================================================================\n")

summary_by_condition <- df %>%
  group_by(partner_condition) %>%
  summarise(
    trials = n_distinct(trial),
    total_rounds = n(),
    matched_rounds = sum(matched),
    match_rate = mean(matched),
    avg_final_score = mean(p1_total[round == 14]),
    switched_to_M_round8 = mean((p1 == "M" | p2 == "M" | p3 == "M" |
                                  p4 == "M" | p5 == "M" | p6 == "M")[round == 8]),
    .groups = "drop"
  )

print(summary_by_condition)

# =============================================================================
# SWITCHING ANALYSIS BY CONDITION
# =============================================================================

cat("\n================================================================================\n")
cat("ROUND 8 SWITCHING BEHAVIOR BY CONDITION\n")
cat("================================================================================\n")

round_8_switching <- df_long %>%
  filter(round == 8) %>%
  group_by(partner_condition, choice) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(partner_condition) %>%
  mutate(proportion = count / sum(count))

print(round_8_switching)

# Visualize Round 8 switching
p4 <- ggplot(round_8_switching, aes(x = partner_condition, y = proportion, fill = choice)) +
  geom_bar(stat = "identity", position = "stack") +
  labs(title = "Round 8 Letter Choices by Partner Condition",
       subtitle = "Critical decision point: stick with J or switch to M/N?",
       x = "Partner Condition",
       y = "Proportion of Choices",
       fill = "Letter") +
  scale_fill_manual(values = c("J" = "#1f77b4", "M" = "#ff7f0e", "N" = "#2ca02c")) +
  scale_y_continuous(labels = percent) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 11),
        legend.position = "bottom")

ggsave("/Users/calypsocalcott/LLM_Coordination/name_game_analysis/round8_switching_by_condition.png",
       p4, width = 10, height = 6, dpi = 300)

cat("\nSaved Round 8 switching plot to: /Users/calypsocalcott/LLM_Coordination/name_game_analysis/round8_switching_by_condition.png\n")

# =============================================================================
# VISUALIZATION 5: ROUND vs PROPORTION - THREE PANELS BY CONDITION
# =============================================================================

p5 <- ggplot(letter_proportions,
             aes(x = round, y = proportion, color = choice, group = choice)) +
  geom_line(size = 1.5) +
  geom_point(size = 3) +
  facet_wrap(~partner_condition, ncol = 2) +
  geom_vline(xintercept = 7.5, color = "red", linetype = "dashed",
             alpha = 0.5, size = 1) +
  labs(title = "Letter Choice Proportions by Round and Partner Condition",
       subtitle = "Vertical line indicates shift from rounds 1-7 (J=10) to rounds 8-14 (J=10, M=15, N=15)",
       x = "Round",
       y = "Proportion of Letter Choice",
       color = "Letter") +
  scale_color_manual(values = c("J" = "#1f77b4", "M" = "#ff7f0e", "N" = "#2ca02c"),
                     labels = c("J" = "J (10 pts)",
                                "M" = "M (15 pts in R8+)",
                                "N" = "N (15 pts in R8+)")) +
  scale_y_continuous(labels = percent, limits = c(0, 1),
                     breaks = seq(0, 1, 0.25)) +
  scale_x_continuous(breaks = seq(1, 14, by = 1)) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    strip.text = element_text(face = "bold", size = 13),
    strip.background = element_rect(fill = "lightgray", color = "black"),
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 12),
    legend.text = element_text(size = 11),
    axis.title = element_text(face = "bold", size = 12),
    axis.text = element_text(size = 10),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, size = 0.5)
  )

ggsave("/Users/calypsocalcott/LLM_Coordination/name_game_analysis/round_by_proportion_four_panels.png",
       p5, width = 14, height = 10, dpi = 300)

cat("\nSaved four-panel round × proportion plot to: /Users/calypsocalcott/LLM_Coordination/name_game_analysis/round_by_proportion_four_panels.png\n")

cat("\n================================================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("Files saved to Downloads folder:\n")
cat("  1. letter_choice_by_condition_stacked.png\n")
cat("  2. letter_choice_by_condition_lines.png\n")
cat("  3. letter_choice_by_condition_sidebyside.png\n")
cat("  4. round8_switching_by_condition.png\n")
cat("  5. round_by_proportion_four_panels.png [MAIN VISUALIZATION]\n")
cat("================================================================================\n")
