###############################################################################
# Name Game — human round-9 prediction study: clean + visualize
#
# Reads the Qualtrics export from ~/Downloads, cleans it (drops previews,
# coalesces the between-subjects condition blocks, scores comprehension),
# and produces the key visualizations:
#   1. Proportion predicted to choose M, by round-8 condition
#   2. Proportion who would THEMSELVES choose M, by condition
#   3. Reasoning: theme frequencies by condition + top words
# Bonus: overlays the gpt-5.4 pilot prediction if the pilot CSV is present.
#
# Design: between-subjects. Each participant saw ONE round-8 condition
#   0M (6J,0M) / 1M (5J,1M) / 2M (4J,2M) / 3M (3J,3M)
# and (a) predicted how many of 6 players choose J/M/N in round 9,
#     (b) said which letter they themselves would pick, (c) explained why.
#
# Question-ID map (from the labeled survey PDF):
#   block      counts(J/M/N)      reasoning  self-choice  likelihood(empty)
#   0M         Q2_1 /Q2_2 /Q2_3   Q21        Q24          Q29_4
#   1M         Q12_1/Q12_2/Q12_3  Q20        Q25          Q30_4
#   2M         Q13_1/Q13_2/Q13_3  Q19        Q26          Q31_4
#   3M         Q14_1/Q14_2/Q14_3  Q18        Q28          Q32_4
#   comprehension: Q5, Q15, Q17
###############################################################################

library(tidyverse)
# (titles/labels use plain ASCII hyphens for portability across locales)

# small helpers for per-facet bar ordering (avoids a tidytext dependency)
reorder_within <- function(x, by, within, sep = "___") {
  stats::reorder(paste(x, within, sep = sep), by)
}
scale_x_reordered <- function(..., sep = "___") {
  ggplot2::scale_x_discrete(labels = function(x) gsub(paste0(sep, ".+$"), "", x), ...)
}

# ---- locate the newest Qualtrics export in Downloads --------------------------
dl <- path.expand("~/Downloads")
csvs <- list.files(dl, pattern = "^Name_game_predictions.*\\.csv$", full.names = TRUE)
stopifnot(length(csvs) > 0)
DATA <- csvs[which.max(file.mtime(csvs))]
message("Reading: ", DATA)

# ---- read, skipping the 2 extra Qualtrics metadata header rows ---------------
hdr <- names(readr::read_csv(DATA, n_max = 0, show_col_types = FALSE))
raw <- readr::read_csv(DATA, skip = 3, col_names = hdr,
                       col_types = cols(.default = "c"))

# =============================================================================
# 1. CLEAN
# =============================================================================
N_PLAYERS <- 6
cond_levels <- c("0M", "1M", "2M", "3M")

clean <- raw %>%
  filter(Status != "Survey Preview", Finished == "True") %>%
  mutate(
    n_blocks = (!is.na(Q2_2)) + (!is.na(Q12_2)) + (!is.na(Q13_2)) + (!is.na(Q14_2)),
    condition = case_when(
      !is.na(Q2_2)  ~ "0M",
      !is.na(Q12_2) ~ "1M",
      !is.na(Q13_2) ~ "2M",
      !is.na(Q14_2) ~ "3M",
      TRUE ~ NA_character_
    ),
    pred_J = as.numeric(coalesce(Q2_1, Q12_1, Q13_1, Q14_1)),
    pred_M = as.numeric(coalesce(Q2_2, Q12_2, Q13_2, Q14_2)),
    pred_N = as.numeric(coalesce(Q2_3, Q12_3, Q13_3, Q14_3)),
    reasoning   = coalesce(Q21, Q20, Q19, Q18),
    self_choice = coalesce(Q24, Q25, Q26, Q28),
    # comprehension (correct-answer strings taken from the response set)
    comp_pass = (Q5  == "All players choose the same letter") &
                (Q15 == "All players choices") &
                (Q17 == "6 players who each choose one letter")
  ) %>%
  filter(!is.na(condition), n_blocks == 1) %>%       # one valid condition each
  mutate(
    condition  = factor(condition, levels = cond_levels),
    m_evidence = as.integer(str_remove(as.character(condition), "M")),  # 0..3
    pred_sum   = pred_J + pred_M + pred_N,
    pred_M_frac = pred_M / N_PLAYERS,
    self_M     = self_choice == "M"
  )

# data-quality notes
message("Completed responses: ", nrow(clean))
message("Comprehension failures: ", sum(!clean$comp_pass))
message("Predictions not summing to 6: ", sum(clean$pred_sum != N_PLAYERS, na.rm = TRUE))
print(count(clean, condition))

# Use comprehension-passers for the main analysis (drop the 1 failer). Swap to
# `clean` if you'd rather keep everyone.
df <- clean %>% filter(comp_pass)

# =============================================================================
# 2. SUMMARIES
# =============================================================================
mean_ci <- function(x) {
  x <- x[!is.na(x)]; m <- mean(x); se <- sd(x) / sqrt(length(x))
  tibble(est = m, lo = m - 1.96 * se, hi = m + 1.96 * se, n = length(x))
}
prop_ci <- function(x) {                       # Wald CI for a proportion
  x <- x[!is.na(x)]; p <- mean(x); n <- length(x); se <- sqrt(p * (1 - p) / n)
  tibble(est = p, lo = pmax(0, p - 1.96 * se), hi = pmin(1, p + 1.96 * se), n = n)
}

pred_summary <- df %>% group_by(condition, m_evidence) %>%
  reframe(mean_ci(pred_M_frac))
self_summary <- df %>% group_by(condition, m_evidence) %>%
  reframe(prop_ci(self_M))

cat("\n-- Proportion predicted to choose M (mean of predicted players / 6) --\n")
print(pred_summary)
cat("\n-- Proportion who would themselves choose M --\n")
print(self_summary)

# =============================================================================
# 3. VISUALIZE
# =============================================================================
pal <- "#E8820C"          # M = orange (matches the stimulus colour)
base_theme <- theme_minimal(base_size = 13) +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold"))

xlab_cond <- "Round-8 evidence (number of 6 players who chose M)"

## --- Plot 1: predicted proportion choosing M — Humans vs. gpt-5.4 vs Claude -
# load AI pilots (newest of each) so all sources sit on the SAME plot
pilot_dir <- path.expand("~/LLM_Coordination/Prediction_game/results")
source_levels <- c("Humans", "gpt-5.4", "Claude (opus-4-8)")
src_cols <- c("Humans" = pal, "gpt-5.4" = "#2E6FDB", "Claude (opus-4-8)" = "#8E44AD")

load_pilot <- function(pattern) {                # returns raw rows or NULL
  fs <- list.files(pilot_dir, pattern = pattern, full.names = TRUE)
  if (length(fs) == 0) return(NULL)
  readr::read_csv(fs[which.max(file.mtime(fs))], show_col_types = FALSE) %>%
    mutate(condition  = factor(str_remove(condition, "^r8_"), levels = cond_levels),
           m_evidence = as.integer(str_remove(as.character(condition), "M")),
           pred_M_frac = pred_M / N_PLAYERS)
}
pred_M_summary <- function(d) d %>% group_by(condition, m_evidence) %>% reframe(mean_ci(pred_M_frac))

gpt_raw    <- load_pilot("^pilot_round9_gpt54.*\\.csv$")
claude_raw <- load_pilot("^pilot_round9_claude.*\\.csv$")

pred_compare <- bind_rows(
  pred_summary %>% mutate(source = "Humans"),
  if (!is.null(gpt_raw))    pred_M_summary(gpt_raw)    %>% mutate(source = "gpt-5.4"),
  if (!is.null(claude_raw)) pred_M_summary(claude_raw) %>% mutate(source = "Claude (opus-4-8)")
) %>% mutate(source = factor(source, levels = source_levels))

dodge <- position_dodge(width = .8)
p1 <- ggplot(pred_compare, aes(condition, est, fill = source, group = source)) +
  geom_col(position = dodge, width = .72, alpha = .9) +
  geom_errorbar(aes(ymin = lo, ymax = hi), position = dodge, width = .14) +
  geom_text(aes(label = scales::percent(est, accuracy = 1)),
            position = dodge, vjust = -0.5, size = 2.8) +
  scale_y_continuous("Predicted proportion choosing M (round 9)",
                     labels = scales::percent, limits = c(0, 1)) +
  scale_fill_manual(values = src_cols) +
  labs(x = xlab_cond, fill = NULL,
       title = "Predicted switching to M: humans vs. gpt-5.4 vs. Claude",
       subtitle = "Mean of (predicted # players choosing M / 6), with 95% CI") +
  base_theme + theme(legend.position = "top")
print(p1)

## --- Plot 2: proportion who would THEMSELVES choose M -----------------------
p2 <- ggplot(self_summary, aes(condition, est)) +
  geom_col(fill = pal, alpha = .85, width = .65) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = .15) +
  geom_text(aes(label = scales::percent(est, accuracy = 1)), vjust = -0.6,
            nudge_y = .015, size = 4) +
  scale_y_continuous("Proportion who would choose M themselves",
                     labels = scales::percent, limits = c(0, 1)) +
  labs(x = xlab_cond,
       title = "Most participants would themselves choose M — even with little evidence",
       subtitle = "Self-reported choice for round 9 ('if you were player 1'), 95% CI") +
  base_theme
print(p2)

## --- Plot 2c: self-choice of M — Humans vs. Claude --------------------------
# (gpt-5.4 pilot had no self-choice item; Claude pilot does)
if (!is.null(claude_raw) && "self_choice" %in% names(claude_raw)) {
  claude_self <- claude_raw %>% group_by(condition, m_evidence) %>%
    reframe(prop_ci(self_choice == "M")) %>% mutate(source = "Claude (opus-4-8)")
  self_compare <- bind_rows(
    self_summary %>% mutate(source = "Humans"),
    claude_self
  ) %>% mutate(source = factor(source, levels = c("Humans", "Claude (opus-4-8)")))

  p2c <- ggplot(self_compare, aes(condition, est, fill = source, group = source)) +
    geom_col(position = dodge, width = .68, alpha = .9) +
    geom_errorbar(aes(ymin = lo, ymax = hi), position = dodge, width = .14) +
    geom_text(aes(label = scales::percent(est, accuracy = 1)),
              position = dodge, vjust = -0.5, size = 3) +
    scale_y_continuous("Proportion who would choose M themselves",
                       labels = scales::percent, limits = c(0, 1)) +
    scale_fill_manual(values = src_cols[c("Humans", "Claude (opus-4-8)")]) +
    labs(x = xlab_cond, fill = NULL,
         title = "Would choose M themselves: humans vs. Claude",
         subtitle = "Self-reported round-9 choice ('if you were player 1'), 95% CI") +
    base_theme + theme(legend.position = "top")
  print(p2c)
}

## --- Plot 2b: predicted-others vs self, overlaid (the key comparison) -------
combo <- bind_rows(
  pred_summary %>% mutate(measure = "Predicted for others"),
  self_summary %>% mutate(measure = "Would choose themselves")
)
p_combo <- ggplot(combo, aes(condition, est, color = measure, group = measure)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = .12) +
  scale_y_continuous("Proportion choosing M", labels = scales::percent,
                     limits = c(0, 1)) +
  scale_color_manual(values = c("Predicted for others" = "#5A6678",
                                "Would choose themselves" = pal)) +
  labs(x = xlab_cond, color = NULL,
       title = "Self choice of M exceeds predicted choice for others",
       subtitle = "Round-9 choice of M, by round-8 evidence (95% CI)") +
  base_theme + theme(legend.position = "top")
print(p_combo)

# =============================================================================
# 4. REASONING
# =============================================================================
reasons <- df %>%
  filter(!is.na(reasoning), str_trim(reasoning) != "") %>%
  mutate(txt = str_to_lower(reasoning))

## --- 4a: theme frequencies by condition ------------------------------------
themes <- c(
  "Higher payoff / switch to M" =
    "15|higher|more point|better|worth more|payoff|incentive|maximi|switch|move to m",
  "Stick with J / convention" =
    "stick|stay|keep j|safe|conventio|establish|familiar|habit|default|secure|comfort",
  "Coordination / matching risk" =
    "risk|coordinat|everyone|all (six|6|of them)|match|consensus|together|follow|same letter",
  "Gradual / uncertain" =
    "wait|gradual|slow|eventual|test|see if|hesitan|unsure|might|some of"
)
theme_df <- imap_dfr(themes, function(rx, nm) {
  reasons %>% group_by(condition) %>%
    summarise(prop = mean(str_detect(txt, rx)), n = n(), .groups = "drop") %>%
    mutate(theme = nm)
})
p3 <- ggplot(theme_df, aes(condition, prop, fill = theme)) +
  geom_col(position = position_dodge(.8), width = .75) +
  scale_y_continuous("Proportion of responses mentioning theme",
                     labels = scales::percent, limits = c(0, 1)) +
  scale_fill_brewer(palette = "Set2") +
  labs(x = xlab_cond, fill = NULL,
       title = "Reasoning themes by condition",
       subtitle = "Keyword-based; responses may mention more than one theme") +
  base_theme + theme(legend.position = "top") +
  guides(fill = guide_legend(nrow = 2))
print(p3)

## --- 4b: top words, split by whether they'd choose M ------------------------
stop_words <- c("the","a","an","and","or","to","of","in","is","it","that","they",
                "i","you","would","will","be","so","if","this","for","are","on",
                "as","with","but","at","by","not","have","has","their","them",
                "because","since","more","most","than","then","there","was","were",
                "do","does","my","me","we","also","all","just","like","get","one",
                "what","who","when","which","up","out","about","other","others",
                "round","players","player","point","points","choose","chose",
                "choosing","choice","letter","letters","everyone","people","think")
word_df <- reasons %>%
  filter(!is.na(self_M)) %>%
  transmute(self_M, w = str_replace_all(txt, "[^a-z ]", " ")) %>%
  separate_rows(w, sep = "\\s+") %>%
  filter(w != "", !(w %in% stop_words), str_length(w) >= 1) %>%
  mutate(group = if_else(self_M, "Would choose M", "Would not choose M")) %>%
  count(group, w, sort = TRUE) %>%
  group_by(group) %>% slice_max(n, n = 12, with_ties = FALSE) %>% ungroup()
p4 <- ggplot(word_df, aes(reorder_within(w, n, group), n, fill = group)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  facet_wrap(~ group, scales = "free") +
  scale_x_reordered() +
  scale_fill_manual(values = c("Would choose M" = pal,
                               "Would not choose M" = "#5A6678")) +
  labs(x = NULL, y = "Word count",
       title = "Most frequent words in reasoning, by self-choice") +
  base_theme
print(p4)

