## =====================================================================
## Summary of every entrenchment / payoff pilot run so far.
##
## One tile per experimental cell: rows are the arms we tried (model x how
## the new option was structured), columns are how many J-only rounds came
## before the unannounced payoff change. The tile reports the share of
## POST-SHIFT choices that went to the new high-value option.
##
##   Rscript fig_pilot_summary.R
## =====================================================================
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(ggplot2); library(stringr)
})

HERE <- "/Users/calypsocalcott/Desktop/Coordination/LLM_Coordination/Name_game_experiment"
RES  <- file.path(HERE, "results")
OUTS <- c("/Users/calypsocalcott/Desktop/Coordination/LLM_Coordination/fig_pilot_summary.png",
          "/Users/calypsocalcott/Downloads/fig_pilot_summary.png")

INK  <- "#0b0b0b"; INK2 <- "#52514e"; SURF <- "#fcfcfb"
COL_STUCK <- "#2a78d6"   # categorical slot 1 - "stayed on J"
COL_MOVED <- "#eb6834"   # categorical slot 2 - reserved for any switching

load_one <- function(p) {
  d <- read_csv(p, show_col_types = FALSE)
  if (!"variant" %in% names(d))      d$variant      <- "MN"
  if (!"high_letters" %in% names(d)) d$high_letters <- "M/N"
  if (!"mn_value" %in% names(d))     d$mn_value     <- 15
  d$high_letters <- tidyr::replace_na(d$high_letters, "M/N")
  d$mn_value     <- tidyr::replace_na(as.numeric(d$mn_value), 15)
  d$stamp <- str_match(basename(p), "_(\\d{8}_\\d{6})\\.csv$")[, 2]
  d
}

files <- list.files(RES, pattern = "^name_game_entrenchment_.*\\.csv$", full.names = TRUE)
raw <- files |> lapply(load_one) |> bind_rows() |>
  group_by(model, variant, mn_value) |> filter(stamp == max(stamp)) |> ungroup()

## post-shift choices only; a choice "moved" if it is one of that arm's high letters
cells <- raw |>
  filter(phase == "post_shift") |>
  pivot_longer(c(p1,p2,p3,p4,p5,p6), names_to = "player", values_to = "letter") |>
  mutate(moved = str_detect(high_letters, fixed(letter))) |>
  group_by(model, variant, high_letters, mn_value, condition) |>
  summarise(n_choices = n(), pct_moved = mean(moved), .groups = "drop") |>
  mutate(
    arm = sprintf("%s   %s = %d", model, high_letters, mn_value),
    condition = factor(condition, levels = c("j7","j4","j1"),
                       labels = c("7 J-only rounds\n(original)", "4 J-only rounds", "1 J-only round"))
  )

## order arms: model, then structure, then payoff
arm_order <- cells |> distinct(model, high_letters, mn_value, arm) |>
  arrange(model, desc(high_letters), mn_value) |> pull(arm)
cells$arm <- factor(cells$arm, levels = rev(arm_order))

## fill in the cells that have not been collected yet
grid <- expand_grid(arm = factor(arm_order, levels = rev(arm_order)),
                    condition = levels(cells$condition) |> factor(levels = levels(cells$condition)))
cells <- grid |> left_join(cells, by = c("arm","condition"))

tot <- sum(cells$n_choices, na.rm = TRUE)
n_cells <- sum(!is.na(cells$n_choices))

p <- ggplot(cells, aes(condition, arm)) +
  geom_tile(aes(fill = ifelse(is.na(pct_moved), NA, pct_moved)),
            colour = SURF, linewidth = 2.5) +
  geom_text(aes(label = ifelse(is.na(pct_moved), "not run",
                               scales::percent(pct_moved, accuracy = 1))),
            colour = "white", fontface = "bold", size = 5) +
  geom_text(aes(label = ifelse(is.na(n_choices), "", paste0(n_choices, " choices"))),
            nudge_y = -0.26, colour = "white", size = 2.9, alpha = .85) +
  scale_fill_gradient(low = COL_STUCK, high = COL_MOVED,
                      limits = c(0, 1), na.value = "#d9d8d3", guide = "none") +
  labs(
    title = "One cell broke the deadlock: gpt-5.5, at 4x, with the longest-running convention",
    subtitle = paste0(
      "Each tile is one game: the share of post-shift choices that went to the new, higher-paying option.\n",
      n_cells, " games, ", format(tot, big.mark = ","),
      " post-shift choices. 22 of 24 cells are exactly 0%. The exception is gpt-5.5 at M=40, where a\n",
      "3-round cascade of failed coordination ended in a full switch (76% of that game's post-shift choices).\n",
      "Note it is the 7-round convention that broke, not the 1-round one - entrenchment did not protect J."),
    x = NULL, y = NULL,
    caption = "Name Game - entrenchment & payoff pilots - n = 1 game per cell"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.background  = element_rect(fill = SURF, colour = NA),
    panel.background = element_rect(fill = SURF, colour = NA),
    plot.title       = element_text(face = "bold", size = 15, colour = INK),
    plot.subtitle    = element_text(size = 9.5, colour = INK2, lineheight = 1.35,
                                    margin = margin(t = 6, b = 14)),
    plot.caption     = element_text(size = 8, colour = INK2, hjust = 0),
    axis.text.x      = element_text(size = 9.5, colour = INK, face = "bold"),
    axis.text.y      = element_text(size = 9.5, colour = INK, family = "mono"),
    panel.grid       = element_blank(),
    plot.margin      = margin(14, 18, 10, 12)
  )

print(p)
cat("\ncells:\n"); print(as.data.frame(cells |> select(arm, condition, n_choices, pct_moved)))

for (o in OUTS) {
  ggsave(o, p, width = 10.5, height = 6.5, dpi = 200, bg = SURF)
  cat("wrote", o, "\n")
}
