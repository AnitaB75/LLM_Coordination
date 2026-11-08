## =====================================================================
## Simple: how often does each condition defect / switch?
## Only gpt-5.5 at 4x has n>1, so only it can carry a likelihood.
##
##   Rscript fig_simple.R
## =====================================================================
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(ggplot2); library(stringr)
})

RES  <- "/Users/calypsocalcott/Desktop/Coordination/LLM_Coordination/Name_game_experiment/results"
OUTS <- c("/Users/calypsocalcott/Desktop/Coordination/LLM_Coordination/fig_simple.png",
          "/Users/calypsocalcott/Downloads/fig_simple.png")

COL <- c("Someone defects" = "#eb6834", "Whole group switches" = "#2a78d6")
INK <- "#1a1a1a"; INK2 <- "#5b5a56"; GRID <- "#e4e3df"

files <- list.files(RES, pattern = "^name_game_entrenchment_singleM_gpt55_j[0-9]+_.*\\.csv$",
                    full.names = TRUE)
raw <- files |> lapply(read_csv, show_col_types = FALSE) |> bind_rows()

games <- raw |>
  filter(phase == "post_shift") |>
  pivot_longer(c(p1,p2,p3,p4,p5,p6), names_to = "player", values_to = "letter") |>
  group_by(condition, trial, round, matched) |>
  summarise(anyM = any(letter == "M"), allM = all(letter == "M"), .groups = "drop") |>
  group_by(condition, trial) |>
  summarise(defect = any(anyM), switch = any(allM & matched), .groups = "drop")

dat <- games |>
  group_by(condition) |>
  summarise(n = n(), `Someone defects` = mean(defect),
            `Whole group switches` = mean(switch), .groups = "drop") |>
  pivot_longer(-c(condition, n), names_to = "outcome", values_to = "p") |>
  mutate(
    condition = factor(c(j7 = "7 rounds", j4 = "4 rounds", j1 = "1 round")[condition],
                       levels = c("7 rounds", "4 rounds", "1 round")),
    outcome   = factor(outcome, levels = names(COL))
  )

p <- ggplot(dat, aes(condition, p, fill = outcome)) +
  geom_col(position = position_dodge(width = .72), width = .62) +
  geom_text(aes(label = scales::percent(p, accuracy = 1)),
            position = position_dodge(width = .72), vjust = -0.6,
            size = 3.6, fontface = "bold", colour = INK) +
  scale_fill_manual(values = COL, name = NULL) +
  scale_y_continuous(limits = c(0, 1.06), breaks = seq(0, 1, .25),
                     labels = scales::percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, .02))) +
  labs(
    title = "gpt-5.5 at a 4× payoff: how often the convention breaks",
    subtitle = paste0("10 games per condition. ",
                      "gpt-5.4 and claude-opus-4-8 never switched, and neither did gpt-5.5 at\n",
                      "1.5× or 2×, but those cells were run once each, so they cannot give a rate."),
    x = "rounds the convention ran before the payoff change",
    y = "share of games"
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title       = element_text(face = "bold", size = 12.5, colour = INK),
    plot.subtitle    = element_text(size = 8.5, colour = INK2, lineheight = 1.3,
                                    margin = margin(t = 4, b = 10)),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(linewidth = .25, colour = GRID),
    panel.border     = element_rect(colour = "#c9c8c3", linewidth = .4),
    axis.title       = element_text(size = 9, colour = INK2),
    axis.text        = element_text(size = 10, colour = INK2),
    legend.position  = "top",
    legend.text      = element_text(size = 9.5, colour = INK),
    legend.margin    = margin(b = -4),
    plot.margin      = margin(12, 16, 10, 12)
  )

print(p)
for (o in OUTS) { ggsave(o, p, width = 6.8, height = 4.4, dpi = 300, bg = "white")
                  cat("wrote", o, "\n") }
