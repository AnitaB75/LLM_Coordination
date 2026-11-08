## =====================================================================
## Full design at n = 10: 2 models x 3 payoffs x 3 convention lengths.
##
## Cells are identified from COLUMNS INSIDE each csv (model, mn_value,
## condition), never from the filename -- early runs were written before
## the payoff was added to the filename, and globbing on the name
## silently dropped them.
##
##   Rscript fig_full_grid.R
## =====================================================================
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(ggplot2); library(stringr)
})

RES  <- "/Users/calypsocalcott/Desktop/Coordination/LLM_Coordination/Name_game_experiment/results"
OUTS <- c("/Users/calypsocalcott/Desktop/Coordination/LLM_Coordination/fig_full_grid.png",
          "/Users/calypsocalcott/Downloads/fig_full_grid.png")

COL <- c("Someone defects" = "#eb6834", "Whole group switches" = "#2a78d6")
INK <- "#1a1a1a"; INK2 <- "#5b5a56"; GRID <- "#e4e3df"

files <- list.files(RES, pattern = "^name_game_entrenchment_singleM_gpt5.*_j[0-9]+.*\\.csv$",
                    full.names = TRUE)
raw <- files |> lapply(read_csv, show_col_types = FALSE) |> bind_rows()

## unit of analysis = the game
games <- raw |>
  filter(phase == "post_shift") |>
  pivot_longer(c(p1,p2,p3,p4,p5,p6), names_to = "player", values_to = "letter") |>
  group_by(model, mn_value, condition, trial, round, matched) |>
  summarise(anyM = any(letter == "M"), allM = all(letter == "M"), .groups = "drop") |>
  group_by(model, mn_value, condition, trial) |>
  summarise(defect = any(anyM), switch = any(allM & matched), .groups = "drop")

dat <- games |>
  group_by(model, mn_value, condition) |>
  summarise(n = n(), `Someone defects` = mean(defect),
            `Whole group switches` = mean(switch), .groups = "drop") |>
  pivot_longer(-c(model, mn_value, condition, n),
               names_to = "outcome", values_to = "p") |>
  mutate(
    condition = factor(c(j7="7 rounds", j4="4 rounds", j1="1 round")[condition],
                       levels = c("7 rounds","4 rounds","1 round")),
    payoff    = factor(mn_value, levels = c(15,20,40),
                       labels = c("1.5× payoff", "2× payoff", "4× payoff")),
    outcome   = factor(outcome, levels = names(COL))
  )

stopifnot("expected 18 cells x 2 outcomes" = nrow(dat) == 36)
stopifnot("every cell should have n = 10" = all(dat$n == 10))

p <- ggplot(dat, aes(condition, p, fill = outcome)) +
  geom_col(position = position_dodge(width = .74), width = .64) +
  geom_text(aes(label = ifelse(p > 0, scales::percent(p, accuracy = 1), "0")),
            position = position_dodge(width = .74), vjust = -0.5,
            size = 2.9, fontface = "bold", colour = INK) +
  facet_grid(model ~ payoff) +
  scale_fill_manual(values = COL, name = NULL) +
  scale_y_continuous(limits = c(0, 1.12), breaks = seq(0, 1, .5),
                     labels = scales::percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, .02))) +
  labs(
    title = "Switch rates across precedent strength and payoff",
    subtitle = paste0("10 independent games per cell (180 games). Six agents per game, no communication, ",
                      "partners described as human;\nthe payoff change is unannounced. ",
                      "gpt-5.4 defected in 1 of its 90 games and never completed a switch."),
    x = "rounds the convention ran before the payoff change",
    y = "share of games"
  ) +
  theme_bw(base_size = 10) +
  theme(
    plot.title       = element_text(face = "bold", size = 12.5, colour = INK),
    plot.subtitle    = element_text(size = 8.5, colour = INK2, lineheight = 1.3,
                                    margin = margin(t = 4, b = 10)),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(linewidth = .25, colour = GRID),
    panel.border     = element_rect(colour = "#c9c8c3", linewidth = .4),
    strip.background = element_rect(fill = "#f2f1ed", colour = "#c9c8c3", linewidth = .4),
    strip.text       = element_text(size = 9.5, face = "bold", colour = INK,
                                    margin = margin(4,4,4,4)),
    axis.title       = element_text(size = 9, colour = INK2),
    axis.text        = element_text(size = 8.5, colour = INK2),
    legend.position  = "top",
    legend.text      = element_text(size = 9.5, colour = INK),
    legend.margin    = margin(b = -4),
    panel.spacing    = unit(.9, "lines"),
    plot.margin      = margin(12, 16, 10, 12)
  )

print(p)
cat("\n=== all 18 cells, n = 10 each ===\n")
print(as.data.frame(dat |> select(model, payoff, condition, outcome, n, p) |>
                    pivot_wider(names_from = outcome, values_from = p)))
for (o in OUTS) { ggsave(o, p, width = 9.2, height = 5.4, dpi = 300, bg = "white")
                  cat("wrote", o, "\n") }
