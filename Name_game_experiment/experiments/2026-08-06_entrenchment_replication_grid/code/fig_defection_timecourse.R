## =====================================================================
## Defection over rounds, by condition (n = 10 games each).
##
## Aligned on rounds SINCE the payoff change, so the three conditions
## share an x-axis. Thin lines are individual games; the heavy line is
## the mean across the 10 games in that condition.
##
##   Rscript fig_defection_timecourse.R
## =====================================================================
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(ggplot2); library(stringr)
})

HERE <- "/Users/calypsocalcott/Desktop/Coordination/LLM_Coordination/Name_game_experiment"
RES  <- file.path(HERE, "results")
OUTS <- c("/Users/calypsocalcott/Desktop/Coordination/LLM_Coordination/fig_defection_timecourse.png",
          "/Users/calypsocalcott/Downloads/fig_defection_timecourse.png")

## validated triad: CVD ΔE 24.7, normal-vision 33.6, all >= 3:1 on light
COL <- c("7 rounds" = "#2a78d6", "4 rounds" = "#eb6834", "1 round" = "#4a3aa7")
INK <- "#1a1a1a"; INK2 <- "#5b5a56"; GRID <- "#e4e3df"

files <- list.files(RES, pattern = "^name_game_entrenchment_singleM_gpt55_j[0-9]+_.*\\.csv$",
                    full.names = TRUE)
stopifnot("no replication files found" = length(files) > 0)

raw <- files |> lapply(read_csv, show_col_types = FALSE) |> bind_rows()

per_game <- raw |>
  pivot_longer(c(p1,p2,p3,p4,p5,p6), names_to = "player", values_to = "letter") |>
  group_by(condition, trial, round_rel_shift) |>
  summarise(defect = mean(letter == "M"), .groups = "drop") |>
  filter(round_rel_shift >= 0) |>                      # 0 = last round before the change
  mutate(cond = factor(c(j7="7 rounds", j4="4 rounds", j1="1 round")[condition],
                       levels = names(COL)))

per_cond <- per_game |>
  group_by(cond, round_rel_shift) |>
  summarise(defect = mean(defect), .groups = "drop")

## label how many of the 10 games ever reached unanimity on M
lab <- raw |>
  pivot_longer(c(p1,p2,p3,p4,p5,p6), names_to="player", values_to="letter") |>
  group_by(condition, trial, round) |>
  summarise(allM = all(letter=="M"), matched = first(matched), .groups="drop") |>
  group_by(condition, trial) |> summarise(conv = any(allM & matched), .groups="drop") |>
  group_by(condition) |> summarise(k = sum(conv), n = n(), .groups="drop") |>
  mutate(cond = factor(c(j7="7 rounds", j4="4 rounds", j1="1 round")[condition],
                       levels = names(COL)),
         txt = sprintf("%d of %d games\nconverged on M", k, n))

p <- ggplot(per_game, aes(round_rel_shift, defect)) +
  geom_vline(xintercept = 0.5, linetype = "22", linewidth = .4, colour = INK2) +
  geom_line(aes(group = trial), colour = "grey72", linewidth = .35, alpha = .85) +
  geom_line(data = per_cond, aes(colour = cond), linewidth = 1.05) +
  geom_point(data = per_cond, aes(colour = cond), size = 1.8) +
  geom_text(data = lab, aes(x = 7, y = .05, label = txt),
            hjust = 1, vjust = 0, size = 2.9, colour = INK2, lineheight = .95) +
  facet_wrap(~ cond, nrow = 1) +
  scale_colour_manual(values = COL, guide = "none") +
  scale_x_continuous(breaks = 0:7,
                     labels = c("last\nJ-only", 1:7)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, .25),
                     labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Defection spreads gradually, and only sometimes completes",
    subtitle = paste0("Share of the 6 agents choosing the new option, by round since the unannounced payoff change.\n",
                      "Grey lines are individual games; coloured lines are the condition mean. ",
                      "gpt-5.5, new option worth 4x, n = 10 games per condition."),
    x = "rounds since the payoff change",
    y = "agents choosing the new option"
  ) +
  theme_bw(base_size = 10) +
  theme(
    plot.title       = element_text(face = "bold", size = 12, colour = INK),
    plot.subtitle    = element_text(size = 8.5, colour = INK2, lineheight = 1.3,
                                    margin = margin(t = 4, b = 10)),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = .25, colour = GRID),
    panel.border     = element_rect(colour = "#c9c8c3", linewidth = .4),
    strip.background = element_rect(fill = "#f2f1ed", colour = "#c9c8c3", linewidth = .4),
    strip.text       = element_text(size = 9.5, face = "bold", colour = INK,
                                    margin = margin(4,4,4,4)),
    axis.title       = element_text(size = 9, colour = INK2),
    axis.text        = element_text(size = 8, colour = INK2),
    panel.spacing    = unit(1.1, "lines"),
    plot.margin      = margin(12, 16, 10, 12)
  )

print(p)
cat("\nmean defection by condition x round:\n")
print(as.data.frame(per_cond |> pivot_wider(names_from = round_rel_shift,
                                            values_from = defect)))
for (o in OUTS) { ggsave(o, p, width = 8.6, height = 4.3, dpi = 300, bg = "white")
                  cat("wrote", o, "\n") }
