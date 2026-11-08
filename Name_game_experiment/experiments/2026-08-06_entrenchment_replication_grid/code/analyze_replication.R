## =====================================================================
## n=10 replication: gpt-5.5, M=40 (4x), three convention lengths.
##
## Per GAME (the independent unit -- not per choice, which is 6 correlated
## agents inside one game):
##   defected  = at least one agent ever chose M post-shift
##   converged = the group ever matched unanimously on M
##
## Wilson intervals, since proportions near 0/1 with n=10 break the normal
## approximation.
##
##   Rscript analyze_replication.R
## =====================================================================
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(ggplot2); library(stringr)
})

HERE <- "/Users/calypsocalcott/Desktop/Coordination/LLM_Coordination/Name_game_experiment"
RES  <- file.path(HERE, "results")
OUTS <- c("/Users/calypsocalcott/Desktop/Coordination/LLM_Coordination/fig_replication.png",
          "/Users/calypsocalcott/Downloads/fig_replication.png")

COL_DEF <- "#eb6834"   # a defector appeared
COL_CON <- "#2a78d6"   # the group converged
INK <- "#1a1a1a"; INK2 <- "#5b5a56"; GRID <- "#e4e3df"

files <- list.files(RES, pattern = "^name_game_entrenchment_singleM_gpt55_j[0-9]+_.*\\.csv$",
                    full.names = TRUE)
stopifnot("no replication files found" = length(files) > 0)

raw <- files |> lapply(read_csv, show_col_types = FALSE) |> bind_rows()

## one row per GAME
games <- raw |>
  filter(phase == "post_shift") |>
  pivot_longer(c(p1,p2,p3,p4,p5,p6), names_to = "player", values_to = "letter") |>
  group_by(condition, trial, round, matched) |>
  summarise(anyM = any(letter == "M"), allM = all(letter == "M"), .groups = "drop") |>
  group_by(condition, trial) |>
  summarise(
    defected     = any(anyM),
    converged    = any(allM & matched),
    failed_rounds= sum(!matched),
    .groups = "drop"
  )

wilson <- function(k, n, z = 1.96) {
  if (n == 0) return(c(NA, NA))
  p <- k / n; d <- 1 + z^2 / n
  c(( p + z^2/(2*n) - z * sqrt(p*(1-p)/n + z^2/(4*n^2)) ) / d,
    ( p + z^2/(2*n) + z * sqrt(p*(1-p)/n + z^2/(4*n^2)) ) / d)
}

summ <- games |>
  group_by(condition) |>
  summarise(n = n(), k_def = sum(defected), k_con = sum(converged),
            med_failed = median(failed_rounds), .groups = "drop") |>
  rowwise() |>
  mutate(
    p_def = k_def/n, def_lo = wilson(k_def, n)[1], def_hi = wilson(k_def, n)[2],
    p_con = k_con/n, con_lo = wilson(k_con, n)[1], con_hi = wilson(k_con, n)[2]
  ) |> ungroup() |>
  mutate(condition = factor(condition, levels = c("j7","j4","j1"),
                            labels = c("7 rounds","4 rounds","1 round")))

cat("=== per-game outcomes, gpt-5.5 at M=40 (4x) ===\n")
print(as.data.frame(summ |> select(condition, n, k_def, p_def, k_con, p_con, med_failed)))

## Fisher test on the key contrast the design hinges on
j7 <- summ |> filter(condition == "7 rounds"); j1 <- summ |> filter(condition == "1 round")
if (nrow(j7) && nrow(j1)) {
  m <- matrix(c(j7$k_con, j7$n - j7$k_con, j1$k_con, j1$n - j1$k_con), nrow = 2)
  ft <- fisher.test(m)
  cat(sprintf("\n7 rounds vs 1 round, convergence: %d/%d vs %d/%d, Fisher p = %.3f\n",
              j7$k_con, j7$n, j1$k_con, j1$n, ft$p.value))
  cat(sprintf("  odds ratio %.2f, 95%% CI [%.2f, %.2f]\n",
              ft$estimate, ft$conf.int[1], ft$conf.int[2]))
}

## ---------------- figure ----------------------------------------------
plot_dat <- bind_rows(
  summ |> transmute(condition, n, stat = "At least one agent defected",
                    p = p_def, lo = def_lo, hi = def_hi),
  summ |> transmute(condition, n, stat = "Group converged on the new option",
                    p = p_con, lo = con_lo, hi = con_hi)
) |>
  mutate(stat = factor(stat, levels = c("At least one agent defected",
                                        "Group converged on the new option")))

pd <- position_dodge(width = .55)
p <- ggplot(plot_dat, aes(condition, p, colour = stat, group = stat)) +
  geom_errorbar(aes(ymin = lo, ymax = hi), position = pd, width = .12, linewidth = .5) +
  geom_point(position = pd, size = 2.8) +
  scale_colour_manual(values = c("At least one agent defected" = COL_DEF,
                                 "Group converged on the new option" = COL_CON),
                      name = NULL) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, .25),
                     labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Defection is common; coordinated change is not",
    subtitle = paste0("gpt-5.5, new option worth 4x the convention, partners described as human. ",
                      "n = 10 games per condition;\nbars are 95% Wilson intervals. ",
                      "The unit is the game, not the choice."),
    x = "rounds the convention ran before the payoff change",
    y = "proportion of games"
  ) +
  theme_bw(base_size = 10) +
  theme(
    plot.title      = element_text(face = "bold", size = 12, colour = INK),
    plot.subtitle   = element_text(size = 8.5, colour = INK2, lineheight = 1.3,
                                   margin = margin(t = 4, b = 10)),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(linewidth = .25, colour = GRID),
    panel.border    = element_rect(colour = "#c9c8c3", linewidth = .4),
    axis.title      = element_text(size = 9, colour = INK2),
    axis.text       = element_text(size = 9, colour = INK2),
    legend.position = "top",
    legend.text     = element_text(size = 9, colour = INK),
    legend.margin   = margin(b = -2),
    plot.margin     = margin(12, 16, 10, 12)
  )

print(p)
for (o in OUTS) { ggsave(o, p, width = 7.2, height = 4.4, dpi = 300, bg = "white")
                  cat("wrote", o, "\n") }
