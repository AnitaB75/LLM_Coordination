## =====================================================================
## Two panels, one per model. Each shows all three convention lengths
## crossed with all three payoff levels.
##
##   Rscript fig_two_panel.R
## =====================================================================
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(ggplot2); library(stringr)
})

HERE <- "/Users/calypsocalcott/Desktop/Coordination/LLM_Coordination/Name_game_experiment"
RES  <- file.path(HERE, "results")
OUTS <- c("/Users/calypsocalcott/Desktop/Coordination/LLM_Coordination/fig_entrenchment_2panel.png",
          "/Users/calypsocalcott/Downloads/fig_entrenchment_2panel.png")

## validated: CVD ΔE 24.7, normal-vision 33.6, all >= 3:1 on the light surface
COL <- c("7 rounds" = "#2a78d6", "4 rounds" = "#eb6834", "1 round" = "#4a3aa7")
INK <- "#1a1a1a"; INK2 <- "#5b5a56"; GRID <- "#e4e3df"

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

raw <- list.files(RES, pattern = "^name_game_entrenchment_.*\\.csv$", full.names = TRUE) |>
  lapply(load_one) |> bind_rows() |>
  group_by(model, variant, mn_value) |> filter(stamp == max(stamp)) |> ungroup()

dat <- raw |>
  filter(variant == "singleM", phase == "post_shift") |>          # the complete 3x3 sweep
  pivot_longer(c(p1,p2,p3,p4,p5,p6), names_to = "player", values_to = "letter") |>
  mutate(chose_high = str_detect(high_letters, fixed(letter))) |>
  group_by(model, mn_value, condition) |>
  summarise(adopt = mean(chose_high), n = n(), .groups = "drop") |>
  mutate(
    payoff = factor(mn_value, levels = c(15, 20, 40),
                    labels = c("1.5×\n(M=15)", "2×\n(M=20)", "4×\n(M=40)")),
    `convention length` = factor(c(j7 = "7 rounds", j4 = "4 rounds", j1 = "1 round")[condition],
                                 levels = names(COL))
  )

dg <- position_dodge(width = .45)   # separate the overlapping zeros

p <- ggplot(dat, aes(payoff, adopt,
                     colour = `convention length`, group = `convention length`)) +
  geom_line(position = dg, linewidth = .6) +
  geom_point(position = dg, size = 2.4) +
  geom_text(data = subset(dat, adopt > 0.03),
            aes(label = scales::percent(adopt, accuracy = 1)),
            position = dg, vjust = -1.1, size = 3, fontface = "bold",
            show.legend = FALSE) +
  facet_wrap(~ model, nrow = 1) +
  scale_colour_manual(values = COL, name = "J-only rounds before the change:  ") +
  scale_y_continuous(limits = c(-.05, 1.02), breaks = seq(0, 1, .25),
                     labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Only gpt-5.5, and only at a 4× payoff, left the established convention",
    subtitle = paste0("Share of post-shift choices on the new, higher-paying option. Six agents, no communication;\n",
                      "the payoff change is unannounced. Partners described as human. n = 1 game per point."),
    x = "payoff of the new option, relative to the convention (J = 10)",
    y = "post-shift choices on the new option"
  ) +
  theme_bw(base_size = 10) +
  theme(
    plot.title       = element_text(face = "bold", size = 12, colour = INK),
    plot.subtitle    = element_text(size = 8.5, colour = INK2, lineheight = 1.3,
                                    margin = margin(t = 4, b = 10)),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(linewidth = .25, colour = GRID),
    panel.border     = element_rect(colour = "#c9c8c3", linewidth = .4),
    strip.background = element_rect(fill = "#f2f1ed", colour = "#c9c8c3", linewidth = .4),
    strip.text       = element_text(size = 10, face = "bold", colour = INK,
                                    margin = margin(4, 4, 4, 4)),
    axis.title       = element_text(size = 9, colour = INK2),
    axis.text        = element_text(size = 8.5, colour = INK2),
    legend.position  = "top",
    legend.title     = element_text(size = 9, colour = INK2),
    legend.text      = element_text(size = 9, colour = INK),
    legend.margin    = margin(b = -2),
    panel.spacing    = unit(1.2, "lines"),
    plot.margin      = margin(12, 16, 10, 12)
  )

print(p)
cat("\n"); print(as.data.frame(dat |> select(model, payoff, `convention length`, n, adopt)))

for (o in OUTS) {
  ggsave(o, p, width = 8.2, height = 4.6, dpi = 300, bg = "white")
  cat("wrote", o, "\n")
}
