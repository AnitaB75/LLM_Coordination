## =====================================================================
## Publication figure.
##
## (A) Adoption of the higher-paying option as a function of its payoff
##     ratio, by convention length and model.
## (B) Round-by-round adoption in the single cell where the convention
##     broke, showing the cascade.
##
##   Rscript fig_academic.R
## =====================================================================
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(ggplot2)
  library(stringr); library(patchwork)
})

HERE <- "/Users/calypsocalcott/Desktop/Coordination/LLM_Coordination/Name_game_experiment"
RES  <- file.path(HERE, "results")
OUTS <- c("/Users/calypsocalcott/Desktop/Coordination/LLM_Coordination/fig_entrenchment_academic.png",
          "/Users/calypsocalcott/Downloads/fig_entrenchment_academic.png")

## muted, print-safe; validated categorical slots 1 and 2
C54 <- "#2a78d6"; C55 <- "#eb6834"
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

long <- raw |>
  pivot_longer(c(p1,p2,p3,p4,p5,p6), names_to = "player", values_to = "letter") |>
  mutate(chose_high = str_detect(high_letters, fixed(letter)))

COND_LAB <- c(j7 = "7 rounds", j4 = "4 rounds", j1 = "1 round")

## ---------------- Panel A: adoption vs payoff ratio -------------------
## single-target arm only, where the payoff sweep is complete and the
## switch target is unambiguous
pa_dat <- long |>
  filter(variant == "singleM", phase == "post_shift") |>
  group_by(model, mn_value, condition) |>
  summarise(adopt = mean(chose_high), .groups = "drop") |>
  mutate(ratio = mn_value / 10,
         condition = factor(COND_LAB[condition], levels = unname(COND_LAB)))

pA <- ggplot(pa_dat, aes(ratio, adopt, colour = model, group = model)) +
  geom_hline(yintercept = 0, linewidth = .3, colour = GRID) +
  geom_line(linewidth = .55) +
  geom_point(size = 1.9) +
  facet_wrap(~ condition, nrow = 1) +
  scale_colour_manual(values = c("gpt-5.4" = C54, "gpt-5.5" = C55), name = NULL) +
  scale_x_continuous(breaks = c(1.5, 2, 4), labels = c("1.5×", "2×", "4×")) +
  scale_y_continuous(limits = c(-.03, 1), breaks = seq(0, 1, .25),
                     labels = scales::percent_format(accuracy = 1)) +
  labs(x = "payoff of new option, relative to the established convention",
       y = "post-shift choices\non the new option") +
  theme_bw(base_size = 9) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = .25, colour = GRID),
    panel.border     = element_rect(colour = "#c9c8c3", linewidth = .4),
    strip.background = element_rect(fill = "#f2f1ed", colour = "#c9c8c3", linewidth = .4),
    strip.text       = element_text(size = 8.5, colour = INK, margin = margin(3,3,3,3)),
    axis.title       = element_text(size = 8.5, colour = INK2),
    axis.text        = element_text(size = 8, colour = INK2),
    legend.position  = "top",
    legend.margin    = margin(b = -4),
    legend.text      = element_text(size = 8.5, colour = INK),
    legend.key.height= unit(.8, "lines")
  )

## ---------------- Panel B: the cascade --------------------------------
pb_dat <- long |>
  filter(model == "gpt-5.5", variant == "singleM", mn_value == 40, condition == "j7") |>
  group_by(round) |>
  summarise(adopt = mean(chose_high), matched = first(matched), .groups = "drop")

pB <- ggplot(pb_dat, aes(round, adopt)) +
  annotate("rect", xmin = 7.5, xmax = 10.5, ymin = -.03, ymax = 1,
           fill = "#f3d9cd", alpha = .55) +
  annotate("text", x = 9, y = .93, label = "3 rounds,\nno one scores",
           size = 2.6, colour = INK2, lineheight = .95) +
  geom_vline(xintercept = 7.5, linewidth = .4, linetype = "22", colour = INK2) +
  annotate("text", x = 7.42, y = .40, label = "payoff change", angle = 90,
           hjust = .5, vjust = -0.5, size = 2.6, colour = INK2) +
  geom_line(linewidth = .55, colour = C55) +
  geom_point(size = 1.9, colour = C55) +
  scale_x_continuous(breaks = 1:14) +
  scale_y_continuous(limits = c(-.03, 1), breaks = seq(0, 1, .25),
                     labels = scales::percent_format(accuracy = 1)) +
  labs(x = "round",
       y = "players choosing\nthe new option") +
  theme_bw(base_size = 9) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = .25, colour = GRID),
    panel.border     = element_rect(colour = "#c9c8c3", linewidth = .4),
    axis.title       = element_text(size = 8.5, colour = INK2),
    axis.text        = element_text(size = 8, colour = INK2)
  )

## ---------------- assemble --------------------------------------------
fig <- (pA / pB) +
  plot_annotation(
    tag_levels = "A",
    title = "Convention change requires a large payoff gap, and proceeds by cascade",
    caption = paste0(
      "Six agents repeatedly choose a letter and score only on unanimous agreement. An established convention (J, 10 points) is replaced\n",
      "mid-game by a higher-paying alternative (M), announced without warning after 7, 4 or 1 rounds. Partners are described as human.\n",
      "(A) Adoption is nil at 1.5× and 2× for both models; only gpt-5.5 at 4× departs, and only where the convention ran longest.\n",
      "(B) That departure is a cascade: 1, then 2, then 5 of 6 agents defect over three unrewarded rounds before all six converge on M.\n",
      "n = 1 game per cell. Not shown: claude-opus-4-8 and a two-target variant (M and N tied at 15), both 0% in all conditions."),
    theme = theme(
      plot.title    = element_text(size = 11, face = "bold", colour = INK,
                                   margin = margin(b = 8)),
      plot.caption  = element_text(size = 7.2, colour = INK2, hjust = 0,
                                   lineheight = 1.4, margin = margin(t = 10)),
      plot.tag      = element_text(size = 10, face = "bold", colour = INK)
    )
  ) & theme(plot.background = element_rect(fill = "white", colour = NA))

print(fig)
for (o in OUTS) {
  ggsave(o, fig, width = 7.4, height = 6.2, dpi = 300, bg = "white")
  cat("wrote", o, "\n")
}
