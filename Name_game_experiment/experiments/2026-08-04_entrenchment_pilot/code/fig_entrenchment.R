## =====================================================================
## Entrenchment pilot -- headline figure.
## Share of the 6 players choosing the old convention (J) vs the new
## higher-paying options (M/N), by round, for each condition and model.
## The vertical rule marks the UNANNOUNCED payoff change.
##
##   Rscript fig_entrenchment.R
## =====================================================================
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(ggplot2); library(stringr)
})

HERE <- "/Users/calypsocalcott/Desktop/Coordination/LLM_Coordination/Name_game_experiment"
RES  <- file.path(HERE, "results")

OUTS <- c("/Users/calypsocalcott/Desktop/Coordination/LLM_Coordination/fig_entrenchment_human.png",
          "/Users/calypsocalcott/Downloads/fig_entrenchment_human.png")

## --- palette (validated: CVD ΔE 24.7, normal 33.6, both >= 3:1 on surface) ---
LAB_J   <- "J  (old convention, 10 pts)"
LAB_NEW <- "new high-value option (15 pts)"

COL_J  <- "#2a78d6"   # categorical slot 1
COL_MN <- "#eb6834"   # categorical slot 2
INK    <- "#0b0b0b"
INK2   <- "#52514e"
SURF   <- "#fcfcfb"

## --- load newest run per provider ------------------------------------
files <- list.files(RES, pattern = "^name_game_entrenchment_.*\\.csv$", full.names = TRUE)
stopifnot("no entrenchment CSVs found" = length(files) > 0)

## Group by what the FILE says, not by its name -- filenames have changed shape
## across variants. Legacy files (pre-variant) are the M/N arm by definition.
load_one <- function(p) {
  d <- read_csv(p, show_col_types = FALSE)
  if (!"variant" %in% names(d))      d$variant <- "MN"
  if (!"high_letters" %in% names(d)) d$high_letters <- "M/N"
  if (!"mn_value" %in% names(d))     d$mn_value <- 15    # pre-sweep runs were all 15
  d$high_letters <- tidyr::replace_na(d$high_letters, "M/N")
  d$mn_value     <- tidyr::replace_na(as.numeric(d$mn_value), 15)
  d$stamp <- str_match(basename(p), "_(\\d{8}_\\d{6})\\.csv$")[, 2]
  d
}

## Deduplicate at the level of an actual experimental CELL. Grouping only by
## (variant, model) would silently drop a payoff level collected in a separate
## file -- e.g. a sweep resumed after a crash.
raw <- files |> lapply(load_one) |> bind_rows() |>
  group_by(variant, model, mn_value) |> filter(stamp == max(stamp)) |> ungroup() |>
  mutate(variant = factor(variant, levels = c("MN", "singleM"),
                          labels = c("two targets (M and N, 15 each)",
                                     "one target (M only, 15)")))

## --- long: one row per player-round ----------------------------------
long <- raw |>
  pivot_longer(c(p1, p2, p3, p4, p5, p6), names_to = "player", values_to = "letter") |>
  mutate(
    ## `high_letters` is "M/N" or "M" depending on the arm, so N counts as a
    ## high-value choice ONLY in the two-target arm (it is worth 0 in the other).
    option = case_when(letter == "J" ~ LAB_J,
                       str_detect(high_letters, fixed(letter)) ~ LAB_NEW,
                       TRUE ~ "other (0 pts)"),
    condition = factor(condition, levels = c("j7", "j4", "j1"),
                       labels = c("7 J-only rounds\n(original design)",
                                  "4 J-only rounds",
                                  "1 J-only round")),
    model = factor(model, levels = c("gpt-5.4", "claude-opus-4-8"))
  )

## keep the two options of interest so a flat 0% line is drawn for M/N
shares <- long |>
  count(variant, model, condition, round, option) |>
  group_by(variant, model, condition, round) |>
  mutate(share = n / sum(n)) |> ungroup() |>
  complete(nesting(variant, model, condition, round),
           option = c(LAB_J, LAB_NEW),
           fill = list(n = 0, share = 0)) |>
  filter(option != "other (0 pts)")

shift <- raw |> distinct(variant, model, condition, shift_round) |>
  mutate(condition = factor(condition, levels = c("j7", "j4", "j1"),
                            labels = levels(shares$condition)),
         model = factor(model, levels = c("gpt-5.4", "claude-opus-4-8")))

## direct labels at the end of each line (identity never by color alone)
ends <- shares |> group_by(variant, model, condition, option) |>
  slice_max(round, n = 1) |> ungroup()

p <- ggplot(shares, aes(round, share, colour = option)) +
  geom_vline(data = shift, aes(xintercept = shift_round - 0.5),
             linetype = "22", linewidth = .5, colour = INK2) +
  geom_text(data = shift, aes(x = shift_round - 0.5, y = .52, label = "payoff change"),
            inherit.aes = FALSE, angle = 90, vjust = -0.4, hjust = 0.5,
            size = 2.7, colour = INK2) +
  geom_line(linewidth = .7) +
  geom_point(size = 1.5) +
  geom_text(data = ends, aes(label = scales::percent(share, accuracy = 1)),
            hjust = -0.3, size = 3, fontface = "bold", show.legend = FALSE) +
  facet_grid(model + variant ~ condition, scales = "free_x", space = "free_x") +
  coord_cartesian(clip = "off") +
  scale_colour_manual(values = setNames(c(COL_J, COL_MN), c(LAB_J, LAB_NEW))) +
  scale_x_continuous(breaks = seq(1, 14, 1),
                     expand = expansion(mult = c(.05, 0), add = c(0, .35))) +
  scale_y_continuous(labels = scales::percent, limits = c(-.04, 1.08),
                     breaks = c(0, .5, 1)) +
  labs(
    title = "The old convention survived every manipulation",
    subtitle = paste0("Share of the 6 players choosing each option. The payoff change is unannounced until it happens; afterwards the\n",
                      "new option pays 15 vs J's 10. Shortening the convention to a single round did not induce switching, and neither\n",
                      "did removing the M-vs-N tie so the switch target was unique. Human partner framing, n = 1 game per cell."),
    x = "round", y = "share of players", colour = NULL,
    caption = paste("Name Game - convention entrenchment pilot - 2026-08-04.",
                    "claude x one-target arm is missing: that run halted on an API credit error.")
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.background   = element_rect(fill = SURF, colour = NA),
    panel.background  = element_rect(fill = SURF, colour = NA),
    plot.title        = element_text(face = "bold", size = 14, colour = INK),
    plot.subtitle     = element_text(size = 9, colour = INK2, lineheight = 1.25,
                                     margin = margin(t = 4, b = 12)),
    plot.caption      = element_text(size = 7.5, colour = INK2, hjust = 0),
    axis.title        = element_text(size = 9, colour = INK2),
    axis.text         = element_text(size = 8, colour = INK2),
    strip.text        = element_text(size = 9, face = "bold", colour = INK),
    strip.text.y      = element_text(angle = 0),
    panel.grid.minor  = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(linewidth = .3, colour = "#e6e5e1"),
    panel.spacing.x   = unit(3.2, "lines"),   # room for the overflowing end labels
    panel.spacing.y   = unit(1.1, "lines"),
    plot.margin       = margin(12, 34, 8, 12),
    legend.position   = "top",
    legend.text       = element_text(size = 9, colour = INK),
    legend.key.width  = unit(1.4, "lines")
  )

print(p)

for (o in OUTS) {
  dir.create(dirname(o), showWarnings = FALSE, recursive = TRUE)
  ggsave(o, p, width = 12, height = 8, dpi = 200, bg = SURF)
  cat("wrote", o, "\n")
}
