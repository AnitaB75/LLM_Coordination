## =====================================================================
## Convention entrenchment: does the length of the J-only phase predict
## whether players abandon J once M/N become worth more?
##
## Conditions differ ONLY in how many J-only rounds precede the shift
## (7 / 4 / 1). The post-shift phase is 7 rounds in every condition, so
## everything is plotted on ROUNDS SINCE THE SHIFT -- that puts the three
## conditions on a common x-axis and makes them directly comparable.
##
## The shift is UNANNOUNCED: players see only the J payoff until the rule
## update arrives (round 8 / 5 / 2). So pre-shift J-play is built in good
## faith, and post-shift behaviour is a genuine response to news.
##
##   Rscript analyze_entrenchment.R
## =====================================================================
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(ggplot2); library(stringr)
})

HERE <- "/Users/calypsocalcott/Desktop/Coordination/LLM_Coordination/Name_game_experiment"
RES  <- file.path(HERE, "results")

## --- load the newest entrenchment CSV per provider ------------------
files <- list.files(RES, pattern = "^name_game_entrenchment_.*\\.csv$", full.names = TRUE)
stopifnot("no entrenchment result CSVs found" = length(files) > 0)

newest_per_provider <- tibble(path = files) |>
  mutate(provider = str_match(basename(path), "entrenchment_([a-z]+)_")[, 2],
         stamp    = str_match(basename(path), "_(\\d{8}_\\d{6})\\.csv$")[, 2]) |>
  group_by(provider) |> slice_max(stamp, n = 1) |> ungroup()

cat("Using:\n"); print(newest_per_provider |> select(provider, path))

raw <- newest_per_provider$path |>
  lapply(read_csv, show_col_types = FALSE) |>
  bind_rows()

## --- long format: one row per player-round --------------------------
long <- raw |>
  pivot_longer(cols = c(p1, p2, p3, p4, p5, p6),
               names_to = "player", values_to = "letter") |>
  mutate(
    condition = factor(condition, levels = c("j7", "j4", "j1"),
                       labels = c("7 J-only rounds", "4 J-only rounds", "1 J-only round")),
    choice = case_when(letter == "J" ~ "J (old convention, 10)",
                       letter %in% c("M", "N") ~ "M/N (new, 15)",
                       TRUE ~ "other (0)")
  )

## --- panel A: share choosing each option, by rounds since shift -----
shares <- long |>
  count(model, condition, round_rel_shift, choice) |>
  group_by(model, condition, round_rel_shift) |>
  mutate(share = n / sum(n)) |> ungroup()

pA <- ggplot(shares, aes(round_rel_shift, share, colour = choice)) +
  geom_vline(xintercept = 0.5, linetype = "dashed", linewidth = .4, colour = "grey40") +
  geom_line(linewidth = .8) + geom_point(size = 1.6) +
  facet_grid(model ~ condition) +
  scale_x_continuous(breaks = seq(-6, 7, 2)) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_colour_manual(values = c("J (old convention, 10)" = "#1f77b4",
                                 "M/N (new, 15)"          = "#d62728",
                                 "other (0)"              = "grey65")) +
  labs(title = "Choice shares by rounds since the M/N payoff shift",
       subtitle = paste("x = 0 is the last J-only round; the dashed line marks the unannounced rule update.",
                        "Human framing, n = 1 game per cell."),
       x = "rounds since payoff shift", y = "share of the 6 players", colour = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "top", panel.grid.minor = element_blank())

print(pA)

## --- panel B: did the group actually coordinate, and on what? -------
group_state <- long |>
  group_by(model, condition, round, round_rel_shift, matched) |>
  summarise(letter = if (n_distinct(letter) == 1) first(letter) else NA_character_,
            .groups = "drop") |>
  mutate(state = case_when(!matched          ~ "no match",
                           letter == "J"     ~ "matched on J",
                           letter %in% c("M","N") ~ "matched on M/N",
                           TRUE              ~ "matched on other"))

pB <- ggplot(group_state, aes(round_rel_shift, condition, fill = state)) +
  geom_tile(colour = "white", linewidth = .6) +
  geom_vline(xintercept = 0.5, linetype = "dashed", linewidth = .4) +
  facet_wrap(~ model, ncol = 1) +
  scale_x_continuous(breaks = seq(-6, 7, 1)) +
  scale_fill_manual(values = c("no match"         = "grey85",
                               "matched on J"     = "#1f77b4",
                               "matched on M/N"   = "#d62728",
                               "matched on other" = "#7f7f7f")) +
  labs(title = "Group outcome per round",
       subtitle = "Whether all 6 matched, and on which letter",
       x = "rounds since payoff shift", y = NULL, fill = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "top", panel.grid = element_blank())

print(pB)

## --- switch latency + summary table ---------------------------------
summary_tbl <- group_state |>
  filter(round_rel_shift >= 1) |>
  group_by(model, condition) |>
  summarise(
    post_rounds        = n(),
    rounds_matched_MN  = sum(state == "matched on M/N"),
    rounds_matched_J   = sum(state == "matched on J"),
    rounds_no_match    = sum(state == "no match"),
    first_MN_round     = { w <- which(state == "matched on M/N")
                           if (length(w)) round_rel_shift[min(w)] else NA_integer_ },
    .groups = "drop"
  ) |>
  mutate(switched = !is.na(first_MN_round))

cat("\n=== Post-shift behaviour (human framing, n = 1 game per cell) ===\n")
print(as.data.frame(summary_tbl))

## individual-level M/N uptake in the post-shift phase
uptake <- long |>
  filter(round_rel_shift >= 1) |>
  group_by(model, condition) |>
  summarise(pct_MN_choices = mean(letter %in% c("M", "N")),
            pct_J_choices  = mean(letter == "J"), .groups = "drop")

cat("\n=== Individual choice mix, post-shift ===\n")
print(as.data.frame(uptake))

cat("\nNOTE: one game per condition per model. These are trajectories, not rates --\n")
cat("      treat any condition difference as a lead to replicate, not an effect.\n")
