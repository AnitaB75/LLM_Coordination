# ============================================================
# HUMAN-LLM COORDINATION -- STUDY 1
#
# Three data sources, one long-format table:
#
#   6H     name_game_2026-08-04.csv
#          oTree long export, human-only sessions
#
#   mixed  all_apps_wide-2026-08-05-2_part2.csv
#          oTree wide export, 2H+4LLM / 3H+3LLM / 4H+2LLM
#          (the non-part2 file is a strict subset; deduped)
#
#   6LLM   name_game_llm_MN20_20260804_131059.csv
#          simulation, condition == "llm"
# ============================================================


library(tidyverse)
library(jsonlite)
library(lme4)
library(broom.mixed)
library(scales)


DATA_DIR <- path.expand("~/Downloads/name_game_20 2")
N_ROUNDS <- 14

stopifnot(dir.exists(DATA_DIR))

f <- function(x) file.path(DATA_DIR, x)


# ============================================================
# 1. ALL-HUMAN TEAMS
# ============================================================

human_only_raw <- read_csv(
  f("name_game_2026-08-04.csv"),
  show_col_types = FALSE
)


human_only <- human_only_raw %>%
  
  transmute(
    group_id =
      paste0(
        "H__",
        session.code,
        "__",
        group.id_in_subsession
      ),
    
    player_uid =
      paste0(
        "H__",
        session.code,
        "__",
        group.id_in_subsession,
        "__",
        participant.code
      ),
    
    round = subsession.round_number,
    
    type = "Human",
    
    choice =
      toupper(
        trimws(player.letter_choice)
      )
  ) %>%
  
  filter(
    !is.na(choice),
    nzchar(choice)
  )


# Keep only teams of exactly six players who each played all
# 14 rounds. Group 1 in several sessions is oTree's catch-all
# bucket and holds up to 120 unassigned participants.

ho_valid <- human_only %>%
  
  count(group_id, player_uid) %>%
  filter(n == N_ROUNDS) %>%
  count(group_id) %>%
  filter(n == 6) %>%
  pull(group_id)


human_only <- human_only %>%
  
  filter(
    group_id %in% ho_valid
  ) %>%
  
  semi_join(
    human_only %>%
      count(group_id, player_uid) %>%
      filter(n == N_ROUNDS),
    by = c("group_id", "player_uid")
  )


cat("All-human teams retained:", n_distinct(human_only$group_id), "\n")


# ============================================================
# 2. MIXED TEAMS
# ============================================================

wide_files <- c(
  f("all_apps_wide-2026-08-05-2.csv"),
  f("all_apps_wide-2026-08-05-2_part2.csv")
)


# The two exports overlap on 132 participant codes. Bind, then
# keep the copy of each participant with the most rounds.

wide <- map_dfr(
  wide_files,
  ~ read_csv(
    .x,
    show_col_types = FALSE,
    col_types = cols(.default = col_character())
  )
) %>%
  
  mutate(
    .n =
      rowSums(
        !is.na(
          across(
            all_of(
              paste0(
                "name_game.",
                1:N_ROUNDS,
                ".player.letter_choice"
              )
            )
          )
        )
      )
  ) %>%
  
  arrange(
    participant.code,
    desc(.n)
  ) %>%
  
  distinct(
    participant.code,
    .keep_all = TRUE
  ) %>%
  
  select(-.n)


cat(
  "Unique participants in wide export:",
  nrow(wide),
  "\n"
)


# One group-round has llm_choices_json == "PENDING", a live-page
# write that never completed. fromJSON errors on it.

parse_llm <- function(x) {
  
  if (
    is.na(x) ||
    !nzchar(trimws(x)) ||
    !startsWith(trimws(x), "[")
  ) {
    return(character(0))
  }
  
  out <- tryCatch(
    fromJSON(x),
    error = function(e) NULL
  )
  
  if (is.null(out)) {
    return(character(0))
  }
  
  toupper(trimws(as.character(out)))
}


mixed_humans <- map_dfr(
  1:N_ROUNDS,
  function(r) {
    
    wide %>%
      
      transmute(
        session = session.code,
        
        group =
          .data[[paste0("name_game.", r, ".group.id_in_subsession")]],
        
        round = r,
        
        num_humans =
          as.integer(
            .data[[paste0("name_game.", r, ".group.num_humans")]]
          ),
        
        num_llms =
          as.integer(
            .data[[paste0("name_game.", r, ".group.num_llms")]]
          ),
        
        label = participant.code,
        
        type = "Human",
        
        choice =
          toupper(
            trimws(
              .data[[paste0("name_game.", r, ".player.letter_choice")]]
            )
          )
      ) %>%
      
      filter(
        !is.na(session),
        !is.na(group),
        !is.na(choice),
        nzchar(choice)
      )
  }
)


mixed_llms <- map_dfr(
  1:N_ROUNDS,
  function(r) {
    
    wide %>%
      
      transmute(
        session = session.code,
        
        group =
          .data[[paste0("name_game.", r, ".group.id_in_subsession")]],
        
        round = r,
        
        num_humans =
          as.integer(
            .data[[paste0("name_game.", r, ".group.num_humans")]]
          ),
        
        num_llms =
          as.integer(
            .data[[paste0("name_game.", r, ".group.num_llms")]]
          ),
        
        json =
          .data[[paste0("name_game.", r, ".group.llm_choices_json")]]
      ) %>%
      
      filter(
        !is.na(session),
        !is.na(group),
        !is.na(json)
      ) %>%
      
      distinct(
        session,
        group,
        round,
        .keep_all = TRUE
      ) %>%
      
      mutate(
        choices = map(json, parse_llm)
      ) %>%
      
      select(-json) %>%
      
      unnest_longer(
        choices,
        values_to = "choice",
        indices_to = "i"
      ) %>%
      
      mutate(
        label = paste0("L", i),
        type = "LLM"
      ) %>%
      
      select(
        session, group, round,
        num_humans, num_llms,
        label, type, choice
      )
  }
)


mixed_all <- bind_rows(
  mixed_humans,
  mixed_llms
) %>%
  
  mutate(
    group_id =
      paste0("M__", session, "__", group),
    
    player_uid =
      paste0("M__", session, "__", group, "__", label)
  ) %>%
  
  distinct(
    group_id,
    player_uid,
    round,
    .keep_all = TRUE
  )


# A group is retained only if every round has exactly the
# stamped number of humans and LLMs, and six players in total.

mixed_valid <- mixed_all %>%
  
  group_by(group_id, round) %>%
  
  summarise(
    oh = n_distinct(player_uid[type == "Human"]),
    ol = n_distinct(player_uid[type == "LLM"]),
    eh = first(num_humans),
    el = first(num_llms),
    .groups = "drop"
  ) %>%
  
  group_by(group_id) %>%
  
  summarise(
    ok =
      n() == N_ROUNDS &
      all(!is.na(eh) & !is.na(el)) &
      all(oh == eh & ol == el & oh + ol == 6),
    
    .groups = "drop"
  ) %>%
  
  filter(ok) %>%
  
  pull(group_id)


mixed <- mixed_all %>%
  
  filter(
    group_id %in% mixed_valid
  ) %>%
  
  select(
    group_id,
    player_uid,
    round,
    type,
    choice
  )


cat("Mixed teams retained:", n_distinct(mixed$group_id), "\n")


# ============================================================
# 3. ALL-LLM TEAMS
# ============================================================

# In this file `condition` is the framing given to the agents
# (told teammates are humans vs LLMs), not the actual
# composition. Every player is an LLM in both arms. The paper's
# LLM-only condition is the "llm" arm.

llm_only <- read_csv(
  f("name_game_llm_MN20_20260804_131059.csv"),
  show_col_types = FALSE
) %>%
  
  filter(condition == "llm") %>%
  
  select(
    trial,
    round,
    p1:p6
  ) %>%
  
  pivot_longer(
    p1:p6,
    names_to = "slot",
    values_to = "choice"
  ) %>%
  
  transmute(
    group_id = paste0("L__", trial),
    player_uid = paste0("L__", trial, "__", slot),
    round,
    type = "LLM",
    choice = toupper(trimws(choice))
  ) %>%
  
  filter(
    !is.na(choice),
    nzchar(choice)
  )


cat("All-LLM teams retained:", n_distinct(llm_only$group_id), "\n")


# ============================================================
# 4. COMBINE AND LABEL
# ============================================================

game <- bind_rows(
  human_only,
  mixed,
  llm_only
) %>%
  
  mutate(
    type =
      factor(
        type,
        levels = c("Human", "LLM")
      ),
    
    better = choice %in% c("M", "N"),
    precedent = choice == "J"
  )


composition <- game %>%
  
  group_by(group_id) %>%
  
  summarise(
    n_human =
      n_distinct(
        player_uid[type == "Human"]
      ),
    
    n_llm =
      n_distinct(
        player_uid[type == "LLM"]
      ),
    
    .groups = "drop"
  ) %>%
  
  mutate(
    condition =
      case_when(
        n_llm == 0   ~ paste0(n_human, "H"),
        n_human == 0 ~ paste0(n_llm, "LLM"),
        TRUE         ~ paste0(n_human, "H + ", n_llm, "LLM")
      )
  )


cond_levels <- composition %>%
  distinct(condition, n_llm) %>%
  arrange(n_llm) %>%
  pull(condition)


game <- game %>%
  
  left_join(
    composition %>%
      select(group_id, condition),
    by = "group_id"
  ) %>%
  
  mutate(
    condition =
      factor(condition, levels = cond_levels)
  )


stopifnot(
  !any(is.na(game$condition))
)

stopifnot(
  all(
    count(game, group_id, round)$n == 6
  )
)


cat("\n=== GROUPS PER COMPOSITION ===\n")
print(count(composition, condition))


# ============================================================
# 5. FIGURE 1: TEAM-LEVEL SWITCHING, ALL COMPOSITIONS
# ============================================================

switch_rates <- function(data, ...) {
  
  data %>%
    
    group_by(group_id, round, ...) %>%
    
    summarise(
      p_group = mean(better),
      .groups = "drop"
    ) %>%
    
    group_by(round, ...) %>%
    
    summarise(
      n_groups = n(),
      p_switch = mean(p_group),
      
      se =
        if_else(
          n() > 1,
          sd(p_group) / sqrt(n()),
          NA_real_
        ),
      
      .groups = "drop"
    ) %>%
    
    mutate(
      ci_low = pmax(0, p_switch - 1.96 * se),
      ci_high = pmin(1, p_switch + 1.96 * se)
    )
}


reward_marker <- list(
  
  geom_vline(
    xintercept = 7.5,
    linetype = "dashed",
    colour = "grey40"
  ),
  
  annotate(
    "text",
    x = 7.4,
    y = 1,
    label = "Reward\nchange",
    hjust = 1,
    vjust = 1,
    size = 3,
    fontface = "italic",
    colour = "grey40"
  )
)


base_theme <- list(
  
  scale_y_continuous(
    limits = c(0, 1),
    labels = percent
  ),
  
  scale_x_continuous(breaks = 1:14),
  
  theme_minimal(base_size = 12),
  
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.title = element_blank()
  )
)


cond_cols <- setNames(
  colorRampPalette(
    c("#2E5FE8", "#7B6BD6", "#A85FA8", "#D14A6B", "#E8342E")
  )(length(cond_levels)),
  cond_levels
)


rates_team <- switch_rates(game, condition)


fig1 <- ggplot(
  rates_team,
  aes(round, p_switch, colour = condition)
) +
  reward_marker +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_colour_manual(values = cond_cols) +
  labs(
    title = "Team-level rate of switching to M or N",
    x = "Round",
    y = "% of all players choosing M or N"
  ) +
  base_theme


print(fig1)


cat("\n=== FIGURE 1 VALUES ===\n")
print(
  rates_team %>%
    select(condition, round, n_groups, p_switch) %>%
    pivot_wider(
      names_from = round,
      values_from = p_switch
    ),
  width = Inf
)


# ============================================================
# 6. FIGURE 1C: AGENT TYPE WITHIN COMPOSITION
# ============================================================

type_cols <- c(
  Human = "#2E5FE8",
  LLM = "#E8342E"
)


fig1c <- ggplot(
  switch_rates(game, condition, type),
  aes(round, p_switch, colour = type)
) +
  reward_marker +
  geom_ribbon(
    aes(
      ymin = ci_low,
      ymax = ci_high,
      fill = type
    ),
    alpha = 0.15,
    colour = NA
  ) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 1.6) +
  scale_colour_manual(values = type_cols) +
  scale_fill_manual(values = type_cols) +
  facet_wrap(~ condition, nrow = 1) +
  scale_y_continuous(
    limits = c(0, 1),
    labels = percent
  ) +
  scale_x_continuous(breaks = c(1, 4, 7, 10, 14)) +
  labs(
    title = "Switching to M or N, by composition and agent type",
    x = "Round",
    y = "% of players choosing M or N"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.title = element_blank()
  )


print(fig1c)


# ============================================================
# 7. ROUND 8
# ============================================================

round8 <- game %>%
  
  filter(round == 8) %>%
  
  mutate(
    switched = choice %in% c("M", "N")
  )


cat("\n=== ROUND 8 BY TYPE ===\n")
print(
  round8 %>%
    group_by(type) %>%
    summarise(
      n = n(),
      n_switch = sum(switched),
      p_switch = mean(switched),
      .groups = "drop"
    )
)


print(
  fisher.test(
    table(round8$type, round8$switched)
  )
)


cat("\n=== ROUND 8 BY COMPOSITION AND TYPE ===\n")
print(
  round8 %>%
    group_by(condition, type) %>%
    summarise(
      n = n(),
      p_switch = mean(switched),
      .groups = "drop"
    ),
  n = Inf
)


# ============================================================
# 8. DYNAMICS: CASCADE AND REVERSION
# ============================================================

dynamic <- game %>%
  
  group_by(group_id, player_uid) %>%
  arrange(round, .by_group = TRUE) %>%
  
  mutate(
    choice_prev = lag(choice),
    better_prev = lag(better)
  ) %>%
  
  ungroup() %>%
  
  left_join(
    game %>%
      group_by(group_id, round) %>%
      summarise(
        total_M = sum(choice == "M"),
        total_N = sum(choice == "N"),
        total_better = sum(choice %in% c("M", "N")),
        .groups = "drop"
      ) %>%
      mutate(round = round + 1),
    
    by = c("group_id", "round")
  ) %>%
  
  mutate(
    others_better_prev =
      total_better - as.integer(better_prev),
    
    others_M_prev =
      total_M - as.integer(choice_prev == "M"),
    
    others_N_prev =
      total_N - as.integer(choice_prev == "N")
  )


adoption <- dynamic %>%
  
  filter(
    round >= 9,
    choice_prev == "J"
  ) %>%
  
  mutate(
    switched_from_J = choice %in% c("M", "N")
  )


cat("\n=== CASCADE ===\n")
print(
  adoption %>%
    group_by(type, others_better_prev) %>%
    summarise(
      n = n(),
      p_switch = mean(switched_from_J),
      .groups = "drop"
    ),
  n = Inf
)


m_cascade <- glmer(
  switched_from_J ~
    type * others_better_prev +
    (1 | group_id),
  
  data = adoption,
  family = binomial,
  
  control = glmerControl(
    optimizer = "bobyqa",
    optCtrl = list(maxfun = 200000)
  )
)

print(summary(m_cascade))

print(
  broom.mixed::tidy(
    m_cascade,
    effects = "fixed",
    conf.int = TRUE,
    exponentiate = TRUE
  )
)


reversion <- dynamic %>%
  
  filter(
    round >= 9,
    choice_prev %in% c("M", "N")
  ) %>%
  
  mutate(
    same_alt_support =
      if_else(
        choice_prev == "M",
        others_M_prev,
        others_N_prev
      ),
    
    reverted_to_J = choice == "J"
  )


cat("\n=== REVERSION ===\n")
print(
  reversion %>%
    group_by(type) %>%
    summarise(
      n = n(),
      p_revert = mean(reverted_to_J),
      .groups = "drop"
    )
)


# ============================================================
# 9. ROUND 14 OUTCOME
# ============================================================

round14 <- game %>%
  
  filter(round == 14) %>%
  
  group_by(group_id, condition) %>%
  
  summarise(
    n_unique = n_distinct(choice),
    
    final =
      if_else(
        n_distinct(choice) == 1,
        first(choice),
        NA_character_
      ),
    
    .groups = "drop"
  ) %>%
  
  mutate(
    outcome =
      case_when(
        n_unique != 1 ~ "Not coordinated",
        final == "J" ~ "Coordinated on J",
        final %in% c("M", "N") ~ "Coordinated on M or N",
        TRUE ~ "Coordinated on a zero-point letter"
      )
  )


cat("\n=== ROUND 14: ALL GROUPS ===\n")
print(
  count(round14, condition, outcome),
  n = Inf
)


round14_coord <- round14 %>%
  
  filter(
    outcome %in%
      c("Coordinated on J", "Coordinated on M or N")
  ) %>%
  
  count(condition, outcome) %>%
  
  group_by(condition) %>%
  
  mutate(
    proportion = n / sum(n)
  ) %>%
  
  ungroup()


fig2 <- ggplot(
  round14_coord,
  aes(condition, proportion, fill = outcome)
) +
  geom_col() +
  geom_text(
    aes(
      label = percent(proportion, accuracy = 1)
    ),
    position = position_stack(vjust = 0.5)
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    labels = percent
  ) +
  labs(
    title = "Round 14 outcome among coordinated groups",
    x = NULL,
    y = "% of coordinated groups",
    fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )


print(fig2)


ggsave(f("fig1_composition.png"), fig1,  width = 8,  height = 4.5, dpi = 300)
ggsave(f("fig1c_faceted.png"),    fig1c, width = 14, height = 4,   dpi = 300)
ggsave(f("fig2_round14.png"),     fig2,  width = 8,  height = 4.5, dpi = 300)