library(tidyverse)
library(ggrepel)
library(patchwork)
library(scales)

# ==============================================================================
# LOAD DATA
# ==============================================================================

results <- read_csv(
  "/Users/calypsocalcott/LLM_Coordination/wvs_cultural_analysis/results_table.csv",
  show_col_types = FALSE
)

# Rebuild country-level merged data for scatter plots
wvs_raw <- read_csv(
  "/Users/calypsocalcott/Downloads/wvs_wave7/WVS_Cross-National_Wave_7_csv_v6_0.csv",
  show_col_types = FALSE
)

moral_items <- paste0("Q", 177:195)
trust_items <- paste0("Q", 57:62)

wvs <- wvs_raw %>%
  select(country = B_COUNTRY_ALPHA, weight = W_WEIGHT,
         all_of(c(moral_items, trust_items))) %>%
  mutate(across(all_of(c(moral_items, trust_items)),
                ~ if_else(. < 0, NA_real_, as.numeric(.))))

wvs_country <- wvs %>%
  group_by(country) %>%
  summarise(
    across(all_of(c(moral_items, trust_items)),
           ~ weighted.mean(., w = weight, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(
    purity   = rowMeans(select(., Q182, Q183, Q186, Q193), na.rm = TRUE),
    harm     = rowMeans(select(., Q189, Q190, Q191, Q192, Q194), na.rm = TRUE),
    fairness = rowMeans(select(., Q177, Q178, Q179, Q180, Q181), na.rm = TRUE),
    autonomy = rowMeans(select(., Q184, Q185, Q187, Q188), na.rm = TRUE),
    trust_ip = rowMeans(select(., Q57, Q58, Q59), na.rm = TRUE)
  )

enke    <- read_csv("/Users/calypsocalcott/Desktop/value_structure/enke_kinship.csv",
                    show_col_types = FALSE) %>%
  select(country = country_code, enke_kinship = enke_kinship_tightness)

gelfand <- read_csv("/Users/calypsocalcott/Desktop/value_structure/gelfand_tightness.csv",
                    show_col_types = FALSE) %>%
  select(country = country_code, gelfand_tl = gelfand_tightness)

gdp <- tribble(
  ~country, ~log_gdp,
  "USA",10.82,"GBR",10.68,"DEU",10.73,"FRA",10.67,"JPN",10.60,
  "CAN",10.65,"AUS",10.72,"NZL",10.55,"SWE",10.70,"NOR",11.07,
  "DNK",10.82,"FIN",10.64,"CHE",11.11,"NLD",10.77,"BEL",10.65,
  "AUT",10.71,"IRL",10.97,"ISR",10.47,"ESP",10.35,"ITA",10.36,
  "PRT",10.18,"GRC",10.00,"CZE",10.33,"HUN",10.09,"POL",10.14,
  "SVK",10.17,"ROU",10.01,"SRB",9.62,"TUR",9.79,"RUS",9.82,
  "CHN",9.43,"KOR",10.47,"SGP",11.04,"HKG",10.93,"TWN",10.54,
  "MYS",9.69,"THA",9.27,"VNM",8.55,"IDN",8.77,"PHL",8.50,
  "IND",8.13,"PAK",7.88,"BGD",7.67,"EGY",8.57,"MAR",8.37,
  "DZA",8.86,"NGA",7.72,"KEN",7.59,"ZAF",9.06,"ETH",7.12,
  "ZWE",7.19,"MEX",9.37,"BRA",9.37,"ARG",9.25,"CHL",9.57,
  "COL",9.07,"PER",9.05,"GTM",8.61,"PAN",9.69,"SAU",10.44,
  "ARE",10.92,"IRN",9.04,"IRQ",8.79,"JOR",9.02,"KAZ",9.41,
  "KGZ",7.72,"TJK",7.22,"AZE",9.02,"ARM",8.74,"GEO",8.81,
  "UKR",8.53,"BLR",9.17,"LTU",10.19,"LVA",10.09,"EST",10.22
)

df <- wvs_country %>%
  left_join(enke,    by = "country") %>%
  left_join(gelfand, by = "country") %>%
  left_join(gdp,     by = "country")

resid_on_gdp <- function(x, g) {
  ok <- !is.na(x) & !is.na(g)
  r  <- rep(NA_real_, length(x))
  if (sum(ok) >= 5) r[ok] <- residuals(lm(x[ok] ~ g[ok]))
  r
}

df <- df %>%
  mutate(across(c(purity, harm, fairness, autonomy, enke_kinship, gelfand_tl),
                ~ resid_on_gdp(., log_gdp),
                .names = "{.col}_r"))

# ==============================================================================
# LABELS & SHARED THEME
# ==============================================================================

pred_labels <- c(
  enke_kinship        = "Enke Kinship\nTightness",
  gelfand_tl          = "Gelfand\nTightness-Looseness",
  rel_mobility        = "Relational\nMobility",
  trust_ip            = "Interpersonal\nTrust (WVS)",
  mfq_individualizing = "MFQ\nIndividualizing",
  mfq_binding         = "MFQ\nBinding"
)

outcome_labels <- c(
  purity   = "Purity",
  harm     = "Harm",
  fairness = "Fairness",
  autonomy = "Autonomy"
)

base_theme <- theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(size = 10, color = "grey40"),
    axis.title    = element_text(size = 11),
    panel.grid.minor = element_blank(),
    panel.border  = element_rect(color = "grey80", fill = NA)
  )

# ==============================================================================
# PLOT 1: Q² HEATMAP
# ==============================================================================

heatmap_data <- results %>%
  filter(!is.na(q2)) %>%
  mutate(
    pred_label = pred_labels[predictor],
    out_label  = outcome_labels[outcome],
    label      = sprintf("%.2f\n(n=%d)", q2, n),
    sig        = case_when(q2 > 0.3 ~ "strong", q2 > 0 ~ "positive", TRUE ~ "negative")
  )

p1 <- ggplot(heatmap_data,
             aes(x = factor(pred_label, levels = pred_labels),
                 y = factor(out_label,  levels = rev(outcome_labels)),
                 fill = q2)) +
  geom_tile(color = "white", linewidth = 1.2) +
  geom_text(aes(label = label), size = 3.2, lineheight = 1.2) +
  scale_fill_gradient2(
    low      = "#d73027",
    mid      = "#ffffbf",
    high     = "#1a9850",
    midpoint = 0,
    limits   = c(-0.35, 0.65),
    name     = "LOO-CV R²"
  ) +
  scale_x_discrete(position = "top") +
  labs(
    title    = "Predictive Accuracy by Cultural Measure and Moral Domain",
    subtitle = "LOO-CV R² (Q²) after GDP residualization — green = good out-of-sample fit, red = worse than mean",
    x = NULL, y = NULL
  ) +
  base_theme +
  theme(
    axis.text.x     = element_text(size = 10, face = "bold"),
    axis.text.y     = element_text(size = 11, face = "bold"),
    legend.position = "right",
    panel.border    = element_blank(),
    panel.grid      = element_blank()
  )

print(p1)

# ==============================================================================
# PLOT 2: OVERALL PREDICTOR RANKING (mean Q²)
# ==============================================================================

ranking <- results %>%
  group_by(predictor) %>%
  summarise(mean_q2 = mean(q2, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    label     = pred_labels[predictor],
    positive  = mean_q2 > 0
  ) %>%
  arrange(mean_q2)

p2 <- ggplot(ranking,
             aes(x = mean_q2,
                 y = reorder(label, mean_q2),
                 fill = positive)) +
  geom_col(width = 0.65) +
  geom_vline(xintercept = 0, linewidth = 0.8, color = "grey30") +
  geom_text(aes(label = sprintf("%.3f", mean_q2),
                hjust = if_else(positive, -0.15, 1.15)),
            size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("TRUE" = "#1a9850", "FALSE" = "#d73027"),
                    guide = "none") +
  scale_x_continuous(limits = c(-0.18, 0.26), expand = c(0, 0)) +
  labs(
    title    = "Overall Predictor Ranking",
    subtitle = "Mean LOO-CV R² averaged across all four moral outcomes",
    x        = "Mean LOO-CV R²",
    y        = NULL
  ) +
  base_theme +
  theme(panel.grid.major.y = element_blank())

print(p2)

# ==============================================================================
# PLOT 3: SCATTER PLOTS — SIGNIFICANT RELATIONSHIPS
# ==============================================================================

make_scatter <- function(data, x_var, y_var, x_lab, y_lab, title) {
  d <- data %>%
    filter(!is.na(.data[[x_var]]), !is.na(.data[[y_var]]))
  r_val <- round(cor(d[[x_var]], d[[y_var]]), 2)
  ggplot(d, aes(x = .data[[x_var]], y = .data[[y_var]], label = country)) +
    geom_hline(yintercept = 0, color = "grey70", linetype = "dashed") +
    geom_vline(xintercept = 0, color = "grey70", linetype = "dashed") +
    geom_smooth(method = "lm", se = TRUE, color = "#2166ac",
                fill = "#2166ac", alpha = 0.15, linewidth = 1) +
    geom_point(size = 2.2, color = "#2166ac", alpha = 0.8) +
    geom_text_repel(size = 2.8, color = "grey30", max.overlaps = 20,
                    segment.color = "grey70", segment.size = 0.3) +
    annotate("text", x = Inf, y = Inf, hjust = 1.1, vjust = 1.5,
             label = sprintf("r = %s", r_val), fontface = "bold", size = 4) +
    labs(title = title, x = x_lab, y = y_lab) +
    base_theme
}

p3a <- make_scatter(df, "enke_kinship_r", "purity_r",
                    "Enke Kinship Tightness (residualized)",
                    "Purity permissiveness (residualized)",
                    "Kinship Tightness → Purity  [Q²=0.58]")

p3b <- make_scatter(df, "gelfand_tl_r", "purity_r",
                    "Gelfand Tightness-Looseness (residualized)",
                    "Purity permissiveness (residualized)",
                    "Tightness-Looseness → Purity  [Q²=0.51]")

p3c <- make_scatter(df, "enke_kinship_r", "autonomy_r",
                    "Enke Kinship Tightness (residualized)",
                    "Autonomy permissiveness (residualized)",
                    "Kinship Tightness → Autonomy  [Q²=0.27]")

p3d <- make_scatter(df, "gelfand_tl_r", "autonomy_r",
                    "Gelfand Tightness-Looseness (residualized)",
                    "Autonomy permissiveness (residualized)",
                    "Tightness-Looseness → Autonomy  [Q²=0.29]")

p3 <- (p3a | p3b) / (p3c | p3d) +
  plot_annotation(
    title    = "Cultural Tightness Predicts Purity and Autonomy Attitudes",
    subtitle = "GDP-residualized values; axes show variance not explained by national wealth",
    theme = theme(
      plot.title    = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 10, color = "grey40")
    )
  )

print(p3)

# ==============================================================================
# PLOT 4: NULL FINDINGS — HARM & FAIRNESS
# ==============================================================================

p4a <- make_scatter(df, "enke_kinship_r", "harm_r",
                    "Enke Kinship Tightness (residualized)",
                    "Harm tolerance (residualized)",
                    "Kinship Tightness → Harm  [Q²=−0.11]")

p4b <- make_scatter(df, "enke_kinship_r", "fairness_r",
                    "Enke Kinship Tightness (residualized)",
                    "Fairness violations (residualized)",
                    "Kinship Tightness → Fairness  [Q²=−0.03]")

p4 <- (p4a | p4b) +
  plot_annotation(
    title    = "Null Results: Harm and Fairness Unpredicted After GDP Control",
    subtitle = "Country-level variance in harm tolerance and fairness violations is explained by wealth, not cultural tightness",
    theme = theme(
      plot.title    = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 10, color = "grey40")
    )
  )

print(p4)
