library(tidyverse)

# ==============================================================================
# DATA LOADING
# ==============================================================================

cat("Loading WVS Wave 7...\n")
wvs_raw <- read_csv(
  "/Users/calypsocalcott/Downloads/wvs_wave7/WVS_Cross-National_Wave_7_csv_v6_0.csv",
  show_col_types = FALSE
)

# Recode negative values (missing codes) as NA
moral_items   <- paste0("Q", 177:195)
trust_items   <- paste0("Q", 57:62)
all_items     <- c(moral_items, trust_items)

wvs <- wvs_raw %>%
  select(country = B_COUNTRY_ALPHA, weight = W_WEIGHT, all_of(all_items)) %>%
  mutate(across(all_of(all_items), ~ if_else(. < 0, NA_real_, as.numeric(.))))

cat(sprintf("  WVS: %d respondents, %d countries\n", nrow(wvs), n_distinct(wvs$country)))

# ==============================================================================
# AGGREGATE WVS TO COUNTRY LEVEL
# ==============================================================================

wvs_country <- wvs %>%
  group_by(country) %>%
  summarise(
    n_wvs = n(),
    across(all_of(all_items), ~ weighted.mean(., w = weight, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(
    purity    = rowMeans(select(., Q182, Q183, Q186, Q193), na.rm = TRUE),
    harm      = rowMeans(select(., Q189, Q190, Q191, Q192, Q194), na.rm = TRUE),
    fairness  = rowMeans(select(., Q177, Q178, Q179, Q180, Q181), na.rm = TRUE),
    autonomy  = rowMeans(select(., Q184, Q185, Q187, Q188), na.rm = TRUE),
    trust_ip  = rowMeans(select(., Q57, Q58, Q59), na.rm = TRUE)
  )

cat(sprintf("  Country-level WVS: %d countries\n", nrow(wvs_country)))

# ==============================================================================
# MFQ: AGGREGATE YOURMORALS DATA BY COUNTRY
# ==============================================================================

cat("Loading MFQ data...\n")
mfq_raw <- read_csv("/Users/calypsocalcott/Downloads/MFQ_clean (2).csv",
                    show_col_types = FALSE)

mfq_country <- mfq_raw %>%
  filter(!is.na(currentCountry), currentCountry != "") %>%
  group_by(country = currentCountry) %>%
  summarise(
    n_mfq          = n(),
    mfq_care       = mean(care_score,            na.rm = TRUE),
    mfq_hierarchy  = mean(hierarchy_score,       na.rm = TRUE),
    mfq_prop       = mean(proportionality_score, na.rm = TRUE),
    mfq_equality   = mean(equality_score,        na.rm = TRUE),
    mfq_loyalty    = mean(loyalty_score,         na.rm = TRUE),
    mfq_purity     = mean(purity_score,          na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(n_mfq >= 30) %>%
  mutate(
    mfq_individualizing = (mfq_care + mfq_equality + mfq_prop) / 3,
    mfq_binding         = (mfq_hierarchy + mfq_loyalty + mfq_purity) / 3
  )

cat(sprintf("  MFQ: %d countries (n>=30)\n", nrow(mfq_country)))

# Standardise country names to ISO3 for MFQ (it uses full names)
name_to_iso3 <- c(
  "United States" = "USA", "Canada" = "CAN", "United Kingdom" = "GBR",
  "Australia" = "AUS", "France" = "FRA", "Indonesia" = "IDN",
  "Germany" = "DEU", "New Zealand" = "NZL", "Switzerland" = "CHE",
  "Belgium" = "BEL", "Brazil" = "BRA", "Ireland" = "IRL",
  "Japan" = "JPN", "Netherlands The" = "NLD", "Mexico" = "MEX",
  "India" = "IND", "South Africa" = "ZAF", "Philippines" = "PHL",
  "Russia" = "RUS", "Argentina" = "ARG", "China" = "CHN",
  "Sweden" = "SWE", "Spain" = "ESP", "Chile" = "CHL",
  "Singapore" = "SGP", "Saudi Arabia" = "SAU", "Peru" = "PER",
  "Kenya" = "KEN", "Nigeria" = "NGA", "Egypt" = "EGY",
  "Poland" = "POL", "Morocco" = "MAR", "UAE" = "ARE",
  "Columbia" = "COL", "Colombia" = "COL", "Italy" = "ITA",
  "Romania" = "ROU", "Austria" = "AUT", "Korea South" = "KOR",
  "Czech Republic" = "CZE", "Israel" = "ISR", "Finland" = "FIN",
  "Denmark" = "DNK", "Norway" = "NOR", "Guatemala" = "GTM",
  "Portugal" = "PRT", "Hong Kong S.A.R." = "HKG", "Turkey" = "TUR",
  "Hungary" = "HUN", "United Arab Emirates" = "ARE",
  "Pakistan" = "PAK", "Malaysia" = "MYS", "Slovakia" = "SVK",
  "Vietnam" = "VNM", "Thailand" = "THA", "Greece" = "GRC",
  "Serbia" = "SRB", "Taiwan" = "TWN", "Afghanistan" = "AFG",
  "Algeria" = "DZA", "Panama" = "PAN"
)

mfq_country <- mfq_country %>%
  mutate(country_iso3 = name_to_iso3[country]) %>%
  filter(!is.na(country_iso3)) %>%
  select(country = country_iso3, n_mfq, mfq_individualizing, mfq_binding)

# ==============================================================================
# CULTURAL MEASURES
# ==============================================================================

cat("Loading cultural measures...\n")

gelfand <- read_csv("/Users/calypsocalcott/Desktop/value_structure/gelfand_tightness.csv",
                    show_col_types = FALSE) %>%
  select(country = country_code, gelfand_tl = gelfand_tightness)

rel_mob <- read_csv("/Users/calypsocalcott/Desktop/value_structure/relational_mobility_scores.csv",
                    show_col_types = FALSE) %>%
  select(country = country_code, rel_mobility = relational_mobility)

enke <- read_csv("/Users/calypsocalcott/Desktop/value_structure/enke_kinship.csv",
                 show_col_types = FALSE) %>%
  select(country = country_code, enke_kinship = enke_kinship_tightness)

cat(sprintf("  Gelfand: %d countries | RelMob: %d | Enke kinship: %d | MFQ: %d\n",
            nrow(gelfand), nrow(rel_mob), nrow(enke), nrow(mfq_country)))

# ==============================================================================
# GDP (World Bank 2019, log)
# ==============================================================================

gdp <- tribble(
  ~country, ~log_gdp,
  "USA",10.82,"GBR",10.68,"DEU",10.73,"FRA",10.67,"JPN",10.60,
  "CAN",10.65,"AUS",10.72,"NZL",10.55,"SWE",10.70,"NOR",11.07,
  "DNK",10.82,"FIN",10.64,"CHE",11.11,"NLD",10.77,"BEL",10.65,
  "AUT",10.71,"IRL",10.97,"ISR",10.47,"ESP",10.35,"ITA",10.36,
  "PRT",10.18,"GRC",10.00,"CZE",10.33,"HUN",10.09,"POL",10.14,
  "SVK",10.17,"ROU",10.01,"SRB",9.62,"TUR",9.79,"RUS",9.82,
  "CHN",9.43,"KOR",10.47,"JPN",10.60,"SGP",11.04,"HKG",10.93,
  "TWN",10.54,"MYS",9.69,"THA",9.27,"VNM",8.55,"IDN",8.77,
  "PHL",8.50,"IND",8.13,"PAK",7.88,"BGD",7.67,"EGY",8.57,
  "MAR",8.37,"DZA",8.86,"TUN",8.59,"NGA",7.72,"KEN",7.59,
  "ZAF",9.06,"ETH",7.12,"ZWE",7.19,"MEX",9.37,"BRA",9.37,
  "ARG",9.25,"CHL",9.57,"COL",9.07,"PER",9.05,"GTM",8.61,
  "PAN",9.69,"PRY",8.59,"ECU",8.98,"BOL",8.27,"NIC",7.87,
  "SAU",10.44,"ARE",10.92,"IRN",9.04,"IRQ",8.79,"JOR",9.02,
  "KAZ",9.41,"KGZ",7.72,"TJK",7.22,"AZE",9.02,"ARM",8.74,
  "GEO",8.81,"UKR",8.53,"BLR",9.17,"LTU",10.19,"LVA",10.09,
  "EST",10.22
)

# ==============================================================================
# MERGE ALL DATA
# ==============================================================================

df <- wvs_country %>%
  left_join(gelfand,     by = "country") %>%
  left_join(rel_mob,     by = "country") %>%
  left_join(enke,        by = "country") %>%
  left_join(mfq_country, by = "country") %>%
  left_join(gdp,         by = "country")

cat(sprintf("\nMerged dataset: %d countries\n", nrow(df)))

outcomes   <- c("purity", "harm", "fairness", "autonomy")
predictors <- c("gelfand_tl", "rel_mobility", "enke_kinship",
                "mfq_individualizing", "mfq_binding", "trust_ip")

coverage <- df %>%
  summarise(across(all_of(c(outcomes, predictors, "log_gdp")),
                   ~ sum(!is.na(.))))
cat("\nCountry coverage per variable:\n")
print(as.data.frame(t(coverage)))

# ==============================================================================
# GDP RESIDUALIZATION
# ==============================================================================

resid_on_gdp <- function(x, gdp) {
  ok <- !is.na(x) & !is.na(gdp)
  if (sum(ok) < 5) return(rep(NA_real_, length(x)))
  r <- rep(NA_real_, length(x))
  r[ok] <- residuals(lm(x[ok] ~ gdp[ok]))
  r
}

df_r <- df %>%
  mutate(across(all_of(c(outcomes, predictors)),
                ~ resid_on_gdp(., log_gdp),
                .names = "{.col}_r"))

# ==============================================================================
# LOO-CV R² (Q²)
# ==============================================================================

loo_cv_r2 <- function(y, x) {
  ok <- !is.na(y) & !is.na(x)
  n  <- sum(ok)
  if (n < 6) return(list(r = NA, r2 = NA, q2 = NA, n = n))
  y_ <- y[ok]; x_ <- x[ok]
  fit <- lm(y_ ~ x_)
  h   <- hatvalues(fit)
  e   <- residuals(fit)
  press <- sum((e / (1 - h))^2)
  sst   <- sum((y_ - mean(y_))^2)
  list(
    r  = cor(y_, x_),
    r2 = summary(fit)$r.squared,
    q2 = 1 - press / sst,
    n  = n
  )
}

# ==============================================================================
# RUN ALL PREDICTOR × OUTCOME COMBINATIONS
# ==============================================================================

cat("\n================================================================================\n")
cat("RESULTS: LOO-CV R² (Q²) for each predictor × outcome (GDP-residualized)\n")
cat("================================================================================\n")

results <- expand_grid(outcome = outcomes, predictor = predictors) %>%
  rowwise() %>%
  mutate(
    res = list(loo_cv_r2(
      df_r[[paste0(outcome,   "_r")]],
      df_r[[paste0(predictor, "_r")]]
    )),
    r    = res$r,
    r2   = res$r2,
    q2   = res$q2,
    n    = res$n
  ) %>%
  select(-res) %>%
  ungroup()

# Print as matrix
cat("\n--- Pearson r ---\n")
results %>%
  select(outcome, predictor, r) %>%
  pivot_wider(names_from = predictor, values_from = r) %>%
  mutate(across(where(is.numeric), ~ round(., 3))) %>%
  print()

cat("\n--- R² ---\n")
results %>%
  select(outcome, predictor, r2) %>%
  pivot_wider(names_from = predictor, values_from = r2) %>%
  mutate(across(where(is.numeric), ~ round(., 3))) %>%
  print()

cat("\n--- LOO-CV R² (Q²) — primary metric ---\n")
results %>%
  select(outcome, predictor, q2) %>%
  pivot_wider(names_from = predictor, values_from = q2) %>%
  mutate(across(where(is.numeric), ~ round(., 3))) %>%
  print()

cat("\n--- N (countries per cell) ---\n")
results %>%
  select(outcome, predictor, n) %>%
  pivot_wider(names_from = predictor, values_from = n) %>%
  print()

# ==============================================================================
# RANKINGS
# ==============================================================================

cat("\n================================================================================\n")
cat("PREDICTOR RANKINGS BY OUTCOME (LOO-CV R²)\n")
cat("================================================================================\n")

for (oc in outcomes) {
  cat(sprintf("\n%s:\n", toupper(oc)))
  results %>%
    filter(outcome == oc, !is.na(q2)) %>%
    arrange(desc(q2)) %>%
    mutate(rank = row_number()) %>%
    select(rank, predictor, r = r, q2, n) %>%
    mutate(across(c(r, q2), ~ round(., 3))) %>%
    { print(.) }
}

cat("\n================================================================================\n")
cat("OVERALL WINNER (mean Q² across outcomes)\n")
cat("================================================================================\n")

results %>%
  group_by(predictor) %>%
  summarise(mean_q2 = mean(q2, na.rm = TRUE),
            n_outcomes = sum(!is.na(q2)),
            .groups = "drop") %>%
  arrange(desc(mean_q2)) %>%
  mutate(across(mean_q2, ~ round(., 3))) %>%
  print()

# ==============================================================================
# TWO-PREDICTOR MODELS FOR TOP COMPETITORS
# ==============================================================================

cat("\n================================================================================\n")
cat("TWO-PREDICTOR MODELS (top 2 per outcome, GDP-residualized)\n")
cat("================================================================================\n")

top2 <- results %>%
  filter(!is.na(q2)) %>%
  group_by(outcome) %>%
  slice_max(q2, n = 2) %>%
  ungroup()

for (oc in outcomes) {
  preds <- top2 %>% filter(outcome == oc) %>% pull(predictor)
  if (length(preds) < 2) next
  p1 <- paste0(preds[1], "_r"); p2 <- paste0(preds[2], "_r"); y <- paste0(oc, "_r")
  ok <- complete.cases(df_r[[y]], df_r[[p1]], df_r[[p2]])
  if (sum(ok) < 6) next
  fit2 <- lm(df_r[[y]][ok] ~ df_r[[p1]][ok] + df_r[[p2]][ok])
  h <- hatvalues(fit2); e <- residuals(fit2)
  press <- sum((e / (1 - h))^2)
  sst   <- sum((df_r[[y]][ok] - mean(df_r[[y]][ok]))^2)
  q2_2  <- round(1 - press / sst, 3)
  cat(sprintf("\n%s ~ %s + %s  (n=%d, Q²=%.3f)\n",
              oc, preds[1], preds[2], sum(ok), q2_2))
  print(summary(fit2)$coefficients %>% round(3))
}

# ==============================================================================
# SAVE
# ==============================================================================

write_csv(results, "/Users/calypsocalcott/LLM_Coordination/wvs_cultural_analysis/results_table.csv")
cat("\nSaved to results_table.csv\n")
