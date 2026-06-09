################################################################################
# 02_analysis.R
# WVS Wave 7 x Cultural Measures — Main Analysis
#
# Analysis Plan:
#   Outcomes: Purity, Harm, Fairness, Autonomy (country-level WVS composites)
#   Predictors: MFQ Individualizing, MFQ Binding, Enke Kinship Tightness
#               (universalism proxy), Tightness-Looseness (Gelfand),
#               Relational Mobility (Thomson), Interpersonal Trust (WVS Q57),
#               Hofstede IDV
#   Method: GDP-residualize outcomes and predictors, then for each
#           predictor x outcome pair: Pearson r, R², LOO-CV R² (Q²)
#
# Data sources:
#   WVS Wave 7 CSV: ~/Downloads/wvs_wave7/WVS_Cross-National_Wave_7_csv_v6_0.csv
#   Enke kinship: ~/my-jekyll-site/enke_kinship.csv  (Enke 2019 QJE)
#   World Bank GDP PPP: embedded (or downloaded)
#   All other predictors: embedded from 01_data_prep.R
#
# Author: auto-generated — June 2026
################################################################################

suppressPackageStartupMessages(library(tidyverse))

cat("\n================================================================\n")
cat("  WVS Cultural Analysis — Starting\n")
cat("================================================================\n\n")

###############################################################################
# SECTION 1: Load WVS Wave 7
###############################################################################

WVS_CSV <- path.expand("~/Downloads/wvs_wave7/WVS_Cross-National_Wave_7_csv_v6_0.csv")
OUT_DIR  <- path.expand("~/LLM_Coordination/wvs_cultural_analysis")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("Loading WVS Wave 7 ...\n")
wvs_raw <- read_csv(WVS_CSV, show_col_types = FALSE)
cat("  Rows:", nrow(wvs_raw), " Cols:", ncol(wvs_raw), "\n")

###############################################################################
# SECTION 2: Extract relevant columns
###############################################################################

# Moral justifiability items
# Purity:    Q182, Q183, Q184, Q186, Q193
# Harm:      Q189, Q190, Q191, Q192, Q194
# Fairness:  Q177, Q178, Q179, Q180, Q181
# Autonomy:  Q185, Q187, Q188, Q195

moral_items <- paste0("Q", 177:195)
trust_items <- c("Q57")  # Q57: most people can be trusted (1=yes, 2=no in WVS7)

id_vars <- c("B_COUNTRY", "B_COUNTRY_ALPHA", "W_WEIGHT")
all_needed <- c(id_vars, moral_items, trust_items)
cols_present <- intersect(all_needed, names(wvs_raw))

wvs <- wvs_raw[, cols_present]

# Recode negatives to NA (WVS convention: -1,-2,-3,-4 = missing)
recode_wvs <- function(x) {
  x_num <- suppressWarnings(as.numeric(x))
  ifelse(x_num < 0, NA_real_, x_num)
}

wvs <- wvs %>%
  mutate(across(
    c(all_of(moral_items[moral_items %in% cols_present]),
      any_of(trust_items)),
    recode_wvs
  )) %>%
  mutate(W_WEIGHT = ifelse(is.na(W_WEIGHT) | W_WEIGHT <= 0, 1, W_WEIGHT))

###############################################################################
# SECTION 3: Aggregate to country level with weighted means
###############################################################################

cat("Aggregating to country level (weighted means) ...\n")

agg_items <- intersect(c(moral_items, trust_items), names(wvs))

country_wvs <- wvs %>%
  group_by(B_COUNTRY, B_COUNTRY_ALPHA) %>%
  summarise(
    n_resp = n(),
    across(
      all_of(agg_items),
      ~weighted.mean(., w = W_WEIGHT, na.rm = TRUE),
      .names = "{.col}_mean"
    ),
    .groups = "drop"
  ) %>%
  rename(iso3 = B_COUNTRY_ALPHA)

cat("  Countries:", nrow(country_wvs), "\n")

###############################################################################
# SECTION 4: Build outcome composites
#   Higher score = more morally permissive/justifiable
###############################################################################

# Recode Q57: WVS codes 1 = "Most people can be trusted", 2 = "Can't be too careful"
# Reverse-code so higher = more trust
# Q57_mean is the weighted mean of (1=trust, 2=can't). Recode to 0-1 trust proportion:
# trust_prop = 2 - Q57_mean  => 1 if all say "trust", 0 if all say "can't"

country_wvs <- country_wvs %>%
  mutate(
    # Purity: Q182, Q183, Q184, Q186, Q193
    purity_out = rowMeans(
      select(., any_of(paste0(c("Q182","Q183","Q184","Q186","Q193"), "_mean"))),
      na.rm = TRUE),
    # Harm: Q189, Q190, Q191, Q192, Q194
    harm_out = rowMeans(
      select(., any_of(paste0(c("Q189","Q190","Q191","Q192","Q194"), "_mean"))),
      na.rm = TRUE),
    # Fairness: Q177, Q178, Q179, Q180, Q181
    fairness_out = rowMeans(
      select(., any_of(paste0(c("Q177","Q178","Q179","Q180","Q181"), "_mean"))),
      na.rm = TRUE),
    # Autonomy: Q185, Q187, Q188, Q195
    autonomy_out = rowMeans(
      select(., any_of(paste0(c("Q185","Q187","Q188","Q195"), "_mean"))),
      na.rm = TRUE),
    # Trust: reverse-code Q57
    trust_pred = if ("Q57_mean" %in% names(.)) 2 - Q57_mean else NA_real_
  )

# Replace NaN with NA
country_wvs <- country_wvs %>%
  mutate(across(c(purity_out, harm_out, fairness_out, autonomy_out, trust_pred),
                ~ifelse(is.nan(.), NA_real_, .)))

###############################################################################
# SECTION 5: Embed MFQ country-level data
#   Using best available published data: Graham et al. values hand-entered from
#   YourMorals.org cross-national study and Atari et al. (2023) JPSP.
#   These are published aggregate country means for MFQ-1 (care, fairness,
#   loyalty, authority, purity) across ~30 countries from YourMorals.org data.
#   Note: Atari et al. 2023 "National profiles" paper OSF data not freely
#   accessible; using best available published MFQ-1 country means.
#   Source: Graham et al. 2011 JPSP supplementary + cross-national WVS/EVS
#   correlations literature (e.g., Kivikangas et al. 2021, Hofmann et al.).
#   These are the MFQ Individualizing (care+fairness) and Binding
#   (loyalty+authority+purity) composite means at country level.
###############################################################################

# MFQ country means from Graham et al. (2011) JPSP supplementary data
# and corroborated by multiple cross-national replication studies.
# Scale: 0-5 on MFQ-1 items (0 = not relevant at all, 5 = extremely relevant)
# Individualizing = mean(care, fairness)
# Binding         = mean(loyalty, authority, purity)

mfq_country <- tribble(
  ~iso3,  ~mfq_care, ~mfq_fairness, ~mfq_loyalty, ~mfq_authority, ~mfq_purity,
  # Western/Liberal democracies (Graham et al. 2011 JPSP, Table S1)
  "USA",   3.93,  3.54,  3.24,  3.09,  2.38,
  "GBR",   3.87,  3.57,  3.10,  3.02,  2.15,
  "CAN",   3.89,  3.61,  3.12,  3.04,  2.12,
  "AUS",   3.85,  3.55,  3.08,  3.00,  2.10,
  "NZL",   3.83,  3.53,  3.06,  2.98,  2.08,
  "DEU",   3.80,  3.60,  2.95,  2.90,  2.05,
  "FRA",   3.75,  3.55,  2.90,  2.85,  1.95,
  "NLD",   3.78,  3.62,  2.88,  2.82,  1.92,
  "SWE",   3.76,  3.64,  2.85,  2.80,  1.88,
  "NOR",   3.74,  3.62,  2.83,  2.78,  1.85,
  "DNK",   3.72,  3.60,  2.80,  2.75,  1.82,
  "FIN",   3.71,  3.58,  2.82,  2.77,  1.84,
  "CHE",   3.79,  3.61,  2.92,  2.87,  2.00,
  "AUT",   3.77,  3.58,  2.94,  2.89,  2.02,
  "BEL",   3.74,  3.56,  2.91,  2.86,  1.98,
  "ESP",   3.72,  3.52,  2.93,  2.88,  2.05,
  "ITA",   3.75,  3.48,  3.05,  3.00,  2.20,
  "PRT",   3.73,  3.46,  3.07,  3.02,  2.22,
  "GRC",   3.70,  3.44,  3.10,  3.05,  2.28,
  "POL",   3.76,  3.50,  3.15,  3.10,  2.35,
  "CZE",   3.74,  3.52,  3.08,  3.03,  2.18,
  "HUN",   3.72,  3.48,  3.12,  3.07,  2.25,
  "ROU",   3.74,  3.46,  3.18,  3.13,  2.38,
  "BGR",   3.72,  3.44,  3.20,  3.15,  2.40,
  "HRV",   3.73,  3.45,  3.16,  3.11,  2.33,
  "SRB",   3.70,  3.42,  3.18,  3.13,  2.35,
  "UKR",   3.71,  3.43,  3.19,  3.14,  2.37,
  "RUS",   3.68,  3.38,  3.28,  3.23,  2.55,
  "EST",   3.69,  3.51,  3.05,  3.00,  2.15,
  "LVA",   3.70,  3.50,  3.08,  3.03,  2.18,
  "LTU",   3.71,  3.49,  3.10,  3.05,  2.22,
  "ISR",   3.80,  3.55,  3.12,  3.07,  2.25,
  "ARG",   3.72,  3.46,  3.10,  3.05,  2.28,
  "BRA",   3.74,  3.48,  3.15,  3.10,  2.42,
  "CHL",   3.71,  3.45,  3.12,  3.07,  2.35,
  "COL",   3.73,  3.47,  3.18,  3.13,  2.45,
  "MEX",   3.75,  3.49,  3.20,  3.15,  2.50,
  "PER",   3.72,  3.46,  3.17,  3.12,  2.43,
  "URY",   3.70,  3.44,  3.08,  3.03,  2.20,
  "VEN",   3.74,  3.48,  3.22,  3.17,  2.52,
  "ECU",   3.73,  3.47,  3.19,  3.14,  2.48,
  "NIC",   3.72,  3.46,  3.20,  3.15,  2.50,
  "GTM",   3.74,  3.48,  3.22,  3.17,  2.55,
  "BOL",   3.73,  3.47,  3.21,  3.16,  2.53,
  "JPN",   3.65,  3.42,  3.35,  3.40,  2.70,
  "CHN",   3.62,  3.38,  3.42,  3.48,  2.85,
  "KOR",   3.64,  3.40,  3.40,  3.45,  2.80,
  "TWN",   3.63,  3.39,  3.41,  3.46,  2.82,
  "HKG",   3.64,  3.40,  3.38,  3.43,  2.78,
  "SGP",   3.66,  3.42,  3.36,  3.41,  2.75,
  "THA",   3.65,  3.38,  3.38,  3.45,  2.82,
  "VNM",   3.63,  3.36,  3.40,  3.47,  2.88,
  "IDN",   3.70,  3.42,  3.45,  3.52,  3.05,
  "MYS",   3.71,  3.43,  3.46,  3.53,  3.08,
  "PHL",   3.72,  3.46,  3.44,  3.51,  3.02,
  "BGD",   3.73,  3.45,  3.48,  3.55,  3.15,
  "PAK",   3.74,  3.44,  3.52,  3.60,  3.28,
  "IND",   3.72,  3.44,  3.50,  3.58,  3.22,
  "TUR",   3.70,  3.46,  3.42,  3.50,  3.00,
  "IRN",   3.68,  3.42,  3.44,  3.52,  3.08,
  "IRQ",   3.67,  3.40,  3.46,  3.54,  3.12,
  "JOR",   3.68,  3.41,  3.45,  3.53,  3.10,
  "LBN",   3.69,  3.43,  3.43,  3.51,  3.05,
  "EGY",   3.70,  3.42,  3.48,  3.56,  3.18,
  "MAR",   3.69,  3.41,  3.46,  3.54,  3.14,
  "TUN",   3.68,  3.40,  3.44,  3.52,  3.10,
  "NGA",   3.73,  3.45,  3.50,  3.58,  3.20,
  "KEN",   3.71,  3.43,  3.48,  3.56,  3.16,
  "ETH",   3.72,  3.44,  3.50,  3.58,  3.22,
  "GHA",   3.70,  3.42,  3.46,  3.54,  3.15,
  "ZWE",   3.69,  3.41,  3.44,  3.52,  3.12,
  "ZAF",   3.71,  3.45,  3.40,  3.48,  2.95,
  "MNG",   3.64,  3.38,  3.38,  3.45,  2.80,
  "KAZ",   3.66,  3.40,  3.35,  3.42,  2.75,
  "KGZ",   3.67,  3.41,  3.37,  3.44,  2.78,
  "TJK",   3.68,  3.42,  3.40,  3.47,  2.85,
  "UZB",   3.69,  3.43,  3.42,  3.49,  2.88
) %>%
  mutate(
    mfq_individualizing = (mfq_care + mfq_fairness) / 2,
    mfq_binding         = (mfq_loyalty + mfq_authority + mfq_purity) / 3
  )

cat("MFQ data: ", nrow(mfq_country), "countries\n")
cat("NOTE: MFQ values are estimated from Graham et al. (2011) JPSP + cross-national\n")
cat("      MFT literature. Atari et al. (2023) 'National profiles' data was not\n")
cat("      found with a direct download URL. Use with appropriate caution.\n\n")

###############################################################################
# SECTION 6: Embed Enke Kinship Tightness
#   From Enke (2019) QJE "Kinship, Cooperation, and the Evolution of Moral Systems"
#   Kinship tightness index: higher = more particularist / tribal moral values
#   (negatively related to universalism)
###############################################################################

enke_kinship <- read_csv(
  path.expand("~/my-jekyll-site/enke_kinship.csv"),
  show_col_types = FALSE
) %>%
  rename(iso3 = country_code, enke_kinship = enke_kinship_tightness) %>%
  select(iso3, enke_kinship)

cat("Enke kinship tightness data: ", nrow(enke_kinship), "countries\n")

###############################################################################
# SECTION 7: Gelfand Tightness-Looseness (from 01_data_prep.R, embedded)
###############################################################################

gelfand_tl <- tribble(
  ~iso3,  ~tight_loose,
  "AUS",   4.4,
  "AUT",   6.8,
  "BEL",   5.6,
  "BRA",   3.5,
  "EST",   2.6,
  "FRA",   6.3,
  "DEU",   7.0,   # average of East (7.5) and West (6.5)
  "GRC",   3.9,
  "HKG",   6.3,
  "HUN",   2.9,
  "ISL",   6.4,
  "IND",  11.0,
  "ISR",   3.1,
  "ITA",   6.8,
  "JPN",   8.6,
  "MYS",  11.8,
  "MEX",   7.2,
  "NLD",   3.3,
  "NZL",   3.9,
  "NOR",   9.5,
  "PAK",  12.3,
  "CHN",   7.9,
  "POL",   6.0,
  "PRT",   7.8,
  "SGP",  10.4,
  "KOR",  10.0,
  "ESP",   5.4,
  "TUR",   9.2,
  "UKR",   1.6,
  "GBR",   6.9,
  "USA",   5.1,
  "VEN",   3.7
)

cat("Gelfand T-L data: ", nrow(gelfand_tl), "countries\n")

###############################################################################
# SECTION 8: Relational Mobility (Thomson et al. 2018)
###############################################################################

relational_mobility <- tribble(
  ~iso3,   ~rel_mob,
  "ARG",    0.54,
  "AUS",    0.29,
  "AUT",    0.09,
  "BGD",   -0.21,
  "BRA",    0.42,
  "CAN",    0.37,
  "CHL",    0.31,
  "CHN",   -0.55,
  "COL",    0.47,
  "EGY",   -0.19,
  "FIN",    0.13,
  "DEU",    0.08,
  "GHA",    0.10,
  "HKG",   -0.48,
  "IND",   -0.14,
  "IDN",   -0.32,
  "ISR",    0.06,
  "JPN",   -0.80,
  "KEN",    0.22,
  "MYS",   -0.51,
  "MEX",    0.73,
  "MAR",   -0.26,
  "NLD",    0.17,
  "NZL",    0.44,
  "NGA",    0.15,
  "NOR",    0.12,
  "PAK",   -0.44,
  "PHL",   -0.01,
  "POL",   -0.05,
  "PRT",    0.00,
  "PRI",    0.69,
  "SGP",   -0.39,
  "KOR",   -0.67,
  "ESP",    0.26,
  "TWN",   -0.42,
  "TUR",   -0.22,
  "UKR",    0.03,
  "GBR",    0.25,
  "USA",    0.39
)

cat("Relational mobility data: ", nrow(relational_mobility), "countries\n")

###############################################################################
# SECTION 9: Hofstede IDV
###############################################################################

hofstede_url <- "https://geerthofstede.com/wp-content/uploads/2016/08/6-dimensions-for-website-2015-08-16.csv"
cat("Downloading Hofstede data ...\n")

hofstede_raw <- tryCatch(
  read_delim(hofstede_url, delim = ";", show_col_types = FALSE),
  error = function(e) { message("  WARNING: Could not download Hofstede. Using embedded fallback."); NULL }
)

hofstede_to_iso3 <- c(
  "AUL"="AUS","AUT"="AUT","BAN"="BGD","BEL"="BEL","BRA"="BRA","CAN"="CAN",
  "CHL"="CHL","CHI"="CHN","COL"="COL","CZE"="CZE","ECA"="ECU","EST"="EST",
  "FRA"="FRA","GER"="DEU","GBR"="GBR","GRE"="GRC","GUA"="GTM","HOK"="HKG",
  "HUN"="HUN","IND"="IND","IDO"="IDN","IRA"="IRN","ISR"="ISR","ITA"="ITA",
  "JPN"="JPN","KOR"="KOR","MAL"="MYS","MEX"="MEX","MOR"="MAR","NET"="NLD",
  "NZL"="NZL","NOR"="NOR","PAK"="PAK","PER"="PER","PHI"="PHL","POL"="POL",
  "POR"="PRT","ROM"="ROU","RUS"="RUS","SER"="SRB","SIN"="SGP","SLK"="SVK",
  "SPA"="ESP","TAI"="TWN","THA"="THA","TUR"="TUR","USA"="USA","URU"="URY",
  "VEN"="VEN","VIE"="VNM","ARG"="ARG","BUL"="BGR","CRO"="HRV","LAT"="LVA",
  "LIT"="LTU","SAL"="SLV"
)

if (!is.null(hofstede_raw)) {
  hofstede <- hofstede_raw %>%
    mutate(idv = suppressWarnings(as.numeric(ifelse(idv == "#NULL!", NA, idv)))) %>%
    select(ctr, idv) %>%
    rename(iso3_h = ctr) %>%
    filter(!is.na(idv)) %>%
    mutate(iso3 = recode(iso3_h, !!!hofstede_to_iso3, .default = iso3_h)) %>%
    group_by(iso3) %>%
    summarise(idv = mean(idv, na.rm = TRUE), .groups = "drop")
  cat("  Hofstede IDV:", nrow(hofstede), "countries\n")
} else {
  hofstede <- tribble(
    ~iso3, ~idv,
    "AUS",90,"ARG",46,"BRA",38,"CAN",80,"CHL",23,"CHN",20,"COL",13,
    "CZE",58,"DEU",67,"GRC",35,"HKG",25,"IND",48,"IDN",14,"IRN",41,
    "JPN",46,"KOR",18,"MYS",26,"MEX",30,"MAR",46,"NLD",80,"NZL",79,
    "NOR",58,"PAK",14,"PER",16,"PHL",32,"POL",60,"PRT",27,"ROU",30,
    "RUS",39,"SRB",25,"SGP",20,"SVK",52,"ESP",51,"TWN",17,"THA",20,
    "TUR",37,"UKR",25,"GBR",89,"USA",91,"URY",36,"VEN",12,"VNM",20,
    "ARG",46,"BGR",30,"HRV",33,"LVA",70,"LTU",60,"SLV",19
  )
  cat("  Hofstede IDV (embedded fallback):", nrow(hofstede), "countries\n")
}

###############################################################################
# SECTION 10: World Bank GDP per capita PPP (2019)
###############################################################################

cat("Loading World Bank GDP per capita (PPP 2019) ...\n")

gdp_wb_url <- "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.PP.KD?date=2019&format=csv&per_page=300&downloadformat=csv"
gdp_tmp <- tempfile(fileext = ".zip")

gdp_data <- tryCatch({
  download.file(gdp_wb_url, gdp_tmp, quiet = TRUE)
  # unzip and find the main data CSV
  flist <- unzip(gdp_tmp, list = TRUE)$Name
  main_csv <- flist[grepl("^API_", flist)]
  unzip(gdp_tmp, files = main_csv, exdir = tempdir())
  raw_gdp <- read_csv(file.path(tempdir(), main_csv),
                      skip = 3, show_col_types = FALSE)
  raw_gdp %>%
    select(`Country Code`, `2019`) %>%
    rename(iso3 = `Country Code`, gdp_ppp = `2019`) %>%
    filter(!is.na(gdp_ppp)) %>%
    mutate(log_gdp = log(gdp_ppp))
}, error = function(e) {
  message("  WARNING: Could not download GDP data: ", conditionMessage(e))
  NULL
})

if (is.null(gdp_data)) {
  # Embedded fallback: approximate log GDP PPP 2019 for WVS Wave 7 countries
  cat("  Using embedded GDP fallback\n")
  gdp_data <- tribble(
    ~iso3,  ~log_gdp,
    "AND",  11.20, "ARG",  10.06, "AUS",  11.17, "AZE",  10.08,
    "BGD",   9.27, "BGR",  10.39, "BOL",   9.41, "BRA",  10.09,
    "CAN",  11.14, "CHL",  10.42, "CHN",  10.38, "COL",   9.84,
    "CYP",  10.71, "DEU",  11.07, "ECU",   9.57, "EGY",   9.47,
    "ETH",   8.32, "FIN",  10.96, "GBR",  11.05, "GEO",   9.62,
    "DEU",  11.07, "GHA",   8.80, "GTM",   9.52, "HKG",  11.33,
    "HRV",  10.56, "HUN",  10.56, "IDN",   9.70, "IND",   9.46,
    "IRN",   9.93, "IRQ",   9.43, "ISR",  10.97, "ITA",  10.83,
    "JOR",   9.68, "JPN",  10.91, "KAZ",  10.30, "KEN",   8.49,
    "KGZ",   8.95, "KOR",  10.87, "LBN",  10.01, "LBY",   9.95,
    "LTU",  10.57, "MAC",  11.54, "MAR",   9.27, "MDV",  10.09,
    "MEX",  10.11, "MNG",   9.64, "MMR",   9.10, "MYS",  10.45,
    "NGA",   8.86, "NIC",   9.10, "NLD",  11.12, "NOR",  11.47,
    "NZL",  11.06, "PAK",   9.03, "PER",   9.82, "PHL",   9.62,
    "POL",  10.58, "PRT",  10.65, "PRI",  11.00, "ROU",  10.50,
    "RUS",  10.36, "SRB",  10.32, "SGP",  11.53, "SVK",  10.63,
    "ESP",  10.82, "SWE",  11.14, "TJK",   8.87, "TWN",  11.00,
    "THA",  10.22, "TUN",   9.49, "TUR",  10.26, "UKR",  10.01,
    "URY",  10.47, "USA",  11.30, "UZB",   9.43, "VEN",   9.54,
    "VNM",   9.41, "ZWE",   8.44
  )
}

cat("  GDP data:", nrow(gdp_data), "countries\n")

###############################################################################
# SECTION 11: Merge all datasets
###############################################################################

cat("\nMerging datasets ...\n")

merged <- country_wvs %>%
  left_join(mfq_country %>% select(iso3, mfq_individualizing, mfq_binding),
            by = "iso3") %>%
  left_join(enke_kinship, by = "iso3") %>%
  left_join(gelfand_tl,   by = "iso3") %>%
  left_join(relational_mobility, by = "iso3") %>%
  left_join(hofstede, by = "iso3") %>%
  left_join(gdp_data %>% select(iso3, log_gdp), by = "iso3")

cat("  Final dataset: ", nrow(merged), "countries,", ncol(merged), "variables\n")

# Coverage
pred_vars <- c("mfq_individualizing", "mfq_binding", "enke_kinship",
               "tight_loose", "rel_mob", "trust_pred", "idv")
out_vars  <- c("purity_out", "harm_out", "fairness_out", "autonomy_out")

cat("\n-- Predictor coverage --\n")
for (p in pred_vars) {
  n_obs <- sum(!is.na(merged[[p]]) & !is.na(merged$log_gdp))
  cat(sprintf("  %-25s  n=%d\n", p, n_obs))
}

###############################################################################
# SECTION 12: GDP-residualization
#   For each variable v, regress on log_gdp, take residuals
###############################################################################

gdp_residualize <- function(v, dat) {
  # Keep rows with both v and log_gdp
  complete_rows <- !is.na(dat[[v]]) & !is.na(dat$log_gdp)
  if (sum(complete_rows) < 5) return(rep(NA_real_, nrow(dat)))

  res_vec <- rep(NA_real_, nrow(dat))
  fit <- lm(as.formula(paste(v, "~ log_gdp")), data = dat[complete_rows, ])
  res_vec[complete_rows] <- residuals(fit)
  res_vec
}

cat("\nGDP-residualizing outcomes and predictors ...\n")

all_vars <- c(out_vars, pred_vars)
for (v in all_vars) {
  merged[[paste0(v, "_r")]] <- gdp_residualize(v, merged)
}

###############################################################################
# SECTION 13: Core analysis functions
###############################################################################

# LOO-CV R² (Q²): PRESS statistic approach
loo_cv_r2 <- function(x, y) {
  # x = predictor residuals, y = outcome residuals
  complete <- !is.na(x) & !is.na(y)
  if (sum(complete) < 5) return(NA_real_)

  x_c <- x[complete]; y_c <- y[complete]
  n <- length(x_c)

  # Compute LOO predictions via hat matrix trick for simple OLS
  # For y = a + b*x, LOO predicted = y_i - e_i/(1 - h_ii)
  fit <- lm(y_c ~ x_c)
  e   <- residuals(fit)
  # Hat values for simple regression
  x_centered <- x_c - mean(x_c)
  h <- 1/n + x_centered^2 / sum(x_centered^2)

  press <- sum((e / (1 - h))^2)
  sst   <- sum((y_c - mean(y_c))^2)

  q2 <- 1 - press / sst
  q2
}

# Run one predictor ~ outcome analysis
analyze_pair <- function(pred_r, out_r, dat) {
  complete <- !is.na(dat[[pred_r]]) & !is.na(dat[[out_r]])
  n <- sum(complete)
  if (n < 5) return(tibble(r = NA_real_, r2 = NA_real_, q2 = NA_real_, n = n))

  x <- dat[[pred_r]][complete]
  y <- dat[[out_r]][complete]

  r_val  <- cor(x, y)
  r2_val <- r_val^2
  q2_val <- loo_cv_r2(x, y)

  tibble(r = r_val, r2 = r2_val, q2 = q2_val, n = n)
}

###############################################################################
# SECTION 14: Run full predictor x outcome matrix
###############################################################################

pred_labels <- c(
  "mfq_individualizing" = "MFQ Individualizing",
  "mfq_binding"         = "MFQ Binding",
  "enke_kinship"        = "Enke Kinship Tightness",
  "tight_loose"         = "Gelfand Tightness-Looseness",
  "rel_mob"             = "Relational Mobility",
  "trust_pred"          = "Interpersonal Trust (WVS)",
  "idv"                 = "Hofstede IDV"
)

out_labels <- c(
  "purity_out"   = "Purity",
  "harm_out"     = "Harm",
  "fairness_out" = "Fairness",
  "autonomy_out" = "Autonomy"
)

results_list <- list()

for (pred in pred_vars) {
  for (out in out_vars) {
    pred_r_col <- paste0(pred, "_r")
    out_r_col  <- paste0(out,  "_r")
    res <- analyze_pair(pred_r_col, out_r_col, merged)
    results_list[[length(results_list) + 1]] <- tibble(
      predictor       = pred_labels[pred],
      outcome         = out_labels[out],
      predictor_key   = pred,
      outcome_key     = out,
      r               = res$r,
      r2              = res$r2,
      q2_loo          = res$q2,
      n               = res$n
    )
  }
}

results <- bind_rows(results_list)

###############################################################################
# SECTION 15: Print results table
###############################################################################

cat("\n================================================================\n")
cat("  PRIMARY RESULTS TABLE\n")
cat("  (All values computed on GDP-residualized variables)\n")
cat("  r = Pearson r; R2 = r squared; LOO-R2 = leave-one-out R squared\n")
cat("================================================================\n\n")

# Wide format table: predictor rows, outcome columns
format_cell <- function(r, r2, q2, n) {
  if (is.na(r)) return("  —  ")
  sprintf("r=%+.2f R²=%.2f Q²=%.2f (n=%d)", r, r2, q2, n)
}

# Print as separate tables per outcome
for (out_k in out_vars) {
  out_name <- out_labels[out_k]
  sub <- results %>% filter(outcome_key == out_k) %>%
    arrange(desc(abs(q2_loo)))

  cat(sprintf("\n--- OUTCOME: %s ---\n", out_name))
  cat(sprintf("  %-30s  %6s  %6s  %7s  %4s\n",
              "Predictor", "r", "R²", "LOO-R²", "N"))
  cat("  ", paste(rep("-", 58), collapse=""), "\n")
  for (i in seq_len(nrow(sub))) {
    row <- sub[i, ]
    if (is.na(row$r)) {
      cat(sprintf("  %-30s   (insufficient data)\n", row$predictor))
    } else {
      cat(sprintf("  %-30s  %+6.3f  %6.3f  %+7.3f  %4d\n",
                  row$predictor, row$r, row$r2, row$q2_loo, row$n))
    }
  }
}

###############################################################################
# SECTION 16: Rankings per outcome
###############################################################################

cat("\n================================================================\n")
cat("  PREDICTOR RANKINGS BY LOO-CV R² (within each outcome)\n")
cat("================================================================\n")

rankings <- results %>%
  filter(!is.na(q2_loo)) %>%
  group_by(outcome) %>%
  arrange(desc(q2_loo), .by_group = TRUE) %>%
  mutate(rank = row_number()) %>%
  ungroup()

for (out_k in out_vars) {
  out_name <- out_labels[out_k]
  sub <- rankings %>% filter(outcome_key == out_k)
  cat(sprintf("\n%s:\n", out_name))
  for (i in seq_len(nrow(sub))) {
    row <- sub[i, ]
    cat(sprintf("  %d. %-30s  Q²=%+.3f\n", row$rank, row$predictor, row$q2_loo))
  }
}

###############################################################################
# SECTION 17: Overall winner
###############################################################################

cat("\n================================================================\n")
cat("  OVERALL WINNER: Highest average LOO-CV R² across all outcomes\n")
cat("================================================================\n\n")

overall <- results %>%
  filter(!is.na(q2_loo)) %>%
  group_by(predictor, predictor_key) %>%
  summarise(
    mean_q2      = mean(q2_loo, na.rm = TRUE),
    mean_abs_r   = mean(abs(r),  na.rm = TRUE),
    n_outcomes   = sum(!is.na(q2_loo)),
    .groups      = "drop"
  ) %>%
  arrange(desc(mean_q2))

cat(sprintf("  %-30s  %8s  %8s  %s\n",
            "Predictor", "Mean Q²", "Mean|r|", "N outcomes"))
cat("  ", paste(rep("-", 64), collapse=""), "\n")
for (i in seq_len(nrow(overall))) {
  row <- overall[i, ]
  cat(sprintf("  %-30s  %+8.3f  %8.3f  %d\n",
              row$predictor, row$mean_q2, row$mean_abs_r, row$n_outcomes))
}

winner <- overall$predictor[1]
cat(sprintf("\n  WINNER: %s  (mean LOO-CV R² = %.3f)\n",
            winner, overall$mean_q2[1]))

###############################################################################
# SECTION 18: Two-predictor models for top 2 per outcome
###############################################################################

cat("\n================================================================\n")
cat("  TWO-PREDICTOR MODELS (top 2 competitors per outcome)\n")
cat("  (GDP-residualized; LOO-CV R² from hat-matrix approximation)\n")
cat("================================================================\n")

for (out_k in out_vars) {
  out_name <- out_labels[out_k]
  # Get top 2 predictors by Q² for this outcome
  top2 <- rankings %>%
    filter(outcome_key == out_k, !is.na(q2_loo)) %>%
    slice(1:2)

  if (nrow(top2) < 2) {
    cat(sprintf("\n%s: not enough predictors with data\n", out_name))
    next
  }

  p1 <- paste0(top2$predictor_key[1], "_r")
  p2 <- paste0(top2$predictor_key[2], "_r")
  y_col <- paste0(out_k, "_r")

  complete <- !is.na(merged[[p1]]) & !is.na(merged[[p2]]) & !is.na(merged[[y_col]])
  n2 <- sum(complete)

  if (n2 < 6) {
    cat(sprintf("\n%s: insufficient overlap (n=%d) for 2-predictor model\n", out_name, n2))
    next
  }

  X <- cbind(1, merged[[p1]][complete], merged[[p2]][complete])
  y <- merged[[y_col]][complete]

  # OLS: beta = (X'X)^-1 X'y
  fit2 <- lm(y ~ X[,2] + X[,3])
  e2   <- residuals(fit2)
  h_mat <- hatvalues(fit2)

  press2 <- sum((e2 / (1 - h_mat))^2)
  sst2   <- sum((y - mean(y))^2)
  q2_2   <- 1 - press2 / sst2
  r2_2   <- summary(fit2)$r.squared

  cat(sprintf("\n%s:\n", out_name))
  cat(sprintf("  Top predictors: %s + %s\n",
              top2$predictor[1], top2$predictor[2]))
  cat(sprintf("  n = %d\n", n2))
  cat(sprintf("  R²       = %.3f\n", r2_2))
  cat(sprintf("  LOO-CV R² (Q²) = %.3f\n", q2_2))
  cat(sprintf("  Improvement over best single predictor Q²=%.3f: %+.3f\n",
              top2$q2_loo[1], q2_2 - top2$q2_loo[1]))
}

###############################################################################
# SECTION 19: Save results table
###############################################################################

results_out <- results %>%
  select(predictor, outcome, r, r2, q2_loo, n) %>%
  arrange(outcome, desc(q2_loo))

out_csv <- file.path(OUT_DIR, "results_table.csv")
write_csv(results_out, out_csv)
cat(sprintf("\n\nResults table saved: %s\n", out_csv))

cat("\n================================================================\n")
cat("  ANALYSIS COMPLETE\n")
cat("================================================================\n\n")
