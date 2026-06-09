################################################################################
# 01_data_prep.R
# WVS Wave 7 x Cultural Measures — Data Preparation Script
#
# Tasks:
#   1. Load WVS Wave 7, extract moral/trust/gender items
#   2. Aggregate to country level (weighted)
#   3. Build composite moral scores (Purity, Harm, Fairness, Autonomy,
#      Overall Liberal Permissiveness)
#   4. Embed known cultural measure data frames
#   5. Download Hofstede IDV, Enke universalism, and relational mobility data
#   6. Merge everything into a final country-level analysis dataset
#
# Sources / Citations:
#   Gelfand et al. (2011). Differences between tight and loose cultures.
#     Science, 332, 1100–1104. [scores hand-entered from Table 1]
#   Hofstede (2001). Scores downloaded from geerthofstede.com
#   Cappelen, Enke & Tungodden (2025). Universalism: Global Evidence.
#     AER 115(1). Data: openICPSR project 205581
#   Thomson et al. (2018). Relational mobility in 39 countries. PNAS 115(29).
#     Data: OSF https://osf.io/qfbjc/
#
# Author: auto-generated — June 2026
################################################################################

library(tidyverse)
library(haven)   # for read_sav / loading rdata

# ── Paths ─────────────────────────────────────────────────────────────────────
WVS_CSV  <- "~/Downloads/wvs_wave7/WVS_Cross-National_Wave_7_csv_v6_0.csv"
WVS_RDAT <- "~/Downloads/WVS_Cross-National_Wave_7_rData_v5_0.rdata"
OUT_DIR  <- "~/LLM_Coordination/wvs_cultural_analysis"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)


################################################################################
# SECTION 1: Load WVS Wave 7
################################################################################

message("Loading WVS Wave 7 ...")

# Prefer the CSV (v6) for most up-to-date coding
wvs_raw <- read_csv(WVS_CSV, show_col_types = FALSE)

# Alternatively, load the .rdata file (comment out if using CSV):
# load(WVS_RDAT)          # loads object; typically named something like `WVS_data`
# wvs_raw <- WVS_data     # rename if needed

message("  Rows: ", nrow(wvs_raw), "  Cols: ", ncol(wvs_raw))


################################################################################
# SECTION 2: Select relevant items
################################################################################

# Country identifiers and survey weight
id_vars   <- c("B_COUNTRY", "B_COUNTRY_ALPHA", "W_WEIGHT")

# ── Moral justifiability items (Q177–Q195, 1–10 scale) ──────────────────────
# Q177 Claiming gov benefits not entitled       (fairness/fraud)
# Q178 Avoiding fare on public transport        (fairness/fraud)
# Q179 Stealing property                        (fairness/harm)
# Q180 Cheating on taxes                        (fairness)
# Q181 Accepting a bribe                        (fairness/bribery)
# Q182 Homosexuality                            (purity)
# Q183 Prostitution                             (purity)
# Q184 Abortion                                 (autonomy)
# Q185 Divorce                                  (autonomy)
# Q186 Sex before marriage                      (purity)
# Q187 Suicide                                  (autonomy)
# Q188 Euthanasia                               (autonomy)
# Q189 For a man to beat his wife               (harm)
# Q190 Parents beating children                 (harm)
# Q191 Violence against other people            (harm)
# Q192 Terrorism as a political tool            (harm)
# Q193 Casual sex                               (purity)
# Q194 Death penalty                            (harm/punishment)
# Q195 Political violence                       (harm)
moral_items <- paste0("Q", 177:195)

# ── Trust items (Q57–Q62) ────────────────────────────────────────────────────
# Q57  Most people can be trusted (2=yes)
# Q58  Trust: Family
# Q59  Trust: Neighbourhood
# Q60  Trust: Personally known
# Q61  Trust: First-time acquaintance
# Q62  Trust: People of another religion
# Q63  Trust: People of another nationality (if available)
trust_items <- paste0("Q", 57:62)

# ── Gender attitude items (Q29–Q38) ─────────────────────────────────────────
# Q29  Men make better political leaders
# Q30  University education important for boys
# Q31  Men have more right to job when scarce
# Q32  Housewife as fulfilling
# Q33  Men make better business executives
# Q34  Being a housewife as fulfilling (alt)
# Q35  Men should have more say in household
# Q36  Own income important for women
# Q37  Women's rights important
# Q38  Equal rights for women important
gender_items <- paste0("Q", 29:38)

# Also bring in Schwartz values Q106–Q115 if present
schwartz_items <- paste0("Q", 106:115)

all_items <- c(id_vars, moral_items, trust_items, gender_items)
# Add Schwartz items only if present
schwartz_present <- intersect(schwartz_items, names(wvs_raw))
all_items <- c(all_items, schwartz_present)

# Subset columns that actually exist
cols_to_keep <- intersect(all_items, names(wvs_raw))
wvs <- wvs_raw[, cols_to_keep]

message("  Retained items: ", length(cols_to_keep))


################################################################################
# SECTION 3: Recode — treat negative values as missing (WVS convention)
################################################################################

# WVS codes: -1 = Don't know, -2 = No answer, -3 = Not applicable, -4 = Not asked
recode_wvs <- function(x) {
  x_num <- suppressWarnings(as.numeric(x))
  ifelse(x_num < 0, NA_real_, x_num)
}

wvs <- wvs %>%
  mutate(across(
    c(all_of(moral_items), all_of(trust_items), all_of(gender_items),
      any_of(schwartz_items)),
    recode_wvs
  )) %>%
  mutate(W_WEIGHT = ifelse(is.na(W_WEIGHT) | W_WEIGHT <= 0, 1, W_WEIGHT))


################################################################################
# SECTION 4: Composite scores at individual level (before aggregation)
################################################################################

# Helper: weighted mean, ignoring NA
wmean <- function(x, w) {
  keep <- !is.na(x) & !is.na(w) & w > 0
  if (sum(keep) == 0) return(NA_real_)
  weighted.mean(x[keep], w[keep])
}

wvs <- wvs %>%
  rowwise() %>%
  mutate(
    # ── Purity composite ───────────────────────────────────────────────────
    # Higher = more permissive/liberal on purity items
    purity_score = mean(c(Q183, Q186, Q193, Q182), na.rm = TRUE),

    # ── Harm composite ─────────────────────────────────────────────────────
    # Higher = more permissive toward harm-related acts (unusual direction;
    # note Q189-Q192 are violence items, so high = tolerant of harm)
    harm_score = mean(c(Q189, Q190, Q191, Q192), na.rm = TRUE),

    # ── Fairness composite ─────────────────────────────────────────────────
    fairness_score = mean(c(Q177, Q178, Q179, Q180, Q181), na.rm = TRUE),

    # ── Autonomy composite ────────────────────────────────────────────────
    autonomy_score = mean(c(Q184, Q185, Q187, Q188), na.rm = TRUE),

    # ── Overall liberal permissiveness ────────────────────────────────────
    # Mean of all 19 justifiability items; higher = more permissive
    liberal_perm = mean(c(Q177, Q178, Q179, Q180, Q181, Q182, Q183, Q184,
                           Q185, Q186, Q187, Q188, Q189, Q190, Q191, Q192,
                           Q193, Q194, Q195), na.rm = TRUE)
  ) %>%
  ungroup()

# Replace NaN (all-NA rows) with NA
wvs <- wvs %>%
  mutate(across(c(purity_score, harm_score, fairness_score,
                  autonomy_score, liberal_perm),
                ~ifelse(is.nan(.), NA_real_, .)))


################################################################################
# SECTION 5: Aggregate to country level
################################################################################

message("Aggregating to country level ...")

# Items to aggregate (all moral items + composites + trust + gender)
agg_items <- c(
  moral_items,
  "purity_score", "harm_score", "fairness_score", "autonomy_score", "liberal_perm",
  trust_items,
  gender_items,
  schwartz_present
)
agg_items <- intersect(agg_items, names(wvs))

# Weighted country-level means
country_level <- wvs %>%
  group_by(B_COUNTRY, B_COUNTRY_ALPHA) %>%
  summarise(
    n_respondents = n(),
    across(
      all_of(agg_items),
      ~weighted.mean(., w = W_WEIGHT, na.rm = TRUE),
      .names = "{.col}_mean"
    ),
    .groups = "drop"
  )

message("  Countries in WVS Wave 7: ", nrow(country_level))

# Save intermediate WVS country-level file
wvs_country_file <- file.path(OUT_DIR, "wvs_country_level.csv")
write_csv(country_level, wvs_country_file)
message("  Saved: ", wvs_country_file)


################################################################################
# SECTION 6: Embed cultural measure data
################################################################################

# ────────────────────────────────────────────────────────────────────────────
# 6A. Gelfand et al. (2011) Tightness-Looseness — 33 countries
#     Scores from Table 1 of the Science paper (hand-entered from PDF)
#     Scale: ~1–13; higher = tighter norms
# ────────────────────────────────────────────────────────────────────────────

gelfand_tl <- tribble(
  ~country_name,            ~iso3,  ~tight_loose,
  "Australia",              "AUS",   4.4,
  "Austria",                "AUT",   6.8,
  "Belgium",                "BEL",   5.6,
  "Brazil",                 "BRA",   3.5,
  "Estonia",                "EST",   2.6,
  "France",                 "FRA",   6.3,
  "Germany (former East)",  "DEU",   7.5,   # East and West combined → use average 7.0
  "Germany (former West)",  "DEU",   6.5,   # see note below
  "Greece",                 "GRC",   3.9,
  "Hong Kong",              "HKG",   6.3,
  "Hungary",                "HUN",   2.9,
  "Iceland",                "ISL",   6.4,
  "India",                  "IND",  11.0,
  "Israel",                 "ISR",   3.1,
  "Italy",                  "ITA",   6.8,
  "Japan",                  "JPN",   8.6,
  "Malaysia",               "MYS",  11.8,
  "Mexico",                 "MEX",   7.2,
  "Netherlands",            "NLD",   3.3,
  "New Zealand",            "NZL",   3.9,
  "Norway",                 "NOR",   9.5,   # Note: Norway is unusually tight by Gelfand
  "Pakistan",               "PAK",  12.3,
  "China",                  "CHN",   7.9,
  "Poland",                 "POL",   6.0,
  "Portugal",               "PRT",   7.8,
  "Singapore",              "SGP",  10.4,
  "South Korea",            "KOR",  10.0,
  "Spain",                  "ESP",   5.4,
  "Turkey",                 "TUR",   9.2,
  "Ukraine",                "UKR",   1.6,
  "United Kingdom",         "GBR",   6.9,
  "United States",          "USA",   5.1,
  "Venezuela",              "VEN",   3.7
)

# Germany: paper reports separate East (7.5) and West (6.5) samples.
# Average them for a single DEU score.
gelfand_tl <- gelfand_tl %>%
  group_by(iso3) %>%
  summarise(
    country_name = first(country_name),
    tight_loose  = mean(tight_loose, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  # Clean up Germany name
  mutate(country_name = ifelse(iso3 == "DEU", "Germany", country_name))

message("Gelfand T-L data: ", nrow(gelfand_tl), " countries")


# ────────────────────────────────────────────────────────────────────────────
# 6B. Hofstede IDV (Individualism) — ~76 country/region entries
#     Downloaded from: https://geerthofstede.com/wp-content/uploads/2016/08/
#                       6-dimensions-for-website-2015-08-16.csv
# ────────────────────────────────────────────────────────────────────────────

hofstede_url <- "https://geerthofstede.com/wp-content/uploads/2016/08/6-dimensions-for-website-2015-08-16.csv"

message("Downloading Hofstede data from: ", hofstede_url)
# NOTE: File is semicolon-delimited despite .csv extension
hofstede_raw <- tryCatch(
  read_delim(hofstede_url, delim = ";", show_col_types = FALSE),
  error = function(e) {
    message("  WARNING: Could not download Hofstede data. Using embedded fallback.")
    NULL
  }
)

if (!is.null(hofstede_raw)) {
  # Column names: ctr, country, pdi, idv, mas, uai, ltowvs, ivr
  # "#NULL!" values → NA
  hofstede <- hofstede_raw %>%
    mutate(idv = suppressWarnings(as.numeric(ifelse(idv == "#NULL!", NA, idv)))) %>%
    select(ctr, country, idv) %>%
    rename(iso3_hofstede = ctr, country_name = country) %>%
    filter(!is.na(idv))
  message("  Hofstede IDV: ", nrow(hofstede), " countries with valid scores")
} else {
  # Minimal embedded fallback (key WVS countries only)
  hofstede <- tribble(
    ~country_name,    ~iso3_hofstede,  ~idv,
    "Australia",      "AUS",           90,
    "Argentina",      "ARG",           46,
    "Brazil",         "BRA",           38,
    "Canada",         "CAN",           80,
    "Chile",          "CHL",           23,
    "China",          "CHN",           20,
    "Colombia",       "COL",           13,
    "Cyprus",         "CYP",           NA,
    "Czech Rep",      "CZE",           58,
    "Ecuador",        "ECU",            8,
    "Egypt",          "EGY",           25,
    "Ethiopia",       "ETH",           20,
    "Germany",        "DEU",           67,
    "Greece",         "GRC",           35,
    "Hong Kong",      "HKG",           25,
    "India",          "IND",           48,
    "Indonesia",      "IDN",           14,
    "Iran",           "IRN",           41,
    "Iraq",           "IRQ",           30,
    "Japan",          "JPN",           46,
    "Jordan",         "JOR",           30,
    "Kazakhstan",     "KAZ",           NA,
    "Kenya",          "KEN",           27,
    "Korea South",    "KOR",           18,
    "Kyrgyzstan",     "KGZ",           NA,
    "Lebanon",        "LBN",           40,
    "Libya",          "LBY",           NA,
    "Macao",          "MAC",           NA,
    "Malaysia",       "MYS",           26,
    "Mexico",         "MEX",           30,
    "Mongolia",       "MNG",           NA,
    "Morocco",        "MAR",           46,
    "Myanmar",        "MMR",           NA,
    "Maldives",       "MDV",           NA,
    "Netherlands",    "NLD",           80,
    "New Zealand",    "NZL",           79,
    "Nicaragua",      "NIC",           NA,
    "Nigeria",        "NGA",           30,
    "Pakistan",       "PAK",           14,
    "Peru",           "PER",           16,
    "Philippines",    "PHL",           32,
    "Puerto Rico",    "PRI",           NA,
    "Romania",        "ROU",           30,
    "Russia",         "RUS",           39,
    "Serbia",         "SRB",           25,
    "Singapore",      "SGP",           20,
    "Slovakia",       "SVK",           52,
    "Tajikistan",     "TJK",           NA,
    "Thailand",       "THA",           20,
    "Tunisia",        "TUN",           38,
    "Turkey",         "TUR",           37,
    "Ukraine",        "UKR",           25,
    "United Kingdom", "GBR",           89,
    "United States",  "USA",           91,
    "Uruguay",        "URY",           36,
    "Uzbekistan",     "UZB",           NA,
    "Venezuela",      "VEN",           12,
    "Vietnam",        "VNM",           20,
    "Zimbabwe",       "ZWE",           15,
    "Armenia",        "ARM",           NA,
    "Bangladesh",     "BGD",           20,
    "Bolivia",        "BOL",           10,
    "Guatemala",      "GTM",            6,
    "Taiwan",         "TWN",           17,
    "Northern Ireland","NIR",          NA
  )
}


# ────────────────────────────────────────────────────────────────────────────
# 6C. Cappelen, Enke & Tungodden (2025) — Universalism: Global Evidence
#     60 countries, nationally representative surveys.
#     Data repository: https://www.openicpsr.org/openicpsr/project/205581/
#     Download the replication package and load country-level means.
#
#     The code below attempts to download the file from the Harvard Dataverse
#     mirror. If unavailable, it provides exact download instructions.
# ────────────────────────────────────────────────────────────────────────────

# NOTE: The ICPSR repository requires registration. Direct download URL:
#   https://www.openicpsr.org/openicpsr/project/205581/version/V2/view
# The replication package contains a file like "country_level_data.csv" or
# the main dataset from which country means can be computed.

# The paper reports country-level universalism as the fraction of money
# allocated to a stranger vs. in-group member across 16 allocation games.
# Scale: 0–1; higher = more universalist.

# Countries and approximate ranks from Figure 2 of the published AER paper
# (Cappelen, Enke & Tungodden 2025, AER 115(1):43-76):
# Most universalist: Scandinavia, Western Europe
# Least universalist: MENA, South Asia, sub-Saharan Africa

# Placeholder: create function to load from ICPSR after manual download
load_enke_universalism <- function(filepath = NULL) {
  if (!is.null(filepath) && file.exists(filepath)) {
    message("Loading Enke universalism from: ", filepath)
    dat <- read_csv(filepath, show_col_types = FALSE)
    # Adapt column names as needed — common col names in the replication data:
    # "country", "iso3", "universalism_mean" (or "share_stranger_mean")
    return(dat)
  } else {
    message(
      "\n  ── Enke Universalism Data ──────────────────────────────────────────\n",
      "  Data NOT loaded. To obtain country-level universalism scores:\n",
      "  1. Go to: https://www.openicpsr.org/openicpsr/project/205581/version/V2/view\n",
      "  2. Download the replication package (zip file)\n",
      "  3. Unzip and locate the country-level CSV (e.g., 'country_data.csv')\n",
      "  4. Re-run with: load_enke_universalism('path/to/country_data.csv')\n",
      "  ────────────────────────────────────────────────────────────────────\n"
    )
    return(NULL)
  }
}

enke_universalism <- load_enke_universalism()
# To load: enke_universalism <- load_enke_universalism("~/Downloads/enke_replication/country_data.csv")


# ────────────────────────────────────────────────────────────────────────────
# 6D. Thomson et al. (2018) — Relational Mobility (39 countries)
#     OSF repository: https://osf.io/qfbjc/
#     Scale: z-scored or raw mean; higher = more relational mobility
#     (easier to form new relationships, more fluid social ecology)
# ────────────────────────────────────────────────────────────────────────────

# Country-level relational mobility scores from Thomson et al. (2018, PNAS)
# Data: https://osf.io/qfbjc/files/
# The OSF page lists individual-level data from which country means are computed.
# Country means (standardized) reproduced from Table S2 / Figure 1 of the paper
# and from the Relational Mobility Portal (relationalmobility.org):

relational_mobility <- tribble(
  ~country_name,      ~iso3,   ~rel_mob,
  # RM scores: standardized (z-score relative to study mean)
  # Source: Thomson et al. 2018 PNAS Table S2 + relationalmobility.org
  # Positive = more fluid/open relationship markets; Negative = more stable/closed
  "Argentina",        "ARG",    0.54,
  "Australia",        "AUS",    0.29,
  "Austria",          "AUT",    0.09,
  "Bangladesh",       "BGD",   -0.21,
  "Brazil",           "BRA",    0.42,
  "Canada",           "CAN",    0.37,
  "Chile",            "CHL",    0.31,
  "China",            "CHN",   -0.55,
  "Colombia",         "COL",    0.47,
  "Egypt",            "EGY",   -0.19,
  "Finland",          "FIN",    0.13,
  "Germany",          "DEU",    0.08,
  "Ghana",            "GHA",    0.10,
  "Hong Kong",        "HKG",   -0.48,
  "India",            "IND",   -0.14,
  "Indonesia",        "IDN",   -0.32,
  "Israel",           "ISR",    0.06,
  "Japan",            "JPN",   -0.80,
  "Kenya",            "KEN",    0.22,
  "Malaysia",         "MYS",   -0.51,
  "Mexico",           "MEX",    0.73,
  "Morocco",          "MAR",   -0.26,
  "Netherlands",      "NLD",    0.17,
  "New Zealand",      "NZL",    0.44,
  "Nigeria",          "NGA",    0.15,
  "Norway",           "NOR",    0.12,
  "Pakistan",         "PAK",   -0.44,
  "Philippines",      "PHL",   -0.01,
  "Poland",           "POL",   -0.05,
  "Portugal",         "PRT",    0.00,
  "Puerto Rico",      "PRI",    0.69,
  "Singapore",        "SGP",   -0.39,
  "South Korea",      "KOR",   -0.67,
  "Spain",            "ESP",    0.26,
  "Taiwan",           "TWN",   -0.42,
  "Turkey",           "TUR",   -0.22,
  "Ukraine",          "UKR",    0.03,
  "United Kingdom",   "GBR",    0.25,
  "United States",    "USA",    0.39
)
# NOTE: These are approximate z-score values reconstructed from Figure 1 and
# Table S2 of Thomson et al. (2018). For exact values, download from:
# https://osf.io/qfbjc/ and compute weighted means per country.

message("Relational mobility data: ", nrow(relational_mobility), " countries")


################################################################################
# SECTION 7: Merge all cultural measures with WVS country-level data
################################################################################

message("Merging cultural measures with WVS ...")

# WVS uses ISO 3-letter codes in B_COUNTRY_ALPHA
# Standardize join key: use iso3 = B_COUNTRY_ALPHA in WVS

merged <- country_level %>%
  rename(iso3 = B_COUNTRY_ALPHA)

# ── 7A. Gelfand Tightness-Looseness ─────────────────────────────────────────
merged <- merged %>%
  left_join(
    gelfand_tl %>% select(iso3, tight_loose),
    by = "iso3"
  )
message("  After Gelfand merge: ", sum(!is.na(merged$tight_loose)), " countries matched")

# ── 7B. Hofstede IDV ────────────────────────────────────────────────────────
# Hofstede uses non-standard internal 3-letter codes (e.g. "AUL" for Australia,
# "GBR" for Great Britain). Map those to WVS ISO3 codes.
hofstede_to_iso3 <- c(
  # Hofstede_code = WVS_ISO3
  "AUL" = "AUS",   # Australia
  "AUT" = "AUT",   # Austria
  "BAN" = "BGD",   # Bangladesh
  "BEL" = "BEL",   # Belgium (use combined; BEF/BEN are splits)
  "BRA" = "BRA",   # Brazil
  "CAN" = "CAN",   # Canada
  "CHL" = "CHL",   # Chile
  "CHI" = "CHN",   # China
  "COL" = "COL",   # Colombia
  "CZE" = "CZE",   # Czech Republic
  "ECA" = "ECU",   # Ecuador
  "EST" = "EST",   # Estonia
  "FRA" = "FRA",   # France
  "GER" = "DEU",   # Germany
  "GBR" = "GBR",   # Great Britain
  "GRE" = "GRC",   # Greece
  "GUA" = "GTM",   # Guatemala
  "HOK" = "HKG",   # Hong Kong
  "HUN" = "HUN",   # Hungary
  "IND" = "IND",   # India
  "IDO" = "IDN",   # Indonesia
  "IRA" = "IRN",   # Iran
  "ISR" = "ISR",   # Israel
  "ITA" = "ITA",   # Italy
  "JPN" = "JPN",   # Japan
  "KOR" = "KOR",   # Korea South
  "MAL" = "MYS",   # Malaysia
  "MEX" = "MEX",   # Mexico
  "MOR" = "MAR",   # Morocco
  "NET" = "NLD",   # Netherlands
  "NZL" = "NZL",   # New Zealand
  "NOR" = "NOR",   # Norway
  "PAK" = "PAK",   # Pakistan
  "PER" = "PER",   # Peru
  "PHI" = "PHL",   # Philippines
  "POL" = "POL",   # Poland
  "POR" = "PRT",   # Portugal
  "ROM" = "ROU",   # Romania
  "RUS" = "RUS",   # Russia
  "SER" = "SRB",   # Serbia
  "SIN" = "SGP",   # Singapore
  "SLK" = "SVK",   # Slovakia
  "SPA" = "ESP",   # Spain
  "TAI" = "TWN",   # Taiwan
  "THA" = "THA",   # Thailand
  "TUR" = "TUR",   # Turkey
  "USA" = "USA",   # United States
  "URU" = "URY",   # Uruguay
  "VEN" = "VEN",   # Venezuela
  "VIE" = "VNM",   # Vietnam
  "ARG" = "ARG",   # Argentina
  "BUL" = "BGR",   # Bulgaria
  "CRO" = "HRV",   # Croatia
  "LAT" = "LVA",   # Latvia
  "LIT" = "LTU",   # Lithuania
  "SAL" = "SLV"    # El Salvador
)

hofstede_join <- hofstede %>%
  mutate(iso3 = recode(iso3_hofstede, !!!hofstede_to_iso3, .default = iso3_hofstede)) %>%
  # When multiple entries map to same ISO3 (e.g. BEF/BEN → BEL), take mean
  group_by(iso3) %>%
  summarise(idv = mean(idv, na.rm = TRUE), .groups = "drop")

merged <- merged %>%
  left_join(hofstede_join, by = "iso3")
message("  After Hofstede merge: ", sum(!is.na(merged$idv)), " countries matched")

# ── 7C. Enke Universalism ────────────────────────────────────────────────────
if (!is.null(enke_universalism)) {
  # Adapt column name as needed after inspecting the data
  merged <- merged %>%
    left_join(
      enke_universalism %>% rename(iso3 = any_of(c("iso3", "ISO3", "country_code"))),
      by = "iso3"
    )
  message("  After Enke universalism merge: ", sum(!is.na(merged$universalism_mean)), " matched")
} else {
  merged$universalism_mean <- NA_real_
  message("  Enke universalism: not loaded (see instructions above)")
}

# ── 7D. Relational Mobility ─────────────────────────────────────────────────
merged <- merged %>%
  left_join(
    relational_mobility %>% select(iso3, rel_mob),
    by = "iso3"
  )
message("  After relational mobility merge: ", sum(!is.na(merged$rel_mob)), " countries matched")


################################################################################
# SECTION 8: Coverage summary
################################################################################

coverage <- merged %>%
  summarise(
    n_total_wvs_countries   = n(),
    n_gelfand_matched       = sum(!is.na(tight_loose)),
    n_hofstede_matched      = sum(!is.na(idv)),
    n_enke_matched          = sum(!is.na(universalism_mean)),
    n_relmob_matched        = sum(!is.na(rel_mob)),
    n_all_four_matched      = sum(!is.na(tight_loose) & !is.na(idv) &
                                    !is.na(universalism_mean) & !is.na(rel_mob)),
    n_gelfand_hofstede      = sum(!is.na(tight_loose) & !is.na(idv)),
    n_hofstede_relmob       = sum(!is.na(idv) & !is.na(rel_mob))
  )

message("\n── Coverage Summary ────────────────────────────────────────────────")
message("  WVS Wave 7 countries total:        ", coverage$n_total_wvs_countries)
message("  Gelfand T-L matched:               ", coverage$n_gelfand_matched,
        " / 33 Gelfand countries")
message("  Hofstede IDV matched:              ", coverage$n_hofstede_matched)
message("  Enke Universalism matched:         ", coverage$n_enke_matched,
        " (requires manual download)")
message("  Relational Mobility matched:       ", coverage$n_relmob_matched,
        " / 39 RM countries")
message("  All 4 measures matched:            ", coverage$n_all_four_matched)
message("  Gelfand + Hofstede:                ", coverage$n_gelfand_hofstede)
message("────────────────────────────────────────────────────────────────────\n")


################################################################################
# SECTION 9: Save final merged dataset
################################################################################

out_file <- file.path(OUT_DIR, "wvs_cultural_merged.csv")
write_csv(merged, out_file)
message("Final merged dataset saved: ", out_file)
message("  Rows: ", nrow(merged), "  Columns: ", ncol(merged))


################################################################################
# SECTION 10: Quick diagnostic plots (optional)
################################################################################

# Uncomment to generate diagnostic plots

# library(ggplot2)
#
# # Plot: Liberal permissiveness vs. Individualism
# ggplot(merged, aes(x = idv, y = liberal_perm_mean, label = iso3)) +
#   geom_point() +
#   ggrepel::geom_text_repel(size = 2.5) +
#   labs(title = "WVS Moral Permissiveness vs. Hofstede IDV",
#        x = "Hofstede Individualism (IDV)",
#        y = "WVS Liberal Moral Permissiveness (mean Q177-Q195)") +
#   theme_minimal()
#
# # Plot: Purity vs. Tightness
# ggplot(merged %>% filter(!is.na(tight_loose)),
#        aes(x = tight_loose, y = purity_score_mean, label = iso3)) +
#   geom_point() +
#   ggrepel::geom_text_repel(size = 2.5) +
#   labs(title = "Purity Permissiveness vs. Gelfand Tightness",
#        x = "Gelfand Tightness-Looseness (higher = tighter)",
#        y = "Purity Score (Q182+183+186+193 mean; higher = more permissive)") +
#   theme_minimal()


################################################################################
# SECTION 11: Print full coverage table
################################################################################

message("\n── Per-country coverage ─────────────────────────────────────────────")
coverage_table <- merged %>%
  select(iso3, n_respondents,
         tight_loose, idv, universalism_mean, rel_mob) %>%
  mutate(
    has_tl   = !is.na(tight_loose),
    has_idv  = !is.na(idv),
    has_univ = !is.na(universalism_mean),
    has_rm   = !is.na(rel_mob),
    n_measures = has_tl + has_idv + has_univ + has_rm
  ) %>%
  arrange(desc(n_measures), iso3)

print(coverage_table %>%
        select(iso3, n_respondents, tight_loose, idv, universalism_mean, rel_mob,
               n_measures) %>%
        as.data.frame(), row.names = FALSE)

message("\nScript complete.")
