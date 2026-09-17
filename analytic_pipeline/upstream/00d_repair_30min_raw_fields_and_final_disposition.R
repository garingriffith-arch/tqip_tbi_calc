# =============================================================================
# 00d_repair_30min_raw_fields_and_final_disposition.R
# TBI-TRACT upstream repair: raw PUF_TRAUMA -> project-local repaired core
#
# PURPOSE
# -------
# Recover early fields that were dropped from the harmonized patient core and
# repair final disposition for patients discharged directly from the ED.
#
# This script DOES NOT overwrite TQIP Harmony. It creates:
#   data/derived/tbi_tract_patient_core_repaired_2020_2024.rds
#
# Recovered raw PUF fields:
#   - EDDISCHARGEDISPOSITION / BIU
#   - all PROTDEV_* flattened protective-device fields
#   - GCSQ_* assessment qualifier fields
#   - SUPPLEMENTALOXYGEN / BIU
#   - TEMPERATURE / BIU
#   - AGEYEARS (QC only)
#
# Final disposition repair for AY 2020-2024:
#   Existing harmonized hospital discharge_group is retained when present.
#   If missing:
#       ED 4 or 9  -> Home/home health
#       ED 5       -> Death/hospice
#       ED 10      -> Left AMA                    [project exclusion]
#       ED 11      -> Short-term hospital transfer [project exclusion]
#       ED 6       -> ED other/institutional/custody [project exclusion]
#       ED 1/2/3/7/8 or unresolved -> unresolved QC
#
# Protective-device repair:
#   Helmet worn / No helmet worn / Unknown-not-recorded
#
# IMPORTANT
# ---------
# The raw PUF source is searched automatically. The script expects one
# PUF_TRAUMA.csv for each admission year 2020-2024 somewhere under the usual
# TQIP project/Harmony roots.
#
# =============================================================================

rm(list = ls())
gc()

suppressPackageStartupMessages({
  library(data.table)
})

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

config_candidates <- c(
  file.path(getwd(), "R", "00_config.R"),
  "R/00_config.R"
)
config_file <- config_candidates[file.exists(config_candidates)][1]

if (is.na(config_file) || length(config_file) == 0L) {
  stop("Could not find R/00_config.R. Run from the TBI-TRACT project root.", call. = FALSE)
}
source(config_file)

repair_years <- 2020:2024
repair_dir <- file.path(output_dir, "repair_30min_raw_fields")
dir.create(repair_dir, recursive = TRUE, showWarnings = FALSE)

repaired_core_file <- file.path(
  data_out_dir,
  "tbi_tract_patient_core_repaired_2020_2024.rds"
)

# Workstation threading.
logical_cores <- parallel::detectCores(logical = TRUE)
if (is.na(logical_cores)) logical_cores <- 32L
threads <- max(1L, min(28L, logical_cores - 4L))
data.table::setDTthreads(threads)

cat("\n============================================================\n")
cat("TBI-TRACT RAW-FIELD / DISPOSITION REPAIR\n")
cat("============================================================\n")
cat("Threads: ", threads, "\n", sep = "")
cat("Harmony source will NOT be modified.\n")
cat("Output repaired core:\n  ", repaired_core_file, "\n\n", sep = "")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

first_existing <- function(paths) {
  hit <- paths[file.exists(paths)]
  if (length(hit) == 0L) return(NA_character_)
  normalizePath(hit[1], winslash = "/", mustWork = TRUE)
}

normalize_inc_key <- function(x) {
  x <- as.character(x)
  x <- trimws(x)
  x <- gsub("\\.0$", "", x)
  x[x %in% c("", "NA", "NaN", "<NA>")] <- NA_character_
  x
}

safe_num <- function(x) suppressWarnings(as.numeric(as.character(x)))

norm_text <- function(x) {
  tolower(trimws(as.character(x)))
}

is_yes_flag <- function(x) {
  y <- norm_text(x)
  yn <- safe_num(x)
  (!is.na(yn) & yn == 1) |
    y %in% c("yes", "y", "true", "t", "present", "selected")
}

is_no_flag <- function(x) {
  y <- norm_text(x)
  yn <- safe_num(x)
  (!is.na(yn) & yn == 0) |
    y %in% c("no", "n", "false", "f", "not selected")
}

extract_year_from_path <- function(path) {
  m <- regexpr("(?<![0-9])(2020|2021|2022|2023|2024)(?![0-9])",
               path, perl = TRUE)
  if (m[1] == -1L) return(NA_integer_)
  as.integer(regmatches(path, m))
}

header_names <- function(path) {
  tryCatch(
    names(fread(path, nrows = 0, showProgress = FALSE)),
    error = function(e) character()
  )
}

# -----------------------------------------------------------------------------
# Locate harmonized patient core
# -----------------------------------------------------------------------------

patient_core_file <- first_existing(c(
  file.path(warehouse_dir, "data", "harmonized_patient_core_2007_2024.rds"),
  file.path(warehouse_dir, "harmonized_patient_core_2007_2024.rds")
))

if (is.na(patient_core_file)) {
  stop("Could not find harmonized_patient_core_2007_2024.rds.", call. = FALSE)
}

# -----------------------------------------------------------------------------
# Locate raw yearly PUF_TRAUMA.csv files
# -----------------------------------------------------------------------------

raw_search_roots <- unique(c(
  project_dir,
  data_dir,
  dirname(warehouse_dir),
  "C:/Users/garin/OneDrive/OHSU/Research/TQIP/Griffith_G-selected",
  "C:/OneDrive/garingriffith/OneDrive/OHSU/Research/TQIP/Griffith_G-selected",
  "C:/Users/garin/OneDrive/OHSU/Research/TQIP/TQIP Harmony",
  "C:/OneDrive/garingriffith/OneDrive/OHSU/Research/TQIP/TQIP Harmony"
))

raw_search_roots <- raw_search_roots[dir.exists(raw_search_roots)]

cat("Searching for PUF_TRAUMA.csv under:\n")
cat(paste0("  ", raw_search_roots, collapse = "\n"), "\n\n")

trauma_files <- unique(unlist(lapply(raw_search_roots, function(root) {
  list.files(
    root,
    pattern = "^PUF_TRAUMA\\.csv$",
    recursive = TRUE,
    full.names = TRUE,
    ignore.case = TRUE
  )
})))

if (length(trauma_files) == 0L) {
  stop(
    "No PUF_TRAUMA.csv files found under the configured TQIP roots.\n",
    "Add the correct raw PUF root to raw_search_roots in this script.",
    call. = FALSE
  )
}

desired_fixed <- c(
  "INC_KEY",
  "AGEYEARS",
  "EDDISCHARGEDISPOSITION",
  "EDDISCHARGEDISPOSITION_BIU",
  "GCSQ_INTUBATED",
  "GCSQ_SEDATEDPARALYZED",
  "GCSQ_EYEOBSTRUCTION",
  "GCSQ_VALID",
  "GCSQ_UK",
  "GCSQ_NA",
  "SUPPLEMENTALOXYGEN",
  "SUPPLEMENTALOXYGEN_BIU",
  "TEMPERATURE",
  "TEMPERATURE_BIU"
)

candidate_inventory <- rbindlist(lapply(trauma_files, function(f) {
  nm <- header_names(f)
  yr <- extract_year_from_path(f)
  prot <- grep("^PROTDEV_", nm, value = TRUE, ignore.case = TRUE)

  data.table(
    path = normalizePath(f, winslash = "/", mustWork = FALSE),
    admission_year = yr,
    file_size_gb = file.info(f)$size / 1024^3,
    n_columns = length(nm),
    n_desired_fixed_present = sum(toupper(desired_fixed) %in% toupper(nm)),
    n_protdev_columns = length(prot),
    has_inc_key = any(toupper(nm) == "INC_KEY"),
    has_ed_disposition = any(toupper(nm) == "EDDISCHARGEDISPOSITION"),
    has_gcs_qualifiers = all(
      c("GCSQ_INTUBATED", "GCSQ_SEDATEDPARALYZED", "GCSQ_EYEOBSTRUCTION") %in%
        toupper(nm)
    ),
    has_supplemental_o2 = any(toupper(nm) == "SUPPLEMENTALOXYGEN"),
    has_temperature = any(toupper(nm) == "TEMPERATURE")
  )
}), fill = TRUE)

fwrite(
  candidate_inventory[order(admission_year, -n_desired_fixed_present, -n_protdev_columns, -file_size_gb)],
  file.path(repair_dir, "00_raw_PUF_TRAUMA_candidate_inventory.csv")
)

# Select one best source per year.
selected_files <- rbindlist(lapply(repair_years, function(yr) {
  cand <- candidate_inventory[
    admission_year == yr & has_inc_key == TRUE
  ][
    order(-n_desired_fixed_present, -n_protdev_columns, -file_size_gb, path)
  ]

  if (nrow(cand) == 0L) {
    return(data.table(
      admission_year = yr,
      selected_path = NA_character_,
      n_desired_fixed_present = NA_integer_,
      n_protdev_columns = NA_integer_
    ))
  }

  cand[1, .(
    admission_year,
    selected_path = path,
    n_desired_fixed_present,
    n_protdev_columns
  )]
}))

fwrite(
  selected_files,
  file.path(repair_dir, "01_selected_raw_PUF_TRAUMA_by_year.csv")
)

if (anyNA(selected_files$selected_path)) {
  stop(
    "Could not identify a raw PUF_TRAUMA.csv for every year 2020-2024.\n",
    "See outputs/repair_30min_raw_fields/01_selected_raw_PUF_TRAUMA_by_year.csv",
    call. = FALSE
  )
}

if (any(selected_files$n_desired_fixed_present < 10L, na.rm = TRUE)) {
  warning(
    "At least one selected raw PUF file is missing several requested fields. ",
    "The script will proceed and QC missing fields explicitly."
  )
}

# -----------------------------------------------------------------------------
# Read only needed raw fields from selected sources
# -----------------------------------------------------------------------------

raw_list <- vector("list", nrow(selected_files))

for (i in seq_len(nrow(selected_files))) {
  yr <- selected_files$admission_year[i]
  f <- selected_files$selected_path[i]

  cat("Reading raw PUF_TRAUMA ", yr, ":\n  ", f, "\n", sep = "")

  nm <- header_names(f)

  # Preserve original case from source header.
  fixed_present <- nm[toupper(nm) %in% toupper(desired_fixed)]
  prot_present <- grep("^PROTDEV_", nm, value = TRUE, ignore.case = TRUE)
  select_cols <- unique(c(fixed_present, prot_present))

  if (!any(toupper(select_cols) == "INC_KEY")) {
    stop("Selected raw file lacks INC_KEY: ", f, call. = FALSE)
  }

  d <- fread(
    f,
    select = select_cols,
    showProgress = TRUE,
    integer64 = "character",
    na.strings = c("", "NA")
  )

  # Standardize raw names to uppercase.
  setnames(d, names(d), toupper(names(d)))
  d[, admission_year := yr]
  d[, inc_key := normalize_inc_key(INC_KEY)]

  if (anyDuplicated(d$inc_key[!is.na(d$inc_key)]) > 0L) {
    dup_n <- d[!is.na(inc_key), .N, by = inc_key][N > 1, .N]
    stop(
      "Duplicate INC_KEY values within selected ", yr,
      " PUF_TRAUMA source (", dup_n, " duplicate keys).",
      call. = FALSE
    )
  }

  raw_list[[i]] <- d
  rm(d)
  gc()
}

raw <- rbindlist(raw_list, fill = TRUE, use.names = TRUE)
rm(raw_list)
gc()

raw[, INC_KEY := NULL]

# -----------------------------------------------------------------------------
# Build clean raw derivatives
# -----------------------------------------------------------------------------

# ED disposition.
if ("EDDISCHARGEDISPOSITION" %in% names(raw)) {
  raw[, ed_disposition_code_recovered := as.integer(safe_num(EDDISCHARGEDISPOSITION))]
} else {
  raw[, ed_disposition_code_recovered := NA_integer_]
}

if ("EDDISCHARGEDISPOSITION_BIU" %in% names(raw)) {
  raw[, ed_disposition_biu_recovered := as.character(EDDISCHARGEDISPOSITION_BIU)]
} else {
  raw[, ed_disposition_biu_recovered := NA_character_]
}

# Raw age QC.
if ("AGEYEARS" %in% names(raw)) {
  raw[, ageyears_raw_recovered := safe_num(AGEYEARS)]
} else {
  raw[, ageyears_raw_recovered := NA_real_]
}

# Protective devices.
prot_cols <- grep("^PROTDEV_", names(raw), value = TRUE)
prot_null_cols <- intersect(c("PROTDEV_UK", "PROTDEV_NA"), prot_cols)
prot_valid_cols <- setdiff(prot_cols, prot_null_cols)

flag_or_false <- function(d, col) {
  if (!col %in% names(d)) return(rep(FALSE, nrow(d)))
  is_yes_flag(d[[col]])
}

helmet_yes <- flag_or_false(raw, "PROTDEV_HELMET")
prot_uk <- flag_or_false(raw, "PROTDEV_UK")
prot_na <- flag_or_false(raw, "PROTDEV_NA")

any_valid_prot <- rep(FALSE, nrow(raw))
if (length(prot_valid_cols) > 0L) {
  for (v in prot_valid_cols) {
    any_valid_prot <- any_valid_prot | is_yes_flag(raw[[v]])
  }
}

raw[, protective_device_conflict_recovered :=
      as.integer((prot_uk | prot_na) & any_valid_prot)]

raw[, helmet_use_recovered := fifelse(
  prot_uk | prot_na,
  "Unknown/not recorded",
  fifelse(
    helmet_yes,
    "Helmet worn",
    fifelse(
      any_valid_prot,
      "No helmet worn",
      "Unknown/not recorded"
    )
  )
)]

# GCS assessment qualifiers.
gcs_valid <- flag_or_false(raw, "GCSQ_VALID")
gcs_intubated <- flag_or_false(raw, "GCSQ_INTUBATED")
gcs_sedated <- flag_or_false(raw, "GCSQ_SEDATEDPARALYZED")
gcs_eye_obstruction <- flag_or_false(raw, "GCSQ_EYEOBSTRUCTION")
gcs_uk <- flag_or_false(raw, "GCSQ_UK")
gcs_na <- flag_or_false(raw, "GCSQ_NA")

gcs_any_known <- gcs_valid | gcs_intubated | gcs_sedated | gcs_eye_obstruction
gcs_unknown <- gcs_uk | gcs_na | !gcs_any_known

raw[, gcsq_intubated_recovered := as.integer(gcs_intubated)]
raw[, gcsq_sedated_paralyzed_recovered := as.integer(gcs_sedated)]
raw[, gcsq_eye_obstruction_recovered := as.integer(gcs_eye_obstruction)]
raw[, gcsq_valid_recovered := as.integer(gcs_valid)]
raw[, gcsq_unknown_recovered := as.integer(gcs_unknown)]
raw[, gcsq_conflict_recovered :=
      as.integer((gcs_uk | gcs_na) & gcs_any_known)]

# Supplemental oxygen. NTDS 2020-2024:
#   1 = No Supplemental Oxygen
#   2 = Supplemental Oxygen
if ("SUPPLEMENTALOXYGEN" %in% names(raw)) {
  so2n <- safe_num(raw$SUPPLEMENTALOXYGEN)
  so2t <- norm_text(raw$SUPPLEMENTALOXYGEN)

  raw[, supplemental_oxygen_recovered := fifelse(
    (!is.na(so2n) & so2n == 1) |
      so2t %in% c("no supplemental oxygen", "no"),
    "No supplemental oxygen",
    fifelse(
      (!is.na(so2n) & so2n == 2) |
        so2t %in% c("supplemental oxygen", "yes"),
      "Supplemental oxygen",
      "Unknown/not recorded"
    )
  )]
} else {
  raw[, supplemental_oxygen_recovered := "Unknown/not recorded"]
}

# Temperature in degrees Celsius; preserve plausible NTDS range only.
if ("TEMPERATURE" %in% names(raw)) {
  raw[, temperature_c_recovered := safe_num(TEMPERATURE)]
  raw[
    !is.na(temperature_c_recovered) &
      (temperature_c_recovered < 10 | temperature_c_recovered > 45),
    temperature_c_recovered := NA_real_
  ]
} else {
  raw[, temperature_c_recovered := NA_real_]
}

# Keep only repaired derivatives needed downstream plus raw code provenance.
raw_keep <- c(
  "admission_year", "inc_key",
  "ageyears_raw_recovered",
  "ed_disposition_code_recovered",
  "ed_disposition_biu_recovered",
  "helmet_use_recovered",
  "protective_device_conflict_recovered",
  "gcsq_intubated_recovered",
  "gcsq_sedated_paralyzed_recovered",
  "gcsq_eye_obstruction_recovered",
  "gcsq_valid_recovered",
  "gcsq_unknown_recovered",
  "gcsq_conflict_recovered",
  "supplemental_oxygen_recovered",
  "temperature_c_recovered"
)

raw_patch <- raw[, ..raw_keep]
rm(raw)
gc()

# -----------------------------------------------------------------------------
# Merge into harmonized patient core
# -----------------------------------------------------------------------------

cat("\nReading harmonized patient core:\n  ", patient_core_file, "\n", sep = "")
core <- as.data.table(readRDS(patient_core_file))
core <- core[admission_year %in% repair_years]

if (!all(c("admission_year", "inc_key", "discharge_group") %in% names(core))) {
  stop("Harmonized patient core lacks required identifiers/discharge_group.", call. = FALSE)
}

core[, inc_key := normalize_inc_key(inc_key)]

pre_n <- nrow(core)
core <- merge(
  core,
  raw_patch,
  by = c("admission_year", "inc_key"),
  all.x = TRUE,
  sort = FALSE
)

if (nrow(core) != pre_n) {
  stop("Raw-field patch merge changed patient-core row count.", call. = FALSE)
}

core[, raw_puf_repair_matched := as.integer(!is.na(ed_disposition_code_recovered) |
                                             !is.na(ageyears_raw_recovered) |
                                             helmet_use_recovered != "Unknown/not recorded" |
                                             gcsq_unknown_recovered == 0 |
                                             supplemental_oxygen_recovered != "Unknown/not recorded" |
                                             !is.na(temperature_c_recovered))]

# -----------------------------------------------------------------------------
# Repair final disposition
# -----------------------------------------------------------------------------

core[, final_discharge_group_repaired := as.character(discharge_group)]
core[, final_disposition_source_repaired :=
       fifelse(!is.na(discharge_group) & trimws(discharge_group) != "",
               "Hospital discharge disposition",
               NA_character_)]

missing_hosp <- is.na(core$final_discharge_group_repaired) |
  trimws(core$final_discharge_group_repaired) == ""

# Direct ED final disposition categories, AY 2020-2024.
core[
  missing_hosp & ed_disposition_code_recovered %in% c(4L, 9L),
  `:=`(
    final_discharge_group_repaired = "Home/home health",
    final_disposition_source_repaired = "ED discharge disposition"
  )
]

core[
  missing_hosp & ed_disposition_code_recovered == 5L,
  `:=`(
    final_discharge_group_repaired = "Death/hospice",
    final_disposition_source_repaired = "ED discharge disposition"
  )
]

core[
  missing_hosp & ed_disposition_code_recovered == 10L,
  `:=`(
    final_discharge_group_repaired = "Left AMA",
    final_disposition_source_repaired = "ED discharge disposition"
  )
]

core[
  missing_hosp & ed_disposition_code_recovered == 11L,
  `:=`(
    final_discharge_group_repaired = "Short-term hospital transfer",
    final_disposition_source_repaired = "ED discharge disposition"
  )
]

core[
  missing_hosp & ed_disposition_code_recovered == 6L,
  `:=`(
    final_discharge_group_repaired = "ED other/institutional/custody",
    final_disposition_source_repaired = "ED discharge disposition"
  )
]

# 2025+ hospice code retained defensively even though current years stop at 2024.
core[
  missing_hosp & ed_disposition_code_recovered == 13L,
  `:=`(
    final_discharge_group_repaired = "Death/hospice",
    final_disposition_source_repaired = "ED discharge disposition"
  )
]

# Project-level trajectory status.
y <- norm_text(core$final_discharge_group_repaired)

core[, project_trajectory_status_repaired := "Retain"]

core[
  is.na(final_discharge_group_repaired) |
    trimws(final_discharge_group_repaired) == "",
  project_trajectory_status_repaired := "Unresolved after disposition repair"
]

core[
  grepl("left ama|against medical advice|\\bama\\b", y),
  project_trajectory_status_repaired := "Exclude: AMA"
]

core[
  grepl("short[- ]?term hospital transfer|transfer/acute care|another acute", y),
  project_trajectory_status_repaired := "Exclude: short-term acute-care transfer"
]

core[
  grepl("court|law enforcement|prison|jail|correction|custody|police", y) |
    final_discharge_group_repaired == "ED other/institutional/custody",
  project_trajectory_status_repaired := "Exclude: custody/ED-other institutional"
]

# If hospital disposition is missing but ED disposition points to an inpatient
# unit, then the final hospital disposition truly remains unresolved.
core[
  (is.na(final_discharge_group_repaired) |
     trimws(final_discharge_group_repaired) == "") &
    ed_disposition_code_recovered %in% c(1L, 2L, 3L, 7L, 8L),
  project_trajectory_status_repaired :=
    "Unresolved: admitted ED disposition but hospital disposition missing"
]

# -----------------------------------------------------------------------------
# QC
# -----------------------------------------------------------------------------

merge_qc <- core[
  ,
  .(
    N = .N,
    N_raw_patch_matched = sum(raw_puf_repair_matched == 1L, na.rm = TRUE),
    pct_raw_patch_matched = 100 * mean(raw_puf_repair_matched == 1L, na.rm = TRUE),
    N_ED_disposition_recovered = sum(!is.na(ed_disposition_code_recovered)),
    N_temperature_recovered = sum(!is.na(temperature_c_recovered)),
    N_supplemental_O2_known = sum(supplemental_oxygen_recovered != "Unknown/not recorded", na.rm = TRUE),
    N_helmet_known = sum(helmet_use_recovered != "Unknown/not recorded", na.rm = TRUE),
    N_GCS_qualifier_known = sum(gcsq_unknown_recovered == 0L, na.rm = TRUE)
  ),
  by = admission_year
][order(admission_year)]

fwrite(merge_qc, file.path(repair_dir, "02_repair_merge_and_field_QC_by_year.csv"))

disposition_qc <- core[
  ,
  .N,
  by = .(
    admission_year,
    original_discharge_group = discharge_group,
    ed_disposition_code_recovered,
    final_discharge_group_repaired,
    final_disposition_source_repaired,
    project_trajectory_status_repaired
  )
][order(admission_year, -N)]

fwrite(
  disposition_qc,
  file.path(repair_dir, "03_final_disposition_repair_QC.csv")
)

disposition_summary <- core[
  ,
  .N,
  by = .(
    admission_year,
    project_trajectory_status_repaired
  )
][order(admission_year, project_trajectory_status_repaired)]

fwrite(
  disposition_summary,
  file.path(repair_dir, "04_project_trajectory_status_after_repair.csv")
)

helmet_qc <- core[
  ,
  .N,
  by = .(admission_year, helmet_use_recovered)
][
  ,
  pct := 100 * N / sum(N),
  by = admission_year
][order(admission_year, helmet_use_recovered)]

fwrite(helmet_qc, file.path(repair_dir, "05_helmet_recovery_by_year.csv"))

gcsq_qc <- core[
  ,
  .(
    N = .N,
    intubated = sum(gcsq_intubated_recovered == 1L, na.rm = TRUE),
    sedated_paralyzed = sum(gcsq_sedated_paralyzed_recovered == 1L, na.rm = TRUE),
    eye_obstruction = sum(gcsq_eye_obstruction_recovered == 1L, na.rm = TRUE),
    valid = sum(gcsq_valid_recovered == 1L, na.rm = TRUE),
    unknown = sum(gcsq_unknown_recovered == 1L, na.rm = TRUE),
    conflicts = sum(gcsq_conflict_recovered == 1L, na.rm = TRUE)
  ),
  by = admission_year
][order(admission_year)]

fwrite(gcsq_qc, file.path(repair_dir, "06_GCS_qualifier_recovery_by_year.csv"))

physiology_qc <- core[
  ,
  .(
    N = .N,
    supplemental_O2_known = sum(
      supplemental_oxygen_recovered != "Unknown/not recorded",
      na.rm = TRUE
    ),
    supplemental_O2_yes = sum(
      supplemental_oxygen_recovered == "Supplemental oxygen",
      na.rm = TRUE
    ),
    temperature_known = sum(!is.na(temperature_c_recovered)),
    temperature_median_C = median(temperature_c_recovered, na.rm = TRUE),
    temperature_q1_C = quantile(temperature_c_recovered, 0.25, na.rm = TRUE),
    temperature_q3_C = quantile(temperature_c_recovered, 0.75, na.rm = TRUE)
  ),
  by = admission_year
][order(admission_year)]

fwrite(physiology_qc, file.path(repair_dir, "07_O2_temperature_recovery_by_year.csv"))

# Adult-known-age disposition resolution QC, because this is the intended
# benchmark cohort before S06 restriction is subsequently applied.
age_num <- safe_num(core$age)
adult_known <- core[!is.na(age_num) & age_num >= 18]

resolution_qc <- adult_known[
  ,
  .N,
  by = project_trajectory_status_repaired
][
  ,
  pct := 100 * N / sum(N)
][order(-N)]

fwrite(
  resolution_qc,
  file.path(repair_dir, "08_adult_known_age_disposition_resolution_QC.csv")
)

# -----------------------------------------------------------------------------
# Save repaired core
# -----------------------------------------------------------------------------

saveRDS(core, repaired_core_file, compress = FALSE)

cat("\n============================================================\n")
cat("REPAIR COMPLETE\n")
cat("============================================================\n")
cat("Saved:\n  ", repaired_core_file, "\n\n", sep = "")
cat("Adult known-age disposition resolution:\n")
print(resolution_qc)

cat("\nReview these before running the repaired benchmark:\n")
cat("  outputs/repair_30min_raw_fields/02_repair_merge_and_field_QC_by_year.csv\n")
cat("  outputs/repair_30min_raw_fields/04_project_trajectory_status_after_repair.csv\n")
cat("  outputs/repair_30min_raw_fields/05_helmet_recovery_by_year.csv\n")
cat("  outputs/repair_30min_raw_fields/06_GCS_qualifier_recovery_by_year.csv\n")
cat("  outputs/repair_30min_raw_fields/07_O2_temperature_recovery_by_year.csv\n")
cat("  outputs/repair_30min_raw_fields/08_adult_known_age_disposition_resolution_QC.csv\n")
