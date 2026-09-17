# =============================================================================
# 08_TBI_TRACT_methods_completion_audit.R
#
# METHODS-COMPLETION AUDIT FOR TBI-TRACT
# FITS NO OUTCOME-PREDICTION MODEL.
#
# Purposes:
#   A) Quantify downstream-selection bias created by excluding direct-presenting
#      patients who later transfer out / leave AMA / have constrained disposition.
#   B) Produce development (2020-23) vs temporal-test (2024) SMD tables.
#   C) Re-audit ICU, ventilator, and hospital LOS thresholds using DEVELOPMENT
#      DATA ONLY, with no threshold selection from 2024 performance.
#   D) Save a frozen patient-level methods dataset for hyperparameter tuning and
#      penalized-regression comparator analyses.
#
# Run after the repaired Harmony/core and v4 adjudication are available.
# =============================================================================

rm(list = ls())
gc()

suppressPackageStartupMessages({
  library(data.table)
})

config_candidates <- c(
  file.path(getwd(), "R", "00_config.R"),
  "R/00_config.R"
)
config_file <- config_candidates[file.exists(config_candidates)][1]
if (is.na(config_file) || length(config_file) == 0L) {
  stop("Could not find R/00_config.R.", call. = FALSE)
}
source(config_file)

audit_dir <- file.path(
  output_dir,
  "METHODS_COMPLETION_TBI_TRACT"
)
dir.create(audit_dir, recursive = TRUE, showWarnings = FALSE)

logical_cores <- parallel::detectCores(logical = TRUE)
if (is.na(logical_cores)) logical_cores <- 32L
setDTthreads(max(1L, min(28L, logical_cores - 4L)))

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

safe_num <- function(x) suppressWarnings(as.numeric(as.character(x)))

normalize_id <- function(x) {
  out <- trimws(as.character(x))
  out <- gsub("\\.0$", "", out)
  out[out %in% c("", "NA", "NaN", "<NA>", "Inf", "-Inf")] <- NA_character_
  out
}

norm_code <- function(x) {
  toupper(gsub("[^A-Z0-9]", "", as.character(x)))
}

first_existing <- function(paths) {
  hit <- paths[file.exists(paths)]
  if (length(hit) == 0L) return(NA_character_)
  normalizePath(hit[1], winslash = "/", mustWork = TRUE)
}

binary_smd <- function(p1, p2) {
  den <- sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
  if (!is.finite(den) || den <= 0) return(0)
  (p2 - p1) / den
}

numeric_smd <- function(x1, x2) {
  x1 <- safe_num(x1)
  x2 <- safe_num(x2)
  m1 <- mean(x1, na.rm = TRUE)
  m2 <- mean(x2, na.rm = TRUE)
  s1 <- sd(x1, na.rm = TRUE)
  s2 <- sd(x2, na.rm = TRUE)
  den <- sqrt((s1^2 + s2^2) / 2)
  if (!is.finite(den) || den <= 0) return(NA_real_)
  (m2 - m1) / den
}

smd_one_predictor <- function(data, group_var, group1, group2, predictor, type) {
  g <- as.character(data[[group_var]])
  keep <- g %in% c(group1, group2)
  d <- data[keep]
  g <- as.character(d[[group_var]])

  if (type == "numeric") {
    x1 <- d[g == group1][[predictor]]
    x2 <- d[g == group2][[predictor]]

    miss1 <- mean(is.na(safe_num(x1)))
    miss2 <- mean(is.na(safe_num(x2)))

    return(data.table(
      predictor = predictor,
      level = NA_character_,
      predictor_type = "numeric",
      group1 = group1,
      group2 = group2,
      n_group1 = length(x1),
      n_group2 = length(x2),
      group1_summary = sprintf(
        "mean %.3f; SD %.3f",
        mean(safe_num(x1), na.rm = TRUE),
        sd(safe_num(x1), na.rm = TRUE)
      ),
      group2_summary = sprintf(
        "mean %.3f; SD %.3f",
        mean(safe_num(x2), na.rm = TRUE),
        sd(safe_num(x2), na.rm = TRUE)
      ),
      smd = numeric_smd(x1, x2),
      abs_smd = abs(numeric_smd(x1, x2)),
      missing_pct_group1 = 100 * miss1,
      missing_pct_group2 = 100 * miss2,
      missing_pct_difference = 100 * (miss2 - miss1)
    ))
  }

  x <- trimws(as.character(d[[predictor]]))
  x[is.na(x) | x == ""] <- "__MISSING__"
  levs <- sort(unique(x))

  out <- rbindlist(lapply(levs, function(lev) {
    p1 <- mean(x[g == group1] == lev)
    p2 <- mean(x[g == group2] == lev)

    data.table(
      predictor = predictor,
      level = lev,
      predictor_type = "categorical_level",
      group1 = group1,
      group2 = group2,
      n_group1 = sum(g == group1),
      n_group2 = sum(g == group2),
      group1_summary = sprintf("%.2f%%", 100 * p1),
      group2_summary = sprintf("%.2f%%", 100 * p2),
      smd = binary_smd(p1, p2),
      abs_smd = abs(binary_smd(p1, p2)),
      missing_pct_group1 =
        100 * mean(x[g == group1] == "__MISSING__"),
      missing_pct_group2 =
        100 * mean(x[g == group2] == "__MISSING__"),
      missing_pct_difference =
        100 * (
          mean(x[g == group2] == "__MISSING__") -
            mean(x[g == group1] == "__MISSING__")
        )
    )
  }))

  overall <- data.table(
    predictor = predictor,
    level = "__OVERALL_MAX_LEVEL_SMD__",
    predictor_type = "categorical_overall",
    group1 = group1,
    group2 = group2,
    n_group1 = sum(g == group1),
    n_group2 = sum(g == group2),
    group1_summary = NA_character_,
    group2_summary = NA_character_,
    smd = NA_real_,
    abs_smd = max(out$abs_smd, na.rm = TRUE),
    missing_pct_group1 = out$missing_pct_group1[1],
    missing_pct_group2 = out$missing_pct_group2[1],
    missing_pct_difference = out$missing_pct_difference[1]
  )

  rbind(overall, out, fill = TRUE)
}

make_smd_table <- function(data, group_var, group1, group2, numeric_vars, categorical_vars) {
  rbindlist(
    c(
      lapply(
        numeric_vars,
        function(v) smd_one_predictor(
          data, group_var, group1, group2, v, "numeric"
        )
      ),
      lapply(
        categorical_vars,
        function(v) smd_one_predictor(
          data, group_var, group1, group2, v, "categorical"
        )
      )
    ),
    fill = TRUE
  )
}

# -----------------------------------------------------------------------------
# Load repaired patient core + ICD diagnoses
# -----------------------------------------------------------------------------

core_file <- file.path(
  data_out_dir,
  "tbi_tract_patient_core_repaired_2020_2024.rds"
)
if (!file.exists(core_file)) {
  stop("Repaired patient core not found.", call. = FALSE)
}

icd_file <- first_existing(c(
  file.path(
    warehouse_dir,
    "data",
    "harmonized_icd_diagnoses_2007_2024.rds"
  ),
  file.path(
    warehouse_dir,
    "harmonized_icd_diagnoses_2007_2024.rds"
  )
))
if (is.na(icd_file)) stop("Harmonized ICD diagnoses not found.", call. = FALSE)

pc_file <- first_existing(c(
  file.path(
    warehouse_dir,
    "data",
    "harmonized_preexisting_conditions_2007_2024.rds"
  ),
  file.path(
    warehouse_dir,
    "harmonized_preexisting_conditions_2007_2024.rds"
  )
))
if (is.na(pc_file)) stop("Harmonized pre-existing conditions not found.", call. = FALSE)

cat("Reading repaired patient core...\n")
core <- as.data.table(readRDS(core_file))
core <- core[admission_year %in% 2020:2024]

id_vars <- c("admission_year", "inc_key", "harmonized_patient_id")
for (v in id_vars[-1]) core[, (v) := normalize_id(get(v))]

cat("Reading ICD diagnoses...\n")
dx <- as.data.table(readRDS(icd_file))
dx <- dx[admission_year %in% 2020:2024]
for (v in id_vars[-1]) dx[, (v) := normalize_id(get(v))]
dx[, code := norm_code(diagnosis_code)]

s06_ids <- unique(
  dx[
    startsWith(code, "S06"),
    ..id_vars
  ]
)
s06_ids[, s06_tbi := 1L]

pre_n <- nrow(core)
core <- merge(core, s06_ids, by = id_vars, all.x = TRUE, sort = FALSE)
if (nrow(core) != pre_n) stop("S06 merge changed row count.", call. = FALSE)
core[, s06_tbi := fifelse(is.na(s06_tbi), 0L, s06_tbi)]
core[, age_audit := safe_num(age)]

# Direct presentation is known at prediction time and defines the target population.
core[, transfer_in_status := {
  x <- trimws(as.character(transfer_clean))
  xl <- tolower(x)
  fcase(
    is.na(x) | x == "" |
      xl %in% c("na", "n/a", "unknown", "unk", "not known", "not recorded"),
      "Unknown",
    xl %in% c("yes", "y", "1", "true"), "Yes",
    xl %in% c("no", "n", "0", "false"), "No",
    default = "Unknown"
  )
}]

direct_source <- core[
  s06_tbi == 1L &
    !is.na(age_audit) &
    age_audit >= 18 &
    age_audit <= 89 &
    transfer_in_status == "No"
]

direct_source[, observation_status := fcase(
  project_trajectory_status_repaired == "Retain",
    "Retained complete trajectory",
  project_trajectory_status_repaired == "Exclude: short-term acute-care transfer",
    "Future transfer-out",
  project_trajectory_status_repaired == "Exclude: AMA",
    "Future AMA",
  project_trajectory_status_repaired == "Exclude: custody/ED-other institutional",
    "Future ED-other/institutional/custody",
  default = "Other/unresolved"
)]

fwrite(
  direct_source[
    ,
    .N,
    by = .(
      admission_year,
      observation_status
    )
  ][order(admission_year, observation_status)],
  file.path(
    audit_dir,
    "01_DIRECT_PRESENTATION_OBSERVATION_STATUS_BY_YEAR.csv"
  )
)

# -----------------------------------------------------------------------------
# Derive the exact frozen acute injury phenotype used by the primary model
# -----------------------------------------------------------------------------

cat("Deriving acute injury phenotype...\n")

dx[, dx_concussion_tmp :=
     startsWith(code, "S060")]
dx[, dx_cerebral_edema_traumatic_tmp :=
     startsWith(code, "S061")]
dx[, dx_diffuse_axonal_injury_tmp :=
     startsWith(code, "S062")]
dx[, dx_focal_contusion_or_iph_tmp :=
     startsWith(code, "S063")]
dx[, dx_epidural_hematoma_tmp :=
     startsWith(code, "S064")]
dx[, dx_subdural_hematoma_tmp :=
     startsWith(code, "S065")]
dx[, dx_subarachnoid_hemorrhage_tmp :=
     startsWith(code, "S066")]
dx[, dx_other_intracranial_injury_tmp :=
     startsWith(code, "S068") |
       startsWith(code, "S069")]
dx[, dx_cranial_skull_fracture_tmp :=
     startsWith(code, "S020") |
       startsWith(code, "S021")]
dx[, dx_facial_fracture_tmp :=
     grepl("^S02[23456]", code)]
dx[, dx_other_skull_or_facial_fracture_tmp :=
     startsWith(code, "S028") |
       startsWith(code, "S029")]
dx[, dx_spinal_cord_injury_tmp :=
     startsWith(code, "S14") |
       startsWith(code, "S24") |
       startsWith(code, "S34")]
dx[, dx_neck_vascular_injury_tmp :=
     startsWith(code, "S15")]
dx[, dx_thoracic_injury_tmp :=
     grepl("^S2[0-9]", code)]
dx[, dx_abdominal_pelvic_injury_tmp :=
     grepl("^S3[0-9]", code)]
dx[, dx_upper_extremity_injury_tmp :=
     grepl("^S[456][0-9]", code)]
dx[, dx_lower_extremity_injury_tmp :=
     grepl("^S[789][0-9]", code)]

injury_summary <- dx[
  ,
  .(
    dx_concussion =
      as.integer(any(dx_concussion_tmp, na.rm = TRUE)),
    dx_cerebral_edema_traumatic =
      as.integer(any(dx_cerebral_edema_traumatic_tmp, na.rm = TRUE)),
    dx_diffuse_axonal_injury =
      as.integer(any(dx_diffuse_axonal_injury_tmp, na.rm = TRUE)),
    dx_focal_contusion_or_iph =
      as.integer(any(dx_focal_contusion_or_iph_tmp, na.rm = TRUE)),
    dx_epidural_hematoma =
      as.integer(any(dx_epidural_hematoma_tmp, na.rm = TRUE)),
    dx_subdural_hematoma =
      as.integer(any(dx_subdural_hematoma_tmp, na.rm = TRUE)),
    dx_subarachnoid_hemorrhage =
      as.integer(any(dx_subarachnoid_hemorrhage_tmp, na.rm = TRUE)),
    dx_other_intracranial_injury =
      as.integer(any(dx_other_intracranial_injury_tmp, na.rm = TRUE)),
    dx_cranial_skull_fracture =
      as.integer(any(dx_cranial_skull_fracture_tmp, na.rm = TRUE)),
    dx_facial_fracture =
      as.integer(any(dx_facial_fracture_tmp, na.rm = TRUE)),
    dx_other_skull_or_facial_fracture =
      as.integer(any(dx_other_skull_or_facial_fracture_tmp, na.rm = TRUE)),
    dx_spinal_cord_injury =
      as.integer(any(dx_spinal_cord_injury_tmp, na.rm = TRUE)),
    dx_neck_vascular_injury =
      as.integer(any(dx_neck_vascular_injury_tmp, na.rm = TRUE)),
    dx_thoracic_injury =
      as.integer(any(dx_thoracic_injury_tmp, na.rm = TRUE)),
    dx_abdominal_pelvic_injury =
      as.integer(any(dx_abdominal_pelvic_injury_tmp, na.rm = TRUE)),
    dx_upper_extremity_injury =
      as.integer(any(dx_upper_extremity_injury_tmp, na.rm = TRUE)),
    dx_lower_extremity_injury =
      as.integer(any(dx_lower_extremity_injury_tmp, na.rm = TRUE))
  ),
  by = id_vars
]

injury_predictors <- setdiff(names(injury_summary), id_vars)

pre_n <- nrow(direct_source)
direct_source <- merge(
  direct_source,
  injury_summary,
  by = id_vars,
  all.x = TRUE,
  sort = FALSE
)
if (nrow(direct_source) != pre_n) {
  stop("Injury merge changed direct-source row count.", call. = FALSE)
}
for (v in injury_predictors) {
  direct_source[is.na(get(v)), (v) := 0L]
}

rm(dx, injury_summary, s06_ids)
gc()

# -----------------------------------------------------------------------------
# Derive exact selected PMHx codes
# -----------------------------------------------------------------------------

cat("Deriving selected PMHx...\n")
pc <- as.data.table(readRDS(pc_file))
pc <- pc[admission_year %in% 2020:2024]
for (v in id_vars[-1]) pc[, (v) := normalize_id(get(v))]

get_pc <- function(nm) {
  if (nm %in% names(pc)) normalize_id(pc[[nm]]) else rep(NA_character_, nrow(pc))
}

pc[, pmhx_code := {
  c1 <- get_pc("condition_code")
  c2 <- get_pc("condition_name")
  c3 <- get_pc("condition_group_clean")

  fcase(
    !is.na(c1) & grepl("^[0-9]+$", c1), c1,
    !is.na(c2) & grepl("^[0-9]+$", c2), c2,
    !is.na(c3) & grepl("^[0-9]+$", c3), c3,
    default = NA_character_
  )
}]

pmhx_map <- data.table(
  pmhx_code = c(
    "4", "31", "23", "11", "19", "8",
    "15", "26", "7", "9", "25", "24"
  ),
  pmhx_variable = c(
    "pmhx_bleeding_disorder",
    "pmhx_anticoagulant_therapy",
    "pmhx_copd",
    "pmhx_diabetes",
    "pmhx_hypertension",
    "pmhx_current_smoker",
    "pmhx_functional_dependence",
    "pmhx_dementia",
    "pmhx_chf",
    "pmhx_chronic_renal_failure",
    "pmhx_cirrhosis",
    "pmhx_steroid_use"
  )
)

pmhx_predictors <- pmhx_map$pmhx_variable

pc_pos <- unique(
  pc[
    !is.na(pmhx_code) &
      pmhx_code %in% pmhx_map$pmhx_code,
    c(id_vars, "pmhx_code"),
    with = FALSE
  ]
)

pc_pos <- merge(
  pc_pos,
  pmhx_map,
  by = "pmhx_code",
  all.x = TRUE
)
pc_pos[, value := 1L]

pc_wide <- dcast(
  pc_pos,
  admission_year + inc_key + harmonized_patient_id ~ pmhx_variable,
  value.var = "value",
  fun.aggregate = max,
  fill = 0L
)

pre_n <- nrow(direct_source)
direct_source <- merge(
  direct_source,
  pc_wide,
  by = id_vars,
  all.x = TRUE,
  sort = FALSE
)
if (nrow(direct_source) != pre_n) {
  stop("PMHx merge changed direct-source row count.", call. = FALSE)
}
for (v in pmhx_predictors) {
  if (!v %in% names(direct_source)) {
    direct_source[, (v) := 0L]
  } else {
    direct_source[is.na(get(v)), (v) := 0L]
  }
}

rm(pc, pc_pos, pc_wide)
gc()

# -----------------------------------------------------------------------------
# Predictor set and validation-only GCS severity
# -----------------------------------------------------------------------------

numeric_predictors <- c(
  "age",
  "gcs_eye_clean",
  "gcs_verbal_clean",
  "gcs_motor_clean",
  "gcsq_intubated_recovered",
  "gcsq_sedated_paralyzed_recovered",
  "gcsq_eye_obstruction_recovered",
  "gcsq_unknown_recovered",
  "sbp_clean",
  "pulse_clean",
  "rr_clean",
  "spo2_clean",
  "temperature_c_recovered",
  injury_predictors,
  pmhx_predictors
)

categorical_predictors <- c(
  "sex_clean",
  "mechanism_clean",
  "helmet_use_recovered",
  "pupil_clean",
  "respiratoryassistance_clean",
  "supplemental_oxygen_recovered"
)

missing_pred <- setdiff(
  c(numeric_predictors, categorical_predictors),
  names(direct_source)
)
if (length(missing_pred) > 0L) {
  stop(
    "Missing frozen predictor(s): ",
    paste(missing_pred, collapse = ", "),
    call. = FALSE
  )
}

direct_source[, gcs_total_validation := {
  e <- safe_num(gcs_eye_clean)
  v <- safe_num(gcs_verbal_clean)
  m <- safe_num(gcs_motor_clean)

  fifelse(
    is.na(e) | is.na(v) | is.na(m),
    NA_real_,
    e + v + m
  )
}]

direct_source[, gcs_severity_validation := fcase(
  is.na(gcs_total_validation), "Unknown",
  gcs_total_validation >= 13, "Mild (13-15)",
  gcs_total_validation >= 9, "Moderate (9-12)",
  gcs_total_validation >= 3, "Severe (3-8)",
  default = "Unknown"
)]

# -----------------------------------------------------------------------------
# A) Downstream observation-selection audit
# -----------------------------------------------------------------------------

cat("Computing downstream-selection SMDs...\n")

selection_groups <- c(
  "Future transfer-out",
  "Future AMA",
  "Future ED-other/institutional/custody",
  "All downstream-excluded"
)

selection_work <- copy(direct_source)
selection_work[, selection_compare := observation_status]
selection_work[
  observation_status != "Retained complete trajectory",
  selection_compare_all := "All downstream-excluded"
]
selection_work[
  observation_status == "Retained complete trajectory",
  selection_compare_all := "Retained complete trajectory"
]

selection_tables <- list()

for (grp in selection_groups) {
  if (grp == "All downstream-excluded") {
    d <- selection_work[
      selection_compare_all %in%
        c(
          "Retained complete trajectory",
          "All downstream-excluded"
        )
    ]

    tab <- make_smd_table(
      d,
      "selection_compare_all",
      "Retained complete trajectory",
      "All downstream-excluded",
      numeric_predictors,
      categorical_predictors
    )
  } else {
    d <- selection_work[
      observation_status %in%
        c(
          "Retained complete trajectory",
          grp
        )
    ]

    tab <- make_smd_table(
      d,
      "observation_status",
      "Retained complete trajectory",
      grp,
      numeric_predictors,
      categorical_predictors
    )
  }

  tab[, comparison := paste0(
    "Retained vs ", grp
  )]
  selection_tables[[length(selection_tables) + 1L]] <- tab
}

selection_smd <- rbindlist(selection_tables, fill = TRUE)

fwrite(
  selection_smd[
    order(
      comparison,
      -abs_smd,
      predictor,
      level
    )
  ],
  file.path(
    audit_dir,
    "02_SELECTION_SMD_RETAINED_VS_DOWNSTREAM_EXCLUDED.csv"
  )
)

selection_summary <- selection_smd[
  predictor_type %in%
    c(
      "numeric",
      "categorical_overall"
    ),
  .(
    n_predictors = .N,
    n_abs_smd_ge_0_10 =
      sum(abs_smd >= 0.10, na.rm = TRUE),
    n_abs_smd_ge_0_20 =
      sum(abs_smd >= 0.20, na.rm = TRUE),
    max_abs_smd =
      max(abs_smd, na.rm = TRUE)
  ),
  by = comparison
]

fwrite(
  selection_summary,
  file.path(
    audit_dir,
    "03_SELECTION_SMD_SUMMARY.csv"
  )
)

# 2024-only selection check.
selection_2024 <- selection_work[admission_year == 2024L]
selection_2024_tabs <- list()

for (grp in selection_groups) {
  if (grp == "All downstream-excluded") {
    d <- selection_2024[
      selection_compare_all %in%
        c(
          "Retained complete trajectory",
          "All downstream-excluded"
        )
    ]
    tab <- make_smd_table(
      d,
      "selection_compare_all",
      "Retained complete trajectory",
      "All downstream-excluded",
      numeric_predictors,
      categorical_predictors
    )
  } else {
    d <- selection_2024[
      observation_status %in%
        c(
          "Retained complete trajectory",
          grp
        )
    ]
    tab <- make_smd_table(
      d,
      "observation_status",
      "Retained complete trajectory",
      grp,
      numeric_predictors,
      categorical_predictors
    )
  }

  tab[, comparison := paste0(
    "2024 retained vs ", grp
  )]
  selection_2024_tabs[[length(selection_2024_tabs) + 1L]] <- tab
}

fwrite(
  rbindlist(
    selection_2024_tabs,
    fill = TRUE
  )[
    order(
      comparison,
      -abs_smd,
      predictor,
      level
    )
  ],
  file.path(
    audit_dir,
    "04_SELECTION_SMD_2024_ONLY.csv"
  )
)

# -----------------------------------------------------------------------------
# B) Development vs 2024 case-mix SMDs in retained final cohort
# -----------------------------------------------------------------------------

retained <- direct_source[
  observation_status ==
    "Retained complete trajectory"
]

# Hard reconciliation with completed v4 / primary model.
expected <- data.table(
  admission_year = 2020:2024,
  expected_N = c(
    145338L,
    153018L,
    152944L,
    152706L,
    151874L
  )
)

observed <- retained[
  ,
  .(observed_N = .N),
  by = admission_year
]

recon <- merge(
  expected,
  observed,
  by = "admission_year",
  all = TRUE
)
recon[, matches := expected_N == observed_N]

fwrite(
  recon,
  file.path(
    audit_dir,
    "05_RETAINED_COHORT_RECONCILIATION.csv"
  )
)

if (
  anyNA(recon$matches) ||
    !all(recon$matches)
) {
  stop(
    "Retained cohort does not reproduce the locked primary cohort.",
    call. = FALSE
  )
}

retained[, temporal_group := fifelse(
  admission_year <= 2023,
  "Development 2020-2023",
  "Temporal test 2024"
)]

temporal_smd <- make_smd_table(
  retained,
  "temporal_group",
  "Development 2020-2023",
  "Temporal test 2024",
  numeric_predictors,
  categorical_predictors
)

fwrite(
  temporal_smd[
    order(
      -abs_smd,
      predictor,
      level
    )
  ],
  file.path(
    audit_dir,
    "06_DEVELOPMENT_2020_23_VS_2024_SMD.csv"
  )
)

# -----------------------------------------------------------------------------
# C) DEVELOPMENT-ONLY threshold audit
# -----------------------------------------------------------------------------

cat("Auditing duration thresholds using 2020-2023 ONLY...\n")

dev <- retained[admission_year <= 2023]

dev[, icu_days := safe_num(icu_los_zero)]
dev[, vent_days := safe_num(vent_days_zero)]
dev[, hospital_days := safe_num(hosp_los_days)]

quantile_probs <- c(
  0.50,
  0.75,
  0.80,
  0.85,
  0.90,
  0.95,
  0.975,
  0.99
)

duration_quantiles <- rbindlist(list(
  data.table(
    outcome = "Hospital LOS: all retained patients",
    percentile = quantile_probs,
    days = as.numeric(
      quantile(
        dev$hospital_days,
        probs = quantile_probs,
        na.rm = TRUE,
        type = 2
      )
    )
  ),
  data.table(
    outcome = "ICU LOS: all retained patients including zero",
    percentile = quantile_probs,
    days = as.numeric(
      quantile(
        dev$icu_days,
        probs = quantile_probs,
        na.rm = TRUE,
        type = 2
      )
    )
  ),
  data.table(
    outcome = "ICU LOS: ICU users only",
    percentile = quantile_probs,
    days = as.numeric(
      quantile(
        dev[icu_days > 0]$icu_days,
        probs = quantile_probs,
        na.rm = TRUE,
        type = 2
      )
    )
  ),
  data.table(
    outcome = "Ventilator days: all retained patients including zero",
    percentile = quantile_probs,
    days = as.numeric(
      quantile(
        dev$vent_days,
        probs = quantile_probs,
        na.rm = TRUE,
        type = 2
      )
    )
  ),
  data.table(
    outcome = "Ventilator days: ventilated patients only",
    percentile = quantile_probs,
    days = as.numeric(
      quantile(
        dev[vent_days > 0]$vent_days,
        probs = quantile_probs,
        na.rm = TRUE,
        type = 2
      )
    )
  )
))

fwrite(
  duration_quantiles,
  file.path(
    audit_dir,
    "07_DEVELOPMENT_ONLY_DURATION_QUANTILES.csv"
  )
)

threshold_definitions <- rbindlist(list(
  data.table(
    outcome = "ICU LOS",
    threshold_days = c(8L, 10L, 14L),
    threshold_label = c(
      ">=8 days (>7 days)",
      ">=10 days",
      ">=14 days"
    )
  ),
  data.table(
    outcome = "Ventilator duration",
    threshold_days = c(8L, 11L, 15L, 21L),
    threshold_label = c(
      ">=8 days (>7 days)",
      ">=11 days (>10 days)",
      ">=15 days (>14 days)",
      ">=21 days"
    )
  ),
  data.table(
    outcome = "Hospital LOS",
    threshold_days = c(14L, 20L, 28L, 30L),
    threshold_label = c(
      ">=14 days",
      ">=20 days",
      ">=28 days (4 weeks)",
      ">=30 days"
    )
  )
))

threshold_prevalence <- list()

for (i in seq_len(nrow(threshold_definitions))) {
  o <- threshold_definitions$outcome[i]
  t <- threshold_definitions$threshold_days[i]
  lab <- threshold_definitions$threshold_label[i]

  x <- switch(
    o,
    "ICU LOS" = dev$icu_days,
    "Ventilator duration" = dev$vent_days,
    "Hospital LOS" = dev$hospital_days
  )

  threshold_prevalence[[length(threshold_prevalence) + 1L]] <- data.table(
    outcome = o,
    threshold_days = t,
    threshold_label = lab,
    subgroup = "Overall development 2020-2023",
    N_known = sum(!is.na(x)),
    events = sum(x >= t, na.rm = TRUE),
    prevalence_pct = 100 * mean(x >= t, na.rm = TRUE),
    percentile_below_threshold =
      100 * mean(x < t, na.rm = TRUE)
  )

  for (sev in c(
    "Mild (13-15)",
    "Moderate (9-12)",
    "Severe (3-8)"
  )) {
    ds <- dev[
      gcs_severity_validation == sev
    ]

    xs <- switch(
      o,
      "ICU LOS" = ds$icu_days,
      "Ventilator duration" = ds$vent_days,
      "Hospital LOS" = ds$hospital_days
    )

    threshold_prevalence[[length(threshold_prevalence) + 1L]] <- data.table(
      outcome = o,
      threshold_days = t,
      threshold_label = lab,
      subgroup = sev,
      N_known = sum(!is.na(xs)),
      events = sum(xs >= t, na.rm = TRUE),
      prevalence_pct = 100 * mean(xs >= t, na.rm = TRUE),
      percentile_below_threshold =
        100 * mean(xs < t, na.rm = TRUE)
    )
  }
}

threshold_prevalence <- rbindlist(
  threshold_prevalence,
  fill = TRUE
)

fwrite(
  threshold_prevalence[
    order(
      outcome,
      threshold_days,
      subgroup
    )
  ],
  file.path(
    audit_dir,
    "08_DEVELOPMENT_ONLY_THRESHOLD_PREVALENCE.csv"
  )
)

# Mathematical equivalence check for integer-day data:
# >7 days should equal >=8 days exactly.
equivalence_qc <- data.table(
  endpoint = c(
    "ICU LOS",
    "Ventilator duration"
  ),
  N_gt7 = c(
    sum(dev$icu_days > 7, na.rm = TRUE),
    sum(dev$vent_days > 7, na.rm = TRUE)
  ),
  N_ge8 = c(
    sum(dev$icu_days >= 8, na.rm = TRUE),
    sum(dev$vent_days >= 8, na.rm = TRUE)
  )
)
equivalence_qc[, exact_equivalence := N_gt7 == N_ge8]

fwrite(
  equivalence_qc,
  file.path(
    audit_dir,
    "09_GT7_EQUIVALENT_TO_GE8_QC.csv"
  )
)

# HLOS tiers from Yue et al. 2023 (0-7, 8-13, 14-27, >=28).
dev[, hlos_tier_yue2023 := fcase(
  is.na(hospital_days), NA_character_,
  hospital_days <= 7, "0-7 days",
  hospital_days <= 13, "8-13 days",
  hospital_days <= 27, "14-27 days",
  hospital_days >= 28, ">=28 days"
)]

fwrite(
  dev[
    ,
    .N,
    by = .(
      hlos_tier_yue2023
    )
  ][
    ,
    prevalence_pct := 100 * N / sum(N)
  ][
    order(
      factor(
        hlos_tier_yue2023,
        levels = c(
          "0-7 days",
          "8-13 days",
          "14-27 days",
          ">=28 days"
        )
      )
    )
  ],
  file.path(
    audit_dir,
    "10_DEVELOPMENT_HLOS_YUE2023_TIERS.csv"
  )
)

# -----------------------------------------------------------------------------
# Save frozen methods dataset for tuning/comparator scripts
# -----------------------------------------------------------------------------

# Outcomes are only valid/comparable in retained patients.
retained[, discharge_3cat_final := fifelse(
  final_discharge_group_repaired %in% c(
    "Home/home health", "Home", "Home health"
  ),
  "Home/home health",
  fifelse(
    final_discharge_group_repaired %in% c(
      "Inpatient rehab",
      "Facility/LTCH/SNF/other institution",
      "Skilled nursing/LTAC",
      "Rehab",
      "Post-acute facility"
    ),
    "Post-acute facility",
    fifelse(
      final_discharge_group_repaired %in% c(
        "Death/hospice", "Hospice", "Death", "Expired"
      ),
      "Death/hospice",
      NA_character_
    )
  )
)]

retained[, icu_days := safe_num(icu_los_zero)]
retained[, vent_days := safe_num(vent_days_zero)]
retained[, hospital_days := safe_num(hosp_los_days)]

retained[, icu_trajectory_final := fcase(
  is.na(icu_days), NA_character_,
  icu_days <= 0, "No ICU",
  icu_days < 8, "ICU 1-7 days",
  icu_days >= 8, "ICU >=8 days"
)]

retained[, vent_ge8_final := fifelse(
  is.na(vent_days),
  NA_integer_,
  as.integer(vent_days >= 8)
)]

retained[, hlos_ge28_final := fifelse(
  is.na(hospital_days),
  NA_integer_,
  as.integer(hospital_days >= 28)
)]

# Merge the same adjudicated neurosurgical endpoint flags used by the primary model.
candidate_neuro_files <- unique(c(
  file.path(
    project_dir,
    "neurosurgical_resource_endpoints_2020_2024",
    "04_neurosurgical_resource_endpoint_flags.csv"
  ),
  file.path(
    data_raw_dir,
    "04_neurosurgical_resource_endpoint_flags.csv"
  ),
  file.path(
    "C:/Users/garin/OneDrive/OHSU/Research/TQIP/Griffith_G-selected",
    "neurosurgical_resource_endpoints_2020_2024",
    "04_neurosurgical_resource_endpoint_flags.csv"
  ),
  file.path(
    "C:/OneDrive/garingriffith/OneDrive/OHSU/Research/TQIP/Griffith_G-selected",
    "neurosurgical_resource_endpoints_2020_2024",
    "04_neurosurgical_resource_endpoint_flags.csv"
  )
))

recursive_neuro_hits <- tryCatch(
  list.files(
    project_dir,
    pattern = "^04_neurosurgical_resource_endpoint_flags\\.csv$",
    recursive = TRUE,
    full.names = TRUE,
    ignore.case = TRUE
  ),
  error = function(e) character()
)

neuro_file <- first_existing(
  unique(
    c(
      candidate_neuro_files,
      recursive_neuro_hits
    )
  )
)

if (is.na(neuro_file)) {
  stop(
    "Neurosurgical endpoint flag file not found; cannot prepare complete comparator/tuning dataset.",
    call. = FALSE
  )
}

nf <- fread(
  neuro_file,
  integer64 = "character"
)
if (!all(c("year", "INC_KEY") %in% names(nf))) {
  stop(
    "Neurosurgical endpoint file missing year/INC_KEY.",
    call. = FALSE
  )
}

nf[, admission_year := as.integer(year)]
nf[, inc_key := normalize_id(INC_KEY)]
nf[, neuro_flag_matched := 1L]

keep_nf <- intersect(
  c(
    "admission_year",
    "inc_key",
    "neuro_flag_matched",
    "icp_evd",
    "icp_parenchymal_bolt",
    "craniotomy_craniectomy"
  ),
  names(nf)
)
nf <- unique(
  nf[, ..keep_nf],
  by = c(
    "admission_year",
    "inc_key"
  )
)

pre_n <- nrow(retained)
retained <- merge(
  retained,
  nf,
  by = c(
    "admission_year",
    "inc_key"
  ),
  all.x = TRUE,
  sort = FALSE
)
if (nrow(retained) != pre_n) {
  stop(
    "Neurosurgical endpoint merge changed retained row count.",
    call. = FALSE
  )
}

if (!all(c("icp_evd", "icp_parenchymal_bolt") %in% names(retained))) {
  stop(
    "Neurosurgical endpoint file lacks EVD/parenchymal ICP flags.",
    call. = FALSE
  )
}

evd <- safe_num(retained$icp_evd)
bolt <- safe_num(retained$icp_parenchymal_bolt)

retained[, icp_pressure_monitor_final := fifelse(
  neuro_flag_matched != 1L,
  NA_integer_,
  fifelse(
    (!is.na(evd) & evd == 1) |
      (!is.na(bolt) & bolt == 1),
    1L,
    fifelse(
      (!is.na(evd) & evd == 0) &
        (!is.na(bolt) & bolt == 0),
      0L,
      NA_integer_
    )
  )
)]

if (!"craniotomy_craniectomy" %in% names(retained)) {
  stop(
    "Neurosurgical endpoint file lacks craniotomy_craniectomy.",
    call. = FALSE
  )
}

cr <- safe_num(retained$craniotomy_craniectomy)
retained[, craniotomy_craniectomy_final := fifelse(
  neuro_flag_matched != 1L | is.na(cr),
  NA_integer_,
  as.integer(cr == 1)
)]

rm(nf, evd, bolt, cr)
gc()

keep_for_methods <- unique(c(
  id_vars,
  "observation_status",
  "gcs_total_validation",
  "gcs_severity_validation",
  numeric_predictors,
  categorical_predictors,
  "race_clean",
  "ethnicity_clean",
  "insurance_clean",
  "discharge_3cat_final",
  "icu_trajectory_final",
  "vent_ge8_final",
  "hlos_ge28_final",
  "icp_pressure_monitor_final",
  "craniotomy_craniectomy_final",
  "icu_days",
  "vent_days",
  "hospital_days"
))

keep_for_methods <- intersect(
  keep_for_methods,
  names(retained)
)

saveRDS(
  retained[
    ,
    ..keep_for_methods
  ],
  file.path(
    audit_dir,
    "frozen_methods_dataset_retained_2020_2024.rds"
  ),
  compress = FALSE
)

fwrite(
  data.table(
    category = c(
      rep("numeric", length(numeric_predictors)),
      rep("categorical", length(categorical_predictors))
    ),
    predictor = c(
      numeric_predictors,
      categorical_predictors
    )
  ),
  file.path(
    audit_dir,
    "11_FROZEN_PREDICTOR_TYPES.csv"
  )
)

summary_lines <- c(
  "TBI-TRACT METHODS-COMPLETION AUDIT FINISHED",
  "",
  "No outcome-prediction model was fit.",
  "",
  "Primary questions answered:",
  "  1. How different are patients retained vs downstream-excluded after direct presentation?",
  "  2. How much did case mix shift from 2020-23 development to 2024?",
  "  3. Where do >=8-day ICU/ventilator and >=28-day hospital LOS thresholds sit in development-only distributions?",
  "  4. Are >7 days and >=8 days exactly equivalent in integer-day TQIP fields?",
  "",
  "Review first:",
  "  03_SELECTION_SMD_SUMMARY.csv",
  "  02_SELECTION_SMD_RETAINED_VS_DOWNSTREAM_EXCLUDED.csv",
  "  06_DEVELOPMENT_2020_23_VS_2024_SMD.csv",
  "  07_DEVELOPMENT_ONLY_DURATION_QUANTILES.csv",
  "  08_DEVELOPMENT_ONLY_THRESHOLD_PREVALENCE.csv",
  "  09_GT7_EQUIVALENT_TO_GE8_QC.csv",
  "  10_DEVELOPMENT_HLOS_YUE2023_TIERS.csv",
  "",
  "Frozen methods dataset saved for subsequent tuning/comparator work."
)

writeLines(
  summary_lines,
  file.path(
    audit_dir,
    "METHODS_COMPLETION_SUMMARY.txt"
  )
)

cat("\n", paste(summary_lines, collapse = "\n"), "\n", sep = "")
cat("\nOutput:\n  ", audit_dir, "\n", sep = "")
