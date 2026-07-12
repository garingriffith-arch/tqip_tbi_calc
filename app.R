# ============================================================
# TBI Resource Utilization Calculator
# Clinician-facing Shiny app with endpoint-specific Quick Mode
#
# Required data files in data/:
#   - model_bundle.rds
#   - predictor_metadata.rds
#
# This version intentionally removes ISS and AIS-derived registry severity
# variables from the user interface and from prediction. The deployed model
# bundle and metadata must therefore be regenerated from the no-ISS/no-AIS
# training pipeline before deployment.
# ============================================================

suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(data.table)
  library(Matrix)
  library(xgboost)
  library(scales)
  library(stringr)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

bundle <- readRDS(file.path("data", "model_bundle.rds"))
meta_obj <- readRDS(file.path("data", "predictor_metadata.rds"))

metadata <- as.data.table(meta_obj$metadata)
for (nm in c("variable", "label", "group", "input_type", "model_type", "choices", "default", "min", "max")) {
  if (!nm %in% names(metadata)) metadata[, (nm) := NA_character_]
}

# ------------------------------------------------------------
# Registry severity variables intentionally retired
# ------------------------------------------------------------

registry_severity_vars <- unique(c(
  "iss_clean",
  "max_head_ais_clean",
  "max_extracranial_ais_clean",
  "max_overall_ais_aug",
  "n_head_ais_codes",
  "n_extracranial_ais_codes",
  "n_total_ais_codes_aug",
  "severe_head_ais3_aug",
  "critical_head_ais5_aug",
  "severe_extracranial_ais3_aug",
  "severe_tbi_ais3_src_aug",
  "severe_extracranial_ais3_src_aug",
  "any_extracranial_injury_aug",
  "isolated_tbi_derived_aug",
  "ais_face_any",
  "ais_thorax_any",
  "ais_abdomen_any",
  "ais_spine_any",
  "ais_upper_ext_any",
  "ais_lower_ext_any",
  "ais_external_any",
  "ais_headneck_severe3",
  "ais_face_severe3",
  "ais_thorax_severe3",
  "ais_abdomen_severe3",
  "ais_spine_severe3",
  "ais_upper_ext_severe3",
  "ais_lower_ext_severe3",
  "ais_region_count",
  "ais_severe_region_count",
  "ais_polyregion_2plus",
  "ais_polyregion_3plus",
  "ais_severe_polyregion_2plus"
))

contains_registry_severity <- function(x) {
  x <- as.character(x)
  x <- x[!is.na(x) & nzchar(x)]
  if (length(x) == 0) return(character(0))
  exact <- x %in% registry_severity_vars
  ais_prefix <- grepl("^ais_", x)
  starts_with_retired <- vapply(x, function(xx) {
    any(startsWith(xx, paste0(registry_severity_vars, "."))) ||
      any(startsWith(xx, paste0(registry_severity_vars, "_"))) ||
      any(startsWith(xx, paste0(registry_severity_vars, "1"))) ||
      any(startsWith(xx, paste0(registry_severity_vars, "TRUE")))
  }, logical(1))
  x[exact | ais_prefix | starts_with_retired]
}

bundle_registry_hits <- unique(unlist(lapply(bundle, function(obj) {
  if (!is.list(obj)) return(character(0))
  contains_registry_severity(c(obj$predictors, obj$feature_names))
}), use.names = FALSE))

metadata_registry_hits <- contains_registry_severity(metadata$variable)

# Remove retired variables from metadata/predictor vectors used by the UI.
metadata <- metadata[!variable %in% registry_severity_vars & !grepl("^ais_", variable)]
predictors_full <- setdiff(meta_obj$predictors_full, registry_severity_vars)
predictors_full <- predictors_full[!grepl("^ais_", predictors_full)]
predictors_no_resp <- meta_obj$predictors_no_resp %||% setdiff(predictors_full, "respiratoryassistance_clean")
predictors_no_resp <- setdiff(predictors_no_resp, registry_severity_vars)
predictors_no_resp <- predictors_no_resp[!grepl("^ais_", predictors_no_resp)]

model_n <- meta_obj$model_n %||% NA_integer_
model_years <- meta_obj$model_years %||% character(0)
discharge_levels <- meta_obj$discharge_levels %||% c("Home/home health", "Post-acute facility", "Death/hospice")

input_id <- function(v) paste0("var__", v)

safe_numeric <- function(x, default = 0) {
  if (is.null(x) || length(x) == 0) return(default)
  y <- suppressWarnings(as.numeric(x[[1]]))
  if (!is.finite(y)) return(default)
  y
}

safe_int01 <- function(x) {
  if (is.null(x) || length(x) == 0) return(0L)
  as.integer(as.character(x[[1]]) %in% c("1", "Yes", "yes", "TRUE", "true", TRUE))
}

get_yesno <- function(input, v) {
  safe_int01(input[[input_id(v)]])
}

get_ext_sev <- function(input, region) {
  val <- input[[paste0("ui__", region, "_severity")]]
  if (is.null(val) || length(val) == 0) "none" else as.character(val[[1]])
}

ext_any <- function(input, region) {
  as.integer(get_ext_sev(input, region) != "none")
}

ext_severe <- function(input, region) {
  as.integer(get_ext_sev(input, region) == "severe")
}

clip_prob <- function(p, eps = 1e-6) {
  pmin(pmax(as.numeric(p), eps), 1 - eps)
}

apply_binary_recalibration <- function(p, obj) {
  cal <- obj$calibration
  if (is.null(cal)) return(clip_prob(p))

  intercept <- suppressWarnings(as.numeric(cal$intercept))
  slope <- suppressWarnings(as.numeric(cal$slope))
  eps <- suppressWarnings(as.numeric(cal$clip_eps %||% 1e-6))

  if (!is.finite(intercept) || !is.finite(slope)) return(clip_prob(p))
  if (!is.finite(eps) || eps <= 0 || eps >= 0.5) eps <- 1e-6

  as.numeric(plogis(intercept + slope * qlogis(clip_prob(p, eps))))
}

find_xgb <- function(obj, max_depth = 10) {
  if (max_depth < 0 || is.null(obj)) return(NULL)
  if (inherits(obj, "xgb.Booster")) return(obj)
  if (is.list(obj)) {
    for (nm in names(obj)) {
      hit <- find_xgb(obj[[nm]], max_depth - 1)
      if (!is.null(hit)) return(hit)
    }
  }
  NULL
}

get_model <- function(obj) {
  model <- find_xgb(obj)
  if (is.null(model)) stop("Could not find xgb.Booster.")
  model
}

# ------------------------------------------------------------
# Metadata curation
# ------------------------------------------------------------

label_map <- c(
  age = "Age, years",
  sex_clean = "Sex",
  race_clean = "Race",
  ethnicity_clean = "Ethnicity",
  insurance_clean = "Insurance",

  transfer_clean = "Interfacility transfer",
  mechanism_clean = "Mechanism of injury",
  helmet_clean = "Helmet use, if applicable",

  gcs_eye_clean = "GCS eye",
  gcs_motor_clean = "GCS motor",
  gcs_verbal_clean = "GCS verbal",
  gcs_total_aug = "Total GCS",
  pupil_clean = "Pupillary response",

  sbp_clean = "Systolic blood pressure, mmHg",
  pulse_clean = "Heart rate, beats/min",
  rr_clean = "Respiratory rate, breaths/min",
  spo2_clean = "Oxygen saturation, %",
  respiratoryassistance_clean = "Respiratory assistance on arrival",

  bleeding_disorder = "Bleeding disorder / anticoagulopathy",
  diabetes = "Diabetes",
  copd = "COPD",
  hypertension = "Hypertension",
  current_smoker = "Current smoker",

  dx_concussion = "Concussion",
  dx_cerebral_edema_traumatic = "Traumatic cerebral edema",
  dx_diffuse_axonal_injury = "Diffuse axonal injury",
  dx_focal_contusion_or_iph = "Contusion / intraparenchymal hemorrhage",
  dx_epidural_hematoma = "Epidural hematoma",
  dx_subdural_hematoma = "Subdural hematoma",
  dx_subarachnoid_hemorrhage = "Traumatic subarachnoid hemorrhage",
  dx_other_intracranial_injury = "Other intracranial injury",
  dx_brain_compression_herniation = "Brain compression / herniation",
  dx_skull_fracture_any = "Any skull fracture",
  dx_vault_skull_fracture = "Vault skull fracture",
  dx_base_skull_fracture = "Basilar skull fracture",
  dx_open_wound_head = "Open wound of head"
)

ui_group_map <- c(
  age = "Demographics",
  sex_clean = "Demographics",
  race_clean = "Demographics",
  ethnicity_clean = "Demographics",
  insurance_clean = "Demographics",

  transfer_clean = "Transfer and mechanism",
  mechanism_clean = "Transfer and mechanism",
  helmet_clean = "Transfer and mechanism",

  gcs_eye_clean = "Neurologic status",
  gcs_motor_clean = "Neurologic status",
  gcs_verbal_clean = "Neurologic status",
  gcs_total_aug = "Neurologic status",
  pupil_clean = "Neurologic status",

  sbp_clean = "Vital signs and respiratory support",
  pulse_clean = "Vital signs and respiratory support",
  rr_clean = "Vital signs and respiratory support",
  spo2_clean = "Vital signs and respiratory support",
  respiratoryassistance_clean = "Vital signs and respiratory support",

  bleeding_disorder = "Comorbidities",
  diabetes = "Comorbidities",
  copd = "Comorbidities",
  hypertension = "Comorbidities",
  current_smoker = "Comorbidities",

  dx_concussion = "Intracranial injury pattern",
  dx_cerebral_edema_traumatic = "Intracranial injury pattern",
  dx_diffuse_axonal_injury = "Intracranial injury pattern",
  dx_focal_contusion_or_iph = "Intracranial injury pattern",
  dx_epidural_hematoma = "Intracranial injury pattern",
  dx_subdural_hematoma = "Intracranial injury pattern",
  dx_subarachnoid_hemorrhage = "Intracranial injury pattern",
  dx_other_intracranial_injury = "Intracranial injury pattern",
  dx_brain_compression_herniation = "Intracranial injury pattern",
  dx_skull_fracture_any = "Intracranial injury pattern",
  dx_vault_skull_fracture = "Intracranial injury pattern",
  dx_base_skull_fracture = "Intracranial injury pattern",
  dx_open_wound_head = "Intracranial injury pattern"
)

visible_vars <- c(
  "age", "sex_clean", "race_clean", "ethnicity_clean", "insurance_clean",
  "transfer_clean", "mechanism_clean", "helmet_clean",
  "gcs_eye_clean", "gcs_motor_clean", "gcs_verbal_clean", "gcs_total_aug", "pupil_clean",
  "sbp_clean", "pulse_clean", "rr_clean", "spo2_clean", "respiratoryassistance_clean",
  "bleeding_disorder", "diabetes", "copd", "hypertension", "current_smoker",
  "dx_concussion", "dx_cerebral_edema_traumatic", "dx_diffuse_axonal_injury",
  "dx_focal_contusion_or_iph", "dx_epidural_hematoma", "dx_subdural_hematoma",
  "dx_subarachnoid_hemorrhage", "dx_other_intracranial_injury",
  "dx_brain_compression_herniation", "dx_skull_fracture_any",
  "dx_vault_skull_fracture", "dx_base_skull_fracture", "dx_open_wound_head"
)

derived_vars <- c(
  "age_group_aug",
  "gcs_severity_aug",
  "hypotension_sbp90_aug",
  "hypoxia_spo2_90_aug",
  "tachycardia_120_aug",
  "abnormal_rr_aug",

  "dx_any_s06_intracranial",
  "dx_intracranial_hemorrhage_any",
  "dx_multiple_intracranial_patterns",
  "dx_polyregion_injury_count",
  "dx_polyregion_2plus",
  "dx_polyregion_3plus",

  "dx_facial_fracture",
  "dx_spinal_cord_injury",
  "dx_neck_vascular_injury",
  "dx_thoracic_injury",
  "dx_abdominal_pelvic_injury",
  "dx_upper_extremity_injury",
  "dx_lower_extremity_injury",

  "n_preexisting_conditions"
)

metadata[variable %in% names(label_map), label := unname(label_map[variable])]
metadata[variable %in% names(ui_group_map), group := unname(ui_group_map[variable])]
metadata[variable %in% derived_vars, input_type := "derived"]
metadata[!variable %in% visible_vars & !variable %in% derived_vars, input_type := "derived"]

yesno_vars <- c(
  "bleeding_disorder", "diabetes", "copd", "hypertension", "current_smoker",
  grep("^dx_", metadata$variable, value = TRUE)
)

metadata[variable %in% yesno_vars & input_type != "derived", `:=`(
  input_type = "yesno",
  default = "0",
  choices = "0||1"
)]

metadata[variable == "age", `:=`(input_type = "numeric", min = 18, max = 120, default = "65")]
metadata[variable == "sbp_clean", `:=`(input_type = "numeric", min = 0, max = 300, default = "120")]
metadata[variable == "pulse_clean", `:=`(input_type = "numeric", min = 0, max = 250, default = "80")]
metadata[variable == "rr_clean", `:=`(input_type = "numeric", min = 0, max = 80, default = "16")]
metadata[variable == "spo2_clean", `:=`(input_type = "numeric", min = 0, max = 100, default = "98")]

metadata[variable == "gcs_eye_clean", `:=`(input_type = "gcs_select", default = "4", choices = "1||2||3||4")]
metadata[variable == "gcs_motor_clean", `:=`(input_type = "gcs_select", default = "6", choices = "1||2||3||4||5||6")]
metadata[variable == "gcs_verbal_clean", `:=`(input_type = "gcs_select", default = "5", choices = "1||2||3||4||5")]
metadata[variable == "gcs_total_aug", `:=`(input_type = "gcs_select", default = "15", choices = paste(3:15, collapse = "||"))]

metadata[, app_order := match(variable, visible_vars)]
metadata[is.na(app_order), app_order := 9999L]
metadata[is.na(group) & variable %in% visible_vars, group := "Other inputs"]
metadata[is.na(label) | label == "", label := variable]

# ------------------------------------------------------------
# Endpoint-specific Quick Mode predictor sets
# These sets intentionally exclude ISS/AIS and related registry severity fields.
# ------------------------------------------------------------

quick_predictor_sets <- list(
  icu_admission = c(
    "dx_intracranial_hemorrhage_any", "gcs_total_aug", "dx_multiple_intracranial_patterns",
    "gcs_verbal_clean", "dx_concussion", "n_preexisting_conditions", "gcs_motor_clean",
    "age", "respiratoryassistance_clean", "gcs_severity_aug", "pupil_clean",
    "dx_subarachnoid_hemorrhage", "dx_spinal_cord_injury", "sbp_clean", "pulse_clean",
    "dx_cerebral_edema_traumatic", "transfer_clean", "dx_epidural_hematoma",
    "spo2_clean", "gcs_eye_clean", "rr_clean", "mechanism_clean", "insurance_clean"
  ),
  mechanical_ventilation = c(
    "gcs_total_aug", "gcs_motor_clean", "gcs_verbal_clean", "age", "pulse_clean",
    "transfer_clean", "dx_multiple_intracranial_patterns", "dx_cerebral_edema_traumatic",
    "sbp_clean", "rr_clean", "n_preexisting_conditions", "spo2_clean", "pupil_clean",
    "dx_intracranial_hemorrhage_any", "dx_concussion", "mechanism_clean",
    "dx_brain_compression_herniation", "dx_diffuse_axonal_injury"
  ),
  hlos_ge20 = c(
    "gcs_total_aug", "age", "gcs_verbal_clean", "n_preexisting_conditions", "pupil_clean",
    "gcs_motor_clean", "dx_intracranial_hemorrhage_any", "dx_multiple_intracranial_patterns",
    "race_clean", "mechanism_clean", "insurance_clean", "dx_diffuse_axonal_injury",
    "dx_concussion", "transfer_clean", "respiratoryassistance_clean", "ethnicity_clean",
    "gcs_eye_clean", "sbp_clean", "sex_clean", "dx_focal_contusion_or_iph",
    "dx_subdural_hematoma", "dx_polyregion_3plus"
  ),
  icu_los_ge8 = c(
    "gcs_total_aug", "gcs_motor_clean", "age", "pupil_clean", "gcs_verbal_clean",
    "dx_diffuse_axonal_injury", "n_preexisting_conditions", "dx_spinal_cord_injury",
    "dx_multiple_intracranial_patterns", "mechanism_clean", "dx_intracranial_hemorrhage_any",
    "insurance_clean", "dx_subarachnoid_hemorrhage", "sex_clean", "rr_clean",
    "transfer_clean", "dx_focal_contusion_or_iph", "race_clean", "spo2_clean",
    "dx_concussion", "dx_subdural_hematoma", "sbp_clean", "age_group_aug", "dx_polyregion_3plus"
  ),
  vent_days_ge8 = c(
    "age", "pupil_clean", "dx_diffuse_axonal_injury", "mechanism_clean", "gcs_total_aug",
    "dx_multiple_intracranial_patterns", "insurance_clean", "sbp_clean",
    "dx_cerebral_edema_traumatic", "dx_subarachnoid_hemorrhage", "n_preexisting_conditions",
    "gcs_motor_clean", "race_clean", "dx_brain_compression_herniation", "dx_epidural_hematoma",
    "rr_clean", "spo2_clean", "transfer_clean", "sex_clean", "dx_polyregion_3plus"
  ),
  icp_monitor_evd_bolt = c(
    "gcs_total_aug", "gcs_motor_clean", "dx_intracranial_hemorrhage_any",
    "dx_cerebral_edema_traumatic", "age", "dx_concussion", "dx_multiple_intracranial_patterns",
    "pupil_clean", "gcs_verbal_clean", "dx_brain_compression_herniation", "n_preexisting_conditions",
    "mechanism_clean", "dx_other_intracranial_injury", "dx_subdural_hematoma",
    "dx_subarachnoid_hemorrhage", "dx_focal_contusion_or_iph", "dx_vault_skull_fracture",
    "dx_diffuse_axonal_injury", "dx_skull_fracture_any", "dx_spinal_cord_injury",
    "age_group_aug", "transfer_clean", "sbp_clean"
  ),
  craniotomy_craniectomy = c(
    "gcs_total_aug", "dx_subdural_hematoma", "dx_epidural_hematoma",
    "dx_cerebral_edema_traumatic", "dx_vault_skull_fracture", "dx_intracranial_hemorrhage_any",
    "age", "mechanism_clean", "pupil_clean", "dx_brain_compression_herniation",
    "dx_skull_fracture_any", "dx_focal_contusion_or_iph", "dx_concussion",
    "dx_diffuse_axonal_injury", "insurance_clean", "dx_multiple_intracranial_patterns",
    "race_clean", "sex_clean", "dx_other_intracranial_injury", "gcs_motor_clean",
    "gcs_verbal_clean", "gcs_eye_clean", "transfer_clean", "sbp_clean"
  )
)

source_var_map <- list(
  age_group_aug = c("age"),
  gcs_severity_aug = c("gcs_total_aug"),
  hypotension_sbp90_aug = c("sbp_clean"),
  hypoxia_spo2_90_aug = c("spo2_clean"),
  tachycardia_120_aug = c("pulse_clean"),
  abnormal_rr_aug = c("rr_clean"),

  dx_any_s06_intracranial = c("dx_concussion", "dx_cerebral_edema_traumatic", "dx_diffuse_axonal_injury",
                              "dx_focal_contusion_or_iph", "dx_epidural_hematoma", "dx_subdural_hematoma",
                              "dx_subarachnoid_hemorrhage", "dx_other_intracranial_injury"),
  dx_intracranial_hemorrhage_any = c("dx_focal_contusion_or_iph", "dx_epidural_hematoma", "dx_subdural_hematoma", "dx_subarachnoid_hemorrhage"),
  dx_multiple_intracranial_patterns = c("dx_concussion", "dx_cerebral_edema_traumatic", "dx_diffuse_axonal_injury",
                                        "dx_focal_contusion_or_iph", "dx_epidural_hematoma", "dx_subdural_hematoma",
                                        "dx_subarachnoid_hemorrhage", "dx_other_intracranial_injury",
                                        "dx_brain_compression_herniation"),

  n_preexisting_conditions = c("bleeding_disorder", "diabetes", "copd", "hypertension", "current_smoker"),

  dx_polyregion_injury_count = c("dx_facial_fracture", "dx_spinal_cord_injury", "dx_thoracic_injury",
                                 "dx_abdominal_pelvic_injury", "dx_upper_extremity_injury", "dx_lower_extremity_injury"),
  dx_polyregion_2plus = c("dx_facial_fracture", "dx_spinal_cord_injury", "dx_thoracic_injury",
                          "dx_abdominal_pelvic_injury", "dx_upper_extremity_injury", "dx_lower_extremity_injury"),
  dx_polyregion_3plus = c("dx_facial_fracture", "dx_spinal_cord_injury", "dx_thoracic_injury",
                          "dx_abdominal_pelvic_injury", "dx_upper_extremity_injury", "dx_lower_extremity_injury")
)

source_vars_for_predictors <- function(preds) {
  preds <- setdiff(preds, registry_severity_vars)
  preds <- preds[!grepl("^ais_", preds)]
  unlist(lapply(preds, function(v) source_var_map[[v]] %||% v), use.names = FALSE)
}

endpoint_visible_vars <- function(endpoint, mode = c("quick", "full")) {
  mode <- match.arg(mode)
  if (mode == "full" || endpoint == "all") return(visible_vars)
  preds <- quick_predictor_sets[[endpoint]]
  vars <- unique(c(source_vars_for_predictors(preds), "age", "gcs_total_aug"))
  vars <- vars[vars %in% visible_vars]
  vars[order(match(vars, visible_vars))]
}

needs_extracranial_profile <- function(endpoint, mode = c("quick", "full")) {
  mode <- match.arg(mode)
  if (mode == "full" || endpoint == "all") return(TRUE)
  preds <- quick_predictor_sets[[endpoint]]
  if (is.null(preds)) return(FALSE)
  any(grepl("dx_polyregion|dx_facial|dx_spinal|dx_thoracic|dx_abdominal|dx_upper|dx_lower", preds))
}

# ------------------------------------------------------------
# Derived values
# ------------------------------------------------------------

split_choices <- function(x, fallback = c("0", "1")) {
  out <- unlist(strsplit(as.character(x), "\\|\\|"))
  out <- out[!is.na(out) & nzchar(out)]
  if (length(out) == 0) fallback else out
}

first_or <- function(x, fallback) {
  if (length(x) == 0 || is.na(x[1]) || !nzchar(as.character(x[1]))) return(fallback)
  x[1]
}

derive_age_group <- function(age, choices) {
  age <- safe_numeric(age, NA)
  if (!is.finite(age)) return(choices[1])
  if (age < 40) return(first_or(choices[grepl("18|39|young", choices, ignore.case = TRUE)], choices[1]))
  if (age < 65) return(first_or(choices[grepl("40|64|adult", choices, ignore.case = TRUE)], choices[1]))
  if (age < 75) return(first_or(choices[grepl("65|74", choices, ignore.case = TRUE)], choices[1]))
  hit <- choices[grepl("75|80|elder|older", choices, ignore.case = TRUE)][1]
  ifelse(is.na(hit), choices[length(choices)], hit)
}

derive_gcs_severity <- function(gcs, choices) {
  gcs <- safe_numeric(gcs, NA)
  if (!is.finite(gcs)) return(choices[1])
  if (gcs >= 13) return(first_or(choices[grepl("mild|13", choices, ignore.case = TRUE)], choices[1]))
  if (gcs >= 9) return(first_or(choices[grepl("moderate|9", choices, ignore.case = TRUE)], choices[1]))
  hit <- choices[grepl("severe|3", choices, ignore.case = TRUE)][1]
  ifelse(is.na(hit), choices[1], hit)
}

derived_value <- function(v, row, input) {
  choices <- split_choices(row$choices, c("0", "1"))

  if (v == "age_group_aug") return(derive_age_group(input[[input_id("age")]], choices))
  if (v == "gcs_severity_aug") return(derive_gcs_severity(input[[input_id("gcs_total_aug")]], choices))
  if (v == "hypotension_sbp90_aug") return(as.integer(safe_numeric(input[[input_id("sbp_clean")]], 999) < 90))
  if (v == "hypoxia_spo2_90_aug") return(as.integer(safe_numeric(input[[input_id("spo2_clean")]], 999) <= 90))
  if (v == "tachycardia_120_aug") return(as.integer(safe_numeric(input[[input_id("pulse_clean")]], 0) >= 120))
  if (v == "abnormal_rr_aug") {
    rr <- safe_numeric(input[[input_id("rr_clean")]], 16)
    return(as.integer(rr < 10 | rr > 29))
  }

  if (v == "dx_any_s06_intracranial") return(1L)

  if (v == "n_preexisting_conditions") {
    return(
      get_yesno(input, "bleeding_disorder") +
        get_yesno(input, "diabetes") +
        get_yesno(input, "copd") +
        get_yesno(input, "hypertension") +
        get_yesno(input, "current_smoker")
    )
  }

  ich_any <- get_yesno(input, "dx_focal_contusion_or_iph") |
    get_yesno(input, "dx_epidural_hematoma") |
    get_yesno(input, "dx_subdural_hematoma") |
    get_yesno(input, "dx_subarachnoid_hemorrhage")

  intracranial_pattern_count <- sum(c(
    get_yesno(input, "dx_concussion"),
    get_yesno(input, "dx_cerebral_edema_traumatic"),
    get_yesno(input, "dx_diffuse_axonal_injury"),
    get_yesno(input, "dx_focal_contusion_or_iph"),
    get_yesno(input, "dx_epidural_hematoma"),
    get_yesno(input, "dx_subdural_hematoma"),
    get_yesno(input, "dx_subarachnoid_hemorrhage"),
    get_yesno(input, "dx_other_intracranial_injury"),
    get_yesno(input, "dx_brain_compression_herniation")
  ), na.rm = TRUE)

  if (v == "dx_intracranial_hemorrhage_any") return(as.integer(ich_any))
  if (v == "dx_multiple_intracranial_patterns") return(as.integer(intracranial_pattern_count >= 2))

  face_any <- ext_any(input, "face")
  thorax_any <- ext_any(input, "thorax")
  abdomen_any <- ext_any(input, "abdomen")
  spine_any <- ext_any(input, "spine")
  upper_any <- ext_any(input, "upper_ext")
  lower_any <- ext_any(input, "lower_ext")

  if (v == "dx_facial_fracture") return(face_any)
  if (v == "dx_thoracic_injury") return(thorax_any)
  if (v == "dx_abdominal_pelvic_injury") return(abdomen_any)
  if (v == "dx_spinal_cord_injury") return(spine_any)
  if (v == "dx_neck_vascular_injury") return(0L)
  if (v == "dx_upper_extremity_injury") return(upper_any)
  if (v == "dx_lower_extremity_injury") return(lower_any)

  dx_poly_count <- 1L + face_any + thorax_any + abdomen_any + spine_any + upper_any + lower_any
  if (v == "dx_polyregion_injury_count") return(dx_poly_count)
  if (v == "dx_polyregion_2plus") return(as.integer(dx_poly_count >= 2))
  if (v == "dx_polyregion_3plus") return(as.integer(dx_poly_count >= 3))

  row$default
}

# ------------------------------------------------------------
# Prediction helpers
# ------------------------------------------------------------

make_one_row <- function(input, predictors) {
  out <- data.frame(row.names = 1)

  predictors <- setdiff(predictors, registry_severity_vars)
  predictors <- predictors[!grepl("^ais_", predictors)]

  for (v in predictors) {
    row <- metadata[variable == v][1]

    if (nrow(row) == 0) {
      out[[v]] <- 0
      next
    }

    val <- input[[input_id(v)]]
    input_type <- as.character(row$input_type %||% "")
    model_type <- as.character(row$model_type %||% "")

    if (input_type == "derived") {
      val <- derived_value(v, row, input)
    }

    if (model_type %in% c("numeric", "numeric_binary") || input_type %in% c("numeric", "gcs_select")) {
      out[[v]] <- safe_numeric(val, safe_numeric(row$default, 0))
    } else if (input_type == "yesno") {
      if (model_type %in% c("numeric_binary", "numeric")) {
        out[[v]] <- safe_int01(val)
      } else {
        choices <- split_choices(row$choices, c("0", "1"))
        val <- as.character(ifelse(safe_int01(val) == 1L, "1", "0"))
        out[[v]] <- factor(val, levels = choices)
      }
    } else {
      choices <- split_choices(row$choices, as.character(row$default))
      default <- as.character(row$default)
      if (is.na(default) || !nzchar(default) || !default %in% choices) default <- choices[1]
      if (is.null(val) || length(val) == 0 || !as.character(val[[1]]) %in% choices) val <- default
      out[[v]] <- factor(as.character(val[[1]]), levels = choices)
    }
  }

  out
}

align_matrix <- function(newdata, predictors, feature_names) {
  f <- as.formula(paste("~", paste(predictors, collapse = " + "), "- 1"))
  mm <- Matrix::sparse.model.matrix(f, data = newdata, na.action = stats::na.pass)

  missing_cols <- setdiff(feature_names, colnames(mm))
  if (length(missing_cols) > 0) {
    z <- Matrix::Matrix(0, nrow = nrow(mm), ncol = length(missing_cols), sparse = TRUE)
    colnames(z) <- missing_cols
    mm <- cbind(mm, z)
  }

  extra_cols <- setdiff(colnames(mm), feature_names)
  if (length(extra_cols) > 0) {
    mm <- mm[, setdiff(colnames(mm), extra_cols), drop = FALSE]
  }

  mm[, feature_names, drop = FALSE]
}

predict_binary <- function(obj, input, fallback_predictors) {
  model <- get_model(obj)
  predictors <- obj$predictors
  if (is.null(predictors)) predictors <- fallback_predictors
  feature_names <- obj$feature_names

  if (is.null(feature_names)) stop("Binary model is missing feature_names.")

  bad <- contains_registry_severity(c(predictors, feature_names))
  if (length(bad) > 0) {
    stop("This model bundle still contains retired ISS/AIS predictors. Rerun the no-ISS/no-AIS training pipeline and replace data/model_bundle.rds and data/predictor_metadata.rds.")
  }

  newdata <- make_one_row(input, predictors)
  mm <- align_matrix(newdata, predictors, feature_names)
  raw <- as.numeric(predict(model, xgb.DMatrix(mm))[1])
  apply_binary_recalibration(raw, obj)
}

predict_multiclass <- function(obj, input, fallback_predictors) {
  model <- get_model(obj)
  predictors <- obj$predictors
  if (is.null(predictors)) predictors <- fallback_predictors
  feature_names <- obj$feature_names
  levels <- obj$class_levels %||% discharge_levels

  if (is.null(feature_names)) stop("Discharge model is missing feature_names.")

  bad <- contains_registry_severity(c(predictors, feature_names))
  if (length(bad) > 0) {
    stop("This model bundle still contains retired ISS/AIS predictors. Rerun the no-ISS/no-AIS training pipeline and replace data/model_bundle.rds and data/predictor_metadata.rds.")
  }

  newdata <- make_one_row(input, predictors)
  mm <- align_matrix(newdata, predictors, feature_names)
  raw <- as.numeric(predict(model, xgb.DMatrix(mm)))

  n_class <- length(levels)
  if (length(raw) > n_class) raw <- raw[seq_len(n_class)]
  raw <- raw / sum(raw)
  names(raw) <- levels
  raw
}

# ------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------

endpoint_labels <- c(
  discharge = "Discharge disposition",
  icu_admission = "ICU admission",
  mechanical_ventilation = "Mechanical ventilation",
  hlos_ge20 = "Hospital LOS ≥20 days",
  icu_los_ge8 = "ICU LOS ≥8 days, if ICU admitted",
  vent_days_ge8 = "Ventilator duration ≥8 days, if ventilated",
  icp_monitor_evd_bolt = "ICP monitor/EVD/BOLT placement",
  craniotomy_craniectomy = "Craniotomy/craniectomy"
)

endpoint_choices <- c(
  "Discharge disposition" = "discharge",
  "ICU admission" = "icu_admission",
  "Mechanical ventilation" = "mechanical_ventilation",
  "ICP monitor/EVD/BOLT placement" = "icp_monitor_evd_bolt",
  "Craniotomy/craniectomy" = "craniotomy_craniectomy",
  "Hospital LOS ≥20 days" = "hlos_ge20",
  "ICU LOS ≥8 days, if ICU admitted" = "icu_los_ge8",
  "Ventilator duration ≥8 days, if ventilated" = "vent_days_ge8"
)

binary_order <- c(
  "icu_admission", "mechanical_ventilation", "icp_monitor_evd_bolt", "craniotomy_craniectomy",
  "hlos_ge20", "icu_los_ge8", "vent_days_ge8"
)

endpoint_family <- c(
  icu_admission = "acute",
  mechanical_ventilation = "acute",
  icp_monitor_evd_bolt = "neuro",
  craniotomy_craniectomy = "neuro",
  hlos_ge20 = "prolonged",
  icu_los_ge8 = "prolonged",
  vent_days_ge8 = "prolonged"
)

result_card <- function(name, prob, family = "acute", subtext = NULL) {
  pct <- percent(prob, accuracy = 0.1)
  width <- paste0(round(100 * prob, 1), "%")
  div(
    class = paste("result-card", paste0("result-", family)),
    div(
      class = "result-card-top",
      div(class = "result-name", name),
      div(class = "result-percent", pct)
    ),
    div(class = "result-bar", div(class = "result-fill", style = paste0("width:", width, ";"))),
    if (!is.null(subtext)) div(class = "result-subtext", subtext)
  )
}

result_section <- function(title, note, cards, grid_class = "") {
  div(
    class = "result-section",
    div(
      class = "result-section-header",
      h3(class = "result-section-title", title),
      div(class = "result-section-note", note)
    ),
    div(class = paste("result-grid", grid_class), cards)
  )
}

split_choice_display <- function(x) {
  x <- as.character(x)
  x <- gsub("_", " ", x)
  x
}

labelled_choices <- function(v, choices) {
  choices <- as.character(choices)
  labs <- choices
  if (v == "sex_clean") {
    labs <- c(Male = "Male", Female = "Female", `Non-binary` = "Non-binary")[choices] %||% choices
  }
  setNames(choices, split_choice_display(labs))
}

gcs_eye_choices <- c("1 - No eye opening" = "1", "2 - To pain" = "2", "3 - To speech" = "3", "4 - Spontaneous" = "4")
gcs_motor_choices <- c("1 - None" = "1", "2 - Extension" = "2", "3 - Flexion" = "3", "4 - Withdraws" = "4", "5 - Localizes" = "5", "6 - Obeys" = "6")
gcs_verbal_choices <- c("1 - None" = "1", "2 - Incomprehensible" = "2", "3 - Inappropriate words" = "3", "4 - Confused" = "4", "5 - Oriented" = "5")
gcs_total_choices <- setNames(as.character(3:15), as.character(3:15))

extracranial_severity_control <- function(region, label) {
  selectInput(
    paste0("ui__", region, "_severity"),
    label,
    choices = c("None documented" = "none", "Present" = "present", "Major/severe" = "severe"),
    selected = "none",
    selectize = FALSE
  )
}

make_input_control <- function(row) {
  input_type <- as.character(row$input_type %||% "")
  if (input_type == "derived") return(NULL)

  id <- input_id(row$variable)

  if (input_type == "numeric") {
    numericInput(
      inputId = id,
      label = row$label,
      value = safe_numeric(row$default, 0),
      min = safe_numeric(row$min, NA),
      max = safe_numeric(row$max, NA),
      step = 1
    )
  } else if (input_type == "gcs_select") {
    choices <- switch(
      row$variable,
      gcs_eye_clean = gcs_eye_choices,
      gcs_motor_clean = gcs_motor_choices,
      gcs_verbal_clean = gcs_verbal_choices,
      gcs_total_aug = gcs_total_choices,
      gcs_total_choices
    )
    selectInput(id, row$label, choices = choices, selected = as.character(row$default), selectize = FALSE)
  } else if (input_type == "yesno") {
    selectInput(id, row$label, choices = c("No" = "0", "Yes" = "1"), selected = "0", selectize = FALSE)
  } else {
    choices <- split_choices(row$choices, as.character(row$default))
    default <- as.character(row$default)
    if (is.na(default) || !nzchar(default) || !default %in% choices) default <- choices[1]
    selectInput(id, row$label, choices = labelled_choices(row$variable, choices), selected = default, selectize = FALSE)
  }
}

input_group_ui <- function(group_name, allowed_vars = visible_vars) {
  rows <- metadata[group == group_name & input_type != "derived" & variable %in% allowed_vars][order(app_order)]
  if (nrow(rows) == 0) return(NULL)

  accordion_panel(
    title = group_name,
    lapply(seq_len(nrow(rows)), function(i) {
      make_input_control(rows[i])
    })
  )
}

extracranial_profile_ui <- function() {
  accordion_panel(
    title = "Major extracranial injury profile",
    extracranial_severity_control("face", "Face injury"),
    extracranial_severity_control("thorax", "Thoracic injury"),
    extracranial_severity_control("abdomen", "Abdominal/pelvic injury"),
    extracranial_severity_control("spine", "Spine/spinal cord injury"),
    extracranial_severity_control("upper_ext", "Upper extremity injury"),
    extracranial_severity_control("lower_ext", "Lower extremity injury")
  )
}

# ------------------------------------------------------------
# App UI
# ------------------------------------------------------------

ui <- page_fluid(
  theme = bs_theme(version = 5, bootswatch = "flatly", primary = "#1f4e79", bg = "#f4f7fb", fg = "#243447"),
  tags$head(
    tags$style(
      HTML("
        body { background: #f4f7fb; }
        .app-container { max-width: 1380px; margin: 0 auto; padding: 24px 22px 36px 22px; }
        .app-header, .card { background: #fff; border-radius: 24px !important; box-shadow: 0 8px 28px rgba(31,52,73,0.07); border: 1px solid #e7edf5 !important; }
        .app-header { padding: 22px 28px; margin-bottom: 24px; }
        .header-grid { display: grid; grid-template-columns: 92px 1fr; gap: 20px; align-items: center; }
        .ohsu-logo { width: 86px; height: auto; display: block; }
        .logo-badge { width: 86px; height: 86px; border-radius: 22px; background: #1f4e79; color: white; display: flex; align-items: center; justify-content: center; font-weight: 900; }
        .header-title { margin: 0; font-weight: 800; font-size: clamp(1.9rem,3.3vw,3.0rem); }
        .header-subtitle { margin-bottom: 0; color: #526579; }
        .sticky-panel { position: sticky; top: 24px; }
        .input-scroll { max-height: calc(100vh - 330px); overflow-y: auto; padding-right: 6px; margin-bottom: 12px; }
        .input-scroll::-webkit-scrollbar { width: 8px; }
        .input-scroll::-webkit-scrollbar-thumb { background: #c9d6e2; border-radius: 8px; }
        .input-scroll::-webkit-scrollbar-track { background: #eef3f8; border-radius: 8px; }
        .section-title { font-weight: 800; margin-bottom: 14px; }
        .subsection-note { color: #526579; font-weight: 650; margin-top: -4px; margin-bottom: 16px; line-height: 1.45; }
        .form-label { font-weight: 650; }
        .form-control, .form-select { border-radius: 14px !important; min-height: 42px; }
        .btn-primary { border-radius: 14px !important; font-weight: 750; min-height: 46px; margin-top: 6px; }
        .mode-box { background: #edf4fb; border: 1px solid #dbe8f5; border-radius: 18px; padding: 14px 14px 2px 14px; margin-bottom: 14px; }
        .quick-note { font-size: 0.88rem; color: #526579; line-height: 1.42; margin-top: -4px; margin-bottom: 12px; }
        .block-gap { height: 18px; }
        .note { font-size: 0.9rem; color: #6b7a8c; margin-top: 10px; line-height: 1.45; }
        .detail-card h3 { font-size: 1.08rem; font-weight: 750; color: #243447; margin-top: 0; margin-bottom: 0.8rem; }
        .detail-card ul { margin-bottom: 0; padding-left: 1.15rem; }
        .detail-card li { color: #425466; margin-bottom: 0.48rem; line-height: 1.5; }
        .detail-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 26px 38px; }
        .result-section { border: 1px solid #e1eaf3; border-radius: 22px; padding: 16px; background: #ffffff; margin-bottom: 14px; }
        .result-section-header { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
        .result-section-title { margin: 0; font-weight: 850; color: #243447; font-size: 1.1rem; }
        .result-section-note { color: #6b7a8c; font-size: 0.86rem; font-weight: 650; white-space: nowrap; }
        .result-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
        .result-grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .result-grid.one { grid-template-columns: minmax(0, 1fr); }
        .result-card { border: 1px solid #dbe8f5; background: #f8fbfe; border-radius: 18px; padding: 14px 14px 12px 14px; min-width: 0; }
        .result-card-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
        .result-name { color: #425466; font-weight: 750; line-height: 1.22; min-width: 0; }
        .result-percent { color: #243447; font-weight: 900; font-size: clamp(1.45rem, 3vw, 2.05rem); line-height: 1; white-space: nowrap; }
        .result-bar { width: 100%; height: 13px; border-radius: 999px; background: #e7edf5; overflow: hidden; }
        .result-fill { height: 100%; border-radius: 999px; background: #1f4e79; }
        .result-subtext { color: #6b7a8c; font-size: 0.82rem; margin-top: 9px; line-height: 1.35; }
        .result-acute .result-fill { background: #b45f06; }
        .result-neuro .result-fill { background: #6f42c1; }
        .result-prolonged .result-fill { background: #2f7d32; }
        .warning-box { border: 1px solid #ffd9a8; background: #fff7ed; color: #7c3e00; border-radius: 18px; padding: 14px; margin-bottom: 14px; font-weight: 650; }
        @media (max-width:1199px) { .sticky-panel { position: static; } .input-scroll { max-height: none; overflow-y: visible; padding-right: 0; } }
        @media (max-width:991px) { .result-grid, .result-grid.two { grid-template-columns: 1fr; } .result-section-note { white-space: normal; } }
        @media (max-width:767px) { .app-container { padding: 18px 14px 28px 14px; } .header-grid, .detail-grid { grid-template-columns: 1fr; } }
      ")
    )
  ),

  div(
    class = "app-container",

    div(
      class = "app-header",
      div(
        class = "header-grid",
        div(
          if (file.exists(file.path("www", "ohsu_logo.png"))) {
            img(src = "ohsu_logo.png", class = "ohsu-logo")
          } else {
            div("OHSU", class = "logo-badge")
          }
        ),
        div(
          h1("TBI Resource Utilization Calculator", class = "header-title"),
          p("Oregon Health & Science University · Department of Neurological Surgery", class = "header-subtitle")
        )
      )
    ),

    layout_columns(
      col_widths = c(4, 8),

      div(
        class = "sticky-panel",
        card(
          card_body(
            h2("Calculator mode", class = "section-title"),
            div(
              class = "mode-box",
              radioButtons(
                "calc_mode",
                label = NULL,
                choices = c("Quick endpoint mode" = "quick", "Full calculator" = "full"),
                selected = "quick",
                inline = FALSE
              ),
              conditionalPanel(
                condition = "input.calc_mode == 'quick'",
                selectInput("quick_endpoint", "Endpoint of interest", choices = endpoint_choices, selected = "craniotomy_craniectomy", selectize = FALSE),
                div(class = "quick-note", "Quick mode shows endpoint-specific inputs and intentionally excludes ISS/AIS registry severity scores.")
              )
            ),

            h2("Admission / initial-workup characteristics", class = "section-title"),
            div(class = "subsection-note", uiOutput("input_mode_note")),
            div(
              class = "input-scroll",
              uiOutput("dynamic_inputs")
            ),
            actionButton("calc", "Calculate risk", class = "btn-primary w-100")
          )
        )
      ),

      div(
        card(
          card_body(
            h2("Predicted inpatient outcomes", class = "section-title"),
            uiOutput("bundle_warning"),
            uiOutput("result_cards"),
            div(class = "note", uiOutput("result_note"))
          )
        )
      )
    ),

    div(class = "block-gap"),

    card(
      class = "detail-card",
      card_body(
        h2("Model details, analysis summary, and intended use", class = "section-title"),
        div(
          class = "detail-grid",

          div(
            h3("Model cohort and intended use"),
            tags$ul(
              tags$li(if (is.na(model_n)) "Model cohort: adult TBI modeling cohort." else paste0("Model cohort: n = ", format(model_n, big.mark = ","), " patients.")),
              tags$li(if (length(model_years) == 0) "Study years: ACS-TQIP model years." else paste0("Study years: ", paste(model_years, collapse = ", "), ".")),
              tags$li("Intended use: this calculator estimates inpatient resource-utilization risk among adults with traumatic brain injury using admission and initial-workup clinical characteristics."),
              tags$li("ISS, AIS severity, and AIS-derived registry summary variables are intentionally excluded from this version.")
            )
          ),

          div(
            h3("Quick Mode"),
            tags$ul(
              tags$li("Quick Mode begins with the endpoint of interest and displays a streamlined input set for that endpoint."),
              tags$li("The deployed full no-ISS/no-AIS model remains the prediction engine; hidden inputs are assigned default/reference values."),
              tags$li("For comprehensive risk estimation across all endpoints, use Full calculator mode.")
            )
          ),

          div(
            h3("Outcomes"),
            tags$ul(
              tags$li("Discharge disposition probabilities: home/home health, post-acute facility, and death/hospice."),
              tags$li("Acute utilization outcomes: ICU admission and mechanical ventilation."),
              tags$li("Neurosurgical resource-utilization outcomes: ICP monitor/EVD/BOLT placement and craniotomy/craniectomy."),
              tags$li("Prolonged utilization outcomes: hospital length of stay ≥20 days, ICU length of stay ≥8 days among ICU patients, and ventilator duration ≥8 days among ventilated patients.")
            )
          ),

          div(
            h3("Interpretation and limitations"),
            tags$ul(
              tags$li("Displayed binary probabilities are recalibrated when calibration parameters are available in the model bundle."),
              tags$li("Predictions reflect observed registry outcomes and should not be interpreted as treatment recommendations or measures of clinical appropriateness."),
              tags$li("This tool is an adjunct to clinical judgment and should undergo external/prospective validation before routine clinical implementation.")
            )
          )
        )
      )
    )
  )
)

# ------------------------------------------------------------
# Server
# ------------------------------------------------------------

server <- function(input, output, session) {

  output$bundle_warning <- renderUI({
    if (length(bundle_registry_hits) == 0 && length(metadata_registry_hits) == 0) return(NULL)

    div(
      class = "warning-box",
      "The currently loaded app data still appears to contain retired ISS/AIS registry-severity fields. ",
      "Rerun the no-ISS/no-AIS pipeline and replace data/model_bundle.rds and data/predictor_metadata.rds before using the calculator."
    )
  })

  output$input_mode_note <- renderUI({
    if (input$calc_mode == "quick") {
      HTML("Quick mode shows inputs most relevant to the selected endpoint and excludes ISS/AIS registry-severity scores.")
    } else {
      HTML("Full mode shows all available clinician-facing inputs used by the no-ISS/no-AIS model family.")
    }
  })

  output$dynamic_inputs <- renderUI({
    mode <- input$calc_mode %||% "quick"
    endpoint <- if (mode == "quick") input$quick_endpoint %||% "craniotomy_craniectomy" else "all"
    allowed <- endpoint_visible_vars(endpoint, mode = mode)

    panels <- list(
      input_group_ui("Demographics", allowed),
      input_group_ui("Transfer and mechanism", allowed),
      input_group_ui("Neurologic status", allowed),
      input_group_ui("Vital signs and respiratory support", allowed),
      input_group_ui("Comorbidities", allowed),
      input_group_ui("Intracranial injury pattern", allowed)
    )

    if (needs_extracranial_profile(endpoint, mode = mode)) {
      panels <- c(panels, list(extracranial_profile_ui()))
    }

    panels <- panels[!vapply(panels, is.null, logical(1))]
    if (length(panels) == 0) return(div("No inputs available for this endpoint."))

    do.call(accordion, c(panels, list(open = c("Demographics", "Neurologic status", "Intracranial injury pattern"))))
  })

  predictions <- eventReactive(input$calc, {
    if (length(bundle_registry_hits) > 0) {
      stop("Loaded model_bundle.rds still contains retired ISS/AIS predictors.")
    }

    out <- list()

    if ("discharge" %in% names(bundle)) {
      out$discharge <- predict_multiclass(bundle$discharge, input, predictors_full)
    } else if ("discharge_3cat_clean" %in% names(bundle)) {
      out$discharge <- predict_multiclass(bundle$discharge_3cat_clean, input, predictors_full)
    }

    for (ep in binary_order) {
      if (ep %in% names(bundle)) {
        fallback <- if (ep %in% c("mechanical_ventilation", "vent_days_ge8")) predictors_no_resp else predictors_full
        out[[ep]] <- predict_binary(bundle[[ep]], input, fallback)
      }
    }

    out
  }, ignoreInit = TRUE)

  output$result_cards <- renderUI({
    preds <- tryCatch(predictions(), error = function(e) e)

    if (inherits(preds, "error")) {
      return(div(class = "warning-box", paste("Prediction unavailable:", preds$message)))
    }

    if (is.null(preds)) {
      return(div(class = "note", "Enter patient characteristics and click Calculate risk."))
    }

    mode <- input$calc_mode %||% "quick"
    endpoint <- input$quick_endpoint %||% "craniotomy_craniectomy"

    sections <- list()

    show_discharge <- mode == "full" || endpoint == "discharge"
    show_binary <- if (mode == "full") binary_order else endpoint

    if (show_discharge && !is.null(preds$discharge)) {
      d <- preds$discharge
      cards <- lapply(names(d), function(nm) {
        result_card(nm, d[[nm]], family = "disposition", subtext = "Three-class discharge model")
      })
      sections <- c(sections, list(result_section("Discharge disposition", "Validation-set probabilities", cards)))
    }

    show_binary <- intersect(show_binary, binary_order)
    show_binary <- show_binary[show_binary %in% names(preds)]
    if (length(show_binary) > 0) {
      acute_eps <- intersect(show_binary, c("icu_admission", "mechanical_ventilation"))
      neuro_eps <- intersect(show_binary, c("icp_monitor_evd_bolt", "craniotomy_craniectomy"))
      prolonged_eps <- intersect(show_binary, c("hlos_ge20", "icu_los_ge8", "vent_days_ge8"))

      if (length(acute_eps) > 0) {
        cards <- lapply(acute_eps, function(ep) result_card(endpoint_labels[[ep]], preds[[ep]], "acute"))
        sections <- c(sections, list(result_section("Acute utilization", "Observed inpatient utilization", cards, ifelse(length(cards) == 1, "one", "two"))))
      }

      if (length(neuro_eps) > 0) {
        cards <- lapply(neuro_eps, function(ep) result_card(endpoint_labels[[ep]], preds[[ep]], "neuro"))
        sections <- c(sections, list(result_section("Neurosurgical resource use", "Observed placement/procedure receipt", cards, ifelse(length(cards) == 1, "one", "two"))))
      }

      if (length(prolonged_eps) > 0) {
        cards <- lapply(prolonged_eps, function(ep) result_card(endpoint_labels[[ep]], preds[[ep]], "prolonged"))
        sections <- c(sections, list(result_section("Prolonged utilization", "Resource-intensity estimates", cards)))
      }
    }

    tagList(sections)
  })

  output$result_note <- renderUI({
    HTML("Predictions estimate observed inpatient resource utilization and discharge outcomes. They are not treatment recommendations and should be interpreted with clinical judgment.")
  })
}

shinyApp(ui, server)
