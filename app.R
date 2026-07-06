# ============================================================
# TBI Resource Utilization Calculator
# Clinician-facing Shiny app with endpoint-specific Quick Mode
#
# Required data files in data/:
#   - model_bundle.rds
#   - predictor_metadata.rds
#
# Quick Mode uses the deployed/full model bundle. It does not require
# separate reduced-model RDS files. It simply shows the endpoint-specific
# high-yield inputs and sends all omitted predictors through the existing
# default/reference-value pathway.
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
predictors_full <- meta_obj$predictors_full
predictors_no_resp <- meta_obj$predictors_no_resp %||% setdiff(predictors_full, "respiratoryassistance_clean")
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

  iss_clean = "Injury Severity Score (ISS; overall trauma injury severity, 1–75)",
  max_head_ais_clean = "Maximum head AIS severity",
  max_extracranial_ais_clean = "Maximum extracranial AIS severity",

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

  iss_clean = "Injury burden",
  max_head_ais_clean = "Injury burden",
  max_extracranial_ais_clean = "Injury burden",

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
  "iss_clean", "max_head_ais_clean", "max_extracranial_ais_clean",
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
  "ais_severe_polyregion_2plus",

  "n_preexisting_conditions"
)

metadata[variable %in% names(label_map), label := unname(label_map[variable])]
metadata[variable %in% names(ui_group_map), group := unname(ui_group_map[variable])]
metadata[variable %in% derived_vars, input_type := "derived"]
metadata[!variable %in% visible_vars & !variable %in% derived_vars, input_type := "derived"]

yesno_vars <- c(
  "bleeding_disorder", "diabetes", "copd", "hypertension", "current_smoker",
  grep("^dx_", metadata$variable, value = TRUE),
  grep("^ais_", metadata$variable, value = TRUE),
  "severe_head_ais3_aug", "critical_head_ais5_aug", "severe_extracranial_ais3_aug",
  "severe_tbi_ais3_src_aug", "severe_extracranial_ais3_src_aug",
  "any_extracranial_injury_aug", "isolated_tbi_derived_aug"
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
metadata[variable == "iss_clean", `:=`(input_type = "numeric", min = 1, max = 75, default = "9")]

metadata[variable == "gcs_eye_clean", `:=`(input_type = "gcs_select", default = "4", choices = "1||2||3||4")]
metadata[variable == "gcs_motor_clean", `:=`(input_type = "gcs_select", default = "6", choices = "1||2||3||4||5||6")]
metadata[variable == "gcs_verbal_clean", `:=`(input_type = "gcs_select", default = "5", choices = "1||2||3||4||5")]
metadata[variable == "gcs_total_aug", `:=`(input_type = "gcs_select", default = "15", choices = paste(3:15, collapse = "||"))]

metadata[variable %in% c("max_head_ais_clean", "max_extracranial_ais_clean"), `:=`(
  input_type = "ais_severity",
  min = 0,
  max = 6,
  default = ifelse(variable == "max_head_ais_clean", "1", "0"),
  choices = "0||1||2||3||4||5||6"
)]

metadata[, app_order := match(variable, visible_vars)]
metadata[is.na(app_order), app_order := 9999L]
metadata[is.na(group) & variable %in% visible_vars, group := "Other inputs"]
metadata[is.na(label) | label == "", label := variable]

# ------------------------------------------------------------
# Endpoint-specific Quick Mode predictor sets
# ------------------------------------------------------------

quick_predictor_sets <- list(
  icu_admission = c(
    "iss_clean",
    "dx_intracranial_hemorrhage_any",
    "max_overall_ais_aug",
    "gcs_total_aug",
    "dx_multiple_intracranial_patterns",
    "gcs_verbal_clean",
    "dx_concussion",
    "n_preexisting_conditions",
    "n_total_ais_codes_aug",
    "n_head_ais_codes",
    "gcs_motor_clean",
    "age",
    "respiratoryassistance_clean",
    "gcs_severity_aug",
    "pupil_clean",
    "dx_subarachnoid_hemorrhage",
    "dx_spinal_cord_injury",
    "max_extracranial_ais_clean",
    "sbp_clean",
    "pulse_clean",
    "max_head_ais_clean",
    "dx_cerebral_edema_traumatic",
    "transfer_clean",
    "n_extracranial_ais_codes",
    "dx_epidural_hematoma",
    "ais_spine_severe3",
    "spo2_clean",
    "gcs_eye_clean",
    "rr_clean",
    "mechanism_clean"
  ),
  mechanical_ventilation = c(
    "gcs_total_aug",
    "iss_clean",
    "gcs_motor_clean",
    "gcs_verbal_clean",
    "n_total_ais_codes_aug",
    "max_overall_ais_aug",
    "age",
    "pulse_clean",
    "transfer_clean",
    "dx_multiple_intracranial_patterns",
    "dx_cerebral_edema_traumatic",
    "sbp_clean",
    "rr_clean",
    "n_preexisting_conditions",
    "spo2_clean",
    "pupil_clean",
    "dx_intracranial_hemorrhage_any",
    "dx_concussion",
    "mechanism_clean",
    "dx_brain_compression_herniation"
  ),
  hlos_ge20 = c(
    "iss_clean",
    "gcs_total_aug",
    "n_total_ais_codes_aug",
    "age",
    "gcs_verbal_clean",
    "n_preexisting_conditions",
    "pupil_clean",
    "ais_severe_region_count",
    "gcs_motor_clean",
    "max_overall_ais_aug",
    "dx_intracranial_hemorrhage_any",
    "max_extracranial_ais_clean",
    "dx_multiple_intracranial_patterns",
    "race_clean",
    "n_extracranial_ais_codes",
    "mechanism_clean",
    "insurance_clean",
    "dx_diffuse_axonal_injury",
    "max_head_ais_clean",
    "dx_concussion",
    "transfer_clean",
    "ais_polyregion_3plus",
    "ais_region_count",
    "respiratoryassistance_clean",
    "ethnicity_clean",
    "gcs_eye_clean",
    "sbp_clean",
    "sex_clean",
    "dx_focal_contusion_or_iph",
    "dx_subdural_hematoma"
  ),
  icu_los_ge8 = c(
    "gcs_total_aug",
    "iss_clean",
    "n_total_ais_codes_aug",
    "gcs_motor_clean",
    "age",
    "pupil_clean",
    "gcs_verbal_clean",
    "max_extracranial_ais_clean",
    "max_overall_ais_aug",
    "dx_diffuse_axonal_injury",
    "n_preexisting_conditions",
    "max_head_ais_clean",
    "dx_spinal_cord_injury",
    "dx_multiple_intracranial_patterns",
    "mechanism_clean",
    "ais_severe_region_count",
    "dx_intracranial_hemorrhage_any",
    "insurance_clean",
    "n_extracranial_ais_codes",
    "dx_subarachnoid_hemorrhage",
    "sex_clean",
    "rr_clean",
    "transfer_clean",
    "dx_focal_contusion_or_iph",
    "race_clean",
    "spo2_clean",
    "dx_concussion",
    "dx_subdural_hematoma",
    "sbp_clean",
    "age_group_aug"
  ),
  vent_days_ge8 = c(
    "iss_clean",
    "age",
    "pupil_clean",
    "n_total_ais_codes_aug",
    "dx_diffuse_axonal_injury",
    "mechanism_clean",
    "gcs_total_aug",
    "max_overall_ais_aug",
    "dx_multiple_intracranial_patterns",
    "insurance_clean",
    "ais_severe_region_count",
    "sbp_clean",
    "dx_cerebral_edema_traumatic",
    "max_head_ais_clean",
    "dx_subarachnoid_hemorrhage",
    "n_preexisting_conditions",
    "gcs_motor_clean",
    "race_clean",
    "dx_brain_compression_herniation",
    "dx_epidural_hematoma"
  ),
  icp_monitor_evd_bolt = c(
    "gcs_total_aug",
    "max_head_ais_clean",
    "gcs_motor_clean",
    "iss_clean",
    "n_head_ais_codes",
    "critical_head_ais5_aug",
    "dx_intracranial_hemorrhage_any",
    "dx_cerebral_edema_traumatic",
    "age",
    "dx_concussion",
    "severe_head_ais3_aug",
    "dx_multiple_intracranial_patterns",
    "n_total_ais_codes_aug",
    "pupil_clean",
    "gcs_verbal_clean",
    "max_overall_ais_aug",
    "dx_brain_compression_herniation",
    "n_preexisting_conditions",
    "mechanism_clean",
    "dx_other_intracranial_injury",
    "dx_subdural_hematoma",
    "dx_subarachnoid_hemorrhage",
    "dx_focal_contusion_or_iph",
    "dx_vault_skull_fracture",
    "dx_diffuse_axonal_injury",
    "max_extracranial_ais_clean",
    "dx_skull_fracture_any",
    "dx_spinal_cord_injury",
    "age_group_aug",
    "severe_tbi_ais3_src_aug"
  ),
  craniotomy_craniectomy = c(
    "max_head_ais_clean",
    "critical_head_ais5_aug",
    "severe_head_ais3_aug",
    "gcs_total_aug",
    "iss_clean",
    "dx_subdural_hematoma",
    "dx_epidural_hematoma",
    "dx_cerebral_edema_traumatic",
    "dx_vault_skull_fracture",
    "dx_intracranial_hemorrhage_any",
    "age",
    "mechanism_clean",
    "n_total_ais_codes_aug",
    "severe_tbi_ais3_src_aug",
    "pupil_clean",
    "max_overall_ais_aug",
    "dx_brain_compression_herniation",
    "n_head_ais_codes",
    "dx_skull_fracture_any",
    "dx_focal_contusion_or_iph",
    "dx_concussion",
    "dx_diffuse_axonal_injury",
    "insurance_clean",
    "dx_multiple_intracranial_patterns",
    "race_clean",
    "sex_clean",
    "dx_other_intracranial_injury",
    "gcs_motor_clean",
    "gcs_verbal_clean",
    "gcs_eye_clean"
  )
)

# Source visible variables needed when a high-yield predictor is derived.
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

  max_overall_ais_aug = c("max_head_ais_clean", "max_extracranial_ais_clean"),
  severe_head_ais3_aug = c("max_head_ais_clean"),
  critical_head_ais5_aug = c("max_head_ais_clean"),
  severe_tbi_ais3_src_aug = c("max_head_ais_clean"),
  severe_extracranial_ais3_aug = c("max_extracranial_ais_clean"),
  severe_extracranial_ais3_src_aug = c("max_extracranial_ais_clean"),
  any_extracranial_injury_aug = c("max_extracranial_ais_clean"),
  isolated_tbi_derived_aug = c("max_extracranial_ais_clean"),

  n_preexisting_conditions = c("bleeding_disorder", "diabetes", "copd", "hypertension", "current_smoker"),
  n_head_ais_codes = c("dx_concussion", "dx_cerebral_edema_traumatic", "dx_diffuse_axonal_injury",
                       "dx_focal_contusion_or_iph", "dx_epidural_hematoma", "dx_subdural_hematoma",
                       "dx_subarachnoid_hemorrhage", "dx_other_intracranial_injury",
                       "dx_brain_compression_herniation", "dx_skull_fracture_any"),
  n_extracranial_ais_codes = c("max_extracranial_ais_clean"),
  n_total_ais_codes_aug = c("max_head_ais_clean", "max_extracranial_ais_clean",
                            "dx_concussion", "dx_cerebral_edema_traumatic", "dx_diffuse_axonal_injury",
                            "dx_focal_contusion_or_iph", "dx_epidural_hematoma", "dx_subdural_hematoma",
                            "dx_subarachnoid_hemorrhage", "dx_other_intracranial_injury",
                            "dx_brain_compression_herniation", "dx_skull_fracture_any"),

  ais_region_count = c("max_head_ais_clean", "max_extracranial_ais_clean"),
  ais_severe_region_count = c("max_head_ais_clean", "max_extracranial_ais_clean"),
  ais_polyregion_2plus = c("max_head_ais_clean", "max_extracranial_ais_clean"),
  ais_polyregion_3plus = c("max_head_ais_clean", "max_extracranial_ais_clean"),
  ais_severe_polyregion_2plus = c("max_head_ais_clean", "max_extracranial_ais_clean")
)

extracranial_derived_prefixes <- c(
  "ais_face", "ais_thorax", "ais_abdomen", "ais_spine", "ais_upper_ext", "ais_lower_ext", "ais_external",
  "dx_facial", "dx_spinal", "dx_thoracic", "dx_abdominal", "dx_upper", "dx_lower",
  "dx_polyregion"
)

source_vars_for_predictors <- function(predictors) {
  src <- character(0)
  for (v in predictors) {
    if (v %in% visible_vars) src <- c(src, v)
    if (v %in% names(source_var_map)) src <- c(src, source_var_map[[v]])
    if (any(startsWith(v, extracranial_derived_prefixes))) {
      src <- c(src, "max_extracranial_ais_clean")
    }
  }
  unique(src[src %in% visible_vars])
}

clinical_core_visible <- c(
  "age", "transfer_clean", "mechanism_clean",
  "gcs_eye_clean", "gcs_motor_clean", "gcs_verbal_clean", "gcs_total_aug", "pupil_clean",
  "sbp_clean", "pulse_clean", "rr_clean", "spo2_clean",
  "iss_clean", "max_head_ais_clean", "max_extracranial_ais_clean",
  "dx_cerebral_edema_traumatic", "dx_diffuse_axonal_injury", "dx_focal_contusion_or_iph",
  "dx_epidural_hematoma", "dx_subdural_hematoma", "dx_subarachnoid_hemorrhage",
  "dx_brain_compression_herniation", "dx_skull_fracture_any"
)

endpoint_visible_vars <- function(endpoint, mode = c("quick", "full")) {
  mode <- match.arg(mode)
  if (mode == "full" || endpoint == "all") return(visible_vars)

  if (endpoint == "discharge") {
    return(unique(c(clinical_core_visible, "sex_clean", "race_clean", "ethnicity_clean", "insurance_clean")))
  }

  preds <- quick_predictor_sets[[endpoint]]
  if (is.null(preds)) return(visible_vars)

  vars <- unique(c(source_vars_for_predictors(preds), "age", "gcs_total_aug"))
  vars <- vars[vars %in% visible_vars]
  vars[order(match(vars, visible_vars))]
}

needs_extracranial_profile <- function(endpoint, mode = c("quick", "full")) {
  mode <- match.arg(mode)
  if (mode == "full" || endpoint == "all") return(TRUE)
  preds <- quick_predictor_sets[[endpoint]]
  if (is.null(preds)) return(FALSE)
  any(grepl("ais_|extracranial|dx_polyregion|dx_facial|dx_spinal|dx_thoracic|dx_abdominal|dx_upper|dx_lower", preds))
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
  external_any <- ext_any(input, "external")

  face_sev <- ext_severe(input, "face")
  thorax_sev <- ext_severe(input, "thorax")
  abdomen_sev <- ext_severe(input, "abdomen")
  spine_sev <- ext_severe(input, "spine")
  upper_sev <- ext_severe(input, "upper_ext")
  lower_sev <- ext_severe(input, "lower_ext")

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

  max_head <- safe_numeric(input[[input_id("max_head_ais_clean")]], 1)
  max_extra <- safe_numeric(input[[input_id("max_extracranial_ais_clean")]], 0)
  max_overall <- max(max_head, max_extra, na.rm = TRUE)

  if (v == "max_overall_ais_aug") return(max_overall)
  if (v == "severe_head_ais3_aug") return(as.integer(max_head >= 3))
  if (v == "critical_head_ais5_aug") return(as.integer(max_head >= 5))
  if (v == "severe_tbi_ais3_src_aug") return(as.integer(max_head >= 3))

  severe_extra <- as.integer(max_extra >= 3 | any(c(face_sev, thorax_sev, abdomen_sev, spine_sev, upper_sev, lower_sev) == 1))
  any_extra <- as.integer(max_extra > 0 | any(c(face_any, thorax_any, abdomen_any, spine_any, upper_any, lower_any, external_any) == 1))

  if (v == "severe_extracranial_ais3_aug") return(severe_extra)
  if (v == "severe_extracranial_ais3_src_aug") return(severe_extra)
  if (v == "any_extracranial_injury_aug") return(any_extra)
  if (v == "isolated_tbi_derived_aug") return(as.integer(severe_extra == 0))

  head_code_count <- max(1L, intracranial_pattern_count + get_yesno(input, "dx_skull_fracture_any"))
  extracranial_count <- face_any + thorax_any + abdomen_any + spine_any + upper_any + lower_any + external_any

  if (v == "n_head_ais_codes") return(head_code_count)
  if (v == "n_extracranial_ais_codes") return(extracranial_count)
  if (v == "n_total_ais_codes_aug") return(head_code_count + extracranial_count)

  if (v == "ais_face_any") return(face_any)
  if (v == "ais_thorax_any") return(thorax_any)
  if (v == "ais_abdomen_any") return(abdomen_any)
  if (v == "ais_spine_any") return(spine_any)
  if (v == "ais_upper_ext_any") return(upper_any)
  if (v == "ais_lower_ext_any") return(lower_any)
  if (v == "ais_external_any") return(external_any)

  if (v == "ais_headneck_severe3") return(as.integer(max_head >= 3))
  if (v == "ais_face_severe3") return(face_sev)
  if (v == "ais_thorax_severe3") return(thorax_sev)
  if (v == "ais_abdomen_severe3") return(abdomen_sev)
  if (v == "ais_spine_severe3") return(spine_sev)
  if (v == "ais_upper_ext_severe3") return(upper_sev)
  if (v == "ais_lower_ext_severe3") return(lower_sev)

  ais_region_count <- as.integer(max_head > 0) + face_any + thorax_any + abdomen_any + spine_any + upper_any + lower_any + external_any
  ais_severe_count <- as.integer(max_head >= 3) + face_sev + thorax_sev + abdomen_sev + spine_sev + upper_sev + lower_sev

  if (v == "ais_region_count") return(ais_region_count)
  if (v == "ais_severe_region_count") return(ais_severe_count)
  if (v == "ais_polyregion_2plus") return(as.integer(ais_region_count >= 2))
  if (v == "ais_polyregion_3plus") return(as.integer(ais_region_count >= 3))
  if (v == "ais_severe_polyregion_2plus") return(as.integer(ais_severe_count >= 2))

  row$default
}

# ------------------------------------------------------------
# Prediction helpers
# ------------------------------------------------------------

make_one_row <- function(input, predictors) {
  out <- data.frame(row.names = 1)

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

    if (model_type %in% c("numeric", "numeric_binary")) {
      out[[v]] <- safe_numeric(val, safe_numeric(row$default, 0))
    } else if (input_type == "yesno") {
      choices <- split_choices(row$choices, c("0", "1"))
      val <- as.character(ifelse(safe_int01(val) == 1L, "1", "0"))
      out[[v]] <- factor(val, levels = choices)
    } else if (input_type %in% c("gcs_select", "ais_severity")) {
      out[[v]] <- safe_numeric(val, safe_numeric(row$default, 0))
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
  class_levels <- obj$class_levels
  if (is.null(class_levels)) class_levels <- discharge_levels

  newdata <- make_one_row(input, predictors)
  mm <- align_matrix(newdata, predictors, feature_names)
  raw <- as.numeric(predict(model, xgb.DMatrix(mm)))
  data.table(class = class_levels, probability = raw)
}

predict_optional_binary <- function(endpoint, input, fallback_predictors) {
  if (!endpoint %in% names(bundle)) return(NA_real_)
  tryCatch(
    predict_binary(bundle[[endpoint]], input, fallback_predictors),
    error = function(e) {
      warning("Could not predict endpoint ", endpoint, ": ", conditionMessage(e))
      NA_real_
    }
  )
}

# ------------------------------------------------------------
# Endpoint display metadata
# ------------------------------------------------------------

binary_endpoint_spec <- data.table(
  endpoint = c(
    "icu_admission",
    "mechanical_ventilation",
    "icp_monitor_evd_bolt",
    "craniotomy_craniectomy",
    "hlos_ge20",
    "icu_los_ge8",
    "vent_days_ge8"
  ),
  outcome = c(
    "ICU admission",
    "Mechanical ventilation",
    "ICP monitor/EVD/BOLT placement",
    "Craniotomy/craniectomy",
    "Hospital LOS ≥20 days",
    "ICU LOS ≥8 days, if ICU admitted",
    "Ventilator duration ≥8 days, if ventilated"
  ),
  section = c(
    "Acute utilization",
    "Acute utilization",
    "Neurosurgical resource utilization",
    "Neurosurgical resource utilization",
    "Prolonged utilization",
    "Prolonged utilization",
    "Prolonged utilization"
  ),
  type_class = c(
    "result-acute",
    "result-acute",
    "result-neuro",
    "result-neuro",
    "result-prolonged",
    "result-prolonged",
    "result-prolonged"
  ),
  fallback = c(
    "full",
    "no_resp",
    "full",
    "full",
    "full",
    "full",
    "no_resp"
  ),
  subtext = c(
    NA_character_,
    NA_character_,
    "Neurosurgical resource endpoint",
    "Neurosurgical operative endpoint",
    NA_character_,
    "Conditional on ICU admission",
    "Conditional on mechanical ventilation"
  )
)

endpoint_choices <- c(
  "Discharge disposition (3-class)" = "discharge",
  stats::setNames(binary_endpoint_spec$endpoint, binary_endpoint_spec$outcome)
)

# ------------------------------------------------------------
# UI controls
# ------------------------------------------------------------

gcs_eye_choices <- c("1 - None" = "1", "2 - To pain" = "2", "3 - To speech" = "3", "4 - Spontaneous" = "4")
gcs_motor_choices <- c("1 - None" = "1", "2 - Extension" = "2", "3 - Flexion" = "3", "4 - Withdraws" = "4", "5 - Localizes" = "5", "6 - Obeys commands" = "6")
gcs_verbal_choices <- c("1 - None" = "1", "2 - Incomprehensible" = "2", "3 - Inappropriate words" = "3", "4 - Confused" = "4", "5 - Oriented" = "5")
gcs_total_choices <- as.character(3:15)

ais_choices <- c(
  "0 - None" = "0",
  "1 - Minor" = "1",
  "2 - Moderate" = "2",
  "3 - Serious" = "3",
  "4 - Severe" = "4",
  "5 - Critical" = "5",
  "6 - Maximal" = "6"
)

choice_display_label <- function(variable, value) {
  value <- as.character(value)
  if (variable == "mechanism_clean" && value == "Transport/MVC") return("Transport-related injury")
  value
}

labelled_choices <- function(variable, choices) {
  choices <- choices[!is.na(choices) & choices != ""]
  stats::setNames(choices, vapply(choices, function(x) choice_display_label(variable, x), character(1)))
}

extracranial_severity_control <- function(id, label) {
  selectInput(
    inputId = paste0("ui__", id, "_severity"),
    label = label,
    choices = c(
      "None" = "none",
      "Minor/moderate injury" = "minor_moderate",
      "Serious/severe/critical injury" = "severe"
    ),
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
  } else if (input_type == "ais_severity") {
    selectInput(id, row$label, choices = ais_choices, selected = as.character(row$default), selectize = FALSE)
  } else if (input_type == "yesno") {
    selectInput(id, row$label, choices = c("No" = "0", "Yes" = "1"), selected = "0", selectize = FALSE)
  } else if (input_type == "count_select") {
    choices <- split_choices(row$choices, as.character(row$default))
    selectInput(id, row$label, choices = choices, selected = row$default, selectize = FALSE)
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
    extracranial_severity_control("face", "Face injury severity"),
    extracranial_severity_control("thorax", "Thoracic injury severity"),
    extracranial_severity_control("abdomen", "Abdominal/pelvic injury severity"),
    extracranial_severity_control("spine", "Spine/spinal cord injury severity"),
    extracranial_severity_control("upper_ext", "Upper extremity injury severity"),
    extracranial_severity_control("lower_ext", "Lower extremity injury severity"),
    extracranial_severity_control("external", "External/skin injury severity")
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
        .result-fill { height: 100%; border-radius: 999px; }
        .result-subtext { color: #6b7a8c; font-size: 0.82rem; margin-top: 9px; line-height: 1.35; }
        .result-disposition .result-fill { background: #1f4e79; }
        .result-acute .result-fill { background: #b45f06; }
        .result-neuro .result-fill { background: #6f42c1; }
        .result-prolonged .result-fill { background: #2f7d32; }
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
                div(class = "quick-note", "Quick mode shows only endpoint-specific high-yield inputs. Omitted predictors are set to the calculator's default/reference values and the deployed full model is used for prediction.")
              )
            ),

            h2("Admission characteristics", class = "section-title"),
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
              tags$li("Intended use: this calculator estimates inpatient resource-utilization risk among adults with traumatic brain injury using admission-era clinical and injury characteristics."),
              tags$li("The calculator is intended to support triage, early disposition planning, and resource-allocation discussions. It does not replace individualized clinical judgment.")
            )
          ),

          div(
            h3("Quick Mode"),
            tags$ul(
              tags$li("Quick Mode begins with the endpoint of interest and displays a streamlined input set for that endpoint."),
              tags$li("The deployed full model remains the prediction engine; hidden inputs are assigned default/reference values."),
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
              tags$li("Conditional outcomes should be interpreted within the relevant population, such as ICU LOS among patients admitted to the ICU."),
              tags$li("Predictions are derived from registry data and may not capture local practice patterns, bed availability, procedural indication, unmet need, or clinician judgment."),
              tags$li("Displayed probabilities are point estimates. Uncertainty intervals are not shown unless a formal uncertainty-estimation procedure is implemented and validated.")
            )
          )
        )
      )
    )
  )
)

# ------------------------------------------------------------
# Results display helpers
# ------------------------------------------------------------

fmt_prob_label <- function(p) {
  p <- suppressWarnings(as.numeric(p))
  if (!is.finite(p)) return("Not available")
  pct <- 100 * p
  if (pct > 0 && pct < 0.1) return("<0.1%")
  if (pct < 5 || pct > 95) return(paste0(formatC(pct, format = "f", digits = 1), "%"))
  paste0(formatC(round(pct), format = "f", digits = 0), "%")
}

result_card <- function(name, probability, type_class, subtext = NULL) {
  p <- suppressWarnings(as.numeric(probability))
  if (!is.finite(p)) p <- NA_real_
  p_for_bar <- if (is.na(p)) 0 else max(0, min(1, p))
  pct_width <- paste0(round(100 * p_for_bar, 1), "%")

  tags$div(
    class = paste("result-card", type_class),
    tags$div(
      class = "result-card-top",
      tags$div(class = "result-name", name),
      tags$div(class = "result-percent", fmt_prob_label(p))
    ),
    tags$div(
      class = "result-bar",
      tags$div(class = "result-fill", style = paste0("width:", pct_width, ";"))
    ),
    if (!is.null(subtext) && !is.na(subtext)) tags$div(class = "result-subtext", subtext)
  )
}

result_section <- function(title, note, cards, grid_class = "") {
  tags$section(
    class = "result-section",
    tags$div(
      class = "result-section-header",
      tags$h3(class = "result-section-title", title),
      tags$div(class = "result-section-note", note)
    ),
    tags$div(class = paste("result-grid", grid_class), cards)
  )
}

# ------------------------------------------------------------
# Server
# ------------------------------------------------------------

server <- function(input, output, session) {
  selected_mode <- reactive({
    mode <- input$calc_mode %||% "quick"
    if (!mode %in% c("quick", "full")) "quick" else mode
  })

  selected_endpoint <- reactive({
    ep <- input$quick_endpoint %||% "craniotomy_craniectomy"
    if (!ep %in% c("discharge", binary_endpoint_spec$endpoint)) "craniotomy_craniectomy" else ep
  })

  current_allowed_vars <- reactive({
    endpoint_visible_vars(
      endpoint = if (selected_mode() == "full") "all" else selected_endpoint(),
      mode = selected_mode()
    )
  })

  output$input_mode_note <- renderUI({
    if (selected_mode() == "full") {
      tags$span("Full calculator mode displays all clinician-facing inputs and returns all model outputs.")
    } else {
      ep <- selected_endpoint()
      label <- if (ep == "discharge") "Discharge disposition" else binary_endpoint_spec[endpoint == ep]$outcome[1]
      n_inputs <- length(current_allowed_vars())
      tags$span(paste0("Quick Mode for ", label, ": ", n_inputs, " focused input fields shown; other predictors use defaults/reference values."))
    }
  })

  output$dynamic_inputs <- renderUI({
    allowed <- current_allowed_vars()
    mode <- selected_mode()
    ep <- if (mode == "full") "all" else selected_endpoint()

    panels <- list(
      input_group_ui("Demographics", allowed),
      input_group_ui("Transfer and mechanism", allowed),
      input_group_ui("Neurologic status", allowed),
      input_group_ui("Vital signs and respiratory support", allowed),
      input_group_ui("Injury burden", allowed),
      input_group_ui("Comorbidities", allowed),
      input_group_ui("Intracranial injury pattern", allowed)
    )

    if (needs_extracranial_profile(ep, mode)) {
      panels <- c(panels, list(extracranial_profile_ui()))
    }

    panels <- panels[!vapply(panels, is.null, logical(1))]

    do.call(
      accordion,
      c(
        list(id = "input_accordion", open = FALSE),
        panels
      )
    )
  })

  observe({
    bounds <- list(
      age = c(18, 120),
      sbp_clean = c(0, 300),
      pulse_clean = c(0, 250),
      rr_clean = c(0, 80),
      spo2_clean = c(0, 100),
      iss_clean = c(1, 75)
    )

    for (v in names(bounds)) {
      id <- input_id(v)
      val <- suppressWarnings(as.numeric(input[[id]]))
      if (length(val) > 0 && is.finite(val)) {
        lo <- bounds[[v]][1]
        hi <- bounds[[v]][2]
        clipped <- min(max(val, lo), hi)
        if (!identical(val, clipped)) {
          updateNumericInput(session, id, value = clipped)
        }
      }
    }
  })

  observeEvent(
    input$calc,
    {
      tryCatch(
        bslib::accordion_panel_set(id = "input_accordion", values = character(0), session = session),
        error = function(e) NULL
      )
    },
    ignoreInit = TRUE
  )

  results <- eventReactive(
    input$calc,
    {
      if (selected_mode() == "full") {
        disp <- predict_multiclass(bundle$discharge, input, predictors_full)

        bin <- copy(binary_endpoint_spec)
        bin[, probability := NA_real_]

        for (i in seq_len(nrow(bin))) {
          fallback_predictors <- if (bin$fallback[i] == "no_resp") predictors_no_resp else predictors_full
          bin$probability[i] <- predict_optional_binary(bin$endpoint[i], input, fallback_predictors)
        }

        bin[, available := is.finite(probability)]

        return(list(mode = "full", endpoint = "all", discharge = disp, binary = bin))
      }

      ep <- selected_endpoint()
      if (ep == "discharge") {
        disp <- predict_multiclass(bundle$discharge, input, predictors_full)
        return(list(mode = "quick", endpoint = ep, discharge = disp, binary = data.table()))
      }

      spec <- copy(binary_endpoint_spec[endpoint == ep])
      spec[, probability := NA_real_]
      fallback_predictors <- if (spec$fallback[1] == "no_resp") predictors_no_resp else predictors_full
      spec$probability[1] <- predict_optional_binary(ep, input, fallback_predictors)
      spec[, available := is.finite(probability)]

      list(mode = "quick", endpoint = ep, discharge = data.table(), binary = spec)
    },
    ignoreInit = FALSE
  )

  output$result_note <- renderUI({
    req(results())
    if (results()$mode == "quick") {
      tags$span("Quick Mode returns only the selected endpoint. It uses the deployed full model, with omitted inputs set to default/reference values. Switch to Full calculator mode to view all outputs.")
    } else {
      tags$span("Disposition probabilities are mutually exclusive and sum to 100%. Other outcomes are independent binary predictions and should be interpreted separately. Binary probabilities use logistic recalibration when calibration parameters are present in the model bundle.")
    }
  })

  output$result_cards <- renderUI({
    req(results())

    if (results()$mode == "quick") {
      if (results()$endpoint == "discharge") {
        disp <- copy(results()$discharge)
        preferred <- c("Home/home health", "Post-acute facility", "Death/hospice")
        disp[, ord := match(class, preferred)]
        disp[is.na(ord), ord := 99L]
        disp <- disp[order(ord, -probability)]

        disp_cards <- lapply(seq_len(nrow(disp)), function(i) {
          result_card(
            name = disp$class[i],
            probability = disp$probability[i],
            type_class = "result-disposition"
          )
        })

        return(result_section(
          title = "Discharge disposition",
          note = "Quick Mode · Mutually exclusive",
          cards = disp_cards
        ))
      }

      bin <- copy(results()$binary)
      if (nrow(bin) == 0) return(NULL)

      subtext <- bin$subtext[1]
      if (!isTRUE(bin$available[1])) {
        subtext <- ifelse(is.na(subtext), "Model not found in uploaded model bundle", paste(subtext, "· Model not found in uploaded model bundle"))
      } else {
        subtext <- paste0(
          ifelse(is.na(subtext), "", paste0(subtext, " · ")),
          "Endpoint-specific Quick Mode"
        )
      }

      return(result_section(
        title = bin$outcome[1],
        note = "Selected endpoint",
        cards = list(result_card(
          name = bin$outcome[1],
          probability = bin$probability[1],
          type_class = bin$type_class[1],
          subtext = subtext
        )),
        grid_class = "one"
      ))
    }

    disp <- copy(results()$discharge)
    preferred <- c("Home/home health", "Post-acute facility", "Death/hospice")
    disp[, ord := match(class, preferred)]
    disp[is.na(ord), ord := 99L]
    disp <- disp[order(ord, -probability)]

    disp_cards <- lapply(seq_len(nrow(disp)), function(i) {
      result_card(
        name = disp$class[i],
        probability = disp$probability[i],
        type_class = "result-disposition"
      )
    })

    bin <- copy(results()$binary)

    make_cards <- function(dat) {
      if (nrow(dat) == 0) return(list())
      lapply(seq_len(nrow(dat)), function(i) {
        subtext <- dat$subtext[i]
        if (!isTRUE(dat$available[i])) {
          subtext <- ifelse(is.na(subtext), "Model not found in uploaded model bundle", paste(subtext, "· Model not found in uploaded model bundle"))
        }
        result_card(
          name = dat$outcome[i],
          probability = dat$probability[i],
          type_class = dat$type_class[i],
          subtext = subtext
        )
      })
    }

    acute <- bin[section == "Acute utilization"]
    neuro <- bin[section == "Neurosurgical resource utilization"]
    prolonged <- bin[section == "Prolonged utilization"]

    tags$div(
      result_section(
        title = "Discharge disposition",
        note = "Mutually exclusive",
        cards = disp_cards
      ),
      result_section(
        title = "Acute utilization",
        note = "Independent binary estimates",
        cards = make_cards(acute),
        grid_class = "two"
      ),
      result_section(
        title = "Neurosurgical resource utilization",
        note = "Independent binary estimates",
        cards = make_cards(neuro),
        grid_class = "two"
      ),
      result_section(
        title = "Prolonged utilization",
        note = "Independent or conditional estimates",
        cards = make_cards(prolonged)
      )
    )
  })
}

shinyApp(ui, server)
