# ============================================================
# TBI Resource Utilization Calculator
# Clean clinician-facing app.R
# ============================================================

suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(ggplot2)
  library(data.table)
  library(Matrix)
  library(xgboost)
  library(scales)
  library(stringr)
})

bundle <- readRDS(file.path("data", "model_bundle.rds"))
meta_obj <- readRDS(file.path("data", "predictor_metadata.rds"))

metadata <- as.data.table(meta_obj$metadata)
predictors_full <- meta_obj$predictors_full
predictors_no_resp <- meta_obj$predictors_no_resp
model_n <- meta_obj$model_n
model_years <- meta_obj$model_years
discharge_levels <- meta_obj$discharge_levels

input_id <- function(v) paste0("var__", v)

safe_numeric <- function(x, default = 0) {
  y <- suppressWarnings(as.numeric(x))
  ifelse(is.finite(y), y, default)
}

safe_int01 <- function(x) {
  as.integer(as.character(x) %in% c("1", "Yes", "yes", "TRUE", "true", TRUE))
}

get_yesno <- function(input, v) {
  safe_int01(input[[input_id(v)]])
}

get_ext_sev <- function(input, region) {
  val <- input[[paste0("ui__", region, "_severity")]]
  if (is.null(val)) "none" else as.character(val)
}

ext_any <- function(input, region) {
  as.integer(get_ext_sev(input, region) != "none")
}

ext_severe <- function(input, region) {
  as.integer(get_ext_sev(input, region) == "severe")
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

# Define input types and safe defaults.
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

# ------------------------------------------------------------
# Derived values
# ------------------------------------------------------------

derive_age_group <- function(age, choices) {
  age <- safe_numeric(age, NA)
  if (!is.finite(age)) return(choices[1])
  if (age < 40) return(choices[grepl("18|39|young", choices, ignore.case = TRUE)][1])
  if (age < 65) return(choices[grepl("40|64|adult", choices, ignore.case = TRUE)][1])
  if (age < 75) return(choices[grepl("65|74", choices, ignore.case = TRUE)][1])
  hit <- choices[grepl("75|80|elder|older", choices, ignore.case = TRUE)][1]
  ifelse(is.na(hit), choices[length(choices)], hit)
}

derive_gcs_severity <- function(gcs, choices) {
  gcs <- safe_numeric(gcs, NA)
  if (!is.finite(gcs)) return(choices[1])
  if (gcs >= 13) return(choices[grepl("mild|13", choices, ignore.case = TRUE)][1])
  if (gcs >= 9) return(choices[grepl("moderate|9", choices, ignore.case = TRUE)][1])
  hit <- choices[grepl("severe|3", choices, ignore.case = TRUE)][1]
  ifelse(is.na(hit), choices[1], hit)
}

derived_value <- function(v, row, input) {
  choices <- strsplit(as.character(row$choices), "\\|\\|")[[1]]
  if (length(choices) == 0 || all(is.na(choices))) choices <- c("0", "1")
  
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
    if (nrow(row) == 0) next
    
    val <- input[[input_id(v)]]
    
    if (row$input_type == "derived") {
      val <- derived_value(v, row, input)
    }
    
    if (row$model_type %in% c("numeric", "numeric_binary")) {
      out[[v]] <- safe_numeric(val, safe_numeric(row$default, 0))
    } else if (row$input_type == "yesno") {
      choices <- strsplit(as.character(row$choices), "\\|\\|")[[1]]
      if (length(choices) == 0 || all(is.na(choices))) choices <- c("0", "1")
      val <- as.character(ifelse(val %in% c("1", "Yes", "yes", TRUE), "1", "0"))
      out[[v]] <- factor(val, levels = choices)
    } else if (row$input_type %in% c("gcs_select", "ais_severity")) {
      out[[v]] <- safe_numeric(val, safe_numeric(row$default, 0))
    } else {
      choices <- strsplit(as.character(row$choices), "\\|\\|")[[1]]
      if (length(choices) == 0 || all(is.na(choices))) choices <- as.character(row$default)
      if (is.null(val) || !as.character(val) %in% choices) val <- row$default
      out[[v]] <- factor(as.character(val), levels = choices)
    }
  }
  
  out
}

align_matrix <- function(newdata, predictors, feature_names) {
  f <- as.formula(paste("~", paste(predictors, collapse = " + "), "- 1"))
  mm <- Matrix::sparse.model.matrix(f, data = newdata)
  
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
  
  newdata <- make_one_row(input, predictors)
  mm <- align_matrix(newdata, predictors, feature_names)
  as.numeric(predict(model, xgb.DMatrix(mm))[1])
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
  if (row$input_type == "derived") return(NULL)
  
  id <- input_id(row$variable)
  
  if (row$input_type == "numeric") {
    numericInput(
      inputId = id,
      label = row$label,
      value = safe_numeric(row$default, 0),
      min = safe_numeric(row$min, NA),
      max = safe_numeric(row$max, NA),
      step = 1
    )
  } else if (row$input_type == "gcs_select") {
    choices <- switch(
      row$variable,
      gcs_eye_clean = gcs_eye_choices,
      gcs_motor_clean = gcs_motor_choices,
      gcs_verbal_clean = gcs_verbal_choices,
      gcs_total_aug = gcs_total_choices,
      gcs_total_choices
    )
    selectInput(id, row$label, choices = choices, selected = as.character(row$default), selectize = FALSE)
  } else if (row$input_type == "ais_severity") {
    selectInput(id, row$label, choices = ais_choices, selected = as.character(row$default), selectize = FALSE)
  } else if (row$input_type == "yesno") {
    selectInput(id, row$label, choices = c("No" = "0", "Yes" = "1"), selected = "0", selectize = FALSE)
  } else if (row$input_type == "count_select") {
    choices <- strsplit(as.character(row$choices), "\\|\\|")[[1]]
    selectInput(id, row$label, choices = choices, selected = row$default, selectize = FALSE)
  } else {
    choices <- strsplit(as.character(row$choices), "\\|\\|")[[1]]
    selectInput(id, row$label, choices = choices, selected = row$default, selectize = FALSE)
  }
}

input_group_ui <- function(group_name) {
  rows <- metadata[group == group_name & input_type != "derived" & variable %in% visible_vars][order(app_order)]
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
        .section-title { font-weight: 800; margin-bottom: 14px; }
        .plot-title { font-weight: 800; color: #243447; margin: 12px 0 10px 0; font-size: 1.15rem; }
        .subsection-note { color: #526579; font-weight: 650; margin-top: -4px; margin-bottom: 16px; }
        .form-label { font-weight: 650; }
        .form-control, .form-select { border-radius: 14px !important; min-height: 42px; }
        .btn-primary { border-radius: 14px !important; font-weight: 750; min-height: 46px; margin-top: 6px; }
        .slim-disp { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 10px; margin-bottom: 14px; }
        .disp-cell { background: #edf4fb; border-radius: 16px; padding: 10px 12px; border: 1px solid #dbe8f5; }
        .disp-label { font-size: 0.82rem; color: #526579; font-weight: 700; }
        .disp-value { font-size: 1.35rem; color: #1f4e79; font-weight: 850; }
        .block-gap { height: 18px; }
        .block-gap-small { height: 8px; }
        .note { font-size: 0.9rem; color: #6b7a8c; margin-top: 10px; }
        .detail-card h3 { font-size: 1.08rem; font-weight: 750; color: #243447; margin-top: 0; margin-bottom: 0.8rem; }
        .detail-card ul { margin-bottom: 0; padding-left: 1.15rem; }
        .detail-card li { color: #425466; margin-bottom: 0.48rem; line-height: 1.5; }
        .detail-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 26px 38px; }
        .detail-section { min-width: 0; }
        @media (max-width:1199px) { .sticky-panel { position: static; } }
        @media (max-width:767px) { .app-container { padding: 18px 14px 28px 14px; } .header-grid, .slim-disp, .detail-grid { grid-template-columns: 1fr; } }
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
            h2("Admission characteristics", class = "section-title"),
            accordion(
              id = "input_accordion",
              open = c("Demographics", "Neurologic status"),
              input_group_ui("Demographics"),
              input_group_ui("Transfer and mechanism"),
              input_group_ui("Neurologic status"),
              input_group_ui("Vital signs and respiratory support"),
              input_group_ui("Injury burden"),
              input_group_ui("Comorbidities"),
              input_group_ui("Intracranial injury pattern"),
              extracranial_profile_ui()
            ),
            actionButton("calc", "Calculate risk", class = "btn-primary w-100")
          )
        )
      ),
      
      div(
        card(
          card_body(
            h2("Predicted inpatient outcomes", class = "section-title"),
            plotOutput("risk_plot", height = "600px"),
            div(class = "note", "Disposition probabilities are mutually exclusive and sum to 100%. Other outcomes are independent binary predictions and should be interpreted separately.")
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
            class = "detail-section",
            h3("Model cohort and intended use"),
            tags$ul(
              tags$li(paste0("Model cohort: n = ", format(model_n, big.mark = ","), " patients.")),
              tags$li(paste0("Study years: ", paste(model_years, collapse = ", "), ".")),
              tags$li("Intended use: this calculator estimates inpatient resource-utilization risk among adults with traumatic brain injury using admission-era clinical and injury characteristics."),
              tags$li("The calculator is intended to support triage, early disposition planning, and resource-allocation discussions. It does not replace individualized clinical judgment.")
            )
          ),
          
          div(
            class = "detail-section",
            h3("Cohort and predictors"),
            tags$ul(
              tags$li("Data source: American College of Surgeons Trauma Quality Improvement Program (ACS-TQIP)."),
              tags$li("Study population: adult trauma patients with traumatic brain injury, defined using ICD-10 traumatic intracranial injury diagnosis codes (S06*)."),
              tags$li("Predictors include demographics, transfer/mechanism, Glasgow Coma Scale components, pupillary response, vital signs, respiratory assistance, ISS, AIS-derived injury severity, selected intracranial injury patterns, major extracranial injury patterns, and comorbidities."),
              tags$li("Several registry-style predictors are derived in the background from clinician-facing inputs to keep the calculator usable at the bedside.")
            )
          ),
          
          div(
            class = "detail-section",
            h3("Outcomes"),
            tags$ul(
              tags$li("Discharge disposition probabilities: home/home health, post-acute facility, and death/hospice."),
              tags$li("Acute utilization outcomes: ICU admission and mechanical ventilation."),
              tags$li("Prolonged utilization outcomes: hospital length of stay ≥20 days, ICU length of stay ≥8 days among ICU patients, and ventilator duration ≥8 days among ventilated patients."),
              tags$li("Conditional outcomes should be interpreted within the relevant population, such as ICU LOS among patients admitted to the ICU.")
            )
          ),
          
          div(
            class = "detail-section",
            h3("Performance, interpretation, and limitations"),
            tags$ul(
              tags$li("***"),
              tags$li("***"),
              tags$li("***"),
              tags$li("Displayed probabilities are point estimates. Uncertainty intervals are not shown unless a bootstrap, ensemble, or other formal uncertainty-estimation procedure is implemented and validated.")
            )
          )
        )
      )
    )
  )
)

server <- function(input, output, session) {
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
      if (is.finite(val)) {
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
      disp <- predict_multiclass(bundle$discharge, input, predictors_full)
      icu <- predict_binary(bundle$icu_admission, input, predictors_full)
      vent <- predict_binary(bundle$mechanical_ventilation, input, predictors_no_resp)
      hlos <- predict_binary(bundle$hlos_ge20, input, predictors_full)
      icu_los <- predict_binary(bundle$icu_los_ge8, input, predictors_full)
      vent_days <- predict_binary(bundle$vent_days_ge8, input, predictors_no_resp)
      
      list(
        discharge = disp,
        binary = data.table(
          outcome = c(
            "ICU admission",
            "Mechanical ventilation",
            "Hospital LOS ≥20 days",
            "ICU LOS ≥8 days, if ICU admitted",
            "Ventilator duration ≥8 days, if ventilated"
          ),
          probability = c(icu, vent, hlos, icu_los, vent_days)
        )
      )
    },
    ignoreInit = FALSE
  )
  
  output$disposition_slim <- renderUI({
    req(results())
    d <- copy(results()$discharge)
    d[, probability_label := sprintf("%.1f%%", 100 * probability)]
    
    preferred <- c("Home/home health", "Post-acute facility", "Death/hospice")
    d[, ord := match(class, preferred)]
    d[is.na(ord), ord := 99L]
    d <- d[order(ord, -probability)]
    
    tags$div(
      class = "slim-disp",
      lapply(seq_len(nrow(d)), function(i) {
        tags$div(
          class = "disp-cell",
          tags$div(class = "disp-label", d$class[i]),
          tags$div(class = "disp-value", d$probability_label[i])
        )
      })
    )
  })
  
  output$risk_plot <- renderPlot({
    req(results())
    
    disp <- copy(results()$discharge)
    preferred <- c("Home/home health", "Post-acute facility", "Death/hospice")
    disp[, ord := match(class, preferred)]
    disp[is.na(ord), ord := 99L]
    disp <- disp[order(ord)]
    
    disp_plot <- data.table(
      outcome = disp$class,
      probability = disp$probability,
      group = "Discharge disposition",
      section_order = 1L,
      within_order = seq_len(nrow(disp))
    )
    
    bin <- copy(results()$binary)
    bin[, outcome := c(
      "ICU admission",
      "Mechanical ventilation",
      "Hospital LOS ≥20 days",
      "ICU LOS ≥8 days, if ICU admitted",
      "Ventilator duration ≥8 days, if ventilated"
    )]
    bin[, group := data.table::fifelse(
      outcome %in% c("ICU admission", "Mechanical ventilation"),
      "Acute utilization",
      "Prolonged utilization"
    )]
    bin[, section_order := data.table::fifelse(group == "Acute utilization", 2L, 3L)]
    bin[, within_order := seq_len(.N), by = group]
    
    d <- rbindlist(list(disp_plot, bin), fill = TRUE)
    d[, group := factor(
      group,
      levels = c("Discharge disposition", "Acute utilization", "Prolonged utilization")
    )]
    d[, label := sprintf("%.1f%%", 100 * probability)]
    d[, outcome_wrapped := stringr::str_wrap(outcome, width = 30)]
    
    # Preserve within-section order while allowing separate facet spacing.
    d[, outcome_id := paste0(section_order, "_", sprintf("%02d", within_order), "_", outcome_wrapped)]
    d[, outcome_id := factor(outcome_id, levels = rev(d$outcome_id))]
    label_lookup <- setNames(d$outcome_wrapped, d$outcome_id)
    
    d[, label_x := pmin(probability + 0.035, 0.98)]
    d[, hjust_val := data.table::fifelse(probability > 0.88, 1.05, -0.05)]
    
    ggplot(d, aes(x = probability, y = outcome_id, fill = group)) +
      geom_col(width = 0.62) +
      geom_text(
        aes(x = label_x, label = label, hjust = hjust_val),
        size = 4.2,
        fontface = "bold",
        color = "#243447"
      ) +
      facet_grid(
        rows = vars(group),
        scales = "free_y",
        space = "free_y",
        switch = "y"
      ) +
      scale_y_discrete(labels = label_lookup) +
      scale_x_continuous(
        labels = scales::percent_format(accuracy = 1),
        limits = c(0, 1.03),
        breaks = seq(0, 1, by = 0.25)
      ) +
      scale_fill_manual(
        values = c(
          "Discharge disposition" = "#1f4e79",
          "Acute utilization" = "#b45f06",
          "Prolonged utilization" = "#2f7d32"
        )
      ) +
      labs(x = "Model-estimated probability", y = NULL) +
      theme_minimal(base_size = 13) +
      theme(
        legend.position = "none",
        strip.placement = "outside",
        strip.text.y.left = element_text(
          angle = 0,
          color = "#243447",
          face = "bold",
          size = 11,
          hjust = 0
        ),
        strip.background = element_rect(fill = "#edf4fb", color = "#dbe8f5"),
        panel.spacing.y = unit(0.8, "lines"),
        panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(color = "#e7edf5", linewidth = 0.7),
        axis.title = element_text(color = "#2f4257", face = "bold", size = 14),
        axis.text = element_text(color = "#425466"),
        plot.background = element_rect(fill = "#ffffff", color = NA),
        panel.background = element_rect(fill = "#ffffff", color = NA),
        plot.margin = margin(8, 12, 8, 4)
      )
  }, res = 120)
}

shinyApp(ui, server)
