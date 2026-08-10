# TBI-TRACT: Trauma Resource and Acute Care Trajectory Calculator
# Clinician-facing Shiny application for TBI resource-utilization prediction.
# Model inputs are derived directly from the bundled predictor metadata.


suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(data.table)
  library(Matrix)
  library(xgboost)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

data_dir <- "data"
model_path <- file.path(data_dir, "model_bundle.rds")
metadata_path <- file.path(data_dir, "predictor_metadata.rds")

required_files <- c(model_path, metadata_path)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing required application file(s): ",
    paste(missing_files, collapse = ", "),
    call. = FALSE
  )
}

bundle <- readRDS(model_path)
meta_obj <- readRDS(metadata_path)

if (is.null(meta_obj$metadata)) {
  stop("predictor_metadata.rds does not contain a 'metadata' object.", call. = FALSE)
}
if (is.null(bundle$discharge)) {
  stop("model_bundle.rds does not contain the required discharge model.", call. = FALSE)
}

metadata <- as.data.table(meta_obj$metadata)
for (nm in c("variable", "label", "group", "input_type", "model_type", "choices", "default", "min", "max")) {
  if (!nm %in% names(metadata)) metadata[, (nm) := NA_character_]
}
if (anyDuplicated(metadata$variable)) {
  stop("Predictor metadata contains duplicate variable names.", call. = FALSE)
}
predictors_full <- meta_obj$predictors_full
predictors_no_resp <- meta_obj$predictors_no_resp %||% setdiff(predictors_full, "respiratoryassistance_clean")
model_n <- meta_obj$model_n %||% NA_integer_
model_years <- meta_obj$model_years %||% character(0)
discharge_levels <- meta_obj$discharge_levels %||% c("Home/home health", "Post-acute facility", "Death/hospice")

input_id <- function(v) paste0("var__", v)

safe_numeric <- function(x, default = 0) {
  if (is.null(x) || length(x) == 0L) return(default)
  y <- suppressWarnings(as.numeric(x[[1]]))
  if (!is.finite(y)) return(default)
  y
}

safe_int01 <- function(x) {
  if (is.null(x) || length(x) == 0L) return(0L)
  value <- tolower(trimws(as.character(x[[1]])))
  as.integer(value %in% c("1", "yes", "true"))
}

get_yesno <- function(input, v) {
  safe_int01(input[[input_id(v)]])
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

find_xgb <- function(obj, max_depth = 10L) {
  if (max_depth < 0L || is.null(obj)) return(NULL)
  if (inherits(obj, "xgb.Booster")) return(obj)

  if (is.list(obj)) {
    for (item in obj) {
      hit <- find_xgb(item, max_depth - 1L)
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
# Metadata and model configuration
# ------------------------------------------------------------

model_entries <- names(bundle)[vapply(
  bundle,
  function(x) is.list(x) && !is.null(x$predictors) && !is.null(x$feature_names),
  logical(1)
)]

bundle_predictors <- unique(unlist(
  lapply(bundle[model_entries], function(x) x$predictors),
  use.names = FALSE
))

missing_predictor_metadata <- setdiff(bundle_predictors, metadata$variable)
if (length(missing_predictor_metadata) > 0L) {
  stop(
    "Predictor metadata is missing: ",
    paste(missing_predictor_metadata, collapse = ", "),
    call. = FALSE
  )
}

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
  dx_open_wound_head = "Open wound of head",

  dx_facial_fracture = "Facial fracture",
  dx_spinal_cord_injury = "Spinal cord injury",
  dx_neck_vascular_injury = "Neck vascular injury",
  dx_thoracic_injury = "Thoracic injury",
  dx_abdominal_pelvic_injury = "Abdominal/pelvic injury",
  dx_upper_extremity_injury = "Upper extremity injury",
  dx_lower_extremity_injury = "Lower extremity injury"
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
  dx_open_wound_head = "Intracranial injury pattern",

  dx_facial_fracture = "Extracranial injury pattern",
  dx_spinal_cord_injury = "Extracranial injury pattern",
  dx_neck_vascular_injury = "Extracranial injury pattern",
  dx_thoracic_injury = "Extracranial injury pattern",
  dx_abdominal_pelvic_injury = "Extracranial injury pattern",
  dx_upper_extremity_injury = "Extracranial injury pattern",
  dx_lower_extremity_injury = "Extracranial injury pattern"
)

visible_vars <- c(
  "age", "sex_clean", "race_clean", "ethnicity_clean", "insurance_clean",
  "transfer_clean", "mechanism_clean", "helmet_clean",
  "gcs_eye_clean", "gcs_motor_clean", "gcs_verbal_clean", "pupil_clean",
  "sbp_clean", "pulse_clean", "rr_clean", "spo2_clean", "respiratoryassistance_clean",
  "bleeding_disorder", "diabetes", "copd", "hypertension", "current_smoker",
  "dx_concussion", "dx_cerebral_edema_traumatic", "dx_diffuse_axonal_injury",
  "dx_focal_contusion_or_iph", "dx_epidural_hematoma", "dx_subdural_hematoma",
  "dx_subarachnoid_hemorrhage", "dx_other_intracranial_injury",
  "dx_brain_compression_herniation", "dx_skull_fracture_any",
  "dx_vault_skull_fracture", "dx_base_skull_fracture", "dx_open_wound_head",
  "dx_facial_fracture", "dx_spinal_cord_injury", "dx_neck_vascular_injury",
  "dx_thoracic_injury", "dx_abdominal_pelvic_injury",
  "dx_upper_extremity_injury", "dx_lower_extremity_injury"
)

derived_vars <- c(
  "age_group_aug",
  "gcs_total_aug",
  "gcs_severity_aug",
  "hypotension_sbp90_aug",
  "hypoxia_spo2_90_aug",
  "tachycardia_120_aug",
  "abnormal_rr_aug",
  "n_preexisting_conditions",
  "dx_any_s06_intracranial",
  "dx_intracranial_hemorrhage_any",
  "dx_multiple_intracranial_patterns",
  "dx_polyregion_injury_count",
  "dx_polyregion_2plus",
  "dx_polyregion_3plus"
)

metadata[variable %in% names(label_map), label := unname(label_map[variable])]
metadata[variable %in% names(ui_group_map), group := unname(ui_group_map[variable])]
metadata[variable %in% derived_vars, input_type := "derived"]

yesno_vars <- c(
  "bleeding_disorder", "diabetes", "copd", "hypertension", "current_smoker",
  grep("^dx_", metadata$variable, value = TRUE)
)

metadata[variable %in% yesno_vars & input_type != "derived", `:=`(
  input_type = "yesno",
  default = "0",
  choices = "0||1"
)]

metadata[variable == "age", `:=`(
  input_type = "numeric", min = 18, max = 120, default = "65"
)]
metadata[variable == "sbp_clean", `:=`(
  input_type = "numeric", min = 0, max = 300, default = "120"
)]
metadata[variable == "pulse_clean", `:=`(
  input_type = "numeric", min = 0, max = 250, default = "80"
)]
metadata[variable == "rr_clean", `:=`(
  input_type = "numeric", min = 0, max = 80, default = "16"
)]
metadata[variable == "spo2_clean", `:=`(
  input_type = "numeric", min = 0, max = 100, default = "98"
)]

metadata[variable == "gcs_eye_clean", `:=`(
  input_type = "gcs_select", default = "4", choices = "1||2||3||4"
)]
metadata[variable == "gcs_motor_clean", `:=`(
  input_type = "gcs_select", default = "6", choices = "1||2||3||4||5||6"
)]
metadata[variable == "gcs_verbal_clean", `:=`(
  input_type = "gcs_select", default = "5", choices = "1||2||3||4||5"
)]

metadata[, app_order := match(variable, visible_vars)]
metadata[is.na(app_order), app_order := 9999L]
metadata[is.na(group) & variable %in% visible_vars, group := "Other inputs"]
metadata[is.na(label) | label == "", label := variable]

# Quick Mode uses the highest-gain predictors from the deployed model itself.
# This keeps the UI synchronized with the model bundle when models are updated.
quick_top_n <- c(
  discharge = 30L,
  icu_admission = 30L,
  mechanical_ventilation = 20L,
  hlos_ge20 = 30L,
  icu_los_ge8 = 30L,
  vent_days_ge8 = 20L,
  icp_monitor_evd_bolt = 30L,
  craniotomy_craniectomy = 30L
)

feature_to_predictor <- function(feature, predictors) {
  hits <- predictors[feature == predictors | startsWith(feature, predictors)]
  if (length(hits) == 0L) return(NA_character_)
  hits[which.max(nchar(hits))]
}

rank_model_predictors <- function(obj) {
  predictors <- obj$predictors
  if (is.null(predictors) || length(predictors) == 0L) return(character(0))

  importance <- obj$importance
  if (is.null(importance) || !all(c("Feature", "Gain") %in% names(importance))) {
    importance <- tryCatch(
      as.data.table(xgboost::xgb.importance(model = get_model(obj))),
      error = function(e) data.table()
    )
  } else {
    importance <- as.data.table(importance)
  }

  if (nrow(importance) == 0L || !all(c("Feature", "Gain") %in% names(importance))) {
    return(predictors)
  }

  importance[, predictor := vapply(
    Feature,
    feature_to_predictor,
    character(1),
    predictors = predictors
  )]

  ranked <- importance[
    !is.na(predictor),
    .(gain = sum(as.numeric(Gain), na.rm = TRUE)),
    by = predictor
  ][order(-gain)]

  unique(c(ranked$predictor, predictors))
}

quick_predictor_sets <- lapply(names(quick_top_n), function(endpoint) {
  obj <- bundle[[endpoint]]
  if (is.null(obj)) return(character(0))
  head(rank_model_predictors(obj), quick_top_n[[endpoint]])
})
names(quick_predictor_sets) <- names(quick_top_n)

source_var_map <- list(
  age_group_aug = "age",
  gcs_total_aug = c("gcs_eye_clean", "gcs_motor_clean", "gcs_verbal_clean"),
  gcs_severity_aug = c("gcs_eye_clean", "gcs_motor_clean", "gcs_verbal_clean"),
  hypotension_sbp90_aug = "sbp_clean",
  hypoxia_spo2_90_aug = "spo2_clean",
  tachycardia_120_aug = "pulse_clean",
  abnormal_rr_aug = "rr_clean",
  n_preexisting_conditions = c(
    "bleeding_disorder", "diabetes", "copd", "hypertension", "current_smoker"
  ),
  dx_intracranial_hemorrhage_any = c(
    "dx_focal_contusion_or_iph", "dx_epidural_hematoma",
    "dx_subdural_hematoma", "dx_subarachnoid_hemorrhage"
  ),
  dx_multiple_intracranial_patterns = c(
    "dx_concussion", "dx_cerebral_edema_traumatic", "dx_diffuse_axonal_injury",
    "dx_focal_contusion_or_iph", "dx_epidural_hematoma", "dx_subdural_hematoma",
    "dx_subarachnoid_hemorrhage", "dx_other_intracranial_injury",
    "dx_brain_compression_herniation"
  ),
  dx_polyregion_injury_count = c(
    "dx_facial_fracture", "dx_spinal_cord_injury", "dx_thoracic_injury",
    "dx_abdominal_pelvic_injury", "dx_upper_extremity_injury",
    "dx_lower_extremity_injury"
  ),
  dx_polyregion_2plus = c(
    "dx_facial_fracture", "dx_spinal_cord_injury", "dx_thoracic_injury",
    "dx_abdominal_pelvic_injury", "dx_upper_extremity_injury",
    "dx_lower_extremity_injury"
  ),
  dx_polyregion_3plus = c(
    "dx_facial_fracture", "dx_spinal_cord_injury", "dx_thoracic_injury",
    "dx_abdominal_pelvic_injury", "dx_upper_extremity_injury",
    "dx_lower_extremity_injury"
  )
)

source_vars_for_predictors <- function(predictors) {
  src <- character(0)

  for (v in predictors) {
    if (v %in% visible_vars) src <- c(src, v)
    if (v %in% names(source_var_map)) src <- c(src, source_var_map[[v]])
  }

  unique(src[src %in% visible_vars])
}

endpoint_visible_vars <- function(endpoint, mode = c("quick", "full")) {
  mode <- match.arg(mode)
  if (mode == "full" || endpoint == "all") return(visible_vars)

  predictors <- quick_predictor_sets[[endpoint]]
  if (is.null(predictors) || length(predictors) == 0L) return(visible_vars)

  vars <- unique(c("age", source_vars_for_predictors(predictors)))
  vars[order(match(vars, visible_vars))]
}

# ------------------------------------------------------------
# Derived values
# ------------------------------------------------------------

split_choices <- function(x, fallback = c("0", "1")) {
  out <- unlist(strsplit(as.character(x), "\\|\\|"))
  out <- out[!is.na(out) & nzchar(out)]
  if (length(out) == 0L) fallback else out
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

match_choice <- function(choices, label_pattern, numeric_pattern = NULL, fallback = choices[1]) {
  hit <- choices[grepl(label_pattern, choices, ignore.case = TRUE)]
  if (length(hit) > 0L) return(hit[1])

  if (!is.null(numeric_pattern)) {
    hit <- choices[grepl(numeric_pattern, choices, perl = TRUE)]
    if (length(hit) > 0L) return(hit[1])
  }

  fallback
}

derive_gcs_severity <- function(gcs, choices) {
  gcs <- safe_numeric(gcs, NA)
  if (!is.finite(gcs)) return(choices[1])

  if (gcs >= 13) {
    return(match_choice(choices, "mild", "(^|[^0-9])13([^0-9]|$)"))
  }
  if (gcs >= 9) {
    return(match_choice(choices, "moderate", "(^|[^0-9])9([^0-9]|$)"))
  }

  match_choice(choices, "severe", "(^|[^0-9])3([^0-9]|$)")
}

calculate_gcs_total <- function(input) {
  eye <- safe_numeric(input[[input_id("gcs_eye_clean")]], 4)
  motor <- safe_numeric(input[[input_id("gcs_motor_clean")]], 6)
  verbal <- safe_numeric(input[[input_id("gcs_verbal_clean")]], 5)

  eye <- min(max(eye, 1), 4)
  motor <- min(max(motor, 1), 6)
  verbal <- min(max(verbal, 1), 5)

  as.integer(eye + motor + verbal)
}

derived_value <- function(v, row, input) {
  choices <- split_choices(row$choices, c("0", "1"))

  if (v == "age_group_aug") {
    return(derive_age_group(input[[input_id("age")]], choices))
  }
  if (v == "gcs_total_aug") {
    return(calculate_gcs_total(input))
  }
  if (v == "gcs_severity_aug") {
    return(derive_gcs_severity(calculate_gcs_total(input), choices))
  }
  if (v == "hypotension_sbp90_aug") {
    return(as.integer(safe_numeric(input[[input_id("sbp_clean")]], 999) < 90))
  }
  if (v == "hypoxia_spo2_90_aug") {
    return(as.integer(safe_numeric(input[[input_id("spo2_clean")]], 999) <= 90))
  }
  if (v == "tachycardia_120_aug") {
    return(as.integer(safe_numeric(input[[input_id("pulse_clean")]], 0) >= 120))
  }
  if (v == "abnormal_rr_aug") {
    rr <- safe_numeric(input[[input_id("rr_clean")]], 16)
    return(as.integer(rr < 10 | rr > 29))
  }

  if (v == "dx_any_s06_intracranial") {
    return(1L)
  }

  if (v == "n_preexisting_conditions") {
    return(
      get_yesno(input, "bleeding_disorder") +
        get_yesno(input, "diabetes") +
        get_yesno(input, "copd") +
        get_yesno(input, "hypertension") +
        get_yesno(input, "current_smoker")
    )
  }

  intracranial_hemorrhage <- any(c(
    get_yesno(input, "dx_focal_contusion_or_iph"),
    get_yesno(input, "dx_epidural_hematoma"),
    get_yesno(input, "dx_subdural_hematoma"),
    get_yesno(input, "dx_subarachnoid_hemorrhage")
  ) == 1L)

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
  ))

  if (v == "dx_intracranial_hemorrhage_any") {
    return(as.integer(intracranial_hemorrhage))
  }
  if (v == "dx_multiple_intracranial_patterns") {
    return(as.integer(intracranial_pattern_count >= 2L))
  }

  extracranial_region_count <- sum(c(
    get_yesno(input, "dx_facial_fracture"),
    get_yesno(input, "dx_spinal_cord_injury"),
    get_yesno(input, "dx_thoracic_injury"),
    get_yesno(input, "dx_abdominal_pelvic_injury"),
    get_yesno(input, "dx_upper_extremity_injury"),
    get_yesno(input, "dx_lower_extremity_injury")
  ))

  polyregion_count <- 1L + extracranial_region_count

  if (v == "dx_polyregion_injury_count") {
    return(polyregion_count)
  }
  if (v == "dx_polyregion_2plus") {
    return(as.integer(polyregion_count >= 2L))
  }
  if (v == "dx_polyregion_3plus") {
    return(as.integer(polyregion_count >= 3L))
  }

  row$default
}

# ------------------------------------------------------------
# Prediction helpers
# ------------------------------------------------------------

make_one_row <- function(input, predictors) {
  out <- data.frame(row.names = 1)

  for (v in predictors) {
    row <- metadata[variable == v][1]

    if (nrow(row) == 0L) {
      stop("Missing metadata for predictor: ", v, call. = FALSE)
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
    } else if (input_type == "gcs_select") {
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
  if (length(missing_cols) > 0L) {
    z <- Matrix::Matrix(0, nrow = nrow(mm), ncol = length(missing_cols), sparse = TRUE)
    colnames(z) <- missing_cols
    mm <- cbind(mm, z)
  }

  extra_cols <- setdiff(colnames(mm), feature_names)
  if (length(extra_cols) > 0L) {
    mm <- mm[, setdiff(colnames(mm), extra_cols), drop = FALSE]
  }

  mm[, feature_names, drop = FALSE]
}

predict_binary <- function(obj, input, fallback_predictors) {
  model <- get_model(obj)
  predictors <- obj$predictors %||% fallback_predictors
  feature_names <- obj$feature_names

  if (is.null(feature_names)) stop("Binary model is missing feature_names.")

  newdata <- make_one_row(input, predictors)
  mm <- align_matrix(newdata, predictors, feature_names)
  raw <- as.numeric(predict(model, xgb.DMatrix(mm))[1])
  apply_binary_recalibration(raw, obj)
}

predict_multiclass <- function(obj, input, fallback_predictors) {
  model <- get_model(obj)
  predictors <- obj$predictors %||% fallback_predictors
  feature_names <- obj$feature_names
  class_levels <- obj$class_levels %||% discharge_levels

  if (is.null(feature_names)) {
    stop("Multiclass model is missing feature_names.")
  }

  newdata <- make_one_row(input, predictors)
  mm <- align_matrix(newdata, predictors, feature_names)
  raw <- as.numeric(predict(model, xgb.DMatrix(mm)))

  if (length(raw) != length(class_levels)) {
    stop("Multiclass prediction length does not match the configured class levels.")
  }

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
    "ICU LOS ≥8 days",
    "Ventilator duration ≥8 days"
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
    "Full-cohort endpoint",
    "Full-cohort endpoint"
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

choice_display_label <- function(variable, value) {
  value <- as.character(value)
  if (variable == "mechanism_clean" && value == "Transport/MVC") return("Transport-related injury")
  value
}

labelled_choices <- function(variable, choices) {
  choices <- choices[!is.na(choices) & choices != ""]
  stats::setNames(choices, vapply(choices, function(x) choice_display_label(variable, x), character(1)))
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
      gcs_verbal_clean = gcs_verbal_choices
    )
    if (is.null(choices) || length(choices) == 0L) {
      choices <- split_choices(row$choices, as.character(row$default))
    }
    selectInput(id, row$label, choices = choices, selected = as.character(row$default), selectize = FALSE)
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
  if (nrow(rows) == 0L) return(NULL)

  controls <- lapply(seq_len(nrow(rows)), function(i) {
    make_input_control(rows[i])
  })

  gcs_components <- c("gcs_eye_clean", "gcs_motor_clean", "gcs_verbal_clean")
  if (group_name == "Neurologic status" && all(gcs_components %in% allowed_vars)) {
    controls <- c(controls, list(uiOutput("gcs_total_display")))
  }

  do.call(accordion_panel, c(list(title = group_name), controls))
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
        .gcs-total-box { display: flex; align-items: center; justify-content: space-between; gap: 14px; padding: 12px 14px; margin: 2px 0 12px 0; border: 1px solid #dbe8f5; border-radius: 14px; background: #f8fbfe; }
        .gcs-total-label { color: #425466; font-weight: 700; }
        .gcs-total-value { color: #1f4e79; font-size: 1.35rem; font-weight: 900; line-height: 1; }
        .gcs-total-note { color: #6b7a8c; font-size: 0.78rem; margin-top: 4px; }
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
          h1("TBI-TRACT: Trauma Resource and Acute Care Trajectory Calculator", class = "header-title"),
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
              tags$li("Prolonged utilization outcomes: hospital length of stay ≥20 days, ICU length of stay ≥8 days, and ventilator duration ≥8 days. ICU and ventilator-duration endpoints are modeled in the full cohort, with patients without the corresponding resource use classified as not experiencing the prolonged-utilization endpoint.")
            )
          ),

          div(
            h3("Interpretation and limitations"),
            tags$ul(
              tags$li("Displayed binary probabilities are recalibrated when calibration parameters are available in the model bundle."),
              tags$li("ICU LOS ≥8 days and ventilator duration ≥8 days are full-cohort endpoints, consistent with model development; patients without ICU admission or mechanical ventilation are classified as not experiencing the corresponding prolonged-utilization outcome."),
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

order_discharge_results <- function(discharge) {
  preferred <- c("Home/home health", "Post-acute facility", "Death/hospice")
  discharge <- copy(discharge)
  discharge[, ord := match(class, preferred)]
  discharge[is.na(ord), ord := 99L]
  discharge[order(ord, -probability)]
}

make_discharge_cards <- function(discharge) {
  discharge <- order_discharge_results(discharge)
  lapply(seq_len(nrow(discharge)), function(i) {
    result_card(
      name = discharge$class[i],
      probability = discharge$probability[i],
      type_class = "result-disposition"
    )
  })
}

endpoint_subtext <- function(subtext, available, quick_mode = FALSE) {
  pieces <- character(0)
  if (!is.na(subtext) && nzchar(subtext)) pieces <- c(pieces, subtext)
  if (!isTRUE(available)) pieces <- c(pieces, "Model unavailable in model bundle")
  if (isTRUE(available) && quick_mode) pieces <- c(pieces, "Endpoint-specific Quick Mode")

  if (length(pieces) == 0L) NULL else paste(pieces, collapse = " · ")
}

make_binary_cards <- function(dat, quick_mode = FALSE) {
  if (nrow(dat) == 0L) return(list())

  lapply(seq_len(nrow(dat)), function(i) {
    result_card(
      name = dat$outcome[i],
      probability = dat$probability[i],
      type_class = dat$type_class[i],
      subtext = endpoint_subtext(dat$subtext[i], dat$available[i], quick_mode)
    )
  })
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

  output$gcs_total_display <- renderUI({
    total <- calculate_gcs_total(input)
    tags$div(
      class = "gcs-total-box",
      tags$div(
        tags$div(class = "gcs-total-label", "Calculated total GCS"),
        tags$div(class = "gcs-total-note", "Automatically calculated from eye + motor + verbal scores")
      ),
      tags$div(class = "gcs-total-value", total)
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

    panels <- list(
      input_group_ui("Demographics", allowed),
      input_group_ui("Transfer and mechanism", allowed),
      input_group_ui("Neurologic status", allowed),
      input_group_ui("Vital signs and respiratory support", allowed),
      input_group_ui("Comorbidities", allowed),
      input_group_ui("Intracranial injury pattern", allowed),
      input_group_ui("Extracranial injury pattern", allowed)
    )

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
      spo2_clean = c(0, 100)
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
        return(result_section(
          title = "Discharge disposition",
          note = "Quick Mode · Mutually exclusive",
          cards = make_discharge_cards(results()$discharge)
        ))
      }

      bin <- copy(results()$binary)
      if (nrow(bin) == 0L) return(NULL)

      return(result_section(
        title = bin$outcome[1],
        note = "Selected endpoint",
        cards = make_binary_cards(bin, quick_mode = TRUE),
        grid_class = "one"
      ))
    }

    disp_cards <- make_discharge_cards(results()$discharge)
    bin <- copy(results()$binary)

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
        cards = make_binary_cards(acute),
        grid_class = "two"
      ),
      result_section(
        title = "Neurosurgical resource utilization",
        note = "Independent binary estimates",
        cards = make_binary_cards(neuro),
        grid_class = "two"
      ),
      result_section(
        title = "Prolonged utilization",
        note = "Independent binary estimates",
        cards = make_binary_cards(prolonged)
      )
    )
  })
}

shinyApp(ui, server)
