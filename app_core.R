# TBI-TRACT: Trauma Resource and Acute Care Trajectory Calculator
# Locked 2020-2024 deployment build.

suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(data.table)
  library(xgboost)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

data_dir <- "data"
model_dir <- file.path(data_dir, "models")
encoder_dir <- file.path(data_dir, "encoders")

model_ids <- c(
  "discharge_3cat_final",
  "hlos_trajectory_final",
  "hospital_los",
  "icu_trajectory_final",
  "icu_los_conditional",
  "ventilation_trajectory_final",
  "ventilator_days_conditional",
  "icp_pressure_monitor_final",
  "craniotomy_craniectomy_final"
)

resolve_model_path <- function(id) {
  candidates <- c(
    file.path(model_dir, paste0(id, ".json")),
    paste0(id, ".json")
  )
  hit <- candidates[file.exists(candidates)]
  if (length(hit)) hit[[1L]] else NA_character_
}

model_paths <- setNames(
  vapply(model_ids, resolve_model_path, character(1)),
  model_ids
)
encoder_paths <- setNames(
  file.path(encoder_dir, paste0(model_ids, "_encoder.rds")),
  model_ids
)

missing_models <- names(model_paths)[is.na(model_paths)]
missing_encoders <- names(encoder_paths)[!file.exists(encoder_paths)]

if (length(missing_models) || length(missing_encoders)) {
  stop(
    paste0(
      "Missing TBI-TRACT deployment files.\n",
      if (length(missing_models)) {
        paste0(
          "Model JSON(s): ", paste(missing_models, collapse = ", "),
          " (checked data/models/ and repo root)\n"
        )
      } else "",
      if (length(missing_encoders)) {
        paste0(
          "Encoder RDS(s): ", paste(missing_encoders, collapse = ", "),
          " (expected in data/encoders/)\n"
        )
      } else ""
    ),
    call. = FALSE
  )
}

load_entry <- function(id) {
  meta <- readRDS(encoder_paths[[id]])
  needed <- c(
    "endpoint_id", "endpoint_label", "type",
    "predictors", "encoder", "feature_names"
  )
  missing_meta <- setdiff(needed, names(meta))
  if (length(missing_meta)) {
    stop(
      id, " encoder metadata missing: ",
      paste(missing_meta, collapse = ", "),
      call. = FALSE
    )
  }

  list(
    model = xgboost::xgb.load(model_paths[[id]]),
    meta = meta
  )
}

models <- setNames(lapply(model_ids, load_entry), model_ids)

# Locked-spec integrity checks.
removed_predictors <- c(
  "helmet_use_recovered",
  "respiratoryassistance_clean"
)
social_predictors <- c(
  "race_clean",
  "ethnicity_clean",
  "insurance_clean"
)

all_used <- unique(unlist(
  lapply(models, function(x) as.character(x$meta$predictors)),
  use.names = FALSE
))

if (length(intersect(removed_predictors, all_used))) {
  stop("Removed predictors remain in deployment metadata.", call. = FALSE)
}

for (id in setdiff(
  model_ids,
  c("discharge_3cat_final", "hlos_trajectory_final")
)) {
  if (length(intersect(social_predictors, models[[id]]$meta$predictors))) {
    stop(id, " unexpectedly contains race/ethnicity/payer.", call. = FALSE)
  }
}

for (id in c("discharge_3cat_final", "hlos_trajectory_final")) {
  if (length(setdiff(social_predictors, models[[id]]$meta$predictors))) {
    stop(
      id,
      " is missing final race/ethnicity/payer context predictors.",
      call. = FALSE
    )
  }
}

hlos_rounds <- suppressWarnings(
  as.integer(models$hospital_los$meta$selected_rounds %||% NA_integer_)
)
if (!length(hlos_rounds) ||
    !is.finite(hlos_rounds[1]) ||
    hlos_rounds[1] != 4341L) {
  warning(
    "Hospital LOS metadata does not report the final 4,341-round extended-CV fit."
  )
}

all_predictors <- unique(
  all_used[!is.na(all_used) & nzchar(all_used)]
)
numeric_predictors <- unique(unlist(
  lapply(models, function(x) x$meta$encoder$numeric_vars %||% character()),
  use.names = FALSE
))
categorical_predictors <- unique(unlist(
  lapply(models, function(x) x$meta$encoder$categorical_vars %||% character()),
  use.names = FALSE
))
binary_predictors <- unique(c(
  "gcsq_intubated_recovered",
  "gcsq_sedated_paralyzed_recovered",
  "gcsq_eye_obstruction_recovered",
  "gcsq_unknown_recovered",
  grep("^dx_", all_predictors, value = TRUE),
  grep("^pmhx_", all_predictors, value = TRUE)
))
continuous_predictors <- setdiff(numeric_predictors, binary_predictors)

labels <- c(
  age = "Age, years",
  sex_clean = "Sex",
  race_clean = "Race",
  ethnicity_clean = "Ethnicity",
  insurance_clean = "Payer / insurance",
  mechanism_clean = "Mechanism of injury",
  gcs_eye_clean = "GCS eye",
  gcs_verbal_clean = "GCS verbal",
  gcs_motor_clean = "GCS motor",
  gcsq_intubated_recovered = "Intubated at GCS assessment",
  gcsq_sedated_paralyzed_recovered = "Sedated / paralyzed at GCS assessment",
  gcsq_eye_obstruction_recovered = "Eye score limited by obstruction",
  gcsq_unknown_recovered = "GCS qualifier unknown / unassessable",
  pupil_clean = "Pupillary response",
  sbp_clean = "Systolic blood pressure, mmHg",
  pulse_clean = "Heart rate, beats/min",
  rr_clean = "Respiratory rate, breaths/min",
  spo2_clean = "Oxygen saturation, %",
  temperature_c_recovered = "Temperature, °C",
  supplemental_oxygen_recovered = "Supplemental oxygen at initial assessment",
  dx_concussion = "Concussion",
  dx_cerebral_edema_traumatic = "Traumatic cerebral edema",
  dx_diffuse_axonal_injury = "Diffuse axonal injury",
  dx_focal_contusion_or_iph = "Contusion / intraparenchymal hemorrhage",
  dx_epidural_hematoma = "Epidural hematoma",
  dx_subdural_hematoma = "Subdural hematoma",
  dx_subarachnoid_hemorrhage = "Traumatic subarachnoid hemorrhage",
  dx_other_intracranial_injury = "Other intracranial injury",
  dx_cranial_skull_fracture = "Cranial skull fracture",
  dx_facial_fracture = "Facial fracture",
  dx_other_skull_or_facial_fracture = "Other skull / facial fracture",
  dx_spinal_cord_injury = "Spinal cord injury",
  dx_neck_vascular_injury = "Neck vascular injury",
  dx_thoracic_injury = "Thoracic injury",
  dx_abdominal_pelvic_injury = "Abdominal / pelvic injury",
  dx_upper_extremity_injury = "Upper-extremity injury",
  dx_lower_extremity_injury = "Lower-extremity injury",
  pmhx_bleeding_disorder = "Bleeding disorder / coagulopathy",
  pmhx_anticoagulant_therapy = "Anticoagulant therapy",
  pmhx_copd = "COPD",
  pmhx_diabetes = "Diabetes",
  pmhx_hypertension = "Hypertension",
  pmhx_current_smoker = "Current smoker",
  pmhx_functional_dependence = "Functional dependence",
  pmhx_dementia = "Dementia",
  pmhx_chf = "Congestive heart failure",
  pmhx_chronic_renal_failure = "Chronic renal failure",
  pmhx_cirrhosis = "Cirrhosis",
  pmhx_steroid_use = "Chronic steroid use"
)

label_for <- function(v) {
  if (v %in% names(labels)) unname(labels[v]) else gsub("_", " ", v)
}

groups <- list(
  "Demographics and context" = c(
    "age", "sex_clean", "race_clean", "ethnicity_clean", "insurance_clean"
  ),
  "Mechanism and neurologic status" = c(
    "mechanism_clean", "gcs_eye_clean", "gcs_verbal_clean", "gcs_motor_clean",
    "gcsq_intubated_recovered", "gcsq_sedated_paralyzed_recovered",
    "gcsq_eye_obstruction_recovered", "gcsq_unknown_recovered", "pupil_clean"
  ),
  "Admission physiology" = c(
    "sbp_clean", "pulse_clean", "rr_clean", "spo2_clean",
    "temperature_c_recovered", "supplemental_oxygen_recovered"
  ),
  "Cranial injury pattern" = c(
    "dx_concussion", "dx_cerebral_edema_traumatic", "dx_diffuse_axonal_injury",
    "dx_focal_contusion_or_iph", "dx_epidural_hematoma", "dx_subdural_hematoma",
    "dx_subarachnoid_hemorrhage", "dx_other_intracranial_injury",
    "dx_cranial_skull_fracture", "dx_facial_fracture",
    "dx_other_skull_or_facial_fracture"
  ),
  "Extracranial injury pattern" = c(
    "dx_spinal_cord_injury", "dx_neck_vascular_injury", "dx_thoracic_injury",
    "dx_abdominal_pelvic_injury", "dx_upper_extremity_injury",
    "dx_lower_extremity_injury"
  ),
  "Pre-existing conditions" = c(
    "pmhx_bleeding_disorder", "pmhx_anticoagulant_therapy", "pmhx_copd",
    "pmhx_diabetes", "pmhx_hypertension", "pmhx_current_smoker",
    "pmhx_functional_dependence", "pmhx_dementia", "pmhx_chf",
    "pmhx_chronic_renal_failure", "pmhx_cirrhosis", "pmhx_steroid_use"
  )
)
groups <- lapply(groups, intersect, y = all_predictors)

get_levels <- function(v) {
  for (x in models) {
    z <- x$meta$encoder$categorical_levels[[v]]
    if (!is.null(z) && length(z)) return(as.character(z))
  }
  character()
}

pretty_level <- function(x) {
  z <- trimws(as.character(x))
  zl <- tolower(z)

  if (zl %in% c(
    "__unknown__", "unknown", "unknown / not recorded",
    "unknown/not recorded", "not recorded", "not known", "unk"
  )) {
    return("Unknown / not recorded")
  }

  if (zl %in% c(
    "__other__", "other", "other / unlisted",
    "other/unlisted", "other / not listed", "other/not listed"
  )) {
    return("Other / not listed")
  }

  if (z == "Transport/MVC") return("Motor vehicle / transport-related")
  z
}

clean_choices <- function(levels) {
  lev <- unique(as.character(levels))
  if (!length(lev)) lev <- "__UNKNOWN__"

  lab <- vapply(lev, pretty_level, character(1))
  priority <- rep(10L, length(lev))
  priority[lev == "__UNKNOWN__"] <- 1L
  priority[tolower(lev) == "unknown"] <- 2L
  priority[tolower(lev) == "other"] <- 1L
  priority[lev == "__OTHER__"] <- 2L

  ord <- order(lab, priority, seq_along(lev))
  lev <- lev[ord]
  lab <- lab[ord]
  keep <- !duplicated(lab)

  setNames(lev[keep], lab[keep])
}

input_id <- function(v) paste0("var__", v)

numeric_limits <- list(
  age = list(min = 18, max = 89, step = 1, note = "Study population: age 18–89 years"),
  sbp_clean = list(min = 0, max = 300, step = 1, note = "Allowed range: 0–300 mmHg"),
  pulse_clean = list(min = 0, max = 300, step = 1, note = "Allowed range: 0–300 beats/min"),
  rr_clean = list(min = 0, max = 100, step = 1, note = "Allowed range: 0–100 breaths/min"),
  spo2_clean = list(min = 0, max = 100, step = 1, note = "Allowed range: 0–100%"),
  temperature_c_recovered = list(min = 25, max = 45, step = 0.1, note = "Allowed range: 25–45 °C")
)

gcs_choices <- list(
  gcs_eye_clean = c(
    "Unknown" = "", "1 - None" = "1", "2 - To pain" = "2",
    "3 - To speech" = "3", "4 - Spontaneous" = "4"
  ),
  gcs_verbal_clean = c(
    "Unknown" = "", "1 - None" = "1", "2 - Incomprehensible sounds" = "2",
    "3 - Inappropriate words" = "3", "4 - Confused" = "4", "5 - Oriented" = "5"
  ),
  gcs_motor_clean = c(
    "Unknown" = "", "1 - None" = "1", "2 - Extension" = "2",
    "3 - Flexion" = "3", "4 - Withdraws" = "4", "5 - Localizes" = "5",
    "6 - Obeys commands" = "6"
  )
)

make_control <- function(v) {
  id <- input_id(v)
  lab <- label_for(v)

  if (v %in% names(gcs_choices)) {
    return(selectInput(
      id, lab, gcs_choices[[v]], selected = "", selectize = FALSE
    ))
  }

  if (v %in% binary_predictors) {
    return(selectInput(
      id,
      lab,
      c("Unknown / not recorded" = "", "No" = "0", "Yes" = "1"),
      selected = "",
      selectize = FALSE
    ))
  }

  if (v %in% continuous_predictors) {
    lim <- numeric_limits[[v]]
    if (!is.null(lim)) {
      return(tagList(
        numberInput(
          id, lab, value = NA,
          min = lim$min, max = lim$max, step = lim$step
        ),
        div(class = "input-hint", lim$note)
      ))
    }
    return(numberInput(id, lab, value = NA))
  }

  if (v %in% categorical_predictors) {
    lev <- get_levels(v)
    if (!length(lev)) lev <- "__UNKNOWN__"

    choices <- clean_choices(lev)
    vals <- unname(choices)
    selected <- if ("__UNKNOWN__" %in% vals) {
      "__UNKNOWN__"
    } else if (any(tolower(vals) == "unknown")) {
      vals[which(tolower(vals) == "unknown")[1L]]
    } else {
      vals[1L]
    }

    return(selectInput(
      id, lab, choices, selected = selected, selectize = FALSE
    ))
  }

  NULL
}

validate_numeric_inputs <- function(input, allowed) {
  errors <- character()

  for (v in intersect(names(numeric_limits), allowed)) {
    x <- input[[input_id(v)]]
    if (is.null(x) || !length(x) || is.na(x) || identical(x, "")) next

    z <- suppressWarnings(as.numeric(x[[1L]]))
    lim <- numeric_limits[[v]]

    if (!is.finite(z) || z < lim$min || z > lim$max) {
      errors <- c(
        errors,
        paste0(
          label_for(v), " must be between ",
          lim$min, " and ", lim$max, "."
        )
      )
    }
  }

  errors
}

quick_common <- c(
  "age", "sex_clean", "gcs_eye_clean", "gcs_verbal_clean", "gcs_motor_clean",
  "gcsq_intubated_recovered", "gcsq_sedated_paralyzed_recovered", "pupil_clean",
  "sbp_clean", "rr_clean", "spo2_clean", "supplemental_oxygen_recovered",
  "mechanism_clean", "dx_cerebral_edema_traumatic", "dx_diffuse_axonal_injury",
  "dx_focal_contusion_or_iph", "dx_epidural_hematoma", "dx_subdural_hematoma",
  "dx_subarachnoid_hemorrhage", "dx_cranial_skull_fracture", "dx_thoracic_injury",
  "dx_abdominal_pelvic_injury", "pmhx_anticoagulant_therapy",
  "pmhx_functional_dependence", "pmhx_dementia"
)

quick_sets <- list(
  disposition = unique(c(
    quick_common, "race_clean", "ethnicity_clean", "insurance_clean",
    "dx_upper_extremity_injury", "dx_lower_extremity_injury",
    "pmhx_chf", "pmhx_chronic_renal_failure"
  )),
  hlos = unique(c(
    quick_common, "race_clean", "ethnicity_clean", "insurance_clean",
    "pulse_clean", "temperature_c_recovered", "dx_spinal_cord_injury",
    "dx_neck_vascular_injury", "dx_upper_extremity_injury",
    "dx_lower_extremity_injury", "pmhx_chf", "pmhx_chronic_renal_failure"
  )),
  icu = unique(c(
    quick_common, "pulse_clean", "temperature_c_recovered",
    "dx_spinal_cord_injury", "dx_neck_vascular_injury"
  )),
  ventilation = unique(c(
    quick_common, "pulse_clean", "temperature_c_recovered", "dx_spinal_cord_injury"
  )),
  icp = unique(c(
    quick_common, "dx_other_intracranial_injury",
    "dx_other_skull_or_facial_fracture"
  )),
  craniotomy = unique(c(
    quick_common, "dx_other_intracranial_injury",
    "dx_other_skull_or_facial_fracture"
  ))
)
quick_sets <- lapply(quick_sets, intersect, y = all_predictors)

quick_choices <- c(
  "Discharge disposition" = "disposition",
  "Hospital length of stay" = "hlos",
  "ICU trajectory" = "icu",
  "Mechanical ventilation trajectory" = "ventilation",
  "EVD / intraparenchymal ICP bolt" = "icp",
  "Craniotomy / craniectomy" = "craniotomy"
)

collect_values <- function(input, allowed) {
  vals <- setNames(vector("list", length(all_predictors)), all_predictors)

  for (v in all_predictors) {
    if (!v %in% allowed) {
      vals[[v]] <- if (v %in% categorical_predictors) {
        "__UNKNOWN__"
      } else {
        NA_real_
      }
      next
    }

    x <- input[[input_id(v)]]

    if (v %in% numeric_predictors) {
      z <- suppressWarnings(as.numeric(if (length(x)) x[[1L]] else NA))
      vals[[v]] <- if (is.finite(z)) z else NA_real_
    } else {
      z <- if (length(x)) as.character(x[[1L]]) else ""
      vals[[v]] <- if (is.na(z) || !nzchar(z)) "__UNKNOWN__" else z
    }
  }

  vals
}

encode_values <- function(entry, vals) {
  encoder <- entry$meta$encoder
  feature_names <- as.character(entry$meta$feature_names)

  X <- matrix(
    0,
    nrow = 1,
    ncol = length(feature_names),
    dimnames = list(NULL, feature_names)
  )

  for (v in encoder$numeric_vars) {
    if (v %in% feature_names) {
      X[1, v] <- suppressWarnings(as.numeric(vals[[v]]))
    }
  }

  for (v in encoder$categorical_vars) {
    lev <- as.character(encoder$categorical_levels[[v]])
    val <- as.character(vals[[v]] %||% "__UNKNOWN__")

    if (is.na(val) || !nzchar(val)) val <- "__UNKNOWN__"
    if (!val %in% lev) {
      val <- if ("__OTHER__" %in% lev) {
        "__OTHER__"
      } else if ("__UNKNOWN__" %in% lev) {
        "__UNKNOWN__"
      } else {
        lev[1L]
      }
    }

    encoded_names <- paste0(v, "__", make.names(lev, unique = TRUE))
    keep <- intersect(encoded_names, feature_names)
    X[1, keep] <- 0

    idx <- match(val, lev)
    if (!is.na(idx) && encoded_names[idx] %in% feature_names) {
      X[1, encoded_names[idx]] <- 1
    }
  }

  storage.mode(X) <- "double"
  X
}

raw_pred <- function(id, vals) {
  entry <- models[[id]]
  X <- encode_values(entry, vals)
  as.numeric(predict(
    entry$model,
    xgboost::xgb.DMatrix(X, missing = NA_real_)
  ))
}

pred_multi <- function(id, vals) {
  levels <- as.character(models[[id]]$meta$levels)
  p <- raw_pred(id, vals)

  if (length(p) != length(levels)) {
    stop(id, " returned wrong number of classes", call. = FALSE)
  }

  p[!is.finite(p)] <- 0
  p <- pmax(p, 0)
  p <- if (sum(p) > 0) p / sum(p) else rep(1 / length(p), length(p))

  data.table(class = levels, probability = p)
}

pred_binary <- function(id, vals) {
  pmin(1, pmax(0, raw_pred(id, vals)[1L]))
}

pred_quantiles <- function(id, vals) {
  p <- raw_pred(id, vals)
  if (length(p) < 3L) {
    stop(id, " did not return Q10/Q50/Q90", call. = FALSE)
  }

  p <- expm1(p[1:3])
  lower_bound <- suppressWarnings(
    as.numeric(models[[id]]$meta$lower_bound %||% 0)
  )
  if (!is.finite(lower_bound)) lower_bound <- 0

  p <- sort(pmax(p, lower_bound))
  setNames(p, c("q10", "q50", "q90"))
}

fmt_prob <- function(p) {
  x <- 100 * as.numeric(p)
  if (!is.finite(x)) return("—")
  if (x > 0 && x < 0.1) return("<0.1%")
  if (x < 5 || x > 95) sprintf("%.1f%%", x) else sprintf("%.0f%%", x)
}

fmt_days <- function(x) {
  x <- as.numeric(x)
  if (!is.finite(x)) return("—")
  if (x < 10) sprintf("%.1f d", x) else sprintf("%.0f d", x)
}

pretty_class <- function(x) {
  map <- c(
    "Home/home health" = "Home / home health",
    "Post-acute facility" = "Post-acute facility",
    "Death/hospice" = "Death / hospice",
    "Hospital LOS <=7 days" = "≤7 days",
    "Hospital LOS 8-27 days" = "8–27 days",
    "Hospital LOS >=28 days" = "≥28 days",
    "No ICU" = "No ICU",
    "ICU 1-7 days" = "ICU 1–7 days",
    "ICU >=8 days" = "ICU ≥8 days",
    "No ventilation" = "No ventilation",
    "Ventilation 1-7 days" = "Ventilation 1–7 days",
    "Ventilation >=8 days" = "Ventilation ≥8 days"
  )
  if (x %in% names(map)) unname(map[x]) else x
}

probability_card <- function(name, probability, cls) {
  div(
    class = paste("result-card", cls),
    div(
      class = "result-top",
      span(class = "result-name", name),
      span(class = "result-pct", fmt_prob(probability))
    ),
    div(
      class = "bar",
      div(
        class = "fill",
        style = paste0("width:", 100 * probability, "%;")
      )
    )
  )
}

prob_cards <- function(d, cls) {
  div(
    class = "result-grid",
    lapply(seq_len(nrow(d)), function(i) {
      probability_card(
        pretty_class(d$class[i]),
        d$probability[i],
        cls
      )
    })
  )
}

duration_card <- function(title, q, cls, note = NULL) {
  div(
    class = paste("duration-card", cls),
    div(class = "duration-label", title),
    div(class = "duration-main", fmt_days(q["q50"])),
    div(
      class = "duration-range",
      paste0(
        "Estimated 10th–90th percentile range: ",
        fmt_days(q["q10"]), " – ", fmt_days(q["q90"])
      )
    ),
    if (!is.null(note)) div(class = "small-note", note)
  )
}

section_ui <- function(title, note, body) {
  div(
    class = "result-section",
    div(
      class = "result-heading",
      h3(title),
      span(note)
    ),
    body
  )
}

traj_duration <- function(d, q, cls, title, note = NULL) {
  tagList(
    prob_cards(d, cls),
    duration_card(title, q, cls, note)
  )
}

neuro_card <- function(name, p) {
  probability_card(name, p, "result-neuro")
}

neuro_section <- function(icp = NULL, craniotomy = NULL) {
  cards <- list()
  if (!is.null(icp)) {
    cards <- c(cards, list(
      neuro_card("EVD or intraparenchymal ICP bolt", icp)
    ))
  }
  if (!is.null(craniotomy)) {
    cards <- c(cards, list(
      neuro_card("Craniotomy / craniectomy", craniotomy)
    ))
  }

  section_ui(
    "Neurosurgical resource utilization",
    if (length(cards) > 1L) {
      "Independent probability estimates"
    } else {
      "Independent probability estimate"
    },
    div(class = "result-grid", cards)
  )
}

css <- "
body{background:#f4f7fb;color:#243447}.app{max-width:1450px;margin:auto;padding:22px}.hero,.card{background:white;border:1px solid #e3ebf3;border-radius:22px;box-shadow:0 8px 26px rgba(31,52,73,.06)}
.hero{padding:22px 26px;margin-bottom:20px}.hero-grid{display:grid;grid-template-columns:84px 1fr;gap:18px;align-items:center}.logo{width:78px}.title{font-weight:850;font-size:clamp(1.8rem,3vw,2.8rem);margin:0}.subtitle{color:#5b6d7f;margin:4px 0 0}
.sticky{position:sticky;top:18px}.scroll{max-height:calc(100vh - 355px);overflow-y:auto;padding-right:5px}.mode{background:#edf4fb;border-radius:15px;padding:12px;margin-bottom:12px}.btn-primary{border-radius:13px;font-weight:750;min-height:44px}.section-title{font-weight:800}.result-section{padding:4px 0 19px;border-bottom:1px solid #edf1f5;margin-bottom:17px}.result-heading{display:flex;justify-content:space-between;align-items:baseline;gap:12px}.result-heading h3{font-size:1.05rem;font-weight:800}.result-heading span{font-size:.82rem;color:#6b7a8c;font-weight:650}.result-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:11px}.result-card,.duration-card{border:1px solid #dbe8f5;border-radius:17px;padding:13px;background:#f9fbfd}.result-top{display:flex;justify-content:space-between;gap:10px}.result-name{font-weight:700;color:#425466}.result-pct{font-size:1.7rem;font-weight:900}.bar{height:11px;background:#e7edf5;border-radius:99px;overflow:hidden;margin-top:10px}.fill{height:100%;border-radius:99px}.result-disposition .fill{background:#1f4e79}.result-hospital .fill{background:#2f7d32}.result-icu .fill{background:#b45f06}.result-vent .fill{background:#187b80}.result-neuro .fill{background:#6f42c1}.duration-card{margin-top:11px;background:white}.duration-main{font-size:2rem;font-weight:900}.duration-label,.duration-range{font-weight:700;color:#425466}.small-note,.footnote,.quick-note,.input-hint{font-size:.84rem;color:#6b7a8c}.input-hint{margin-top:-10px;margin-bottom:10px}.notice{background:#fff8e6;border:1px solid #f1d99a;border-radius:14px;padding:11px 13px;margin-bottom:13px}.details{margin-top:18px}.detail-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px 28px}.detail-grid h3{font-size:1rem;font-weight:800}@media(max-width:1199px){.sticky{position:static}.scroll{max-height:none}}@media(max-width:900px){.result-grid,.detail-grid{grid-template-columns:1fr}.hero-grid{grid-template-columns:1fr}.app{padding:12px}}
"

ui <- page_fluid(
  theme = bs_theme(
    version = 5,
    bootswatch = "flatly",
    primary = "#1f4e79",
    bg = "#f4f7fb",
    fg = "#243447"
  ),
  tags$head(tags$style(HTML(css))),
  div(
    class = "app",
    div(
      class = "hero",
      div(
        class = "hero-grid",
        div(
          if (file.exists("www/ohsu_logo.png")) {
            img(src = "ohsu_logo.png", class = "logo")
          } else {
            strong("OHSU")
          }
        ),
        div(
          h1(
            "TBI-TRACT: Trauma Resource and Acute Care Trajectory Calculator",
            class = "title"
          ),
          p(
            paste(
              "Early inpatient trajectory estimates for adults with traumatic",
              "brain injury after initial trauma-center evaluation and diagnostic workup"
            ),
            class = "subtitle"
          )
        )
      )
    ),
    layout_columns(
      col_widths = c(4, 8),
      div(
        class = "sticky",
        card(
          card_body(
            h2("Calculator mode", class = "section-title"),
            div(
              class = "mode",
              radioButtons(
                "mode",
                NULL,
                c(
                  "Complete calculator (recommended)" = "complete",
                  "Quick endpoint preview" = "quick"
                ),
                "complete"
              ),
              conditionalPanel(
                "input.mode == 'quick'",
                selectInput(
                  "endpoint",
                  "Endpoint of interest",
                  quick_choices,
                  "craniotomy",
                  selectize = FALSE
                ),
                div(
                  class = "quick-note",
                  paste(
                    "Quick preview uses the same deployed endpoint model; hidden",
                    "predictors are passed as missing/unknown and this reduced-input",
                    "workflow was not separately validated."
                  )
                )
              )
            ),
            h2("Initial evaluation", class = "section-title"),
            uiOutput("mode_note"),
            div(class = "scroll", uiOutput("inputs")),
            actionButton(
              "calc",
              "Calculate trajectory",
              class = "btn-primary w-100"
            )
          )
        )
      ),
      card(
        card_body(
          h2("Predicted inpatient trajectory", class = "section-title"),
          uiOutput("results"),
          div(class = "footnote", uiOutput("result_note"))
        )
      )
    ),
    card(
      class = "details",
      card_body(
        h2(
          "Clinical context, performance, and intended use",
          class = "section-title"
        ),
        div(
          class = "detail-grid",
          div(
            h3("Who this model represents"),
            p(paste(
              "Developed from 755,880 direct-presenting adults aged 18–89 years",
              "with traumatic intracranial injury in ACS TQIP/TQP 2020–2024.",
              "The most recent temporal evaluation cohort contained 151,874 patients",
              "from 2024. Transfer-in patients and patients without an observable",
              "index-hospital trajectory were excluded."
            ))
          ),
          div(
            h3("When to use it"),
            p(paste(
              "Use after the initial trauma-center evaluation and diagnostic workup,",
              "when neurologic examination, physiology, mechanism, relevant",
              "comorbidities, and initial imaging-derived injury phenotypes are",
              "available. Registry data do not timestamp when each diagnosis was recognized."
            ))
          ),
          div(
            h3("What it estimates"),
            p(paste(
              "Three-class discharge disposition; hospital LOS trajectory; ICU and",
              "mechanical-ventilation trajectories; continuous hospital, ICU, and",
              "ventilator-duration forecasts; EVD or intraparenchymal ICP bolt",
              "utilization; and craniotomy/craniectomy. ICU and ventilator-duration",
              "forecasts are conditional on use of that resource."
            ))
          ),
          div(
            h3("2024 temporal performance"),
            p(paste(
              "AUROC: post-acute facility 0.785; death/hospice 0.939; any ICU 0.853;",
              "ICU ≥8 days 0.885; any ventilation 0.940; ventilation ≥8 days 0.916;",
              "hospital LOS ≥28 days 0.875; EVD/intraparenchymal ICP bolt 0.925;",
              "craniotomy/craniectomy 0.894. Calibration slopes for these headline",
              "classification outputs were approximately 0.99–1.04."
            ))
          ),
          div(
            h3("Duration performance"),
            p(paste(
              "2024 median-prediction mean absolute error was 4.54 days for hospital LOS,",
              "3.41 days for ICU LOS conditional on ICU use, and 4.79 days for",
              "ventilator duration conditional on ventilation. Observed coverage of",
              "estimated Q10–Q90 ranges was 79.5%, 80.0%, and 76.8%, respectively."
            ))
          ),
          div(
            h3("How to interpret it"),
            p(paste(
              "Predictions are risk estimates, not treatment recommendations or guarantees.",
              "Duration outputs are model-estimated 10th–90th percentile ranges rather",
              "than formal guaranteed 80% prediction intervals. Race, ethnicity, and",
              "payer are used only in disposition and categorical hospital-LOS models",
              "as social/health-system context, not as biological or causal attributes."
            ))
          ),
          div(
            h3("Validation status"),
            p(paste(
              "Architecture and predictor policies were locked using forward-temporal",
              "development procedures. Final deployment models were fit on the full",
              "2020–2024 cohort. Reported performance comes from rolling-origin temporal",
              "evaluation; independent external and prospective validation remain pending."
            ))
          ),
          div(
            h3("Important limitation"),
            p(paste(
              "Use clinical judgment, especially for patient groups with case mix unlike",
              "the development population. This calculator is intended to support counseling",
              "and resource planning; it should not be used as a stand-alone basis to",
              "initiate, withhold, or withdraw treatment."
            ))
          )
        )
      )
    )
  )
)

server <- function(input, output, session) {
  mode <- reactive(input$mode %||% "complete")
  endpoint <- reactive(input$endpoint %||% "craniotomy")
  allowed <- reactive({
    if (mode() == "complete") {
      all_predictors
    } else {
      quick_sets[[endpoint()]]
    }
  })

  output$mode_note <- renderUI({
    if (mode() == "complete") {
      div(
        class = "quick-note",
        paste0(
          "Complete mode displays all ", length(allowed()),
          " inputs used across the final endpoint-specific models. Leave unavailable ",
          "information unknown/blank. Numeric entries are constrained to clinically ",
          "plausible interface ranges and are rechecked before prediction."
        )
      )
    } else {
      div(
        class = "quick-note",
        paste0(
          "Quick preview shows ", length(allowed()),
          " focused inputs; hidden inputs are treated as unknown/missing."
        )
      )
    }
  })

  output$inputs <- renderUI({
    a <- allowed()

    panels <- lapply(names(groups), function(g) {
      vars <- intersect(groups[[g]], a)
      if (!length(vars)) return(NULL)

      do.call(
        accordion_panel,
        c(list(title = g), lapply(vars, make_control))
      )
    })

    panels <- panels[!vapply(panels, is.null, logical(1))]
    do.call(
      accordion,
      c(list(id = "acc", open = FALSE), panels)
    )
  })

  results_data <- eventReactive(
    input$calc,
    {
      input_errors <- validate_numeric_inputs(input, allowed())
      validate(need(
        length(input_errors) == 0L,
        paste(input_errors, collapse = " ")
      ))

      vals <- collect_values(input, allowed())
      ep <- endpoint()

      disp <- function() {
        pred_multi("discharge_3cat_final", vals)
      }
      hlos <- function() {
        list(
          trajectory = pred_multi("hlos_trajectory_final", vals),
          duration = pred_quantiles("hospital_los", vals)
        )
      }
      icu <- function() {
        list(
          trajectory = pred_multi("icu_trajectory_final", vals),
          duration = pred_quantiles("icu_los_conditional", vals)
        )
      }
      vent <- function() {
        list(
          trajectory = pred_multi("ventilation_trajectory_final", vals),
          duration = pred_quantiles("ventilator_days_conditional", vals)
        )
      }
      neuro <- function() {
        list(
          icp = pred_binary("icp_pressure_monitor_final", vals),
          craniotomy = pred_binary("craniotomy_craniectomy_final", vals)
        )
      }

      intub <- identical(vals$gcsq_intubated_recovered, 1)

      if (mode() == "quick") {
        if (ep == "disposition") {
          return(list(mode = "quick", ep = ep, disposition = disp()))
        }
        if (ep == "hlos") {
          return(c(list(mode = "quick", ep = ep), hlos()))
        }
        if (ep == "icu") {
          return(c(list(mode = "quick", ep = ep), icu()))
        }
        if (ep == "ventilation") {
          return(c(
            list(mode = "quick", ep = ep, intub = intub),
            vent()
          ))
        }

        n <- neuro()
        return(list(
          mode = "quick",
          ep = ep,
          icp = n$icp,
          craniotomy = n$craniotomy
        ))
      }

      n <- neuro()
      list(
        mode = "complete",
        ep = "all",
        disposition = disp(),
        hlos = hlos(),
        icu = icu(),
        ventilation = vent(),
        icp = n$icp,
        craniotomy = n$craniotomy,
        intub = intub
      )
    },
    ignoreInit = FALSE
  )

  output$results <- renderUI({
    r <- results_data()
    req(r)

    airway_notice <- if (isTRUE(r$intub)) {
      div(
        class = "notice",
        paste(
          "Baseline intubation is present. Focus on the ≥8-day ventilation",
          "trajectory and conditional duration rather than treating any ventilation",
          "as a purely future yes/no event."
        )
      )
    } else {
      NULL
    }

    if (r$mode == "quick") {
      if (r$ep == "disposition") {
        return(section_ui(
          "Discharge disposition",
          "Mutually exclusive probabilities",
          prob_cards(r$disposition, "result-disposition")
        ))
      }

      if (r$ep == "hlos") {
        return(section_ui(
          "Hospital length of stay",
          "Trajectory + continuous forecast",
          traj_duration(
            r$trajectory,
            r$duration,
            "result-hospital",
            "Predicted median hospital LOS"
          )
        ))
      }

      if (r$ep == "icu") {
        return(section_ui(
          "ICU trajectory",
          "Trajectory + conditional duration",
          traj_duration(
            r$trajectory,
            r$duration,
            "result-icu",
            "If ICU care occurs: predicted median ICU LOS",
            "Duration is conditional on ICU use."
          )
        ))
      }

      if (r$ep == "ventilation") {
        return(tagList(
          airway_notice,
          section_ui(
            "Mechanical ventilation trajectory",
            "Trajectory + conditional duration",
            traj_duration(
              r$trajectory,
              r$duration,
              "result-vent",
              "If ventilation occurs: predicted median ventilator duration",
              "Duration is conditional on mechanical ventilation."
            )
          )
        ))
      }

      if (r$ep == "icp") {
        return(neuro_section(icp = r$icp))
      }

      return(neuro_section(craniotomy = r$craniotomy))
    }

    tagList(
      airway_notice,
      section_ui(
        "Discharge disposition",
        "Mutually exclusive probabilities",
        prob_cards(r$disposition, "result-disposition")
      ),
      section_ui(
        "Hospital length of stay",
        "Trajectory + continuous forecast",
        traj_duration(
          r$hlos$trajectory,
          r$hlos$duration,
          "result-hospital",
          "Predicted median hospital LOS"
        )
      ),
      section_ui(
        "ICU trajectory",
        "Trajectory + conditional duration",
        traj_duration(
          r$icu$trajectory,
          r$icu$duration,
          "result-icu",
          "If ICU care occurs: predicted median ICU LOS",
          "Duration is conditional on ICU use."
        )
      ),
      section_ui(
        "Mechanical ventilation trajectory",
        "Trajectory + conditional duration",
        traj_duration(
          r$ventilation$trajectory,
          r$ventilation$duration,
          "result-vent",
          "If ventilation occurs: predicted median ventilator duration",
          "Duration is conditional on mechanical ventilation."
        )
      ),
      neuro_section(
        icp = r$icp,
        craniotomy = r$craniotomy
      )
    )
  })

  output$result_note <- renderUI({
    if (mode() == "quick") {
      span(paste(
        "Quick preview uses the same deployed endpoint model, but omitted",
        "information is treated as unknown/missing; use Complete mode for",
        "manuscript-concordant estimates."
      ))
    } else {
      span(paste(
        "Trajectory probabilities within each multiclass domain sum to 100%.",
        "ICU and ventilation duration forecasts are conditional on use of the",
        "corresponding resource. Duration ranges are estimated 10th–90th percentile ranges."
      ))
    }
  })
}

shinyApp(ui, server)
