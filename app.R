# TBI-TRACT: Trauma Resource and Acute Care Trajectory Calculator
# Locked 2020-2024 deployment build.
# Complete mode is manuscript-concordant. Quick preview uses the same endpoint
# model with unentered predictors passed as missing/unknown.

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
  "discharge_3cat_final", "hlos_trajectory_final", "hospital_los",
  "icu_trajectory_final", "icu_los_conditional",
  "ventilation_trajectory_final", "ventilator_days_conditional",
  "icp_pressure_monitor_final", "craniotomy_craniectomy_final"
)

# Large JSONs may live either in data/models/ (preferred) or the repo root.
resolve_model_path <- function(id) {
  p <- c(file.path(model_dir, paste0(id, ".json")), paste0(id, ".json"))
  hit <- p[file.exists(p)]
  if (length(hit)) hit[[1L]] else NA_character_
}

model_paths <- setNames(vapply(model_ids, resolve_model_path, character(1)), model_ids)
encoder_paths <- setNames(file.path(encoder_dir, paste0(model_ids, "_encoder.rds")), model_ids)
missing_models <- names(model_paths)[is.na(model_paths)]
missing_encoders <- names(encoder_paths)[!file.exists(encoder_paths)]
if (length(missing_models) || length(missing_encoders)) {
  stop(paste0(
    "Missing TBI-TRACT deployment files.\n",
    if (length(missing_models)) paste0("Model JSON(s): ", paste(missing_models, collapse = ", "),
                                      " (checked data/models/ and repo root)\n") else "",
    if (length(missing_encoders)) paste0("Encoder RDS(s): ", paste(missing_encoders, collapse = ", "),
                                        " (expected in data/encoders/)\n") else ""
  ), call. = FALSE)
}

load_entry <- function(id) {
  meta <- readRDS(encoder_paths[[id]])
  needed <- c("endpoint_id", "endpoint_label", "type", "predictors", "encoder", "feature_names")
  miss <- setdiff(needed, names(meta))
  if (length(miss)) stop(id, " encoder metadata missing: ", paste(miss, collapse = ", "))
  list(model = xgboost::xgb.load(model_paths[[id]]), meta = meta)
}
models <- setNames(lapply(model_ids, load_entry), model_ids)

# Locked-spec integrity checks.
removed_predictors <- c("helmet_use_recovered", "respiratoryassistance_clean")
social_predictors <- c("race_clean", "ethnicity_clean", "insurance_clean")
all_used <- unique(unlist(lapply(models, function(x) as.character(x$meta$predictors)), use.names = FALSE))
if (length(intersect(removed_predictors, all_used))) stop("Removed predictors remain in deployment metadata.")
for (id in setdiff(model_ids, c("discharge_3cat_final", "hlos_trajectory_final"))) {
  if (length(intersect(social_predictors, models[[id]]$meta$predictors)))
    stop(id, " unexpectedly contains race/ethnicity/payer.")
}
for (id in c("discharge_3cat_final", "hlos_trajectory_final")) {
  if (length(setdiff(social_predictors, models[[id]]$meta$predictors)))
    stop(id, " is missing final race/ethnicity/payer context predictors.")
}
hlos_rounds <- suppressWarnings(as.integer(models$hospital_los$meta$selected_rounds %||% NA_integer_))
if (!length(hlos_rounds) || !is.finite(hlos_rounds[1]) || hlos_rounds[1] != 4341L)
  warning("Hospital LOS metadata does not report the final 4,341-round extended-CV fit.")

all_predictors <- unique(all_used[!is.na(all_used) & nzchar(all_used)])
numeric_predictors <- unique(unlist(lapply(models, function(x) x$meta$encoder$numeric_vars %||% character()), use.names = FALSE))
categorical_predictors <- unique(unlist(lapply(models, function(x) x$meta$encoder$categorical_vars %||% character()), use.names = FALSE))
binary_predictors <- unique(c(
  "gcsq_intubated_recovered", "gcsq_sedated_paralyzed_recovered",
  "gcsq_eye_obstruction_recovered", "gcsq_unknown_recovered",
  grep("^dx_", all_predictors, value = TRUE), grep("^pmhx_", all_predictors, value = TRUE)
))
continuous_predictors <- setdiff(numeric_predictors, binary_predictors)

labels <- c(
  age="Age, years", sex_clean="Sex", race_clean="Race", ethnicity_clean="Ethnicity",
  insurance_clean="Payer / insurance", mechanism_clean="Mechanism of injury",
  gcs_eye_clean="GCS eye", gcs_verbal_clean="GCS verbal", gcs_motor_clean="GCS motor",
  gcsq_intubated_recovered="Intubated at GCS assessment",
  gcsq_sedated_paralyzed_recovered="Sedated / paralyzed at GCS assessment",
  gcsq_eye_obstruction_recovered="Eye score limited by obstruction",
  gcsq_unknown_recovered="GCS qualifier unknown / unassessable",
  pupil_clean="Pupillary response", sbp_clean="Systolic blood pressure, mmHg",
  pulse_clean="Heart rate, beats/min", rr_clean="Respiratory rate, breaths/min",
  spo2_clean="Oxygen saturation, %", temperature_c_recovered="Temperature, °C",
  supplemental_oxygen_recovered="Supplemental oxygen",
  dx_concussion="Concussion", dx_cerebral_edema_traumatic="Traumatic cerebral edema",
  dx_diffuse_axonal_injury="Diffuse axonal injury",
  dx_focal_contusion_or_iph="Contusion / intraparenchymal hemorrhage",
  dx_epidural_hematoma="Epidural hematoma", dx_subdural_hematoma="Subdural hematoma",
  dx_subarachnoid_hemorrhage="Traumatic subarachnoid hemorrhage",
  dx_other_intracranial_injury="Other intracranial injury",
  dx_cranial_skull_fracture="Cranial skull fracture", dx_facial_fracture="Facial fracture",
  dx_other_skull_or_facial_fracture="Other skull / facial fracture",
  dx_spinal_cord_injury="Spinal cord injury", dx_neck_vascular_injury="Neck vascular injury",
  dx_thoracic_injury="Thoracic injury", dx_abdominal_pelvic_injury="Abdominal / pelvic injury",
  dx_upper_extremity_injury="Upper-extremity injury", dx_lower_extremity_injury="Lower-extremity injury",
  pmhx_bleeding_disorder="Bleeding disorder / coagulopathy",
  pmhx_anticoagulant_therapy="Anticoagulant therapy", pmhx_copd="COPD",
  pmhx_diabetes="Diabetes", pmhx_hypertension="Hypertension",
  pmhx_current_smoker="Current smoker", pmhx_functional_dependence="Functional dependence",
  pmhx_dementia="Dementia", pmhx_chf="Congestive heart failure",
  pmhx_chronic_renal_failure="Chronic renal failure", pmhx_cirrhosis="Cirrhosis",
  pmhx_steroid_use="Chronic steroid use"
)
label_for <- function(v) if (v %in% names(labels)) unname(labels[v]) else gsub("_", " ", v)

groups <- list(
  "Demographics and context"=c("age","sex_clean","race_clean","ethnicity_clean","insurance_clean"),
  "Mechanism and neurologic status"=c("mechanism_clean","gcs_eye_clean","gcs_verbal_clean","gcs_motor_clean",
    "gcsq_intubated_recovered","gcsq_sedated_paralyzed_recovered","gcsq_eye_obstruction_recovered",
    "gcsq_unknown_recovered","pupil_clean"),
  "Admission physiology"=c("sbp_clean","pulse_clean","rr_clean","spo2_clean","temperature_c_recovered","supplemental_oxygen_recovered"),
  "Cranial injury pattern"=c("dx_concussion","dx_cerebral_edema_traumatic","dx_diffuse_axonal_injury",
    "dx_focal_contusion_or_iph","dx_epidural_hematoma","dx_subdural_hematoma","dx_subarachnoid_hemorrhage",
    "dx_other_intracranial_injury","dx_cranial_skull_fracture","dx_facial_fracture","dx_other_skull_or_facial_fracture"),
  "Extracranial injury pattern"=c("dx_spinal_cord_injury","dx_neck_vascular_injury","dx_thoracic_injury",
    "dx_abdominal_pelvic_injury","dx_upper_extremity_injury","dx_lower_extremity_injury"),
  "Pre-existing conditions"=c("pmhx_bleeding_disorder","pmhx_anticoagulant_therapy","pmhx_copd","pmhx_diabetes",
    "pmhx_hypertension","pmhx_current_smoker","pmhx_functional_dependence","pmhx_dementia","pmhx_chf",
    "pmhx_chronic_renal_failure","pmhx_cirrhosis","pmhx_steroid_use")
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
  if (x == "__UNKNOWN__") return("Unknown / not recorded")
  if (x == "__OTHER__") return("Other / unlisted")
  if (x == "Transport/MVC") return("Transport-related")
  x
}
input_id <- function(v) paste0("var__", v)

gcs_choices <- list(
  gcs_eye_clean=c("Unknown"="", "1 - None"="1", "2 - To pain"="2", "3 - To speech"="3", "4 - Spontaneous"="4"),
  gcs_verbal_clean=c("Unknown"="", "1 - None"="1", "2 - Incomprehensible sounds"="2", "3 - Inappropriate words"="3", "4 - Confused"="4", "5 - Oriented"="5"),
  gcs_motor_clean=c("Unknown"="", "1 - None"="1", "2 - Extension"="2", "3 - Flexion"="3", "4 - Withdraws"="4", "5 - Localizes"="5", "6 - Obeys commands"="6")
)

make_control <- function(v) {
  id <- input_id(v); lab <- label_for(v)
  if (v %in% names(gcs_choices)) return(selectInput(id, lab, gcs_choices[[v]], selected="", selectize=FALSE))
  if (v %in% binary_predictors) return(selectInput(id, lab, c("Unknown"="", "No"="0", "Yes"="1"), selected="", selectize=FALSE))
  if (v %in% continuous_predictors) return(textInput(id, lab, value="", placeholder="Unknown"))
  if (v %in% categorical_predictors) {
    lev <- get_levels(v); if (!length(lev)) lev <- "__UNKNOWN__"
    ch <- setNames(lev, vapply(lev, pretty_level, character(1)))
    sel <- if ("__UNKNOWN__" %in% lev) "__UNKNOWN__" else lev[1]
    return(selectInput(id, lab, ch, selected=sel, selectize=FALSE))
  }
  NULL
}

# Quick preview is intentionally focused, not a separately validated reduced-input model.
quick_common <- c("age","sex_clean","gcs_eye_clean","gcs_verbal_clean","gcs_motor_clean",
  "gcsq_intubated_recovered","gcsq_sedated_paralyzed_recovered","pupil_clean","sbp_clean","rr_clean","spo2_clean",
  "supplemental_oxygen_recovered","mechanism_clean","dx_cerebral_edema_traumatic","dx_diffuse_axonal_injury",
  "dx_focal_contusion_or_iph","dx_epidural_hematoma","dx_subdural_hematoma","dx_subarachnoid_hemorrhage",
  "dx_cranial_skull_fracture","dx_thoracic_injury","dx_abdominal_pelvic_injury","pmhx_anticoagulant_therapy",
  "pmhx_functional_dependence","pmhx_dementia")
quick_sets <- list(
  disposition=unique(c(quick_common,"race_clean","ethnicity_clean","insurance_clean","dx_upper_extremity_injury","dx_lower_extremity_injury","pmhx_chf","pmhx_chronic_renal_failure")),
  hlos=unique(c(quick_common,"race_clean","ethnicity_clean","insurance_clean","pulse_clean","temperature_c_recovered","dx_spinal_cord_injury","dx_neck_vascular_injury","dx_upper_extremity_injury","dx_lower_extremity_injury","pmhx_chf","pmhx_chronic_renal_failure")),
  icu=unique(c(quick_common,"pulse_clean","temperature_c_recovered","dx_spinal_cord_injury","dx_neck_vascular_injury")),
  ventilation=unique(c(quick_common,"pulse_clean","temperature_c_recovered","dx_spinal_cord_injury")),
  icp=unique(c(quick_common,"dx_other_intracranial_injury","dx_other_skull_or_facial_fracture")),
  craniotomy=unique(c(quick_common,"dx_other_intracranial_injury","dx_other_skull_or_facial_fracture"))
)
quick_sets <- lapply(quick_sets, intersect, y=all_predictors)
quick_choices <- c("Discharge disposition"="disposition", "Hospital length of stay"="hlos",
  "ICU trajectory"="icu", "Mechanical ventilation trajectory"="ventilation",
  "ICP monitoring"="icp", "Craniotomy / craniectomy"="craniotomy")

collect_values <- function(input, allowed) {
  vals <- setNames(vector("list", length(all_predictors)), all_predictors)
  for (v in all_predictors) {
    if (!v %in% allowed) {
      vals[[v]] <- if (v %in% categorical_predictors) "__UNKNOWN__" else NA_real_
      next
    }
    x <- input[[input_id(v)]]
    if (v %in% numeric_predictors) {
      z <- suppressWarnings(as.numeric(if (length(x)) x[[1]] else NA))
      vals[[v]] <- if (is.finite(z)) z else NA_real_
    } else {
      z <- if (length(x)) as.character(x[[1]]) else ""
      vals[[v]] <- if (is.na(z) || !nzchar(z)) "__UNKNOWN__" else z
    }
  }
  vals
}

encode_values <- function(entry, vals) {
  e <- entry$meta$encoder; f <- as.character(entry$meta$feature_names)
  X <- matrix(0, 1, length(f), dimnames=list(NULL, f))
  for (v in e$numeric_vars) if (v %in% f) X[1,v] <- suppressWarnings(as.numeric(vals[[v]]))
  for (v in e$categorical_vars) {
    lev <- as.character(e$categorical_levels[[v]])
    val <- as.character(vals[[v]] %||% "__UNKNOWN__")
    if (is.na(val) || !nzchar(val)) val <- "__UNKNOWN__"
    if (!val %in% lev) val <- if ("__OTHER__" %in% lev) "__OTHER__" else if ("__UNKNOWN__" %in% lev) "__UNKNOWN__" else lev[1]
    names_f <- paste0(v, "__", make.names(lev, unique=TRUE))
    keep <- intersect(names_f, f); X[1,keep] <- 0
    idx <- match(val, lev); if (!is.na(idx) && names_f[idx] %in% f) X[1,names_f[idx]] <- 1
  }
  storage.mode(X) <- "double"; X
}
raw_pred <- function(id, vals) {
  e <- models[[id]]; X <- encode_values(e, vals)
  as.numeric(predict(e$model, xgboost::xgb.DMatrix(X, missing=NA_real_)))
}
pred_multi <- function(id, vals) {
  lev <- as.character(models[[id]]$meta$levels); p <- raw_pred(id, vals)
  if (length(p) != length(lev)) stop(id, " returned wrong number of classes")
  p[!is.finite(p)] <- 0; p <- pmax(p,0); p <- if (sum(p)>0) p/sum(p) else rep(1/length(p),length(p))
  data.table(class=lev, probability=p)
}
pred_binary <- function(id, vals) pmin(1,pmax(0,raw_pred(id,vals)[1]))
pred_quantiles <- function(id, vals) {
  p <- raw_pred(id, vals); if (length(p)<3) stop(id," did not return Q10/Q50/Q90")
  p <- expm1(p[1:3]); lo <- suppressWarnings(as.numeric(models[[id]]$meta$lower_bound %||% 0)); if(!is.finite(lo)) lo<-0
  p <- sort(pmax(p,lo)); setNames(p,c("q10","q50","q90"))
}

fmt_prob <- function(p) {
  x <- 100*as.numeric(p); if (!is.finite(x)) return("—")
  if (x>0 && x<0.1) return("<0.1%")
  if (x<5 || x>95) sprintf("%.1f%%",x) else sprintf("%.0f%%",x)
}
fmt_days <- function(x) if (!is.finite(as.numeric(x))) "—" else if (x<10) sprintf("%.1f d",x) else sprintf("%.0f d",x)
pretty_class <- function(x) {
  map <- c("Home/home health"="Home / home health","Post-acute facility"="Post-acute facility","Death/hospice"="Death / hospice",
    "Hospital LOS <=7 days"="≤7 days","Hospital LOS 8-27 days"="8–27 days","Hospital LOS >=28 days"="≥28 days",
    "No ICU"="No ICU","ICU 1-7 days"="ICU 1–7 days","ICU >=8 days"="ICU ≥8 days",
    "No ventilation"="No ventilation","Ventilation 1-7 days"="Ventilation 1–7 days","Ventilation >=8 days"="Ventilation ≥8 days")
  if (x %in% names(map)) unname(map[x]) else x
}
prob_cards <- function(d, cls) div(class="result-grid", lapply(seq_len(nrow(d)), function(i)
  div(class=paste("result-card",cls), div(class="result-top",span(class="result-name",pretty_class(d$class[i])),
      span(class="result-pct",fmt_prob(d$probability[i]))),
      div(class="bar",div(class="fill",style=paste0("width:",100*d$probability[i],"%;"))))))
duration_card <- function(title,q,cls,note=NULL) div(class=paste("duration-card",cls),
  div(class="duration-label",title),div(class="duration-main",fmt_days(q["q50"])),
  div(class="duration-range",paste0("Estimated 10th–90th percentile range: ",fmt_days(q["q10"])," – ",fmt_days(q["q90"]))),
  if(!is.null(note)) div(class="small-note",note))
section_ui <- function(title,note,body) div(class="result-section",div(class="result-heading",h3(title),span(note)),body)
traj_duration <- function(d,q,cls,title,note=NULL) tagList(prob_cards(d,cls),duration_card(title,q,cls,note))

css <- "
body{background:#f4f7fb;color:#243447}.app{max-width:1450px;margin:auto;padding:22px}.hero,.card{background:white;border:1px solid #e3ebf3;border-radius:22px;box-shadow:0 8px 26px rgba(31,52,73,.06)}
.hero{padding:22px 26px;margin-bottom:20px}.hero-grid{display:grid;grid-template-columns:84px 1fr;gap:18px;align-items:center}.logo{width:78px}.title{font-weight:850;font-size:clamp(1.8rem,3vw,2.8rem);margin:0}.subtitle{color:#5b6d7f;margin:4px 0 0}
.sticky{position:sticky;top:18px}.scroll{max-height:calc(100vh - 355px);overflow-y:auto;padding-right:5px}.mode{background:#edf4fb;border-radius:15px;padding:12px;margin-bottom:12px}.btn-primary{border-radius:13px;font-weight:750;min-height:44px}.section-title{font-weight:800}.result-section{padding:4px 0 19px;border-bottom:1px solid #edf1f5;margin-bottom:17px}.result-heading{display:flex;justify-content:space-between;align-items:baseline;gap:12px}.result-heading h3{font-size:1.05rem;font-weight:800}.result-heading span{font-size:.82rem;color:#6b7a8c;font-weight:650}.result-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:11px}.result-card,.duration-card{border:1px solid #dbe8f5;border-radius:17px;padding:13px;background:#f9fbfd}.result-top{display:flex;justify-content:space-between;gap:10px}.result-name{font-weight:700;color:#425466}.result-pct{font-size:1.7rem;font-weight:900}.bar{height:11px;background:#e7edf5;border-radius:99px;overflow:hidden;margin-top:10px}.fill{height:100%;border-radius:99px}.result-disposition .fill{background:#1f4e79}.result-hospital .fill{background:#2f7d32}.result-icu .fill{background:#b45f06}.result-vent .fill{background:#187b80}.result-neuro .fill{background:#6f42c1}.duration-card{margin-top:11px;background:white}.duration-main{font-size:2rem;font-weight:900}.duration-label,.duration-range{font-weight:700;color:#425466}.small-note,.footnote,.quick-note{font-size:.84rem;color:#6b7a8c}.notice{background:#fff8e6;border:1px solid #f1d99a;border-radius:14px;padding:11px 13px;margin-bottom:13px}.details{margin-top:18px}.detail-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px 28px}.detail-grid h3{font-size:1rem;font-weight:800}@media(max-width:1199px){.sticky{position:static}.scroll{max-height:none}}@media(max-width:900px){.result-grid,.detail-grid{grid-template-columns:1fr}.hero-grid{grid-template-columns:1fr}.app{padding:12px}}
"

ui <- page_fluid(
  theme=bs_theme(version=5,bootswatch="flatly",primary="#1f4e79",bg="#f4f7fb",fg="#243447"),
  tags$head(tags$style(HTML(css))),
  div(class="app",
    div(class="hero",div(class="hero-grid",
      div(if(file.exists("www/ohsu_logo.png")) img(src="ohsu_logo.png",class="logo") else strong("OHSU")),
      div(h1("TBI-TRACT: Trauma Resource and Acute Care Trajectory Calculator",class="title"),
          p("Post-initial-evaluation prediction of disposition, resource trajectory, neurosurgical utilization, and duration",class="subtitle")))),
    layout_columns(col_widths=c(4,8),
      div(class="sticky",card(card_body(
        h2("Calculator mode",class="section-title"),
        div(class="mode",radioButtons("mode",NULL,c("Complete calculator (recommended)"="complete","Quick endpoint preview"="quick"),"complete"),
          conditionalPanel("input.mode == 'quick'",selectInput("endpoint","Endpoint of interest",quick_choices,"craniotomy",selectize=FALSE),
            div(class="quick-note","Quick preview uses the same deployed endpoint model; hidden predictors are passed as missing/unknown and this reduced-input workflow was not separately validated."))),
        h2("Initial evaluation",class="section-title"),uiOutput("mode_note"),div(class="scroll",uiOutput("inputs")),
        actionButton("calc","Calculate trajectory",class="btn-primary w-100")))),
      card(card_body(h2("Predicted inpatient trajectory",class="section-title"),uiOutput("results"),div(class="footnote",uiOutput("result_note"))))),
    card(class="details",card_body(h2("Model details and intended use",class="section-title"),div(class="detail-grid",
      div(h3("Prediction time"),p("Use after the initial trauma-center evaluation and diagnostic workup, before subsequent inpatient disposition and resource utilization. Initial imaging-derived injury phenotypes may be entered.")),
      div(h3("Outputs"),p("Disposition; HLOS, ICU and ventilation trajectories; conditional duration forecasts; invasive ICP monitoring; and craniotomy/craniectomy.")),
      div(h3("Development"),p("Final deployment models were fit on the complete 2020–2024 development cohort after architecture was locked. Performance estimates shown in the manuscript come from rolling-origin 2022–2024 temporal evaluation; independent external validation remains pending.")),
      div(h3("Interpretation"),p("Duration outputs are estimated 10th–90th percentile ranges, not guaranteed 80% prediction intervals. Race/ethnicity/payer are used only for disposition and categorical HLOS as social/health-system context, not as biological or causal attributes."))
    )))
  )
)

server <- function(input, output, session) {
  mode <- reactive(input$mode %||% "complete")
  endpoint <- reactive(input$endpoint %||% "craniotomy")
  allowed <- reactive(if(mode()=="complete") all_predictors else quick_sets[[endpoint()]])

  output$mode_note <- renderUI(div(class="quick-note",if(mode()=="complete")
    paste0("Complete mode displays all ",length(allowed())," inputs used across the final endpoint-specific models. Leave unavailable information unknown/blank.") else
    paste0("Quick preview shows ",length(allowed())," focused inputs; hidden inputs are treated as unknown/missing.")))

  output$inputs <- renderUI({
    a <- allowed()
    panels <- lapply(names(groups),function(g){
      vars <- intersect(groups[[g]],a); if(!length(vars)) return(NULL)
      do.call(accordion_panel,c(list(title=g),lapply(vars,make_control)))
    })
    panels <- panels[!vapply(panels,is.null,logical(1))]
    do.call(accordion,c(list(id="acc",open=FALSE),panels))
  })

  R <- eventReactive(input$calc,{
    vals <- collect_values(input,allowed()); ep <- endpoint()
    disp <- function() pred_multi("discharge_3cat_final",vals)
    hlos <- function() list(trajectory=pred_multi("hlos_trajectory_final",vals),duration=pred_quantiles("hospital_los",vals))
    icu <- function() list(trajectory=pred_multi("icu_trajectory_final",vals),duration=pred_quantiles("icu_los_conditional",vals))
    vent <- function() list(trajectory=pred_multi("ventilation_trajectory_final",vals),duration=pred_quantiles("ventilator_days_conditional",vals))
    neuro <- function() list(icp=pred_binary("icp_pressure_monitor_final",vals),craniotomy=pred_binary("craniotomy_craniectomy_final",vals))
    intub <- identical(vals$gcsq_intubated_recovered,1)
    if(mode()=="quick") {
      if(ep=="disposition") return(list(mode="quick",ep=ep,disposition=disp()))
      if(ep=="hlos") return(c(list(mode="quick",ep=ep),hlos()))
      if(ep=="icu") return(c(list(mode="quick",ep=ep),icu()))
      if(ep=="ventilation") return(c(list(mode="quick",ep=ep,intub=intub),vent()))
      n<-neuro(); return(list(mode="quick",ep=ep,icp=n$icp,craniotomy=n$craniotomy))
    }
    n<-neuro(); list(mode="complete",ep="all",disposition=disp(),hlos=hlos(),icu=icu(),ventilation=vent(),icp=n$icp,craniotomy=n$craniotomy,intub=intub)
  },ignoreInit=FALSE)

  output$results <- renderUI({
    r<-R(); req(r)
    airway <- function() if(isTRUE(r$intub)) div(class="notice","Baseline intubation is present. Focus on the ≥8-day ventilation trajectory and conditional duration rather than treating any ventilation as a purely future yes/no event.")
    neuro_ui <- function(p,title) section_ui("Neurosurgical resource utilization","Independent probability estimate",div(class="result-grid",div(class="result-card result-neuro",div(class="result-top",span(class="result-name",title),span(class="result-pct",fmt_prob(p))),div(class="bar",div(class="fill",style=paste0("width:",100*p,"%;"))))))
    if(r$mode=="quick") {
      if(r$ep=="disposition") return(section_ui("Discharge disposition","Mutually exclusive probabilities",prob_cards(r$disposition,"result-disposition")))
      if(r$ep=="hlos") return(section_ui("Hospital length of stay","Trajectory + continuous forecast",traj_duration(r$trajectory,r$duration,"result-hospital","Predicted median hospital LOS")))
      if(r$ep=="icu") return(section_ui("ICU trajectory","Trajectory + conditional duration",traj_duration(r$trajectory,r$duration,"result-icu","If ICU care occurs: predicted median ICU LOS","Duration is conditional on ICU use.")))
      if(r$ep=="ventilation") return(tagList(airway(),section_ui("Mechanical ventilation trajectory","Trajectory + conditional duration",traj_duration(r$trajectory,r$duration,"result-vent","If ventilation occurs: predicted median ventilator duration","Duration is conditional on mechanical ventilation."))))
      return(if(r$ep=="icp") neuro_ui(r$icp,"Invasive ICP monitoring") else neuro_ui(r$craniotomy,"Craniotomy / craniectomy"))
    }
    tagList(airway(),
      section_ui("Discharge disposition","Mutually exclusive probabilities",prob_cards(r$disposition,"result-disposition")),
      section_ui("Hospital length of stay","Trajectory + continuous forecast",traj_duration(r$hlos$trajectory,r$hlos$duration,"result-hospital","Predicted median hospital LOS")),
      section_ui("ICU trajectory","Trajectory + conditional duration",traj_duration(r$icu$trajectory,r$icu$duration,"result-icu","If ICU care occurs: predicted median ICU LOS","Duration is conditional on ICU use.")),
      section_ui("Mechanical ventilation trajectory","Trajectory + conditional duration",traj_duration(r$ventilation$trajectory,r$ventilation$duration,"result-vent","If ventilation occurs: predicted median ventilator duration","Duration is conditional on mechanical ventilation.")),
      section_ui("Neurosurgical resource utilization","Independent probability estimates",div(class="result-grid",
        div(class="result-card result-neuro",div(class="result-top",span(class="result-name","Invasive ICP monitoring"),span(class="result-pct",fmt_prob(r$icp))),div(class="bar",div(class="fill",style=paste0("width:",100*r$icp,"%;")))),
        div(class="result-card result-neuro",div(class="result-top",span(class="result-name","Craniotomy / craniectomy"),span(class="result-pct",fmt_prob(r$craniotomy))),div(class="bar",div(class="fill",style=paste0("width:",100*r$craniotomy,"%;"))))))
    )
  })

  output$result_note <- renderUI(span(if(mode()=="quick")
    "Quick preview uses the same deployed endpoint model, but omitted information is treated as unknown/missing; use Complete mode for manuscript-concordant estimates." else
    "Trajectory probabilities within each multiclass domain sum to 100%. ICU and ventilation duration forecasts are conditional on use of the corresponding resource. Duration ranges are estimated 10th–90th percentile ranges."))
}

shinyApp(ui,server)
