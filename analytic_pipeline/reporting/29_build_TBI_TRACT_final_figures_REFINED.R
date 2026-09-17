# =============================================================================
# 29_build_TBI_TRACT_final_figures.R
#
# FINAL FIGURE PIPELINE
#
# Produces the locked manuscript figure package:
#
# MAIN
#   Figure 1  TBI-TRACT prediction framework / temporal design / architecture
#   Figure 2  Temporal performance fingerprint
#   Figure 3A 2024 probability calibration
#   Figure 3B 2024 continuous-trajectory calibration
#
# SUPPLEMENT
#   eFigure 1 Cohort flow + year-specific observation status
#   eFigure 2 Temporal case-mix drift love plot
#   eFigure 3 Complete 2024 calibration grid
#   eFigure 4 Duration quantile calibration over time
#   eFigure 5 Predictor-family ablation heatmap
#   eFigure 6 Subgroup robustness + social/context calibration shift
#
# DESIGN PRINCIPLES
#   - manuscript-facing performance comes from locked rolling-origin predictions
#   - no ROC-curve wall and no argmax confusion matrices
#   - calibration uses flexible individual-level curves; grouped points are anchors
#   - AUPRC is shown relative to prevalence for uncommon outcomes
#   - duration Q10-Q90 is called an estimated 10th-90th percentile range
#   - predictor ablation, not generic feature importance, is used to explain
#     why information was retained or removed
#
# EXPECTED PREREQUISITES
#   - script 25 final sensitivity suite completed
#   - script 27 manuscript output data completed
#   - final pragmatic classification and predictor-ablation outputs available
#
# OUTPUT
#   output/TBI_TRACT_FINAL_FIGURES/
#   Each figure is saved as:
#       PNG, 300 dpi (review)
#       TIFF, 600 dpi, LZW compression (submission)
# =============================================================================

rm(list = ls())
gc()

required_packages <- c(
  "data.table",
  "ggplot2",
  "patchwork",
  "scales",
  "grid"
)

missing_packages <- required_packages[
  !vapply(
    required_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_packages) > 0L) {
  stop(
    "Missing package(s): ",
    paste(missing_packages, collapse = ", "),
    "\nInstall them before running this figure pipeline.",
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(grid)
})

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

config_candidates <- c(
  file.path(getwd(), "R", "00_config.R"),
  "R/00_config.R"
)

config_file <- config_candidates[
  file.exists(config_candidates)
][1L]

if (
  length(config_file) == 0L ||
    is.na(config_file)
) {
  stop(
    "Could not find R/00_config.R.",
    call. = FALSE
  )
}

source(config_file)

if (
  !exists("output_dir") ||
    is.null(output_dir)
) {
  stop(
    "00_config.R did not define output_dir.",
    call. = FALSE
  )
}

figure_dir <- file.path(
  output_dir,
  "TBI_TRACT_FINAL_FIGURES"
)

dir.create(
  figure_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

# Index files once; this makes the script robust to one extra outer ZIP folder.
all_files <- list.files(
  output_dir,
  recursive = TRUE,
  full.names = TRUE,
  include.dirs = FALSE
)

find_file <- function(
    filename,
    prefer = NULL,
    required = TRUE
) {
  hits <- all_files[
    basename(all_files) ==
      filename
  ]

  if (
    !is.null(prefer) &&
      length(hits) > 0L
  ) {
    preferred <- hits[
      grepl(
        prefer,
        hits,
        fixed = TRUE
      )
    ]

    if (
      length(preferred) > 0L
    ) {
      hits <- preferred
    }
  }

  if (
    length(hits) == 0L
  ) {
    if (
      isTRUE(required)
    ) {
      stop(
        "Could not find required file: ",
        filename,
        if (!is.null(prefer)) {
          paste0(
            "\nPreferred path contained: ",
            prefer
          )
        } else {
          ""
        },
        call. = FALSE
      )
    }

    return(
      NA_character_
    )
  }

  # Prefer the shortest canonical path if duplicate archived copies exist.
  hits[
    which.min(
      nchar(hits)
    )
  ]
}

# -----------------------------------------------------------------------------
# Required source files
# -----------------------------------------------------------------------------

files <- list(
  temporal =
    find_file(
      "09_TEMPORAL_PERFORMANCE_FIGURE_DATA.csv",
      "TBI_TRACT_MANUSCRIPT_OUTPUT_DATA"
    ),
  headline_ci =
    find_file(
      "03_2024_HEADLINE_CLASSIFICATION_BOOTSTRAP_CI.csv",
      "TBI_TRACT_MANUSCRIPT_OUTPUT_DATA"
    ),
  calibration =
    find_file(
      "04_2024_FLEXIBLE_CALIBRATION_CURVES.csv",
      "TBI_TRACT_MANUSCRIPT_OUTPUT_DATA"
    ),
  calibration_points =
    find_file(
      "05_2024_CALIBRATION_DECILE_POINTS.csv",
      "TBI_TRACT_MANUSCRIPT_OUTPUT_DATA"
    ),
  duration_calibration =
    find_file(
      "08_2024_DURATION_CALIBRATION_BINS.csv",
      "TBI_TRACT_MANUSCRIPT_OUTPUT_DATA"
    ),
  duration_year =
    find_file(
      "06_DURATION_YEAR_SPECIFIC_METRICS.csv",
      "TBI_TRACT_MANUSCRIPT_OUTPUT_DATA"
    ),
  flow =
    find_file(
      "01_DIRECT_PRESENTATION_OBSERVATION_STATUS_BY_YEAR.csv",
      "METHODS_COMPLETION_TBI_TRACT"
    ),
  drift =
    find_file(
      "06_DEVELOPMENT_2020_23_VS_2024_SMD.csv",
      "METHODS_COMPLETION_TBI_TRACT"
    ),
  ablation =
    find_file(
      "11_CLASS_LEVEL_TEMPORAL_ABLATION_SUMMARY.csv",
      "TBI_TRACT_PREDICTOR_STRESS_TEST"
    ),
  subgroup =
    find_file(
      "10_2024_SUBGROUP_ROBUSTNESS_HEATMAP_DATA.csv",
      "TBI_TRACT_MANUSCRIPT_OUTPUT_DATA"
    ),
  social =
    find_file(
      "11_2024_SOCIAL_AUGMENTATION_CALIBRATION_SHIFT_DATA.csv",
      "TBI_TRACT_MANUSCRIPT_OUTPUT_DATA"
    )
)

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

temporal <- fread(
  files$temporal
)

headline_ci <- fread(
  files$headline_ci
)

calibration <- fread(
  files$calibration
)

calibration_points <- fread(
  files$calibration_points
)

duration_calibration <- fread(
  files$duration_calibration
)

duration_year <- fread(
  files$duration_year
)

flow <- fread(
  files$flow
)

drift <- fread(
  files$drift
)

ablation <- fread(
  files$ablation
)

subgroup <- fread(
  files$subgroup
)

social <- fread(
  files$social
)

# -----------------------------------------------------------------------------
# Visual system
# -----------------------------------------------------------------------------

BASE_FONT <- "Arial"

NAVY <- "#17365D"
BLUE <- "#4472C4"
GREEN <- "#548235"
PURPLE <- "#8064A2"
ORANGE <- "#C55A11"
GOLD <- "#BF9000"
RED <- "#C00000"
MID_GRAY <- "#7F7F7F"
LIGHT_GRAY <- "#D9D9D9"
VERY_LIGHT_GRAY <- "#F2F2F2"
BLACK <- "#1F1F1F"

domain_palette <- c(
  "Disposition" =
    BLUE,
  "Hospital LOS" =
    GOLD,
  "ICU" =
    GREEN,
  "Ventilation" =
    PURPLE,
  "Neurosurgical resources" =
    ORANGE
)

domain_shapes <- c(
  "Disposition" =
    16,
  "Hospital LOS" =
    15,
  "ICU" =
    17,
  "Ventilation" =
    18,
  "Neurosurgical resources" =
    8
)

theme_tract <- function(
    base_size = 9
) {
  theme_minimal(
    base_family = BASE_FONT,
    base_size = base_size
  ) +
    theme(
      plot.title =
        element_text(
          face = "bold",
          color = NAVY,
          size = base_size +
            2,
          hjust = 0
        ),
      plot.subtitle =
        element_text(
          color = MID_GRAY,
          size = base_size
        ),
      plot.caption =
        element_text(
          color = MID_GRAY,
          size = base_size -
            1,
          hjust = 0
        ),
      axis.title =
        element_text(
          color = BLACK,
          face = "bold"
        ),
      axis.text =
        element_text(
          color = BLACK
        ),
      strip.text =
        element_text(
          face = "bold",
          color = NAVY
        ),
      strip.background =
        element_rect(
          fill = VERY_LIGHT_GRAY,
          color = NA
        ),
      panel.grid.minor =
        element_blank(),
      panel.grid.major.y =
        element_blank(),
      legend.title =
        element_text(
          face = "bold"
        ),
      legend.position =
        "bottom",
      plot.margin =
        margin(
          8,
          12,
          8,
          8
        )
    )
}

save_figure <- function(
    plot,
    stem,
    width,
    height
) {
  png_file <- file.path(
    figure_dir,
    paste0(
      stem,
      ".png"
    )
  )

  tiff_file <- file.path(
    figure_dir,
    paste0(
      stem,
      ".tiff"
    )
  )

  ggsave(
    filename = png_file,
    plot = plot,
    width = width,
    height = height,
    units = "in",
    dpi = 300,
    bg = "white"
  )

  ggsave(
    filename = tiff_file,
    plot = plot,
    width = width,
    height = height,
    units = "in",
    dpi = 600,
    device = "tiff",
    compression = "lzw",
    bg = "white"
  )

  invisible(
    list(
      png = png_file,
      tiff = tiff_file
    )
  )
}

# -----------------------------------------------------------------------------
# Label helpers
# -----------------------------------------------------------------------------

target_order <- c(
  "Post-acute facility",
  "Death/hospice",
  "Any ICU",
  "ICU >=8 days",
  "Any ventilation",
  "Ventilation >=8 days",
  "Hospital LOS >=28 days",
  "Invasive ICP monitoring",
  "Craniotomy/craniectomy"
)

target_labels <- c(
  "Post-acute facility" =
    "Post-acute disposition",
  "Death/hospice" =
    "Death / hospice",
  "Any ICU" =
    "Any ICU",
  "ICU >=8 days" =
    "ICU \u22658 d",
  "Any ventilation" =
    "Any ventilation",
  "Ventilation >=8 days" =
    "Ventilation \u22658 d",
  "Hospital LOS >=28 days" =
    "Hospital LOS \u226528 d",
  "Invasive ICP monitoring" =
    "ICP monitoring",
  "Craniotomy/craniectomy" =
    "Craniotomy / craniectomy"
)

publication_label <- function(x) {
  x <- gsub(
    ">=",
    "\u2265",
    x,
    fixed = TRUE
  )
  x <- gsub(
    "<=",
    "\u2264",
    x,
    fixed = TRUE
  )
  x
}

domain_for_target <- function(
    x
) {
  fcase(
    x %in%
      c(
        "Post-acute facility",
        "Death/hospice"
      ),
    "Disposition",
    x ==
      "Hospital LOS >=28 days",
    "Hospital LOS",
    x %in%
      c(
        "Any ICU",
        "ICU >=8 days"
      ),
    "ICU",
    x %in%
      c(
        "Any ventilation",
        "Ventilation >=8 days"
      ),
    "Ventilation",
    x %in%
      c(
        "Invasive ICP monitoring",
        "Craniotomy/craniectomy"
      ),
    "Neurosurgical resources",
    default =
      "Other"
  )
}

pretty_predictor <- function(
    x
) {
  known <- c(
    "age" =
      "Age",
    "sex_clean" =
      "Sex",
    "gcs_eye_clean" =
      "GCS eye",
    "gcs_verbal_clean" =
      "GCS verbal",
    "gcs_motor_clean" =
      "GCS motor",
    "gcsq_intubated_recovered" =
      "GCS qualifier: intubated",
    "gcsq_sedated_paralyzed_recovered" =
      "GCS qualifier: sedated/paralyzed",
    "gcsq_eye_obstruction_recovered" =
      "GCS qualifier: eye obstruction",
    "gcsq_unknown_recovered" =
      "GCS qualifier: unknown",
    "pupil_clean" =
      "Pupillary response",
    "sbp_clean" =
      "Systolic blood pressure",
    "pulse_clean" =
      "Pulse",
    "rr_clean" =
      "Respiratory rate",
    "spo2_clean" =
      "Oxygen saturation",
    "temperature_c_recovered" =
      "Temperature",
    "supplemental_oxygen_recovered" =
      "Supplemental oxygen",
    "mechanism_clean" =
      "Mechanism",
    "helmet_use_recovered" =
      "Helmet documentation",
    "respiratoryassistance_clean" =
      "Respiratory assistance",
    "dx_concussion" =
      "Concussion",
    "dx_cerebral_edema_traumatic" =
      "Traumatic cerebral edema",
    "dx_diffuse_axonal_injury" =
      "Diffuse axonal injury",
    "dx_focal_contusion_or_iph" =
      "Contusion / IPH",
    "dx_epidural_hematoma" =
      "Epidural hematoma",
    "dx_subdural_hematoma" =
      "Subdural hematoma",
    "dx_subarachnoid_hemorrhage" =
      "Traumatic SAH",
    "dx_other_intracranial_injury" =
      "Other intracranial injury",
    "dx_cranial_skull_fracture" =
      "Cranial skull fracture",
    "dx_facial_fracture" =
      "Facial fracture",
    "dx_other_skull_or_facial_fracture" =
      "Other skull/facial fracture",
    "dx_spinal_cord_injury" =
      "Spinal cord injury",
    "dx_neck_vascular_injury" =
      "Neck vascular injury",
    "dx_thoracic_injury" =
      "Thoracic injury",
    "dx_abdominal_pelvic_injury" =
      "Abdominal/pelvic injury",
    "dx_upper_extremity_injury" =
      "Upper-extremity injury",
    "dx_lower_extremity_injury" =
      "Lower-extremity injury",
    "pmhx_bleeding_disorder" =
      "Bleeding disorder",
    "pmhx_anticoagulant_therapy" =
      "Anticoagulant therapy",
    "pmhx_copd" =
      "COPD",
    "pmhx_diabetes" =
      "Diabetes",
    "pmhx_hypertension" =
      "Hypertension",
    "pmhx_current_smoker" =
      "Current smoking",
    "pmhx_functional_dependence" =
      "Functional dependence",
    "pmhx_dementia" =
      "Dementia",
    "pmhx_chf" =
      "Congestive heart failure",
    "pmhx_chronic_renal_failure" =
      "Chronic renal failure",
    "pmhx_cirrhosis" =
      "Cirrhosis",
    "pmhx_steroid_use" =
      "Steroid use"
  )

  out <- unname(
    known[
      x
    ]
  )

  missing <- is.na(
    out
  )

  out[
    missing
  ] <- gsub(
    "_",
    " ",
    x[
      missing
    ],
    fixed = TRUE
  )

  out
}

# =============================================================================
# FIGURE 1 — Prediction framework / temporal design / architecture
# =============================================================================

box_layer <- function(
    xmin,
    xmax,
    ymin,
    ymax,
    label,
    fill = "white",
    color = NAVY,
    text_color = NAVY,
    size = 3.0,
    fontface = "plain"
) {
  list(
    annotate(
      "rect",
      xmin = xmin,
      xmax = xmax,
      ymin = ymin,
      ymax = ymax,
      fill = fill,
      color = color,
      linewidth = 0.7
    ),
    annotate(
      "text",
      x = (
        xmin +
          xmax
      ) /
        2,
      y = (
        ymin +
          ymax
      ) /
        2,
      label = label,
      color = text_color,
      family = BASE_FONT,
      fontface = fontface,
      size = size,
      lineheight = 0.95
    )
  )
}

arrow_layer <- function(
    x,
    y,
    xend,
    yend,
    color = MID_GRAY
) {
  annotate(
    "segment",
    x = x,
    y = y,
    xend = xend,
    yend = yend,
    color = color,
    linewidth = 0.7,
    arrow =
      arrow(
        length = unit(
          0.12,
          "in"
        ),
        type = "closed"
      )
  )
}

# Panel A: prediction time
fig1_a <- ggplot() +
  box_layer(
    0.02,
    0.19,
    0.36,
    0.64,
    "Trauma-center\narrival",
    fill = VERY_LIGHT_GRAY,
    fontface = "bold"
  ) +
  box_layer(
    0.25,
    0.47,
    0.31,
    0.69,
    "Initial evaluation\n+ physiology\n+ diagnostic imaging",
    fill = "#EAF0F8",
    fontface = "bold"
  ) +
  box_layer(
    0.53,
    0.72,
    0.27,
    0.73,
    "TBI-TRACT\nprediction point",
    fill = "#D9E2F3",
    color = NAVY,
    text_color = NAVY,
    size = 3.3,
    fontface = "bold"
  ) +
  box_layer(
    0.78,
    0.98,
    0.31,
    0.69,
    "Future inpatient\nresource + disposition\ntrajectory",
    fill = "#FCE4D6",
    color = ORANGE,
    text_color = BLACK,
    fontface = "bold"
  ) +
  arrow_layer(
    0.19,
    0.50,
    0.25,
    0.50
  ) +
  arrow_layer(
    0.47,
    0.50,
    0.53,
    0.50
  ) +
  arrow_layer(
    0.72,
    0.50,
    0.78,
    0.50
  ) +
  annotate(
    "text",
    x = 0.625,
    y = 0.14,
    label =
      "Prediction occurs after the initial diagnostic workup but before subsequent inpatient utilization.",
    family = BASE_FONT,
    color = MID_GRAY,
    size = 2.7
  ) +
  coord_cartesian(
    xlim = c(
      0,
      1
    ),
    ylim = c(
      0,
      1
    ),
    clip = "off"
  ) +
  theme_void(
    base_family = BASE_FONT
  ) +
  labs(
    title =
      "A. Intended prediction time"
  ) +
  theme(
    plot.title =
      element_text(
        face = "bold",
        color = NAVY,
        size = 11,
        hjust = 0
      )
  )

# Panel B: rolling-origin design
fig1_b <- ggplot() +
  # Uniform node geometry across all rolling-origin rows.
  box_layer(
    0.02,
    0.20,
    0.68,
    0.88,
    "Train\n2020",
    fill = "#EAF0F8",
    fontface = "bold"
  ) +
  box_layer(
    0.28,
    0.46,
    0.68,
    0.88,
    "Tune\n2021",
    fill = "#FFF2CC",
    color = GOLD,
    text_color = BLACK,
    fontface = "bold"
  ) +
  box_layer(
    0.54,
    0.72,
    0.68,
    0.88,
    "Evaluate\n2022",
    fill = "#E2F0D9",
    color = GREEN,
    text_color = BLACK,
    fontface = "bold"
  ) +
  arrow_layer(
    0.20,
    0.78,
    0.28,
    0.78
  ) +
  arrow_layer(
    0.46,
    0.78,
    0.54,
    0.78
  ) +
  box_layer(
    0.02,
    0.20,
    0.39,
    0.59,
    "Train\n2020\u20132021",
    fill = "#EAF0F8",
    fontface = "bold"
  ) +
  box_layer(
    0.28,
    0.46,
    0.39,
    0.59,
    "Tune\n2022",
    fill = "#FFF2CC",
    color = GOLD,
    text_color = BLACK,
    fontface = "bold"
  ) +
  box_layer(
    0.54,
    0.72,
    0.39,
    0.59,
    "Evaluate\n2023",
    fill = "#E2F0D9",
    color = GREEN,
    text_color = BLACK,
    fontface = "bold"
  ) +
  arrow_layer(
    0.20,
    0.49,
    0.28,
    0.49
  ) +
  arrow_layer(
    0.46,
    0.49,
    0.54,
    0.49
  ) +
  box_layer(
    0.02,
    0.20,
    0.10,
    0.30,
    "Train\n2020\u20132022",
    fill = "#EAF0F8",
    fontface = "bold"
  ) +
  box_layer(
    0.28,
    0.46,
    0.10,
    0.30,
    "Tune\n2023",
    fill = "#FFF2CC",
    color = GOLD,
    text_color = BLACK,
    fontface = "bold"
  ) +
  box_layer(
    0.54,
    0.72,
    0.10,
    0.30,
    "Evaluate\n2024",
    fill = "#E2F0D9",
    color = GREEN,
    text_color = BLACK,
    fontface = "bold"
  ) +
  arrow_layer(
    0.20,
    0.20,
    0.28,
    0.20
  ) +
  arrow_layer(
    0.46,
    0.20,
    0.54,
    0.20
  ) +
  box_layer(
    0.80,
    0.98,
    0.31,
    0.67,
    "LOCKED MODEL\n\nFit on all\n2020\u20132024\n\nFuture external\nvalidation",
    fill = "#D9E2F3",
    fontface = "bold"
  ) +
  arrow_layer(
    0.72,
    0.78,
    0.80,
    0.67,
    color = NAVY
  ) +
  arrow_layer(
    0.72,
    0.49,
    0.80,
    0.49,
    color = NAVY
  ) +
  arrow_layer(
    0.72,
    0.20,
    0.80,
    0.31,
    color = NAVY
  ) +
  coord_cartesian(
    xlim = c(
      0,
      1
    ),
    ylim = c(
      0,
      1
    ),
    clip = "off"
  ) +
  theme_void(
    base_family = BASE_FONT
  ) +
  labs(
    title =
      "B. Rolling-origin temporal evaluation"
  ) +
  theme(
    plot.title =
      element_text(
        face = "bold",
        color = NAVY,
        size = 11,
        hjust = 0
      )
  )

# Panel C: predictor policy + outputs
fig1_c <- ggplot() +
  box_layer(
    0.02,
    0.30,
    0.33,
    0.67,
    "Pragmatic clinical backbone\n\n46 admission-era predictors\n\nHelmet + respiratory assistance removed",
    fill = "#D9E2F3",
    size = 2.8,
    fontface = "bold"
  ) +
  box_layer(
    0.38,
    0.62,
    0.56,
    0.90,
    "System-mediated outcomes\n\nDisposition\nHLOS trajectory\n\n+ race + ethnicity + payer",
    fill = "#EAF0F8",
    color = BLUE,
    text_color = BLACK,
    size = 2.7,
    fontface = "bold"
  ) +
  box_layer(
    0.38,
    0.62,
    0.10,
    0.44,
    "Clinical-only outputs\n\nICU trajectory\nVentilation trajectory\nICP monitoring\nCraniotomy / craniectomy",
    fill = "#E2F0D9",
    color = GREEN,
    text_color = BLACK,
    size = 2.6,
    fontface = "bold"
  ) +
  arrow_layer(
    0.30,
    0.53,
    0.38,
    0.73,
    color = BLUE
  ) +
  arrow_layer(
    0.30,
    0.47,
    0.38,
    0.27,
    color = GREEN
  ) +
  box_layer(
    0.70,
    0.98,
    0.56,
    0.90,
    "Probability trajectories\n\nDisposition: 3 classes\nHLOS: \u22647 / 8\u201327 / \u226528 d\nICU: none / 1\u20137 / \u22658 d\nVentilation: none / 1\u20137 / \u22658 d\nICP + craniotomy: probabilities",
    fill = "white",
    color = ORANGE,
    text_color = BLACK,
    size = 2.5
  ) +
  box_layer(
    0.70,
    0.98,
    0.10,
    0.44,
    "Continuous trajectories\n\nHospital LOS\nICU LOS among ICU users\nVentilator duration among ventilated patients\n\nMedian + estimated Q10\u2013Q90 range",
    fill = "white",
    color = PURPLE,
    text_color = BLACK,
    size = 2.5
  ) +
  arrow_layer(
    0.62,
    0.73,
    0.70,
    0.73,
    color = ORANGE
  ) +
  arrow_layer(
    0.62,
    0.27,
    0.70,
    0.27,
    color = PURPLE
  ) +
  coord_cartesian(
    xlim = c(
      0,
      1
    ),
    ylim = c(
      0,
      1
    ),
    clip = "off"
  ) +
  theme_void(
    base_family = BASE_FONT
  ) +
  labs(
    title =
      "C. Final hybrid architecture and endpoint-specific predictor policy"
  ) +
  theme(
    plot.title =
      element_text(
        face = "bold",
        color = NAVY,
        size = 11,
        hjust = 0
      )
  )

figure_1 <-
  fig1_a /
    fig1_b /
    fig1_c +
  plot_layout(
    heights = c(
      0.72,
      1.10,
      1.05
    )
  ) +
  plot_annotation(
    title =
      "TBI-TRACT: prediction framework, temporal development, and final architecture",
    theme =
      theme(
        plot.title =
          element_text(
            family = BASE_FONT,
            face = "bold",
            color = NAVY,
            size = 14,
            hjust = 0
          )
      )
  )

save_figure(
  figure_1,
  "Figure_1_TBI_TRACT_Framework",
  width = 10.5,
  height = 10.0
)

# =============================================================================
# FIGURE 2 — Temporal performance fingerprint
# =============================================================================

figure2_data <- temporal[
  target %in%
    target_order
]

figure2_data[
  ,
  domain :=
    domain_for_target(
      target
    )
]

figure2_data[
  ,
  display_target :=
    factor(
      target_labels[
        target
      ],
      levels =
        rev(
          unname(
            target_labels[
              target_order
            ]
          )
        )
    )
]

summary2 <- figure2_data[
  ,
  .(
    AUROC_temporal_min =
      min(
        AUROC
      ),
    AUROC_temporal_max =
      max(
        AUROC
      ),
    PR_lift_temporal_min =
      min(
        prevalence_lift
      ),
    PR_lift_temporal_max =
      max(
        prevalence_lift
      ),
    prevalence_2024 =
      prevalence[
        year ==
          2024
      ][1L],
    AUROC_2024 =
      AUROC[
        year ==
          2024
      ][1L],
    PR_lift_2024 =
      prevalence_lift[
        year ==
          2024
      ][1L],
    domain =
      domain[1L]
  ),
  by = .(
    target,
    display_target
  )
]

ci2 <- headline_ci[
  target %in%
    target_order
]

summary2 <- merge(
  summary2,
  ci2[
    ,
    .(
      target,
      AUROC_low,
      AUROC_high,
      AUPRC_low,
      AUPRC_high,
      prevalence
    )
  ],
  by = "target",
  all.x = TRUE
)

summary2[
  ,
  `:=`(
    PR_lift_low =
      AUPRC_low /
        prevalence,
    PR_lift_high =
      AUPRC_high /
        prevalence
  )
]

summary2[
  ,
  domain :=
    factor(
      domain,
      levels =
        names(
          domain_palette
        )
    )
]

p2a <- ggplot(
  summary2,
  aes(
    y = display_target,
    color = domain,
    shape = domain
  )
) +
  geom_segment(
    aes(
      x = AUROC_temporal_min,
      xend = AUROC_temporal_max,
      yend = display_target
    ),
    linewidth = 2.5,
    alpha = 0.22
  ) +
  geom_segment(
    aes(
      x = AUROC_low,
      xend = AUROC_high,
      yend = display_target
    ),
    linewidth = 0.8
  ) +
  geom_point(
    aes(
      x = AUROC_2024
    ),
    size = 2.8,
    fill = "white",
    stroke = 1.0
  ) +
  scale_color_manual(
    values = domain_palette
  ) +
  scale_shape_manual(
    values = domain_shapes
  ) +
  scale_x_continuous(
    limits = c(
      0.72,
      0.97
    ),
    breaks = seq(
      0.75,
      0.95,
      by = 0.05
    )
  ) +
  labs(
    title =
      "A. Discrimination",
    subtitle =
      "2024 estimate (95% CI)\nPale bar = 2022\u20132024 range",
    x =
      "AUROC",
    y =
      NULL,
    color =
      NULL,
    shape =
      NULL
  ) +
  theme_tract(
    9
  ) +
  theme(
    legend.position =
      "bottom"
  )

p2b <- ggplot(
  summary2,
  aes(
    y = display_target,
    color = domain,
    shape = domain
  )
) +
  geom_segment(
    aes(
      x = PR_lift_temporal_min,
      xend = PR_lift_temporal_max,
      yend = display_target
    ),
    linewidth = 2.5,
    alpha = 0.22
  ) +
  geom_segment(
    aes(
      x = PR_lift_low,
      xend = PR_lift_high,
      yend = display_target
    ),
    linewidth = 0.8
  ) +
  geom_point(
    aes(
      x = PR_lift_2024
    ),
    size = 2.8,
    fill = "white",
    stroke = 1.0
  ) +
  scale_color_manual(
    values = domain_palette
  ) +
  scale_shape_manual(
    values = domain_shapes
  ) +
  guides(
    color = guide_legend(
      nrow = 1,
      byrow = TRUE
    ),
    shape = guide_legend(
      nrow = 1,
      byrow = TRUE
    )
  ) +
  scale_x_continuous(
    labels =
      function(x) {
        paste0(
          x,
          "\u00d7"
        )
      }
  ) +
  labs(
    title =
      "B. Precision-recall performance",
    subtitle =
      "AUPRC \u00f7 prevalence\n1\u00d7 = no-skill baseline",
    x =
      "Precision-recall lift",
    y =
      NULL,
    color =
      NULL,
    shape =
      NULL
  ) +
  theme_tract(
    9
  ) +
  theme(
    axis.text.y =
      element_blank(),
    legend.position =
      "bottom"
  )

figure_2 <-
  (
    p2a +
      p2b +
      plot_layout(
        widths = c(
          1.08,
          1
        ),
        guides = "collect"
      )
  ) &
    theme(
      legend.position = "bottom",
      legend.box = "horizontal"
    )

figure_2 <-
  figure_2 +
    plot_annotation(
      title =
        "Temporal predictive performance of clinically salient TBI-TRACT outputs",
      theme =
        theme(
          plot.title =
            element_text(
              family = BASE_FONT,
              face = "bold",
              color = NAVY,
              size = 14,
              hjust = 0
            )
        )
    )

save_figure(
  figure_2,
  "Figure_2_Temporal_Performance_Fingerprint",
  width = 11.0,
  height = 6.2
)

# =============================================================================
# FIGURE 3A — Curated 2024 probability calibration
# =============================================================================

calibration_targets <- c(
  "Post-acute facility",
  "Death/hospice",
  "Any ICU",
  "ICU >=8 days",
  "Any ventilation",
  "Ventilation >=8 days",
  "Hospital LOS >=28 days",
  "Invasive ICP monitoring",
  "Craniotomy/craniectomy"
)

make_calibration_panel <- function(
    target_name,
    show_legend = FALSE
) {
  curve <- calibration[
    target ==
      target_name
  ][
    order(
      predicted_probability
    )
  ]

  points <- calibration_points[
    target ==
      target_name
  ][
    order(
      mean_predicted
    )
  ]

  if (
    nrow(curve) == 0L
  ) {
    stop(
      "No calibration curve found for target: ",
      target_name,
      call. = FALSE
    )
  }

  metric <- headline_ci[
    target ==
      target_name
  ]

  upper <- max(
    c(
      curve$predicted_probability,
      curve$calibrated_probability,
      curve$upper_95,
      points$mean_predicted,
      points$observed
    ),
    na.rm = TRUE
  )

  upper <- min(
    1,
    max(
      0.12,
      upper *
        1.05
    )
  )

  domain <- domain_for_target(
    target_name
  )

  p <- ggplot() +
    geom_abline(
      intercept = 0,
      slope = 1,
      color = MID_GRAY,
      linetype = 2,
      linewidth = 0.6
    ) +
    geom_ribbon(
      data = curve,
      aes(
        x = predicted_probability,
        ymin = lower_95,
        ymax = upper_95
      ),
      fill = domain_palette[
        domain
      ],
      alpha = 0.13
    ) +
    geom_line(
      data = curve,
      aes(
        x = predicted_probability,
        y = calibrated_probability
      ),
      color = domain_palette[
        domain
      ],
      linewidth = 0.9
    ) +
    geom_point(
      data = points,
      aes(
        x = mean_predicted,
        y = observed,
        size = N
      ),
      shape = 21,
      fill = "white",
      color = domain_palette[
        domain
      ],
      stroke = 0.7
    ) +
    annotate(
      "label",
      x = upper *
        0.04,
      y = upper *
        0.96,
      hjust = 0,
      vjust = 1,
      size = 2.25,
      family = BASE_FONT,
      label =
        paste0(
          "Intercept ",
          sprintf(
            "%+.2f",
            metric$calibration_intercept[1L]
          ),
          "\nSlope ",
          sprintf(
            "%.2f",
            metric$calibration_slope[1L]
          )
        ),
      label.size = 0.15,
      fill = alpha(
        "white",
        0.85
      ),
      color = BLACK
    ) +
    scale_x_continuous(
      limits = c(
        0,
        upper
      ),
      labels =
        label_percent(
          accuracy = 1
        ),
      expand =
        expansion(
          mult = c(
            0,
            0.02
          )
        )
    ) +
    scale_y_continuous(
      limits = c(
        0,
        upper
      ),
      labels =
        label_percent(
          accuracy = 1
        ),
      expand =
        expansion(
          mult = c(
            0,
            0.02
          )
        )
    ) +
    scale_size_continuous(
      range = c(
        1.8,
        5.0
      ),
      breaks =
        pretty_breaks(
          n = 3
        ),
      labels =
        label_comma(
          accuracy = 1
        )
    ) +
    coord_equal() +
    labs(
      title =
        unname(
          target_labels[
            target_name
          ]
        ),
      x =
        "Predicted",
      y =
        "Observed",
      size =
        "Patients per group"
    ) +
    theme_tract(
      7.7
    ) +
    theme(
      plot.title =
        element_text(
          size = 9,
          face = "bold",
          color = NAVY
        ),
      legend.position =
        if (
          show_legend
        ) {
          "bottom"
        } else {
          "none"
        }
    )

  p
}

cal_panels <- lapply(
  seq_along(
    calibration_targets
  ),
  function(i) {
    make_calibration_panel(
      calibration_targets[i],
      show_legend =
        i ==
          9L
    )
  }
)

figure_3a <-
  (
    wrap_plots(
      cal_panels,
      ncol = 3,
      guides = "collect"
    )
  ) &
    theme(
      legend.position = "bottom",
      legend.box = "horizontal"
    )

figure_3a <-
  figure_3a +
    plot_annotation(
      title =
        "Figure 3A. Calibration of clinically salient probabilities in the 2024 temporal cohort",
      subtitle =
        "Flexible calibration curve with 95% confidence band; grouped observations are visual anchors",
      theme =
        theme(
          plot.title =
            element_text(
              family = BASE_FONT,
              face = "bold",
              color = NAVY,
              size = 13
            ),
          plot.subtitle =
            element_text(
              family = BASE_FONT,
              color = MID_GRAY,
              size = 9
            )
        )
    )

save_figure(
  figure_3a,
  "Figure_3A_2024_Probability_Calibration",
  width = 10.5,
  height = 10.0
)

# =============================================================================
# FIGURE 3B — 2024 continuous trajectory calibration
# =============================================================================

duration_order <- c(
  "hospital_los",
  "icu_los_conditional",
  "ventilator_days_conditional"
)

duration_colors <- c(
  "hospital_los" =
    GOLD,
  "icu_los_conditional" =
    GREEN,
  "ventilator_days_conditional" =
    PURPLE
)

make_duration_panel <- function(duration_id_value) {
  
  d <- duration_calibration[
    get("duration_id") == duration_id_value
  ]
  
  if (nrow(d) == 0L) {
    stop(
      "No duration-calibration data found for: ",
      duration_id_value,
      call. = FALSE
    )
  }
  
  d <- copy(d)
  
  # Keep plotting order deterministic
  setorder(
    d,
    predicted_median
  )
  
  duration_label_value <- unique(d$duration_label)
  
  if (length(duration_label_value) != 1L) {
    stop(
      "Expected one duration label for ",
      duration_id_value,
      ".",
      call. = FALSE
    )
  }
  
  ggplot(
    d,
    aes(
      x = predicted_median,
      y = observed_median
    )
  ) +
    geom_abline(
      intercept = 0,
      slope = 1,
      linetype = "dashed",
      linewidth = 0.45
    ) +
    geom_errorbar(
      aes(
        ymin = observed_q10,
        ymax = observed_q90
      ),
      width = 0,
      linewidth = 0.45
    ) +
    geom_errorbarh(
      aes(
        xmin = predicted_q10,
        xmax = predicted_q90
      ),
      height = 0,
      linewidth = 0.45
    ) +
    geom_point(
      size = 1.8
    ) +
    labs(
      title = duration_label_value,
      x = "Predicted duration, days",
      y = "Observed duration, days"
    ) +
    coord_equal() +
    theme_classic(
      base_family = "Arial"
    ) +
    theme(
      plot.title = element_text(
        size = 9,
        face = "bold"
      ),
      axis.title = element_text(
        size = 8
      ),
      axis.text = element_text(
        size = 7
      ),
      plot.subtitle = element_text(
        size = 7
      )
    )
}

duration_panels <- lapply(
  duration_order,
  make_duration_panel
)

figure_3b <-
  wrap_plots(
    duration_panels,
    nrow = 1
  ) +
  plot_annotation(
    title =
      "Figure 3B. Calibration of continuous resource-duration predictions in 2024",
    subtitle =
      "Each point represents an equal-frequency group defined by predicted median duration",
    theme =
      theme(
        plot.title =
          element_text(
            family = BASE_FONT,
            face = "bold",
            color = NAVY,
            size = 13
          ),
        plot.subtitle =
          element_text(
            family = BASE_FONT,
            color = MID_GRAY,
            size = 9
          )
      )
  )

save_figure(
  figure_3b,
  "Figure_3B_2024_Continuous_Trajectory_Calibration",
  width = 12.0,
  height = 4.4
)

# =============================================================================
# eFIGURE 1 — Cohort flow + year-specific observation status
# =============================================================================

flow[
  ,
  status_label :=
    fcase(
      observation_status ==
        "Retained complete trajectory",
      "Retained complete trajectory",
      observation_status ==
        "Future transfer-out",
      "Future transfer-out",
      observation_status ==
        "Future AMA",
      "Future AMA",
      observation_status ==
        "Future ED-other/institutional/custody",
      "Future institutional / custody / other",
      default =
        observation_status
    )
]

flow_totals <- flow[
  ,
  .(
    N =
      sum(
        N
      )
  ),
  by = status_label
]

grand_total <- sum(
  flow_totals$N
)

retained_total <- flow_totals[
  status_label ==
    "Retained complete trajectory",
  N
]

exclusion_total <- grand_total -
  retained_total

node_y <- c(
  "Retained complete trajectory" =
    0.78,
  "Future transfer-out" =
    0.55,
  "Future AMA" =
    0.32,
  "Future institutional / custody / other" =
    0.10
)

status_colors <- c(
  "Retained complete trajectory" =
    NAVY,
  "Future transfer-out" =
    ORANGE,
  "Future AMA" =
    RED,
  "Future institutional / custody / other" =
    MID_GRAY
)

flow_a <- ggplot() +
  box_layer(
    0.02,
    0.30,
    0.37,
    0.63,
    paste0(
      "Direct presentations\n2020-2024\n\nN = ",
      comma(
        grand_total
      )
    ),
    fill = "#D9E2F3",
    size = 3.2,
    fontface = "bold"
  )

for (
  s in names(
    node_y
  )
) {
  n_s <- flow_totals[
    status_label ==
      s,
    N
  ]

  pct_s <- n_s /
    grand_total

  y_mid <- node_y[
    s
  ]

  flow_a <- flow_a +
    annotate(
      "rect",
      xmin = 0.62,
      xmax = 0.98,
      ymin = y_mid -
        0.085,
      ymax = y_mid +
        0.085,
      fill = alpha(
        status_colors[
          s
        ],
        0.11
      ),
      color = status_colors[
        s
      ],
      linewidth = 0.7
    ) +
    annotate(
      "text",
      x = 0.80,
      y = y_mid,
      label =
        paste0(
          if (
            s ==
              "Future institutional / custody / other"
          ) {
            "Future institutional /\ncustody / other"
          } else {
            s
          },
          "\n",
          comma(
            n_s
          ),
          " (",
          percent(
            pct_s,
            accuracy = 0.1
          ),
          ")"
        ),
      family = BASE_FONT,
      color = BLACK,
      size = 2.7,
      fontface =
        if (
          s ==
            "Retained complete trajectory"
        ) {
          "bold"
        } else {
          "plain"
        }
    ) +
    annotate(
      "segment",
      x = 0.30,
      y = 0.50,
      xend = 0.62,
      yend = y_mid,
      color = status_colors[
        s
      ],
      linewidth = max(
        0.45,
        2.5 *
          pct_s
      ),
      alpha = 0.65,
      arrow =
        arrow(
          length = unit(
            0.08,
            "in"
          ),
          type = "closed"
        )
    )
}

flow_a <- flow_a +
  coord_cartesian(
    xlim = c(
      0,
      1
    ),
    ylim = c(
      0,
      1
    ),
    clip = "off"
  ) +
  theme_void(
    base_family = BASE_FONT
  ) +
  labs(
    title =
      "A. Observation of the index-hospital trajectory"
  ) +
  theme(
    plot.title =
      element_text(
        face = "bold",
        color = NAVY,
        size = 11
      )
  )

flow[
  ,
  year_total :=
    sum(
      N
    ),
  by = admission_year
]

flow[
  ,
  proportion :=
    N /
      year_total
]

flow[
  ,
  status_label :=
    factor(
      status_label,
      levels = c(
        "Retained complete trajectory",
        "Future transfer-out",
        "Future AMA",
        "Future institutional / custody / other"
      )
    )
]

flow_b <- ggplot(
  flow,
  aes(
    x = factor(
      admission_year
    ),
    y = proportion,
    fill = status_label
  )
) +
  geom_col(
    width = 0.72
  ) +
  scale_fill_manual(
    values = status_colors
  ) +
  guides(
    fill =
      guide_legend(
        nrow = 2,
        byrow = TRUE
      )
  ) +
  scale_y_continuous(
    labels =
      label_percent(
        accuracy = 1
      ),
    expand =
      expansion(
        mult = c(
          0,
          0.02
        )
      )
  ) +
  labs(
    title =
      "B. Observation status by admission year",
    x =
      "Admission year",
    y =
      "Proportion of direct presentations",
    fill =
      NULL
  ) +
  theme_tract(
    9
  ) +
  theme(
    legend.position =
      "bottom",
    panel.grid.major.x =
      element_blank()
  )

efigure_1 <-
  (
    flow_a +
      flow_b +
      plot_layout(
        widths = c(
          1.05,
          1
        ),
        guides = "collect"
      )
  ) &
    theme(
      legend.position = "bottom",
      legend.box = "horizontal"
    )

efigure_1 <-
  efigure_1 +
    plot_annotation(
    title =
      "eFigure 1. Cohort-flow and trajectory-observation restriction",
    theme =
      theme(
        plot.title =
          element_text(
            family = BASE_FONT,
            face = "bold",
            color = NAVY,
            size = 14
          )
      )
  )

save_figure(
  efigure_1,
  "eFigure_1_Cohort_Flow",
  width = 11.0,
  height = 6.0
)

# =============================================================================
# eFIGURE 2 — Case-mix drift love plot
# =============================================================================

drift_plot <- drift[
  predictor_type %in%
    c(
      "numeric",
      "categorical_overall"
    )
]

drift_plot[
  ,
  display_predictor :=
    pretty_predictor(
      predictor
    )
]

# Retain all conventionally notable changes plus enough leading rows to show
# the structure of smaller drift without creating a 46-row wall.
top_predictors <- unique(
  c(
    drift_plot[
      abs_smd >=
        0.05,
      predictor
    ],
    drift_plot[
      order(
        -abs_smd
      )
    ][
      seq_len(
        min(
          25L,
          .N
        )
      ),
      predictor
    ]
  )
)

drift_plot <- drift_plot[
  predictor %in%
    top_predictors
][
  order(
    abs_smd
  )
]

drift_plot[
  ,
  display_predictor :=
    factor(
      display_predictor,
      levels =
        display_predictor
    )
]

drift_plot[
  ,
  notable :=
    abs_smd >=
      0.10
]

efigure_2 <- ggplot(
  drift_plot,
  aes(
    x = abs_smd,
    y = display_predictor
  )
) +
  geom_vline(
    xintercept = 0.10,
    linetype = 2,
    color = RED,
    linewidth = 0.7
  ) +
  geom_segment(
    aes(
      x = 0,
      xend = abs_smd,
      yend = display_predictor
    ),
    color = LIGHT_GRAY,
    linewidth = 0.7
  ) +
  geom_point(
    aes(
      fill = notable
    ),
    shape = 21,
    size = 2.8,
    color = NAVY,
    stroke = 0.5
  ) +
  scale_fill_manual(
    values = c(
      "FALSE" =
        "white",
      "TRUE" =
        NAVY
    ),
    guide = "none"
  ) +
  scale_x_continuous(
    limits = c(
      0,
      max(
        0.16,
        max(
          drift_plot$abs_smd
        ) *
          1.08
      )
    )
  ) +
  labs(
    title =
      "eFigure 2. Temporal case-mix drift from 2020-2023 to 2024",
    subtitle =
      "Absolute standardized mean differences; dashed reference denotes |SMD| = 0.10",
    x =
      "Absolute standardized mean difference",
    y =
      NULL,
    caption =
      "Categorical predictors are represented by the maximum level-specific SMD."
  ) +
  theme_tract(
    9
  ) +
  theme(
    legend.position =
      "none"
  )

save_figure(
  efigure_2,
  "eFigure_2_Case_Mix_Drift_Love_Plot",
  width = 8.0,
  height = 7.2
)

# =============================================================================
# eFIGURE 3 — Complete 2024 calibration grid
# =============================================================================

# This supplement intentionally computes curves from the already-saved locked
# prediction cache so every probability output can be displayed, not only the
# curated headline outputs in Figure 3A.

prediction_cache_files <- c(
  discharge_3cat_final =
    find_file(
      "TEST_2024__discharge_3cat_final__final_predictions.rds",
      "prediction_cache"
    ),
  icu_trajectory_final =
    find_file(
      "TEST_2024__icu_trajectory_final__final_predictions.rds",
      "prediction_cache"
    ),
  ventilation_trajectory_final =
    find_file(
      "TEST_2024__ventilation_trajectory_final__final_predictions.rds",
      "prediction_cache"
    ),
  hlos_trajectory_final =
    find_file(
      "TEST_2024__hlos_trajectory_final__final_predictions.rds",
      "prediction_cache"
    ),
  icp_pressure_monitor_final =
    find_file(
      "TEST_2024__icp_pressure_monitor_final__final_predictions.rds",
      "prediction_cache"
    ),
  craniotomy_craniectomy_final =
    find_file(
      "TEST_2024__craniotomy_craniectomy_final__final_predictions.rds",
      "prediction_cache"
    )
)

full_endpoint_specs <- list(
  discharge_3cat_final = list(
    label =
      "Disposition",
    classes = c(
      "Home/home health",
      "Post-acute facility",
      "Death/hospice"
    ),
    derived = NULL
  ),
  icu_trajectory_final = list(
    label =
      "ICU",
    classes = c(
      "No ICU",
      "ICU 1-7 days",
      "ICU >=8 days"
    ),
    derived = list(
      label =
        "Any ICU",
      reference =
        "No ICU"
    )
  ),
  ventilation_trajectory_final = list(
    label =
      "Ventilation",
    classes = c(
      "No ventilation",
      "Ventilation 1-7 days",
      "Ventilation >=8 days"
    ),
    derived = list(
      label =
        "Any ventilation",
      reference =
        "No ventilation"
    )
  ),
  hlos_trajectory_final = list(
    label =
      "Hospital LOS",
    classes = c(
      "Hospital LOS <=7 days",
      "Hospital LOS 8-27 days",
      "Hospital LOS >=28 days"
    ),
    derived = NULL
  ),
  icp_pressure_monitor_final = list(
    label =
      "Neurosurgical resources",
    classes = NULL,
    binary_label =
      "Invasive ICP monitoring"
  ),
  craniotomy_craniectomy_final = list(
    label =
      "Neurosurgical resources",
    classes = NULL,
    binary_label =
      "Craniotomy/craniectomy"
  )
)

clamp_probability <- function(
    p,
    eps = 1e-6
) {
  pmin(
    pmax(
      as.numeric(
        p
      ),
      eps
    ),
    1 -
      eps
  )
}

flexible_curve_from_predictions <- function(
    y,
    p,
    n_grid = 120L
) {
  keep <- !is.na(
    y
  ) &
    !is.na(
      p
    ) &
    is.finite(
      p
    )

  y <- as.integer(
    y[
      keep
    ]
  )

  p <- clamp_probability(
    p[
      keep
    ]
  )

  lp <- qlogis(
    p
  )

  fit <- glm(
    y ~ splines::ns(
      lp,
      df = 4
    ),
    family = binomial()
  )

  q <- quantile(
    p,
    probs = c(
      0.005,
      0.995
    ),
    na.rm = TRUE
  )

  grid_p <- seq(
    q[1L],
    q[2L],
    length.out = n_grid
  )

  pred <- predict(
    fit,
    newdata =
      data.frame(
        lp =
          qlogis(
            clamp_probability(
              grid_p
            )
          )
      ),
    type = "link",
    se.fit = TRUE
  )

  data.table(
    predicted_probability =
      grid_p,
    calibrated_probability =
      plogis(
        pred$fit
      ),
    lower_95 =
      plogis(
        pred$fit -
          1.96 *
            pred$se.fit
      ),
    upper_95 =
      plogis(
        pred$fit +
          1.96 *
            pred$se.fit
      )
  )
}

grouped_calibration_points <- function(
    y,
    p,
    groups = 10L
) {
  d <- data.table(
    y =
      as.integer(
        y
      ),
    p =
      as.numeric(
        p
      )
  )

  d <- d[
    !is.na(
      y
    ) &
      !is.na(
        p
      ) &
      is.finite(
        p
      )
  ]

  d[
    ,
    group :=
      pmin(
        groups,
        pmax(
          1L,
          ceiling(
            groups *
              frank(
                p,
                ties.method = "average"
              ) /
              .N
          )
        )
      )
  ]

  d[
    ,
    .(
      N =
        .N,
      mean_predicted =
        mean(
          p
        ),
      observed =
        mean(
          y
        )
    ),
    by = group
  ]
}

full_calibration_list <- list()
full_points_list <- list()

for (
  endpoint_id in names(
    prediction_cache_files
  )
) {
  cache <- as.data.table(
    readRDS(
      prediction_cache_files[
        endpoint_id
      ]
    )
  )

  spec <- full_endpoint_specs[[
    endpoint_id
  ]]

  if (
    endpoint_id %in%
      c(
        "icp_pressure_monitor_final",
        "craniotomy_craniectomy_final"
      )
  ) {
    y <- as.integer(
      cache$observed
    )

    p <- as.numeric(
      cache$predicted_probability
    )

    target_name <- spec$binary_label

    curve <- flexible_curve_from_predictions(
      y,
      p
    )

    points <- grouped_calibration_points(
      y,
      p
    )

    curve[
      ,
      `:=`(
        endpoint_id =
          endpoint_id,
        domain =
          spec$label,
        target =
          target_name
      )
    ]

    points[
      ,
      `:=`(
        endpoint_id =
          endpoint_id,
        domain =
          spec$label,
        target =
          target_name
      )
    ]

    full_calibration_list[[
      length(
        full_calibration_list
      ) +
        1L
    ]] <- curve

    full_points_list[[
      length(
        full_points_list
      ) +
        1L
    ]] <- points

  } else {
    for (
      class_label in spec$classes
    ) {
      prob_column <- paste0(
        "p__",
        make.names(
          class_label,
          unique = TRUE
        )
      )

      if (
        !prob_column %in%
          names(
            cache
          )
      ) {
        stop(
          "Missing probability column ",
          prob_column,
          " in ",
          endpoint_id,
          call. = FALSE
        )
      }

      y <- as.integer(
        as.character(
          cache$observed
        ) ==
          class_label
      )

      p <- as.numeric(
        cache[[
          prob_column
        ]]
      )

      curve <- flexible_curve_from_predictions(
        y,
        p
      )

      points <- grouped_calibration_points(
        y,
        p
      )

      curve[
        ,
        `:=`(
          endpoint_id =
            endpoint_id,
          domain =
            spec$label,
          target =
            class_label
        )
      ]

      points[
        ,
        `:=`(
          endpoint_id =
            endpoint_id,
          domain =
            spec$label,
          target =
            class_label
        )
      ]

      full_calibration_list[[
        length(
          full_calibration_list
        ) +
          1L
      ]] <- curve

      full_points_list[[
        length(
          full_points_list
        ) +
          1L
      ]] <- points
    }

    if (
      !is.null(
        spec$derived
      )
    ) {
      reference <- spec$derived$reference

      prob_column <- paste0(
        "p__",
        make.names(
          reference,
          unique = TRUE
        )
      )

      y <- as.integer(
        as.character(
          cache$observed
        ) !=
          reference
      )

      p <- 1 -
        as.numeric(
          cache[[
            prob_column
          ]]
        )

      target_name <- spec$derived$label

      curve <- flexible_curve_from_predictions(
        y,
        p
      )

      points <- grouped_calibration_points(
        y,
        p
      )

      curve[
        ,
        `:=`(
          endpoint_id =
            endpoint_id,
          domain =
            spec$label,
          target =
            target_name
        )
      ]

      points[
        ,
        `:=`(
          endpoint_id =
            endpoint_id,
          domain =
            spec$label,
          target =
            target_name
        )
      ]

      full_calibration_list[[
        length(
          full_calibration_list
        ) +
          1L
      ]] <- curve

      full_points_list[[
        length(
          full_points_list
        ) +
          1L
      ]] <- points
    }
  }
}

full_calibration <- rbindlist(
  full_calibration_list,
  fill = TRUE
)

full_points <- rbindlist(
  full_points_list,
  fill = TRUE
)

all_calibration_targets <- unique(
  full_calibration$target
)

make_full_cal_panel <- function(
    target_name
) {
  curve <- full_calibration[
    target ==
      target_name
  ]

  points <- full_points[
    target ==
      target_name
  ]

  domain <- curve$domain[1L]

  color <- fcase(
    domain ==
      "Disposition",
    BLUE,
    domain ==
      "Hospital LOS",
    GOLD,
    domain ==
      "ICU",
    GREEN,
    domain ==
      "Ventilation",
    PURPLE,
    domain ==
      "Neurosurgical resources",
    ORANGE,
    default =
      NAVY
  )

  upper <- max(
    c(
      curve$predicted_probability,
      curve$calibrated_probability,
      curve$upper_95,
      points$mean_predicted,
      points$observed
    ),
    na.rm = TRUE
  )

  upper <- min(
    1,
    max(
      0.10,
      upper *
        1.05
    )
  )

  ggplot() +
    geom_abline(
      intercept = 0,
      slope = 1,
      color = MID_GRAY,
      linetype = 2,
      linewidth = 0.45
    ) +
    geom_ribbon(
      data = curve,
      aes(
        x = predicted_probability,
        ymin = lower_95,
        ymax = upper_95
      ),
      fill = color,
      alpha = 0.12
    ) +
    geom_line(
      data = curve,
      aes(
        x = predicted_probability,
        y = calibrated_probability
      ),
      color = color,
      linewidth = 0.7
    ) +
    geom_point(
      data = points,
      aes(
        x = mean_predicted,
        y = observed
      ),
      shape = 21,
      fill = "white",
      color = color,
      size = 1.6,
      stroke = 0.5
    ) +
    scale_x_continuous(
      limits = c(
        0,
        upper
      ),
      labels =
        label_percent(
          accuracy = 1
        )
    ) +
    scale_y_continuous(
      limits = c(
        0,
        upper
      ),
      labels =
        label_percent(
          accuracy = 1
        )
    ) +
    coord_equal() +
    labs(
      title =
        publication_label(
          gsub(
            "Hospital LOS ",
            "HLOS ",
            target_name,
            fixed = TRUE
          )
        ),
      x =
        "Predicted",
      y =
        "Observed"
    ) +
    theme_tract(
      6.6
    ) +
    theme(
      legend.position =
        "none",
      plot.title =
        element_text(
          size = 7.4
        ),
      axis.title =
        element_text(
          size = 6.5
        ),
      axis.text =
        element_text(
          size = 5.8
        )
    )
}

full_cal_panels <- lapply(
  all_calibration_targets,
  make_full_cal_panel
)

efigure_3 <-
  wrap_plots(
    full_cal_panels,
    ncol = 4
  ) +
  plot_annotation(
    title =
      "eFigure 3. Complete 2024 calibration grid",
    subtitle =
      "Every multiclass probability, derived resource-use probability, and binary procedural endpoint",
    theme =
      theme(
        plot.title =
          element_text(
            family = BASE_FONT,
            face = "bold",
            color = NAVY,
            size = 14
          ),
        plot.subtitle =
          element_text(
            family = BASE_FONT,
            color = MID_GRAY,
            size = 9
          )
      )
  )

save_figure(
  efigure_3,
  "eFigure_3_Complete_2024_Calibration_Grid",
  width = 12.0,
  height = 12.0
)

# =============================================================================
# eFIGURE 4 — Duration quantile calibration over time
# =============================================================================

duration_long <- melt(
  duration_year,
  id.vars = c(
    "year",
    "duration_id",
    "duration_label"
  ),
  measure.vars = c(
    "empirical_q10",
    "empirical_q50",
    "empirical_q90"
  ),
  variable.name =
    "quantile",
  value.name =
    "empirical_fraction"
)

duration_long[
  ,
  nominal_fraction :=
    fcase(
      quantile ==
        "empirical_q10",
      0.10,
      quantile ==
        "empirical_q50",
      0.50,
      quantile ==
        "empirical_q90",
      0.90
    )
]

duration_long[
  ,
  year :=
    factor(
      year
    )
]

p4a <- ggplot(
  duration_long,
  aes(
    x = nominal_fraction,
    y = empirical_fraction,
    color = year,
    shape = year
  )
) +
  geom_abline(
    intercept = 0,
    slope = 1,
    color = MID_GRAY,
    linetype = 2,
    linewidth = 0.6
  ) +
  geom_line(
    aes(
      group = year
    ),
    linewidth = 0.6,
    alpha = 0.7
  ) +
  geom_point(
    size = 2.5
  ) +
  facet_wrap(
    ~ duration_label,
    nrow = 1
  ) +
  scale_x_continuous(
    breaks = c(
      0.10,
      0.50,
      0.90
    ),
    labels = c(
      "Q10",
      "Q50",
      "Q90"
    )
  ) +
  scale_y_continuous(
    limits = c(
      0,
      1
    ),
    labels =
      label_percent(
        accuracy = 1
      )
  ) +
  labs(
    title =
      "A. Quantile reliability",
    subtitle =
      "Observed fraction at or below each predicted quantile",
    x =
      "Predicted quantile",
    y =
      "Empirical cumulative probability",
    color =
      "Evaluation year",
    shape =
      "Evaluation year"
  ) +
  theme_tract(
    8
  ) +
  theme(
    legend.position = "bottom",
    legend.box = "horizontal"
  )

coverage_data <- copy(
  duration_year
)

coverage_data[
  ,
  year :=
    as.integer(
      year
    )
]

p4b <- ggplot(
  coverage_data,
  aes(
    x = year,
    y = central_80_coverage,
    color = duration_label,
    shape = duration_label
  )
) +
  geom_hline(
    yintercept = 0.80,
    linetype = 2,
    color = MID_GRAY,
    linewidth = 0.6
  ) +
  geom_line(
    linewidth = 0.7
  ) +
  geom_point(
    size = 2.7
  ) +
  scale_x_continuous(
    breaks = 2022:2024
  ) +
  scale_y_continuous(
    limits = c(
      0.70,
      0.84
    ),
    labels =
      label_percent(
        accuracy = 1
      )
  ) +
  labs(
    title =
      "B. Empirical Q10\u2013Q90 coverage",
    subtitle =
      "Dashed line = nominal central 80% reference",
    x =
      "Evaluation year",
    y =
      "Empirical coverage",
    color =
      NULL,
    shape =
      NULL
  ) +
  theme_tract(
    8
  ) +
  theme(
    legend.position = "bottom",
    legend.box = "horizontal"
  )

efigure_4 <-
  p4a /
    p4b +
  plot_layout(
    heights = c(
      1,
      0.85
    )
  ) +
  plot_annotation(
    title =
      "eFigure 4. Temporal calibration of continuous duration quantiles",
    theme =
      theme(
        plot.title =
          element_text(
            family = BASE_FONT,
            face = "bold",
            color = NAVY,
            size = 14
          )
      )
  )

save_figure(
  efigure_4,
  "eFigure_4_Duration_Quantile_Calibration_Over_Time",
  width = 10.5,
  height = 7.2
)

# =============================================================================
# eFIGURE 5 — Predictor-family ablation heatmap
# =============================================================================

ablation_variants <- c(
  "NO_RESP_ASSISTANCE",
  "NO_HELMET",
  "NO_GCS_AIRWAY_QUALIFIERS",
  "NO_AIRWAY_PROXY_SET",
  "NO_MECHANISM_HELMET",
  "NO_NONCRANIAL_INJURY_PHENOTYPES",
  "NO_PMHX",
  "CORE_NEURO"
)

ablation_labels <- c(
  "NO_RESP_ASSISTANCE" =
    "Remove respiratory assistance",
  "NO_HELMET" =
    "Remove helmet",
  "NO_GCS_AIRWAY_QUALIFIERS" =
    "Remove GCS airway qualifiers",
  "NO_AIRWAY_PROXY_SET" =
    "Remove airway proxy set",
  "NO_MECHANISM_HELMET" =
    "Remove mechanism + helmet",
  "NO_NONCRANIAL_INJURY_PHENOTYPES" =
    "Remove extracranial injuries",
  "NO_PMHX" =
    "Remove selected PMHx",
  "CORE_NEURO" =
    "Core neurologic predictors only"
)

endpoint_short <- c(
  "Disposition" =
    "Disposition",
  "Hospital LOS trajectory" =
    "HLOS",
  "ICU trajectory" =
    "ICU",
  "Mechanical ventilation trajectory" =
    "Ventilation",
  "Invasive ICP monitoring" =
    "ICP",
  "Craniotomy/craniectomy" =
    "Craniotomy"
)

ablation_plot <- ablation[
  variant_id %in%
    ablation_variants
]

ablation_plot[
  ,
  `:=`(
    ablation =
      factor(
        ablation_labels[
          variant_id
        ],
        levels =
          rev(
            unname(
              ablation_labels[
                ablation_variants
              ]
            )
          )
      ),
    endpoint =
      factor(
        endpoint_short[
          endpoint_label
        ],
        levels = c(
          "Disposition",
          "HLOS",
          "ICU",
          "Ventilation",
          "ICP",
          "Craniotomy"
        )
      )
  )
]

heat_auroc <- ablation_plot[
  ,
  .(
    ablation,
    endpoint,
    value =
      median_delta_AUROC,
    metric =
      "Delta AUROC"
  )
]

heat_auprc <- ablation_plot[
  ,
  .(
    ablation,
    endpoint,
    value =
      median_delta_AUPRC,
    metric =
      "Delta AUPRC"
  )
]

heat <- rbindlist(
  list(
    heat_auroc,
    heat_auprc
  )
)

limit_heat <- max(
  abs(
    heat$value
  ),
  na.rm = TRUE
)

limit_heat <- max(
  0.03,
  limit_heat
)

efigure_5 <- ggplot(
  heat,
  aes(
    x = endpoint,
    y = ablation,
    fill = value
  )
) +
  geom_tile(
    color = "white",
    linewidth = 0.6
  ) +
  geom_text(
    aes(
      label =
        sprintf(
          "%+.3f",
          value
        )
    ),
    family = BASE_FONT,
    size = 2.4,
    color = BLACK
  ) +
  facet_wrap(
    ~ metric,
    ncol = 1
  ) +
  scale_fill_gradient2(
    low = "#B2182B",
    mid = "white",
    high = "#2166AC",
    midpoint = 0,
    limits = c(
      -limit_heat,
      limit_heat
    ),
    name =
      "Change vs\nfull reference"
  ) +
  labs(
    title =
      "eFigure 5. Temporal predictor-family ablation",
    subtitle =
      "Negative values indicate worse performance after removing the specified information",
    x =
      NULL,
    y =
      NULL
  ) +
  theme_tract(
    8
  ) +
  theme(
    axis.text.x =
      element_text(
        angle = 30,
        hjust = 1
      ),
    panel.grid =
      element_blank(),
    legend.position =
      "right"
  )

save_figure(
  efigure_5,
  "eFigure_5_Predictor_Ablation_Heatmap",
  width = 9.0,
  height = 8.0
)

# =============================================================================
# eFIGURE 6 — Subgroup robustness + social/context calibration shift
# =============================================================================

subgroup_targets <- target_order

subgroup_levels <- c(
  "Mild GCS 13-15",
  "Moderate GCS 9-12",
  "Severe GCS 3-8",
  "18-39",
  "40-64",
  "65-79",
  "80-89",
  "Male",
  "Female",
  "0 missing/unknown",
  "1 missing/unknown",
  "2 missing/unknown",
  ">=3 missing/unknown"
)

subgroup_level_labels <- c(
  "Mild GCS 13-15" =
    "GCS 13-15",
  "Moderate GCS 9-12" =
    "GCS 9-12",
  "Severe GCS 3-8" =
    "GCS 3-8",
  "18-39" =
    "Age 18-39",
  "40-64" =
    "Age 40-64",
  "65-79" =
    "Age 65-79",
  "80-89" =
    "Age 80-89",
  "Male" =
    "Male",
  "Female" =
    "Female",
  "0 missing/unknown" =
    "0 missing inputs",
  "1 missing/unknown" =
    "1 missing input",
  "2 missing/unknown" =
    "2 missing inputs",
  ">=3 missing/unknown" =
    "\u22653 missing inputs"
)

sub_plot <- subgroup[
  target %in%
    subgroup_targets &
    subgroup_level %in%
      subgroup_levels
]

sub_plot[
  ,
  `:=`(
    output =
      factor(
        unname(
          target_labels[
            target
          ]
        ),
        levels =
          unname(
            target_labels[
              subgroup_targets
            ]
          )
      ),
    subgroup_display =
      factor(
        subgroup_level_labels[
          subgroup_level
        ],
        levels =
          rev(
            unname(
              subgroup_level_labels[
                subgroup_levels
              ]
            )
          )
      )
  )
]

limit_sub <- max(
  abs(
    sub_plot$delta_AUROC_vs_overall
  ),
  na.rm = TRUE
)

limit_sub <- max(
  0.08,
  limit_sub
)

p6a <- ggplot(
  sub_plot,
  aes(
    x = output,
    y = subgroup_display,
    fill = delta_AUROC_vs_overall
  )
) +
  geom_tile(
    color = "white",
    linewidth = 0.5
  ) +
  geom_text(
    data =
      sub_plot[
        abs(
          delta_AUROC_vs_overall
        ) >=
          0.02
      ],
    aes(
      label =
        sprintf(
          "%+.02f",
          delta_AUROC_vs_overall
        )
    ),
    family = BASE_FONT,
    size = 2.0,
    color = BLACK
  ) +
  scale_fill_gradient2(
    low = "#B2182B",
    mid = "white",
    high = "#2166AC",
    midpoint = 0,
    limits = c(
      -limit_sub,
      limit_sub
    ),
    name =
      "Delta AUROC\nvs overall"
  ) +
  labs(
    title =
      "A. 2024 subgroup discrimination relative to overall performance",
    subtitle =
      "Only absolute differences \u22650.02 are annotated",
    x =
      NULL,
    y =
      NULL
  ) +
  theme_tract(
    7.4
  ) +
  theme(
    axis.text.x =
      element_text(
        angle = 38,
        hjust = 1
      ),
    panel.grid =
      element_blank(),
    legend.position =
      "right"
  )

social_key <- social[
  test_year ==
    2024 &
    variant_id ==
      "PRAGMATIC_PLUS_RACE_ETHNICITY_PAYER" &
    (
      (
        subgroup_domain ==
          "race" &
          subgroup_level ==
            "Black" &
          grepl(
            "^Hospital LOS",
            target
          )
      ) |
        (
          subgroup_domain ==
            "payer" &
            subgroup_level ==
              "Self-pay" &
            target %in%
              c(
                "Home/home health",
                "Post-acute facility"
              )
        ) |
        (
          subgroup_domain ==
            "ethnicity" &
            subgroup_level ==
              "Hispanic/Latino" &
            target %in%
              c(
                "Home/home health",
                "Post-acute facility"
              )
        )
    )
]

social_key[
  ,
  example :=
    fcase(
      subgroup_domain ==
        "race",
      paste0(
        "Black: ",
        gsub(
          "Hospital LOS ",
          "HLOS ",
          target,
          fixed = TRUE
        )
      ),
      subgroup_domain ==
        "payer",
      paste0(
        "Self-pay: ",
        target
      ),
      subgroup_domain ==
        "ethnicity",
      paste0(
        "Hispanic/Latino: ",
        target
      ),
      default =
        paste(
          subgroup_level,
          target
        )
    )
]

social_long <- rbindlist(
  list(
    social_key[
      ,
      .(
        example,
        model =
          "Pragmatic clinical",
        abs_calibration_intercept =
          abs(
            calibration_intercept_reference
          )
      )
    ],
    social_key[
      ,
      .(
        example,
        model =
          "+ race, ethnicity, payer",
        abs_calibration_intercept =
          abs(
            calibration_intercept
          )
      )
    ]
  )
)

example_order <- social_key[
  order(
    -(
      abs(
        calibration_intercept_reference
      ) -
        abs(
          calibration_intercept
        )
    )
  ),
  example
]

social_long[
  ,
  example :=
    publication_label(
      example
    )
]

example_order <- publication_label(
  example_order
)

social_long[
  ,
  example :=
    factor(
      example,
      levels =
        rev(
          example_order
        )
    )
]

p6b <- ggplot(
  social_long,
  aes(
    x = abs_calibration_intercept,
    y = example,
    group = example
  )
) +
  geom_line(
    color = LIGHT_GRAY,
    linewidth = 1.0
  ) +
  geom_point(
    aes(
      shape = model,
      fill = model
    ),
    size = 3.0,
    color = NAVY,
    stroke = 0.7
  ) +
  scale_shape_manual(
    values = c(
      "Pragmatic clinical" =
        21,
      "+ race, ethnicity, payer" =
        24
    )
  ) +
  scale_fill_manual(
    values = c(
      "Pragmatic clinical" =
        "white",
      "+ race, ethnicity, payer" =
        BLUE
    )
  ) +
  labs(
    title =
      "B. Selected social/context calibration shifts",
    subtitle =
      "Absolute calibration intercept; closer to 0 indicates less calibration-in-the-large error",
    x =
      "Absolute calibration intercept",
    y =
      NULL,
    shape =
      NULL,
    fill =
      NULL
  ) +
  theme_tract(
    8
  ) +
  theme(
    legend.position =
      "bottom"
  )

efigure_6 <-
  p6a /
    p6b +
  plot_layout(
    heights = c(
      1.25,
      0.85
    )
  ) +
  plot_annotation(
    title =
      "eFigure 6. Subgroup robustness and social/health-system contextual calibration",
    theme =
      theme(
        plot.title =
          element_text(
            family = BASE_FONT,
            face = "bold",
            color = NAVY,
            size = 14
          )
      )
  )

save_figure(
  efigure_6,
  "eFigure_6_Subgroup_and_Context_Calibration",
  width = 11.0,
  height = 10.0
)

# -----------------------------------------------------------------------------
# Figure manifest
# -----------------------------------------------------------------------------

manifest <- data.table(
  exhibit = c(
    "Figure 1",
    "Figure 2",
    "Figure 3A",
    "Figure 3B",
    "eFigure 1",
    "eFigure 2",
    "eFigure 3",
    "eFigure 4",
    "eFigure 5",
    "eFigure 6"
  ),
  file_stem = c(
    "Figure_1_TBI_TRACT_Framework",
    "Figure_2_Temporal_Performance_Fingerprint",
    "Figure_3A_2024_Probability_Calibration",
    "Figure_3B_2024_Continuous_Trajectory_Calibration",
    "eFigure_1_Cohort_Flow",
    "eFigure_2_Case_Mix_Drift_Love_Plot",
    "eFigure_3_Complete_2024_Calibration_Grid",
    "eFigure_4_Duration_Quantile_Calibration_Over_Time",
    "eFigure_5_Predictor_Ablation_Heatmap",
    "eFigure_6_Subgroup_and_Context_Calibration"
  ),
  purpose = c(
    "Prediction time, temporal design, and final architecture",
    "Temporal AUROC and precision-recall lift",
    "Curated 2024 probability calibration",
    "2024 continuous-duration calibration",
    "Cohort flow and downstream observation restriction",
    "Development-to-2024 case-mix drift",
    "Complete 2024 calibration audit",
    "Temporal quantile calibration",
    "Predictor-family ablation",
    "Subgroup robustness and social/context calibration"
  )
)

fwrite(
  manifest,
  file.path(
    figure_dir,
    "FIGURE_MANIFEST.csv"
  )
)

writeLines(
  c(
    "TBI-TRACT FINAL FIGURE PIPELINE COMPLETE",
    "",
    paste0(
      "Output directory: ",
      figure_dir
    ),
    "",
    "Main:",
    "  Figure 1  framework / rolling temporal design / architecture",
    "  Figure 2  temporal performance fingerprint",
    "  Figure 3A probability calibration",
    "  Figure 3B continuous-duration calibration",
    "",
    "Supplement:",
    "  eFigure 1 cohort flow",
    "  eFigure 2 case-mix drift",
    "  eFigure 3 complete calibration grid",
    "  eFigure 4 duration quantile calibration over time",
    "  eFigure 5 predictor ablation",
    "  eFigure 6 subgroup/context calibration",
    "",
    "All submission TIFFs: 600 dpi, LZW compression.",
    "PNG copies are generated at 300 dpi for review."
  ),
  file.path(
    figure_dir,
    "FIGURE_PIPELINE_SUMMARY.txt"
  )
)

cat(
  "\n============================================================\n",
  "TBI-TRACT FINAL FIGURES COMPLETE\n",
  "Output: ",
  figure_dir,
  "\n============================================================\n",
  sep = ""
)
