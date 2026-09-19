# TBI-TRACT Shiny entry point
#
# app_core.R contains model loading, prediction logic, input validation, and
# result rendering. This file defines clinician-facing menu labels/order while
# preserving the encoder values expected by the deployed models.

suppressPackageStartupMessages(library(shiny))

# app_core.R historically used numberInput() for numeric controls. Keep this
# alias so the deployed implementation remains backward-compatible with Shiny's
# numericInput() constructor.
numberInput <- shiny::numericInput

app <- source("app_core.R", local = TRUE)$value

# -----------------------------------------------------------------------------
# Clinician-facing categorical menus
# -----------------------------------------------------------------------------

choice_display_label <- function(v, raw_value) {
  raw_value <- as.character(raw_value)
  base <- pretty_level(raw_value)
  low <- tolower(trimws(raw_value))

  if (v == "supplemental_oxygen_recovered") {
    if (low %in% c("supplemental oxygen", "yes", "y", "2")) return("Yes")
    if (low %in% c("no supplemental oxygen", "no", "n", "1")) return("No")
    if (raw_value == "__OTHER__" || base == "Other / not listed") return("Other")
    if (raw_value == "__UNKNOWN__" || base == "Unknown / not recorded") {
      return("Unknown / not recorded")
    }
  }

  base
}

choice_rank <- function(v, label) {
  x <- tolower(trimws(as.character(label)))

  if (grepl("unknown|not recorded", x)) return(990L)
  if (x %in% c("other", "other / not listed")) return(950L)

  if (v == "race_clean") {
    if (x == "white") return(10L)
    if (x == "black") return(20L)
    if (grepl("asian", x)) return(30L)
    if (grepl("american indian|alaska", x)) return(40L)
    if (grepl("pacific islander|native hawaiian", x)) return(50L)
    if (grepl("multiple", x)) return(60L)
    if (grepl("other race", x)) return(70L)
    return(500L)
  }

  if (v == "sex_clean") {
    if (x == "male") return(10L)
    if (x == "female") return(20L)
    if (grepl("other|nonbinary|non-binary", x)) return(30L)
    return(500L)
  }

  if (v == "ethnicity_clean") {
    if (grepl("hispanic|latino", x) && !grepl("not|non", x)) return(10L)
    if (grepl("not hispanic|non-hispanic|non hispanic", x)) return(20L)
    return(500L)
  }

  if (v == "insurance_clean") {
    if (grepl("private|commercial", x)) return(10L)
    if (grepl("medicare", x)) return(20L)
    if (grepl("medicaid", x)) return(30L)
    if (grepl("self[- ]?pay|uninsured", x)) return(40L)
    if (grepl("government", x)) return(50L)
    return(500L)
  }

  if (v == "mechanism_clean") {
    if (grepl("motor vehicle|transport", x)) return(10L)
    if (grepl("fall", x)) return(20L)
    if (grepl("firearm|gunshot", x)) return(30L)
    if (grepl("blunt", x)) return(40L)
    if (grepl("penetrat|stab", x)) return(50L)
    return(500L)
  }

  if (v == "pupil_clean") {
    if (grepl("both.*react|bilateral.*react", x)) return(10L)
    if (grepl("one.*react|unilateral.*react", x)) return(20L)
    if (grepl("neither|none.*react|nonreact", x)) return(30L)
    return(500L)
  }

  if (v == "supplemental_oxygen_recovered") {
    if (x == "yes") return(10L)
    if (x == "no") return(20L)
    if (x == "other") return(30L)
    return(500L)
  }

  500L
}

ordered_categorical_choices <- function(v, levels) {
  lev <- unique(as.character(levels))
  if (!length(lev)) lev <- "__UNKNOWN__"

  # Prefer the model's explicit unknown sentinel when duplicate unknown labels
  # exist in the training encoder.
  if ("__UNKNOWN__" %in% lev) {
    duplicate_unknown <- vapply(
      lev,
      function(z) z != "__UNKNOWN__" && pretty_level(z) == "Unknown / not recorded",
      logical(1)
    )
    lev <- lev[!duplicate_unknown]
  }

  # __OTHER__ is an encoder fallback rather than a clinical response option.
  # Supplemental oxygen is the deliberate exception because the interface uses
  # Yes / No / Other / Unknown.
  if (v != "supplemental_oxygen_recovered") {
    lev <- lev[lev != "__OTHER__"]
  }

  labels_out <- vapply(
    lev,
    function(z) choice_display_label(v, z),
    character(1)
  )

  keep <- !duplicated(labels_out)
  lev <- lev[keep]
  labels_out <- labels_out[keep]

  ranks <- vapply(labels_out, function(z) choice_rank(v, z), integer(1))
  ord <- order(ranks, tolower(labels_out), seq_along(labels_out))

  setNames(lev[ord], labels_out[ord])
}

gcs_choices <- list(
  gcs_eye_clean = c(
    "1 - None" = "1",
    "2 - To pain" = "2",
    "3 - To speech" = "3",
    "4 - Spontaneous" = "4",
    "Unknown / not recorded" = ""
  ),
  gcs_verbal_clean = c(
    "1 - None" = "1",
    "2 - Incomprehensible sounds" = "2",
    "3 - Inappropriate words" = "3",
    "4 - Confused" = "4",
    "5 - Oriented" = "5",
    "Unknown / not recorded" = ""
  ),
  gcs_motor_clean = c(
    "1 - None" = "1",
    "2 - Extension" = "2",
    "3 - Flexion" = "3",
    "4 - Withdraws" = "4",
    "5 - Localizes" = "5",
    "6 - Obeys commands" = "6",
    "Unknown / not recorded" = ""
  )
)

make_control <- function(v, cached = NULL) {
  id <- input_id(v)
  lab <- label_for(v)

  cached_chr <- if (!is.null(cached) && length(cached)) {
    as.character(cached[[1L]])
  } else {
    NA_character_
  }

  if (v %in% names(gcs_choices)) {
    valid <- unname(gcs_choices[[v]])
    selected <- if (!is.na(cached_chr) && cached_chr %in% valid) {
      cached_chr
    } else {
      ""
    }

    return(selectInput(
      id,
      lab,
      gcs_choices[[v]],
      selected = selected,
      selectize = FALSE
    ))
  }

  if (v %in% binary_predictors) {
    choices <- c(
      "No" = "0",
      "Yes" = "1",
      "Unknown / not recorded" = ""
    )
    selected <- if (!is.na(cached_chr) && cached_chr %in% unname(choices)) {
      cached_chr
    } else {
      ""
    }

    return(selectInput(
      id,
      lab,
      choices,
      selected = selected,
      selectize = FALSE
    ))
  }

  if (v %in% continuous_predictors) {
    cached_num <- if (!is.null(cached) && length(cached)) {
      suppressWarnings(as.numeric(cached[[1L]]))
    } else {
      NA_real_
    }
    value <- if (is.finite(cached_num)) cached_num else NA_real_

    lim <- numeric_limits[[v]]
    if (!is.null(lim)) {
      return(tagList(
        shiny::numericInput(
          id,
          lab,
          value = value,
          min = lim$min,
          max = lim$max,
          step = lim$step
        ),
        div(class = "input-hint", lim$note)
      ))
    }
    return(shiny::numericInput(id, lab, value = value))
  }

  if (v %in% categorical_predictors) {
    lev <- get_levels(v)
    if (!length(lev)) lev <- "__UNKNOWN__"

    choices <- ordered_categorical_choices(v, lev)
    vals <- unname(choices)

    default_selected <- if ("__UNKNOWN__" %in% vals) {
      "__UNKNOWN__"
    } else {
      unknown_idx <- which(
        vapply(
          vals,
          function(z) pretty_level(z) == "Unknown / not recorded",
          logical(1)
        )
      )
      if (length(unknown_idx)) vals[unknown_idx[1L]] else vals[1L]
    }

    selected <- if (!is.na(cached_chr) && cached_chr %in% vals) {
      cached_chr
    } else {
      default_selected
    }

    return(selectInput(
      id,
      lab,
      choices,
      selected = selected,
      selectize = FALSE
    ))
  }

  NULL
}

app
