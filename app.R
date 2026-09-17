# TBI-TRACT Shiny entrypoint
#
# The full application implementation is preserved in app_core.R. This wrapper
# applies small presentation/compatibility overrides without changing the locked
# model objects, encoder values, or prediction logic.

suppressPackageStartupMessages(library(shiny))

# Compatibility alias: Shiny's constructor is numericInput().
numberInput <- shiny::numericInput

# Load the locked application implementation into this environment. The server
# function created by app_core.R resolves helpers from this same environment at
# runtime, so the clinician-facing ordering overrides below are used by renderUI.
app <- source("app_core.R", local = TRUE)$value

# -----------------------------------------------------------------------------
# Clinician-facing categorical menus
# -----------------------------------------------------------------------------
# Keep encoder values unchanged while presenting categories in a deliberate,
# clinically readable order. Internal __OTHER__ is not exposed when a real
# observed "Other" category exists; __UNKNOWN__ remains the user-facing unknown
# state. Supplemental oxygen intentionally exposes __OTHER__ as "Other" because
# the desired clinical control is Yes / No / Other / Unknown.

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

  # Unknown is always last; generic/internal other is immediately before it.
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

  # Prefer the model's explicit unknown sentinel for an unrecorded value.
  if ("__UNKNOWN__" %in% lev) {
    is_other_unknown <- vapply(
      lev,
      function(z) z != "__UNKNOWN__" && pretty_level(z) == "Unknown / not recorded",
      logical(1)
    )
    lev <- lev[!is_other_unknown]
  }

  # __OTHER__ is an encoder safety net, not a clinical response option. Hide it
  # whenever possible. Supplemental oxygen is the one deliberate exception so
  # the interface can offer Yes / No / Other / Unknown as requested.
  if (v != "supplemental_oxygen_recovered") {
    lev <- lev[lev != "__OTHER__"]
  }

  labels_out <- vapply(
    lev,
    function(z) choice_display_label(v, z),
    character(1)
  )

  # If two raw levels would display identically, retain the first deliberate
  # encoder choice only. This avoids duplicate Unknown/Other-looking options.
  keep <- !duplicated(labels_out)
  lev <- lev[keep]
  labels_out <- labels_out[keep]

  ranks <- vapply(labels_out, function(z) choice_rank(v, z), integer(1))
  ord <- order(ranks, tolower(labels_out), seq_along(labels_out))

  setNames(lev[ord], labels_out[ord])
}

# Put unknown last in the GCS menus while keeping it selected initially.
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

# Replace only the control-construction presentation layer. Encoder values and
# prediction preprocessing remain unchanged.
make_control <- function(v) {
  id <- input_id(v)
  lab <- label_for(v)

  if (v %in% names(gcs_choices)) {
    return(selectInput(
      id,
      lab,
      gcs_choices[[v]],
      selected = "",
      selectize = FALSE
    ))
  }

  if (v %in% binary_predictors) {
    return(selectInput(
      id,
      lab,
      c(
        "No" = "0",
        "Yes" = "1",
        "Unknown / not recorded" = ""
      ),
      selected = "",
      selectize = FALSE
    ))
  }

  if (v %in% continuous_predictors) {
    lim <- numeric_limits[[v]]
    if (!is.null(lim)) {
      return(tagList(
        shiny::numericInput(
          id,
          lab,
          value = NA,
          min = lim$min,
          max = lim$max,
          step = lim$step
        ),
        div(class = "input-hint", lim$note)
      ))
    }
    return(shiny::numericInput(id, lab, value = NA))
  }

  if (v %in% categorical_predictors) {
    lev <- get_levels(v)
    if (!length(lev)) lev <- "__UNKNOWN__"

    choices <- ordered_categorical_choices(v, lev)
    vals <- unname(choices)

    selected <- if ("__UNKNOWN__" %in% vals) {
      "__UNKNOWN__"
    } else {
      unknown_idx <- which(
        vapply(vals, function(z) pretty_level(z) == "Unknown / not recorded", logical(1))
      )
      if (length(unknown_idx)) vals[unknown_idx[1L]] else vals[1L]
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
