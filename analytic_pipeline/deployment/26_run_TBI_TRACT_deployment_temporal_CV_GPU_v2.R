# =============================================================================
# 26_run_TBI_TRACT_deployment_temporal_CV_GPU_v2.R
#
# PURPOSE
#   Lock the final 2020-2024 deployment-model specification without changing
#   the manuscript-facing temporal performance estimates.
#
# STRATEGY
#   1) Structural hyperparameters are selected across the already completed
#      rolling-origin forward-temporal tuning folds (2022, 2023, 2024).
#
#      For each configuration and endpoint:
#        - rank configuration within each temporal tuning year
#        - compute relative excess tuning loss above the best configuration
#        - select the configuration with the lowest mean relative excess loss
#          across the three temporal folds
#        - tie-break by mean rank, worst rank, then simpler/more regularized fit
#
#   2) With structure LOCKED, choose boosting rounds using 5-fold CV on the
#      entire 2020-2024 development cohort.
#
#      These folds are stratified across calendar year (and outcome class where
#      applicable) and are used ONLY to select the final number of boosting
#      rounds for the deployment model.
#
#      They are NOT reported as model validation. Manuscript-facing performance
#      remains the rolling-origin temporal evaluation from scripts 23-25.
#
#   3) Fit final deployment models on all 2020-2024 observations and save
#      XGBoost JSON models plus endpoint-specific encoders/specifications.
#
# FINAL PREDICTOR POLICY
#   Disposition + HLOS trajectory:
#       PRAGMATIC_CLINICAL + race + ethnicity + payer
#
#   ICU trajectory + ventilation trajectory + ICP + craniotomy:
#       PRAGMATIC_CLINICAL
#
#   Continuous HLOS + conditional ICU LOS + conditional ventilator duration:
#       PRAGMATIC_CLINICAL
#
# PRAGMATIC_CLINICAL removes only:
#       helmet_use_recovered
#       respiratoryassistance_clean
# =============================================================================

rm(list = ls())
gc()

required <- c("data.table", "reticulate")
missing_pkgs <- required[
  !vapply(required, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_pkgs) > 0L) {
  stop("Missing package(s): ", paste(missing_pkgs, collapse = ", "), call. = FALSE)
}

suppressPackageStartupMessages({
  library(data.table)
  library(reticulate)
})

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

config_candidates <- c(
  file.path(getwd(), "R", "00_config.R"),
  "R/00_config.R"
)
config_file <- config_candidates[file.exists(config_candidates)[1L]
if (length(config_file) == 0L  || is.na(config_file)) {
  stop("Could not find R/00_config.R.", call. = FALSE)
}
source(config_file)

methods_dir <- file.path(output_dir, "METHODS_COMPLETION_TBI_TRACT")
class_dir <- file.path(output_dir, "TBI_TRACT_PRAGMATIC_FINALIST_RETUNE")
duration_dir <- file.path(output_dir, "TBI_TRACT_PRAGMATIC_DURATION_FINALIZATION")

dataset_file <- file.path(
  methods_dir,
  "frozen_methods_dataset_retained_2020_2024.rds"
)
types_file <- file.path(
  methods_dir,
  "11_FROZEN_PREDICTOR_TYPES.csv"
)

needed <- c(dataset_file, types_file, class_dir, duration_dir)
if (!all(file.exists(needed))) {
  stop(
    "Missing prerequisite file/folder(s):\n",
    paste(needed[!file.exists(needed)], collapse = "\n"),
    call. = FALSE
  )
}

out_dir <- file.path(output_dir, "TBI_TRACT_FINAL_DEPLOYMENT_CV")
model_dir <- file.path(out_dir, "models")
encoder_dir <- file.path(out_dir, "encoders")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(encoder_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# Compute backend
# -----------------------------------------------------------------------------

ENV_NAME <- "tbi-tract-xgb-gpu"

logical_cores <- parallel::detectCores(logical = TRUE)
if (is.na(logical_cores) || logical_cores < 2L) {
  logical_cores <- 32L
}

cpu_threads <- max(
  1L,
  min(logical_cores - 1L, floor(0.94 * logical_cores))
)

setDTthreads(cpu_threads)

Sys.setenv(
  OMP_NUM_THREADS = as.character(cpu_threads),
  MKL_NUM_THREADS = as.character(cpu_threads),
  OPENBLAS_NUM_THREADS = as.character(cpu_threads)
)

reticulate::use_condaenv(ENV_NAME, required = TRUE)
py_cfg <- reticulate::py_config()
xgb <- reticulate::import("xgboost", convert = TRUE)

# CUDA probe
set.seed(20260913L)
probe_x <- matrix(rnorm(4000), nrow = 500L, ncol = 8L)
probe_y <- as.integer(rbinom(500L, 1L, 0.3))
probe_d <- xgb$DMatrix(probe_x, label = probe_y)

probe_fit <- xgb$train(
  params = reticulate::dict(
    objective = "binary:logistic",
    tree_method = "hist",
    device = "cuda",
    max_depth = 2L,
    eta = 0.2,
    nthread = as.integer(cpu_threads),
    seed = 20260913L
  ),
  dtrain = probe_d,
  num_boost_round = 3L,
  verbose_eval = FALSE
)

probe_cfg_raw <- probe_fit$save_config()
probe_cfg <- tryCatch(
  as.character(reticulate::py_to_r(probe_cfg_raw))[1L],
  error = function(e) as.character(probe_cfg_raw)[1L]
)

if (!grepl('"device"[[:space:]]*:[[:space:]]*"cuda', probe_cfg, ignore.case = TRUE)) {
  stop("CUDA was not verified.", call. = FALSE)
}

rm(probe_x, probe_y, probe_d, probe_fit)
gc()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

safe_num <- function(x) suppressWarnings(as.numeric(as.character(x)))

normalize_character <- function(x) {
  x <- trimws(as.character(x))
  x[
    is.na(x) |
      x == "" |
      x %in% c("NA", "NaN", "<NA>")
  ] <- "__UNKNOWN__"
  x
}

fit_encoder <- function(d, numeric_vars, categorical_vars) {
  levels_list <- lapply(
    categorical_vars,
    function(v) {
      x <- normalize_character(d[[v]])
      unique(
        c(
          sort(unique(x)),
          "__UNKNOWN__",
          "__OTHER__"
        )
      )
    }
  )
  names(levels_list) <- categorical_vars

  list(
    numeric_vars = numeric_vars,
    categorical_vars = categorical_vars,
    categorical_levels = levels_list
  )
}

encode_dense <- function(d, encoder) {
  n <- nrow(d)
  n_numeric <- length(encoder$numeric_vars)
  n_cat <- sum(
    vapply(
      encoder$categorical_levels,
      length,
      integer(1)
    )
  )

  X <- matrix(
    0,
    nrow = n,
    ncol = n_numeric + n_cat
  )

  feature_names <- character(ncol(X))
  parent_names <- character(ncol(X))
  j <- 1L

  for (v in encoder$numeric_vars) {
    X[, j] <- safe_num(d[[v]])
    feature_names[j] <- v
    parent_names[j] <- v
    j <- j + 1L
  }

  for (v in encoder$categorical_vars) {
    lev <- encoder$categorical_levels[[v]]
    x <- normalize_character(d[[v]])
    x[!x %in% lev] <- "__OTHER__"
    idx <- match(x, lev)

    cols <- j:(j + length(lev) - 1L)
    X[
      cbind(
        seq_len(n),
        cols[idx]
      )
    ] <- 1

    feature_names[cols] <- paste0(
      v,
      "__",
      make.names(lev, unique = TRUE)
    )
    parent_names[cols] <- v

    j <- max(cols) + 1L
  }

  colnames(X) <- feature_names
  attr(X, "parent_predictor") <- parent_names
  storage.mode(X) <- "double"
  X
}

make_balanced_folds <- function(year, strata = NULL, k = 5L, seed = 20260913L) {
  set.seed(seed)

  d <- data.table(
    row_id = seq_along(year),
    year = as.character(year),
    strata = if (is.null(strata!,ƒÒ! ãG7V2FVæGö–çEö–BÀ¢VæGö–çEöÆ&VÂÒ7V2FVæGö–çEöÆ&VÂÀ¢ÖöFVÅöfÖ–Ç’Ò&6Æ76–f–6F–öâ"À¢6öæf–uö–BÒ7G'V7GW&ÂF6öæf–uö–BÀ¢6VÆV7FVE÷&÷VæG5óVföÆEö7bÒ6VÆV7FVE÷&÷VæG2À¢âÒç&÷r……÷W6R¢ ¢Öæ–fW7E÷&÷w5µ°¢ÆVæwF‚†Öæ–fW7E÷&÷w2’²À¢ÕÒÃÂÒFFçF&ÆR€¢VæGö–çEö–BÒ7V2FVæGö–çEö–BÀ¢VæGö–çEöÆ&VÂÒ7V2FVæGö–çEöÆ&VÂÀ¢ÖöFVÅöfÖ–Ç’Ò&6Æ76–f–6F–öâ"À¢&VF–7F÷%÷öÆ–7’Ð¢–b€¢7V2Gf&–çBÓÐ¢%$tÔD”5õÅU5õ$4UôUD„ä”4•E•õ”U" ¢’°¢%&vÖF–26Æ–æ–6Â²&6R²WF†æ–6—G’²–W" ¢ÒVÇ6R°¢%&vÖF–26Æ–æ–6Â ¢ÒÀ¢6öæf–uö–BÒ7G'V7GW&ÂF6öæf–uö–BÀ¢Ö…öFWF‚Ò7G'V7GW&ÂFÖ…öFWF‚À¢Ö–åö6†–ÆE÷vV–v‡BÒ7G'V7GW&ÂFÖ–åö6†–ÆE÷vV–v‡BÀ¢7V'6×ÆRÒ7G'V7GW&ÂG7V'6×ÆRÀ¢6öÇ6×ÆUö'—G&VRÒ7G'V7GW&ÂF6öÇ6×ÆUö'—G&VRÀ¢ÆÖ&FÒ7G'V7GW&ÂFÆÖ&FÀ¢6VÆV7FVE÷&÷VæG2Ò6VÆV7FVE÷&÷VæG2À¢ÖöFVÅ÷F‚ÒÖöFVÅ÷F‚À¢Væ6öFW%÷F‚ÒVæ6öFW%÷F‚À¢âÒç&÷r……÷W6R¢ ¢&Ò€¢‚À¢…÷W6RÀ¢EöÆÂÀ¢f–æÅöf—@¢¢v2‚§Ð ¦f—EöæE÷6fUöGW&F–öâÂÒgVæ7F–öâ‡7V2’°¢7G'V7GW&ÂÂÒ6VÆV7FVE÷F&ÆU°¢VæGö–çEö–BÓÐ¢7V2FVæGö–çEö–@¢Ð ¢–b†ç&÷r‡7G'V7GW&Â’ÒÂ’°¢7F÷€¢%7G'V7GW&Â6VÆV7F–öâ—2æ÷BVæ—VRf÷""À¢7V2FVæGö–çEö–BÀ¢6ÆÂâÒdÅ4P¢¢Ð ¢åö¶VWÂÒ–çFW'6V7B€¢çVÖW&–5÷&VF–7F÷'2À¢7V2G&VF–7F÷'0¢¢5ö¶VWÂÒ–çFW'6V7B€¢6FVv÷&–6Å÷&VF–7F÷'2À¢7V2G&VF–7F÷'0¢ ¢Væ6öFW"ÂÒf—EöVæ6öFW"€¢GBÀ¢åö¶VWÀ¢5ö¶VW ¢ ¢‚ÂÒVæ6öFUöFVç6R€¢GBÀ¢Væ6öFW ¢ ¢•÷&rÂÒ6fUöçVÒ€¢GEµ·7V2G•÷f%ÕÐ¢ ¢¶VWÂÒ7V2Ff–ÇFW"€¢•÷&p¢ ¢…÷W6RÂÒ…°¢¶VWÀ¢À¢G&÷ÒdÅ4P¢Ð¢•÷W6RÂÒÆös€¢•÷&u°¢¶VW ¢Ð¢¢–V%÷W6RÂÒGBFFÖ—76–öå÷–V%°¢¶VW ¢Ð ¢2f÷"GW&F–öâ5bÂF—7G&–'WFRF†R÷WF6öÖR&ævR7&÷72föÆG2v—F†–âV6‚–V ¢2W6–ær–V"×7V6–f–2V–çF–ÆR7G&FâF†—2—2ôäÅ’f÷"&÷VæB6VÆV7F–öâà¢7G&EöGBÂÒFFçF&ÆR€¢–V"Ò–V%÷W6RÀ¢’Ò•÷&u°¢¶VW ¢Ð¢ ¢7G&EöGE°¢À¢÷WF6öÖU÷7G&GVÒ£Ò°¢'"ÂÒg&æ²€¢’À¢F–W2æÖWF†öBÒ&fW&vR ¢’ð¢äà ¢7FS€¢%"À¢Ö–â€¢TÂÀ¢Ö‚€¢ÂÀ¢6V–Æ–ær€¢R ¢' ¢¢¢¢¢ÒÀ¢'’Ò–V ¢Ð ¢&×2ÂÒ&WF–7VÆFS£¦F–7B€¢ö&¦V7F—fRÒ'&Vs§VçF–ÆVW'&÷""À¢VçF–ÆUöÇ†Ò2€¢ãÀ¢ãSÀ¢ã“ ¢’À¢WFÒãRÀ¢Ö…öFWF‚Ò2æ–çFVvW"‡7G'V7GW&ÂFÖ…öFWF‚’À¢Ö–åö6†–ÆE÷vV–v‡BÒ2æçVÖW&–2‡7G'V7GW&ÂFÖ–åö6†–ÆE÷vV–v‡B’À¢7V'6×ÆRÒ2æçVÖW&–2‡7G'V7GW&ÂG7V'6×ÆR’À¢6öÇ6×ÆUö'—G&VRÒ2æçVÖW&–2‡7G'V7GW&ÂF6öÇ6×ÆUö'—G&VR’À¢ÆÖ&FÒ2æçVÖW&–2‡7G'V7GW&ÂFÆÖ&F’À¢G&VUöÖWF†öBÒ&†—7B"À¢FWf–6RÒ&7VF"À¢çF‡&VBÒ2æ–çFVvW"†7U÷F‡&VG2’À¢6VVBÒ##c“4À¢ ¢7bÂÒ6VÆV7Eö7e÷&÷VæG2€¢‚Ò…÷W6RÀ¢’Ò•÷W6RÀ¢–V"Ò–V%÷W6RÀ¢7G&FÒ7G&EöGBF÷WF6öÖU÷7G&GVÒÀ¢&×2Ò&×2À¢Ö…÷&÷VæG2Ò#SÂÀ¢V&Ç•÷7F÷ÒÂÀ¢²ÒTÂÀ¢6VVBÒ##c“4À¢ ¢6VÆV7FVE÷&÷VæG2ÂÒ7bG6VÆV7FVE÷&÷VæG0 ¢gw&—FR€¢7bF7eöÆörÀ¢f–ÆRçF‚€¢÷WEöF—"À¢7FS€¢&7eöÆöuõò"À¢7V2FVæGö–çEö–BÀ¢"æ77b ¢¢¢ ¢EöÆÂÂÒ†v"DDÖG&—‚€¢…÷W6RÀ¢Æ&VÂÒ•÷W6P¢ ¢f–æÅöf—BÂÒ†v"GG&–â€¢&×2Ò&×2À¢GG&–âÒEöÆÂÀ¢çVÕö&ö÷7E÷&÷VæBÒ2æ–çFVvW"€¢6VÆV7FVE÷&÷VæG0¢’À¢fW&&÷6UöWfÂÒdÅ4P¢ ¢ÖöFVÅ÷F‚ÂÒf–ÆRçF‚€¢ÖöFVÅöF—"À¢7FS€¢7V2FVæGö–çEö–BÀ¢"æ§6öâ ¢¢ ¢f–æÅöf—BG6fUöÖöFVÂ€¢ÖöFVÅ÷F€¢ ¢Væ6öFW%÷F‚ÂÒf–ÆRçF‚€¢Væ6öFW%öF—"À¢7FS€¢7V2FVæGö–çEö–BÀ¢%öVæ6öFW"ç&G2 ¢¢ ¢6fU$E2€¢Æ—7B€¢VæGö–çEö–BÒ7V2FVæGö–çEö–BÀ¢VæGö–çEöÆ&VÂÒ7V2FVæGö–çEöÆ&VÂÀ¢G—RÒ'VçF–ÆR"À¢VçF–ÆUöÇ†Ò2€¢ãÀ¢ãSÀ¢ã“ ¢’À¢Æ÷vW%ö&÷VæBÒ7V2FÆ÷vW%ö&÷VæBÀ¢&VF–7F÷'2Ò7V2G&VF–7F÷'2À¢Væ6öFW"ÒVæ6öFW"À¢fVGW&UöæÖW2Ò6öÆæÖW2……÷W6R’À¢7G'V7GW&Åö6öæf–uö–BÒ7G'V7GW&ÂF6öæf–uö–BÀ¢6VÆV7FVE÷&÷VæG2Ò6VÆV7FVE÷&÷VæG2À¢G&–æ–æu÷–V'2Ò##£##BÀ¢âÒç&÷r……÷W6R¢’À¢Væ6öFW%÷F€¢ ¢&÷VæE÷&÷w5µ°¢ÆVæwF‚‡&÷VæE÷&÷w2’²À¢ÕÒÃÂÒFFçF&ÆR€¢VæGö–çEö–BÒ7V2FVæGö–çEö–BÀ¢VæGö–çEöÆ&VÂÒ7V2FVæGö–çEöÆ&VÂÀ¢ÖöFVÅöfÖ–Ç’Ò&GW&F–öâ"À¢6öæf–uö–BÒ7G'V7GW&ÂF6öæf–uö–BÀ¢6VÆV7FVE÷&÷VæG5óVföÆEö7bÒ6VÆV7FVE÷&÷VæG2À¢âÒç&÷r……÷W6R¢ ¢Öæ–fW7E÷&÷w5µ°¢ÆVæwF‚†Öæ–fW7E÷&÷w2’²À¢ÕÒÃÂÒFFçF&ÆR€¢VæGö–çEö–BÒ7V2FVæGö–çEö–BÀ¢VæGö–çEöÆ&VÂÒ7V2FVæGö–çEöÆ&VÂÀ¢ÖöFVÅöfÖ–Ç’Ò&GW&F–öâ"À¢&VF–7F÷%÷öÆ–7’Ò%&vÖF–26Æ–æ–6Â"À¢6öæf–uö–BÒ7G'V7GW&ÂF6öæf–uö–BÀ¢Ö…öFWF‚Ò7G'V7GW&ÂFÖ…öFWF‚À¢Ö–åö6†–ÆE÷vV–v‡BÒ7G'V7GW&ÂFÖ–åö6†–ÆE÷vV–v‡BÀ¢7V'6×ÆRÒ7G'V7GW&ÂG7V'6×ÆRÀ¢6öÇ6×ÆUö'—G&VRÒ7G'V7GW&ÂF6öÇ6×ÆUö'—G&VRÀ¢ÆÖ&FÒ7G'V7GW&ÂFÆÖ&FÀ¢6VÆV7FVE÷&÷VæG2Ò6VÆV7FVE÷&÷VæG2À¢ÖöFVÅ÷F‚ÒÖöFVÅ÷F‚À¢Væ6öFW%÷F‚ÒVæ6öFW%÷F‚À¢âÒç&÷r……÷W6R¢ ¢&Ò€¢‚À¢…÷W6RÀ¢EöÆÂÀ¢f–æÅöf—@¢¢v2‚§Ð ¦f÷"‡7V2–â6Æ75÷7V72’°¢6B€¢%ÆãÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÕÆâ"À¢$DUÄõ”ÔTåB5c¢"À¢7V2FVæGö–çEöÆ&VÂÀ¢%Æâ"À¢#ÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÕÆâ"À¢6WÒ" ¢¢f—EöæE÷6fUö6Æ76–f–6F–öâ€¢7V0¢§Ð ¦f÷"‡7V2–âGW&F–öå÷7V72’°¢6B€¢%ÆãÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÕÆâ"À¢$DUÄõ”ÔTåB5c¢"À¢7V2FVæGö–çEöÆ&VÂÀ¢%Æâ"À¢#ÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÕÆâ"À¢6WÒ" ¢¢f—EöæE÷6fUöGW&F–öâ€¢7V0¢§Ð §&÷VæE÷F&ÆRÂÒ&&–æFÆ—7B€¢&÷VæE÷&÷w2À¢f–ÆÂÒE%TP¢¦Öæ–fW7BÂÒ&&–æFÆ—7B€¢Öæ–fW7E÷&÷w2À¢f–ÆÂÒE%TP¢ ¦gw&—FR€¢&÷VæE÷F&ÆRÀ¢f–ÆRçF‚€¢÷WEöF—"À¢#5ôeTÄÅôDUdTÄõÔTåEóTdôÄEô5eõ$õTäE2æ77b ¢¢ ¦gw&—FR€¢Öæ–fW7BÀ¢f–ÆRçF‚€¢÷WEöF—"À¢#Eôd”äÅôDUÄõ”ÔTåEôÔôDTÅôÔä”dU5Bæ77b ¢¢ §6fU$E2€¢Æ—7B€¢6Æ76–f–6F–öâÒ6Æ75÷7V72À¢GW&F–öâÒGW&F–öå÷7V72À¢&vÖF–5ö6Æ–æ–6Å÷&VF–7F÷'2Ò&vÖF–5ö6Æ–æ–6ÂÀ¢6ö6–Åö6öçFW‡E÷&VF–7F÷'2Ò6ö6–Å÷f'2À¢FWÆ÷–ÖVçEöÖæ–fW7BÒÖæ–fW7@¢’À¢f–ÆRçF‚€¢÷WEöF—"À¢#Uôd”äÅôDUÄõ”ÔTåEõ5T4”d”4D”ôâç&G2 ¢¢ §w&—FTÆ–æW2€¢2€¢%D$’ÕE$5Bd”äÂDUÄõ”ÔTåB5b4ôÕÄUDR"À¢""À¢%7G'V7GW&Â6öæf–wW&F–öã¢"À¢"6VÆV7FVB'’F†R&W7V6–f–VBf÷'v&B×FV×÷&Â7G'V7GW&Â×6VÆV7F–öâ'VÆR7&÷72F†RF‡&VR&öÆÆ–ærÖ÷&–v–âGVæ–ærföÆG2â"À¢""À¢$&ö÷7F–ær&÷VæG3¢"À¢"6VÆV7FVB'’–V"ö÷WF6öÖR×7G&F–f–VBRÖföÆB5bW6–ærF†RVçF—&R##Ó##BFWfVÆ÷ÖVçB6ö†÷'Bâ"À¢""À¢$–×÷'FçC¢"À¢"F†RRÖföÆB5bW7F–ÖFW2&RäõBÖçW67&—BÖf6–ærfÆ–FF–öâW7F–ÖFW2â"À¢"ÖçW67&—BÖf6–ærW&f÷&Öæ6R&VÖ–ç2F†Rf÷'v&BFV×÷&Â##"ó##2ó##BWfÇVF–öââ"À¢""À¢$f–æÂFWÆ÷–ÖVçBÖöFVÇ2vW&Rf—BöâÆÂ##Ó##Bö'6W'fF–öç2æB6fVB2¥4ôââ"À¢""À¢%&Wf–Ws¢"À¢"õDTÕõ$Åô5eõ5E%T5EU$Åô4ôäd”uôtu$TtD”ôâæ77b"À¢"%õDTÕõ$Åô5eõ4TÄT5DTEõ5E%T5EU$Åô4ôäd”ræ77b"À¢"5ôeTÄÅôDUdTÄõÔTåEóTdôÄEô5eõ$õTäE2æ77b"À¢"Eôd”äÅôDUÄõ”ÔTåEôÔôDTÅôÔä”dU5Bæ77b ¢’À¢f–ÆRçF‚€¢÷WEöF—"À¢$DUÄõ”ÔTåEô5eõ5TÔÔ%’çG‡B ¢¢ ¦6B‚%ÆäDôäS¢"Â÷WEöF—"Â%Æâ"Â6WÒ""