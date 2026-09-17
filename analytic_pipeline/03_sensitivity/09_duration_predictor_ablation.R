# =============================================================================
# 19_run_TBI_TRACT_duration_stress_test_GPU_v2.R
#
# PURPOSE
#   Extend the predictor-family stress test to the FINAL duration models:
#
#     - Hospital LOS: unconditional Q10 / Q50 / Q90
#     - ICU LOS: conditional on ICU use, Q10 / Q50 / Q90
#     - Ventilator duration: conditional on ventilation, Q10 / Q50 / Q90
#
#   Uses the SAME rolling forward-chaining folds and predictor variants as
#   script 18.
#
#   Structural hyperparameters are NOT re-invented here. For each fold:
#     HLOS duration <- fold-specific HLOS trajectory architecture from script 18
#     ICU duration  <- fold-specific ICU trajectory architecture from script 18
#     Vent duration <- fold-specific ventilation trajectory architecture
#
#   Each duration ablation selects its own boosting rounds on the pre-test tune
#   year, refits on all pre-test years, and evaluates on the held-out test year.
#
# RUN AFTER:
#   18_run_TBI_TRACT_predictor_stress_test_GPU.R
#
# =============================================================================

rm(list = ls())
gc()

required <- c("data.table", "reticulate")
missing_pkgs <- required[
  !vapply(required, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_pkgs) > 0L) {
  stop(
    "Missing package(s): ",
    paste(missing_pkgs, collapse = ", "),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(data.table)
  library(reticulate)
})

# -----------------------------------------------------------------------------
# Project config
# -----------------------------------------------------------------------------

config_candidates <- c(
  file.path(getwd(), "R", "00_config.R"),
  "R/00_config.R"
)

config_file <- config_candidates[file.exists(config_candidates)][1L]

if (
  length(config_file) == 0L ||
    is.na(config_file)
) {
  stop("Could not find R/00_config.R.", call. = FALSE)
}

source(config_file)

methods_dir <- file.path(
  output_dir,
  "METHODS_COMPLETION_TBI_TRACT"
)

stress_dir <- file.path(
  output_dir,
  "TBI_TRACT_PREDICTOR_STRESS_TEST"
)

dataset_file <- file.path(
  methods_dir,
  "frozen_methods_dataset_retained_2020_2024.rds"
)

types_file <- file.path(
  methods_dir,
  "11_FROZEN_PREDICTOR_TYPES.csv"
)

hp_file <- file.path(
  stress_dir,
  "04_SELECTED_REFERENCE_HYPERPARAMETERS_BY_FOLD_ENDPOINT.csv"
)

needed_files <- c(
  dataset_file,
  types_file,
  hp_file
)

if (!all(file.exists(needed_files))) {
  stop(
    "Missing prerequisite file(s). Run script 18 first:\n",
    paste0(
      "  ",
      needed_files[!file.exists(needed_files)],
      collapse = "\n"
    ),
    call. = FALSE
  )
}

out_dir <- file.path(
  stress_dir,
  "DURATION_STRESS_TEST"
)

dir.create(
  out_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

# -----------------------------------------------------------------------------
# Compute backend
# -----------------------------------------------------------------------------

ENV_NAME <- "tbi-tract-xgb-gpu"

logical_cores <- parallel::detectCores(logical = TRUE)

if (
  is.na(logical_cores) ||
    logical_cores < 2L
) {
  logical_cores <- 32L
}

cpu_threads <- max(
  1L,
  min(
    logical_cores - 1L,
    floor(0.94 * logical_cores)
  )
)

setDTthreads(cpu_threads)

Sys.setenv(
  OMP_NUM_THREADS = as.character(cpu_threads),
  MKL_NUM_THREADS = as.character(cpu_threads),
  OPENBLAS_NUM_THREADS = as.character(cpu_threads)
)

reticulate::use_condaenv(
  ENV_NAME,
  required = TRUE
)

xgb <- reticulate::import(
  "xgboost",
  convert = TRUE
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

safe_num <- function(x) {
  suppressWarnings(
    as.numeric(
      as.character(x)
    )
  )
}

fit_encoder <- function(
    d,
    numeric_vars,
    categorical_vars
) {
  cat_levels <- lapply(
    categorical_vars,
    function(v) {
      x <- trimws(
        as.character(
          d[[v]]
        )
      )

      x[
        is.na(x) |
          x == ""
      ] <- "__UNKNOWN__"

      unique(
        c(
          sort(unique(x)),
          "__UNKNOWN__",
          "__OTHER__"
        )
      )
    }
  )

  names(cat_levels) <- categorical_vars

  list(
    numeric_vars = numeric_vars,
    categorical_vars = categorical_vars,
    categorical_levels = cat_levels
  )
}

encode_dense <- function(
    d,
    encoder
) {
  n <- nrow(d)

  n_numeric <- length(
    encoder$numeric_vars
  )

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

  feature_names <- character(
    ncol(X)
  )

  parent_names <- character(
    ncol(X)
  )

  j <- 1L

  for (v in encoder$numeric_vars) {
    X[, j] <- safe_num(
      d[[v]]
    )

    feature_names[j] <- v
    parent_names[j] <- v

    j <- j + 1L
  }

  for (v in encoder$categorical_vars) {
    lev <- encoder$categorical_levels[[v]]

    x <- trimws(
      as.character(
        d[[v]]
      )
    )

    x[
      is.na(x) |
        x == ""
    ] <- "__UNKNOWN__"

    x[
      !x %in% lev
    ] <- "__OTHER__"

    idx <- match(
      x,
      lev
    )

    cols <- j:(
      j + length(lev) - 1L
    )

    X[
      cbind(
        seq_len(n),
        cols[idx]
      )
    ] <- 1

    feature_names[cols] <- paste0(
      v,
      "__",
      make.names(
        lev,
        unique = TRUE
      )
    )

    parent_names[cols] <- v

    j <- max(cols) + 1L
  }

  colnames(X) <- feature_names
  attr(X, "parent_predictor") <- parent_names

  storage.mode(X) <- "double"

  X
}

variant_column_mask <- function(
    X,
    keep_predictors
) {
  parent <- attr(
    X,
    "parent_predictor"
  )

  if (
    is.null(parent) ||
      length(parent) != ncol(X)
  ) {
    stop(
      "Encoded parent-predictor map is missing.",
      call. = FALSE
    )
  }

  parent %in% keep_predictors
}

extract_best_iteration <- function(
    model,
    fallback
) {
  out <- tryCatch(
    as.numeric(
      reticulate::py_to_r(
        model$best_iteration
      )
    ),
    error = function(e) NA_real_
  )

  if (
    length(out) >= 1L &&
      is.finite(out[1L])
  ) {
    return(
      as.integer(
        out[1L] + 1L
      )
    )
  }

  as.integer(fallback)
}

extract_best_score <- function(model) {
  out <- tryCatch(
    as.numeric(
      reticulate::py_to_r(
        model$best_score
      )
    ),
    error = function(e) NA_real_
  )

  if (
    length(out) >= 1L &&
      is.finite(out[1L])
  ) {
    return(out[1L])
  }

  NA_real_
}

as_quantile_matrix <- function(
    pred,
    n,
    k
) {
  p <- tryCatch(
    reticulate::py_to_r(pred),
    error = function(e) pred
  )

  if (
    is.matrix(p) ||
      is.data.frame(p)
  ) {
    p <- as.matrix(p)
  } else {
    p <- matrix(
      as.numeric(p),
      nrow = n,
      ncol = k,
      byrow = TRUE
    )
  }

  if (
    nrow(p) != n ||
      ncol(p) != k
  ) {
    stop(
      "Unexpected quantile prediction shape.",
      call. = FALSE
    )
  }

  storage.mode(p) <- "double"

  p
}

rearrange_three_quantiles <- function(p) {
  if (
    ncol(p) != 3L
  ) {
    stop(
      "Expected exactly 3 quantiles.",
      call. = FALSE
    )
  }

  lo <- pmin(
    p[, 1L],
    p[, 2L],
    p[, 3L]
  )

  hi <- pmax(
    p[, 1L],
    p[, 2L],
    p[, 3L]
  )

  mid <- rowSums(p) -
    lo -
    hi

  cbind(
    q10 = lo,
    q50 = mid,
    q90 = hi
  )
}

pinball_loss <- function(
    y,
    q,
    alpha
) {
  e <- y - q

  mean(
    ifelse(
      e >= 0,
      alpha * e,
      (alpha - 1) * e
    ),
    na.rm = TRUE
  )
}

# -----------------------------------------------------------------------------
# Load data / predictor definitions
# -----------------------------------------------------------------------------

dt <- as.data.table(
  readRDS(
    dataset_file
  )
)

types <- fread(
  types_file
)

hp_table <- fread(
  hp_file
)

numeric_predictors <- types[
  category == "numeric",
  predictor
]

categorical_predictors <- types[
  category == "categorical",
  predictor
]

all_predictors <- c(
  numeric_predictors,
  categorical_predictors
)

dt[
  ,
  hospital_days := safe_num(hospital_days)
]

dt[
  ,
  icu_days := safe_num(icu_days)
]

dt[
  ,
  vent_days := safe_num(vent_days)
]

dt[
  hospital_days < 0,
  hospital_days := NA_real_
]

dt[
  icu_days < 0,
  icu_days := NA_real_
]

dt[
  vent_days < 0,
  vent_days := NA_real_
]

# -----------------------------------------------------------------------------
# Same variants as script 18
# -----------------------------------------------------------------------------

cranial_injury_predictors <- c(
  "dx_concussion",
  "dx_cerebral_edema_traumatic",
  "dx_diffuse_axonal_injury",
  "dx_focal_contusion_or_iph",
  "dx_epidural_hematoma",
  "dx_subdural_hematoma",
  "dx_subarachnoid_hemorrhage",
  "dx_other_intracranial_injury",
  "dx_cranial_skull_fracture",
  "dx_facial_fracture",
  "dx_other_skull_or_facial_fracture"
)

noncranial_injury_predictors <- c(
  "dx_spinal_cord_injury",
  "dx_neck_vascular_injury",
  "dx_thoracic_injury",
  "dx_abdominal_pelvic_injury",
  "dx_upper_extremity_injury",
  "dx_lower_extremity_injury"
)

pmhx_predictors <- grep(
  "^pmhx_",
  all_predictors,
  value = TRUE
)

core_neuro_predictors <- intersect(
  c(
    "age",
    "sex_clean",
    "gcs_eye_clean",
    "gcs_verbal_clean",
    "gcs_motor_clean",
    "gcsq_eye_obstruction_recovered",
    "gcsq_unknown_recovered",
    "sbp_clean",
    "pulse_clean",
    "rr_clean",
    "spo2_clean",
    "temperature_c_recovered",
    "pupil_clean",
    cranial_injury_predictors
  ),
  all_predictors
)

core_neuro_plus_resp_predictors <- unique(
  c(
    core_neuro_predictors,
    "respiratoryassistance_clean",
    "supplemental_oxygen_recovered"
  )
)

variant_keep <- list(
  FULL_REFERENCE =
    all_predictors,
  NO_RESP_ASSISTANCE =
    setdiff(
      all_predictors,
      "respiratoryassistance_clean"
    ),
  NO_RESP_SUPPORT =
    setdiff(
      all_predictors,
      c(
        "respiratoryassistance_clean",
        "supplemental_oxygen_recovered"
      )
    ),
  NO_GCS_AIRWAY_QUALIFIERS =
    setdiff(
      all_predictors,
      c(
        "gcsq_intubated_recovered",
        "gcsq_sedated_paralyzed_recovered"
      )
    ),
  NO_AIRWAY_PROXY_SET =
    setdiff(
      all_predictors,
      c(
        "gcsq_intubated_recovered",
        "gcsq_sedated_paralyzed_recovered",
        "respiratoryassistance_clean",
        "supplemental_oxygen_recovered"
      )
    ),
  NO_HELMET =
    setdiff(
      all_predictors,
      "helmet_use_recovered"
    ),
  NO_MECHANISM_HELMET =
    setdiff(
      all_predictors,
      c(
        "mechanism_clean",
        "helmet_use_recovered"
      )
    ),
  NO_NONCRANIAL_INJURY_PHENOTYPES =
    setdiff(
      all_predictors,
      noncranial_injury_predictors
    ),
  NO_PMHX =
    setdiff(
      all_predictors,
      pmhx_predictors
    ),
  CORE_NEURO =
    core_neuro_predictors,
  CORE_NEURO_PLUS_RESP =
    core_neuro_plus_resp_predictors
)

folds <- list(
  TEST_2022 = list(
    train_years = 2020L,
    tune_year = 2021L,
    dev_years = 2020:2021,
    test_year = 2022L
  ),
  TEST_2023 = list(
    train_years = 2020:2021,
    tune_year = 2022L,
    dev_years = 2020:2022,
    test_year = 2023L
  ),
  TEST_2024 = list(
    train_years = 2020:2022,
    tune_year = 2023L,
    dev_years = 2020:2023,
    test_year = 2024L
  )
)

duration_specs <- list(
  list(
    id = "hospital_los",
    label = "Hospital LOS",
    y_var = "hospital_days",
    architecture_endpoint = "hlos_trajectory_final",
    lower_bound = 0,
    risk_filter = function(x) {
      !is.na(x) &
        x >= 0
    }
  ),
  list(
    id = "icu_los_conditional",
    label = "ICU LOS conditional on ICU use",
    y_var = "icu_days",
    architecture_endpoint = "icu_trajectory_final",
    lower_bound = 1,
    risk_filter = function(x) {
      !is.na(x) &
        x > 0
    }
  ),
  list(
    id = "ventilator_days_conditional",
    label = "Ventilator duration conditional on ventilation",
    y_var = "vent_days",
    architecture_endpoint = "ventilation_trajectory_final",
    lower_bound = 1,
    risk_filter = function(x) {
      !is.na(x) &
        x > 0
    }
  )
)

ALPHAS <- c(
  0.10,
  0.50,
  0.90
)

ETA <- 0.05
MAX_ROUNDS <- 3000L
EARLY_STOP <- 75L

results <- list()
run_qc <- list()

# -----------------------------------------------------------------------------
# Rolling temporal duration stress test
# -----------------------------------------------------------------------------

for (fold_name in names(folds)) {
  fold <- folds[[fold_name]]

  cat(
    "\n============================================================\n",
    "DURATION STRESS FOLD: ",
    fold_name,
    "\n",
    "============================================================\n",
    sep = ""
  )

  idx_train <- which(
    dt$admission_year %in%
      fold$train_years
  )

  idx_tune <- which(
    dt$admission_year ==
      fold$tune_year
  )

  idx_dev <- which(
    dt$admission_year %in%
      fold$dev_years
  )

  idx_test <- which(
    dt$admission_year ==
      fold$test_year
  )

  encoder_tune <- fit_encoder(
    dt[idx_train],
    numeric_predictors,
    categorical_predictors
  )

  X_train <- encode_dense(
    dt[idx_train],
    encoder_tune
  )

  X_tune <- encode_dense(
    dt[idx_tune],
    encoder_tune
  )

  encoder_final <- fit_encoder(
    dt[idx_dev],
    numeric_predictors,
    categorical_predictors
  )

  X_dev <- encode_dense(
    dt[idx_dev],
    encoder_final
  )

  X_test <- encode_dense(
    dt[idx_test],
    encoder_final
  )

  for (spec in duration_specs) {
    cat(
      "\n--- ",
      spec$label,
      " ---\n",
      sep = ""
    )

    hp <- hp_table[
      hp_table[["fold_id"]] == fold_name &
        hp_table[["endpoint_id"]] ==
          spec$architecture_endpoint
    ]

    if (
      nrow(hp) != 1L
    ) {
      stop(
        "Could not uniquely recover fold-specific architecture for ",
        fold_name,
        " / ",
        spec$architecture_endpoint,
        call. = FALSE
      )
    }

    required_hp <- c(
      "max_depth",
      "min_child_weight",
      "subsample",
      "colsample_bytree",
      "lambda"
    )

    for (h in required_hp) {
      if (
        !h %in% names(hp) ||
          is.na(hp[[h]][1L]) ||
          !is.finite(
            safe_num(
              hp[[h]][1L]
            )
          )
      ) {
        stop(
          "Invalid hyperparameter ",
          h,
          " for ",
          fold_name,
          " / ",
          spec$label,
          call. = FALSE
        )
      }
    }

    y_all <- safe_num(
      dt[[spec$y_var]]
    )

    train_keep <- spec$risk_filter(
      y_all[idx_train]
    )

    tune_keep <- spec$risk_filter(
      y_all[idx_tune]
    )

    dev_keep <- spec$risk_filter(
      y_all[idx_dev]
    )

    test_keep <- spec$risk_filter(
      y_all[idx_test]
    )

    y_train_raw <- y_all[
      idx_train
    ][
      train_keep
    ]

    y_tune_raw <- y_all[
      idx_tune
    ][
      tune_keep
    ]

    y_dev_raw <- y_all[
      idx_dev
    ][
      dev_keep
    ]

    y_test <- y_all[
      idx_test
    ][
      test_keep
    ]

    y_train <- log1p(
      y_train_raw
    )

    y_tune <- log1p(
      y_tune_raw
    )

    y_dev <- log1p(
      y_dev_raw
    )

    for (variant_id in names(variant_keep)) {
      keep_predictors <- variant_keep[[variant_id]]

      cat(
        "  Variant: ",
        variant_id,
        "\n",
        sep = ""
      )

      cols_train <- variant_column_mask(
        X_train,
        keep_predictors
      )

      cols_tune <- variant_column_mask(
        X_tune,
        keep_predictors
      )

      cols_dev <- variant_column_mask(
        X_dev,
        keep_predictors
      )

      cols_test <- variant_column_mask(
        X_test,
        keep_predictors
      )

      if (
        !identical(
          colnames(X_train)[cols_train],
          colnames(X_tune)[cols_tune]
        )
      ) {
        stop(
          "Train/tune feature mismatch for ",
          variant_id,
          call. = FALSE
        )
      }

      if (
        !identical(
          colnames(X_dev)[cols_dev],
          colnames(X_test)[cols_test]
        )
      ) {
        stop(
          "Dev/test feature mismatch for ",
          variant_id,
          call. = FALSE
        )
      }

      params <- reticulate::dict(
        objective = "reg:quantileerror",
        quantile_alpha = ALPHAS,
        eta = ETA,
        max_depth = as.integer(hp$max_depth),
        min_child_weight = as.numeric(hp$min_child_weight),
        subsample = as.numeric(hp$subsample),
        colsample_bytree = as.numeric(hp$colsample_bytree),
        lambda = as.numeric(hp$lambda),
        tree_method = "hist",
        device = "cuda",
        nthread = as.integer(cpu_threads),
        seed = 20260910L
      )

      dtrain <- xgb$DMatrix(
        X_train[
          train_keep,
          cols_train,
          drop = FALSE
        ],
        label = y_train
      )

      dtune <- xgb$DMatrix(
        X_tune[
          tune_keep,
          cols_tune,
          drop = FALSE
        ],
        label = y_tune
      )

      tune_fit <- xgb$train(
        params = params,
        dtrain = dtrain,
        num_boost_round = as.integer(MAX_ROUNDS),
        evals = list(
          reticulate::tuple(
            dtrain,
            "train"
          ),
          reticulate::tuple(
            dtune,
            "tune"
          )
        ),
        early_stopping_rounds = as.integer(EARLY_STOP),
        maximize = FALSE,
        verbose_eval = FALSE
      )

      rounds <- extract_best_iteration(
        tune_fit,
        MAX_ROUNDS
      )

      ddev <- xgb$DMatrix(
        X_dev[
          dev_keep,
          cols_dev,
          drop = FALSE
        ],
        label = y_dev
      )

      dtest <- xgb$DMatrix(
        X_test[
          test_keep,
          cols_test,
          drop = FALSE
        ]
      )

      final_fit <- xgb$train(
        params = params,
        dtrain = ddev,
        num_boost_round = as.integer(rounds),
        verbose_eval = FALSE
      )

      pred_trans_raw <- final_fit$predict(
        dtest
      )

      pred_trans <- as_quantile_matrix(
        pred_trans_raw,
        length(y_test),
        3L
      )

      pred_days_raw <- expm1(
        pred_trans
      )

      pred_days_raw <- pmax(
        pred_days_raw,
        spec$lower_bound
      )

      raw_crossing <- (
        pred_days_raw[, 1L] >
          pred_days_raw[, 2L]
      ) |
        (
          pred_days_raw[, 2L] >
            pred_days_raw[, 3L]
        )

      pred_days <- rearrange_three_quantiles(
        pred_days_raw
      )

      q10 <- pred_days[, 1L]
      q50 <- pred_days[, 2L]
      q90 <- pred_days[, 3L]

      q10_empirical <- mean(
        y_test <= q10
      )

      q50_empirical <- mean(
        y_test <= q50
      )

      q90_empirical <- mean(
        y_test <= q90
      )

      mean_pinball <- mean(
        c(
          pinball_loss(
            y_test,
            q10,
            0.10
          ),
          pinball_loss(
            y_test,
            q50,
            0.50
          ),
          pinball_loss(
            y_test,
            q90,
            0.90
          )
        )
      )

      coverage <- mean(
        y_test >= q10 &
          y_test <= q90
      )

      results[[
        length(results) + 1L
      ]] <- data.table(
        fold_id = fold_name,
        test_year = fold$test_year,
        duration_id = spec$id,
        duration_label = spec$label,
        architecture_endpoint =
          spec$architecture_endpoint,
        variant_id = variant_id,
        N = length(y_test),
        selected_rounds = rounds,
        median_MAE_days =
          mean(
            abs(
              y_test -
                q50
            )
          ),
        median_bias_days =
          median(
            q50 -
              y_test
          ),
        mean_pinball =
          mean_pinball,
        central_80_coverage =
          coverage,
        absolute_80_coverage_error =
          abs(
            coverage -
              0.80
          ),
        median_80PI_width_days =
          median(
            q90 -
              q10
          ),
        q10_empirical_cdf =
          q10_empirical,
        q50_empirical_cdf =
          q50_empirical,
        q90_empirical_cdf =
          q90_empirical,
        quantile_calibration_MAE =
          mean(
            abs(
              c(
                q10_empirical - 0.10,
                q50_empirical - 0.50,
                q90_empirical - 0.90
              )
            )
          ),
        raw_crossing_rate =
          mean(
            raw_crossing
          )
      )

      run_qc[[
        length(run_qc) + 1L
      ]] <- data.table(
        fold_id = fold_name,
        test_year = fold$test_year,
        duration_id = spec$id,
        duration_label = spec$label,
        variant_id = variant_id,
        n_raw_predictors =
          length(
            keep_predictors
          ),
        n_encoded_features_train =
          sum(
            cols_train
          ),
        n_encoded_features_final =
          sum(
            cols_dev
          ),
        selected_rounds = rounds,
        tune_score =
          extract_best_score(
            tune_fit
          )
      )

      rm(
        dtrain,
        dtune,
        tune_fit,
        ddev,
        dtest,
        final_fit,
        pred_trans_raw,
        pred_trans,
        pred_days_raw,
        pred_days
      )

      gc()
    }
  }

  rm(
    X_train,
    X_tune,
    X_dev,
    X_test,
    encoder_tune,
    encoder_final
  )

  gc()
}

# -----------------------------------------------------------------------------
# Save / compare to full reference
# -----------------------------------------------------------------------------

results <- rbindlist(
  results,
  fill = TRUE
)

run_qc <- rbindlist(
  run_qc,
  fill = TRUE
)

fwrite(
  results,
  file.path(
    out_dir,
    "01_DURATION_ABLATION_METRICS.csv"
  )
)

fwrite(
  run_qc,
  file.path(
    out_dir,
    "02_DURATION_ABLATION_RUN_QC.csv"
  )
)

reference <- results[
  variant_id == "FULL_REFERENCE"
]

delta <- merge(
  results,
  reference,
  by = c(
    "fold_id",
    "test_year",
    "duration_id",
    "duration_label",
    "architecture_endpoint"
  ),
  suffixes = c(
    "",
    "_reference"
  )
)

metrics_for_delta <- c(
  "median_MAE_days",
  "median_bias_days",
  "mean_pinball",
  "central_80_coverage",
  "absolute_80_coverage_error",
  "median_80PI_width_days",
  "q10_empirical_cdf",
  "q50_empirical_cdf",
  "q90_empirical_cdf",
  "quantile_calibration_MAE",
  "raw_crossing_rate"
)

for (m in metrics_for_delta) {
  delta[
    ,
    paste0(
      "delta_",
      m
    ) :=
      get(m) -
        get(
          paste0(
            m,
            "_reference"
          )
        )
  ]
}

fwrite(
  delta,
  file.path(
    out_dir,
    "03_DURATION_ABLATION_DELTAS_VS_FULL.csv"
  )
)

summary <- delta[
  variant_id != "FULL_REFERENCE",
  .(
    n_folds = .N,
    median_delta_MAE_days =
      median(
        delta_median_MAE_days,
        na.rm = TRUE
      ),
    worst_delta_MAE_days =
      max(
        delta_median_MAE_days,
        na.rm = TRUE
      ),
    median_delta_pinball =
      median(
        delta_mean_pinball,
        na.rm = TRUE
      ),
    worst_delta_pinball =
      max(
        delta_mean_pinball,
        na.rm = TRUE
      ),
    median_delta_absolute_coverage_error =
      median(
        delta_absolute_80_coverage_error,
        na.rm = TRUE
      ),
    worst_delta_absolute_coverage_error =
      max(
        delta_absolute_80_coverage_error,
        na.rm = TRUE
      ),
    median_delta_quantile_calibration_MAE =
      median(
        delta_quantile_calibration_MAE,
        na.rm = TRUE
      ),
    worst_delta_quantile_calibration_MAE =
      max(
        delta_quantile_calibration_MAE,
        na.rm = TRUE
      )
  ),
  by = .(
    duration_id,
    duration_label,
    variant_id
  )
]

summary[
  ,
  review_flag := fcase(
    median_delta_MAE_days < -0.05 &
      median_delta_pinball < 0,
    "POSSIBLE IMPROVEMENT - REVIEW",
    median_delta_MAE_days <= 0.05 &
      worst_delta_MAE_days <= 0.15 &
      median_delta_absolute_coverage_error <= 0.01,
    "PERFORMANCE-NEUTRAL CANDIDATE FOR SIMPLIFICATION",
    median_delta_MAE_days > 0.20 |
      worst_delta_MAE_days > 0.35 |
      median_delta_pinball > 0.02,
    "PREDICTIVE INFORMATION LIKELY LOST",
    default =
      "MIXED - REVIEW TEMPORAL PATTERN"
  )
]

summary[
  ,
  rule_note :=
    "Heuristic triage only; inspect all folds and quantile calibration before model revision."
]

fwrite(
  summary,
  file.path(
    out_dir,
    "04_DURATION_TEMPORAL_ABLATION_SUMMARY.csv"
  )
)

summary_lines <- c(
  "TBI-TRACT DURATION PREDICTOR STRESS TEST COMPLETE",
  "",
  "Duration models were evaluated with:",
  "  median absolute error",
  "  mean pinball loss",
  "  empirical Q10/Q50/Q90 calibration",
  "  central 80% prediction-interval coverage",
  "  prediction-interval width",
  "  raw quantile crossing",
  "",
  "The categorical trajectory architecture chosen inside each rolling fold",
  "was reused for the corresponding duration model.",
  "",
  "Start review with:",
  "  03_DURATION_ABLATION_DELTAS_VS_FULL.csv",
  "  04_DURATION_TEMPORAL_ABLATION_SUMMARY.csv"
)

writeLines(
  summary_lines,
  file.path(
    out_dir,
    "DURATION_STRESS_TEST_SUMMARY.txt"
  )
)

cat(
  "\n============================================================\n",
  paste(
    summary_lines,
    collapse = "\n"
  ),
  "\n============================================================\n",
  sep = ""
)

print(
  summary[
    order(
      duration_label,
      variant_id
    )
  ]
)
