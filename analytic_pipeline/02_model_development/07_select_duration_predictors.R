# =============================================================================
# 24_run_TBI_TRACT_pragmatic_duration_finalization_GPU.R
#
# PURPOSE
#   Compare CURRENT_REFERENCE vs an exact PRAGMATIC_CLINICAL predictor set for
#   HLOS, ICU LOS, and ventilator-duration Q10/Q50/Q90 models.
#
#   Unlike v1, this script DOES NOT borrow classification hyperparameters from
#   script 21. Each duration endpoint / temporal fold / predictor variant is
#   fully tuned over the original prespecified structural XGBoost grid using
#   the quantile objective itself.
#
# PRAGMATIC_CLINICAL removes:
#   helmet_use_recovered
#   respiratoryassistance_clean
#   gcsq_intubated_recovered
#   gcsq_sedated_paralyzed_recovered
# while retaining supplemental_oxygen_recovered.
#
# DESIGN
#   train 2020      | tune 2021 | refit 2020-21 | test 2022
#   train 2020-21   | tune 2022 | refit 2020-22 | test 2023
#   train 2020-22   | tune 2023 | refit 2020-23 | test 2024
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

tuning_dir <- file.path(
  output_dir,
  "TBI_TRACT_TEMPORAL_HYPERPARAMETER_TUNING"
)

dataset_file <- file.path(
  methods_dir,
  "frozen_methods_dataset_retained_2020_2024.rds"
)

types_file <- file.path(
  methods_dir,
  "11_FROZEN_PREDICTOR_TYPES.csv"
)

grid_file <- file.path(
  tuning_dir,
  "01_PRESPECIFIED_SEARCH_GRID.csv"
)

needed_files <- c(
  dataset_file,
  types_file,
  grid_file
)

if (!all(file.exists(needed_files))) {
  stop(
    "Missing prerequisite file(s):\n",
    paste0(
      "  ",
      needed_files[!file.exists(needed_files)],
      collapse = "\n"
    ),
    call. = FALSE
  )
}

out_dir <- file.path(
  output_dir,
  "TBI_TRACT_PRAGMATIC_DURATION_FINALIZATION"
)

grid_out_dir <- file.path(
  out_dir,
  "grid_search"
)

dir.create(
  out_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  grid_out_dir,
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

search_grid <- fread(
  grid_file
)

required_grid_cols <- c(
  "config_id",
  "max_depth",
  "min_child_weight",
  "subsample",
  "colsample_bytree",
  "lambda"
)

if (!all(required_grid_cols %in% names(search_grid))) {
  stop("Prespecified search grid is malformed.", call. = FALSE)
}

search_grid <- search_grid[, ..required_grid_cols]

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
# Exact finalist predictor sets
# -----------------------------------------------------------------------------

pragmatic_removed <- c(
  "helmet_use_recovered",
  "respiratoryassistance_clean"
)

pragmatic_clinical <- setdiff(
  all_predictors,
  pragmatic_removed
)

required_pragmatic_retained <- c(
  "gcsq_intubated_recovered",
  "gcsq_sedated_paralyzed_recovered",
  "supplemental_oxygen_recovered"
)

if (!all(required_pragmatic_retained %in% pragmatic_clinical)) {
  stop(
    "PRAGMATIC_CLINICAL is missing a required retained airway-context predictor.",
    call. = FALSE
  )
}

variant_keep <- list(
  CURRENT_REFERENCE =
    all_predictors,
  PRAGMATIC_CLINICAL =
    pragmatic_clinical
)

fwrite(
  rbindlist(
    lapply(
      names(variant_keep),
      function(v) {
        data.table(
          variant_id = v,
          n_predictors = length(variant_keep[[v]]),
          removed_from_current = paste(
            setdiff(all_predictors, variant_keep[[v]]),
            collapse = ";"
          )
        )
      }
    )
  ),
  file.path(out_dir, "01_VARIANT_DEFINITIONS.csv")
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
selected_hp_rows <- list()

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

      grid_rows <- vector(
        "list",
        nrow(search_grid)
      )

      for (grid_i in seq_len(nrow(search_grid))) {
        g <- search_grid[grid_i]

        params_grid <- reticulate::dict(
          objective = "reg:quantileerror",
          quantile_alpha = ALPHAS,
          eta = ETA,
          max_depth = as.integer(g$max_depth),
          min_child_weight = as.numeric(g$min_child_weight),
          subsample = as.numeric(g$subsample),
          colsample_bytree = as.numeric(g$colsample_bytree),
          lambda = as.numeric(g$lambda),
          tree_method = "hist",
          device = "cuda",
          nthread = as.integer(cpu_threads),
          seed = 20260912L
        )

        grid_fit <- xgb$train(
          params = params_grid,
          dtrain = dtrain,
          num_boost_round = as.integer(MAX_ROUNDS),
          evals = list(
            reticulate::tuple(dtrain, "train"),
            reticulate::tuple(dtune, "tune")
          ),
          early_stopping_rounds = as.integer(EARLY_STOP),
          maximize = FALSE,
          verbose_eval = FALSE
        )

        grid_rows[[grid_i]] <- cbind(
          data.table(
            fold_id = fold_name,
            test_year = fold$test_year,
            duration_id = spec$id,
            duration_label = spec$label,
            variant_id = variant_id,
            best_iteration = extract_best_iteration(grid_fit, MAX_ROUNDS),
            tune_score = extract_best_score(grid_fit)
          ),
          g
        )

        rm(grid_fit)
        gc()
      }

      grid_table <- rbindlist(
        grid_rows,
        fill = TRUE
      )

      setorder(
        grid_table,
        tune_score,
        best_iteration
      )

      hp <- copy(grid_table[1L])

      hp[, selected_by :=
        "Minimum fold-specific pre-test quantile tuning loss"]

      selected_hp_rows[[
        length(selected_hp_rows) + 1L
      ]] <- hp

      fwrite(
        grid_table,
        file.path(
          grid_out_dir,
          paste0(
            fold_name,
            "__",
            spec$id,
            "__",
            variant_id,
            ".csv"
          )
        )
      )

      rounds <- as.integer(hp$best_iteration)

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
        seed = 20260912L
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
        selected_config_id = hp$config_id,
        tune_score = as.numeric(hp$tune_score)
      )

      rm(
        dtrain,
        dtune,
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

selected_hp <- rbindlist(
  selected_hp_rows,
  fill = TRUE
)

fwrite(
  selected_hp,
  file.path(
    out_dir,
    "02_SELECTED_DURATION_HYPERPARAMETERS.csv"
  )
)

fwrite(
  results,
  file.path(
    out_dir,
    "03_PRAGMATIC_DURATION_METRICS.csv"
  )
)

fwrite(
  run_qc,
  file.path(
    out_dir,
    "04_PRAGMATIC_DURATION_RUN_QC.csv"
  )
)

reference <- results[
  variant_id == "CURRENT_REFERENCE"
]

delta <- merge(
  results,
  reference,
  by = c(
    "fold_id",
    "test_year",
    "duration_id",
    "duration_label"
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
    "05_PRAGMATIC_DURATION_DELTAS_VS_CURRENT.csv"
  )
)

summary <- delta[
  variant_id == "PRAGMATIC_CLINICAL",
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
    "06_PRAGMATIC_DURATION_TEMPORAL_SUMMARY.csv"
  )
)

summary_lines <- c(
  "TBI-TRACT PRAGMATIC DURATION TEST v2 COMPLETE",
  "",
  "Duration models were evaluated with:",
  "  median absolute error",
  "  mean pinball loss",
  "  empirical Q10/Q50/Q90 calibration",
  "  central 80% prediction-interval coverage",
  "  prediction-interval width",
  "  raw quantile crossing",
  "",
  "Each duration endpoint and predictor variant was fully retuned over the",
  "original prespecified XGBoost structural grid using quantile loss.",
  "",
  "Start review with:",
  "  05_PRAGMATIC_DURATION_DELTAS_VS_CURRENT.csv",
  "  06_PRAGMATIC_DURATION_TEMPORAL_SUMMARY.csv"
)

writeLines(
  summary_lines,
  file.path(
    out_dir,
    "PRAGMATIC_DURATION_SUMMARY.txt"
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
