# =============================================================================
# 09_TBI_TRACT_temporal_hyperparameter_tuning_v3.R
#
# PRESPECIFIED TEMPORAL HYPERPARAMETER TUNING FOR TBI-TRACT
#
# IMPORTANT:
#   - Uses 2020-2022 for training.
#   - Uses 2023 ONLY for hyperparameter + boosting-round selection.
#   - DOES NOT LOAD OR EVALUATE 2024 OUTCOMES.
#   - Uses the exact frozen predictor set produced by script 08.
#
# This addresses the remaining methodological weakness that the previous
# XGBoost structural hyperparameters were fixed heuristically rather than
# selected within the development period.
#
# Run after:
#   source("R/08_TBI_TRACT_methods_completion_audit.R")
# =============================================================================

rm(list = ls())
gc()

required <- c("data.table", "xgboost")
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
  library(xgboost)
})

config_candidates <- c(
  file.path(getwd(), "R", "00_config.R"),
  "R/00_config.R"
)
config_file <- config_candidates[file.exists(config_candidates)][1]
if (is.na(config_file) || length(config_file) == 0L) {
  stop("Could not find R/00_config.R.", call. = FALSE)
}
source(config_file)

methods_dir <- file.path(
  output_dir,
  "METHODS_COMPLETION_TBI_TRACT"
)

dataset_file <- file.path(
  methods_dir,
  "frozen_methods_dataset_retained_2020_2024.rds"
)

types_file <- file.path(
  methods_dir,
  "11_FROZEN_PREDICTOR_TYPES.csv"
)

if (!file.exists(dataset_file) || !file.exists(types_file)) {
  stop(
    "Run 08_TBI_TRACT_methods_completion_audit.R first.",
    call. = FALSE
  )
}

tune_dir <- file.path(
  output_dir,
  "TBI_TRACT_TEMPORAL_HYPERPARAMETER_TUNING"
)
dir.create(tune_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(20260907L)

# -----------------------------------------------------------------------------
# Compute
# -----------------------------------------------------------------------------

logical_cores <- parallel::detectCores(logical = TRUE)
if (is.na(logical_cores)) logical_cores <- 32L
cpu_threads <- max(1L, min(28L, logical_cores - 4L))
setDTthreads(cpu_threads)

detect_xgb_compute_params <- function(cpu_threads) {
  set.seed(1L)
  x <- matrix(rnorm(800), nrow = 100, ncol = 8)
  y <- rbinom(100, 1, 0.4)
  d <- xgb.DMatrix(x, label = y, missing = NA_real_)

  booster_uses_cuda <- function(fit) {
    cfg <- tryCatch(
      xgb.config(fit),
      error = function(e) NA_character_
    )

    if (
      length(cfg) == 1L &&
        !is.na(cfg)
    ) {
      cfg_lower <- tolower(cfg)

      # Reject explicit CPU fallback.
      if (
        grepl('"device"[[:space:]]*:[[:space:]]*"cpu"', cfg_lower) ||
          grepl('"device"[[:space:]]*:[[:space:]]*"cpu:', cfg_lower)
      ) {
        return(FALSE)
      }

      # Accept CUDA devices/configuration.
      if (
        grepl('"device"[[:space:]]*:[[:space:]]*"cuda', cfg_lower) ||
          grepl("gpu_id", cfg_lower) &&
            !grepl('"gpu_id"[[:space:]]*:[[:space:]]*-1', cfg_lower)
      ) {
        return(TRUE)
      }
    }

    FALSE
  }

  modern <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    tree_method = "hist",
    device = "cuda",
    max_depth = 2,
    eta = 0.2,
    nthread = cpu_threads,
    seed = 1L
  )

  fit_modern <- tryCatch(
    suppressWarnings(
      xgb.train(
        params = modern,
        data = d,
        nrounds = 2,
        verbose = 0
      )
    ),
    error = function(e) NULL
  )

  if (
    !is.null(fit_modern) &&
      booster_uses_cuda(fit_modern)
  ) {
    return(list(
      mode = "CUDA",
      extra = list(
        tree_method = "hist",
        device = "cuda"
      )
    ))
  }

  legacy <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    tree_method = "gpu_hist",
    predictor = "gpu_predictor",
    max_depth = 2,
    eta = 0.2,
    nthread = cpu_threads,
    seed = 1L
  )

  fit_legacy <- tryCatch(
    suppressWarnings(
      xgb.train(
        params = legacy,
        data = d,
        nrounds = 2,
        verbose = 0
      )
    ),
    error = function(e) NULL
  )

  if (
    !is.null(fit_legacy) &&
      booster_uses_cuda(fit_legacy)
  ) {
    return(list(
      mode = "CUDA legacy",
      extra = list(
        tree_method = "gpu_hist",
        predictor = "gpu_predictor"
      )
    ))
  }

  # If CUDA is unavailable in this R/xgboost session, deliberately use CPU
  # rather than requesting CUDA and silently falling back every fit.
  list(
    mode = "CPU",
    extra = list(
      tree_method = "hist"
    )
  )
}

compute <- detect_xgb_compute_params(cpu_threads)

fwrite(
  data.table(
    compute_mode = compute$mode,
    cpu_threads = cpu_threads,
    xgboost_version =
      as.character(packageVersion("xgboost"))
  ),
  file.path(
    tune_dir,
    "00_COMPUTE_ENVIRONMENT.csv"
  )
)


cat(
  "\nXGBoost compute mode selected: ",
  compute$mode,
  "\n",
  sep = ""
)

if (compute$mode == "CPU") {
  cat(
    "CUDA was not verified in this R/xgboost session; ",
    "the tuning run will deliberately use CPU rather than silent GPU fallback.\n",
    sep = ""
  )
}

# -----------------------------------------------------------------------------
# Data + predictors
# -----------------------------------------------------------------------------

dt <- as.data.table(readRDS(dataset_file))

# HARD RULE: this tuning script never uses 2024 rows.
dev <- dt[admission_year <= 2023]
rm(dt)
gc()

if (any(dev$admission_year == 2024L)) {
  stop(
    "2024 data entered tuning dataset. Abort.",
    call. = FALSE
  )
}

types <- fread(types_file)
numeric_predictors <- types[
  category == "numeric",
  predictor
]
categorical_predictors <- types[
  category == "categorical",
  predictor
]

safe_num <- function(x) {
  suppressWarnings(as.numeric(as.character(x)))
}

fit_encoder <- function(d, numeric_vars, categorical_vars) {
  cat_levels <- lapply(
    categorical_vars,
    function(v) {
      x <- trimws(as.character(d[[v]]))
      x[is.na(x) | x == ""] <- "__UNKNOWN__"
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

encode_dense <- function(d, encoder) {
  n <- nrow(d)
  parts <- vector(
    "list",
    length(encoder$numeric_vars) +
      length(encoder$categorical_vars)
  )
  k <- 1L

  for (v in encoder$numeric_vars) {
    x <- safe_num(d[[v]])
    m <- matrix(x, nrow = n, ncol = 1)
    colnames(m) <- v
    parts[[k]] <- m
    k <- k + 1L
  }

  for (v in encoder$categorical_vars) {
    lev <- encoder$categorical_levels[[v]]
    x <- trimws(as.character(d[[v]]))
    x[is.na(x) | x == ""] <- "__UNKNOWN__"
    x[!x %in% lev] <- "__OTHER__"

    j <- match(x, lev)
    m <- matrix(
      0,
      nrow = n,
      ncol = length(lev)
    )
    m[cbind(seq_len(n), j)] <- 1
    colnames(m) <- paste0(
      v,
      "__",
      make.names(lev, unique = TRUE)
    )
    parts[[k]] <- m
    k <- k + 1L
  }

  X <- do.call(cbind, parts)
  storage.mode(X) <- "double"
  X
}

idx_train <- which(
  dev$admission_year %in% 2020:2022
)
idx_tune <- which(
  dev$admission_year == 2023L
)

# Encoder is fitted on 2020-22 only for strict temporal tuning.
encoder <- fit_encoder(
  dev[idx_train],
  numeric_predictors,
  categorical_predictors
)

X <- encode_dense(
  dev,
  encoder
)

saveRDS(
  encoder,
  file.path(
    tune_dir,
    "encoder_2020_2022_for_temporal_tuning.rds"
  )
)

# -----------------------------------------------------------------------------
# Prespecified randomized search space
# -----------------------------------------------------------------------------

full_grid <- CJ(
  max_depth = c(2L, 3L, 4L, 5L),
  min_child_weight = c(5, 10, 20),
  subsample = c(0.75, 0.90, 1.00),
  colsample_bytree = c(0.75, 0.90, 1.00),
  lambda = c(1, 5),
  unique = TRUE
)

# Current heuristic default is always included as a benchmark.
default_row <- data.table(
  max_depth = 3L,
  min_child_weight = 10,
  subsample = 0.85,
  colsample_bytree = 0.85,
  lambda = 1
)

# Select 23 grid points before any outcome-specific model fitting.
set.seed(20260907L)
sampled_grid <- full_grid[
  sample(
    .N,
    size = 23L,
    replace = FALSE
  )
]

search_grid <- unique(
  rbind(
    default_row,
    sampled_grid,
    fill = TRUE
  )
)

search_grid[
  ,
  config_id := sprintf(
    "cfg_%02d",
    seq_len(.N)
  )
]

setcolorder(
  search_grid,
  c(
    "config_id",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "lambda"
  )
)

fwrite(
  search_grid,
  file.path(
    tune_dir,
    "01_PRESPECIFIED_SEARCH_GRID.csv"
  )
)

# ETA is held fixed. Boosting rounds are selected by early stopping.
ETA_FIXED <- 0.05
MAX_ROUNDS <- 3000L
EARLY_STOP <- 75L

writeLines(
  c(
    "Hyperparameter selection was prespecified before fitting:",
    "  max_depth, min_child_weight, subsample, colsample_bytree, lambda",
    "  eta fixed at 0.05",
    "  2020-22 train",
    "  2023 validation",
    "  selection metric = minimum 2023 log loss / multiclass log loss",
    "  2024 is not loaded or evaluated in this script"
  ),
  file.path(
    tune_dir,
    "TUNING_PROTOCOL.txt"
  )
)

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

endpoint_specs <- list(
  list(
    endpoint = "discharge_3cat_final",
    label = "Discharge 3-class",
    type = "multiclass",
    levels = c(
      "Home/home health",
      "Post-acute facility",
      "Death/hospice"
    )
  ),
  list(
    endpoint = "icu_trajectory_final",
    label = "ICU trajectory",
    type = "multiclass",
    levels = c(
      "No ICU",
      "ICU 1-7 days",
      "ICU >=8 days"
    )
  ),
  list(
    endpoint = "vent_ge8_final",
    label = "Ventilation >=8 days",
    type = "binary"
  ),
  list(
    endpoint = "hlos_ge28_final",
    label = "Hospital LOS >=28 days",
    type = "binary"
  ),
  list(
    endpoint = "icp_pressure_monitor_final",
    label = "True ICP pressure monitor",
    type = "binary"
  ),
  list(
    endpoint = "craniotomy_craniectomy_final",
    label = "Craniotomy/craniectomy",
    type = "binary"
  )
)

missing_endpoints <- setdiff(
  vapply(
    endpoint_specs,
    function(z) z$endpoint,
    character(1)
  ),
  names(dev)
)

if (length(missing_endpoints) > 0L) {
  stop(
    "Frozen methods dataset lacks endpoint(s): ",
    paste(missing_endpoints, collapse = ", "),
    call. = FALSE
  )
}

# -----------------------------------------------------------------------------
# Tuning functions
# -----------------------------------------------------------------------------

extract_best_iteration <- function(model, fallback) {
  out <- tryCatch(
    suppressWarnings(
      as.numeric(
        model$best_iteration
      )
    ),
    error = function(e) NULL
  )

  if (
    !is.null(out) &&
      length(out) >= 1L &&
      is.finite(out[1L])
  ) {
    return(
      max(
        1L,
        as.integer(out[1L])
      )
    )
  }

  a <- tryCatch(
    suppressWarnings(
      as.numeric(
        xgb.attr(
          model,
          "best_iteration"
        )
      )
    ),
    error = function(e) NULL
  )

  if (
    !is.null(a) &&
      length(a) >= 1L &&
      is.finite(a[1L])
  ) {
    # Booster attribute is commonly zero-based.
    return(
      max(
        1L,
        as.integer(a[1L]) + 1L
      )
    )
  }

  as.integer(fallback)
}

extract_best_score <- function(model, metric_name) {
  # Preferred route: XGBoost's early-stopping metadata. This is more robust
  # across R package versions than assuming evaluation_log is populated.
  candidates <- list(
    tryCatch(model$best_score, error = function(e) NULL),
    tryCatch(xgb.attr(model, "best_score"), error = function(e) NULL)
  )

  for (z in candidates) {
    if (!is.null(z) && length(z) >= 1L) {
      val <- suppressWarnings(as.numeric(z[1L]))
      if (length(val) == 1L && is.finite(val)) {
        return(val)
      }
    }
  }

  # Fallback only for builds that do retain evaluation_log.
  elog <- tryCatch(
    as.data.table(model$evaluation_log),
    error = function(e) data.table()
  )

  if (ncol(elog) > 0L && nrow(elog) > 0L) {
    nms <- names(elog)

    normalize_metric_name <- function(x) {
      tolower(
        gsub(
          "[^a-z0-9]+",
          "",
          x
        )
      )
    }

    nms_norm <- normalize_metric_name(nms)
    metric_norm <- normalize_metric_name(metric_name)

    metric_candidates <- which(
      grepl(
        metric_norm,
        nms_norm,
        fixed = TRUE
      )
    )

    if (length(metric_candidates) > 0L) {
      candidate_norm <- nms_norm[metric_candidates]

      preferred <- metric_candidates[
        grepl(
          "2023|tune|valid|validation|eval|test",
          candidate_norm
        ) &
          !grepl(
            "^train|training",
            candidate_norm
          )
      ]

      if (length(preferred) == 0L) {
        nontrain <- metric_candidates[
          !grepl(
            "^train|training",
            candidate_norm
          )
        ]

        if (length(nontrain) >= 1L) {
          preferred <- nontrain
        }
      }

      if (length(preferred) >= 1L) {
        values <- suppressWarnings(
          as.numeric(
            elog[[nms[preferred[1L]]]]
          )
        )

        if (
          length(values) > 0L &&
            any(is.finite(values))
        ) {
          return(
            min(
              values,
              na.rm = TRUE
            )
          )
        }
      }
    }
  }

  # Last-resort diagnostics. At this point early stopping itself may not have
  # been recorded by the installed xgboost build.
  model_best_iteration <- tryCatch(
    model$best_iteration,
    error = function(e) NULL
  )
  attr_best_iteration <- tryCatch(
    xgb.attr(model, "best_iteration"),
    error = function(e) NULL
  )

  stop(
    paste0(
      "Could not recover XGBoost early-stopping best_score for metric '",
      metric_name,
      "'. model$best_iteration=",
      paste(model_best_iteration, collapse = ","),
      "; xgb.attr(best_iteration)=",
      paste(attr_best_iteration, collapse = ","),
      ". This indicates an xgboost-package compatibility issue rather than ",
      "a data/model failure."
    ),
    call. = FALSE
  )
}

extract_tune_score <- function(model, metric_name) {
  extract_best_score(
    model,
    metric_name
  )
}


tune_binary <- function(spec) {
  y_all <- as.integer(dev[[spec$endpoint]])

  tr <- idx_train[
    !is.na(y_all[idx_train]) &
      y_all[idx_train] %in% c(0L, 1L)
  ]
  va <- idx_tune[
    !is.na(y_all[idx_tune]) &
      y_all[idx_tune] %in% c(0L, 1L)
  ]

  dtr <- xgb.DMatrix(
    X[tr, , drop = FALSE],
    label = y_all[tr],
    missing = NA_real_
  )
  dva <- xgb.DMatrix(
    X[va, , drop = FALSE],
    label = y_all[va],
    missing = NA_real_
  )

  out <- vector(
    "list",
    nrow(search_grid)
  )

  for (i in seq_len(nrow(search_grid))) {
    g <- search_grid[i]

    params <- c(
      list(
        objective = "binary:logistic",
        eval_metric = "logloss",
        eta = ETA_FIXED,
        max_depth = g$max_depth,
        min_child_weight =
          g$min_child_weight,
        subsample = g$subsample,
        colsample_bytree =
          g$colsample_bytree,
        lambda = g$lambda,
        nthread = cpu_threads,
        seed = 20260907L
      ),
      compute$extra
    )

    cat(
      "\n",
      spec$label,
      " | ",
      g$config_id,
      " (",
      i,
      "/",
      nrow(search_grid),
      ")\n",
      sep = ""
    )

    set.seed(20260907L)
    fit <- xgb.train(
      params = params,
      data = dtr,
      nrounds = MAX_ROUNDS,
      evals = list(
        train = dtr,
        tune_2023 = dva
      ),
      early_stopping_rounds =
        EARLY_STOP,
      maximize = FALSE,
      verbose = 0
    )

    if (
      i == 1L
    ) {
      diag_best_score <- tryCatch(
        suppressWarnings(
          as.numeric(
            fit$best_score
          )
        ),
        error = function(e) NA_real_
      )

      if (
        length(diag_best_score) == 0L
      ) {
        diag_best_score <- NA_real_
      }

      diag_attr_best_score <- tryCatch(
        suppressWarnings(
          as.numeric(
            xgb.attr(
              fit,
              "best_score"
            )
          )
        ),
        error = function(e) NA_real_
      )

      if (
        length(diag_attr_best_score) == 0L
      ) {
        diag_attr_best_score <- NA_real_
      }

      fwrite(
        data.table(
          endpoint = spec$endpoint,
          model_best_iteration =
            tryCatch(
              as.integer(
                fit$best_iteration
              ),
              error = function(e) NA_integer_
            ),
          attr_best_iteration =
            tryCatch(
              as.integer(
                xgb.attr(
                  fit,
                  "best_iteration"
                )
              ),
              error = function(e) NA_integer_
            ),
          model_best_score =
            diag_best_score[1L],
          attr_best_score =
            diag_attr_best_score[1L],
          evaluation_log_present =
            tryCatch(
              !is.null(
                fit$evaluation_log
              ) &&
                nrow(
                  as.data.table(
                    fit$evaluation_log
                  )
                ) > 0L,
              error = function(e) FALSE
            )
        ),
        file.path(
          tune_dir,
          paste0(
            "DEBUG_early_stopping_metadata_",
            spec$endpoint,
            ".csv"
          )
        )
      )
    }

    out[[i]] <- cbind(
      data.table(
        endpoint = spec$endpoint,
        label = spec$label,
        endpoint_type = "binary",
        tune_metric = "logloss",
        best_iteration =
          extract_best_iteration(
            fit,
            MAX_ROUNDS
          ),
        tune_2023_score =
          extract_tune_score(
            fit,
            "logloss"
          )
      ),
      g
    )

    rm(fit)
    gc()
  }

  rbindlist(out, fill = TRUE)
}

tune_multiclass <- function(spec) {
  yf <- factor(
    as.character(
      dev[[spec$endpoint]]
    ),
    levels = spec$levels
  )
  y_all <- as.integer(yf) - 1L

  tr <- idx_train[
    !is.na(y_all[idx_train])
  ]
  va <- idx_tune[
    !is.na(y_all[idx_tune])
  ]

  dtr <- xgb.DMatrix(
    X[tr, , drop = FALSE],
    label = y_all[tr],
    missing = NA_real_
  )
  dva <- xgb.DMatrix(
    X[va, , drop = FALSE],
    label = y_all[va],
    missing = NA_real_
  )

  out <- vector(
    "list",
    nrow(search_grid)
  )

  for (i in seq_len(nrow(search_grid))) {
    g <- search_grid[i]

    params <- c(
      list(
        objective = "multi:softprob",
        eval_metric = "mlogloss",
        num_class = length(spec$levels),
        eta = ETA_FIXED,
        max_depth = g$max_depth,
        min_child_weight =
          g$min_child_weight,
        subsample = g$subsample,
        colsample_bytree =
          g$colsample_bytree,
        lambda = g$lambda,
        nthread = cpu_threads,
        seed = 20260907L
      ),
      compute$extra
    )

    cat(
      "\n",
      spec$label,
      " | ",
      g$config_id,
      " (",
      i,
      "/",
      nrow(search_grid),
      ")\n",
      sep = ""
    )

    set.seed(20260907L)
    fit <- xgb.train(
      params = params,
      data = dtr,
      nrounds = MAX_ROUNDS,
      evals = list(
        train = dtr,
        tune_2023 = dva
      ),
      early_stopping_rounds =
        EARLY_STOP,
      maximize = FALSE,
      verbose = 0
    )

    if (
      i == 1L
    ) {
      diag_best_score <- tryCatch(
        suppressWarnings(
          as.numeric(
            fit$best_score
          )
        ),
        error = function(e) NA_real_
      )

      if (
        length(diag_best_score) == 0L
      ) {
        diag_best_score <- NA_real_
      }

      diag_attr_best_score <- tryCatch(
        suppressWarnings(
          as.numeric(
            xgb.attr(
              fit,
              "best_score"
            )
          )
        ),
        error = function(e) NA_real_
      )

      if (
        length(diag_attr_best_score) == 0L
      ) {
        diag_attr_best_score <- NA_real_
      }

      fwrite(
        data.table(
          endpoint = spec$endpoint,
          model_best_iteration =
            tryCatch(
              as.integer(
                fit$best_iteration
              ),
              error = function(e) NA_integer_
            ),
          attr_best_iteration =
            tryCatch(
              as.integer(
                xgb.attr(
                  fit,
                  "best_iteration"
                )
              ),
              error = function(e) NA_integer_
            ),
          model_best_score =
            diag_best_score[1L],
          attr_best_score =
            diag_attr_best_score[1L],
          evaluation_log_present =
            tryCatch(
              !is.null(
                fit$evaluation_log
              ) &&
                nrow(
                  as.data.table(
                    fit$evaluation_log
                  )
                ) > 0L,
              error = function(e) FALSE
            )
        ),
        file.path(
          tune_dir,
          paste0(
            "DEBUG_early_stopping_metadata_",
            spec$endpoint,
            ".csv"
          )
        )
      )
    }

    out[[i]] <- cbind(
      data.table(
        endpoint = spec$endpoint,
        label = spec$label,
        endpoint_type = "multiclass",
        tune_metric = "mlogloss",
        best_iteration =
          extract_best_iteration(
            fit,
            MAX_ROUNDS
          ),
        tune_2023_score =
          extract_tune_score(
            fit,
            "mlogloss"
          )
      ),
      g
    )

    rm(fit)
    gc()
  }

  rbindlist(out, fill = TRUE)
}

# -----------------------------------------------------------------------------
# Run search
# -----------------------------------------------------------------------------

all_results <- list()

for (spec in endpoint_specs) {
  if (spec$type == "binary") {
    res <- tune_binary(spec)
  } else {
    res <- tune_multiclass(spec)
  }

  fwrite(
    res[
      order(tune_2023_score)
    ],
    file.path(
      tune_dir,
      paste0(
        "tuning_",
        spec$endpoint,
        ".csv"
      )
    )
  )

  all_results[[
    length(all_results) + 1L
  ]] <- res
}

all_results <- rbindlist(
  all_results,
  fill = TRUE
)

fwrite(
  all_results[
    order(
      endpoint,
      tune_2023_score
    )
  ],
  file.path(
    tune_dir,
    "02_ALL_TEMPORAL_TUNING_RESULTS.csv"
  )
)

selected <- all_results[
  ,
  .SD[
    which.min(
      tune_2023_score
    )
  ],
  by = endpoint
]

selected[
  ,
  selected_by :=
    "Minimum 2023 temporal validation log loss"
]

fwrite(
  selected,
  file.path(
    tune_dir,
    "03_SELECTED_HYPERPARAMETERS.csv"
  )
)

# Compare selected configuration with the old heuristic default.
default_perf <- all_results[
  config_id == "cfg_01"
]

comparison <- merge(
  selected[
    ,
    .(
      endpoint,
      selected_config =
        config_id,
      selected_score =
        tune_2023_score,
      selected_rounds =
        best_iteration
    )
  ],
  default_perf[
    ,
    .(
      endpoint,
      heuristic_default_score =
        tune_2023_score,
      heuristic_default_rounds =
        best_iteration
    )
  ],
  by = "endpoint",
  all.x = TRUE
)

comparison[
  ,
  improvement_in_logloss :=
    heuristic_default_score -
      selected_score
]

fwrite(
  comparison,
  file.path(
    tune_dir,
    "04_SELECTED_VS_HEURISTIC_DEFAULT.csv"
  )
)

writeLines(
  c(
    "TBI-TRACT TEMPORAL HYPERPARAMETER TUNING COMPLETE",
    "",
    "2024 was not loaded/evaluated.",
    "Selected configurations are in 03_SELECTED_HYPERPARAMETERS.csv.",
    "Selection used minimum 2023 log loss/mlogloss only."
  ),
  file.path(
    tune_dir,
    "TUNING_SUMMARY.txt"
  )
)

cat("\n============================================================\n")
cat("SELECTED HYPERPARAMETERS\n")
cat("============================================================\n")
print(selected)
cat("\nOutput:\n  ", tune_dir, "\n", sep = "")
