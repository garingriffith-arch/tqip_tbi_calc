# =============================================================================
# 13_run_FINAL_TBI_TRACT_ventilation_trajectory_v2.R
#
# TBI-TRACT FINAL VENTILATION TRAJECTORY MODEL
#
# Outcome:
#   1) No invasive mechanical ventilation
#   2) 1-7 ventilator days
#   3) >=8 ventilator days
#
# Derived clinically useful probabilities from the SAME multiclass model:
#   P(any ventilation)        = 1 - P(no ventilation)
#   P(extended ventilation)   = P(>=8 ventilator days)
#
# Temporal architecture:
#   2020-2022 -> hyperparameter + boosting training
#   2023      -> hyperparameter + boosting-round selection
#   2020-2023 -> final refit using selected hyperparameters/rounds
#   2024      -> temporal evaluation only
#
# Compute target:
#   - Actively probes NVIDIA CUDA first.
#   - If CUDA is unavailable to the installed R/xgboost build, falls back to
#     ~75% of detected logical CPU threads.
#   - Data encoding is kept deliberately memory-conscious for a 64-GB system.
#
# IMPORTANT:
#   This script DOES NOT alter any existing TBI-TRACT models or outputs.
#   It writes to a new output directory.
#
# Prerequisite:
#   source("R/08_TBI_TRACT_methods_completion_audit.R")
# or otherwise ensure that:
#   outputs/METHODS_COMPLETION_TBI_TRACT/
#     frozen_methods_dataset_retained_2020_2024.rds
#     11_FROZEN_PREDICTOR_TYPES.csv
#
# =============================================================================

rm(list = ls())
gc()

# -----------------------------------------------------------------------------
# Packages
# -----------------------------------------------------------------------------

required <- c("data.table", "xgboost")
missing_pkgs <- required[
  !vapply(required, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_pkgs) > 0L) {
  stop(
    "Missing required package(s): ",
    paste(missing_pkgs, collapse = ", "),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(data.table)
  library(xgboost)
})

# -----------------------------------------------------------------------------
# Project config
# -----------------------------------------------------------------------------

config_candidates <- c(
  file.path(getwd(), "R", "00_config.R"),
  "R/00_config.R"
)
config_file <- config_candidates[file.exists(config_candidates)][1]

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

if (
  !file.exists(dataset_file) ||
    !file.exists(types_file)
) {
  stop(
    "Required frozen methods dataset/predictor dictionary not found. ",
    "Run 08_TBI_TRACT_methods_completion_audit.R first.",
    call. = FALSE
  )
}

out_dir <- file.path(
  output_dir,
  "FINAL_TBI_TRACT_VENTILATION_TRAJECTORY_2024"
)
model_dir <- file.path(out_dir, "models")
pred_dir <- file.path(out_dir, "predictions")
cal_dir <- file.path(out_dir, "calibration")

for (d in c(out_dir, model_dir, pred_dir, cal_dir)) {
  dir.create(
    d,
    recursive = TRUE,
    showWarnings = FALSE
  )
}

# -----------------------------------------------------------------------------
# Compute policy: ~75% CPU capacity, CUDA if genuinely available
# -----------------------------------------------------------------------------

logical_cores <- parallel::detectCores(logical = TRUE)
if (
  is.na(logical_cores) ||
    logical_cores < 1L
) {
  logical_cores <- 12L
}

cpu_threads <- 30L

# Leave at least one logical thread for Windows / RStudio responsiveness.
if (
  cpu_threads >= logical_cores &&
    logical_cores > 1L
) {
  cpu_threads <- logical_cores - 1L
}

setDTthreads(30L)

detect_xgb_compute <- function(cpu_threads) {
  set.seed(20260908L)

  x <- matrix(
    rnorm(1600),
    nrow = 200,
    ncol = 8
  )
  y <- rbinom(200, 1, 0.4)

  d <- xgb.DMatrix(
    x,
    label = y,
    missing = NA_real_
  )

  booster_uses_cuda <- function(fit) {
    cfg <- tryCatch(
      xgb.config(fit),
      error = function(e) NA_character_
    )

    if (
      length(cfg) != 1L ||
        is.na(cfg)
    ) {
      return(FALSE)
    }

    cfg_lower <- tolower(cfg)

    if (
      grepl(
        '"device"[[:space:]]*:[[:space:]]*"cpu',
        cfg_lower
      )
    ) {
      return(FALSE)
    }

    if (
      grepl(
        '"device"[[:space:]]*:[[:space:]]*"cuda',
        cfg_lower
      )
    ) {
      return(TRUE)
    }

    if (
      grepl(
        '"gpu_id"[[:space:]]*:[[:space:]]*[0-9]+',
        cfg_lower
      ) &&
        !grepl(
          '"gpu_id"[[:space:]]*:[[:space:]]*-1',
          cfg_lower
        )
    ) {
      return(TRUE)
    }

    FALSE
  }

  # Modern xgboost syntax.
  modern_params <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    tree_method = "hist",
    device = "cuda",
    max_depth = 2L,
    eta = 0.2,
    nthread = cpu_threads,
    seed = 20260908L
  )

  fit_modern <- tryCatch(
    suppressWarnings(
      xgb.train(
        params = modern_params,
        data = d,
        nrounds = 2L,
        verbose = 0
      )
    ),
    error = function(e) NULL
  )

  if (
    !is.null(fit_modern) &&
      booster_uses_cuda(fit_modern)
  ) {
    return(
      list(
        mode = "CUDA",
        extra = list(
          tree_method = "hist",
          device = "cuda"
        )
      )
    )
  }

  # Legacy GPU syntax for older CUDA-enabled builds.
  legacy_params <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    tree_method = "gpu_hist",
    predictor = "gpu_predictor",
    max_depth = 2L,
    eta = 0.2,
    nthread = cpu_threads,
    seed = 20260908L
  )

  fit_legacy <- tryCatch(
    suppressWarnings(
      xgb.train(
        params = legacy_params,
        data = d,
        nrounds = 2L,
        verbose = 0
      )
    ),
    error = function(e) NULL
  )

  if (
    !is.null(fit_legacy) &&
      booster_uses_cuda(fit_legacy)
  ) {
    return(
      list(
        mode = "CUDA legacy",
        extra = list(
          tree_method = "gpu_hist",
          predictor = "gpu_predictor"
        )
      )
    )
  }

  list(
    mode = "CPU",
    extra = list(
      tree_method = "hist"
    )
  )
}

compute <- detect_xgb_compute(
  cpu_threads
)

cat(
  "\n============================================================\n",
  "TBI-TRACT VENTILATION TRAJECTORY\n",
  "============================================================\n",
  "Logical CPU threads detected: ", logical_cores, "\n",
  "Threads allocated (~75%):     ", cpu_threads, "\n",
  "XGBoost compute mode:         ", compute$mode, "\n",
  "XGBoost version:              ",
  as.character(packageVersion("xgboost")),
  "\n",
  sep = ""
)

if (
  compute$mode == "CPU"
) {
  cat(
    "\nNOTE: CUDA was not verified by this installed R/xgboost build.\n",
    "The run will use ~75% of detected logical CPU threads instead.\n",
    sep = ""
  )
}

fwrite(
  data.table(
    logical_cpu_threads = logical_cores,
    allocated_threads = cpu_threads,
    approximate_cpu_fraction =
      cpu_threads / logical_cores,
    compute_mode = compute$mode,
    xgboost_version =
      as.character(
        packageVersion("xgboost")
      )
  ),
  file.path(
    out_dir,
    "00_COMPUTE_ENVIRONMENT.csv"
  )
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

clamp_prob <- function(
    p,
    eps = 1e-7
) {
  pmin(
    pmax(
      as.numeric(p),
      eps
    ),
    1 - eps
  )
}

fast_auc <- function(y, p) {
  keep <- !is.na(y) &
    !is.na(p) &
    is.finite(p)

  y <- as.integer(y[keep])
  p <- as.numeric(p[keep])

  keep2 <- y %in% c(0L, 1L)
  y <- y[keep2]
  p <- p[keep2]

  n1 <- as.double(
    sum(y == 1L)
  )
  n0 <- as.double(
    sum(y == 0L)
  )

  if (
    n1 == 0 ||
      n0 == 0
  ) {
    return(NA_real_)
  }

  r <- rank(
    p,
    ties.method = "average"
  )

  (
    sum(r[y == 1L]) -
      n1 * (n1 + 1) / 2
  ) / (
    n1 * n0
  )
}

fast_auprc <- function(y, p) {
  keep <- !is.na(y) &
    !is.na(p) &
    is.finite(p)

  y <- as.integer(y[keep])
  p <- as.numeric(p[keep])

  keep2 <- y %in% c(0L, 1L)
  y <- y[keep2]
  p <- p[keep2]

  n_pos <- sum(y == 1L)

  if (
    n_pos == 0L
  ) {
    return(NA_real_)
  }

  ord <- order(
    p,
    decreasing = TRUE
  )

  ys <- y[ord]

  tp <- cumsum(
    ys == 1L
  )
  fp <- cumsum(
    ys == 0L
  )

  precision <- tp / (
    tp + fp
  )
  recall <- tp / n_pos
  recall_prev <- c(
    0,
    head(
      recall,
      -1L
    )
  )

  sum(
    precision *
      (
        recall -
          recall_prev
      ),
    na.rm = TRUE
  )
}

binary_calibration <- function(y, p) {
  keep <- !is.na(y) &
    !is.na(p) &
    is.finite(p)

  y <- as.integer(y[keep])
  p <- clamp_prob(
    p[keep]
  )

  if (
    length(unique(y)) < 2L
  ) {
    return(
      c(
        intercept = NA_real_,
        slope = NA_real_
      )
    )
  }

  lp <- qlogis(p)

  fit_intercept <- tryCatch(
    glm(
      y ~ 1,
      family = binomial(),
      offset = lp
    ),
    error = function(e) NULL
  )

  fit_slope <- tryCatch(
    glm(
      y ~ lp,
      family = binomial()
    ),
    error = function(e) NULL
  )

  c(
    intercept =
      if (
        is.null(fit_intercept)
      ) {
        NA_real_
      } else {
        unname(
          coef(fit_intercept)[1L]
        )
      },
    slope =
      if (
        is.null(fit_slope)
      ) {
        NA_real_
      } else {
        unname(
          coef(fit_slope)["lp"]
        )
      }
  )
}

one_vs_rest_metrics <- function(
    y_class,
    prob,
    class_levels,
    model_label
) {
  rbindlist(
    lapply(
      seq_along(class_levels),
      function(k) {
        lev <- class_levels[k]

        y <- as.integer(
          y_class == lev
        )
        p <- prob[, k]

        cal <- binary_calibration(
          y,
          p
        )

        data.table(
          model = model_label,
          metric_target = lev,
          N = length(y),
          events =
            sum(
              y == 1L
            ),
          event_rate =
            mean(
              y == 1L
            ),
          AUROC =
            fast_auc(
              y,
              p
            ),
          AUPRC =
            fast_auprc(
              y,
              p
            ),
          Brier =
            mean(
              (
                p - y
              )^2
            ),
          calibration_intercept =
            unname(
              cal["intercept"]
            ),
          calibration_slope =
            unname(
              cal["slope"]
            )
        )
      }
    )
  )
}

multiclass_logloss <- function(
    y_index_1based,
    prob
) {
  prob <- pmax(
    pmin(
      prob,
      1 - 1e-15
    ),
    1e-15
  )

  -mean(
    log(
      prob[
        cbind(
          seq_along(
            y_index_1based
          ),
          y_index_1based
        )
      ]
    )
  )
}

multiclass_brier <- function(
    y_index_1based,
    prob
) {
  K <- ncol(prob)

  truth <- matrix(
    0,
    nrow = nrow(prob),
    ncol = K
  )

  truth[
    cbind(
      seq_len(nrow(prob)),
      y_index_1based
    )
  ] <- 1

  # Mean squared probability error summed across classes.
  mean(
    rowSums(
      (
        prob - truth
      )^2
    )
  )
}

make_calibration_deciles <- function(
    y,
    p,
    target
) {
  d <- data.table(
    y = as.integer(y),
    p = as.numeric(p)
  )

  d <- d[
    !is.na(y) &
      !is.na(p)
  ]

  if (
    nrow(d) == 0L
  ) {
    return(data.table())
  }

  # rank-based deciles are robust to tied probabilities.
  d[
    ,
    decile := pmin(
      10L,
      ceiling(
        10 *
          frank(
            p,
            ties.method = "average"
          ) /
          .N
      )
    )
  ]

  d[
    ,
    .(
      target = target,
      N = .N,
      mean_predicted =
        mean(p),
      observed_rate =
        mean(y),
      absolute_difference =
        abs(
          mean(p) -
            mean(y)
        )
    ),
    by = decile
  ][
    order(decile)
  ]
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
          sort(
            unique(x)
          ),
          "__UNKNOWN__",
          "__OTHER__"
        )
      )
    }
  )

  names(cat_levels) <-
    categorical_vars

  list(
    numeric_vars = numeric_vars,
    categorical_vars =
      categorical_vars,
    categorical_levels =
      cat_levels
  )
}

encode_dense <- function(
    d,
    encoder
) {
  n <- nrow(d)

  # Pre-compute final column count to avoid repeated large reallocations.
  n_numeric <-
    length(
      encoder$numeric_vars
    )
  n_cat <-
    sum(
      vapply(
        encoder$categorical_levels,
        length,
        integer(1)
      )
    )

  X <- matrix(
    0,
    nrow = n,
    ncol =
      n_numeric +
      n_cat
  )

  feature_names <- character(
    ncol(X)
  )

  j <- 1L

  for (
    v in encoder$numeric_vars
  ) {
    X[, j] <-
      safe_num(
        d[[v]]
      )

    feature_names[j] <- v
    j <- j + 1L
  }

  for (
    v in encoder$categorical_vars
  ) {
    lev <-
      encoder$categorical_levels[[v]]

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
      j +
        length(lev) -
        1L
    )

    # One-hot encode without constructing intermediate model.matrix objects.
    X[
      cbind(
        seq_len(n),
        cols[idx]
      )
    ] <- 1

    feature_names[cols] <-
      paste0(
        v,
        "__",
        make.names(
          lev,
          unique = TRUE
        )
      )

    j <- max(cols) + 1L
  }

  colnames(X) <-
    feature_names
  storage.mode(X) <-
    "double"

  X
}

extract_best_iteration <- function(
    model,
    fallback
) {
  x <- tryCatch(
    suppressWarnings(
      as.numeric(
        model$best_iteration
      )
    ),
    error = function(e) NULL
  )

  if (
    !is.null(x) &&
      length(x) >= 1L &&
      is.finite(x[1L])
  ) {
    return(
      max(
        1L,
        as.integer(
          x[1L]
        )
      )
    )
  }

  x <- tryCatch(
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
    !is.null(x) &&
      length(x) >= 1L &&
      is.finite(x[1L])
  ) {
    # Booster attribute is commonly zero-based.
    return(
      max(
        1L,
        as.integer(
          x[1L]
        ) +
          1L
      )
    )
  }

  as.integer(fallback)
}

extract_best_score <- function(
    model,
    metric_name = "mlogloss"
) {
  candidates <- list(
    tryCatch(
      model$best_score,
      error = function(e) NULL
    ),
    tryCatch(
      xgb.attr(
        model,
        "best_score"
      ),
      error = function(e) NULL
    )
  )

  for (z in candidates) {
    if (
      !is.null(z) &&
        length(z) >= 1L
    ) {
      val <- suppressWarnings(
        as.numeric(
          z[1L]
        )
      )

      if (
        length(val) == 1L &&
          is.finite(val)
      ) {
        return(val)
      }
    }
  }

  # Optional fallback for builds that retain evaluation_log.
  elog <- tryCatch(
    as.data.table(
      model$evaluation_log
    ),
    error = function(e) data.table()
  )

  if (
    nrow(elog) > 0L &&
      ncol(elog) > 0L
  ) {
    nms <- names(elog)
    nms_norm <- tolower(
      gsub(
        "[^a-z0-9]+",
        "",
        nms
      )
    )
    metric_norm <- tolower(
      gsub(
        "[^a-z0-9]+",
        "",
        metric_name
      )
    )

    idx <- which(
      grepl(
        metric_norm,
        nms_norm,
        fixed = TRUE
      ) &
        grepl(
          "2023|tune|valid|validation|eval|test",
          nms_norm
        ) &
        !grepl(
          "^train|training",
          nms_norm
        )
    )

    if (
      length(idx) >= 1L
    ) {
      vals <- suppressWarnings(
        as.numeric(
          elog[[nms[idx[1L]]]]
        )
      )

      if (
        any(
          is.finite(vals)
        )
      ) {
        return(
          min(
            vals,
            na.rm = TRUE
          )
        )
      }
    }
  }

  stop(
    "Could not recover early-stopping best_score from this xgboost build.",
    call. = FALSE
  )
}

# -----------------------------------------------------------------------------
# Load frozen cohort and define the NEW ventilation trajectory
# -----------------------------------------------------------------------------

cat("\nReading frozen methods dataset...\n")

dt <- as.data.table(
  readRDS(
    dataset_file
  )
)

types <- fread(
  types_file
)

numeric_predictors <- types[
  category == "numeric",
  predictor
]

categorical_predictors <- types[
  category == "categorical",
  predictor
]

required_cols <- unique(
  c(
    "admission_year",
    "inc_key",
    "vent_days",
    numeric_predictors,
    categorical_predictors
  )
)

missing_cols <- setdiff(
  required_cols,
  names(dt)
)

if (
  length(missing_cols) > 0L
) {
  stop(
    "Frozen dataset is missing required column(s): ",
    paste(
      missing_cols,
      collapse = ", "
    ),
    call. = FALSE
  )
}

dt[
  ,
  vent_days_model :=
    safe_num(
      vent_days
    )
]

dt[
  ,
  ventilation_trajectory_final :=
    fcase(
      is.na(
        vent_days_model
      ),
      NA_character_,
      vent_days_model <= 0,
      "No ventilation",
      vent_days_model >= 1 &
        vent_days_model <= 7,
      "Ventilation 1-7 days",
      vent_days_model >= 8,
      "Ventilation >=8 days",
      default =
        NA_character_
    )
]

VENT_LEVELS <- c(
  "No ventilation",
  "Ventilation 1-7 days",
  "Ventilation >=8 days"
)

dt[
  ,
  ventilation_trajectory_final :=
    factor(
      ventilation_trajectory_final,
      levels = VENT_LEVELS
    )
]

# Cohort reconciliation.
cohort_counts <- dt[
  ,
  .(
    N = .N,
    known_ventilation_outcome =
      sum(
        !is.na(
          ventilation_trajectory_final
        )
      ),
    no_ventilation =
      sum(
        ventilation_trajectory_final ==
          "No ventilation",
        na.rm = TRUE
      ),
    vent_1_7 =
      sum(
        ventilation_trajectory_final ==
          "Ventilation 1-7 days",
        na.rm = TRUE
      ),
    vent_ge8 =
      sum(
        ventilation_trajectory_final ==
          "Ventilation >=8 days",
        na.rm = TRUE
      )
  ),
  by = admission_year
][
  order(admission_year)
]

fwrite(
  cohort_counts,
  file.path(
    out_dir,
    "01_VENTILATION_TRAJECTORY_COUNTS_BY_YEAR.csv"
  )
)

print(cohort_counts)

expected_N <- data.table(
  admission_year = 2020:2024,
  expected_N = c(
    145338L,
    153018L,
    152944L,
    152706L,
    151874L
  )
)

recon <- merge(
  expected_N,
  cohort_counts[
    ,
    .(
      admission_year,
      observed_N = N
    )
  ],
  by = "admission_year",
  all = TRUE
)

recon[
  ,
  matches :=
    expected_N ==
      observed_N
]

fwrite(
  recon,
  file.path(
    out_dir,
    "02_PRIMARY_COHORT_RECONCILIATION.csv"
  )
)

if (
  anyNA(
    recon$matches
  ) ||
    !all(
      recon$matches
    )
) {
  stop(
    "Frozen cohort does not reproduce the locked v4 cohort counts.",
    call. = FALSE
  )
}

# -----------------------------------------------------------------------------
# Strict temporal partitions
# -----------------------------------------------------------------------------

idx_train <- which(
  dt$admission_year %in%
    2020:2022 &
    !is.na(
      dt$ventilation_trajectory_final
    )
)

idx_tune <- which(
  dt$admission_year ==
    2023L &
    !is.na(
      dt$ventilation_trajectory_final
    )
)

idx_dev <- which(
  dt$admission_year %in%
    2020:2023 &
    !is.na(
      dt$ventilation_trajectory_final
    )
)

idx_test <- which(
  dt$admission_year ==
    2024L &
    !is.na(
      dt$ventilation_trajectory_final
    )
)

split_counts <- data.table(
  split = c(
    "Training 2020-2022",
    "Tuning 2023",
    "Final development 2020-2023",
    "Temporal evaluation 2024"
  ),
  N = c(
    length(idx_train),
    length(idx_tune),
    length(idx_dev),
    length(idx_test)
  )
)

fwrite(
  split_counts,
  file.path(
    out_dir,
    "03_TEMPORAL_SPLIT_COUNTS.csv"
  )
)

# -----------------------------------------------------------------------------
# Encoder:
#   - Tuning encoder sees 2020-22 only.
#   - Final encoder sees 2020-23 only.
#   - 2024 never defines predictor levels.
# -----------------------------------------------------------------------------

cat("\nEncoding 2020-2023 tuning data...\n")

pp_tune <- fit_encoder(
  dt[idx_train],
  numeric_predictors,
  categorical_predictors
)

tune_rows <- c(
  idx_train,
  idx_tune
)

X_tune_all <- encode_dense(
  dt[tune_rows],
  pp_tune
)

n_train <- length(
  idx_train
)

X_train <- X_tune_all[
  seq_len(n_train),
  ,
  drop = FALSE
]

X_tune23 <- X_tune_all[
  n_train +
    seq_len(
      length(idx_tune)
    ),
  ,
  drop = FALSE
]

rm(X_tune_all)
gc()

y_train_factor <-
  dt$ventilation_trajectory_final[
    idx_train
  ]

y_tune_factor <-
  dt$ventilation_trajectory_final[
    idx_tune
  ]

y_train <- as.integer(
  y_train_factor
) - 1L

y_tune <- as.integer(
  y_tune_factor
) - 1L

dtrain <- xgb.DMatrix(
  X_train,
  label = y_train,
  missing = NA_real_
)

dtune <- xgb.DMatrix(
  X_tune23,
  label = y_tune,
  missing = NA_real_
)

# Free the raw matrices after DMatrix construction to reduce RAM use.
rm(
  X_train,
  X_tune23
)
gc()

# -----------------------------------------------------------------------------
# Prespecified search grid
# -----------------------------------------------------------------------------

grid_file <- file.path(
  output_dir,
  "TBI_TRACT_TEMPORAL_HYPERPARAMETER_TUNING",
  "01_PRESPECIFIED_SEARCH_GRID.csv"
)

if (
  file.exists(grid_file)
) {
  search_grid <- fread(
    grid_file
  )

  needed_grid_cols <- c(
    "config_id",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "lambda"
  )

  if (
    !all(
      needed_grid_cols %in%
        names(search_grid)
    )
  ) {
    stop(
      "Existing tuning search grid is malformed.",
      call. = FALSE
    )
  }

  search_grid <-
    search_grid[
      ,
      ..needed_grid_cols
    ]

  cat(
    "\nReusing the exact prespecified 24-configuration search grid from script 09.\n"
  )
} else {
  # Deterministically reconstruct the same search strategy.
  full_grid <- CJ(
    max_depth =
      c(
        2L,
        3L,
        4L,
        5L
      ),
    min_child_weight =
      c(
        5,
        10,
        20
      ),
    subsample =
      c(
        0.75,
        0.90,
        1.00
      ),
    colsample_bytree =
      c(
        0.75,
        0.90,
        1.00
      ),
    lambda =
      c(
        1,
        5
      ),
    unique = TRUE
  )

  default_row <- data.table(
    max_depth = 3L,
    min_child_weight = 10,
    subsample = 0.85,
    colsample_bytree = 0.85,
    lambda = 1
  )

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
    config_id :=
      sprintf(
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
}

fwrite(
  search_grid,
  file.path(
    out_dir,
    "04_VENT_TRAJECTORY_SEARCH_GRID.csv"
  )
)

ETA_FIXED <- 0.05
MAX_ROUNDS <- 3000L
EARLY_STOP <- 75L

# -----------------------------------------------------------------------------
# Tune the NEW multiclass outcome using 2023 mlogloss only
# -----------------------------------------------------------------------------

cat(
  "\n============================================================\n",
  "TEMPORAL HYPERPARAMETER TUNING\n",
  "2020-22 train -> 2023 selection\n",
  "============================================================\n",
  sep = ""
)

tuning_results <- vector(
  "list",
  nrow(search_grid)
)

for (
  i in seq_len(
    nrow(search_grid)
  )
) {
  g <- search_grid[i]

  params <- c(
    list(
      objective =
        "multi:softprob",
      eval_metric =
        "mlogloss",
      num_class =
        length(VENT_LEVELS),
      eta =
        ETA_FIXED,
      max_depth =
        as.integer(
          g$max_depth
        ),
      min_child_weight =
        g$min_child_weight,
      subsample =
        g$subsample,
      colsample_bytree =
        g$colsample_bytree,
      lambda =
        g$lambda,
      nthread =
        cpu_threads,
      seed =
        20260908L
    ),
    compute$extra
  )

  cat(
    "\nVentilation trajectory | ",
    g$config_id,
    " (",
    i,
    "/",
    nrow(search_grid),
    ")\n",
    sep = ""
  )

  set.seed(
    20260908L
  )

  fit <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = MAX_ROUNDS,
    evals = list(
      train_2020_22 =
        dtrain,
      tune_2023 =
        dtune
    ),
    early_stopping_rounds =
      EARLY_STOP,
    maximize = FALSE,
    verbose = 0
  )

  best_round <-
    extract_best_iteration(
      fit,
      MAX_ROUNDS
    )

  best_score <-
    extract_best_score(
      fit,
      "mlogloss"
    )

  tuning_results[[i]] <-
    cbind(
      data.table(
        endpoint =
          "ventilation_trajectory_final",
        best_iteration =
          best_round,
        tune_2023_mlogloss =
          best_score,
        hit_round_ceiling =
          best_round >=
          MAX_ROUNDS
      ),
      g
    )

  rm(fit)
  gc()
}

tuning_results <- rbindlist(
  tuning_results,
  fill = TRUE
)

setorder(
  tuning_results,
  tune_2023_mlogloss,
  best_iteration
)

fwrite(
  tuning_results,
  file.path(
    out_dir,
    "05_VENT_TRAJECTORY_TEMPORAL_TUNING_RESULTS.csv"
  )
)

selected <- copy(
  tuning_results[1L]
)

selected[
  ,
  selected_by :=
    "Minimum 2023 multiclass log loss"
]

fwrite(
  selected,
  file.path(
    out_dir,
    "06_SELECTED_VENT_TRAJECTORY_HYPERPARAMETERS.csv"
  )
)

cat(
  "\nSelected ventilation-trajectory configuration:\n"
)
print(selected)

# Tuning objects are no longer needed.
rm(
  dtrain,
  dtune,
  pp_tune
)
gc()

# -----------------------------------------------------------------------------
# Final 2020-23 refit; encoder remains frozen before 2024
# -----------------------------------------------------------------------------

cat(
  "\n============================================================\n",
  "FINAL REFIT: 2020-2023\n",
  "============================================================\n",
  sep = ""
)

pp_final <- fit_encoder(
  dt[idx_dev],
  numeric_predictors,
  categorical_predictors
)

saveRDS(
  pp_final,
  file.path(
    model_dir,
    "ventilation_trajectory_encoder_2020_2023.rds"
  )
)

final_rows <- c(
  idx_dev,
  idx_test
)

X_final_all <- encode_dense(
  dt[final_rows],
  pp_final
)

n_dev <- length(
  idx_dev
)

X_dev <- X_final_all[
  seq_len(n_dev),
  ,
  drop = FALSE
]

X_test <- X_final_all[
  n_dev +
    seq_len(
      length(idx_test)
    ),
  ,
  drop = FALSE
]

rm(
  X_final_all
)
gc()

y_dev_factor <-
  dt$ventilation_trajectory_final[
    idx_dev
  ]

y_test_factor <-
  dt$ventilation_trajectory_final[
    idx_test
  ]

y_dev <- as.integer(
  y_dev_factor
) - 1L

ddev <- xgb.DMatrix(
  X_dev,
  label = y_dev,
  missing = NA_real_
)

dtest <- xgb.DMatrix(
  X_test,
  missing = NA_real_
)

rm(
  X_dev,
  X_test
)
gc()

s <- selected[1L]

final_params <- c(
  list(
    objective =
      "multi:softprob",
    eval_metric =
      "mlogloss",
    num_class =
      length(VENT_LEVELS),
    eta =
      ETA_FIXED,
    max_depth =
      as.integer(
        s$max_depth
      ),
    min_child_weight =
      s$min_child_weight,
    subsample =
      s$subsample,
    colsample_bytree =
      s$colsample_bytree,
    lambda =
      s$lambda,
    nthread =
      cpu_threads,
    seed =
      20260908L
  ),
  compute$extra
)

selected_rounds <-
  as.integer(
    s$best_iteration
  )

set.seed(
  20260908L
)

final_model <- xgb.train(
  params =
    final_params,
  data =
    ddev,
  nrounds =
    selected_rounds,
  verbose =
    0
)

saveRDS(
  list(
    model =
      final_model,
    endpoint =
      "ventilation_trajectory_final",
    class_levels =
      VENT_LEVELS,
    selected_hyperparameters =
      selected,
    eta =
      ETA_FIXED,
    selected_rounds =
      selected_rounds,
    predictor_types =
      types,
    compute_environment =
      list(
        compute_mode =
          compute$mode,
        cpu_threads =
          cpu_threads,
        xgboost_version =
          as.character(
            packageVersion(
              "xgboost"
            )
          )
      )
  ),
  file.path(
    model_dir,
    "FINAL_ventilation_trajectory_model.rds"
  )
)

# -----------------------------------------------------------------------------
# 2024 temporal evaluation
# -----------------------------------------------------------------------------

raw_pred <- predict(
  final_model,
  dtest
)

K <- length(
  VENT_LEVELS
)

# xgboost R builds differ: some return an n x K matrix for multi:softprob,
# others return a flattened vector. Handle both without changing class order.
if (
  is.matrix(raw_pred) ||
    is.data.frame(raw_pred)
) {
  prob <- as.matrix(
    raw_pred
  )
} else {
  prob <- matrix(
    as.numeric(
      raw_pred
    ),
    ncol = K,
    byrow = TRUE
  )
}

if (
  ncol(prob) != K
) {
  stop(
    "Unexpected multiclass prediction dimensions.",
    call. = FALSE
  )
}

colnames(prob) <-
  VENT_LEVELS

if (
  nrow(prob) !=
    length(idx_test)
) {
  stop(
    "Prediction row count does not match 2024 test cohort.",
    call. = FALSE
  )
}

# Guard numerical normalization.
row_prob_sum <- rowSums(prob)

if (
  any(
    !is.finite(
      row_prob_sum
    )
  )
) {
  stop(
    "Non-finite probability rows detected.",
    call. = FALSE
  )
}

prob <- prob /
  row_prob_sum

y_test_index <-
  as.integer(
    y_test_factor
  )

predicted_class <- VENT_LEVELS[
  max.col(
    prob,
    ties.method = "first"
  )
]

overall_accuracy <-
  mean(
    predicted_class ==
      as.character(
        y_test_factor
      )
  )

overall_mlogloss <-
  multiclass_logloss(
    y_test_index,
    prob
  )

overall_brier <-
  multiclass_brier(
    y_test_index,
    prob
  )

overall_metrics <- data.table(
  endpoint =
    "Ventilation trajectory",
  N =
    length(
      y_test_factor
    ),
  accuracy =
    overall_accuracy,
  multiclass_log_loss =
    overall_mlogloss,
  multiclass_brier =
    overall_brier,
  selected_rounds =
    selected_rounds,
  compute_mode =
    compute$mode
)

fwrite(
  overall_metrics,
  file.path(
    out_dir,
    "07_2024_VENT_TRAJECTORY_OVERALL_METRICS.csv"
  )
)

class_metrics <-
  one_vs_rest_metrics(
    y_class =
      as.character(
        y_test_factor
      ),
    prob =
      prob,
    class_levels =
      VENT_LEVELS,
    model_label =
      "Ventilation trajectory"
  )

fwrite(
  class_metrics,
  file.path(
    out_dir,
    "08_2024_VENT_TRAJECTORY_CLASS_METRICS.csv"
  )
)

# -----------------------------------------------------------------------------
# Derived probabilities from the SAME trajectory model
# -----------------------------------------------------------------------------

p_none <-
  prob[
    ,
    "No ventilation"
  ]

p_1_7 <-
  prob[
    ,
    "Ventilation 1-7 days"
  ]

p_ge8 <-
  prob[
    ,
    "Ventilation >=8 days"
  ]

p_any <- 1 -
  p_none

y_any <- as.integer(
  as.character(
    y_test_factor
  ) !=
    "No ventilation"
)

y_ge8 <- as.integer(
  as.character(
    y_test_factor
  ) ==
    "Ventilation >=8 days"
)

derived_metrics <- rbindlist(
  list(
    {
      cal <- binary_calibration(
        y_any,
        p_any
      )

      data.table(
        derived_target =
          "Any invasive mechanical ventilation",
        N =
          length(y_any),
        events =
          sum(y_any),
        event_rate =
          mean(y_any),
        AUROC =
          fast_auc(
            y_any,
            p_any
          ),
        AUPRC =
          fast_auprc(
            y_any,
            p_any
          ),
        Brier =
          mean(
            (
              p_any -
                y_any
            )^2
          ),
        calibration_intercept =
          unname(
            cal["intercept"]
          ),
        calibration_slope =
          unname(
            cal["slope"]
          )
      )
    },
    {
      cal <- binary_calibration(
        y_ge8,
        p_ge8
      )

      data.table(
        derived_target =
          "Extended mechanical ventilation >=8 days",
        N =
          length(y_ge8),
        events =
          sum(y_ge8),
        event_rate =
          mean(y_ge8),
        AUROC =
          fast_auc(
            y_ge8,
            p_ge8
          ),
        AUPRC =
          fast_auprc(
            y_ge8,
            p_ge8
          ),
        Brier =
          mean(
            (
              p_ge8 -
                y_ge8
            )^2
          ),
        calibration_intercept =
          unname(
            cal["intercept"]
          ),
        calibration_slope =
          unname(
            cal["slope"]
          )
      )
    }
  )
)

fwrite(
  derived_metrics,
  file.path(
    out_dir,
    "09_2024_DERIVED_ANY_AND_EXTENDED_VENT_METRICS.csv"
  )
)

# -----------------------------------------------------------------------------
# Calibration deciles
# -----------------------------------------------------------------------------

calibration_deciles <- rbindlist(
  list(
    make_calibration_deciles(
      as.integer(
        as.character(
          y_test_factor
        ) ==
          "No ventilation"
      ),
      p_none,
      "No ventilation"
    ),
    make_calibration_deciles(
      as.integer(
        as.character(
          y_test_factor
        ) ==
          "Ventilation 1-7 days"
      ),
      p_1_7,
      "Ventilation 1-7 days"
    ),
    make_calibration_deciles(
      y_ge8,
      p_ge8,
      "Ventilation >=8 days"
    ),
    make_calibration_deciles(
      y_any,
      p_any,
      "Any invasive mechanical ventilation"
    )
  ),
  fill = TRUE
)

fwrite(
  calibration_deciles,
  file.path(
    cal_dir,
    "10_2024_VENTILATION_CALIBRATION_DECILES.csv"
  )
)

# -----------------------------------------------------------------------------
# Patient-level predictions
# -----------------------------------------------------------------------------

pred_2024 <- data.table(
  admission_year =
    dt$admission_year[
      idx_test
    ],
  inc_key =
    as.character(
      dt$inc_key[
        idx_test
      ]
    ),
  observed_class =
    as.character(
      y_test_factor
    ),
  predicted_class =
    predicted_class,
  p_no_ventilation =
    p_none,
  p_ventilation_1_7_days =
    p_1_7,
  p_ventilation_ge8_days =
    p_ge8,
  p_any_ventilation =
    p_any
)

fwrite(
  pred_2024,
  file.path(
    pred_dir,
    "ventilation_trajectory_2024_predictions.csv"
  )
)

# -----------------------------------------------------------------------------
# Feature importance
# -----------------------------------------------------------------------------

importance <- as.data.table(
  xgb.importance(
    model =
      final_model
  )
)

fwrite(
  importance,
  file.path(
    out_dir,
    "11_VENT_TRAJECTORY_XGBOOST_FEATURE_IMPORTANCE.csv"
  )
)

# -----------------------------------------------------------------------------
# Headline summary
# -----------------------------------------------------------------------------

headline <- rbindlist(
  list(
    class_metrics[
      ,
      .(
        target =
          metric_target,
        N,
        events,
        event_rate,
        AUROC,
        AUPRC,
        Brier,
        calibration_intercept,
        calibration_slope
      )
    ],
    derived_metrics[
      ,
      .(
        target =
          derived_target,
        N,
        events,
        event_rate,
        AUROC,
        AUPRC,
        Brier,
        calibration_intercept,
        calibration_slope
      )
    ]
  ),
  fill = TRUE
)

fwrite(
  headline,
  file.path(
    out_dir,
    "12_HEADLINE_2024_VENTILATION_RESULTS.csv"
  )
)

summary_lines <- c(
  "TBI-TRACT VENTILATION TRAJECTORY COMPLETE",
  "",
  "Endpoint:",
  "  No invasive mechanical ventilation",
  "  1-7 ventilator days",
  "  >=8 ventilator days",
  "",
  "Derived from the SAME multiclass model:",
  "  P(any ventilation) = 1 - P(no ventilation)",
  "  P(extended ventilation >=8 d) = P(>=8 d)",
  "",
  "Temporal architecture:",
  "  2020-22 training",
  "  2023 hyperparameter + round selection",
  "  2020-23 final refit",
  "  2024 temporal evaluation",
  "",
  paste0(
    "Compute mode: ",
    compute$mode
  ),
  paste0(
    "CPU threads allocated: ",
    cpu_threads,
    " / ",
    logical_cores
  ),
  paste0(
    "Selected rounds: ",
    selected_rounds
  ),
  "",
  "Review first:",
  "  06_SELECTED_VENT_TRAJECTORY_HYPERPARAMETERS.csv",
  "  07_2024_VENT_TRAJECTORY_OVERALL_METRICS.csv",
  "  08_2024_VENT_TRAJECTORY_CLASS_METRICS.csv",
  "  09_2024_DERIVED_ANY_AND_EXTENDED_VENT_METRICS.csv",
  "  12_HEADLINE_2024_VENTILATION_RESULTS.csv",
  "  calibration/10_2024_VENTILATION_CALIBRATION_DECILES.csv"
)

writeLines(
  summary_lines,
  file.path(
    out_dir,
    "VENTILATION_TRAJECTORY_SUMMARY.txt"
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

cat(
  "\n2024 headline metrics:\n"
)
print(headline)

# Explicit cleanup at end.
rm(
  ddev,
  dtest,
  raw_pred
)
gc()
