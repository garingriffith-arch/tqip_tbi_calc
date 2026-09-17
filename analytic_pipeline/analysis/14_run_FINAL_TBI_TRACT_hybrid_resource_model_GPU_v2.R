# =============================================================================
# 14_run_FINAL_TBI_TRACT_hybrid_resource_model_GPU_v2.R
#
# FINAL TBI-TRACT HYBRID RESOURCE-TRAJECTORY EXTENSION
#
# Adds the two pieces needed to complete the final bedside architecture:
#
# A) Hospital LOS trajectory:
#      <=7 days / 8-27 days / >=28 days
#
# B) Individualized duration estimates:
#      Q10 / Q50 / Q90  -> median + 80% prediction interval
#
#    Hospital LOS:
#      unconditional among patients with observed hospital LOS
#
#    ICU LOS:
#      conditional duration model among patients who used ICU resources;
#      deployed as "if ICU is required: median X days (80% PI Y-Z)"
#
#    Ventilator duration:
#      conditional duration model among patients who received invasive
#      mechanical ventilation;
#      deployed as "if ventilation is required: median X days (80% PI Y-Z)"
#
# Existing final components are NOT replaced:
#   - discharge 3-class
#   - ICU trajectory: no ICU / 1-7 / >=8 days
#   - ventilation trajectory: none / 1-7 / >=8 days
#   - ICP monitoring
#   - craniotomy/craniectomy
#
# TEMPORAL ARCHITECTURE
#   2020-2022 -> training / early stopping
#   2023      -> tuning / boosting-round selection
#   2020-2023 -> final refit
#   2024      -> temporal validation only
#
# IMPORTANT METHODOLOGIC DECISIONS
#   - 2024 is never used to choose cut points, transformations, structural
#     hyperparameters, or boosting rounds.
#   - HLOS classes are <=7 / 8-27 / >=28 days.
#     This retains the literature-supported first-week and >=28-day boundaries
#     while avoiding unnecessary fragmentation of the intermediate range.
#   - Duration models use only Q10/Q50/Q90, not the prior six-quantile
#     experiment.
#   - log1p duration modeling was selected previously using 2023 only.
#   - noncrossing display quantiles are produced by row-wise rearrangement.
#
# GPU BACKEND
#   Uses the already-created conda environment:
#     tbi-tract-xgb-gpu
#   with Python XGBoost CUDA.
#
# RUN AFTER:
#   - 08_TBI_TRACT_methods_completion_audit.R
#   - GPU backend verification
#   - final ventilation trajectory model
#
# =============================================================================

rm(list = ls())
gc()

# -----------------------------------------------------------------------------
# Packages / config
# -----------------------------------------------------------------------------

required <- c("data.table", "reticulate")
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
  library(reticulate)
})

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

set.seed(20260908L)

methods_dir <- file.path(
  output_dir,
  "METHODS_COMPLETION_TBI_TRACT"
)

tuning_dir <- file.path(
  output_dir,
  "TBI_TRACT_TEMPORAL_HYPERPARAMETER_TUNING"
)

vent_dir <- file.path(
  output_dir,
  "FINAL_TBI_TRACT_VENTILATION_TRAJECTORY_2024"
)

dataset_file <- file.path(
  methods_dir,
  "frozen_methods_dataset_retained_2020_2024.rds"
)

types_file <- file.path(
  methods_dir,
  "11_FROZEN_PREDICTOR_TYPES.csv"
)

search_grid_file <- file.path(
  tuning_dir,
  "01_PRESPECIFIED_SEARCH_GRID.csv"
)

selected_hp_file <- file.path(
  tuning_dir,
  "03_SELECTED_HYPERPARAMETERS.csv"
)

vent_selected_hp_file <- file.path(
  vent_dir,
  "06_SELECTED_VENT_TRAJECTORY_HYPERPARAMETERS.csv"
)

needed_files <- c(
  dataset_file,
  types_file,
  search_grid_file,
  selected_hp_file
)

if (!all(file.exists(needed_files))) {
  missing <- needed_files[
    !file.exists(needed_files)
  ]

  stop(
    "Required prerequisite file(s) missing:\n",
    paste0(
      "  ",
      missing,
      collapse = "\n"
    ),
    call. = FALSE
  )
}

out_dir <- file.path(
  output_dir,
  "FINAL_TBI_TRACT_HYBRID_RESOURCE_MODEL_2024"
)

model_dir <- file.path(
  out_dir,
  "models"
)

prediction_dir <- file.path(
  out_dir,
  "predictions"
)

validation_dir <- file.path(
  out_dir,
  "validation"
)

for (d in c(
  out_dir,
  model_dir,
  prediction_dir,
  validation_dir
)) {
  dir.create(
    d,
    recursive = TRUE,
    showWarnings = FALSE
  )
}

# -----------------------------------------------------------------------------
# Compute backend
# -----------------------------------------------------------------------------

ENV_NAME <- "tbi-tract-xgb-gpu"

logical_cores <- parallel::detectCores(
  logical = TRUE
)

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
    floor(
      0.94 * logical_cores
    )
  )
)

setDTthreads(
  cpu_threads
)

Sys.setenv(
  OMP_NUM_THREADS =
    as.character(cpu_threads),
  MKL_NUM_THREADS =
    as.character(cpu_threads),
  OPENBLAS_NUM_THREADS =
    as.character(cpu_threads)
)

reticulate::use_condaenv(
  ENV_NAME,
  required = TRUE
)

py_cfg <- reticulate::py_config()

py_xgb <- reticulate::import(
  "xgboost",
  convert = TRUE
)

py_np <- reticulate::import(
  "numpy",
  convert = TRUE
)

py_xgb_version <- tryCatch(
  as.character(
    reticulate::py_to_r(
      py_xgb$`__version__`
    )
  )[1L],
  error = function(e) {
    as.character(
      py_xgb$`__version__`
    )[1L]
  }
)

# Fail fast if CUDA cannot actually be used.
probe_x <- matrix(
  rnorm(4000),
  nrow = 500L,
  ncol = 8L
)

probe_y <- as.integer(
  rbinom(
    500L,
    1L,
    0.4
  )
)

probe_d <- py_xgb$DMatrix(
  probe_x,
  label = probe_y
)

probe_fit <- py_xgb$train(
  params = reticulate::dict(
    objective = "binary:logistic",
    tree_method = "hist",
    device = "cuda",
    max_depth = 2L,
    eta = 0.2,
    nthread = as.integer(cpu_threads),
    seed = 20260908L
  ),
  dtrain = probe_d,
  num_boost_round = 3L,
  verbose_eval = FALSE
)

probe_config_raw <- probe_fit$save_config()

probe_config <- tryCatch(
  as.character(
    reticulate::py_to_r(
      probe_config_raw
    )
  )[1L],
  error = function(e) {
    as.character(
      probe_config_raw
    )[1L]
  }
)

if (
  !grepl(
    '"device"[[:space:]]*:[[:space:]]*"cuda',
    probe_config,
    ignore.case = TRUE
  )
) {
  stop(
    "CUDA was not verified by the trained Python XGBoost booster.",
    call. = FALSE
  )
}

# Probe multi-quantile support before the long run.
q_probe_y <- abs(
  rnorm(500L)
)

q_probe_d <- py_xgb$DMatrix(
  probe_x,
  label = q_probe_y
)

q_probe_fit <- tryCatch(
  py_xgb$train(
    params = reticulate::dict(
      objective = "reg:quantileerror",
      quantile_alpha =
        c(
          0.10,
          0.50,
          0.90
        ),
      tree_method = "hist",
      device = "cuda",
      max_depth = 2L,
      eta = 0.2,
      nthread = as.integer(cpu_threads),
      seed = 20260908L
    ),
    dtrain = q_probe_d,
    num_boost_round = 3L,
    verbose_eval = FALSE
  ),
  error = function(e) e
)

if (
  inherits(
    q_probe_fit,
    "error"
  )
) {
  stop(
    "Python XGBoost CUDA does not support the requested multi-quantile objective: ",
    conditionMessage(q_probe_fit),
    call. = FALSE
  )
}

fwrite(
  data.table(
    python =
      as.character(
        py_cfg$python
      )[1L],
    python_xgboost_version =
      py_xgb_version,
    device = "cuda",
    logical_cpu_threads =
      logical_cores,
    allocated_host_threads =
      cpu_threads,
    quantiles =
      "0.10;0.50;0.90"
  ),
  file.path(
    out_dir,
    "00_COMPUTE_ENVIRONMENT.csv"
  )
)

rm(
  probe_x,
  probe_y,
  probe_d,
  probe_fit,
  q_probe_y,
  q_probe_d,
  q_probe_fit
)
gc()

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

normalize_id <- function(x) {
  out <- trimws(
    as.character(x)
  )

  out <- gsub(
    "\\.0$",
    "",
    out
  )

  out[
    out %in% c(
      "",
      "NA",
      "NaN",
      "<NA>"
    )
  ] <- NA_character_

  out
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

  y <- as.integer(
    y[keep]
  )

  p <- as.numeric(
    p[keep]
  )

  keep2 <- y %in% c(
    0L,
    1L
  )

  y <- y[keep2]
  p <- p[keep2]

  n1 <- as.double(
    sum(
      y == 1L
    )
  )

  n0 <- as.double(
    sum(
      y == 0L
    )
  )

  if (
    n1 == 0 ||
      n0 == 0
  ) {
    return(
      NA_real_
    )
  }

  r <- rank(
    p,
    ties.method = "average"
  )

  (
    sum(
      r[
        y == 1L
      ]
    ) -
      n1 *
        (
          n1 + 1
        ) /
        2
  ) /
    (
      n1 * n0
    )
}

fast_auprc <- function(y, p) {
  keep <- !is.na(y) &
    !is.na(p) &
    is.finite(p)

  y <- as.integer(
    y[keep]
  )

  p <- as.numeric(
    p[keep]
  )

  keep2 <- y %in% c(
    0L,
    1L
  )

  y <- y[keep2]
  p <- p[keep2]

  n_pos <- sum(
    y == 1L
  )

  if (
    n_pos == 0L
  ) {
    return(
      NA_real_
    )
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

  precision <- tp /
    (
      tp + fp
    )

  recall <- tp /
    n_pos

  recall_previous <- c(
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
          recall_previous
      ),
    na.rm = TRUE
  )
}

binary_calibration <- function(
    y,
    p
) {
  keep <- !is.na(y) &
    !is.na(p) &
    is.finite(p)

  y <- as.integer(
    y[keep]
  )

  p <- clamp_prob(
    p[keep]
  )

  if (
    length(
      unique(y)
    ) < 2L
  ) {
    return(
      c(
        intercept = NA_real_,
        slope = NA_real_
      )
    )
  }

  lp <- qlogis(p)

  fit_i <- tryCatch(
    glm(
      y ~ 1,
      family = binomial(),
      offset = lp
    ),
    error = function(e) NULL
  )

  fit_s <- tryCatch(
    glm(
      y ~ lp,
      family = binomial()
    ),
    error = function(e) NULL
  )

  c(
    intercept =
      if (
        is.null(fit_i)
      ) {
        NA_real_
      } else {
        unname(
          coef(fit_i)[1L]
        )
      },
    slope =
      if (
        is.null(fit_s)
      ) {
        NA_real_
      } else {
        unname(
          coef(fit_s)["lp"]
        )
      }
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
  truth <- matrix(
    0,
    nrow = nrow(prob),
    ncol = ncol(prob)
  )

  truth[
    cbind(
      seq_len(
        nrow(prob)
      ),
      y_index_1based
    )
  ] <- 1

  mean(
    rowSums(
      (
        prob -
          truth
      )^2
    )
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
      (
        alpha - 1
      ) * e
    ),
    na.rm = TRUE
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
    numeric_vars =
      numeric_vars,
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

    feature_names[j] <-
      v

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
    # Python XGBoost best_iteration is zero-based.
    return(
      as.integer(
        out[1L] +
          1L
      )
    )
  }

  as.integer(
    fallback
  )
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
    return(
      out[1L]
    )
  }

  NA_real_
}

as_multiclass_matrix <- function(
    pred,
    n,
    k
) {
  p <- tryCatch(
    reticulate::py_to_r(
      pred
    ),
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
      "Unexpected multiclass prediction shape.",
      call. = FALSE
    )
  }

  p
}

as_quantile_matrix <- function(
    pred,
    n,
    k
) {
  p <- tryCatch(
    reticulate::py_to_r(
      pred
    ),
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

  storage.mode(p) <-
    "double"

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

  out <- cbind(
    q10 = lo,
    q50 = mid,
    q90 = hi
  )

  out
}


python_gain_importance <- function(
    model,
    feature_names,
    model_label
) {
  score_obj <- tryCatch(
    model$get_score(
      importance_type = "gain"
    ),
    error = function(e) NULL
  )

  if (is.null(score_obj)) {
    return(
      data.table()
    )
  }

  score_r <- tryCatch(
    reticulate::py_to_r(
      score_obj
    ),
    error = function(e) score_obj
  )

  vals <- suppressWarnings(
    as.numeric(
      unlist(
        score_r,
        use.names = TRUE
      )
    )
  )

  nms <- names(
    unlist(
      score_r,
      use.names = TRUE
    )
  )

  if (
    length(vals) == 0L ||
      length(nms) == 0L
  ) {
    return(
      data.table()
    )
  }

  feature_index <- suppressWarnings(
    as.integer(
      sub(
        "^f",
        "",
        nms
      )
    ) +
      1L
  )

  mapped <- rep(
    NA_character_,
    length(
      feature_index
    )
  )

  valid <- !is.na(
    feature_index
  ) &
    feature_index >= 1L &
    feature_index <=
      length(
        feature_names
      )

  mapped[valid] <-
    feature_names[
      feature_index[valid]
    ]

  mapped[
    is.na(mapped)
  ] <- nms[
    is.na(mapped)
  ]

  out <- data.table(
    model =
      model_label,
    feature =
      mapped,
    gain =
      vals
  )

  out <- out[
    is.finite(gain) &
      gain >= 0
  ]

  if (
    nrow(out) > 0L &&
      sum(
        out$gain
      ) > 0
  ) {
    out[
      ,
      normalized_gain :=
        gain /
          sum(gain)
    ]
  } else {
    out[
      ,
      normalized_gain :=
        NA_real_
    ]
  }

  setorder(
    out,
    -gain
  )

  out
}

one_vs_rest_metrics <- function(
    y_class,
    prob,
    class_levels,
    endpoint_label
) {
  rbindlist(
    lapply(
      seq_along(
        class_levels
      ),
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
          endpoint =
            endpoint_label,
          target =
            lev,
          N =
            length(y),
          events =
            sum(
              y == 1L
            ),
          prevalence =
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
                p -
                  y
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

# -----------------------------------------------------------------------------
# Load frozen cohort / predictor dictionary
# -----------------------------------------------------------------------------

cat(
  "\nReading frozen methods dataset...\n"
)

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

required_columns <- unique(
  c(
    "admission_year",
    "inc_key",
    "hospital_days",
    "icu_days",
    "vent_days",
    numeric_predictors,
    categorical_predictors
  )
)

missing_columns <- setdiff(
  required_columns,
  names(dt)
)

if (
  length(
    missing_columns
  ) > 0L
) {
  stop(
    "Frozen dataset is missing required column(s): ",
    paste(
      missing_columns,
      collapse = ", "
    ),
    call. = FALSE
  )
}

dt[
  ,
  inc_key :=
    normalize_id(
      inc_key
    )
]

dt[
  ,
  hospital_days :=
    safe_num(
      hospital_days
    )
]

dt[
  ,
  icu_days :=
    safe_num(
      icu_days
    )
]

dt[
  ,
  vent_days :=
    safe_num(
      vent_days
    )
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

HLOS_LEVELS <- c(
  "Hospital LOS <=7 days",
  "Hospital LOS 8-27 days",
  "Hospital LOS >=28 days"
)

dt[
  ,
  hlos_trajectory_final :=
    fcase(
      is.na(
        hospital_days
      ),
      NA_character_,
      hospital_days <= 7,
      HLOS_LEVELS[1L],
      hospital_days <= 27,
      HLOS_LEVELS[2L],
      hospital_days >= 28,
      HLOS_LEVELS[3L],
      default =
        NA_character_
    )
]

dt[
  ,
  hlos_trajectory_final :=
    factor(
      hlos_trajectory_final,
      levels =
        HLOS_LEVELS
    )
]

# Cohort counts by year.
hlos_counts <- dt[
  !is.na(
    hlos_trajectory_final
  ),
  .(
    N = .N,
    hlos_le7 =
      sum(
        hlos_trajectory_final ==
          HLOS_LEVELS[1L]
      ),
    hlos_8_27 =
      sum(
        hlos_trajectory_final ==
          HLOS_LEVELS[2L]
      ),
    hlos_ge28 =
      sum(
        hlos_trajectory_final ==
          HLOS_LEVELS[3L]
      )
  ),
  by =
    admission_year
][
  order(
    admission_year
  )
]

fwrite(
  hlos_counts,
  file.path(
    out_dir,
    "01_HLOS_TRAJECTORY_COUNTS_BY_YEAR.csv"
  )
)

print(
  hlos_counts
)

# -----------------------------------------------------------------------------
# Structural hyperparameters
# -----------------------------------------------------------------------------

selected_hp <- fread(
  selected_hp_file
)

search_grid <- fread(
  search_grid_file
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
    "Prespecified tuning grid is malformed.",
    call. = FALSE
  )
}

search_grid <-
  search_grid[
    ,
    ..needed_grid_cols
  ]

hp_for <- function(endpoint_name) {
  row <- selected_hp[
    selected_hp$endpoint ==
      endpoint_name
  ]

  if (
    nrow(row) != 1L
  ) {
    stop(
      "Could not uniquely recover structural hyperparameters for ",
      endpoint,
      call. = FALSE
    )
  }

  row
}

hp_hlos_quantile <-
  hp_for(
    "hlos_ge28_final"
  )

hp_icu_quantile <-
  hp_for(
    "icu_trajectory_final"
  )

if (
  file.exists(
    vent_selected_hp_file
  )
) {
  hp_vent_quantile <-
    fread(
      vent_selected_hp_file
    )[1L]
} else {
  hp_vent_quantile <-
    hp_for(
      "vent_ge8_final"
    )
}

fwrite(
  rbindlist(
    list(
      data.table(
        duration_model =
          "Hospital LOS",
        max_depth =
          hp_hlos_quantile$max_depth,
        min_child_weight =
          hp_hlos_quantile$min_child_weight,
        subsample =
          hp_hlos_quantile$subsample,
        colsample_bytree =
          hp_hlos_quantile$colsample_bytree,
        lambda =
          hp_hlos_quantile$lambda,
        source =
          "Prior 2023 temporal tuning: HLOS >=28"
      ),
      data.table(
        duration_model =
          "ICU LOS conditional on ICU",
        max_depth =
          hp_icu_quantile$max_depth,
        min_child_weight =
          hp_icu_quantile$min_child_weight,
        subsample =
          hp_icu_quantile$subsample,
        colsample_bytree =
          hp_icu_quantile$colsample_bytree,
        lambda =
          hp_icu_quantile$lambda,
        source =
          "Prior 2023 temporal tuning: ICU trajectory"
      ),
      data.table(
        duration_model =
          "Ventilator duration conditional on ventilation",
        max_depth =
          hp_vent_quantile$max_depth,
        min_child_weight =
          hp_vent_quantile$min_child_weight,
        subsample =
          hp_vent_quantile$subsample,
        colsample_bytree =
          hp_vent_quantile$colsample_bytree,
        lambda =
          hp_vent_quantile$lambda,
        source =
          if (
            file.exists(
              vent_selected_hp_file
            )
          ) {
            "Final ventilation trajectory 2023 tuning"
          } else {
            "Prior 2023 temporal tuning: ventilation >=8"
          }
      )
    )
  ),
  file.path(
    out_dir,
    "02_DURATION_STRUCTURAL_HYPERPARAMETERS.csv"
  )
)

# -----------------------------------------------------------------------------
# Temporal encoders
# -----------------------------------------------------------------------------

idx_train_years <- which(
  dt$admission_year %in%
    2020:2022
)

idx_dev_years <- which(
  dt$admission_year %in%
    2020:2023
)

idx_test_year <- which(
  dt$admission_year ==
    2024L
)

cat(
  "\nEncoding data for 2023 tuning...\n"
)

encoder_tune <- fit_encoder(
  dt[
    idx_train_years
  ],
  numeric_predictors,
  categorical_predictors
)

X_tune_all <- encode_dense(
  dt[
    c(
      idx_train_years,
      which(
        dt$admission_year ==
          2023L
      )
    )
  ],
  encoder_tune
)

n_train_year_rows <-
  length(
    idx_train_years
  )

idx_2023_rows <- which(
  dt$admission_year ==
    2023L
)

X_train_years <- X_tune_all[
  seq_len(
    n_train_year_rows
  ),
  ,
  drop = FALSE
]

X_2023 <- X_tune_all[
  n_train_year_rows +
    seq_len(
      length(
        idx_2023_rows
      )
    ),
  ,
  drop = FALSE
]

rm(
  X_tune_all
)
gc()

cat(
  "\nEncoding final 2020-2024 model matrices...\n"
)

encoder_final <- fit_encoder(
  dt[
    idx_dev_years
  ],
  numeric_predictors,
  categorical_predictors
)

saveRDS(
  encoder_final,
  file.path(
    model_dir,
    "FINAL_hybrid_encoder_2020_2023.rds"
  )
)

X_final_all <- encode_dense(
  dt[
    c(
      idx_dev_years,
      idx_test_year
    )
  ],
  encoder_final
)

n_dev_year_rows <-
  length(
    idx_dev_years
  )

X_dev_years <- X_final_all[
  seq_len(
    n_dev_year_rows
  ),
  ,
  drop = FALSE
]

X_2024 <- X_final_all[
  n_dev_year_rows +
    seq_len(
      length(
        idx_test_year
      )
    ),
  ,
  drop = FALSE
]

rm(
  X_final_all
)
gc()

# Local-row lookups permit outcome-specific filtering without reconstructing
# encoders or leaking 2024 category levels.
train_global_to_local <- setNames(
  seq_along(
    idx_train_years
  ),
  idx_train_years
)

tune_global_to_local <- setNames(
  seq_along(
    idx_2023_rows
  ),
  idx_2023_rows
)

dev_global_to_local <- setNames(
  seq_along(
    idx_dev_years
  ),
  idx_dev_years
)

test_global_to_local <- setNames(
  seq_along(
    idx_test_year
  ),
  idx_test_year
)

local_rows <- function(
    global_rows,
    lookup
) {
  as.integer(
    lookup[
      as.character(
        global_rows
      )
    ]
  )
}

ETA <- 0.05
MAX_ROUNDS <- 3000L
EARLY_STOP <- 75L

# -----------------------------------------------------------------------------
# A) NEW HLOS 3-CLASS TRAJECTORY
# -----------------------------------------------------------------------------

cat(
  "\n============================================================\n",
  "A) HOSPITAL LOS TRAJECTORY: <=7 / 8-27 / >=28 DAYS\n",
  "============================================================\n",
  sep = ""
)

hlos_train_global <- which(
  dt$admission_year %in%
    2020:2022 &
    !is.na(
      dt$hlos_trajectory_final
    )
)

hlos_tune_global <- which(
  dt$admission_year ==
    2023L &
    !is.na(
      dt$hlos_trajectory_final
    )
)

hlos_dev_global <- which(
  dt$admission_year %in%
    2020:2023 &
    !is.na(
      dt$hlos_trajectory_final
    )
)

hlos_test_global <- which(
  dt$admission_year ==
    2024L &
    !is.na(
      dt$hlos_trajectory_final
    )
)

hlos_train_local <-
  local_rows(
    hlos_train_global,
    train_global_to_local
  )

hlos_tune_local <-
  local_rows(
    hlos_tune_global,
    tune_global_to_local
  )

hlos_dev_local <-
  local_rows(
    hlos_dev_global,
    dev_global_to_local
  )

hlos_test_local <-
  local_rows(
    hlos_test_global,
    test_global_to_local
  )

y_hlos_train <-
  as.integer(
    dt$hlos_trajectory_final[
      hlos_train_global
    ]
  ) -
    1L

y_hlos_tune <-
  as.integer(
    dt$hlos_trajectory_final[
      hlos_tune_global
    ]
  ) -
    1L

d_hlos_train <- py_xgb$DMatrix(
  X_train_years[
    hlos_train_local,
    ,
    drop = FALSE
  ],
  label = y_hlos_train
)

d_hlos_tune <- py_xgb$DMatrix(
  X_2023[
    hlos_tune_local,
    ,
    drop = FALSE
  ],
  label = y_hlos_tune
)

hlos_tuning_results <- vector(
  "list",
  nrow(
    search_grid
  )
)

for (
  i in seq_len(
    nrow(
      search_grid
    )
  )
) {
  g <- search_grid[i]

  cat(
    "HLOS trajectory | ",
    g$config_id,
    " (",
    i,
    "/",
    nrow(
      search_grid
    ),
    ")\n",
    sep = ""
  )

  params <- reticulate::dict(
    objective = "multi:softprob",
    eval_metric = "mlogloss",
    num_class = 3L,
    eta = ETA,
    max_depth =
      as.integer(
        g$max_depth
      ),
    min_child_weight =
      as.numeric(
        g$min_child_weight
      ),
    subsample =
      as.numeric(
        g$subsample
      ),
    colsample_bytree =
      as.numeric(
        g$colsample_bytree
      ),
    lambda =
      as.numeric(
        g$lambda
      ),
    tree_method = "hist",
    device = "cuda",
    nthread =
      as.integer(
        cpu_threads
      ),
    seed = 20260908L
  )

  fit <- py_xgb$train(
    params = params,
    dtrain = d_hlos_train,
    num_boost_round =
      as.integer(
        MAX_ROUNDS
      ),
    evals = list(
      reticulate::tuple(
        d_hlos_train,
        "train_2020_22"
      ),
      reticulate::tuple(
        d_hlos_tune,
        "tune_2023"
      )
    ),
    early_stopping_rounds =
      as.integer(
        EARLY_STOP
      ),
    maximize = FALSE,
    verbose_eval = FALSE
  )

  hlos_tuning_results[[i]] <-
    cbind(
      data.table(
        endpoint =
          "hlos_trajectory_final",
        best_iteration =
          extract_best_iteration(
            fit,
            MAX_ROUNDS
          ),
        tune_2023_mlogloss =
          extract_best_score(
            fit
          )
      ),
      g
    )

  rm(fit)
  gc()
}

hlos_tuning_results <- rbindlist(
  hlos_tuning_results,
  fill = TRUE
)

setorder(
  hlos_tuning_results,
  tune_2023_mlogloss,
  best_iteration
)

fwrite(
  hlos_tuning_results,
  file.path(
    out_dir,
    "03_HLOS_TRAJECTORY_TEMPORAL_TUNING_RESULTS.csv"
  )
)

hlos_selected <- copy(
  hlos_tuning_results[
    1L
  ]
)

hlos_selected[
  ,
  selected_by :=
    "Minimum 2023 multiclass log loss"
]

fwrite(
  hlos_selected,
  file.path(
    out_dir,
    "04_SELECTED_HLOS_TRAJECTORY_HYPERPARAMETERS.csv"
  )
)

print(
  hlos_selected
)

d_hlos_dev <- py_xgb$DMatrix(
  X_dev_years[
    hlos_dev_local,
    ,
    drop = FALSE
  ],
  label =
    as.integer(
      dt$hlos_trajectory_final[
        hlos_dev_global
      ]
    ) -
      1L
)

d_hlos_test <- py_xgb$DMatrix(
  X_2024[
    hlos_test_local,
    ,
    drop = FALSE
  ]
)

s <- hlos_selected[1L]

hlos_final_params <- reticulate::dict(
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = 3L,
  eta = ETA,
  max_depth =
    as.integer(
      s$max_depth
    ),
  min_child_weight =
    as.numeric(
      s$min_child_weight
    ),
  subsample =
    as.numeric(
      s$subsample
    ),
  colsample_bytree =
    as.numeric(
      s$colsample_bytree
    ),
  lambda =
    as.numeric(
      s$lambda
    ),
  tree_method = "hist",
  device = "cuda",
  nthread =
    as.integer(
      cpu_threads
    ),
  seed = 20260908L
)

hlos_final_fit <- py_xgb$train(
  params =
    hlos_final_params,
  dtrain =
    d_hlos_dev,
  num_boost_round =
    as.integer(
      s$best_iteration
    ),
  verbose_eval = FALSE
)

hlos_model_file <- file.path(
  model_dir,
  "FINAL_hlos_trajectory_model.ubj"
)

hlos_final_fit$save_model(
  hlos_model_file
)

saveRDS(
  list(
    model_file =
      hlos_model_file,
    endpoint =
      "hlos_trajectory_final",
    levels =
      HLOS_LEVELS,
    selected_hyperparameters =
      hlos_selected,
    backend =
      "Python XGBoost CUDA",
    development_years =
      "2020-2023",
    validation_year =
      2024L
  ),
  file.path(
    model_dir,
    "FINAL_hlos_trajectory_model_metadata.rds"
  )
)


hlos_importance <- python_gain_importance(
  hlos_final_fit,
  colnames(
    X_dev_years
  ),
  "Hospital LOS trajectory"
)

fwrite(
  hlos_importance,
  file.path(
    validation_dir,
    "HLOS_TRAJECTORY_FEATURE_IMPORTANCE.csv"
  )
)

hlos_pred_raw <- hlos_final_fit$predict(
  d_hlos_test
)

hlos_prob <- as_multiclass_matrix(
  hlos_pred_raw,
  length(
    hlos_test_global
  ),
  3L
)

hlos_prob <- hlos_prob /
  rowSums(
    hlos_prob
  )

colnames(
  hlos_prob
) <- HLOS_LEVELS

hlos_observed <- as.character(
  dt$hlos_trajectory_final[
    hlos_test_global
  ]
)

hlos_predicted <- HLOS_LEVELS[
  max.col(
    hlos_prob,
    ties.method = "first"
  )
]

hlos_test_index <- as.integer(
  dt$hlos_trajectory_final[
    hlos_test_global
  ]
)

hlos_overall <- data.table(
  endpoint =
    "Hospital LOS trajectory",
  N =
    length(
      hlos_test_global
    ),
  accuracy =
    mean(
      hlos_predicted ==
        hlos_observed
    ),
  multiclass_log_loss =
    multiclass_logloss(
      hlos_test_index,
      hlos_prob
    ),
  multiclass_brier =
    multiclass_brier(
      hlos_test_index,
      hlos_prob
    )
)

hlos_class_metrics <-
  one_vs_rest_metrics(
    hlos_observed,
    hlos_prob,
    HLOS_LEVELS,
    "Hospital LOS trajectory"
  )

fwrite(
  hlos_overall,
  file.path(
    validation_dir,
    "05_2024_HLOS_TRAJECTORY_OVERALL_METRICS.csv"
  )
)

fwrite(
  hlos_class_metrics,
  file.path(
    validation_dir,
    "06_2024_HLOS_TRAJECTORY_CLASS_METRICS.csv"
  )
)

hlos_pred_dt <- data.table(
  admission_year =
    2024L,
  inc_key =
    dt$inc_key[
      hlos_test_global
    ],
  observed_class =
    hlos_observed,
  predicted_class =
    hlos_predicted,
  p_hlos_le7 =
    hlos_prob[, 1L],
  p_hlos_8_27 =
    hlos_prob[, 2L],
  p_hlos_ge28 =
    hlos_prob[, 3L]
)

fwrite(
  hlos_pred_dt,
  file.path(
    prediction_dir,
    "hlos_trajectory_2024_predictions.csv"
  )
)

# -----------------------------------------------------------------------------
# B) FINAL THREE-QUANTILE DURATION MODELS
# -----------------------------------------------------------------------------

cat(
  "\n============================================================\n",
  "B) FINAL Q10 / Q50 / Q90 DURATION MODELS\n",
  "============================================================\n",
  sep = ""
)

ALPHAS <- c(
  0.10,
  0.50,
  0.90
)

duration_specs <- list(
  list(
    id =
      "hospital_los",
    label =
      "Hospital LOS",
    y_var =
      "hospital_days",
    lower_bound =
      0,
    risk_filter =
      function(d) {
        !is.na(
          d$hospital_days
        ) &
          d$hospital_days >= 0
      },
    hp =
      hp_hlos_quantile
  ),
  list(
    id =
      "icu_los_conditional",
    label =
      "ICU LOS conditional on ICU use",
    y_var =
      "icu_days",
    lower_bound =
      1,
    risk_filter =
      function(d) {
        !is.na(
          d$icu_days
        ) &
          d$icu_days > 0
      },
    hp =
      hp_icu_quantile
  ),
  list(
    id =
      "ventilator_days_conditional",
    label =
      "Ventilator duration conditional on ventilation",
    y_var =
      "vent_days",
    lower_bound =
      1,
    risk_filter =
      function(d) {
        !is.na(
          d$vent_days
        ) &
          d$vent_days > 0
      },
    hp =
      hp_vent_quantile
  )
)

duration_summary <- list()

for (
  spec in duration_specs
) {
  cat(
    "\n--- ",
    spec$label,
    " ---\n",
    sep = ""
  )

  train_global <- which(
    dt$admission_year %in%
      2020:2022 &
      spec$risk_filter(dt)
  )

  tune_global <- which(
    dt$admission_year ==
      2023L &
      spec$risk_filter(dt)
  )

  dev_global <- which(
    dt$admission_year %in%
      2020:2023 &
      spec$risk_filter(dt)
  )

  test_validation_global <- which(
    dt$admission_year ==
      2024L &
      spec$risk_filter(dt)
  )

  train_local <-
    local_rows(
      train_global,
      train_global_to_local
    )

  tune_local <-
    local_rows(
      tune_global,
      tune_global_to_local
    )

  dev_local <-
    local_rows(
      dev_global,
      dev_global_to_local
    )

  # For deployment, conditional models generate predictions for every 2024
  # patient. Validation remains restricted to the observed conditional risk set.
  test_all_local <- seq_len(
    nrow(
      X_2024
    )
  )

  y_train_raw <-
    dt[[spec$y_var]][
      train_global
    ]

  y_tune_raw <-
    dt[[spec$y_var]][
      tune_global
    ]

  y_dev_raw <-
    dt[[spec$y_var]][
      dev_global
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

  dtrain <- py_xgb$DMatrix(
    X_train_years[
      train_local,
      ,
      drop = FALSE
    ],
    label =
      y_train
  )

  dtune <- py_xgb$DMatrix(
    X_2023[
      tune_local,
      ,
      drop = FALSE
    ],
    label =
      y_tune
  )

  hp <- spec$hp[1L]

  q_params <- reticulate::dict(
    objective =
      "reg:quantileerror",
    quantile_alpha =
      ALPHAS,
    eta =
      ETA,
    max_depth =
      as.integer(
        hp$max_depth
      ),
    min_child_weight =
      as.numeric(
        hp$min_child_weight
      ),
    subsample =
      as.numeric(
        hp$subsample
      ),
    colsample_bytree =
      as.numeric(
        hp$colsample_bytree
      ),
    lambda =
      as.numeric(
        hp$lambda
      ),
    tree_method =
      "hist",
    device =
      "cuda",
    nthread =
      as.integer(
        cpu_threads
      ),
    seed =
      20260908L
  )

  q_tune_fit <- py_xgb$train(
    params =
      q_params,
    dtrain =
      dtrain,
    num_boost_round =
      as.integer(
        MAX_ROUNDS
      ),
    evals = list(
      reticulate::tuple(
        dtrain,
        "train_2020_22"
      ),
      reticulate::tuple(
        dtune,
        "tune_2023"
      )
    ),
    early_stopping_rounds =
      as.integer(
        EARLY_STOP
      ),
    maximize = FALSE,
    verbose_eval = FALSE
  )

  selected_rounds <-
    extract_best_iteration(
      q_tune_fit,
      MAX_ROUNDS
    )

  selected_score <-
    extract_best_score(
      q_tune_fit
    )

  ddev <- py_xgb$DMatrix(
    X_dev_years[
      dev_local,
      ,
      drop = FALSE
    ],
    label =
      y_dev
  )

  dtest_all <- py_xgb$DMatrix(
    X_2024[
      test_all_local,
      ,
      drop = FALSE
    ]
  )

  q_final_fit <- py_xgb$train(
    params =
      q_params,
    dtrain =
      ddev,
    num_boost_round =
      as.integer(
        selected_rounds
      ),
    verbose_eval =
      FALSE
  )

  model_file <- file.path(
    model_dir,
    paste0(
      "FINAL_",
      spec$id,
      "_Q10_Q50_Q90_model.ubj"
    )
  )

  q_final_fit$save_model(
    model_file
  )

  saveRDS(
    list(
      model_file =
        model_file,
      endpoint =
        spec$id,
      label =
        spec$label,
      quantile_alpha =
        ALPHAS,
      transform =
        "log1p",
      inverse_transform =
        "expm1",
      lower_bound =
        spec$lower_bound,
      selected_rounds =
        selected_rounds,
      tune_2023_score =
        selected_score,
      structural_hyperparameters =
        hp,
      backend =
        "Python XGBoost CUDA"
    ),
    file.path(
      model_dir,
      paste0(
        "FINAL_",
        spec$id,
        "_Q10_Q50_Q90_metadata.rds"
      )
    )
  )


  q_importance <- python_gain_importance(
    q_final_fit,
    colnames(
      X_dev_years
    ),
    spec$label
  )

  fwrite(
    q_importance,
    file.path(
      validation_dir,
      paste0(
        "FEATURE_IMPORTANCE_",
        spec$id,
        ".csv"
      )
    )
  )

  pred_trans_raw <- q_final_fit$predict(
    dtest_all
  )

  pred_trans <- as_quantile_matrix(
    pred_trans_raw,
    nrow(
      X_2024
    ),
    length(
      ALPHAS
    )
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

  pred_dt <- data.table(
    admission_year =
      dt$admission_year[
        idx_test_year
      ],
    inc_key =
      dt$inc_key[
        idx_test_year
      ],
    q10_days =
      pred_days[, 1L],
    q50_median_days =
      pred_days[, 2L],
    q90_days =
      pred_days[, 3L],
    raw_quantile_crossing =
      raw_crossing
  )

  pred_dt[
    ,
    validation_risk_set :=
      spec$risk_filter(
        dt[
          idx_test_year
        ]
      )
  ]

  pred_dt[
    ,
    observed_days :=
      dt[[spec$y_var]][
        idx_test_year
      ]
  ]

  fwrite(
    pred_dt,
    file.path(
      prediction_dir,
      paste0(
        spec$id,
        "_2024_Q10_Q50_Q90_predictions.csv"
      )
    )
  )

  eval_dt <- pred_dt[
    validation_risk_set &
      !is.na(
        observed_days
      )
  ]

  dev_median <- median(
    y_dev_raw,
    na.rm = TRUE
  )

  baseline_mae <- mean(
    abs(
      eval_dt$observed_days -
        dev_median
    ),
    na.rm = TRUE
  )

  median_mae <- mean(
    abs(
      eval_dt$observed_days -
        eval_dt$q50_median_days
    ),
    na.rm = TRUE
  )

  mean_pinball <- mean(
    c(
      pinball_loss(
        eval_dt$observed_days,
        eval_dt$q10_days,
        0.10
      ),
      pinball_loss(
        eval_dt$observed_days,
        eval_dt$q50_median_days,
        0.50
      ),
      pinball_loss(
        eval_dt$observed_days,
        eval_dt$q90_days,
        0.90
      )
    )
  )

  constant_q <- as.numeric(
    quantile(
      y_dev_raw,
      probs =
        ALPHAS,
      na.rm = TRUE,
      type = 8
    )
  )

  baseline_pinball <- mean(
    c(
      pinball_loss(
        eval_dt$observed_days,
        rep(
          constant_q[1L],
          nrow(eval_dt)
        ),
        0.10
      ),
      pinball_loss(
        eval_dt$observed_days,
        rep(
          constant_q[2L],
          nrow(eval_dt)
        ),
        0.50
      ),
      pinball_loss(
        eval_dt$observed_days,
        rep(
          constant_q[3L],
          nrow(eval_dt)
        ),
        0.90
      )
    )
  )

  eval_row <- data.table(
    outcome =
      spec$label,
    N_validation =
      nrow(
        eval_dt
      ),
    selected_rounds =
      selected_rounds,
    tune_2023_score =
      selected_score,
    selected_transform =
      "log1p",
    median_MAE_days =
      median_mae,
    baseline_development_median_MAE_days =
      baseline_mae,
    median_MAE_improvement_pct =
      100 *
        (
          baseline_mae -
            median_mae
        ) /
        baseline_mae,
    mean_pinball =
      mean_pinball,
    baseline_constant_quantile_pinball =
      baseline_pinball,
    mean_pinball_improvement_pct =
      100 *
        (
          baseline_pinball -
            mean_pinball
        ) /
        baseline_pinball,
    central_80_coverage =
      mean(
        eval_dt$observed_days >=
          eval_dt$q10_days &
          eval_dt$observed_days <=
            eval_dt$q90_days
      ),
    median_80PI_width_days =
      median(
        eval_dt$q90_days -
          eval_dt$q10_days,
        na.rm = TRUE
      ),
    raw_crossing_pct =
      100 *
        mean(
          eval_dt$raw_quantile_crossing
        )
  )

  fwrite(
    eval_row,
    file.path(
      validation_dir,
      paste0(
        "07_QUANTILE_VALIDATION_",
        spec$id,
        ".csv"
      )
    )
  )

  duration_summary[[
    length(
      duration_summary
    ) +
      1L
  ]] <- eval_row

  rm(
    dtrain,
    dtune,
    q_tune_fit,
    ddev,
    dtest_all,
    q_final_fit,
    pred_trans_raw,
    pred_trans,
    pred_days_raw,
    pred_days
  )

  gc()
}

duration_summary <- rbindlist(
  duration_summary,
  fill = TRUE
)

fwrite(
  duration_summary,
  file.path(
    validation_dir,
    "08_ALL_DURATION_MODEL_2024_VALIDATION.csv"
  )
)

# -----------------------------------------------------------------------------
# C) Assemble final bedside output manifest
# -----------------------------------------------------------------------------

manifest <- data.table(
  domain = c(
    "Disposition",
    "ICU trajectory",
    "Mechanical ventilation trajectory",
    "Hospital LOS trajectory",
    "Hospital LOS duration",
    "ICU LOS duration",
    "Ventilator duration",
    "ICP monitoring",
    "Craniotomy/craniectomy"
  ),
  final_output = c(
    "P(home/home health), P(post-acute), P(death/hospice)",
    "P(no ICU), P(ICU 1-7 d), P(ICU >=8 d)",
    "P(no ventilation), P(ventilation 1-7 d), P(ventilation >=8 d)",
    "P(HLOS <=7 d), P(HLOS 8-27 d), P(HLOS >=28 d)",
    "Median HLOS + 80% PI (Q10-Q90)",
    "If ICU required: median ICU days + 80% PI (Q10-Q90)",
    "If ventilation required: median ventilator days + 80% PI (Q10-Q90)",
    "P(invasive ICP monitoring)",
    "P(craniotomy/craniectomy)"
  ),
  model_type = c(
    "Multiclass XGBoost",
    "Multiclass XGBoost",
    "Multiclass XGBoost",
    "Multiclass XGBoost",
    "Conditional/unconditional quantile XGBoost",
    "Conditional quantile XGBoost",
    "Conditional quantile XGBoost",
    "Binary XGBoost",
    "Binary XGBoost"
  ),
  status = c(
    "Existing final model",
    "Existing final model",
    "Existing final ventilation trajectory model",
    "NEW final model from script 14",
    "NEW final Q10/Q50/Q90 model from script 14",
    "NEW final Q10/Q50/Q90 model from script 14",
    "NEW final Q10/Q50/Q90 model from script 14",
    "Existing final model",
    "Existing final model"
  )
)

fwrite(
  manifest,
  file.path(
    out_dir,
    "09_FINAL_BEDSIDE_MODEL_MANIFEST.csv"
  )
)

# -----------------------------------------------------------------------------
# D) Model-development / validation checklist
# -----------------------------------------------------------------------------

checklist <- data.table(
  item = c(
    "Prediction time explicitly defined",
    "Admission-era predictor set frozen before final modeling",
    "Race/ethnicity/payer excluded from bedside predictor set",
    "Missing predictors retained without complete-case deletion",
    "2020-2022 used for early-stopping training",
    "2023 used for model selection / boosting rounds",
    "2024 reserved for temporal performance evaluation",
    "Multiclass probabilities used rather than forced labels",
    "Clinically interpretable resource categories retained",
    "Continuous duration estimates include prediction intervals",
    "Duration interval is an 80% prediction interval, not a confidence interval",
    "Quantile transformation selected without using 2024",
    "Quantile crossing corrected only for presentation after prediction",
    "Derived any-use / extended-use probabilities come from trajectory models",
    "Ridge comparator to be rerun on final endpoint architecture",
    "Calibration / discrimination / proper scoring to be rerun",
    "Subgroup fairness / severity validation to be rerun",
    "COVID-era development sensitivity to be rerun",
    "Selection-bias / cohort-ascertainment audit retained from script 08",
    "Feature importance remains descriptive only"
  ),
  status = c(
    "DONE",
    "DONE",
    "DONE",
    "DONE",
    "DONE",
    "DONE",
    "DONE",
    "DONE",
    "DONE",
    "DONE",
    "DONE",
    "DONE",
    "DONE",
    "DONE",
    "NEXT: post-model analysis",
    "NEXT: post-model analysis",
    "NEXT: post-model analysis",
    "NEXT: post-model analysis",
    "DONE",
    "DONE"
  )
)

fwrite(
  checklist,
  file.path(
    out_dir,
    "10_METHODS_COMPLETION_CHECKLIST.csv"
  )
)

# -----------------------------------------------------------------------------
# E) Summary
# -----------------------------------------------------------------------------

summary_lines <- c(
  "FINAL TBI-TRACT HYBRID RESOURCE MODEL EXTENSION COMPLETE",
  "",
  "Final clinical architecture:",
  "  Disposition: home/home health | post-acute | death/hospice",
  "  ICU: no ICU | 1-7 d | >=8 d",
  "  Ventilation: none | 1-7 d | >=8 d",
  "  Hospital LOS: <=7 d | 8-27 d | >=28 d",
  "  HLOS duration: median + 80% PI",
  "  ICU duration if ICU required: median + 80% PI",
  "  Ventilator duration if ventilation required: median + 80% PI",
  "  ICP monitoring probability",
  "  Craniotomy/craniectomy probability",
  "",
  "Temporal design:",
  "  2020-2022 training",
  "  2023 tuning / round selection",
  "  2020-2023 final refit",
  "  2024 temporal validation",
  "",
  "Next step:",
  "  Rerun the publication-grade validation, ridge comparator, subgroup",
  "  validation, and COVID-era sensitivity against this final endpoint makeup.",
  "",
  "Review:",
  "  04_SELECTED_HLOS_TRAJECTORY_HYPERPARAMETERS.csv",
  "  validation/05_2024_HLOS_TRAJECTORY_OVERALL_METRICS.csv",
  "  validation/06_2024_HLOS_TRAJECTORY_CLASS_METRICS.csv",
  "  validation/08_ALL_DURATION_MODEL_2024_VALIDATION.csv",
  "  09_FINAL_BEDSIDE_MODEL_MANIFEST.csv",
  "  10_METHODS_COMPLETION_CHECKLIST.csv"
)

writeLines(
  summary_lines,
  file.path(
    out_dir,
    "FINAL_HYBRID_MODEL_SUMMARY.txt"
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
  "\nHLOS trajectory 2024 metrics:\n"
)

print(
  hlos_class_metrics
)

cat(
  "\nDuration-model 2024 metrics:\n"
)

print(
  duration_summary
)

rm(
  X_train_years,
  X_2023,
  X_dev_years,
  X_2024
)

gc()
