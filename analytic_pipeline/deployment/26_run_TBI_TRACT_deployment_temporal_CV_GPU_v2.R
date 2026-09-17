# =============================================================================
# 26_run_TBI_TRACT_deployment_temporal_CV_GPU_v2.R
#
# PURPOSE
#   Lock the final 2020-2024 deployment-model specification without changing
#   the manuscript-facing temporal performance estimates.
#
# STRATEGY
#   1) Structural hyperparameters are selected by temporal-CV consensus from the
#      already completed forward folds (2022, 2023, 2024).
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
config_file <- config_candidates[file.exists(config_candidates)][1L]
if (length(config_file) == 0L || is.na(config_file)) {
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
    strata = if (is.null(strata)) {
      "__ALL__"
    } else {
      as.character(strata)
    }
  )

  d[
    is.na(strata) | strata == "",
    strata := "__UNKNOWN__"
  ]

  d[
    ,
    fold_id := {
      ord <- sample.int(.N)
      out <- integer(.N)
      out[ord] <- rep(seq_len(k), length.out = .N)
      out
    },
    by = .(
      year,
      strata
    )
  ]

  d[order(row_id), fold_id]
}

to_python_folds <- function(fold_id, k = 5L) {
  pairs <- lapply(
    seq_len(k),
    function(f) {
      reticulate::tuple(
        as.integer(which(fold_id != f) - 1L),
        as.integer(which(fold_id == f) - 1L)
      )
    }
  )
  reticulate::r_to_py(pairs)
}

select_cv_rounds <- function(
    X,
    y,
    year,
    strata,
    params,
    max_rounds = 2500L,
    early_stop = 100L,
    k = 5L,
    seed = 20260913L
) {
  fold_id <- make_balanced_folds(
    year = year,
    strata = strata,
    k = k,
    seed = seed
  )

  folds_py <- to_python_folds(
    fold_id,
    k
  )

  d <- xgb$DMatrix(
    X,
    label = y
  )

  cv <- xgb$cv(
    params = params,
    dtrain = d,
    num_boost_round = as.integer(max_rounds),
    folds = folds_py,
    early_stopping_rounds = as.integer(early_stop),
    maximize = FALSE,
    verbose_eval = FALSE,
    shuffle = FALSE,
    seed = as.integer(seed)
  )

  cv_r <- as.data.table(
    reticulate::py_to_r(cv)
  )

  if (nrow(cv_r) < 1L) {
    stop("XGBoost CV returned no rows.", call. = FALSE)
  }

  list(
    selected_rounds = nrow(cv_r),
    fold_id = fold_id,
    cv_log = cv_r
  )
}

# -----------------------------------------------------------------------------
# Load data / predictor policy
# -----------------------------------------------------------------------------

dt <- as.data.table(readRDS(dataset_file))
types <- fread(types_file)

numeric_predictors <- types[
  category == "numeric",
  predictor
]
categorical_predictors <- types[
  category == "categorical",
  predictor
]
current_predictors <- c(
  numeric_predictors,
  categorical_predictors
)

pragmatic_clinical <- setdiff(
  current_predictors,
  c(
    "helmet_use_recovered",
    "respiratoryassistance_clean"
  )
)

required_retained <- c(
  "gcsq_intubated_recovered",
  "gcsq_sedated_paralyzed_recovered",
  "supplemental_oxygen_recovered"
)

if (!all(required_retained %in% pragmatic_clinical)) {
  stop("Final pragmatic predictor set is malformed.", call. = FALSE)
}

social_vars <- c(
  "race_clean",
  "ethnicity_clean",
  "insurance_clean"
)

system_context_predictors <- unique(
  c(
    pragmatic_clinical,
    social_vars
  )
)

# -----------------------------------------------------------------------------
# Outcomes
# -----------------------------------------------------------------------------

dt[, hospital_days := safe_num(hospital_days)]
dt[, icu_days := safe_num(icu_days)]
dt[, vent_days := safe_num(vent_days)]

dt[hospital_days < 0, hospital_days := NA_real_]
dt[icu_days < 0, icu_days := NA_real_]
dt[vent_days < 0, vent_days := NA_real_]

DISCHARGE_LEVELS <- c(
  "Home/home health",
  "Post-acute facility",
  "Death/hospice"
)

ICU_LEVELS <- c(
  "No ICU",
  "ICU 1-7 days",
  "ICU >=8 days"
)

VENT_LEVELS <- c(
  "No ventilation",
  "Ventilation 1-7 days",
  "Ventilation >=8 days"
)

HLOS_LEVELS <- c(
  "Hospital LOS <=7 days",
  "Hospital LOS 8-27 days",
  "Hospital LOS >=28 days"
)

dt[
  ,
  ventilation_trajectory_final :=
    fcase(
      is.na(vent_days),
      NA_character_,
      vent_days <= 0,
      VENT_LEVELS[1L],
      vent_days <= 7,
      VENT_LEVELS[2L],
      vent_days >= 8,
      VENT_LEVELS[3L],
      default = NA_character_
    )
]

dt[
  ,
  hlos_trajectory_final :=
    fcase(
      is.na(hospital_days),
      NA_character_,
      hospital_days <= 7,
      HLOS_LEVELS[1L],
      hospital_days <= 27,
      HLOS_LEVELS[2L],
      hospital_days >= 28,
      HLOS_LEVELS[3L],
      default = NA_character_
    )
]

# -----------------------------------------------------------------------------
# Final endpoint specifications
# -----------------------------------------------------------------------------

class_specs <- list(
  list(
    endpoint_id = "discharge_3cat_final",
    endpoint_label = "Disposition",
    type = "multiclass",
    levels = DISCHARGE_LEVELS,
    phase = "B_SYSTEM_MEDIATED",
    variant = "PRAGMATIC_PLUS_RACE_ETHNICITY_PAYER",
    predictors = system_context_predictors
  ),
  list(
    endpoint_id = "hlos_trajectory_final",
    endpoint_label = "Hospital LOS trajectory",
    type = "multiclass",
    levels = HLOS_LEVELS,
    phase = "B_SYSTEM_MEDIATED",
    variant = "PRAGMATIC_PLUS_RACE_ETHNICITY_PAYER",
    predictors = system_context_predictors
  ),
  list(
    endpoint_id = "icu_trajectory_final",
    endpoint_label = "ICU trajectory",
    type = "multiclass",
    levels = ICU_LEVELS,
    phase = "A_ALL_ENDPOINTS",
    variant = "PRAGMATIC_CLINICAL",
    predictors = pragmatic_clinical
  ),
  list(
    endpoint_id = "ventilation_trajectory_final",
    endpoint_label = "Mechanical ventilation trajectory",
    type = "multiclass",
    levels = VENT_LEVELS,
    phase = "A_ALL_ENDPOINTS",
    variant = "PRAGMATIC_CLINICAL",
    predictors = pragmatic_clinical
  ),
  list(
    endpoint_id = "icp_pressure_monitor_final",
    endpoint_label = "Invasive ICP monitoring",
    type = "binary",
    levels = NULL,
    phase = "A_ALL_ENDPOINTS",
    variant = "PRAGMATIC_CLINICAL",
    predictors = pragmatic_clinical
  ),
  list(
    endpoint_id = "craniotomy_craniectomy_final",
    endpoint_label = "Craniotomy/craniectomy",
    type = "binary",
    levels = NULL,
    phase = "A_ALL_ENDPOINTS",
    variant = "PRAGMATIC_CLINICAL",
    predictors = pragmatic_clinical
  )
)

duration_specs <- list(
  list(
    endpoint_id = "hospital_los",
    endpoint_label = "Hospital LOS",
    y_var = "hospital_days",
    type = "quantile",
    lower_bound = 0,
    filter = function(x) !is.na(x) & x >= 0,
    predictors = pragmatic_clinical
  ),
  list(
    endpoint_id = "icu_los_conditional",
    endpoint_label = "ICU LOS conditional on ICU use",
    y_var = "icu_days",
    type = "quantile",
    lower_bound = 1,
    filter = function(x) !is.na(x) & x > 0,
    predictors = pragmatic_clinical
  ),
  list(
    endpoint_id = "ventilator_days_conditional",
    endpoint_label = "Ventilator duration conditional on ventilation",
    y_var = "vent_days",
    type = "quantile",
    lower_bound = 1,
    filter = function(x) !is.na(x) & x > 0,
    predictors = pragmatic_clinical
  )
)

# -----------------------------------------------------------------------------
# Temporal-CV consensus structural selection
# -----------------------------------------------------------------------------

read_grid_family <- function(spec, family = c("classification", "duration")) {
  family <- match.arg(family)

  files <- if (family == "classification") {
    file.path(
      class_dir,
      "grid_search",
      paste0(
        spec$phase,
        "__",
        c("TEST_2022", "TEST_2023", "TEST_2024"),
        "__",
        spec$endpoint_id,
        "__",
        spec$variant,
        ".csv"
      )
    )
  } else {
    file.path(
      duration_dir,
      "grid_search",
      paste0(
        c("TEST_2022", "TEST_2023", "TEST_2024"),
        "__",
        spec$endpoint_id,
        "__PRAGMATIC_CLINICAL.csv"
      )
    )
  }

  if (!all(file.exists(files))) {
    stop(
      "Missing temporal grid file(s) for ",
      spec$endpoint_id,
      call. = FALSE
    )
  }

  folds <- c(
    "TEST_2022",
    "TEST_2023",
    "TEST_2024"
  )

  rbindlist(
    lapply(
      seq_along(files),
      function(i) {
        d <- fread(files[i])

        required_cols <- c(
          "config_id",
          "tune_score",
          "max_depth",
          "min_child_weight",
          "subsample",
          "colsample_bytree",
          "lambda"
        )

        missing_cols <- setdiff(
          required_cols,
          names(d)
        )

        if (length(missing_cols) > 0L) {
          stop(
            "Temporal grid file is missing required column(s): ",
            paste(missing_cols, collapse = ", "),
            "\nFile: ",
            files[i],
            "\nAvailable columns: ",
            paste(names(d), collapse = ", "),
            call. = FALSE
          )
        }

        d[, fold_id := folds[i]]

        score <- as.numeric(
          d[["tune_score"]]
        )

        if (!any(is.finite(score))) {
          stop(
            "No finite tune_score values in temporal grid file: ",
            files[i],
            call. = FALSE
          )
        }

        min_loss <- min(
          score,
          na.rm = TRUE
        )

        loss_denom <- max(
          abs(min_loss),
          .Machine$double.eps
        )

        d[
          ,
          within_fold_rank :=
            frank(
              get("tune_score"),
              ties.method = "min"
            )
        ]

        d[
          ,
          relative_excess_loss :=
            (
              get("tune_score") -
                min_loss
            ) /
            loss_denom
        ]

        d
      }
    ),
    fill = TRUE
  )
}

aggregate_grid <- function(d, endpoint_id_value, endpoint_label_value, model_family_value) {
  key_check <- d[
    ,
    .N,
    by = .(
      fold_id,
      config_id
    )
  ]

  if (any(key_check$N != 1L)) {
    stop(
      "Temporal grid contains duplicate fold/config rows for ",
      endpoint_id_value,
      call. = FALSE
    )
  }

  fold_count <- uniqueN(
    d$fold_id
  )

  if (fold_count != 3L) {
    stop(
      "Expected 3 temporal folds for ",
      endpoint_id_value,
      " but found ",
      fold_count,
      ".",
      call. = FALSE
    )
  }

  a <- d[
    ,
    .(
      n_temporal_folds = .N,
      wins = sum(
        within_fold_rank == 1L
      ),
      mean_rank = mean(
        within_fold_rank
      ),
      median_rank = median(
        within_fold_rank
      ),
      worst_rank = max(
        within_fold_rank
      ),
      mean_relative_excess_loss = mean(
        relative_excess_loss
      ),
      worst_relative_excess_loss = max(
        relative_excess_loss
      ),
      max_depth = first(
        max_depth
      ),
      min_child_weight = first(
        min_child_weight
      ),
      subsample = first(
        subsample
      ),
      colsample_bytree = first(
        colsample_bytree
      ),
      lambda = first(
        lambda
      )
    ),
    by = config_id
  ]

  a[
    ,
    `:=`(
      endpoint_id = endpoint_id_value,
      endpoint_label = endpoint_label_value,
      model_family = model_family_value
    )
  ]

  # Consensus selection:
  # 1. lowest mean relative excess temporal tuning loss
  # 2. lowest mean rank
  # 3. lowest worst rank
  # 4. shallower depth
  # 5. larger lambda
  setorder(
    a,
    mean_relative_excess_loss,
    mean_rank,
    worst_rank,
    max_depth,
    -lambda
  )

  a[
    ,
    selected_temporal_consensus :=
      seq_len(.N) == 1L
  ]

  a
}

all_agg <- list()
selected_specs <- list()

for (spec in class_specs) {
  d <- read_grid_family(
    spec,
    "classification"
  )

  a <- aggregate_grid(
    d,
    spec$endpoint_id,
    spec$endpoint_label,
    "classification"
  )

  all_agg[[
    length(all_agg) + 1L
  ]] <- a

  s <- copy(
    a[
      selected_temporal_consensus == TRUE
    ]
  )

  s[
    ,
    `:=`(
      type = spec$type,
      variant = spec$variant,
      predictor_policy =
        if (
          spec$variant ==
            "PRAGMATIC_PLUS_RACE_ETHNICITY_PAYER"
        ) {
          "Pragmatic clinical + race + ethnicity + payer"
        } else {
          "Pragmatic clinical"
        }
    )
  ]

  selected_specs[[
    length(selected_specs) + 1L
  ]] <- s
}

for (spec in duration_specs) {
  d <- read_grid_family(
    spec,
    "duration"
  )

  a <- aggregate_grid(
    d,
    spec$endpoint_id,
    spec$endpoint_label,
    "duration"
  )

  all_agg[[
    length(all_agg) + 1L
  ]] <- a

  s <- copy(
    a[
      selected_temporal_consensus == TRUE
    ]
  )

  s[
    ,
    `:=`(
      type = "quantile",
      variant = "PRAGMATIC_CLINICAL",
      predictor_policy = "Pragmatic clinical"
    )
  ]

  selected_specs[[
    length(selected_specs) + 1L
  ]] <- s
}

agg_table <- rbindlist(
  all_agg,
  fill = TRUE
)
selected_table <- rbindlist(
  selected_specs,
  fill = TRUE
)

fwrite(
  agg_table,
  file.path(
    out_dir,
    "01_TEMPORAL_CV_STRUCTURAL_CONFIG_AGGREGATION.csv"
  )
)

fwrite(
  selected_table,
  file.path(
    out_dir,
    "02_TEMPORAL_CV_SELECTED_STRUCTURAL_CONFIG.csv"
  )
)

# -----------------------------------------------------------------------------
# Full-development 5-fold CV for boosting rounds, then final fit
# -----------------------------------------------------------------------------

round_rows <- list()
manifest_rows <- list()

fit_and_save_classification <- function(spec) {
  structural <- selected_table[
    endpoint_id ==
      spec$endpoint_id
  ]

  if (nrow(structural) != 1L) {
    stop(
      "Structural selection is not unique for ",
      spec$endpoint_id,
      call. = FALSE
    )
  }

  n_keep <- intersect(
    numeric_predictors,
    spec$predictors
  )
  c_keep <- intersect(
    unique(
      c(
        categorical_predictors,
        social_vars
      )
    ),
    spec$predictors
  )

  encoder <- fit_encoder(
    dt,
    n_keep,
    c_keep
  )

  X <- encode_dense(
    dt,
    encoder
  )

  if (spec$type == "binary") {
    y <- safe_num(
      dt[[spec$endpoint_id]]
    )
    y[
      !y %in%
        c(
          0,
          1
        )
    ] <- NA_real_

    keep <- !is.na(y)

    X_use <- X[
      keep,
      ,
      drop = FALSE
    ]
    y_use <- as.integer(
      y[
        keep
      ]
    )
    year_use <- dt$admission_year[
      keep
    ]
    strata_use <- y_use

    params <- reticulate::dict(
      objective = "binary:logistic",
      eval_metric = "logloss",
      eta = 0.05,
      max_depth = as.integer(structural$max_depth),
      min_child_weight = as.numeric(structural$min_child_weight),
      subsample = as.numeric(structural$subsample),
      colsample_bytree = as.numeric(structural$colsample_bytree),
      lambda = as.numeric(structural$lambda),
      tree_method = "hist",
      device = "cuda",
      nthread = as.integer(cpu_threads),
      seed = 20260913L
    )
  } else {
    yf <- factor(
      as.character(
        dt[[spec$endpoint_id]]
      ),
      levels = spec$levels
    )

    keep <- !is.na(yf)

    X_use <- X[
      keep,
      ,
      drop = FALSE
    ]
    y_use <- as.integer(
      yf[
        keep
      ]
    ) -
      1L
    year_use <- dt$admission_year[
      keep
    ]
    strata_use <- as.character(
      yf[
        keep
      ]
    )

    params <- reticulate::dict(
      objective = "multi:softprob",
      eval_metric = "mlogloss",
      num_class = as.integer(
        length(
          spec$levels
        )
      ),
      eta = 0.05,
      max_depth = as.integer(structural$max_depth),
      min_child_weight = as.numeric(structural$min_child_weight),
      subsample = as.numeric(structural$subsample),
      colsample_bytree = as.numeric(structural$colsample_bytree),
      lambda = as.numeric(structural$lambda),
      tree_method = "hist",
      device = "cuda",
      nthread = as.integer(cpu_threads),
      seed = 20260913L
    )
  }

  cv <- select_cv_rounds(
    X = X_use,
    y = y_use,
    year = year_use,
    strata = strata_use,
    params = params,
    max_rounds = 2500L,
    early_stop = 100L,
    k = 5L,
    seed = 20260913L
  )

  selected_rounds <- cv$selected_rounds

  fwrite(
    cv$cv_log,
    file.path(
      out_dir,
      paste0(
        "cv_log__",
        spec$endpoint_id,
        ".csv"
      )
    )
  )

  d_all <- xgb$DMatrix(
    X_use,
    label = y_use
  )

  final_fit <- xgb$train(
    params = params,
    dtrain = d_all,
    num_boost_round = as.integer(
      selected_rounds
    ),
    verbose_eval = FALSE
  )

  model_path <- file.path(
    model_dir,
    paste0(
      spec$endpoint_id,
      ".json"
    )
  )

  final_fit$save_model(
    model_path
  )

  encoder_path <- file.path(
    encoder_dir,
    paste0(
      spec$endpoint_id,
      "_encoder.rds"
    )
  )

  saveRDS(
    list(
      endpoint_id = spec$endpoint_id,
      endpoint_label = spec$endpoint_label,
      type = spec$type,
      levels = spec$levels,
      predictors = spec$predictors,
      encoder = encoder,
      feature_names = colnames(X_use),
      structural_config_id = structural$config_id,
      selected_rounds = selected_rounds,
      training_years = 2020:2024,
      n_training = nrow(X_use)
    ),
    encoder_path
  )

  round_rows[[
    length(round_rows) + 1L
  ]] <<- data.table(
    endpoint_id = spec$endpoint_id,
    endpoint_label = spec$endpoint_label,
    model_family = "classification",
    config_id = structural$config_id,
    selected_rounds_5fold_cv = selected_rounds,
    N = nrow(X_use)
  )

  manifest_rows[[
    length(manifest_rows) + 1L
  ]] <<- data.table(
    endpoint_id = spec$endpoint_id,
    endpoint_label = spec$endpoint_label,
    model_family = "classification",
    predictor_policy =
      if (
        spec$variant ==
          "PRAGMATIC_PLUS_RACE_ETHNICITY_PAYER"
      ) {
        "Pragmatic clinical + race + ethnicity + payer"
      } else {
        "Pragmatic clinical"
      },
    config_id = structural$config_id,
    max_depth = structural$max_depth,
    min_child_weight = structural$min_child_weight,
    subsample = structural$subsample,
    colsample_bytree = structural$colsample_bytree,
    lambda = structural$lambda,
    selected_rounds = selected_rounds,
    model_path = model_path,
    encoder_path = encoder_path,
    N = nrow(X_use)
  )

  rm(
    X,
    X_use,
    d_all,
    final_fit
  )
  gc()
}

fit_and_save_duration <- function(spec) {
  structural <- selected_table[
    endpoint_id ==
      spec$endpoint_id
  ]

  if (nrow(structural) != 1L) {
    stop(
      "Structural selection is not unique for ",
      spec$endpoint_id,
      call. = FALSE
    )
  }

  n_keep <- intersect(
    numeric_predictors,
    spec$predictors
  )
  c_keep <- intersect(
    categorical_predictors,
    spec$predictors
  )

  encoder <- fit_encoder(
    dt,
    n_keep,
    c_keep
  )

  X <- encode_dense(
    dt,
    encoder
  )

  y_raw <- safe_num(
    dt[[spec$y_var]]
  )

  keep <- spec$filter(
    y_raw
  )

  X_use <- X[
    keep,
    ,
    drop = FALSE
  ]
  y_use <- log1p(
    y_raw[
      keep
    ]
  )
  year_use <- dt$admission_year[
    keep
  ]

  # For duration CV, distribute the outcome range across folds within each year
  # using year-specific quintile strata. This is ONLY for round selection.
  strat_dt <- data.table(
    year = year_use,
    y = y_raw[
      keep
    ]
  )

  strat_dt[
    ,
    outcome_stratum := {
      rr <- frank(
        y,
        ties.method = "average"
      ) /
        .N

      paste0(
        "Q",
        pmin(
          5L,
          pmax(
            1L,
            ceiling(
              5 *
                rr
            )
          )
        )
      )
    },
    by = year
  ]

  params <- reticulate::dict(
    objective = "reg:quantileerror",
    quantile_alpha = c(
      0.10,
      0.50,
      0.90
    ),
    eta = 0.05,
    max_depth = as.integer(structural$max_depth),
    min_child_weight = as.numeric(structural$min_child_weight),
    subsample = as.numeric(structural$subsample),
    colsample_bytree = as.numeric(structural$colsample_bytree),
    lambda = as.numeric(structural$lambda),
    tree_method = "hist",
    device = "cuda",
    nthread = as.integer(cpu_threads),
    seed = 20260913L
  )

  cv <- select_cv_rounds(
    X = X_use,
    y = y_use,
    year = year_use,
    strata = strat_dt$outcome_stratum,
    params = params,
    max_rounds = 2500L,
    early_stop = 100L,
    k = 5L,
    seed = 20260913L
  )

  selected_rounds <- cv$selected_rounds

  fwrite(
    cv$cv_log,
    file.path(
      out_dir,
      paste0(
        "cv_log__",
        spec$endpoint_id,
        ".csv"
      )
    )
  )

  d_all <- xgb$DMatrix(
    X_use,
    label = y_use
  )

  final_fit <- xgb$train(
    params = params,
    dtrain = d_all,
    num_boost_round = as.integer(
      selected_rounds
    ),
    verbose_eval = FALSE
  )

  model_path <- file.path(
    model_dir,
    paste0(
      spec$endpoint_id,
      ".json"
    )
  )

  final_fit$save_model(
    model_path
  )

  encoder_path <- file.path(
    encoder_dir,
    paste0(
      spec$endpoint_id,
      "_encoder.rds"
    )
  )

  saveRDS(
    list(
      endpoint_id = spec$endpoint_id,
      endpoint_label = spec$endpoint_label,
      type = "quantile",
      quantile_alpha = c(
        0.10,
        0.50,
        0.90
      ),
      lower_bound = spec$lower_bound,
      predictors = spec$predictors,
      encoder = encoder,
      feature_names = colnames(X_use),
      structural_config_id = structural$config_id,
      selected_rounds = selected_rounds,
      training_years = 2020:2024,
      N = nrow(X_use)
    ),
    encoder_path
  )

  round_rows[[
    length(round_rows) + 1L
  ]] <<- data.table(
    endpoint_id = spec$endpoint_id,
    endpoint_label = spec$endpoint_label,
    model_family = "duration",
    config_id = structural$config_id,
    selected_rounds_5fold_cv = selected_rounds,
    N = nrow(X_use)
  )

  manifest_rows[[
    length(manifest_rows) + 1L
  ]] <<- data.table(
    endpoint_id = spec$endpoint_id,
    endpoint_label = spec$endpoint_label,
    model_family = "duration",
    predictor_policy = "Pragmatic clinical",
    config_id = structural$config_id,
    max_depth = structural$max_depth,
    min_child_weight = structural$min_child_weight,
    subsample = structural$subsample,
    colsample_bytree = structural$colsample_bytree,
    lambda = structural$lambda,
    selected_rounds = selected_rounds,
    model_path = model_path,
    encoder_path = encoder_path,
    N = nrow(X_use)
  )

  rm(
    X,
    X_use,
    d_all,
    final_fit
  )
  gc()
}

for (spec in class_specs) {
  cat(
    "\n============================================================\n",
    "DEPLOYMENT CV: ",
    spec$endpoint_label,
    "\n",
    "============================================================\n",
    sep = ""
  )
  fit_and_save_classification(
    spec
  )
}

for (spec in duration_specs) {
  cat(
    "\n============================================================\n",
    "DEPLOYMENT CV: ",
    spec$endpoint_label,
    "\n",
    "============================================================\n",
    sep = ""
  )
  fit_and_save_duration(
    spec
  )
}

round_table <- rbindlist(
  round_rows,
  fill = TRUE
)
manifest <- rbindlist(
  manifest_rows,
  fill = TRUE
)

fwrite(
  round_table,
  file.path(
    out_dir,
    "03_FULL_DEVELOPMENT_5FOLD_CV_ROUNDS.csv"
  )
)

fwrite(
  manifest,
  file.path(
    out_dir,
    "04_FINAL_DEPLOYMENT_MODEL_MANIFEST.csv"
  )
)

saveRDS(
  list(
    classification = class_specs,
    duration = duration_specs,
    pragmatic_clinical_predictors = pragmatic_clinical,
    social_context_predictors = social_vars,
    deployment_manifest = manifest
  ),
  file.path(
    out_dir,
    "05_FINAL_DEPLOYMENT_SPECIFICATION.rds"
  )
)

writeLines(
  c(
    "TBI-TRACT FINAL DEPLOYMENT CV COMPLETE",
    "",
    "Structural configuration:",
    "  selected by consensus across the three rolling-origin temporal tuning folds.",
    "",
    "Boosting rounds:",
    "  selected by year/outcome-stratified 5-fold CV using the entire 2020-2024 development cohort.",
    "",
    "Important:",
    "  the 5-fold CV estimates are NOT manuscript-facing validation estimates.",
    "  manuscript-facing performance remains the forward temporal 2022/2023/2024 evaluation.",
    "",
    "Final deployment models were fit on all 2020-2024 observations and saved as JSON.",
    "",
    "Review:",
    "  01_TEMPORAL_CV_STRUCTURAL_CONFIG_AGGREGATION.csv",
    "  02_TEMPORAL_CV_SELECTED_STRUCTURAL_CONFIG.csv",
    "  03_FULL_DEVELOPMENT_5FOLD_CV_ROUNDS.csv",
    "  04_FINAL_DEPLOYMENT_MODEL_MANIFEST.csv"
  ),
  file.path(
    out_dir,
    "DEPLOYMENT_CV_SUMMARY.txt"
  )
)

cat("\nDONE: ", out_dir, "\n", sep = "")
