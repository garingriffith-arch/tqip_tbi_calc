# =============================================================================
# 23_run_TBI_TRACT_pragmatic_finalist_retune_GPU.R
#
# PURPOSE
#   Final targeted adjudication of the clinical predictor backbone and
#   sociodemographic augmentation strategy.
#
# PHASE A — CLINICAL BACKBONE, ALL CLASSIFICATION ENDPOINTS
#   Compare:
#     CURRENT_REFERENCE
#     PRAGMATIC_CLINICAL
#
#   PRAGMATIC_CLINICAL removes ONLY:
#     - helmet_use_recovered
#     - respiratoryassistance_clean
#
#   It RETAINS:
#     - gcsq_intubated_recovered
#     - gcsq_sedated_paralyzed_recovered
#     - supplemental_oxygen_recovered
#
#   Endpoints:
#     - disposition
#     - ICU trajectory
#     - ventilation trajectory
#     - HLOS trajectory
#     - ICP monitoring
#     - craniotomy/craniectomy
#
# PHASE B — SYSTEM-MEDIATED OUTCOMES ONLY
#   For disposition and HLOS trajectory, compare:
#     PRAGMATIC_CLINICAL
#     PRAGMATIC_PLUS_PAYER
#     PRAGMATIC_PLUS_RACE_PAYER
#     PRAGMATIC_PLUS_RACE_ETHNICITY_PAYER
#
# RATIONALE
#   Race, ethnicity, and payer are not being evaluated as biological risk
#   factors. They are evaluated as social / health-system contextual predictors
#   for outcomes that are materially mediated by care pathways and access.
#
# TEMPORAL DESIGN
#   Fold 1: train 2020      | tune 2021 | refit 2020-2021 | test 2022
#   Fold 2: train 2020-2021 | tune 2022 | refit 2020-2022 | test 2023
#   Fold 3: train 2020-2022 | tune 2023 | refit 2020-2023 | test 2024
#
# EACH candidate is fully retuned over the original prespecified structural
# XGBoost grid within each fold and endpoint.
#
# PERFORMANCE
#   AUROC, AUPRC, Brier, log loss, calibration intercept/slope, multiclass
#   log loss/Brier, subgroup performance by race/ethnicity/payer.
#
# IMPORTANT
#   This is model development. If model choices are made from these results,
#   2024 is a temporally later development evaluation cohort, not an untouched
#   final evaluation set. External validation remains the cleanest final test.
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
# Config / files
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
    paste(
      needed_files[!file.exists(needed_files)],
      collapse = "\n"
    ),
    call. = FALSE
  )
}

out_dir <- file.path(
  output_dir,
  "TBI_TRACT_PRAGMATIC_FINALIST_RETUNE"
)

grid_dir <- file.path(
  out_dir,
  "grid_search"
)

dir.create(
  out_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  grid_dir,
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
    floor(
      0.94 *
        logical_cores
    )
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

py_cfg <- reticulate::py_config()

xgb <- reticulate::import(
  "xgboost",
  convert = TRUE
)

# CUDA probe.
set.seed(20260912L)

probe_x <- matrix(
  rnorm(4000),
  nrow = 500L,
  ncol = 8L
)

probe_y <- as.integer(
  rbinom(500L, 1L, 0.3)
)

probe_d <- xgb$DMatrix(
  probe_x,
  label = probe_y
)

probe_fit <- xgb$train(
  params = reticulate::dict(
    objective = "binary:logistic",
    tree_method = "hist",
    device = "cuda",
    max_depth = 2L,
    eta = 0.2,
    nthread = as.integer(cpu_threads),
    seed = 20260912L
  ),
  dtrain = probe_d,
  num_boost_round = 3L,
  verbose_eval = FALSE
)

probe_cfg_raw <- probe_fit$save_config()

probe_cfg <- tryCatch(
  as.character(
    reticulate::py_to_r(
      probe_cfg_raw
    )
  )[1L],
  error = function(e) {
    as.character(
      probe_cfg_raw
    )[1L]
  }
)

if (
  !grepl(
    '"device"[[:space:]]*:[[:space:]]*"cuda',
    probe_cfg,
    ignore.case = TRUE
  )
) {
  stop(
    "CUDA was not verified by the XGBoost probe.",
    call. = FALSE
  )
}

rm(
  probe_x,
  probe_y,
  probe_d,
  probe_fit
)

gc()

fwrite(
  data.table(
    python = as.character(py_cfg$python)[1L],
    device = "cuda",
    logical_cpu_threads = logical_cores,
    allocated_host_threads = cpu_threads
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

normalize_character <- function(x) {
  x <- trimws(
    as.character(x)
  )

  x[
    is.na(x) |
      x == "" |
      x %in%
        c(
          "NA",
          "NaN",
          "<NA>"
        )
  ] <- "__UNKNOWN__"

  x
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

# AUROC with double-precision event counts to avoid integer overflow.
fast_auc <- function(
    y,
    p
) {
  keep <- !is.na(y) &
    !is.na(p) &
    is.finite(p)

  y <- as.integer(
    y[keep]
  )

  p <- as.numeric(
    p[keep]
  )

  keep2 <- y %in%
    c(
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
      n1 *
        n0
    )
}

fast_auprc <- function(
    y,
    p
) {
  keep <- !is.na(y) &
    !is.na(p) &
    is.finite(p)

  y <- as.integer(
    y[keep]
  )

  p <- as.numeric(
    p[keep]
  )

  keep2 <- y %in%
    c(
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
      tp +
        fp
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

binary_logloss <- function(
    y,
    p
) {
  p <- clamp_prob(p)

  -mean(
    y *
      log(p) +
      (
        1 -
          y
      ) *
        log(
          1 -
            p
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
      unique(
        y
      )
    ) <
      2L
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
        is.null(
          fit_i
        )
      ) {
        NA_real_
      } else {
        unname(
          coef(
            fit_i
          )[1L]
        )
      },
    slope =
      if (
        is.null(
          fit_s
        )
      ) {
        NA_real_
      } else {
        unname(
          coef(
            fit_s
          )["lp"]
        )
      }
  )
}

binary_metrics <- function(
    y,
    p
) {
  cal <- binary_calibration(
    y,
    p
  )

  data.table(
    N =
      length(
        y
      ),
    events =
      sum(
        y ==
          1L
      ),
    prevalence =
      mean(
        y ==
          1L
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
    log_loss =
      binary_logloss(
        y,
        p
      ),
    calibration_intercept =
      unname(
        cal[
          "intercept"
        ]
      ),
    calibration_slope =
      unname(
        cal[
          "slope"
        ]
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
      1 -
        1e-15
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
    nrow =
      nrow(
        prob
      ),
    ncol =
      ncol(
        prob
      )
  )

  truth[
    cbind(
      seq_len(
        nrow(
          prob
        )
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

fit_encoder <- function(
    d,
    numeric_vars,
    categorical_vars
) {
  cat_levels <- lapply(
    categorical_vars,
    function(v) {
      x <- normalize_character(
        d[[v]]
      )

      unique(
        c(
          sort(
            unique(
              x
            )
          ),
          "__UNKNOWN__",
          "__OTHER__"
        )
      )
    }
  )

  names(
    cat_levels
  ) <- categorical_vars

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
  n <- nrow(
    d
  )

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
    nrow =
      n,
    ncol =
      n_numeric +
        n_cat
  )

  feature_names <- character(
    ncol(
      X
    )
  )

  parent_names <- character(
    ncol(
      X
    )
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

    parent_names[j] <-
      v

    j <- j +
      1L
  }

  for (
    v in encoder$categorical_vars
  ) {
    lev <-
      encoder$categorical_levels[[v]]

    x <- normalize_character(
      d[[v]]
    )

    x[
      !x %in%
        lev
    ] <- "__OTHER__"

    idx <- match(
      x,
      lev
    )

    cols <- j:(
      j +
        length(
          lev
        ) -
        1L
    )

    X[
      cbind(
        seq_len(
          n
        ),
        cols[
          idx
        ]
      )
    ] <- 1

    feature_names[
      cols
    ] <- paste0(
      v,
      "__",
      make.names(
        lev,
        unique = TRUE
      )
    )

    parent_names[
      cols
    ] <- v

    j <- max(
      cols
    ) +
      1L
  }

  colnames(
    X
  ) <- feature_names

  attr(
    X,
    "parent_predictor"
  ) <- parent_names

  storage.mode(
    X
  ) <- "double"

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
    is.null(
      parent
    ) ||
      length(
        parent
      ) !=
        ncol(
          X
        )
  ) {
    stop(
      "Encoded parent-predictor map is missing.",
      call. = FALSE
    )
  }

  parent %in%
    keep_predictors
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
    error =
      function(e) NA_real_
  )

  if (
    length(
      out
    ) >=
      1L &&
      is.finite(
        out[1L]
      )
  ) {
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

extract_best_score <- function(
    model
) {
  out <- tryCatch(
    as.numeric(
      reticulate::py_to_r(
        model$best_score
      )
    ),
    error =
      function(e) NA_real_
  )

  if (
    length(
      out
    ) >=
      1L &&
      is.finite(
        out[1L]
      )
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
    error =
      function(e) pred
  )

  if (
    is.matrix(
      p
    ) ||
      is.data.frame(
        p
      )
  ) {
    p <- as.matrix(
      p
    )
  } else {
    p <- matrix(
      as.numeric(
        p
      ),
      nrow =
        n,
      ncol =
        k,
      byrow =
        TRUE
    )
  }

  if (
    nrow(
      p
    ) !=
      n ||
      ncol(
        p
      ) !=
        k
  ) {
    stop(
      "Unexpected multiclass prediction shape.",
      call. = FALSE
    )
  }

  storage.mode(
    p
  ) <- "double"

  p
}

endpoint_observed <- function(
    d,
    endpoint_id,
    endpoint_type,
    class_levels = NULL
) {
  if (
    endpoint_type ==
      "binary"
  ) {
    y <- safe_num(
      d[[endpoint_id]]
    )

    y[
      !y %in%
        c(
          0,
          1
        )
    ] <- NA_real_

    return(
      y
    )
  }

  factor(
    as.character(
      d[[endpoint_id]]
    ),
    levels =
      class_levels
  )
}

subgroup_binary_metrics <- function(
    y,
    p,
    subgroup,
    subgroup_domain,
    min_n = 500L,
    min_events = 25L,
    min_nonevents = 25L
) {
  d <- data.table(
    y =
      as.integer(
        y
      ),
    p =
      as.numeric(
        p
      ),
    subgroup =
      normalize_character(
        subgroup
      )
  )

  rows <- list()

  for (
    lev in unique(
      d$subgroup
    )
  ) {
    ds <- d[
      subgroup ==
        lev
    ]

    n <- nrow(
      ds
    )

    events <- sum(
      ds$y ==
        1L
    )

    nonevents <-
      n -
        events

    if (
      n <
        min_n ||
        events <
          min_events ||
        nonevents <
          min_nonevents
    ) {
      next
    }

    met <- binary_metrics(
      ds$y,
      ds$p
    )

    met[
      ,
      `:=`(
        subgroup_domain =
          subgroup_domain,
        subgroup_level =
          lev
      )
    ]

    rows[[
      length(
        rows
      ) +
        1L
    ]] <- met
  }

  if (
    length(
      rows
    ) ==
      0L
  ) {
    return(
      data.table()
    )
  }

  rbindlist(
    rows,
    fill = TRUE
  )
}

# -----------------------------------------------------------------------------
# Data / candidates
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

if (
  !all(
    required_grid_cols %in%
      names(
        search_grid
      )
  )
) {
  stop(
    "Prespecified search grid is malformed.",
    call. = FALSE
  )
}

search_grid <- search_grid[
  ,
  ..required_grid_cols
]

numeric_predictors <- types[
  category ==
    "numeric",
  predictor
]

categorical_predictors <- types[
  category ==
    "categorical",
  predictor
]

current_predictors <- c(
  numeric_predictors,
  categorical_predictors
)

SOCIAL_VARS <- c(
  "race_clean",
  "ethnicity_clean",
  "insurance_clean"
)

missing_social <- setdiff(
  SOCIAL_VARS,
  names(
    dt
  )
)

if (
  length(
    missing_social
  ) >
    0L
) {
  stop(
    "Missing social variable(s): ",
    paste(
      missing_social,
      collapse = ", "
    ),
    call. = FALSE
  )
}

pragmatic_removed <- c(
  "helmet_use_recovered",
  "respiratoryassistance_clean"
)

pragmatic_clinical <- setdiff(
  current_predictors,
  pragmatic_removed
)

required_retained <- c(
  "gcsq_intubated_recovered",
  "gcsq_sedated_paralyzed_recovered",
  "supplemental_oxygen_recovered"
)

if (
  !all(
    required_retained %in%
      pragmatic_clinical
  )
) {
  stop(
    "PRAGMATIC_CLINICAL unexpectedly removed required airway-context variables.",
    call. = FALSE
  )
}

phase_a_variants <- list(
  CURRENT_REFERENCE =
    current_predictors,
  PRAGMATIC_CLINICAL =
    pragmatic_clinical
)

phase_b_variants <- list(
  PRAGMATIC_CLINICAL =
    pragmatic_clinical,
  PRAGMATIC_PLUS_PAYER =
    unique(
      c(
        pragmatic_clinical,
        "insurance_clean"
      )
    ),
  PRAGMATIC_PLUS_RACE_PAYER =
    unique(
      c(
        pragmatic_clinical,
        "race_clean",
        "insurance_clean"
      )
    ),
  PRAGMATIC_PLUS_RACE_ETHNICITY_PAYER =
    unique(
      c(
        pragmatic_clinical,
        SOCIAL_VARS
      )
    )
)

all_numeric_predictors <- numeric_predictors

all_categorical_predictors <- unique(
  c(
    categorical_predictors,
    SOCIAL_VARS
  )
)

variant_defs <- rbindlist(
  list(
    rbindlist(
      lapply(
        names(
          phase_a_variants
        ),
        function(v) {
          data.table(
            phase =
              "A_ALL_ENDPOINTS",
            variant_id =
              v,
            n_predictors =
              length(
                phase_a_variants[[v]]
              ),
            removed_from_current =
              paste(
                setdiff(
                  current_predictors,
                  phase_a_variants[[v]]
                ),
                collapse = ";"
              ),
            added_to_current =
              paste(
                setdiff(
                  phase_a_variants[[v]],
                  current_predictors
                ),
                collapse = ";"
              )
          )
        }
      )
    ),
    rbindlist(
      lapply(
        names(
          phase_b_variants
        ),
        function(v) {
          data.table(
            phase =
              "B_SYSTEM_MEDIATED",
            variant_id =
              v,
            n_predictors =
              length(
                phase_b_variants[[v]]
              ),
            removed_from_current =
              paste(
                setdiff(
                  current_predictors,
                  phase_b_variants[[v]]
                ),
                collapse = ";"
              ),
            added_to_current =
              paste(
                setdiff(
                  phase_b_variants[[v]],
                  current_predictors
                ),
                collapse = ";"
              )
          )
        }
      )
    )
  ),
  fill = TRUE
)

fwrite(
  variant_defs,
  file.path(
    out_dir,
    "01_PRAGMATIC_VARIANT_DEFINITIONS.csv"
  )
)

# -----------------------------------------------------------------------------
# Outcomes / folds
# -----------------------------------------------------------------------------

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
  hospital_days <
    0,
  hospital_days :=
    NA_real_
]

dt[
  icu_days <
    0,
  icu_days :=
    NA_real_
]

dt[
  vent_days <
    0,
  vent_days :=
    NA_real_
]

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
      is.na(
        vent_days
      ),
      NA_character_,
      vent_days <=
        0,
      VENT_LEVELS[1L],
      vent_days <=
        7,
      VENT_LEVELS[2L],
      vent_days >=
        8,
      VENT_LEVELS[3L],
      default =
        NA_character_
    )
]

dt[
  ,
  hlos_trajectory_final :=
    fcase(
      is.na(
        hospital_days
      ),
      NA_character_,
      hospital_days <=
        7,
      HLOS_LEVELS[1L],
      hospital_days <=
        27,
      HLOS_LEVELS[2L],
      hospital_days >=
        28,
      HLOS_LEVELS[3L],
      default =
        NA_character_
    )
]

all_endpoint_specs <- list(
  list(
    id =
      "discharge_3cat_final",
    label =
      "Disposition",
    type =
      "multiclass",
    levels =
      DISCHARGE_LEVELS
  ),
  list(
    id =
      "icu_trajectory_final",
    label =
      "ICU trajectory",
    type =
      "multiclass",
    levels =
      ICU_LEVELS
  ),
  list(
    id =
      "ventilation_trajectory_final",
    label =
      "Mechanical ventilation trajectory",
    type =
      "multiclass",
    levels =
      VENT_LEVELS
  ),
  list(
    id =
      "hlos_trajectory_final",
    label =
      "Hospital LOS trajectory",
    type =
      "multiclass",
    levels =
      HLOS_LEVELS
  ),
  list(
    id =
      "icp_pressure_monitor_final",
    label =
      "Invasive ICP monitoring",
    type =
      "binary",
    levels =
      NULL
  ),
  list(
    id =
      "craniotomy_craniectomy_final",
    label =
      "Craniotomy/craniectomy",
    type =
      "binary",
    levels =
      NULL
  )
)

system_endpoint_specs <- all_endpoint_specs[
  vapply(
    all_endpoint_specs,
    function(x) {
      x$id %in%
        c(
          "discharge_3cat_final",
          "hlos_trajectory_final"
        )
    },
    logical(1)
  )
]

folds <- list(
  TEST_2022 = list(
    train_years =
      2020L,
    tune_year =
      2021L,
    dev_years =
      2020:2021,
    test_year =
      2022L
  ),
  TEST_2023 = list(
    train_years =
      2020:2021,
    tune_year =
      2022L,
    dev_years =
      2020:2022,
    test_year =
      2023L
  ),
  TEST_2024 = list(
    train_years =
      2020:2022,
    tune_year =
      2023L,
    dev_years =
      2020:2023,
    test_year =
      2024L
  )
)

ETA <- 0.05
MAX_ROUNDS <- 3000L
EARLY_STOP <- 75L

# -----------------------------------------------------------------------------
# Generic fully-retuned phase runner
# -----------------------------------------------------------------------------

run_phase <- function(
    phase_name,
    endpoint_specs,
    variant_keep
) {
  selected_hp_rows <- list()
  overall_rows <- list()
  class_rows <- list()
  subgroup_rows <- list()

  for (
    fold_name in names(
      folds
    )
  ) {
    fold <- folds[[fold_name]]

    cat(
      "\n============================================================\n",
      phase_name,
      " | ",
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
      dt[
        idx_train
      ],
      all_numeric_predictors,
      all_categorical_predictors
    )

    X_train <- encode_dense(
      dt[
        idx_train
      ],
      encoder_tune
    )

    X_tune <- encode_dense(
      dt[
        idx_tune
      ],
      encoder_tune
    )

    encoder_final <- fit_encoder(
      dt[
        idx_dev
      ],
      all_numeric_predictors,
      all_categorical_predictors
    )

    X_dev <- encode_dense(
      dt[
        idx_dev
      ],
      encoder_final
    )

    X_test <- encode_dense(
      dt[
        idx_test
      ],
      encoder_final
    )

    subgroup_test <- data.table(
      race =
        normalize_character(
          dt$race_clean[
            idx_test
          ]
        ),
      ethnicity =
        normalize_character(
          dt$ethnicity_clean[
            idx_test
          ]
        ),
      payer =
        normalize_character(
          dt$insurance_clean[
            idx_test
          ]
        )
    )

    for (
      spec in endpoint_specs
    ) {
      cat(
        "\n--- ",
        spec$label,
        " ---\n",
        sep = ""
      )

      y_all <- endpoint_observed(
        dt,
        spec$id,
        spec$type,
        spec$levels
      )

      train_keep <- !is.na(
        y_all[
          idx_train
        ]
      )

      tune_keep <- !is.na(
        y_all[
          idx_tune
        ]
      )

      dev_keep <- !is.na(
        y_all[
          idx_dev
        ]
      )

      test_keep <- !is.na(
        y_all[
          idx_test
        ]
      )

      for (
        variant_id in names(
          variant_keep
        )
      ) {
        cat(
          "  Candidate: ",
          variant_id,
          "\n",
          sep = ""
        )

        keep_predictors <- variant_keep[[variant_id]]

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
            colnames(
              X_train
            )[
              cols_train
            ],
            colnames(
              X_tune
            )[
              cols_tune
            ]
          )
        ) {
          stop(
            "Train/tune feature mismatch: ",
            phase_name,
            " / ",
            fold_name,
            " / ",
            spec$id,
            " / ",
            variant_id,
            call. = FALSE
          )
        }

        if (
          !identical(
            colnames(
              X_dev
            )[
              cols_dev
            ],
            colnames(
              X_test
            )[
              cols_test
            ]
          )
        ) {
          stop(
            "Dev/test feature mismatch: ",
            phase_name,
            " / ",
            fold_name,
            " / ",
            spec$id,
            " / ",
            variant_id,
            call. = FALSE
          )
        }

        if (
          spec$type ==
            "binary"
        ) {
          y_train <- as.integer(
            y_all[
              idx_train
            ][
              train_keep
            ]
          )

          y_tune <- as.integer(
            y_all[
              idx_tune
            ][
              tune_keep
            ]
          )

          dtrain <- xgb$DMatrix(
            X_train[
              train_keep,
              cols_train,
              drop = FALSE
            ],
            label =
              y_train
          )

          dtune <- xgb$DMatrix(
            X_tune[
              tune_keep,
              cols_tune,
              drop = FALSE
            ],
            label =
              y_tune
          )

        } else {
          y_train <- as.integer(
            y_all[
              idx_train
            ][
              train_keep
            ]
          ) -
            1L

          y_tune <- as.integer(
            y_all[
              idx_tune
            ][
              tune_keep
            ]
          ) -
            1L

          dtrain <- xgb$DMatrix(
            X_train[
              train_keep,
              cols_train,
              drop = FALSE
            ],
            label =
              y_train
          )

          dtune <- xgb$DMatrix(
            X_tune[
              tune_keep,
              cols_tune,
              drop = FALSE
            ],
            label =
              y_tune
          )
        }

        grid_rows <- vector(
          "list",
          nrow(
            search_grid
          )
        )

        for (
          grid_i in seq_len(
            nrow(
              search_grid
            )
          )
        ) {
          g <- search_grid[
            grid_i
          ]

          if (
            spec$type ==
              "binary"
          ) {
            params <- reticulate::dict(
              objective =
                "binary:logistic",
              eval_metric =
                "logloss",
              eta =
                ETA,
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
              tree_method =
                "hist",
              device =
                "cuda",
              nthread =
                as.integer(
                  cpu_threads
                ),
              seed =
                20260912L
            )
          } else {
            params <- reticulate::dict(
              objective =
                "multi:softprob",
              eval_metric =
                "mlogloss",
              num_class =
                as.integer(
                  length(
                    spec$levels
                  )
                ),
              eta =
                ETA,
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
              tree_method =
                "hist",
              device =
                "cuda",
              nthread =
                as.integer(
                  cpu_threads
                ),
              seed =
                20260912L
            )
          }

          fit <- xgb$train(
            params =
              params,
            dtrain =
              dtrain,
            num_boost_round =
              as.integer(
                MAX_ROUNDS
              ),
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
            early_stopping_rounds =
              as.integer(
                EARLY_STOP
              ),
            maximize =
              FALSE,
            verbose_eval =
              FALSE
          )

          grid_rows[[
            grid_i
          ]] <- cbind(
            data.table(
              phase =
                phase_name,
              fold_id =
                fold_name,
              test_year =
                fold$test_year,
              endpoint_id =
                spec$id,
              endpoint_label =
                spec$label,
              variant_id =
                variant_id,
              best_iteration =
                extract_best_iteration(
                  fit,
                  MAX_ROUNDS
                ),
              tune_score =
                extract_best_score(
                  fit
                )
            ),
            g
          )

          rm(
            fit
          )

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

        selected <- copy(
          grid_table[
            1L
          ]
        )

        selected[
          ,
          selected_by :=
            "Minimum fold-specific pre-test temporal tuning loss"
        ]

        selected_hp_rows[[
          length(
            selected_hp_rows
          ) +
            1L
        ]] <- selected

        fwrite(
          grid_table,
          file.path(
            grid_dir,
            paste0(
              phase_name,
              "__",
              fold_name,
              "__",
              spec$id,
              "__",
              variant_id,
              ".csv"
            )
          )
        )

        hp <- selected[
          1L
        ]

        rm(
          dtrain,
          dtune
        )

        gc()

        if (
          spec$type ==
            "binary"
        ) {
          y_dev <- as.integer(
            y_all[
              idx_dev
            ][
              dev_keep
            ]
          )

          y_test <- as.integer(
            y_all[
              idx_test
            ][
              test_keep
            ]
          )

          params_final <- reticulate::dict(
            objective =
              "binary:logistic",
            eval_metric =
              "logloss",
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
              20260912L
          )

          ddev <- xgb$DMatrix(
            X_dev[
              dev_keep,
              cols_dev,
              drop = FALSE
            ],
            label =
              y_dev
          )

          dtest <- xgb$DMatrix(
            X_test[
              test_keep,
              cols_test,
              drop = FALSE
            ]
          )

          final_fit <- xgb$train(
            params =
              params_final,
            dtrain =
              ddev,
            num_boost_round =
              as.integer(
                hp$best_iteration
              ),
            verbose_eval =
              FALSE
          )

          pred <- as.numeric(
            final_fit$predict(
              dtest
            )
          )

          met <- binary_metrics(
            y_test,
            pred
          )

          met[
            ,
            `:=`(
              phase =
                phase_name,
              fold_id =
                fold_name,
              test_year =
                fold$test_year,
              endpoint_id =
                spec$id,
              endpoint_label =
                spec$label,
              target =
                spec$label,
              variant_id =
                variant_id,
              selected_rounds =
                as.integer(
                  hp$best_iteration
                )
            )
          ]

          overall_rows[[
            length(
              overall_rows
            ) +
              1L
          ]] <- met

          subgroup_slice <- subgroup_test[
            test_keep
          ]

          for (
            subgroup_name in c(
              "race",
              "ethnicity",
              "payer"
            )
          ) {
            sm <- subgroup_binary_metrics(
              y_test,
              pred,
              subgroup_slice[[subgroup_name]],
              subgroup_name
            )

            if (
              nrow(
                sm
              ) >
                0L
            ) {
              sm[
                ,
                `:=`(
                  phase =
                    phase_name,
                  fold_id =
                    fold_name,
                  test_year =
                    fold$test_year,
                  endpoint_id =
                    spec$id,
                  endpoint_label =
                    spec$label,
                  target =
                    spec$label,
                  variant_id =
                    variant_id
                )
              ]

              subgroup_rows[[
                length(
                  subgroup_rows
                ) +
                  1L
              ]] <- sm
            }
          }

        } else {
          y_dev <- as.integer(
            y_all[
              idx_dev
            ][
              dev_keep
            ]
          ) -
            1L

          y_test_factor <- y_all[
            idx_test
          ][
            test_keep
          ]

          params_final <- reticulate::dict(
            objective =
              "multi:softprob",
            eval_metric =
              "mlogloss",
            num_class =
              as.integer(
                length(
                  spec$levels
                )
              ),
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
              20260912L
          )

          ddev <- xgb$DMatrix(
            X_dev[
              dev_keep,
              cols_dev,
              drop = FALSE
            ],
            label =
              y_dev
          )

          dtest <- xgb$DMatrix(
            X_test[
              test_keep,
              cols_test,
              drop = FALSE
            ]
          )

          final_fit <- xgb$train(
            params =
              params_final,
            dtrain =
              ddev,
            num_boost_round =
              as.integer(
                hp$best_iteration
              ),
            verbose_eval =
              FALSE
          )

          prob <- as_multiclass_matrix(
            final_fit$predict(
              dtest
            ),
            sum(
              test_keep
            ),
            length(
              spec$levels
            )
          )

          prob <- prob /
            rowSums(
              prob
            )

          colnames(
            prob
          ) <- spec$levels

          class_metric_rows <- list()

          subgroup_slice <- subgroup_test[
            test_keep
          ]

          for (
            class_i in seq_along(
              spec$levels
            )
          ) {
            lev <- spec$levels[
              class_i
            ]

            y_binary <- as.integer(
              as.character(
                y_test_factor
              ) ==
                lev
            )

            p_binary <- prob[
              ,
              class_i
            ]

            cm <- binary_metrics(
              y_binary,
              p_binary
            )

            cm[
              ,
              `:=`(
                phase =
                  phase_name,
                fold_id =
                  fold_name,
                test_year =
                  fold$test_year,
                endpoint_id =
                  spec$id,
                endpoint_label =
                  spec$label,
                target =
                  lev,
                variant_id =
                  variant_id,
                selected_rounds =
                  as.integer(
                    hp$best_iteration
                  )
              )
            ]

            class_metric_rows[[
              length(
                class_metric_rows
              ) +
                1L
            ]] <- cm

            for (
              subgroup_name in c(
                "race",
                "ethnicity",
                "payer"
              )
            ) {
              sm <- subgroup_binary_metrics(
                y_binary,
                p_binary,
                subgroup_slice[[subgroup_name]],
                subgroup_name
              )

              if (
                nrow(
                  sm
                ) >
                  0L
              ) {
                sm[
                  ,
                  `:=`(
                    phase =
                      phase_name,
                    fold_id =
                      fold_name,
                    test_year =
                      fold$test_year,
                    endpoint_id =
                      spec$id,
                    endpoint_label =
                      spec$label,
                    target =
                      lev,
                    variant_id =
                      variant_id
                  )
                ]

                subgroup_rows[[
                  length(
                    subgroup_rows
                  ) +
                    1L
                ]] <- sm
              }
            }
          }

          if (
            spec$id ==
              "icu_trajectory_final"
          ) {
            y_any <- as.integer(
              as.character(
                y_test_factor
              ) !=
                "No ICU"
            )

            p_any <- 1 -
              prob[
                ,
                "No ICU"
              ]

            cm <- binary_metrics(
              y_any,
              p_any
            )

            cm[
              ,
              `:=`(
                phase =
                  phase_name,
                fold_id =
                  fold_name,
                test_year =
                  fold$test_year,
                endpoint_id =
                  spec$id,
                endpoint_label =
                  spec$label,
                target =
                  "Any ICU use",
                variant_id =
                  variant_id,
                selected_rounds =
                  as.integer(
                    hp$best_iteration
                  )
              )
            ]

            class_metric_rows[[
              length(
                class_metric_rows
              ) +
                1L
            ]] <- cm
          }

          if (
            spec$id ==
              "ventilation_trajectory_final"
          ) {
            y_any <- as.integer(
              as.character(
                y_test_factor
              ) !=
                "No ventilation"
            )

            p_any <- 1 -
              prob[
                ,
                "No ventilation"
              ]

            cm <- binary_metrics(
              y_any,
              p_any
            )

            cm[
              ,
              `:=`(
                phase =
                  phase_name,
                fold_id =
                  fold_name,
                test_year =
                  fold$test_year,
                endpoint_id =
                  spec$id,
                endpoint_label =
                  spec$label,
                target =
                  "Any ventilation",
                variant_id =
                  variant_id,
                selected_rounds =
                  as.integer(
                    hp$best_iteration
                  )
              )
            ]

            class_metric_rows[[
              length(
                class_metric_rows
              ) +
                1L
            ]] <- cm
          }

          class_metric_table <- rbindlist(
            class_metric_rows,
            fill = TRUE
          )

          class_rows[[
            length(
              class_rows
            ) +
              1L
          ]] <- class_metric_table

          y_index <- as.integer(
            y_test_factor
          )

          base_classes <- class_metric_table[
            !target %in%
              c(
                "Any ICU use",
                "Any ventilation"
              )
          ]

          overall_rows[[
            length(
              overall_rows
            ) +
              1L
          ]] <- data.table(
            phase =
              phase_name,
            fold_id =
              fold_name,
            test_year =
              fold$test_year,
            endpoint_id =
              spec$id,
            endpoint_label =
              spec$label,
            target =
              "<multiclass overall>",
            variant_id =
              variant_id,
            selected_rounds =
              as.integer(
                hp$best_iteration
              ),
            N =
              length(
                y_index
              ),
            accuracy =
              mean(
                max.col(
                  prob,
                  ties.method =
                    "first"
                ) ==
                  y_index
              ),
            multiclass_log_loss =
              multiclass_logloss(
                y_index,
                prob
              ),
            multiclass_brier =
              multiclass_brier(
                y_index,
                prob
              ),
            macro_AUROC =
              mean(
                base_classes$AUROC,
                na.rm = TRUE
              ),
            macro_AUPRC =
              mean(
                base_classes$AUPRC,
                na.rm = TRUE
              )
          )
        }

        rm(
          ddev,
          dtest,
          final_fit
        )

        if (
          exists(
            "pred"
          )
        ) {
          rm(
            pred
          )
        }

        if (
          exists(
            "prob"
          )
        ) {
          rm(
            prob
          )
        }

        gc()
      }

      rm(
        y_all
      )

      gc()
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

  list(
    selected_hp =
      rbindlist(
        selected_hp_rows,
        fill = TRUE
      ),
    overall =
      rbindlist(
        overall_rows,
        fill = TRUE
      ),
    class_metrics =
      if (
        length(
          class_rows
        ) >
          0L
      ) {
        rbindlist(
          class_rows,
          fill = TRUE
        )
      } else {
        data.table()
      },
    subgroup =
      if (
        length(
          subgroup_rows
        ) >
          0L
      ) {
        rbindlist(
          subgroup_rows,
          fill = TRUE
        )
      } else {
        data.table()
      }
  )
}

# -----------------------------------------------------------------------------
# Run Phase A and Phase B
# -----------------------------------------------------------------------------

phase_a <- run_phase(
  "A_ALL_ENDPOINTS",
  all_endpoint_specs,
  phase_a_variants
)

phase_b <- run_phase(
  "B_SYSTEM_MEDIATED",
  system_endpoint_specs,
  phase_b_variants
)

# -----------------------------------------------------------------------------
# Save raw outputs
# -----------------------------------------------------------------------------

selected_hp <- rbindlist(
  list(
    phase_a$selected_hp,
    phase_b$selected_hp
  ),
  fill = TRUE
)

overall <- rbindlist(
  list(
    phase_a$overall,
    phase_b$overall
  ),
  fill = TRUE
)

class_metrics <- rbindlist(
  list(
    phase_a$class_metrics,
    phase_b$class_metrics
  ),
  fill = TRUE
)

subgroup_metrics <- rbindlist(
  list(
    phase_a$subgroup,
    phase_b$subgroup
  ),
  fill = TRUE
)

fwrite(
  selected_hp,
  file.path(
    out_dir,
    "02_SELECTED_PRAGMATIC_HYPERPARAMETERS.csv"
  )
)

fwrite(
  overall,
  file.path(
    out_dir,
    "03_PRAGMATIC_OVERALL_METRICS.csv"
  )
)

fwrite(
  class_metrics,
  file.path(
    out_dir,
    "04_PRAGMATIC_CLASS_METRICS.csv"
  )
)

fwrite(
  subgroup_metrics,
  file.path(
    out_dir,
    "05_PRAGMATIC_SUBGROUP_METRICS.csv"
  )
)

# -----------------------------------------------------------------------------
# Phase A: pragmatic vs current
# -----------------------------------------------------------------------------

phase_a_class <- class_metrics[
  phase ==
    "A_ALL_ENDPOINTS"
]

phase_a_ref <- phase_a_class[
  variant_id ==
    "CURRENT_REFERENCE"
]

phase_a_delta <- merge(
  phase_a_class,
  phase_a_ref,
  by = c(
    "phase",
    "fold_id",
    "test_year",
    "endpoint_id",
    "endpoint_label",
    "target"
  ),
  suffixes = c(
    "",
    "_reference"
  )
)

for (
  m in c(
    "AUROC",
    "AUPRC",
    "Brier",
    "log_loss",
    "calibration_intercept",
    "calibration_slope"
  )
) {
  phase_a_delta[
    ,
    paste0(
      "delta_",
      m
    ) :=
      get(
        m
      ) -
        get(
          paste0(
            m,
            "_reference"
          )
        )
  ]
}

phase_a_pragmatic <- phase_a_delta[
  variant_id ==
    "PRAGMATIC_CLINICAL"
]

fwrite(
  phase_a_pragmatic,
  file.path(
    out_dir,
    "06_PRAGMATIC_CLINICAL_VS_CURRENT_CLASS_DELTAS.csv"
  )
)

phase_a_summary <- phase_a_pragmatic[
  ,
  .(
    n_fold_target_comparisons =
      .N,
    median_delta_AUROC =
      median(
        delta_AUROC,
        na.rm = TRUE
      ),
    worst_delta_AUROC =
      min(
        delta_AUROC,
        na.rm = TRUE
      ),
    best_delta_AUROC =
      max(
        delta_AUROC,
        na.rm = TRUE
      ),
    median_delta_AUPRC =
      median(
        delta_AUPRC,
        na.rm = TRUE
      ),
    worst_delta_AUPRC =
      min(
        delta_AUPRC,
        na.rm = TRUE
      ),
    best_delta_AUPRC =
      max(
        delta_AUPRC,
        na.rm = TRUE
      ),
    median_delta_Brier =
      median(
        delta_Brier,
        na.rm = TRUE
      ),
    median_delta_log_loss =
      median(
        delta_log_loss,
        na.rm = TRUE
      )
  ),
  by = .(
    endpoint_id,
    endpoint_label
  )
]

fwrite(
  phase_a_summary,
  file.path(
    out_dir,
    "07_PRAGMATIC_CLINICAL_VS_CURRENT_TEMPORAL_SUMMARY.csv"
  )
)

# -----------------------------------------------------------------------------
# Phase B: social/system variables vs pragmatic clinical
# -----------------------------------------------------------------------------

phase_b_class <- class_metrics[
  phase ==
    "B_SYSTEM_MEDIATED"
]

phase_b_ref <- phase_b_class[
  variant_id ==
    "PRAGMATIC_CLINICAL"
]

phase_b_delta <- merge(
  phase_b_class,
  phase_b_ref,
  by = c(
    "phase",
    "fold_id",
    "test_year",
    "endpoint_id",
    "endpoint_label",
    "target"
  ),
  suffixes = c(
    "",
    "_reference"
  )
)

for (
  m in c(
    "AUROC",
    "AUPRC",
    "Brier",
    "log_loss",
    "calibration_intercept",
    "calibration_slope"
  )
) {
  phase_b_delta[
    ,
    paste0(
      "delta_",
      m
    ) :=
      get(
        m
      ) -
        get(
          paste0(
            m,
            "_reference"
          )
        )
  ]
}

fwrite(
  phase_b_delta,
  file.path(
    out_dir,
    "08_SYSTEM_MEDIATED_CLASS_DELTAS_VS_PRAGMATIC.csv"
  )
)

phase_b_summary <- phase_b_delta[
  variant_id !=
    "PRAGMATIC_CLINICAL",
  .(
    n_fold_target_comparisons =
      .N,
    median_delta_AUROC =
      median(
        delta_AUROC,
        na.rm = TRUE
      ),
    worst_delta_AUROC =
      min(
        delta_AUROC,
        na.rm = TRUE
      ),
    best_delta_AUROC =
      max(
        delta_AUROC,
        na.rm = TRUE
      ),
    median_delta_AUPRC =
      median(
        delta_AUPRC,
        na.rm = TRUE
      ),
    worst_delta_AUPRC =
      min(
        delta_AUPRC,
        na.rm = TRUE
      ),
    best_delta_AUPRC =
      max(
        delta_AUPRC,
        na.rm = TRUE
      ),
    median_delta_Brier =
      median(
        delta_Brier,
        na.rm = TRUE
      ),
    median_delta_log_loss =
      median(
        delta_log_loss,
        na.rm = TRUE
      )
  ),
  by = .(
    endpoint_id,
    endpoint_label,
    variant_id
  )
]

fwrite(
  phase_b_summary,
  file.path(
    out_dir,
    "09_SYSTEM_MEDIATED_TEMPORAL_SUMMARY.csv"
  )
)

# -----------------------------------------------------------------------------
# Phase B subgroup calibration deltas
# -----------------------------------------------------------------------------

phase_b_subgroup <- subgroup_metrics[
  phase ==
    "B_SYSTEM_MEDIATED"
]

phase_b_subgroup_ref <- phase_b_subgroup[
  variant_id ==
    "PRAGMATIC_CLINICAL"
]

phase_b_subgroup_delta <- merge(
  phase_b_subgroup,
  phase_b_subgroup_ref,
  by = c(
    "phase",
    "fold_id",
    "test_year",
    "endpoint_id",
    "endpoint_label",
    "target",
    "subgroup_domain",
    "subgroup_level"
  ),
  suffixes = c(
    "",
    "_reference"
  )
)

for (
  m in c(
    "AUROC",
    "AUPRC",
    "Brier",
    "log_loss",
    "calibration_intercept",
    "calibration_slope"
  )
) {
  phase_b_subgroup_delta[
    ,
    paste0(
      "delta_",
      m
    ) :=
      get(
        m
      ) -
        get(
          paste0(
            m,
            "_reference"
          )
        )
  ]
}

phase_b_subgroup_delta[
  ,
  delta_abs_calibration_intercept :=
    abs(
      calibration_intercept
    ) -
      abs(
        calibration_intercept_reference
      )
]

phase_b_subgroup_delta[
  ,
  delta_abs_calibration_slope_error :=
    abs(
      calibration_slope -
        1
    ) -
      abs(
        calibration_slope_reference -
          1
      )
]

fwrite(
  phase_b_subgroup_delta,
  file.path(
    out_dir,
    "10_SYSTEM_MEDIATED_SUBGROUP_DELTAS_VS_PRAGMATIC.csv"
  )
)

phase_b_subgroup_summary <- phase_b_subgroup_delta[
  variant_id !=
    "PRAGMATIC_CLINICAL",
  .(
    n_subgroup_comparisons =
      .N,
    median_delta_AUROC =
      median(
        delta_AUROC,
        na.rm = TRUE
      ),
    median_delta_AUPRC =
      median(
        delta_AUPRC,
        na.rm = TRUE
      ),
    median_delta_Brier =
      median(
        delta_Brier,
        na.rm = TRUE
      ),
    median_delta_abs_calibration_intercept =
      median(
        delta_abs_calibration_intercept,
        na.rm = TRUE
      ),
    worst_delta_abs_calibration_intercept =
      max(
        delta_abs_calibration_intercept,
        na.rm = TRUE
      ),
    median_delta_abs_calibration_slope_error =
      median(
        delta_abs_calibration_slope_error,
        na.rm = TRUE
      ),
    worst_delta_abs_calibration_slope_error =
      max(
        delta_abs_calibration_slope_error,
        na.rm = TRUE
      )
  ),
  by = .(
    endpoint_id,
    endpoint_label,
    variant_id,
    subgroup_domain
  )
]

fwrite(
  phase_b_subgroup_summary,
  file.path(
    out_dir,
    "11_SYSTEM_MEDIATED_SUBGROUP_TEMPORAL_SUMMARY.csv"
  )
)

# -----------------------------------------------------------------------------
# Human-readable completion note
# -----------------------------------------------------------------------------

summary_lines <- c(
  "TBI-TRACT PRAGMATIC FINALIST RETUNE COMPLETE",
  "",
  "Phase A tested CURRENT_REFERENCE vs PRAGMATIC_CLINICAL for all classification endpoints.",
  "",
  "PRAGMATIC_CLINICAL removes only:",
  "  helmet_use_recovered",
  "  respiratoryassistance_clean",
  "",
  "It retains:",
  "  gcsq_intubated_recovered",
  "  gcsq_sedated_paralyzed_recovered",
  "  supplemental_oxygen_recovered",
  "",
  "Phase B tested only disposition and HLOS trajectory:",
  "  PRAGMATIC_CLINICAL",
  "  PRAGMATIC_PLUS_PAYER",
  "  PRAGMATIC_PLUS_RACE_PAYER",
  "  PRAGMATIC_PLUS_RACE_ETHNICITY_PAYER",
  "",
  "Every candidate was fully retuned over the original prespecified structural grid in each temporal fold.",
  "",
  "Review first:",
  "  07_PRAGMATIC_CLINICAL_VS_CURRENT_TEMPORAL_SUMMARY.csv",
  "  09_SYSTEM_MEDIATED_TEMPORAL_SUMMARY.csv",
  "  10_SYSTEM_MEDIATED_SUBGROUP_DELTAS_VS_PRAGMATIC.csv",
  "  11_SYSTEM_MEDIATED_SUBGROUP_TEMPORAL_SUMMARY.csv",
  "",
  "Do not select predictors using one metric or a universal AUROC cutoff.",
  "Final adjudication should integrate temporal predictive performance, calibration,",
  "rare-outcome precision-recall performance, subgroup behavior, prediction-time",
  "availability, measurement burden, clinical interpretability, and fairness."
)

writeLines(
  summary_lines,
  file.path(
    out_dir,
    "PRAGMATIC_FINALIST_SUMMARY.txt"
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
  phase_a_summary[
    order(
      endpoint_label
    )
  ]
)

print(
  phase_b_summary[
    order(
      endpoint_label,
      variant_id
    )
  ]
)
