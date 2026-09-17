# =============================================================================
# 17_run_FINAL_TBI_TRACT_hybrid_COVID_era_sensitivity_GPU_v4.R
#
# FINAL TBI-TRACT COVID-ERA DEVELOPMENT SENSITIVITY
#
# Sensitivity temporal architecture:
#   2022      -> early-stopping training
#   2023      -> boosting-round selection
#   2022-2023 -> sensitivity final refit
#   2024      -> same temporal validation cohort
#
# Structural hyperparameters are FIXED from the final primary model.
# 2024 is not used for model selection.
#
# Repeats the FINAL endpoint architecture:
#   - disposition 3-class
#   - ICU: none / 1-7 / >=8
#   - ventilation: none / 1-7 / >=8
#   - HLOS: <=7 / 8-27 / >=28
#   - ICP monitoring
#   - craniotomy/craniectomy
#   - Q10/Q50/Q90 HLOS / conditional ICU LOS / conditional ventilator duration
#
# Run AFTER scripts 14 and 15.
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

hybrid_dir <- file.path(
  output_dir,
  "FINAL_TBI_TRACT_HYBRID_RESOURCE_MODEL_2024"
)

dataset_file <- file.path(
  methods_dir,
  "frozen_methods_dataset_retained_2020_2024.rds"
)

types_file <- file.path(
  methods_dir,
  "11_FROZEN_PREDICTOR_TYPES.csv"
)

selected_hp_file <- file.path(
  tuning_dir,
  "03_SELECTED_HYPERPARAMETERS.csv"
)

vent_hp_file <- file.path(
  vent_dir,
  "06_SELECTED_VENT_TRAJECTORY_HYPERPARAMETERS.csv"
)

hlos_hp_file <- file.path(
  hybrid_dir,
  "04_SELECTED_HLOS_TRAJECTORY_HYPERPARAMETERS.csv"
)

duration_hp_file <- file.path(
  hybrid_dir,
  "02_DURATION_STRUCTURAL_HYPERPARAMETERS.csv"
)

needed <- c(
  dataset_file,
  types_file,
  selected_hp_file,
  vent_hp_file,
  hlos_hp_file,
  duration_hp_file
)

if (!all(file.exists(needed))) {
  stop(
    "Run the final hybrid model (script 14) before this sensitivity.",
    call. = FALSE
  )
}

out_dir <- file.path(
  output_dir,
  "SENSITIVITY_FINAL_TBI_TRACT_2022_2023_to_2024"
)

pred_dir <- file.path(
  out_dir,
  "predictions"
)

model_dir <- file.path(
  out_dir,
  "models"
)

for (d in c(
  out_dir,
  pred_dir,
  model_dir
)) {
  dir.create(
    d,
    recursive = TRUE,
    showWarnings = FALSE
  )
}

# -----------------------------------------------------------------------------
# Compute
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
      0.94 *
        logical_cores
    )
  )
)

setDTthreads(
  cpu_threads
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

binary_metrics <- function(
    y,
    p,
    target
) {
  p <- clamp_prob(
    p
  )

  data.table(
    target =
      target,
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
    log_loss =
      -mean(
        y *
          log(p) +
          (
            1 - y
          ) *
            log(
              1 - p
            )
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
    ncol =
      n_numeric +
        n_cat
  )

  j <- 1L

  for (
    v in encoder$numeric_vars
  ) {
    X[, j] <-
      safe_num(
        d[[v]]
      )

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

    j <- max(cols) +
      1L
  }

  storage.mode(X) <-
    "double"

  X
}

best_iteration <- function(
    fit,
    fallback
) {
  out <- tryCatch(
    as.numeric(
      reticulate::py_to_r(
        fit$best_iteration
      )
    ),
    error = function(e) NA_real_
  )

  if (
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

pred_matrix <- function(
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

  p
}

rearrange_q <- function(p) {
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

# -----------------------------------------------------------------------------
# Data / endpoints
# -----------------------------------------------------------------------------

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

dt[
  ,
  vent_days :=
    safe_num(
      vent_days
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
  hospital_days :=
    safe_num(
      hospital_days
    )
]

dt[
  ,
  ventilation_trajectory_final :=
    fcase(
      is.na(
        vent_days
      ),
      NA_character_,
      vent_days <= 0,
      "No ventilation",
      vent_days <= 7,
      "Ventilation 1-7 days",
      vent_days >= 8,
      "Ventilation >=8 days",
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
      hospital_days <= 7,
      "Hospital LOS <=7 days",
      hospital_days <= 27,
      "Hospital LOS 8-27 days",
      hospital_days >= 28,
      "Hospital LOS >=28 days",
      default =
        NA_character_
    )
]

selected_hp <- fread(
  selected_hp_file
)

vent_hp <- fread(
  vent_hp_file
)[1L]

hlos_hp <- fread(
  hlos_hp_file
)[1L]

get_hp <- function(endpoint_name) {
  z <- selected_hp[
    selected_hp$endpoint ==
      endpoint_name
  ]

  if (
    nrow(z) !=
      1L
  ) {
    stop(
      "Could not recover hyperparameters for ",
      endpoint_name,
      call. = FALSE
    )
  }

  z[1L]
}

endpoint_specs <- list(
  list(
    id =
      "discharge_3cat_final",
    label =
      "Disposition",
    type =
      "multiclass",
    levels =
      c(
        "Home/home health",
        "Post-acute facility",
        "Death/hospice"
      ),
    hp =
      get_hp(
        "discharge_3cat_final"
      )
  ),
  list(
    id =
      "icu_trajectory_final",
    label =
      "ICU trajectory",
    type =
      "multiclass",
    levels =
      c(
        "No ICU",
        "ICU 1-7 days",
        "ICU >=8 days"
      ),
    hp =
      get_hp(
        "icu_trajectory_final"
      )
  ),
  list(
    id =
      "ventilation_trajectory_final",
    label =
      "Mechanical ventilation trajectory",
    type =
      "multiclass",
    levels =
      c(
        "No ventilation",
        "Ventilation 1-7 days",
        "Ventilation >=8 days"
      ),
    hp =
      vent_hp
  ),
  list(
    id =
      "hlos_trajectory_final",
    label =
      "Hospital LOS trajectory",
    type =
      "multiclass",
    levels =
      c(
        "Hospital LOS <=7 days",
        "Hospital LOS 8-27 days",
        "Hospital LOS >=28 days"
      ),
    hp =
      hlos_hp
  ),
  list(
    id =
      "icp_pressure_monitor_final",
    label =
      "Invasive ICP monitoring",
    type =
      "binary",
    hp =
      get_hp(
        "icp_pressure_monitor_final"
      )
  ),
  list(
    id =
      "craniotomy_craniectomy_final",
    label =
      "Craniotomy/craniectomy",
    type =
      "binary",
    hp =
      get_hp(
        "craniotomy_craniectomy_final"
      )
  )
)

# -----------------------------------------------------------------------------
# Sensitivity temporal matrices
# -----------------------------------------------------------------------------

idx_train <- which(
  dt$admission_year ==
    2022L
)

idx_tune <- which(
  dt$admission_year ==
    2023L
)

idx_dev <- which(
  dt$admission_year %in%
    2022:2023
)

idx_test <- which(
  dt$admission_year ==
    2024L
)

pp_tune <- fit_encoder(
  dt[
    idx_train
  ],
  numeric_predictors,
  categorical_predictors
)

X_tmp <- encode_dense(
  dt[
    c(
      idx_train,
      idx_tune
    )
  ],
  pp_tune
)

X_train <- X_tmp[
  seq_len(
    length(
      idx_train
    )
  ),
  ,
  drop = FALSE
]

X_tune <- X_tmp[
  length(
    idx_train
  ) +
    seq_len(
      length(
        idx_tune
      )
    ),
  ,
  drop = FALSE
]

rm(
  X_tmp
)
gc()

pp_final <- fit_encoder(
  dt[
    idx_dev
  ],
  numeric_predictors,
  categorical_predictors
)

X_tmp <- encode_dense(
  dt[
    c(
      idx_dev,
      idx_test
    )
  ],
  pp_final
)

X_dev <- X_tmp[
  seq_len(
    length(
      idx_dev
    )
  ),
  ,
  drop = FALSE
]

X_test <- X_tmp[
  length(
    idx_dev
  ) +
    seq_len(
      length(
        idx_test
      )
    ),
  ,
  drop = FALSE
]

rm(
  X_tmp
)
gc()

ETA <- 0.05
MAX_ROUNDS <- 3000L
EARLY_STOP <- 75L

class_results <- list()

# -----------------------------------------------------------------------------
# Classification sensitivity
# -----------------------------------------------------------------------------

for (
  spec in endpoint_specs
) {
  cat(
    "\nSensitivity model: ",
    spec$label,
    "\n",
    sep = ""
  )

  hp <- spec$hp[1L]

  if (
    spec$type ==
      "binary"
  ) {
    y_all <- as.integer(
      dt[[spec$id]]
    )

    tr_keep <- !is.na(
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

    dtrain <- xgb$DMatrix(
      X_train[
        tr_keep,
        ,
        drop = FALSE
      ],
      label =
        y_all[
          idx_train
        ][
          tr_keep
        ]
    )

    dtune <- xgb$DMatrix(
      X_tune[
        tune_keep,
        ,
        drop = FALSE
      ],
      label =
        y_all[
          idx_tune
        ][
          tune_keep
        ]
    )

    params <- reticulate::dict(
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
        20260908L
    )

    tune_fit <- xgb$train(
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
          "train_2022"
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
      maximize =
        FALSE,
      verbose_eval =
        FALSE
    )

    rounds <- best_iteration(
      tune_fit,
      MAX_ROUNDS
    )

    ddev <- xgb$DMatrix(
      X_dev[
        dev_keep,
        ,
        drop = FALSE
      ],
      label =
        y_all[
          idx_dev
        ][
          dev_keep
        ]
    )

    dtest <- xgb$DMatrix(
      X_test[
        test_keep,
        ,
        drop = FALSE
      ]
    )

    final_fit <- xgb$train(
      params =
        params,
      dtrain =
        ddev,
      num_boost_round =
        as.integer(
          rounds
        ),
      verbose_eval =
        FALSE
    )

    p <- as.numeric(
      final_fit$predict(
        dtest
      )
    )

    y <- y_all[
      idx_test
    ][
      test_keep
    ]

    met <- binary_metrics(
      y,
      p,
      spec$label
    )

    met[
      ,
      `:=`(
        endpoint =
          spec$id,
        selected_rounds =
          rounds
      )
    ]

    class_results[[
      length(
        class_results
      ) +
        1L
    ]] <- met
  } else {
    yf <- factor(
      as.character(
        dt[[spec$id]]
      ),
      levels =
        spec$levels
    )

    tr_keep <- !is.na(
      yf[
        idx_train
      ]
    )

    tune_keep <- !is.na(
      yf[
        idx_tune
      ]
    )

    dev_keep <- !is.na(
      yf[
        idx_dev
      ]
    )

    test_keep <- !is.na(
      yf[
        idx_test
      ]
    )

    ytr <- as.integer(
      yf[
        idx_train
      ][
        tr_keep
      ]
    ) -
      1L

    ytu <- as.integer(
      yf[
        idx_tune
      ][
        tune_keep
      ]
    ) -
      1L

    dtrain <- xgb$DMatrix(
      X_train[
        tr_keep,
        ,
        drop = FALSE
      ],
      label =
        ytr
    )

    dtune <- xgb$DMatrix(
      X_tune[
        tune_keep,
        ,
        drop = FALSE
      ],
      label =
        ytu
    )

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

    tune_fit <- xgb$train(
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
          "train_2022"
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
      maximize =
        FALSE,
      verbose_eval =
        FALSE
    )

    rounds <- best_iteration(
      tune_fit,
      MAX_ROUNDS
    )

    ddev <- xgb$DMatrix(
      X_dev[
        dev_keep,
        ,
        drop = FALSE
      ],
      label =
        as.integer(
          yf[
            idx_dev
          ][
            dev_keep
          ]
        ) -
          1L
    )

    dtest <- xgb$DMatrix(
      X_test[
        test_keep,
        ,
        drop = FALSE
      ]
    )

    final_fit <- xgb$train(
      params =
        params,
      dtrain =
        ddev,
      num_boost_round =
        as.integer(
          rounds
        ),
      verbose_eval =
        FALSE
    )

    prob <- pred_matrix(
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

    ytest <- as.character(
      yf[
        idx_test
      ][
        test_keep
      ]
    )

    for (
      k in seq_along(
        spec$levels
      )
    ) {
      lev <- spec$levels[k]

      met <- binary_metrics(
        as.integer(
          ytest ==
            lev
        ),
        prob[, k],
        paste0(
          spec$label,
          ": ",
          lev
        )
      )

      met[
        ,
        `:=`(
          endpoint =
            spec$id,
          selected_rounds =
            rounds
        )
      ]

      class_results[[
        length(
          class_results
        ) +
          1L
      ]] <- met
    }

    if (
      spec$id ==
        "icu_trajectory_final"
    ) {
      met <- binary_metrics(
        as.integer(
          ytest !=
            "No ICU"
        ),
        1 -
          prob[
            ,
            "No ICU"
          ],
        "Any ICU use"
      )

      met[
        ,
        `:=`(
          endpoint =
            spec$id,
          selected_rounds =
            rounds
        )
      ]

      class_results[[
        length(
          class_results
        ) +
          1L
      ]] <- met
    }

    if (
      spec$id ==
        "ventilation_trajectory_final"
    ) {
      met <- binary_metrics(
        as.integer(
          ytest !=
            "No ventilation"
        ),
        1 -
          prob[
            ,
            "No ventilation"
          ],
        "Any ventilation"
      )

      met[
        ,
        `:=`(
          endpoint =
            spec$id,
          selected_rounds =
            rounds
        )
      ]

      class_results[[
        length(
          class_results
        ) +
          1L
      ]] <- met
    }
  }

  rm(
    dtrain,
    dtune,
    tune_fit,
    ddev,
    dtest,
    final_fit
  )

  gc()
}

class_results <- rbindlist(
  class_results,
  fill = TRUE
)

fwrite(
  class_results,
  file.path(
    out_dir,
    "01_COVID_SENSITIVITY_2024_CLASSIFICATION_PERFORMANCE.csv"
  )
)

# -----------------------------------------------------------------------------
# Duration sensitivity
# -----------------------------------------------------------------------------

duration_hp <- fread(
  duration_hp_file
)

duration_specs <- list(
  list(
    id =
      "hospital_los",
    label =
      "Hospital LOS",
    hp_key =
      "Hospital LOS",
    y_var =
      "hospital_days",
    lower_bound =
      0,
    filter =
      function(d) {
        !is.na(
          d$hospital_days
        ) &
          d$hospital_days >= 0
      }
  ),
  list(
    id =
      "icu_los_conditional",
    label =
      "ICU LOS conditional on ICU use",
    hp_key =
      "ICU LOS conditional on ICU",
    y_var =
      "icu_days",
    lower_bound =
      1,
    filter =
      function(d) {
        !is.na(
          d$icu_days
        ) &
          d$icu_days > 0
      }
  ),
  list(
    id =
      "ventilator_days_conditional",
    label =
      "Ventilator duration conditional on ventilation",
    hp_key =
      "Ventilator duration conditional on ventilation",
    y_var =
      "vent_days",
    lower_bound =
      1,
    filter =
      function(d) {
        !is.na(
          d$vent_days
        ) &
          d$vent_days > 0
      }
  )
)

duration_results <- list()
ALPHAS <- c(
  0.10,
  0.50,
  0.90
)

for (
  spec in duration_specs
) {
  hp <- duration_hp[
    duration_model ==
      spec$hp_key
  ]

  if (
    nrow(hp) != 1L
  ) {
    stop(
      "Could not uniquely recover duration hyperparameters for '",
      spec$label,
      "' using key '",
      spec$hp_key,
      "'. Available duration_model values are: ",
      paste(
        unique(
          duration_hp$duration_model
        ),
        collapse = " | "
      ),
      call. = FALSE
    )
  }

  hp <- hp[1L]

  required_hp <- c(
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "lambda"
  )

  missing_hp <- required_hp[
    !required_hp %in%
      names(hp) |
      vapply(
        required_hp,
        function(v) {
          length(hp[[v]]) == 0L ||
            is.na(hp[[v]][1L]) ||
            !is.finite(
              suppressWarnings(
                as.numeric(
                  hp[[v]][1L]
                )
              )
            )
        },
        logical(1)
      )
  ]

  if (
    length(missing_hp) > 0L
  ) {
    stop(
      "Invalid/missing duration hyperparameters for ",
      spec$label,
      ": ",
      paste(
        missing_hp,
        collapse = ", "
      ),
      call. = FALSE
    )
  }

  cat(
    "  Hyperparameters: depth=",
    hp$max_depth,
    ", min_child_weight=",
    hp$min_child_weight,
    ", subsample=",
    hp$subsample,
    ", colsample=",
    hp$colsample_bytree,
    ", lambda=",
    hp$lambda,
    "\n",
    sep = ""
  )

  train_keep <- spec$filter(
    dt[
      idx_train
    ]
  )

  tune_keep <- spec$filter(
    dt[
      idx_tune
    ]
  )

  dev_keep <- spec$filter(
    dt[
      idx_dev
    ]
  )

  test_keep <- spec$filter(
    dt[
      idx_test
    ]
  )

  ytr_raw <- dt[[spec$y_var]][
    idx_train
  ][
    train_keep
  ]

  ytu_raw <- dt[[spec$y_var]][
    idx_tune
  ][
    tune_keep
  ]

  ydev_raw <- dt[[spec$y_var]][
    idx_dev
  ][
    dev_keep
  ]

  ytest <- dt[[spec$y_var]][
    idx_test
  ][
    test_keep
  ]

  params <- reticulate::dict(
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

  dtrain <- xgb$DMatrix(
    X_train[
      train_keep,
      ,
      drop = FALSE
    ],
    label =
      log1p(
        ytr_raw
      )
  )

  dtune <- xgb$DMatrix(
    X_tune[
      tune_keep,
      ,
      drop = FALSE
    ],
    label =
      log1p(
        ytu_raw
      )
  )

  tune_fit <- xgb$train(
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
        "train_2022"
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
    maximize =
      FALSE,
    verbose_eval =
      FALSE
  )

  rounds <- best_iteration(
    tune_fit,
    MAX_ROUNDS
  )

  ddev <- xgb$DMatrix(
    X_dev[
      dev_keep,
      ,
      drop = FALSE
    ],
    label =
      log1p(
        ydev_raw
      )
  )

  dtest <- xgb$DMatrix(
    X_test[
      test_keep,
      ,
      drop = FALSE
    ]
  )

  final_fit <- xgb$train(
    params =
      params,
    dtrain =
      ddev,
    num_boost_round =
      as.integer(
        rounds
      ),
    verbose_eval =
      FALSE
  )

  pred <- pred_matrix(
    final_fit$predict(
      dtest
    ),
    sum(
      test_keep
    ),
    3L
  )

  pred <- expm1(
    pred
  )

  pred <- pmax(
    pred,
    spec$lower_bound
  )

  pred <- rearrange_q(
    pred
  )

  duration_results[[
    length(
      duration_results
    ) +
      1L
  ]] <- data.table(
    outcome =
      spec$label,
    N =
      length(
        ytest
      ),
    selected_rounds =
      rounds,
    median_MAE_days =
      mean(
        abs(
          ytest -
            pred[, 2L]
        )
      ),
    central_80_coverage =
      mean(
        ytest >=
          pred[, 1L] &
          ytest <=
            pred[, 3L]
      ),
    median_80PI_width_days =
      median(
        pred[, 3L] -
          pred[, 1L]
      ),
    mean_pinball =
      mean(
        c(
          pinball_loss(
            ytest,
            pred[, 1L],
            0.10
          ),
          pinball_loss(
            ytest,
            pred[, 2L],
            0.50
          ),
          pinball_loss(
            ytest,
            pred[, 3L],
            0.90
          )
        )
      )
  )

  rm(
    dtrain,
    dtune,
    tune_fit,
    ddev,
    dtest,
    final_fit,
    pred
  )

  gc()
}

duration_results <- rbindlist(
  duration_results,
  fill = TRUE
)

fwrite(
  duration_results,
  file.path(
    out_dir,
    "02_COVID_SENSITIVITY_2024_DURATION_PERFORMANCE.csv"
  )
)

# -----------------------------------------------------------------------------
# Primary vs sensitivity comparison
# -----------------------------------------------------------------------------

primary_perf_file <- file.path(
  hybrid_dir,
  "publication_validation",
  "01_OVERALL_2024_PERFORMANCE_WITH_95CI.csv"
)

primary_duration_file <- file.path(
  hybrid_dir,
  "publication_validation",
  "04_DURATION_MODEL_OVERALL_2024_VALIDATION.csv"
)

if (
  file.exists(
    primary_perf_file
  )
) {
  primary <- fread(
    primary_perf_file
  )

  sensitivity <- copy(
    class_results
  )

  sensitivity[
    ,
    target_clean :=
      sub(
        "^[^:]+: ",
        "",
        target
      )
  ]

  primary[
    ,
    target_clean :=
      target
  ]

  cmp <- merge(
    primary[
      ,
      .(
        target_clean,
        primary_AUROC =
          AUROC,
        primary_Brier =
          Brier
      )
    ],
    sensitivity[
      ,
      .(
        target_clean,
        sensitivity_AUROC =
          AUROC,
        sensitivity_Brier =
          Brier
      )
    ],
    by =
      "target_clean"
  )

  cmp[
    ,
    `:=`(
      delta_AUROC_sensitivity_minus_primary =
        sensitivity_AUROC -
          primary_AUROC,
      delta_Brier_sensitivity_minus_primary =
        sensitivity_Brier -
          primary_Brier
    )
  ]

  fwrite(
    cmp,
    file.path(
      out_dir,
      "03_PRIMARY_VS_COVID_SENSITIVITY_CLASSIFICATION.csv"
    )
  )
}

if (
  file.exists(
    primary_duration_file
  )
) {
  pqd <- fread(
    primary_duration_file
  )

  qcmp <- merge(
    pqd[
      ,
      .(
        outcome =
          domain,
        primary_median_MAE_days =
          median_MAE_days,
        primary_central_80_coverage =
          central_80_coverage
      )
    ],
    duration_results[
      ,
      .(
        outcome,
        sensitivity_median_MAE_days =
          median_MAE_days,
        sensitivity_central_80_coverage =
          central_80_coverage
      )
    ],
    by =
      "outcome"
  )

  qcmp[
    ,
    `:=`(
      delta_MAE_days =
        sensitivity_median_MAE_days -
          primary_median_MAE_days,
      delta_80PI_coverage =
        sensitivity_central_80_coverage -
          primary_central_80_coverage
    )
  ]

  fwrite(
    qcmp,
    file.path(
      out_dir,
      "04_PRIMARY_VS_COVID_SENSITIVITY_DURATION.csv"
    )
  )
}

summary_lines <- c(
  "FINAL TBI-TRACT COVID-ERA SENSITIVITY COMPLETE",
  "",
  "Sensitivity development:",
  "  2022 training",
  "  2023 boosting-round selection",
  "  2022-23 refit",
  "  2024 temporal validation",
  "",
  "Structural hyperparameters were fixed from the final primary model.",
  "The sensitivity repeats both categorical and duration components."
)

writeLines(
  summary_lines,
  file.path(
    out_dir,
    "COVID_SENSITIVITY_SUMMARY.txt"
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
  class_results
)

print(
  duration_results
)
