# =============================================================================
# 26b_extend_TBI_TRACT_hospital_LOS_deployment_CV_GPU.R
#
# PURPOSE
#   Targeted follow-up for the final Hospital LOS quantile deployment model.
#
# WHY
#   In script 26, the 5-fold full-development CV for hospital_los reached the
#   prespecified MAX_ROUNDS = 2500 ceiling. The minimum test quantile loss was
#   at/near the final round, so early stopping did not establish the plateau.
#
#   This script extends ONLY hospital_los CV to 5000 rounds with 150-round
#   patience, using the already locked temporal-consensus structural config.
#   It then overwrites ONLY the hospital_los deployment JSON/encoder and updates
#   the hospital_los rows in the deployment rounds/manifest CSVs.
#
#   Manuscript-facing temporal performance is NOT changed.
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

cfg_candidates <- c(
  file.path(getwd(), "R", "00_config.R"),
  "R/00_config.R"
)
cfg_file <- cfg_candidates[file.exists(cfg_candidates)][1L]
if (length(cfg_file) == 0L || is.na(cfg_file)) {
  stop("Could not find R/00_config.R.", call. = FALSE)
}
source(cfg_file)

methods_dir <- file.path(output_dir, "METHODS_COMPLETION_TBI_TRACT")
deploy_dir <- file.path(output_dir, "TBI_TRACT_FINAL_DEPLOYMENT_CV")

dataset_file <- file.path(
  methods_dir,
  "frozen_methods_dataset_retained_2020_2024.rds"
)
types_file <- file.path(
  methods_dir,
  "11_FROZEN_PREDICTOR_TYPES.csv"
)
struct_file <- file.path(
  deploy_dir,
  "02_TEMPORAL_CV_SELECTED_STRUCTURAL_CONFIG.csv"
)
rounds_file <- file.path(
  deploy_dir,
  "03_FULL_DEVELOPMENT_5FOLD_CV_ROUNDS.csv"
)
manifest_file <- file.path(
  deploy_dir,
  "04_FINAL_DEPLOYMENT_MODEL_MANIFEST.csv"
)

needed <- c(
  dataset_file,
  types_file,
  struct_file,
  rounds_file,
  manifest_file
)

if (!all(file.exists(needed))) {
  stop(
    "Missing prerequisite file(s):\n",
    paste(needed[!file.exists(needed)], collapse = "\n"),
    call. = FALSE
  )
}

# -----------------------------------------------------------------------------
# GPU
# -----------------------------------------------------------------------------

cores <- parallel::detectCores(logical = TRUE)
if (is.na(cores) || cores < 2L) {
  cores <- 32L
}

threads <- max(
  1L,
  min(
    cores - 1L,
    floor(0.94 * cores)
  )
)

setDTthreads(threads)

Sys.setenv(
  OMP_NUM_THREADS = as.character(threads),
  MKL_NUM_THREADS = as.character(threads),
  OPENBLAS_NUM_THREADS = as.character(threads)
)

reticulate::use_condaenv(
  "tbi-tract-xgb-gpu",
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

make_balanced_folds <- function(
    year,
    strata,
    k = 5L,
    seed = 20260913L
) {
  set.seed(seed)

  d <- data.table(
    row_id = seq_along(year),
    year = as.character(year),
    strata = as.character(strata)
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

to_python_folds <- function(
    fold_id,
    k = 5L
) {
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

# -----------------------------------------------------------------------------
# Load locked data/predictors/config
# -----------------------------------------------------------------------------

dt <- as.data.table(
  readRDS(dataset_file)
)
types <- fread(types_file)
struct <- fread(struct_file)
rounds_table <- fread(rounds_file)
manifest <- fread(manifest_file)

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
  stop("Malformed pragmatic predictor set.", call. = FALSE)
}

hp <- struct[
  endpoint_id == "hospital_los"
]

if (nrow(hp) != 1L) {
  stop("Could not uniquely recover locked hospital_los structure.", call. = FALSE)
}

dt[
  ,
  hospital_days :=
    safe_num(
      hospital_days
    )
]

keep <- !is.na(dt$hospital_days) &
  dt$hospital_days >= 0

numeric_keep <- intersect(
  numeric_predictors,
  pragmatic_clinical
)

categorical_keep <- intersect(
  categorical_predictors,
  pragmatic_clinical
)

encoder <- fit_encoder(
  dt,
  numeric_keep,
  categorical_keep
)

X <- encode_dense(
  dt,
  encoder
)

X_use <- X[
  keep,
  ,
  drop = FALSE
]

y_raw <- dt$hospital_days[
  keep
]

y <- log1p(
  y_raw
)

year <- dt$admission_year[
  keep
]

# Same year/outcome-balanced fold construction used by script 26.
strat_dt <- data.table(
  year = year,
  y = y_raw
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
            5 * rr
          )
        )
      )
    )
  },
  by = year
]

fold_id <- make_balanced_folds(
  year = year,
  strata = strat_dt$outcome_stratum,
  k = 5L,
  seed = 20260913L
)

folds_py <- to_python_folds(
  fold_id,
  k = 5L
)

params <- reticulate::dict(
  objective = "reg:quantileerror",
  quantile_alpha = c(
    0.10,
    0.50,
    0.90
  ),
  eta = 0.05,
  max_depth = as.integer(hp$max_depth),
  min_child_weight = as.numeric(hp$min_child_weight),
  subsample = as.numeric(hp$subsample),
  colsample_bytree = as.numeric(hp$colsample_bytree),
  lambda = as.numeric(hp$lambda),
  tree_method = "hist",
  device = "cuda",
  nthread = as.integer(threads),
  seed = 20260913L
)

d_all <- xgb$DMatrix(
  X_use,
  label = y
)

cat(
  "\nExtending Hospital LOS deployment CV to 5000 rounds...\n"
)

cv <- xgb$cv(
  params = params,
  dtrain = d_all,
  num_boost_round = 5000L,
  folds = folds_py,
  early_stopping_rounds = 150L,
  maximize = FALSE,
  verbose_eval = FALSE,
  shuffle = FALSE,
  seed = 20260913L
)

cv_log <- as.data.table(
  reticulate::py_to_r(cv)
)

if (nrow(cv_log) >= 5000L) {
  warning(
    "Hospital LOS CV again reached the 5000-round ceiling. Review before app deployment."
  )
}

selected_rounds <- nrow(cv_log)

fwrite(
  cv_log,
  file.path(
    deploy_dir,
    "cv_log__hospital_los_extended.csv"
  )
)

final_fit <- xgb$train(
  params = params,
  dtrain = d_all,
  num_boost_round = as.integer(selected_rounds),
  verbose_eval = FALSE
)

model_path <- file.path(
  deploy_dir,
  "models",
  "hospital_los.json"
)

encoder_path <- file.path(
  deploy_dir,
  "encoders",
  "hospital_los_encoder.rds"
)

final_fit$save_model(
  model_path
)

saveRDS(
  list(
    endpoint_id = "hospital_los",
    endpoint_label = "Hospital LOS",
    type = "quantile",
    quantile_alpha = c(
      0.10,
      0.50,
      0.90
    ),
    lower_bound = 0,
    predictors = pragmatic_clinical,
    encoder = encoder,
    feature_names = colnames(X_use),
    structural_config_id = hp$config_id,
    selected_rounds = selected_rounds,
    training_years = 2020:2024,
    N = nrow(X_use),
    deployment_cv_extension = TRUE,
    deployment_cv_max_rounds = 5000L,
    deployment_cv_patience = 150L
  ),
  encoder_path
)

rounds_table[
  endpoint_id == "hospital_los",
  selected_rounds_5fold_cv :=
    selected_rounds
]

manifest[
  endpoint_id == "hospital_los",
  `:=`(
    selected_rounds =
      selected_rounds,
    model_path =
      model_path,
    encoder_path =
      encoder_path
  )
]

fwrite(
  rounds_table,
  rounds_file
)

fwrite(
  manifest,
  manifest_file
)

writeLines(
  c(
    "TBI-TRACT Hospital LOS deployment CV extension complete.",
    paste0("Selected rounds: ", selected_rounds),
    paste0("Structural configuration: ", hp$config_id),
    "Maximum rounds evaluated: 5000",
    "Early-stopping patience: 150",
    "",
    "Only the hospital_los deployment object and corresponding manifest/rounds rows were updated.",
    "Manuscript-facing temporal performance was not changed."
  ),
  file.path(
    deploy_dir,
    "HOSPITAL_LOS_DEPLOYMENT_CV_EXTENSION_SUMMARY.txt"
  )
)

cat(
  "\nHospital LOS deployment CV extension complete.\n",
  "Selected rounds: ",
  selected_rounds,
  "\n",
  sep = ""
)
