# =============================================================================
# 25_run_TBI_TRACT_final_sensitivity_suite_GPU.R
#
# Remaining manuscript-facing sensitivities after predictor adjudication.
#
# New analyses:
#   1) Final prediction cache for 2022/2023/2024 forward temporal folds.
#   2) Final subgroup robustness: GCS severity, age, sex, input missingness.
#   3) Ventilation timing/leakage sensitivity by baseline airway state.
#   4) Survivor-only HLOS/ICU/vent trajectory sensitivity.
#   5) Survivor-only continuous-duration sensitivity.
#   6) Fully retuned composite sensitivities:
#        death/hospice OR HLOS >=28 d
#        death/hospice OR ICU LOS >=8 d
#        death/hospice OR ventilation >=8 d
#
# Already completed elsewhere and intentionally not re-run:
#   - transfer-out/AMA/custody selection SMD audit
#   - development-vs-2024 case-mix SMD audit
#   - development-only threshold audit
#   - COVID-era sensitivity
#   - penalized-regression comparator
#   - predictor-family ablation
#   - race/ethnicity/payer augmentation/fairness analysis
#   - final pragmatic predictor selection
#
# Facility heterogeneity is not evaluated because facility keys are unavailable.
#
# RUN AFTER:
#   23_run_TBI_TRACT_pragmatic_finalist_retune_GPU.R
#   24_run_TBI_TRACT_pragmatic_duration_finalization_GPU.R
# =============================================================================

rm(list = ls()); gc()

req <- c("data.table","reticulate")
miss <- req[!vapply(req, requireNamespace, logical(1), quietly=TRUE)]
if(length(miss)) stop("Missing package(s): ", paste(miss, collapse=", "))

suppressPackageStartupMessages({
  library(data.table)
  library(reticulate)
})

cfg <- c(file.path(getwd(),"R","00_config.R"),"R/00_config.R")
cfg <- cfg[file.exists(cfg)][1]
if(is.na(cfg)) stop("Could not find R/00_config.R")
source(cfg)

methods_dir <- file.path(output_dir,"METHODS_COMPLETION_TBI_TRACT")
class_dir <- file.path(output_dir,"TBI_TRACT_PRAGMATIC_FINALIST_RETUNE")
duration_dir <- file.path(output_dir,"TBI_TRACT_PRAGMATIC_DURATION_FINALIZATION")
tuning_dir <- file.path(output_dir,"TBI_TRACT_TEMPORAL_HYPERPARAMETER_TUNING")

dataset_file <- file.path(methods_dir,"frozen_methods_dataset_retained_2020_2024.rds")
types_file <- file.path(methods_dir,"11_FROZEN_PREDICTOR_TYPES.csv")
class_hp_file <- file.path(class_dir,"02_SELECTED_PRAGMATIC_HYPERPARAMETERS.csv")
duration_hp_file <- file.path(duration_dir,"02_SELECTED_DURATION_HYPERPARAMETERS.csv")
grid_file <- file.path(tuning_dir,"01_PRESPECIFIED_SEARCH_GRID.csv")

needed <- c(dataset_file,types_file,class_hp_file,duration_hp_file,grid_file)
if(!all(file.exists(needed))) {
  stop("Missing prerequisite file(s):\n",paste(needed[!file.exists(needed)],collapse="\n"))
}

out_dir <- file.path(output_dir,"TBI_TRACT_FINAL_SENSITIVITY_SUITE")
pred_dir <- file.path(out_dir,"prediction_cache")
grid_out <- file.path(out_dir,"composite_grid_search")
dir.create(out_dir,recursive=TRUE,showWarnings=FALSE)
dir.create(pred_dir,recursive=TRUE,showWarnings=FALSE)
dir.create(grid_out,recursive=TRUE,showWarnings=FALSE)

# -----------------------------------------------------------------------------
# GPU
# -----------------------------------------------------------------------------

cores <- parallel::detectCores(logical=TRUE)
if(is.na(cores)||cores<2L) cores <- 32L
threads <- max(1L,min(cores-1L,floor(.94*cores)))
setDTthreads(threads)
Sys.setenv(OMP_NUM_THREADS=threads,MKL_NUM_THREADS=threads,OPENBLAS_NUM_THREADS=threads)

reticulate::use_condaenv("tbi-tract-xgb-gpu",required=TRUE)
xgb <- reticulate::import("xgboost",convert=TRUE)

set.seed(20260913L)
px <- matrix(rnorm(4000),500,8)
py <- as.integer(rbinom(500,1,.3))
pd <- xgb$DMatrix(px,label=py)
pf <- xgb$train(
  reticulate::dict(
    objective="binary:logistic",tree_method="hist",device="cuda",
    max_depth=2L,eta=.2,nthread=as.integer(threads),seed=20260913L
  ),
  pd,3L,verbose_eval=FALSE
)
cfg_raw <- pf$save_config()
cfg_txt <- tryCatch(as.character(reticulate::py_to_r(cfg_raw))[1L],
                    error=function(e) as.character(cfg_raw)[1L])
if(!grepl('"device"[[:space:]]*:[[:space:]]*"cuda',cfg_txt,ignore.case=TRUE))
  stop("CUDA was not verified.")
rm(px,py,pd,pf); gc()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

num <- function(x) suppressWarnings(as.numeric(as.character(x)))
norm_chr <- function(x) {
  x <- trimws(as.character(x))
  x[is.na(x)|x==""|x %in% c("NA","NaN","<NA>")] <- "__UNKNOWN__"
  x
}
clamp <- function(p,e=1e-7) pmin(pmax(as.numeric(p),e),1-e)

auc <- function(y,p) {
  k <- !is.na(y)&!is.na(p)&is.finite(p)
  y <- as.integer(y[k]); p <- as.numeric(p[k])
  k <- y %in% c(0L,1L); y <- y[k]; p <- p[k]
  n1 <- as.double(sum(y==1L)); n0 <- as.double(sum(y==0L))
  if(n1==0||n0==0) return(NA_real_)
  r <- rank(p,ties.method="average")
  (sum(r[y==1L])-n1*(n1+1)/2)/(n1*n0)
}
auprc <- function(y,p) {
  k <- !is.na(y)&!is.na(p)&is.finite(p)
  y <- as.integer(y[k]); p <- as.numeric(p[k])
  k <- y %in% c(0L,1L); y <- y[k]; p <- p[k]
  np <- sum(y==1L); if(!np) return(NA_real_)
  o <- order(p,decreasing=TRUE); yy <- y[o]
  tp <- cumsum(yy==1L); fp <- cumsum(yy==0L)
  pr <- tp/(tp+fp); rc <- tp/np; prev <- c(0,head(rc,-1L))
  sum(pr*(rc-prev),na.rm=TRUE)
}
calibration <- function(y,p) {
  k <- !is.na(y)&!is.na(p)&is.finite(p)
  y <- as.integer(y[k]); p <- clamp(p[k])
  if(length(unique(y))<2L) return(c(intercept=NA_real_,slope=NA_real_))
  lp <- qlogis(p)
  fi <- tryCatch(glm(y~1,family=binomial(),offset=lp),error=function(e)NULL)
  fs <- tryCatch(glm(y~lp,family=binomial()),error=function(e)NULL)
  c(intercept=if(is.null(fi)) NA_real_ else unname(coef(fi)[1]),
    slope=if(is.null(fs)) NA_real_ else unname(coef(fs)["lp"]))
}
bmetrics <- function(y,p) {
  c0 <- calibration(y,p); pp <- clamp(p)
  data.table(
    N=length(y),events=sum(y==1L),prevalence=mean(y==1L),
    AUROC=auc(y,p),AUPRC=auprc(y,p),Brier=mean((p-y)^2),
    log_loss=-mean(y*log(pp)+(1-y)*log(1-pp)),
    calibration_intercept=unname(c0["intercept"]),
    calibration_slope=unname(c0["slope"])
  )
}
pinball <- function(y,q,a) {
  e <- y-q
  mean(ifelse(e>=0,a*e,(a-1)*e),na.rm=TRUE)
}
dmetrics <- function(y,q10,q50,q90) {
  cov <- mean(y>=q10 & y<=q90)
  data.table(
    N=length(y),
    median_MAE_days=mean(abs(y-q50)),
    median_bias_days=median(q50-y),
    mean_pinball=mean(c(pinball(y,q10,.1),pinball(y,q50,.5),pinball(y,q90,.9))),
    central_80_coverage=cov,
    absolute_80_coverage_error=abs(cov-.8),
    median_80PI_width_days=median(q90-q10)
  )
}

fit_encoder <- function(d,nvars,cvars) {
  lev <- lapply(cvars,function(v) unique(c(sort(unique(norm_chr(d[[v]]))),
                                          "__UNKNOWN__","__OTHER__")))
  names(lev) <- cvars
  list(num=nvars,cat=cvars,lev=lev)
}
encode <- function(d,enc) {
  n <- nrow(d); nc <- length(enc$num)+sum(vapply(enc$lev,length,integer(1)))
  X <- matrix(0,n,nc); fn <- character(nc); par <- character(nc); j <- 1L
  for(v in enc$num) {
    X[,j] <- num(d[[v]]); fn[j] <- v; par[j] <- v; j <- j+1L
  }
  for(v in enc$cat) {
    lev <- enc$lev[[v]]; x <- norm_chr(d[[v]]); x[!x %in% lev] <- "__OTHER__"
    idx <- match(x,lev); cols <- j:(j+length(lev)-1L)
    X[cbind(seq_len(n),cols[idx])] <- 1
    fn[cols] <- paste0(v,"__",make.names(lev,unique=TRUE)); par[cols] <- v
    j <- max(cols)+1L
  }
  colnames(X) <- fn; attr(X,"parent_predictor") <- par
  storage.mode(X) <- "double"; X
}
mask <- function(X,keep) attr(X,"parent_predictor") %in% keep

best_iter <- function(fit,fallback=3000L) {
  z <- tryCatch(as.numeric(reticulate::py_to_r(fit$best_iteration)),
                error=function(e) NA_real_)
  if(length(z)&&is.finite(z[1])) as.integer(z[1]+1L) else as.integer(fallback)
}
best_score <- function(fit) {
  z <- tryCatch(as.numeric(reticulate::py_to_r(fit$best_score)),
                error=function(e) NA_real_)
  if(length(z)&&is.finite(z[1])) z[1] else NA_real_
}
pred_matrix <- function(pred,n,k) {
  p <- tryCatch(reticulate::py_to_r(pred),error=function(e)pred)
  if(is.matrix(p)||is.data.frame(p)) p <- as.matrix(p)
  else p <- matrix(as.numeric(p),nrow=n,ncol=k,byrow=TRUE)
  if(nrow(p)!=n||ncol(p)!=k) stop("Unexpected prediction shape.")
  storage.mode(p) <- "double"; p
}
rearrange_q <- function(p) {
  lo <- pmin(p[,1],p[,2],p[,3]); hi <- pmax(p[,1],p[,2],p[,3])
  mid <- rowSums(p)-lo-hi
  cbind(q10=lo,q50=mid,q90=hi)
}
bind0 <- function(x) if(length(x)) rbindlist(x,fill=TRUE) else data.table()

# -----------------------------------------------------------------------------
# Load data / final predictor policy
# -----------------------------------------------------------------------------

dt <- as.data.table(readRDS(dataset_file))
types <- fread(types_file)
class_hp <- fread(class_hp_file)
duration_hp <- fread(duration_hp_file)
grid <- fread(grid_file)

grid_cols <- c("config_id","max_depth","min_child_weight","subsample","colsample_bytree","lambda")
if(!all(grid_cols %in% names(grid))) stop("Malformed prespecified search grid.")
grid <- grid[,..grid_cols]

nvars <- types[category=="numeric",predictor]
cvars <- types[category=="categorical",predictor]
current <- c(nvars,cvars)

pragmatic <- setdiff(current,c("helmet_use_recovered","respiratoryassistance_clean"))
required_retained <- c("gcsq_intubated_recovered","gcsq_sedated_paralyzed_recovered",
                       "supplemental_oxygen_recovered")
if(!all(required_retained %in% pragmatic)) stop("Malformed pragmatic predictor set.")

social <- c("race_clean","ethnicity_clean","insurance_clean")
system_context <- unique(c(pragmatic,social))
all_cvars <- unique(c(cvars,social))

dt[,hospital_days:=num(hospital_days)]
dt[,icu_days:=num(icu_days)]
dt[,vent_days:=num(vent_days)]
dt[hospital_days<0,hospital_days:=NA_real_]
dt[icu_days<0,icu_days:=NA_real_]
dt[vent_days<0,vent_days:=NA_real_]

DIS <- c("Home/home health","Post-acute facility","Death/hospice")
ICU <- c("No ICU","ICU 1-7 days","ICU >=8 days")
VENT <- c("No ventilation","Ventilation 1-7 days","Ventilation >=8 days")
HLOS <- c("Hospital LOS <=7 days","Hospital LOS 8-27 days","Hospital LOS >=28 days")

dt[,ventilation_trajectory_final:=fcase(
  is.na(vent_days),NA_character_,
  vent_days<=0,VENT[1],
  vent_days<=7,VENT[2],
  vent_days>=8,VENT[3],
  default=NA_character_
)]
dt[,hlos_trajectory_final:=fcase(
  is.na(hospital_days),NA_character_,
  hospital_days<=7,HLOS[1],
  hospital_days<=27,HLOS[2],
  hospital_days>=28,HLOS[3],
  default=NA_character_
)]

dt[,death_hospice:=as.integer(as.character(discharge_3cat_final)=="Death/hospice")]
dt[,death_or_hlos28:=fcase(
  death_hospice==1L,1L,
  death_hospice==0L & !is.na(hospital_days) & hospital_days>=28,1L,
  death_hospice==0L & !is.na(hospital_days),0L,
  default=NA_integer_
)]
dt[,death_or_icu8:=fcase(
  death_hospice==1L,1L,
  death_hospice==0L & !is.na(icu_days) & icu_days>=8,1L,
  death_hospice==0L & !is.na(icu_days),0L,
  default=NA_integer_
)]
dt[,death_or_vent8:=fcase(
  death_hospice==1L,1L,
  death_hospice==0L & !is.na(vent_days) & vent_days>=8,1L,
  death_hospice==0L & !is.na(vent_days),0L,
  default=NA_integer_
)]

# -----------------------------------------------------------------------------
# Sensitivity strata
# -----------------------------------------------------------------------------

ge <- num(dt$gcs_eye_clean); gv <- num(dt$gcs_verbal_clean); gm <- num(dt$gcs_motor_clean)
gt <- rep(NA_real_,nrow(dt))
okg <- !is.na(ge)&!is.na(gv)&!is.na(gm)
gt[okg] <- ge[okg]+gv[okg]+gm[okg]

dt[,gcs_severity_sens:=fcase(
  !is.na(gt)&gt>=13&gt<=15,"Mild GCS 13-15",
  !is.na(gt)&gt>=9&gt<=12,"Moderate GCS 9-12",
  !is.na(gt)&gt>=3&gt<=8,"Severe GCS 3-8",
  default="GCS unknown/unclassifiable"
)]

agev <- num(dt$age)
dt[,age_group_sens:=fcase(
  !is.na(agev)&agev>=18&agev<=39,"18-39",
  !is.na(agev)&agev>=40&agev<=64,"40-64",
  !is.na(agev)&agev>=65&agev<=79,"65-79",
  !is.na(agev)&agev>=80&agev<=89,"80-89",
  default="Age unknown"
)]

intub <- num(dt$gcsq_intubated_recovered)
resp <- norm_chr(dt$respiratoryassistance_clean)
assisted <- tolower(resp)=="assisted"
resp_known <- !resp %in% c("__UNKNOWN__","__OTHER__")
dt[,airway_state_sens:=fcase(
  intub==1 | assisted,"Baseline airway-positive",
  intub==0 & resp_known & !assisted,"Baseline airway-negative",
  default="Baseline airway uncertain"
)]

num_prag <- intersect(nvars,pragmatic)
cat_prag <- intersect(cvars,pragmatic)
miss_count <- integer(nrow(dt))
for(v in num_prag) miss_count <- miss_count + as.integer(is.na(num(dt[[v]])))
for(v in cat_prag) {
  x <- norm_chr(dt[[v]])
  miss_count <- miss_count + as.integer(x %in% c("__UNKNOWN__","__OTHER__"))
}
dt[,clinical_input_missing_count:=miss_count]
dt[,missingness_sens:=fcase(
  clinical_input_missing_count==0L,"0 missing/unknown",
  clinical_input_missing_count==1L,"1 missing/unknown",
  clinical_input_missing_count==2L,"2 missing/unknown",
  clinical_input_missing_count>=3L,">=3 missing/unknown",
  default="Unknown"
)]

stratum_counts <- rbindlist(list(
  dt[,.(N=.N),by=.(admission_year,subgroup_level=gcs_severity_sens)][,subgroup_domain:="GCS severity"],
  dt[,.(N=.N),by=.(admission_year,subgroup_level=age_group_sens)][,subgroup_domain:="Age"],
  dt[,.(N=.N),by=.(admission_year,subgroup_level=norm_chr(sex_clean))][,subgroup_domain:="Sex"],
  dt[,.(N=.N),by=.(admission_year,subgroup_level=missingness_sens)][,subgroup_domain:="Clinical input missingness"],
  dt[,.(N=.N),by=.(admission_year,subgroup_level=airway_state_sens)][,subgroup_domain:="Baseline airway state"]
),fill=TRUE)
fwrite(stratum_counts,file.path(out_dir,"01_SENSITIVITY_STRATUM_COUNTS.csv"))

folds <- list(
  TEST_2022=list(train=2020L,tune=2021L,dev=2020:2021,test=2022L),
  TEST_2023=list(train=2020:2021,tune=2022L,dev=2020:2022,test=2023L),
  TEST_2024=list(train=2020:2022,tune=2023L,dev=2020:2023,test=2024L)
)

class_specs <- list(
  list(id="discharge_3cat_final",label="Disposition",type="multi",levels=DIS,
       variant="PRAGMATIC_PLUS_RACE_ETHNICITY_PAYER",phase="B_SYSTEM_MEDIATED",predictors=system_context),
  list(id="icu_trajectory_final",label="ICU trajectory",type="multi",levels=ICU,
       variant="PRAGMATIC_CLINICAL",phase="A_ALL_ENDPOINTS",predictors=pragmatic),
  list(id="ventilation_trajectory_final",label="Mechanical ventilation trajectory",type="multi",levels=VENT,
       variant="PRAGMATIC_CLINICAL",phase="A_ALL_ENDPOINTS",predictors=pragmatic),
  list(id="hlos_trajectory_final",label="Hospital LOS trajectory",type="multi",levels=HLOS,
       variant="PRAGMATIC_PLUS_RACE_ETHNICITY_PAYER",phase="B_SYSTEM_MEDIATED",predictors=system_context),
  list(id="icp_pressure_monitor_final",label="Invasive ICP monitoring",type="binary",levels=NULL,
       variant="PRAGMATIC_CLINICAL",phase="A_ALL_ENDPOINTS",predictors=pragmatic),
  list(id="craniotomy_craniectomy_final",label="Craniotomy/craniectomy",type="binary",levels=NULL,
       variant="PRAGMATIC_CLINICAL",phase="A_ALL_ENDPOINTS",predictors=pragmatic)
)

dur_specs <- list(
  list(id="hospital_los",label="Hospital LOS",y="hospital_days",lower=0,filt=function(x)!is.na(x)&x>=0),
  list(id="icu_los_conditional",label="ICU LOS conditional on ICU use",y="icu_days",lower=1,filt=function(x)!is.na(x)&x>0),
  list(id="ventilator_days_conditional",label="Ventilator duration conditional on ventilation",y="vent_days",lower=1,filt=function(x)!is.na(x)&x>0)
)

# -----------------------------------------------------------------------------
# Final classification prediction cache + subgroup / survivor / airway tests
# -----------------------------------------------------------------------------

overall_rows <- list()
sub_rows <- list()
survivor_rows <- list()
airway_rows <- list()

for(fold_name in names(folds)) {
  f <- folds[[fold_name]]
  idev <- which(dt$admission_year %in% f$dev)
  itest <- which(dt$admission_year==f$test)

  enc <- fit_encoder(dt[idev],nvars,all_cvars)
  Xd <- encode(dt[idev],enc); Xt <- encode(dt[itest],enc)

  for(sp in class_specs) {
    cat("\nFINAL CLASSIFICATION | ",fold_name," | ",sp$label,"\n",sep="")
    hp <- class_hp[
      class_hp[["phase"]]==sp$phase &
      class_hp[["fold_id"]]==fold_name &
      class_hp[["endpoint_id"]]==sp$id &
      class_hp[["variant_id"]]==sp$variant
    ]
    if(nrow(hp)!=1L) stop("Bad classification HP lookup: ",fold_name," / ",sp$id," / ",sp$variant)

    cd <- mask(Xd,sp$predictors); ct <- mask(Xt,sp$predictors)
    if(!identical(colnames(Xd)[cd],colnames(Xt)[ct])) stop("Classification feature mismatch.")

    if(sp$type=="binary") {
      ya <- num(dt[[sp$id]]); ya[!ya %in% c(0,1)] <- NA_real_
      kd <- !is.na(ya[idev]); kt <- !is.na(ya[itest])
      yd <- as.integer(ya[idev][kd]); yt <- as.integer(ya[itest][kt])

      pars <- reticulate::dict(
        objective="binary:logistic",eval_metric="logloss",eta=.05,
        max_depth=as.integer(hp$max_depth),min_child_weight=as.numeric(hp$min_child_weight),
        subsample=as.numeric(hp$subsample),colsample_bytree=as.numeric(hp$colsample_bytree),
        lambda=as.numeric(hp$lambda),tree_method="hist",device="cuda",
        nthread=as.integer(threads),seed=20260913L
      )
      dd <- xgb$DMatrix(Xd[kd,cd,drop=FALSE],label=yd)
      dtest <- xgb$DMatrix(Xt[kt,ct,drop=FALSE])
      fit <- xgb$train(pars,dd,as.integer(hp$best_iteration),verbose_eval=FALSE)
      pp <- as.numeric(fit$predict(dtest))

      z <- bmetrics(yt,pp)
      z[,`:=`(fold_id=fold_name,test_year=f$test,endpoint_id=sp$id,endpoint_label=sp$label,target=sp$label)]
      overall_rows[[length(overall_rows)+1L]] <- z

      rows <- itest[kt]
      subdefs <- list(
        "GCS severity"=dt$gcs_severity_sens[rows],
        "Age"=dt$age_group_sens[rows],
        "Sex"=dt$sex_clean[rows],
        "Clinical input missingness"=dt$missingness_sens[rows],
        "Race"=dt$race_clean[rows],
        "Ethnicity"=dt$ethnicity_clean[rows],
        "Payer"=dt$insurance_clean[rows]
      )
      for(dom in names(subdefs)) {
        g <- norm_chr(subdefs[[dom]])
        for(lv in unique(g)) {
          ii <- which(g==lv)
          if(length(ii)<500L) next
          yy <- yt[ii]
          if(sum(yy==1L)<25L||sum(yy==0L)<25L) next
          zz <- bmetrics(yy,pp[ii])
          zz[,`:=`(fold_id=fold_name,test_year=f$test,endpoint_id=sp$id,endpoint_label=sp$label,
                   target=sp$label,subgroup_domain=dom,subgroup_level=lv)]
          sub_rows[[length(sub_rows)+1L]] <- zz
        }
      }

      cache <- data.table(row_index=rows,admission_year=f$test,endpoint_id=sp$id,
                          observed=yt,predicted_probability=pp)
    } else {
      ya <- factor(as.character(dt[[sp$id]]),levels=sp$levels)
      kd <- !is.na(ya[idev]); kt <- !is.na(ya[itest])
      yd <- as.integer(ya[idev][kd])-1L; ytf <- ya[itest][kt]

      pars <- reticulate::dict(
        objective="multi:softprob",eval_metric="mlogloss",num_class=as.integer(length(sp$levels)),eta=.05,
        max_depth=as.integer(hp$max_depth),min_child_weight=as.numeric(hp$min_child_weight),
        subsample=as.numeric(hp$subsample),colsample_bytree=as.numeric(hp$colsample_bytree),
        lambda=as.numeric(hp$lambda),tree_method="hist",device="cuda",
        nthread=as.integer(threads),seed=20260913L
      )
      dd <- xgb$DMatrix(Xd[kd,cd,drop=FALSE],label=yd)
      dtest <- xgb$DMatrix(Xt[kt,ct,drop=FALSE])
      fit <- xgb$train(pars,dd,as.integer(hp$best_iteration),verbose_eval=FALSE)
      pp <- pred_matrix(fit$predict(dtest),sum(kt),length(sp$levels))
      pp <- pp/rowSums(pp); colnames(pp) <- sp$levels

      for(j in seq_along(sp$levels)) {
        yb <- as.integer(as.character(ytf)==sp$levels[j])
        z <- bmetrics(yb,pp[,j])
        z[,`:=`(fold_id=fold_name,test_year=f$test,endpoint_id=sp$id,endpoint_label=sp$label,target=sp$levels[j])]
        overall_rows[[length(overall_rows)+1L]] <- z
      }

      rows <- itest[kt]
      subdefs <- list(
        "GCS severity"=dt$gcs_severity_sens[rows],
        "Age"=dt$age_group_sens[rows],
        "Sex"=dt$sex_clean[rows],
        "Clinical input missingness"=dt$missingness_sens[rows],
        "Race"=dt$race_clean[rows],
        "Ethnicity"=dt$ethnicity_clean[rows],
        "Payer"=dt$insurance_clean[rows]
      )

      for(dom in names(subdefs)) {
        g <- norm_chr(subdefs[[dom]])
        for(lv in unique(g)) {
          ii <- which(g==lv)
          if(length(ii)<500L) next
          for(j in seq_along(sp$levels)) {
            yy <- as.integer(as.character(ytf[ii])==sp$levels[j])
            if(sum(yy==1L)<25L||sum(yy==0L)<25L) next
            zz <- bmetrics(yy,pp[ii,j])
            zz[,`:=`(fold_id=fold_name,test_year=f$test,endpoint_id=sp$id,endpoint_label=sp$label,
                     target=sp$levels[j],subgroup_domain=dom,subgroup_level=lv)]
            sub_rows[[length(sub_rows)+1L]] <- zz
          }
        }
      }

      if(sp$id=="ventilation_trajectory_final") {
        g <- dt$airway_state_sens[rows]
        for(lv in unique(g)) {
          ii <- which(g==lv)
          if(length(ii)<500L) next
          for(j in seq_along(sp$levels)) {
            yy <- as.integer(as.character(ytf[ii])==sp$levels[j])
            if(sum(yy==1L)<25L||sum(yy==0L)<25L) next
            zz <- bmetrics(yy,pp[ii,j])
            zz[,`:=`(fold_id=fold_name,test_year=f$test,endpoint_id=sp$id,endpoint_label=sp$label,
                     target=sp$levels[j],airway_state=lv)]
            airway_rows[[length(airway_rows)+1L]] <- zz
          }
        }
      }

      if(sp$id %in% c("icu_trajectory_final","ventilation_trajectory_final","hlos_trajectory_final")) {
        si <- which(dt$death_hospice[rows]==0L)
        if(length(si)>=500L) {
          for(j in seq_along(sp$levels)) {
            yy <- as.integer(as.character(ytf[si])==sp$levels[j])
            if(sum(yy==1L)<25L||sum(yy==0L)<25L) next
            zz <- bmetrics(yy,pp[si,j])
            zz[,`:=`(fold_id=fold_name,test_year=f$test,endpoint_id=sp$id,endpoint_label=sp$label,
                     target=sp$levels[j],sensitivity_population="Exclude death/hospice")]
            survivor_rows[[length(survivor_rows)+1L]] <- zz
          }
        }
      }

      cache <- data.table(row_index=rows,admission_year=f$test,endpoint_id=sp$id,
                          observed=as.character(ytf))
      for(j in seq_along(sp$levels)) {
        cache[,paste0("p__",make.names(sp$levels[j],unique=TRUE)):=pp[,j]]
      }
    }

    saveRDS(cache,file.path(pred_dir,paste0(fold_name,"__",sp$id,"__final_predictions.rds")),compress=FALSE)
    rm(dd,dtest,fit,cache); if(exists("pp")) rm(pp); gc()
  }
  rm(Xd,Xt,enc); gc()
}

class_overall <- bind0(overall_rows)
class_sub <- bind0(sub_rows)
survivor_class <- bind0(survivor_rows)
airway <- bind0(airway_rows)

fwrite(class_overall,file.path(out_dir,"02_FINAL_CLASSIFICATION_TEMPORAL_METRICS.csv"))
fwrite(class_sub,file.path(out_dir,"03_FINAL_CLASSIFICATION_SUBGROUP_ROBUSTNESS.csv"))
fwrite(survivor_class,file.path(out_dir,"04_SURVIVOR_ONLY_RESOURCE_TRAJECTORY_SENSITIVITY.csv"))
fwrite(airway,file.path(out_dir,"05_VENTILATION_BASELINE_AIRWAY_STATE_SENSITIVITY.csv"))

# -----------------------------------------------------------------------------
# Final pragmatic duration prediction cache + subgroup/survivor sensitivities
# -----------------------------------------------------------------------------

dur_overall_rows <- list()
dur_sub_rows <- list()
dur_surv_rows <- list()
alphas <- c(.1,.5,.9)

for(fold_name in names(folds)) {
  f <- folds[[fold_name]]
  idev <- which(dt$admission_year %in% f$dev)
  itest <- which(dt$admission_year==f$test)

  enc <- fit_encoder(dt[idev],nvars,cvars)
  Xd <- encode(dt[idev],enc); Xt <- encode(dt[itest],enc)
  cd <- mask(Xd,pragmatic); ct <- mask(Xt,pragmatic)

  for(sp in dur_specs) {
    cat("\nFINAL DURATION | ",fold_name," | ",sp$label,"\n",sep="")
    hp <- duration_hp[
      duration_hp[["fold_id"]]==fold_name &
      duration_hp[["duration_id"]]==sp$id &
      duration_hp[["variant_id"]]=="PRAGMATIC_CLINICAL"
    ]
    if(nrow(hp)!=1L) stop("Bad duration HP lookup: ",fold_name," / ",sp$id)

    ya <- num(dt[[sp$y]])
    kd <- sp$filt(ya[idev]); kt <- sp$filt(ya[itest])
    yd <- log1p(ya[idev][kd]); yt <- ya[itest][kt]

    pars <- reticulate::dict(
      objective="reg:quantileerror",quantile_alpha=alphas,eta=.05,
      max_depth=as.integer(hp$max_depth),min_child_weight=as.numeric(hp$min_child_weight),
      subsample=as.numeric(hp$subsample),colsample_bytree=as.numeric(hp$colsample_bytree),
      lambda=as.numeric(hp$lambda),tree_method="hist",device="cuda",
      nthread=as.integer(threads),seed=20260913L
    )
    dd <- xgb$DMatrix(Xd[kd,cd,drop=FALSE],label=yd)
    dtest <- xgb$DMatrix(Xt[kt,ct,drop=FALSE])
    fit <- xgb$train(pars,dd,as.integer(hp$best_iteration),verbose_eval=FALSE)

    qp <- expm1(pred_matrix(fit$predict(dtest),length(yt),3L))
    qp <- pmax(qp,sp$lower); qp <- rearrange_q(qp)
    q10 <- qp[,1]; q50 <- qp[,2]; q90 <- qp[,3]

    z <- dmetrics(yt,q10,q50,q90)
    z[,`:=`(fold_id=fold_name,test_year=f$test,duration_id=sp$id,duration_label=sp$label,
            sensitivity_population="Overall")]
    dur_overall_rows[[length(dur_overall_rows)+1L]] <- z

    rows <- itest[kt]
    subdefs <- list(
      "GCS severity"=dt$gcs_severity_sens[rows],
      "Age"=dt$age_group_sens[rows],
      "Sex"=dt$sex_clean[rows],
      "Clinical input missingness"=dt$missingness_sens[rows]
    )
    for(dom in names(subdefs)) {
      g <- norm_chr(subdefs[[dom]])
      for(lv in unique(g)) {
        ii <- which(g==lv); if(length(ii)<500L) next
        zz <- dmetrics(yt[ii],q10[ii],q50[ii],q90[ii])
        zz[,`:=`(fold_id=fold_name,test_year=f$test,duration_id=sp$id,duration_label=sp$label,
                 subgroup_domain=dom,subgroup_level=lv)]
        dur_sub_rows[[length(dur_sub_rows)+1L]] <- zz
      }
    }

    si <- which(dt$death_hospice[rows]==0L)
    if(length(si)>=500L) {
      zz <- dmetrics(yt[si],q10[si],q50[si],q90[si])
      zz[,`:=`(fold_id=fold_name,test_year=f$test,duration_id=sp$id,duration_label=sp$label,
               sensitivity_population="Exclude death/hospice")]
      dur_surv_rows[[length(dur_surv_rows)+1L]] <- zz
    }

    cache <- data.table(row_index=rows,admission_year=f$test,duration_id=sp$id,
                        observed_days=yt,q10=q10,q50=q50,q90=q90)
    saveRDS(cache,file.path(pred_dir,paste0(fold_name,"__",sp$id,"__final_duration_predictions.rds")),compress=FALSE)

    rm(dd,dtest,fit,qp,cache); gc()
  }
  rm(Xd,Xt,enc); gc()
}

dur_overall <- bind0(dur_overall_rows)
dur_sub <- bind0(dur_sub_rows)
dur_surv <- bind0(dur_surv_rows)

fwrite(dur_overall,file.path(out_dir,"06_FINAL_DURATION_TEMPORAL_METRICS.csv"))
fwrite(dur_sub,file.path(out_dir,"07_FINAL_DURATION_SUBGROUP_ROBUSTNESS.csv"))
fwrite(dur_surv,file.path(out_dir,"08_SURVIVOR_ONLY_DURATION_SENSITIVITY.csv"))

# -----------------------------------------------------------------------------
# Fully retuned early-death composite sensitivities
# -----------------------------------------------------------------------------

composites <- list(
  list(id="death_or_hlos28",label="Death/hospice or HLOS >=28 days",predictors=system_context),
  list(id="death_or_icu8",label="Death/hospice or ICU LOS >=8 days",predictors=pragmatic),
  list(id="death_or_vent8",label="Death/hospice or ventilation >=8 days",predictors=pragmatic)
)

selected_rows <- list()
composite_rows <- list()
ETA <- .05; MAX_ROUNDS <- 3000L; EARLY <- 75L

for(fold_name in names(folds)) {
  f <- folds[[fold_name]]
  itr <- which(dt$admission_year %in% f$train)
  itu <- which(dt$admission_year==f$tune)
  idv <- which(dt$admission_year %in% f$dev)
  ite <- which(dt$admission_year==f$test)

  et <- fit_encoder(dt[itr],nvars,all_cvars)
  ef <- fit_encoder(dt[idv],nvars,all_cvars)
  Xtr <- encode(dt[itr],et); Xtu <- encode(dt[itu],et)
  Xdv <- encode(dt[idv],ef); Xte <- encode(dt[ite],ef)

  for(sp in composites) {
    cat("\nCOMPOSITE | ",fold_name," | ",sp$label,"\n",sep="")
    ya <- num(dt[[sp$id]])
    ktr <- !is.na(ya[itr]); ktu <- !is.na(ya[itu]); kdv <- !is.na(ya[idv]); kte <- !is.na(ya[ite])
    ytr <- as.integer(ya[itr][ktr]); ytu <- as.integer(ya[itu][ktu])
    ydv <- as.integer(ya[idv][kdv]); yte <- as.integer(ya[ite][kte])

    ctr <- mask(Xtr,sp$predictors); ctu <- mask(Xtu,sp$predictors)
    cdv <- mask(Xdv,sp$predictors); cte <- mask(Xte,sp$predictors)

    dtr <- xgb$DMatrix(Xtr[ktr,ctr,drop=FALSE],label=ytr)
    dtu <- xgb$DMatrix(Xtu[ktu,ctu,drop=FALSE],label=ytu)
    grows <- vector("list",nrow(grid))

    for(i in seq_len(nrow(grid))) {
      g <- grid[i]
      pars <- reticulate::dict(
        objective="binary:logistic",eval_metric="logloss",eta=ETA,
        max_depth=as.integer(g$max_depth),min_child_weight=as.numeric(g$min_child_weight),
        subsample=as.numeric(g$subsample),colsample_bytree=as.numeric(g$colsample_bytree),
        lambda=as.numeric(g$lambda),tree_method="hist",device="cuda",
        nthread=as.integer(threads),seed=20260913L
      )
      fit <- xgb$train(
        pars,dtr,MAX_ROUNDS,
        evals=list(reticulate::tuple(dtr,"train"),reticulate::tuple(dtu,"tune")),
        early_stopping_rounds=EARLY,maximize=FALSE,verbose_eval=FALSE
      )
      grows[[i]] <- cbind(
        data.table(fold_id=fold_name,test_year=f$test,endpoint_id=sp$id,endpoint_label=sp$label,
                   best_iteration=best_iter(fit,MAX_ROUNDS),tune_score=best_score(fit)),
        g
      )
      rm(fit); gc()
    }

    gt <- rbindlist(grows,fill=TRUE); setorder(gt,tune_score,best_iteration)
    hp <- copy(gt[1L])
    hp[,selected_by:="Minimum fold-specific pre-test temporal tuning loss"]
    selected_rows[[length(selected_rows)+1L]] <- hp
    fwrite(gt,file.path(grid_out,paste0(fold_name,"__",sp$id,".csv")))

    pars <- reticulate::dict(
      objective="binary:logistic",eval_metric="logloss",eta=ETA,
      max_depth=as.integer(hp$max_depth),min_child_weight=as.numeric(hp$min_child_weight),
      subsample=as.numeric(hp$subsample),colsample_bytree=as.numeric(hp$colsample_bytree),
      lambda=as.numeric(hp$lambda),tree_method="hist",device="cuda",
      nthread=as.integer(threads),seed=20260913L
    )
    dd <- xgb$DMatrix(Xdv[kdv,cdv,drop=FALSE],label=ydv)
    dtest <- xgb$DMatrix(Xte[kte,cte,drop=FALSE])
    ff <- xgb$train(pars,dd,as.integer(hp$best_iteration),verbose_eval=FALSE)
    pp <- as.numeric(ff$predict(dtest))

    z <- bmetrics(yte,pp)
    z[,`:=`(fold_id=fold_name,test_year=f$test,endpoint_id=sp$id,endpoint_label=sp$label,
            selected_config_id=hp$config_id,selected_rounds=as.integer(hp$best_iteration))]
    composite_rows[[length(composite_rows)+1L]] <- z

    rm(dtr,dtu,dd,dtest,ff,pp); gc()
  }
  rm(Xtr,Xtu,Xdv,Xte,et,ef); gc()
}

selected <- bind0(selected_rows)
composite_metrics <- bind0(composite_rows)
fwrite(selected,file.path(out_dir,"09_COMPOSITE_EARLY_DEATH_SELECTED_HYPERPARAMETERS.csv"))
fwrite(composite_metrics,file.path(out_dir,"10_COMPOSITE_EARLY_DEATH_TEMPORAL_PERFORMANCE.csv"))

# -----------------------------------------------------------------------------
# Completion manifest
# -----------------------------------------------------------------------------

manifest <- data.table(
  sensitivity_domain=c(
    "Downstream trajectory-selection bias",
    "Development vs 2024 case mix",
    "Duration threshold justification",
    "COVID-era robustness",
    "Conventional-model comparator",
    "Predictor-family parsimony",
    "Sensitive sociodemographic augmentation",
    "Pragmatic predictor finalization",
    "TBI severity robustness",
    "Age/sex robustness",
    "Missing-input robustness",
    "Ventilation timing/airway-state leakage",
    "Early-death survivor-only sensitivity",
    "Early-death composite sensitivity",
    "Final pragmatic duration robustness",
    "Facility-level heterogeneity"
  ),
  status=c(
    "COMPLETE PRIOR","COMPLETE PRIOR","COMPLETE PRIOR","COMPLETE PRIOR",
    "COMPLETE PRIOR","COMPLETE PRIOR","COMPLETE PRIOR","COMPLETE PRIOR",
    "COMPLETE THIS SCRIPT","COMPLETE THIS SCRIPT","COMPLETE THIS SCRIPT",
    "COMPLETE THIS SCRIPT","COMPLETE THIS SCRIPT","COMPLETE THIS SCRIPT",
    "COMPLETE THIS SCRIPT","NOT FEASIBLE - FACILITY KEYS UNAVAILABLE"
  ),
  primary_output=c(
    file.path(methods_dir,"03_SELECTION_SMD_SUMMARY.csv"),
    file.path(methods_dir,"06_DEVELOPMENT_2020_23_VS_2024_SMD.csv"),
    file.path(methods_dir,"08_DEVELOPMENT_ONLY_THRESHOLD_PREVALENCE.csv"),
    file.path(output_dir,"SENSITIVITY_FINAL_TBI_TRACT_2022_2023_to_2024"),
    file.path(output_dir,"FINAL_TBI_TRACT_HYBRID_RIDGE_COMPARATOR_FIXED"),
    file.path(output_dir,"TBI_TRACT_PREDICTOR_STRESS_TEST"),
    file.path(output_dir,"TBI_TRACT_DEMOGRAPHIC_AUGMENTATION_STRESS_TEST"),
    file.path(class_dir,"07_PRAGMATIC_CLINICAL_VS_CURRENT_TEMPORAL_SUMMARY.csv"),
    file.path(out_dir,"03_FINAL_CLASSIFICATION_SUBGROUP_ROBUSTNESS.csv"),
    file.path(out_dir,"03_FINAL_CLASSIFICATION_SUBGROUP_ROBUSTNESS.csv"),
    file.path(out_dir,"03_FINAL_CLASSIFICATION_SUBGROUP_ROBUSTNESS.csv"),
    file.path(out_dir,"05_VENTILATION_BASELINE_AIRWAY_STATE_SENSITIVITY.csv"),
    file.path(out_dir,"04_SURVIVOR_ONLY_RESOURCE_TRAJECTORY_SENSITIVITY.csv"),
    file.path(out_dir,"10_COMPOSITE_EARLY_DEATH_TEMPORAL_PERFORMANCE.csv"),
    file.path(out_dir,"06_FINAL_DURATION_TEMPORAL_METRICS.csv"),
    NA_character_
  )
)
manifest[,output_exists:=ifelse(is.na(primary_output),NA,file.exists(primary_output))]
fwrite(manifest,file.path(out_dir,"11_SENSITIVITY_COMPLETION_MANIFEST.csv"))

writeLines(c(
  "TBI-TRACT FINAL SENSITIVITY SUITE COMPLETE",
  "",
  "Review first:",
  "  03_FINAL_CLASSIFICATION_SUBGROUP_ROBUSTNESS.csv",
  "  04_SURVIVOR_ONLY_RESOURCE_TRAJECTORY_SENSITIVITY.csv",
  "  05_VENTILATION_BASELINE_AIRWAY_STATE_SENSITIVITY.csv",
  "  07_FINAL_DURATION_SUBGROUP_ROBUSTNESS.csv",
  "  08_SURVIVOR_ONLY_DURATION_SENSITIVITY.csv",
  "  10_COMPOSITE_EARLY_DEATH_TEMPORAL_PERFORMANCE.csv",
  "  11_SENSITIVITY_COMPLETION_MANIFEST.csv",
  "",
  "Canonical forward-temporal prediction caches were saved under prediction_cache/.",
  "Facility-level heterogeneity was not evaluated because facility keys are unavailable."
),file.path(out_dir,"FINAL_SENSITIVITY_SUMMARY.txt"))

cat("\nDONE:",out_dir,"\n")
