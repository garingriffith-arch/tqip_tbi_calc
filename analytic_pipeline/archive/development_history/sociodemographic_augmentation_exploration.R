# =============================================================================
# 20_run_TBI_TRACT_demographic_augmentation_stress_test_GPU_v2.R
#
# Tests incremental predictive value of race, ethnicity, and payer using the
# same rolling temporal folds and fold-specific reference architectures from
# script 18. Includes classification, subgroup calibration, and Q10/Q50/Q90
# duration models.
#
# Run AFTER script 18. Do not run concurrently with script 19 on the same GPU.
# =============================================================================

rm(list = ls()); gc()

req <- c("data.table","reticulate")
miss <- req[!vapply(req, requireNamespace, logical(1), quietly=TRUE)]
if (length(miss)) stop("Missing package(s): ", paste(miss, collapse=", "))

suppressPackageStartupMessages({
  library(data.table)
  library(reticulate)
})

cfg <- c(file.path(getwd(),"R","00_config.R"),"R/00_config.R")
cfg <- cfg[file.exists(cfg)][1]
if (is.na(cfg)) stop("Could not find R/00_config.R")
source(cfg)

methods_dir <- file.path(output_dir,"METHODS_COMPLETION_TBI_TRACT")
stress_dir  <- file.path(output_dir,"TBI_TRACT_PREDICTOR_STRESS_TEST")
dataset_file <- file.path(methods_dir,"frozen_methods_dataset_retained_2020_2024.rds")
types_file   <- file.path(methods_dir,"11_FROZEN_PREDICTOR_TYPES.csv")
hp_file      <- file.path(stress_dir,"04_SELECTED_REFERENCE_HYPERPARAMETERS_BY_FOLD_ENDPOINT.csv")

needed <- c(dataset_file,types_file,hp_file)
if (!all(file.exists(needed))) {
  stop("Missing prerequisite file(s); script 18 must finish first:\n",
       paste(needed[!file.exists(needed)], collapse="\n"))
}

out_dir <- file.path(output_dir,"TBI_TRACT_DEMOGRAPHIC_AUGMENTATION_STRESS_TEST")
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)

ENV_NAME <- "tbi-tract-xgb-gpu"
cores <- parallel::detectCores(logical=TRUE)
if (is.na(cores) || cores < 2) cores <- 32L
threads <- max(1L,min(cores-1L,floor(.94*cores)))
setDTthreads(threads)
Sys.setenv(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
reticulate::use_condaenv(ENV_NAME, required=TRUE)
xgb <- reticulate::import("xgboost", convert=TRUE)

safe_num <- function(x) suppressWarnings(as.numeric(as.character(x)))
norm_chr <- function(x) {
  x <- trimws(as.character(x))
  x[is.na(x) | x=="" | x %in% c("NA","NaN","<NA>")] <- "__UNKNOWN__"
  x
}
clamp <- function(p,eps=1e-7) pmin(pmax(as.numeric(p),eps),1-eps)

auc_fast <- function(y,p) {
  k <- !is.na(y)&!is.na(p)&is.finite(p); y <- as.integer(y[k]); p <- as.numeric(p[k])
  k <- y %in% c(0L,1L); y <- y[k]; p <- p[k]
  n1 <- sum(y==1L); n0 <- sum(y==0L)
  if (!n1 || !n0) return(NA_real_)
  r <- rank(p,ties.method="average")
  (sum(r[y==1L])-n1*(n1+1)/2)/(n1*n0)
}
auprc_fast <- function(y,p) {
  k <- !is.na(y)&!is.na(p)&is.finite(p); y <- as.integer(y[k]); p <- as.numeric(p[k])
  k <- y %in% c(0L,1L); y <- y[k]; p <- p[k]
  npos <- sum(y==1L); if (!npos) return(NA_real_)
  o <- order(p,decreasing=TRUE); ys <- y[o]
  tp <- cumsum(ys==1L); fp <- cumsum(ys==0L)
  pr <- tp/(tp+fp); rc <- tp/npos; rc0 <- c(0,head(rc,-1L))
  sum(pr*(rc-rc0),na.rm=TRUE)
}
calib <- function(y,p) {
  k <- !is.na(y)&!is.na(p)&is.finite(p); y <- as.integer(y[k]); p <- clamp(p[k])
  if (length(unique(y))<2) return(c(intercept=NA_real_,slope=NA_real_))
  lp <- qlogis(p)
  fi <- tryCatch(glm(y~1,family=binomial(),offset=lp),error=function(e)NULL)
  fs <- tryCatch(glm(y~lp,family=binomial()),error=function(e)NULL)
  c(intercept=if(is.null(fi)) NA_real_ else unname(coef(fi)[1]),
    slope=if(is.null(fs)) NA_real_ else unname(coef(fs)["lp"]))
}
bin_metrics <- function(y,p) {
  c0 <- calib(y,p); pp <- clamp(p)
  data.table(N=length(y),events=sum(y==1L),prevalence=mean(y==1L),
             AUROC=auc_fast(y,p),AUPRC=auprc_fast(y,p),
             Brier=mean((p-y)^2),
             log_loss=-mean(y*log(pp)+(1-y)*log(1-pp)),
             calibration_intercept=unname(c0["intercept"]),
             calibration_slope=unname(c0["slope"]))
}
mlogloss <- function(yidx,p) {
  p <- pmax(pmin(p,1-1e-15),1e-15)
  -mean(log(p[cbind(seq_along(yidx),yidx)]))
}
mbrier <- function(yidx,p) {
  yy <- matrix(0,nrow=nrow(p),ncol=ncol(p)); yy[cbind(seq_len(nrow(p)),yidx)] <- 1
  mean(rowSums((p-yy)^2))
}
pinball <- function(y,q,a) {
  e <- y-q; mean(ifelse(e>=0,a*e,(a-1)*e),na.rm=TRUE)
}

fit_encoder <- function(d,numvars,catvars) {
  lev <- lapply(catvars,function(v) unique(c(sort(unique(norm_chr(d[[v]]))),
                                            "__UNKNOWN__","__OTHER__")))
  names(lev) <- catvars
  list(num=numvars,cat=catvars,lev=lev)
}
encode <- function(d,enc) {
  n <- nrow(d); nc <- length(enc$num)+sum(vapply(enc$lev,length,integer(1)))
  X <- matrix(0,n,nc); fn <- character(nc); parent <- character(nc); j <- 1L
  for (v in enc$num) {
    X[,j] <- safe_num(d[[v]]); fn[j] <- v; parent[j] <- v; j <- j+1L
  }
  for (v in enc$cat) {
    lev <- enc$lev[[v]]; x <- norm_chr(d[[v]]); x[!x %in% lev] <- "__OTHER__"
    idx <- match(x,lev); cols <- j:(j+length(lev)-1L)
    X[cbind(seq_len(n),cols[idx])] <- 1
    fn[cols] <- paste0(v,"__",make.names(lev,unique=TRUE)); parent[cols] <- v
    j <- max(cols)+1L
  }
  colnames(X) <- fn; attr(X,"parent_predictor") <- parent; storage.mode(X) <- "double"; X
}
mask_cols <- function(X,keep) attr(X,"parent_predictor") %in% keep

best_iter <- function(fit,fallback=3000L) {
  z <- tryCatch(as.numeric(reticulate::py_to_r(fit$best_iteration)),
                error=function(e) NA_real_)
  if (length(z) && is.finite(z[1])) as.integer(z[1]+1L) else as.integer(fallback)
}
best_score <- function(fit) {
  z <- tryCatch(as.numeric(reticulate::py_to_r(fit$best_score)),
                error=function(e) NA_real_)
  if (length(z)&&is.finite(z[1])) z[1] else NA_real_
}
pred_mat <- function(pred,n,k) {
  p <- tryCatch(reticulate::py_to_r(pred),error=function(e)pred)
  if (is.matrix(p)||is.data.frame(p)) p <- as.matrix(p)
  else p <- matrix(as.numeric(p),nrow=n,ncol=k,byrow=TRUE)
  if (nrow(p)!=n || ncol(p)!=k) stop("Unexpected prediction shape")
  storage.mode(p) <- "double"; p
}
rearrange_q <- function(p) {
  lo <- pmin(p[,1],p[,2],p[,3]); hi <- pmax(p[,1],p[,2],p[,3]); mid <- rowSums(p)-lo-hi
  cbind(q10=lo,q50=mid,q90=hi)
}
bind0 <- function(x) if(length(x)) rbindlist(x,fill=TRUE) else data.table()

subgroup_bin <- function(y,p,g,domain,min_n=500L,min_events=25L) {
  d <- data.table(y=as.integer(y),p=as.numeric(p),level=norm_chr(g)); out <- list()
  for (lv in unique(d$level)) {
    s <- d[level==lv]; ev <- sum(s$y==1L); nev <- nrow(s)-ev
    if (nrow(s)<min_n || ev<min_events || nev<min_events) next
    z <- bin_metrics(s$y,s$p); z[,`:=`(subgroup_domain=domain,subgroup_level=lv)]
    out[[length(out)+1L]] <- z
  }
  bind0(out)
}

dt <- as.data.table(readRDS(dataset_file))
types <- fread(types_file)
hp_tab <- fread(hp_file)

num0 <- types[category=="numeric",predictor]
cat0 <- types[category=="categorical",predictor]
current <- c(num0,cat0)

social <- c("race_clean","ethnicity_clean","insurance_clean")
if (length(setdiff(social,names(dt)))) {
  stop("Missing social variable(s): ",paste(setdiff(social,names(dt)),collapse=", "))
}

num_all <- num0
cat_all <- unique(c(cat0,social))

variants <- list(
  CURRENT_REFERENCE=current,
  PLUS_RACE=unique(c(current,"race_clean")),
  PLUS_ETHNICITY=unique(c(current,"ethnicity_clean")),
  PLUS_RACE_ETHNICITY=unique(c(current,"race_clean","ethnicity_clean")),
  PLUS_PAYER=unique(c(current,"insurance_clean")),
  PLUS_RACE_ETHNICITY_PAYER=unique(c(current,social))
)
fwrite(rbindlist(lapply(names(variants),function(v) data.table(
  variant_id=v,added_variables=paste(setdiff(variants[[v]],current),collapse=";"),
  n_predictors=length(variants[[v]])
))),file.path(out_dir,"01_VARIANTS.csv"))

aud <- bind0(lapply(social,function(v) {
  z <- data.table(variable=v,admission_year=dt$admission_year,level=norm_chr(dt[[v]]))
  z[,.(N=.N),by=.(variable,admission_year,level)][,proportion:=N/sum(N),by=.(variable,admission_year)]
}))
fwrite(aud,file.path(out_dir,"02_SOCIAL_VARIABLE_LEVEL_AUDIT.csv"))

dt[,hospital_days:=safe_num(hospital_days)]
dt[,icu_days:=safe_num(icu_days)]
dt[,vent_days:=safe_num(vent_days)]
dt[hospital_days<0,hospital_days:=NA_real_]
dt[icu_days<0,icu_days:=NA_real_]
dt[vent_days<0,vent_days:=NA_real_]

DIS <- c("Home/home health","Post-acute facility","Death/hospice")
ICU <- c("No ICU","ICU 1-7 days","ICU >=8 days")
VENT <- c("No ventilation","Ventilation 1-7 days","Ventilation >=8 days")
HLOS <- c("Hospital LOS <=7 days","Hospital LOS 8-27 days","Hospital LOS >=28 days")

dt[,ventilation_trajectory_final:=fcase(
  is.na(vent_days),NA_character_,vent_days<=0,VENT[1],vent_days<=7,VENT[2],
  vent_days>=8,VENT[3],default=NA_character_)]
dt[,hlos_trajectory_final:=fcase(
  is.na(hospital_days),NA_character_,hospital_days<=7,HLOS[1],hospital_days<=27,HLOS[2],
  hospital_days>=28,HLOS[3],default=NA_character_)]

eps <- list(
  list(id="discharge_3cat_final",label="Disposition",type="multi",levels=DIS),
  list(id="icu_trajectory_final",label="ICU trajectory",type="multi",levels=ICU),
  list(id="ventilation_trajectory_final",label="Mechanical ventilation trajectory",type="multi",levels=VENT),
  list(id="hlos_trajectory_final",label="Hospital LOS trajectory",type="multi",levels=HLOS),
  list(id="icp_pressure_monitor_final",label="Invasive ICP monitoring",type="binary",levels=NULL),
  list(id="craniotomy_craniectomy_final",label="Craniotomy/craniectomy",type="binary",levels=NULL)
)
dur_eps <- list(
  list(id="hospital_los",label="Hospital LOS",y="hospital_days",arch="hlos_trajectory_final",lower=0,
       filt=function(x)!is.na(x)&x>=0),
  list(id="icu_los_conditional",label="ICU LOS conditional on ICU use",y="icu_days",arch="icu_trajectory_final",lower=1,
       filt=function(x)!is.na(x)&x>0),
  list(id="ventilator_days_conditional",label="Ventilator duration conditional on ventilation",y="vent_days",
       arch="ventilation_trajectory_final",lower=1,filt=function(x)!is.na(x)&x>0)
)
folds <- list(
  TEST_2022=list(train=2020L,tune=2021L,dev=2020:2021,test=2022L),
  TEST_2023=list(train=2020:2021,tune=2022L,dev=2020:2022,test=2023L),
  TEST_2024=list(train=2020:2022,tune=2023L,dev=2020:2023,test=2024L)
)

ETA <- .05; MAX_ROUNDS <- 3000L; EARLY <- 75L
overall_rows <- class_rows <- subgroup_rows <- qc_rows <- list()

for (fold_name in names(folds)) {
  f <- folds[[fold_name]]
  itrain <- which(dt$admission_year %in% f$train)
  itune  <- which(dt$admission_year==f$tune)
  idev   <- which(dt$admission_year %in% f$dev)
  itest  <- which(dt$admission_year==f$test)

  et <- fit_encoder(dt[itrain],num_all,cat_all)
  ef <- fit_encoder(dt[idev],num_all,cat_all)
  Xtr <- encode(dt[itrain],et); Xtu <- encode(dt[itune],et)
  Xdv <- encode(dt[idev],ef);   Xte <- encode(dt[itest],ef)
  sg <- data.table(race=norm_chr(dt$race_clean[itest]),
                   ethnicity=norm_chr(dt$ethnicity_clean[itest]),
                   payer=norm_chr(dt$insurance_clean[itest]))

  for (sp in eps) {
    cat("\n",fold_name," | ",sp$label,"\n",sep="")
    hp <- hp_tab[hp_tab$fold_id==fold_name & hp_tab$endpoint_id==sp$id]
    if (nrow(hp)!=1L) stop("Bad HP lookup: ",fold_name," / ",sp$id)
    reqhp <- c("max_depth","min_child_weight","subsample","colsample_bytree","lambda")
    if (any(!vapply(reqhp,function(h) h %in% names(hp)&&length(hp[[h]])&&
                    is.finite(safe_num(hp[[h]][1])),logical(1))))
      stop("Invalid HP for ",fold_name," / ",sp$id)

    if (sp$type=="binary") {
      yall <- safe_num(dt[[sp$id]]); yall[!yall %in% c(0,1)] <- NA_real_
    } else yall <- factor(as.character(dt[[sp$id]]),levels=sp$levels)

    kt <- !is.na(yall[itrain]); ku <- !is.na(yall[itune])
    kd <- !is.na(yall[idev]);   ke <- !is.na(yall[itest])

    for (v in names(variants)) {
      keep <- variants[[v]]
      ctr <- mask_cols(Xtr,keep); ctu <- mask_cols(Xtu,keep)
      cdv <- mask_cols(Xdv,keep); cte <- mask_cols(Xte,keep)
      if (!identical(colnames(Xtr)[ctr],colnames(Xtu)[ctu])) stop("train/tune feature mismatch: ",v)
      if (!identical(colnames(Xdv)[cdv],colnames(Xte)[cte])) stop("dev/test feature mismatch: ",v)

      if (sp$type=="binary") {
        ytr <- as.integer(yall[itrain][kt]); ytu <- as.integer(yall[itune][ku])
        ydv <- as.integer(yall[idev][kd]);   yte <- as.integer(yall[itest][ke])
        pars <- reticulate::dict(objective="binary:logistic",eval_metric="logloss",
          eta=ETA,max_depth=as.integer(hp$max_depth),min_child_weight=as.numeric(hp$min_child_weight),
          subsample=as.numeric(hp$subsample),colsample_bytree=as.numeric(hp$colsample_bytree),
          lambda=as.numeric(hp$lambda),tree_method="hist",device="cuda",
          nthread=as.integer(threads),seed=20260911L)
        dtr <- xgb$DMatrix(Xtr[kt,ctr,drop=FALSE],label=ytr)
        dtu <- xgb$DMatrix(Xtu[ku,ctu,drop=FALSE],label=ytu)
        tf <- xgb$train(pars,dtr,MAX_ROUNDS,
          evals=list(reticulate::tuple(dtr,"train"),reticulate::tuple(dtu,"tune")),
          early_stopping_rounds=EARLY,maximize=FALSE,verbose_eval=FALSE)
        nr <- best_iter(tf,MAX_ROUNDS)
        ddv <- xgb$DMatrix(Xdv[kd,cdv,drop=FALSE],label=ydv)
        dte <- xgb$DMatrix(Xte[ke,cte,drop=FALSE])
        ff <- xgb$train(pars,ddv,nr,verbose_eval=FALSE)
        p <- as.numeric(ff$predict(dte))
        z <- bin_metrics(yte,p); z[,`:=`(fold_id=fold_name,test_year=f$test,endpoint_id=sp$id,
             endpoint_label=sp$label,target=sp$label,variant_id=v,selected_rounds=nr)]
        overall_rows[[length(overall_rows)+1L]] <- z
        sge <- sg[ke]
        for (gname in c("race","ethnicity","payer")) {
          zz <- subgroup_bin(yte,p,sge[[gname]],gname)
          if (nrow(zz)) {
            zz[,`:=`(fold_id=fold_name,test_year=f$test,endpoint_id=sp$id,endpoint_label=sp$label,
                     target=sp$label,variant_id=v)]
            subgroup_rows[[length(subgroup_rows)+1L]] <- zz
          }
        }
      } else {
        ytr <- as.integer(yall[itrain][kt])-1L; ytu <- as.integer(yall[itune][ku])-1L
        ydv <- as.integer(yall[idev][kd])-1L; ytef <- yall[itest][ke]
        pars <- reticulate::dict(objective="multi:softprob",eval_metric="mlogloss",
          num_class=as.integer(length(sp$levels)),eta=ETA,max_depth=as.integer(hp$max_depth),
          min_child_weight=as.numeric(hp$min_child_weight),subsample=as.numeric(hp$subsample),
          colsample_bytree=as.numeric(hp$colsample_bytree),lambda=as.numeric(hp$lambda),
          tree_method="hist",device="cuda",nthread=as.integer(threads),seed=20260911L)
        dtr <- xgb$DMatrix(Xtr[kt,ctr,drop=FALSE],label=ytr)
        dtu <- xgb$DMatrix(Xtu[ku,ctu,drop=FALSE],label=ytu)
        tf <- xgb$train(pars,dtr,MAX_ROUNDS,
          evals=list(reticulate::tuple(dtr,"train"),reticulate::tuple(dtu,"tune")),
          early_stopping_rounds=EARLY,maximize=FALSE,verbose_eval=FALSE)
        nr <- best_iter(tf,MAX_ROUNDS)
        ddv <- xgb$DMatrix(Xdv[kd,cdv,drop=FALSE],label=ydv)
        dte <- xgb$DMatrix(Xte[ke,cte,drop=FALSE])
        ff <- xgb$train(pars,ddv,nr,verbose_eval=FALSE)
        p <- pred_mat(ff$predict(dte),sum(ke),length(sp$levels)); p <- p/rowSums(p); colnames(p) <- sp$levels

        cm <- list(); sge <- sg[ke]
        for (j in seq_along(sp$levels)) {
          lev <- sp$levels[j]; yb <- as.integer(as.character(ytef)==lev); pb <- p[,j]
          zz <- bin_metrics(yb,pb); zz[,`:=`(fold_id=fold_name,test_year=f$test,endpoint_id=sp$id,
               endpoint_label=sp$label,target=lev,variant_id=v,selected_rounds=nr)]
          cm[[length(cm)+1L]] <- zz
          for (gname in c("race","ethnicity","payer")) {
            ss <- subgroup_bin(yb,pb,sge[[gname]],gname)
            if (nrow(ss)) {
              ss[,`:=`(fold_id=fold_name,test_year=f$test,endpoint_id=sp$id,endpoint_label=sp$label,
                       target=lev,variant_id=v)]
              subgroup_rows[[length(subgroup_rows)+1L]] <- ss
            }
          }
        }
        if (sp$id=="icu_trajectory_final") {
          yb <- as.integer(as.character(ytef)!="No ICU"); pb <- 1-p[,"No ICU"]
          zz <- bin_metrics(yb,pb); zz[,`:=`(fold_id=fold_name,test_year=f$test,endpoint_id=sp$id,
               endpoint_label=sp$label,target="Any ICU use",variant_id=v,selected_rounds=nr)]
          cm[[length(cm)+1L]] <- zz
        }
        if (sp$id=="ventilation_trajectory_final") {
          yb <- as.integer(as.character(ytef)!="No ventilation"); pb <- 1-p[,"No ventilation"]
          zz <- bin_metrics(yb,pb); zz[,`:=`(fold_id=fold_name,test_year=f$test,endpoint_id=sp$id,
               endpoint_label=sp$label,target="Any ventilation",variant_id=v,selected_rounds=nr)]
          cm[[length(cm)+1L]] <- zz
        }
        cmt <- rbindlist(cm,fill=TRUE); class_rows[[length(class_rows)+1L]] <- cmt
        idx <- as.integer(ytef)
        baseclasses <- cmt[!target %in% c("Any ICU use","Any ventilation")]
        overall_rows[[length(overall_rows)+1L]] <- data.table(
          fold_id=fold_name,test_year=f$test,endpoint_id=sp$id,endpoint_label=sp$label,
          target="<multiclass overall>",variant_id=v,selected_rounds=nr,N=length(idx),
          accuracy=mean(max.col(p,ties.method="first")==idx),
          multiclass_log_loss=mlogloss(idx,p),multiclass_brier=mbrier(idx,p),
          macro_AUROC=mean(baseclasses$AUROC,na.rm=TRUE),macro_AUPRC=mean(baseclasses$AUPRC,na.rm=TRUE))
      }
      qc_rows[[length(qc_rows)+1L]] <- data.table(
        fold_id=fold_name,test_year=f$test,endpoint_id=sp$id,endpoint_label=sp$label,variant_id=v,
        n_raw_predictors=length(keep),n_encoded_features_train=sum(ctr),n_encoded_features_final=sum(cdv),
        selected_rounds=nr,tune_score=best_score(tf))
      rm(dtr,dtu,tf,ddv,dte,ff); if(exists("p")) rm(p); gc()
    }
  }
  rm(Xtr,Xtu,Xdv,Xte,et,ef); gc()
}

overall <- bind0(overall_rows); classm <- bind0(class_rows); subm <- bind0(subgroup_rows); qct <- bind0(qc_rows)
fwrite(overall,file.path(out_dir,"03_CLASSIFICATION_OVERALL_METRICS.csv"))
fwrite(classm,file.path(out_dir,"04_CLASSIFICATION_CLASS_METRICS.csv"))
fwrite(subm,file.path(out_dir,"05_CLASSIFICATION_SUBGROUP_METRICS.csv"))
fwrite(qct,file.path(out_dir,"06_CLASSIFICATION_RUN_QC.csv"))

cref <- classm[variant_id=="CURRENT_REFERENCE"]
cd <- merge(classm,cref,by=c("fold_id","test_year","endpoint_id","endpoint_label","target"),
            suffixes=c("","_reference"))
for(m in c("AUROC","AUPRC","Brier","log_loss","calibration_intercept","calibration_slope"))
  cd[,paste0("delta_",m):=get(m)-get(paste0(m,"_reference"))]
fwrite(cd,file.path(out_dir,"07_CLASSIFICATION_CLASS_DELTAS_VS_REFERENCE.csv"))

oref <- overall[variant_id=="CURRENT_REFERENCE"]
od <- merge(overall,oref,by=c("fold_id","test_year","endpoint_id","endpoint_label","target"),
            suffixes=c("","_reference"))
for(m in intersect(c("AUROC","AUPRC","Brier","log_loss","accuracy","multiclass_log_loss",
                     "multiclass_brier","macro_AUROC","macro_AUPRC"),names(overall))) {
  rn <- paste0(m,"_reference"); if(rn %in% names(od)) od[,paste0("delta_",m):=get(m)-get(rn)]
}
fwrite(od,file.path(out_dir,"08_CLASSIFICATION_OVERALL_DELTAS_VS_REFERENCE.csv"))

sref <- subm[variant_id=="CURRENT_REFERENCE"]
sd <- merge(subm,sref,by=c("fold_id","test_year","endpoint_id","endpoint_label","target",
                           "subgroup_domain","subgroup_level"),suffixes=c("","_reference"))
for(m in c("AUROC","AUPRC","Brier","log_loss","calibration_intercept","calibration_slope"))
  sd[,paste0("delta_",m):=get(m)-get(paste0(m,"_reference"))]
sd[,delta_abs_calibration_intercept:=abs(calibration_intercept)-abs(calibration_intercept_reference)]
sd[,delta_abs_calibration_slope_error:=abs(calibration_slope-1)-abs(calibration_slope_reference-1)]
fwrite(sd,file.path(out_dir,"09_CLASSIFICATION_SUBGROUP_DELTAS_VS_REFERENCE.csv"))

csum <- cd[variant_id!="CURRENT_REFERENCE",.(
  n_fold_target_comparisons=.N,
  median_delta_AUROC=median(delta_AUROC,na.rm=TRUE),
  worst_delta_AUROC=min(delta_AUROC,na.rm=TRUE),
  best_delta_AUROC=max(delta_AUROC,na.rm=TRUE),
  median_delta_AUPRC=median(delta_AUPRC,na.rm=TRUE),
  worst_delta_AUPRC=min(delta_AUPRC,na.rm=TRUE),
  best_delta_AUPRC=max(delta_AUPRC,na.rm=TRUE),
  median_delta_Brier=median(delta_Brier,na.rm=TRUE),
  median_delta_log_loss=median(delta_log_loss,na.rm=TRUE)
),by=.(endpoint_id,endpoint_label,variant_id)]
fwrite(csum,file.path(out_dir,"10_CLASSIFICATION_TEMPORAL_SUMMARY.csv"))

ssum <- sd[variant_id!="CURRENT_REFERENCE",.(
  n_subgroup_comparisons=.N,
  median_delta_AUROC=median(delta_AUROC,na.rm=TRUE),
  median_delta_AUPRC=median(delta_AUPRC,na.rm=TRUE),
  median_delta_Brier=median(delta_Brier,na.rm=TRUE),
  median_delta_abs_calibration_intercept=median(delta_abs_calibration_intercept,na.rm=TRUE),
  worst_delta_abs_calibration_intercept=max(delta_abs_calibration_intercept,na.rm=TRUE),
  median_delta_abs_calibration_slope_error=median(delta_abs_calibration_slope_error,na.rm=TRUE),
  worst_delta_abs_calibration_slope_error=max(delta_abs_calibration_slope_error,na.rm=TRUE)
),by=.(endpoint_id,endpoint_label,variant_id,subgroup_domain)]
fwrite(ssum,file.path(out_dir,"11_SUBGROUP_CALIBRATION_TEMPORAL_SUMMARY.csv"))

dur_rows <- dur_sub_rows <- dur_qc_rows <- list()
ALPHAS <- c(.10,.50,.90)

for (fold_name in names(folds)) {
  f <- folds[[fold_name]]
  itrain <- which(dt$admission_year %in% f$train); itune <- which(dt$admission_year==f$tune)
  idev <- which(dt$admission_year %in% f$dev); itest <- which(dt$admission_year==f$test)
  et <- fit_encoder(dt[itrain],num_all,cat_all); ef <- fit_encoder(dt[idev],num_all,cat_all)
  Xtr <- encode(dt[itrain],et); Xtu <- encode(dt[itune],et)
  Xdv <- encode(dt[idev],ef); Xte <- encode(dt[itest],ef)
  sg <- data.table(race=norm_chr(dt$race_clean[itest]),ethnicity=norm_chr(dt$ethnicity_clean[itest]),
                   payer=norm_chr(dt$insurance_clean[itest]))

  for (sp in dur_eps) {
    cat("\nDURATION | ",fold_name," | ",sp$label,"\n",sep="")
    hp <- hp_tab[hp_tab$fold_id==fold_name & hp_tab$endpoint_id==sp$arch]
    if(nrow(hp)!=1L) stop("Bad duration HP lookup: ",fold_name," / ",sp$arch)
    yall <- safe_num(dt[[sp$y]])
    kt <- sp$filt(yall[itrain]); ku <- sp$filt(yall[itune]); kd <- sp$filt(yall[idev]); ke <- sp$filt(yall[itest])
    ytr <- log1p(yall[itrain][kt]); ytu <- log1p(yall[itune][ku]); ydv <- log1p(yall[idev][kd]); yte <- yall[itest][ke]

    for(v in names(variants)) {
      keep <- variants[[v]]; ctr <- mask_cols(Xtr,keep); ctu <- mask_cols(Xtu,keep)
      cdv <- mask_cols(Xdv,keep); cte <- mask_cols(Xte,keep)
      pars <- reticulate::dict(objective="reg:quantileerror",quantile_alpha=ALPHAS,eta=ETA,
        max_depth=as.integer(hp$max_depth),min_child_weight=as.numeric(hp$min_child_weight),
        subsample=as.numeric(hp$subsample),colsample_bytree=as.numeric(hp$colsample_bytree),
        lambda=as.numeric(hp$lambda),tree_method="hist",device="cuda",
        nthread=as.integer(threads),seed=20260911L)
      dtr <- xgb$DMatrix(Xtr[kt,ctr,drop=FALSE],label=ytr); dtu <- xgb$DMatrix(Xtu[ku,ctu,drop=FALSE],label=ytu)
      tf <- xgb$train(pars,dtr,MAX_ROUNDS,
        evals=list(reticulate::tuple(dtr,"train"),reticulate::tuple(dtu,"tune")),
        early_stopping_rounds=EARLY,maximize=FALSE,verbose_eval=FALSE)
      nr <- best_iter(tf,MAX_ROUNDS)
      ddv <- xgb$DMatrix(Xdv[kd,cdv,drop=FALSE],label=ydv); dte <- xgb$DMatrix(Xte[ke,cte,drop=FALSE])
      ff <- xgb$train(pars,ddv,nr,verbose_eval=FALSE)
      pr <- expm1(pred_mat(ff$predict(dte),length(yte),3L)); pr <- pmax(pr,sp$lower)
      crossing <- (pr[,1]>pr[,2])|(pr[,2]>pr[,3]); pr <- rearrange_q(pr)
      q10 <- pr[,1]; q50 <- pr[,2]; q90 <- pr[,3]; cov <- mean(yte>=q10 & yte<=q90)
      dur_rows[[length(dur_rows)+1L]] <- data.table(
        fold_id=fold_name,test_year=f$test,duration_id=sp$id,duration_label=sp$label,
        architecture_endpoint=sp$arch,variant_id=v,N=length(yte),selected_rounds=nr,
        median_MAE_days=mean(abs(yte-q50)),median_bias_days=median(q50-yte),
        mean_pinball=mean(c(pinball(yte,q10,.1),pinball(yte,q50,.5),pinball(yte,q90,.9))),
        central_80_coverage=cov,absolute_80_coverage_error=abs(cov-.8),
        median_80PI_width_days=median(q90-q10),q10_empirical_cdf=mean(yte<=q10),
        q50_empirical_cdf=mean(yte<=q50),q90_empirical_cdf=mean(yte<=q90),
        quantile_calibration_MAE=mean(abs(c(mean(yte<=q10)-.1,mean(yte<=q50)-.5,mean(yte<=q90)-.9))),
        raw_crossing_rate=mean(crossing))

      sge <- sg[ke]
      for(gname in c("race","ethnicity","payer")) {
        gv <- sge[[gname]]
        for(lv in unique(gv)) {
          ii <- which(gv==lv); if(length(ii)<500L) next
          ys <- yte[ii]; l <- q10[ii]; m <- q50[ii]; u <- q90[ii]
          dur_sub_rows[[length(dur_sub_rows)+1L]] <- data.table(
            fold_id=fold_name,test_year=f$test,duration_id=sp$id,duration_label=sp$label,
            variant_id=v,subgroup_domain=gname,subgroup_level=lv,N=length(ii),
            median_MAE_days=mean(abs(ys-m)),median_bias_days=median(m-ys),
            central_80_coverage=mean(ys>=l & ys<=u),median_80PI_width_days=median(u-l))
        }
      }
      dur_qc_rows[[length(dur_qc_rows)+1L]] <- data.table(
        fold_id=fold_name,test_year=f$test,duration_id=sp$id,duration_label=sp$label,variant_id=v,
        n_raw_predictors=length(keep),n_encoded_features_train=sum(ctr),n_encoded_features_final=sum(cdv),
        selected_rounds=nr,tune_score=best_score(tf))
      rm(dtr,dtu,tf,ddv,dte,ff,pr); gc()
    }
  }
  rm(Xtr,Xtu,Xdv,Xte,et,ef); gc()
}

dur <- bind0(dur_rows); dsub <- bind0(dur_sub_rows); dqc <- bind0(dur_qc_rows)
fwrite(dur,file.path(out_dir,"12_DURATION_METRICS.csv"))
fwrite(dsub,file.path(out_dir,"13_DURATION_SUBGROUP_METRICS.csv"))
fwrite(dqc,file.path(out_dir,"14_DURATION_RUN_QC.csv"))

dref <- dur[variant_id=="CURRENT_REFERENCE"]
dd <- merge(dur,dref,by=c("fold_id","test_year","duration_id","duration_label","architecture_endpoint"),
            suffixes=c("","_reference"))
for(m in c("median_MAE_days","median_bias_days","mean_pinball","central_80_coverage",
           "absolute_80_coverage_error","median_80PI_width_days","q10_empirical_cdf",
           "q50_empirical_cdf","q90_empirical_cdf","quantile_calibration_MAE","raw_crossing_rate"))
  dd[,paste0("delta_",m):=get(m)-get(paste0(m,"_reference"))]
fwrite(dd,file.path(out_dir,"15_DURATION_DELTAS_VS_REFERENCE.csv"))

dsref <- dsub[variant_id=="CURRENT_REFERENCE"]
dsd <- merge(dsub,dsref,by=c("fold_id","test_year","duration_id","duration_label","subgroup_domain","subgroup_level"),
             suffixes=c("","_reference"))
for(m in c("median_MAE_days","median_bias_days","central_80_coverage","median_80PI_width_days"))
  dsd[,paste0("delta_",m):=get(m)-get(paste0(m,"_reference"))]
dsd[,delta_absolute_coverage_error:=abs(central_80_coverage-.8)-abs(central_80_coverage_reference-.8)]
fwrite(dsd,file.path(out_dir,"16_DURATION_SUBGROUP_DELTAS_VS_REFERENCE.csv"))

dsum <- dd[variant_id!="CURRENT_REFERENCE",.(
  n_folds=.N,median_delta_MAE_days=median(delta_median_MAE_days,na.rm=TRUE),
  worst_delta_MAE_days=max(delta_median_MAE_days,na.rm=TRUE),
  median_delta_pinball=median(delta_mean_pinball,na.rm=TRUE),
  worst_delta_pinball=max(delta_mean_pinball,na.rm=TRUE),
  median_delta_absolute_coverage_error=median(delta_absolute_80_coverage_error,na.rm=TRUE),
  worst_delta_absolute_coverage_error=max(delta_absolute_80_coverage_error,na.rm=TRUE),
  median_delta_quantile_calibration_MAE=median(delta_quantile_calibration_MAE,na.rm=TRUE)
),by=.(duration_id,duration_label,variant_id)]
fwrite(dsum,file.path(out_dir,"17_DURATION_TEMPORAL_SUMMARY.csv"))

sub_all <- ssum[,.(median_subgroup_delta_abs_calibration_intercept=
                     median(median_delta_abs_calibration_intercept,na.rm=TRUE),
                   worst_subgroup_delta_abs_calibration_intercept=
                     max(worst_delta_abs_calibration_intercept,na.rm=TRUE),
                   median_subgroup_delta_abs_calibration_slope_error=
                     median(median_delta_abs_calibration_slope_error,na.rm=TRUE)),
                by=.(endpoint_id,endpoint_label,variant_id)]
decision <- merge(csum,sub_all,by=c("endpoint_id","endpoint_label","variant_id"),all.x=TRUE)
decision[,review_flag:=fcase(
  median_delta_AUROC>=.005 | median_delta_AUPRC>=.010,
  "POTENTIALLY MEANINGFUL PREDICTIVE GAIN - INVESTIGATE",
  abs(median_delta_AUROC)<.002 & abs(median_delta_AUPRC)<.005 &
    median_subgroup_delta_abs_calibration_intercept < -.05,
  "OVERALL NEUTRAL BUT SUBGROUP CALIBRATION MAY IMPROVE",
  abs(median_delta_AUROC)<.002 & abs(median_delta_AUPRC)<.005,
  "NEGLIGIBLE INCREMENTAL PREDICTIVE VALUE",
  median_delta_AUROC < -0.003 | median_delta_AUPRC < -0.010,
  "PERFORMANCE WORSE THAN REFERENCE",
  default="MIXED - REVIEW FOLD/TARGET PATTERN")]
decision[,caveat:="Heuristic triage only; inclusion requires clinical/equity/temporal-stability review."]
fwrite(decision,file.path(out_dir,"18_DEMOGRAPHIC_AUGMENTATION_DECISION_SUMMARY.csv"))

writeLines(c(
  "TBI-TRACT DEMOGRAPHIC AUGMENTATION STRESS TEST COMPLETE",
  "",
  "Variants: current reference; +race; +ethnicity; +race+ethnicity; +payer; +race+ethnicity+payer.",
  "Same rolling temporal folds as script 18.",
  "Structural hyperparameters fixed to script-18 fold/endpoint reference architectures.",
  "Each augmentation selected its own boosting rounds on the pre-test tune year.",
  "",
  "Review first:",
  "07_CLASSIFICATION_CLASS_DELTAS_VS_REFERENCE.csv",
  "09_CLASSIFICATION_SUBGROUP_DELTAS_VS_REFERENCE.csv",
  "10_CLASSIFICATION_TEMPORAL_SUMMARY.csv",
  "11_SUBGROUP_CALIBRATION_TEMPORAL_SUMMARY.csv",
  "15_DURATION_DELTAS_VS_REFERENCE.csv",
  "16_DURATION_SUBGROUP_DELTAS_VS_REFERENCE.csv",
  "17_DURATION_TEMPORAL_SUMMARY.csv",
  "18_DEMOGRAPHIC_AUGMENTATION_DECISION_SUMMARY.csv",
  "",
  "If an augmented variant materially wins, fully retune that candidate before changing the bedside model."
),file.path(out_dir,"DEMOGRAPHIC_AUGMENTATION_SUMMARY.txt"))

cat("\nDONE. Output:",out_dir,"\n")
print(decision[order(endpoint_label,variant_id)])
