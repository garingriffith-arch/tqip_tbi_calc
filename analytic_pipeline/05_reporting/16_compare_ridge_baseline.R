# =============================================================================
# 31_build_TBI_TRACT_fair_ridge_comparator_PARALLEL_AUC_FIXED.R
#
# PURPOSE
#   Rebuild the Ridge-vs-XGBoost comparator using:
#     - identical 2020-2023 development / 2024 temporal evaluation split
#     - identical endpoint definitions
#     - identical final predictor policies
#     - canonical LOCKED XGBoost 2024 performance from script 27
#
# IMPORTANT
#   This script fits ONLY the Ridge comparator.
#   It does NOT refit or modify the primary XGBoost models.
# =============================================================================

rm(list = ls())
gc()

required <- c("data.table","glmnet","Matrix","doParallel","foreach")
missing_pkgs <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0L) stop("Missing package(s): ", paste(missing_pkgs, collapse = ", "), call. = FALSE)

suppressPackageStartupMessages({
  library(data.table); library(glmnet); library(Matrix); library(doParallel); library(foreach)
})

config_candidates <- c(file.path(getwd(),"R","00_config.R"),"R/00_config.R")
config_file <- config_candidates[file.exists(config_candidates)][1L]
if (length(config_file)==0L || is.na(config_file)) stop("Could not find R/00_config.R.", call.=FALSE)
source(config_file)

methods_dir <- file.path(output_dir,"METHODS_COMPLETION_TBI_TRACT")
manuscript_dir <- file.path(output_dir,"TBI_TRACT_MANUSCRIPT_OUTPUT_DATA")
report_dir <- file.path(output_dir,"TBI_TRACT_REPORTING_QC_COMPLETION")
dir.create(report_dir,recursive=TRUE,showWarnings=FALSE)

dataset_file <- file.path(methods_dir,"frozen_methods_dataset_retained_2020_2024.rds")
types_file <- file.path(methods_dir,"11_FROZEN_PREDICTOR_TYPES.csv")
xgb_file <- file.path(manuscript_dir,"03_2024_HEADLINE_CLASSIFICATION_BOOTSTRAP_CI.csv")
needed <- c(dataset_file,types_file,xgb_file)
if(!all(file.exists(needed))) stop("Missing prerequisite file(s):\n",paste(needed[!file.exists(needed)],collapse="\n"),call.=FALSE)

dt <- as.data.table(readRDS(dataset_file)); types <- fread(types_file)
safe_num <- function(x) suppressWarnings(as.numeric(as.character(x)))
normalize_character <- function(x){x<-trimws(as.character(x));x[is.na(x)|x==""|x%in%c("NA","NaN","<NA>")]<-"__UNKNOWN__";x}
clamp_prob <- function(p,eps=1e-7) pmin(pmax(as.numeric(p),eps),1-eps)

fast_auc <- function(y,p){
  keep<-!is.na(y)&!is.na(p)&is.finite(p); y<-as.integer(y[keep]); p<-as.numeric(p[keep])
  n1<-as.double(sum(y==1L)); n0<-as.double(sum(y==0L)); if(n1==0L||n0==0L)return(NA_real_)
  r<-rank(p,ties.method="average"); (sum(r[y==1L])-n1*(n1+1)/2)/(n1*n0)
}
fast_auprc <- function(y,p){
  keep<-!is.na(y)&!is.na(p)&is.finite(p); y<-as.integer(y[keep]); p<-as.numeric(p[keep]); n_pos<-sum(y==1L); if(n_pos==0L)return(NA_real_)
  ord<-order(p,decreasing=TRUE); yy<-y[ord]; tp<-cumsum(yy==1L); fp<-cumsum(yy==0L); precision<-tp/(tp+fp); recall<-tp/n_pos; recall_previous<-c(0,head(recall,-1L)); sum(precision*(recall-recall_previous),na.rm=TRUE)
}
binary_calibration <- function(y,p){
  keep<-!is.na(y)&!is.na(p)&is.finite(p); y<-as.integer(y[keep]); p<-clamp_prob(p[keep]); if(length(unique(y))<2L)return(c(intercept=NA_real_,slope=NA_real_)); lp<-qlogis(p)
  fit_i<-tryCatch(glm(y~1,family=binomial(),offset=lp),error=function(e)NULL); fit_s<-tryCatch(glm(y~lp,family=binomial()),error=function(e)NULL)
  c(intercept=if(is.null(fit_i))NA_real_ else unname(coef(fit_i)[1L]),slope=if(is.null(fit_s))NA_real_ else unname(coef(fit_s)["lp"]))
}
binary_metrics <- function(y,p){
  y<-as.integer(y);p<-as.numeric(p);cal<-binary_calibration(y,p); data.table(N=length(y),events=sum(y==1L),prevalence=mean(y==1L),AUROC=fast_auc(y,p),AUPRC=fast_auprc(y,p),AUPRC_to_prevalence=fast_auprc(y,p)/mean(y==1L),Brier=mean((p-y)^2),calibration_intercept=unname(cal["intercept"]),calibration_slope=unname(cal["slope"]))
}

numeric_predictors<-types[category=="numeric",predictor]; categorical_predictors<-types[category=="categorical",predictor]
current_predictors<-c(numeric_predictors,categorical_predictors)
pragmatic_clinical<-setdiff(current_predictors,c("helmet_use_recovered","respiratoryassistance_clean"))
social_vars<-c("race_clean","ethnicity_clean","insurance_clean")
system_context_predictors<-unique(c(pragmatic_clinical,social_vars))

dt[,hospital_days:=safe_num(hospital_days)];dt[,icu_days:=safe_num(icu_days)];dt[,vent_days:=safe_num(vent_days)]
dt[hospital_days<0,hospital_days:=NA_real_];dt[icu_days<0,icu_days:=NA_real_];dt[vent_days<0,vent_days:=NA_real_]
DISCHARGE_LEVELS<-c("Home/home health","Post-acute facility","Death/hospice");ICU_LEVELS<-c("No ICU","ICU 1-7 days","ICU >=8 days");VENT_LEVELS<-c("No ventilation","Ventilation 1-7 days","Ventilation >=8 days");HLOS_LEVELS<-c("Hospital LOS <=7 days","Hospital LOS 8-27 days","Hospital LOS >=28 days")
dt[,hlos_trajectory_final:=fcase(is.na(hospital_days),NA_character_,hospital_days<=7,HLOS_LEVELS[1L],hospital_days<=27,HLOS_LEVELS[2L],hospital_days>=28,HLOS_LEVELS[3L],default=NA_character_)]
icu_rebuilt<-fcase(is.na(dt$icu_days),NA_character_,dt$icu_days<=0,ICU_LEVELS[1L],dt$icu_days<=7,ICU_LEVELS[2L],dt$icu_days>=8,ICU_LEVELS[3L],default=NA_character_)
if("icu_trajectory_final"%in%names(dt)){mismatch<-sum(!is.na(dt$icu_trajectory_final)&!is.na(icu_rebuilt)&as.character(dt$icu_trajectory_final)!=icu_rebuilt);if(mismatch>0L)stop("ICU trajectory reconstruction mismatch: ",mismatch," rows.",call.=FALSE)}
dt[,icu_trajectory_final:=icu_rebuilt]
dt[,ventilation_trajectory_final:=fcase(is.na(vent_days),NA_character_,vent_days<=0,VENT_LEVELS[1L],vent_days<=7,VENT_LEVELS[2L],vent_days>=8,VENT_LEVELS[3L],default=NA_character_)]

class_specs<-list(
 discharge_3cat_final=list(endpoint_label="Disposition",type="multiclass",levels=DISCHARGE_LEVELS,predictors=system_context_predictors),
 hlos_trajectory_final=list(endpoint_label="Hospital LOS trajectory",type="multiclass",levels=HLOS_LEVELS,predictors=system_context_predictors),
 icu_trajectory_final=list(endpoint_label="ICU trajectory",type="multiclass",levels=ICU_LEVELS,predictors=pragmatic_clinical),
 ventilation_trajectory_final=list(endpoint_label="Mechanical ventilation trajectory",type="multiclass",levels=VENT_LEVELS,predictors=pragmatic_clinical),
 icp_pressure_monitor_final=list(endpoint_label="Invasive ICP monitoring",type="binary",levels=NULL,predictors=pragmatic_clinical),
 craniotomy_craniectomy_final=list(endpoint_label="Craniotomy/craniectomy",type="binary",levels=NULL,predictors=pragmatic_clinical))
headline_targets<-data.table(endpoint_id=c("discharge_3cat_final","discharge_3cat_final","icu_trajectory_final","icu_trajectory_final","ventilation_trajectory_final","ventilation_trajectory_final","hlos_trajectory_final","icp_pressure_monitor_final","craniotomy_craniectomy_final"),target=c("Post-acute facility","Death/hospice","Any ICU","ICU >=8 days","Any ventilation","Ventilation >=8 days","Hospital LOS >=28 days","Invasive ICP monitoring","Craniotomy/craniectomy"))

fit_ridge_encoder<-function(d,predictors){
 num_vars<-intersect(numeric_predictors,predictors);cat_vars<-setdiff(predictors,num_vars);missing_vars<-setdiff(predictors,names(d));if(length(missing_vars)>0L)stop("Missing predictor(s): ",paste(missing_vars,collapse=", "),call.=FALSE)
 medians<-setNames(numeric(length(num_vars)),num_vars);for(v in num_vars){z<-safe_num(d[[v]]);med<-median(z[is.finite(z)],na.rm=TRUE);if(!is.finite(med))med<-0;medians[v]<-med}
 levels_list<-lapply(cat_vars,function(v){x<-normalize_character(d[[v]]);sort(unique(c(x,"__UNKNOWN__","__OTHER__")))});names(levels_list)<-cat_vars
 list(numeric_vars=num_vars,categorical_vars=cat_vars,medians=medians,categorical_levels=levels_list)
}
apply_ridge_encoder<-function(d,encoder){
 dd<-data.frame(row.names=seq_len(nrow(d)));for(v in encoder$numeric_vars){z<-safe_num(d[[v]]);z[!is.finite(z)]<-encoder$medians[v];dd[[v]]<-z}
 for(v in encoder$categorical_vars){x<-normalize_character(d[[v]]);lev<-encoder$categorical_levels[[v]];x[!x%in%lev]<-"__OTHER__";dd[[v]]<-factor(x,levels=lev)}
 Matrix::sparse.model.matrix(~.-1,data=dd)
}
make_balanced_folds<-function(year,strata,k=5L,seed=20260917L){set.seed(seed);d<-data.table(row_id=seq_along(year),year=as.character(year),strata=as.character(strata));d[is.na(strata)|strata=="",strata:="__UNKNOWN__"];d[,fold_id:={ord<-sample.int(.N);out<-integer(.N);out[ord]<-rep(seq_len(k),length.out=.N);out},by=.(year,strata)];d[order(row_id),fold_id]}

logical_cores<-parallel::detectCores(logical=TRUE);if(is.na(logical_cores)||logical_cores<2L)logical_cores<-32L;TARGET_CPU_THREADS<-max(1L,floor(.75*logical_cores));CV_FOLDS<-5L;CV_WORKERS<-min(CV_FOLDS,TARGET_CPU_THREADS);THREADS_PER_WORKER<-max(1L,ceiling(TARGET_CPU_THREADS/CV_WORKERS));Sys.setenv(OMP_NUM_THREADS=as.character(THREADS_PER_WORKER),MKL_NUM_THREADS=as.character(THREADS_PER_WORKER),OPENBLAS_NUM_THREADS=as.character(THREADS_PER_WORKER));cv_cluster<-parallel::makePSOCKcluster(CV_WORKERS);doParallel::registerDoParallel(cv_cluster)
cat("\nRIDGE COMPUTE BACKEND\n","Logical CPU threads detected: ",logical_cores,"\nTarget CPU threads (~75%): ",TARGET_CPU_THREADS,"\nParallel CV workers: ",CV_WORKERS,"\nThreads allowed per worker: ",THREADS_PER_WORKER,"\nGPU: not used (glmnet is CPU-based)\n\n",sep="")

ridge_rows<-list();ridge_meta<-list()
for(endpoint_id in names(class_specs)){
 cat("\n============================================================\n","RIDGE COMPARATOR: ",endpoint_id,"\n============================================================\n",sep="");endpoint_start_time<-proc.time()[["elapsed"]];spec<-class_specs[[endpoint_id]]
 if(spec$type=="binary"){y_all<-safe_num(dt[[endpoint_id]]);y_all[!y_all%in%c(0,1)]<-NA_real_;observed<-!is.na(y_all)}else{y_all<-factor(as.character(dt[[endpoint_id]]),levels=spec$levels);observed<-!is.na(y_all)}
 train_idx<-observed&dt$admission_year%in%2020:2023;test_idx<-observed&dt$admission_year==2024L;if(sum(train_idx)==0L||sum(test_idx)==0L)stop("Empty train/test set for ",endpoint_id,call.=FALSE)
 encoder<-fit_ridge_encoder(dt[train_idx],spec$predictors);x_train<-apply_ridge_encoder(dt[train_idx],encoder);x_test<-apply_ridge_encoder(dt[test_idx],encoder)
 if(!identical(colnames(x_train),colnames(x_test)))stop("Train/test ridge design matrices do not align for ",endpoint_id,call.=FALSE)
 y_train<-y_all[train_idx];y_test<-y_all[test_idx];fold_id<-make_balanced_folds(year=dt$admission_year[train_idx],strata=y_train,k=5L,seed=20260917L)
 if(spec$type=="binary"){
  cvfit<-glmnet::cv.glmnet(x=x_train,y=as.integer(y_train),family="binomial",alpha=0,type.measure="deviance",foldid=fold_id,standardize=TRUE,intercept=TRUE,parallel=TRUE)
  p_test<-as.numeric(predict(cvfit,newx=x_test,s="lambda.min",type="response"));target_name<-spec$endpoint_label;met<-binary_metrics(as.integer(y_test),p_test)
  met[,`:=`(endpoint_id=endpoint_id,endpoint_label=spec$endpoint_label,target=target_name,model="Ridge",predictor_policy=if(endpoint_id%in%c("discharge_3cat_final","hlos_trajectory_final"))"Pragmatic clinical + race + ethnicity + payer" else "Pragmatic clinical")];ridge_rows[[length(ridge_rows)+1L]]<-met
 } else {
  cvfit<-glmnet::cv.glmnet(x=x_train,y=y_train,family="multinomial",alpha=0,type.measure="deviance",foldid=fold_id,standardize=TRUE,intercept=TRUE,parallel=TRUE)
  pred_array<-predict(cvfit,newx=x_test,s="lambda.min",type="response");if(length(dim(pred_array))==3L){p_test<-pred_array[,,1L,drop=TRUE]}else{p_test<-as.matrix(pred_array)};if(is.null(colnames(p_test)))colnames(p_test)<-spec$levels
  endpoint_id_current<-endpoint_id;targets<-headline_targets[endpoint_id==endpoint_id_current,target]
  for(target_name in targets){
   if(target_name%in%spec$levels){y_binary<-as.integer(as.character(y_test)==target_name);p_binary<-as.numeric(p_test[,target_name])}else if(endpoint_id=="icu_trajectory_final"&&target_name=="Any ICU"){y_binary<-as.integer(as.character(y_test)!="No ICU");p_binary<-1-as.numeric(p_test[,"No ICU"])}else if(endpoint_id=="ventilation_trajectory_final"&&target_name=="Any ventilation"){y_binary<-as.integer(as.character(y_test)!="No ventilation");p_binary<-1-as.numeric(p_test[,"No ventilation"])}else next
   met<-binary_metrics(y_binary,p_binary);met[,`:=`(endpoint_id=endpoint_id,endpoint_label=spec$endpoint_label,target=target_name,model="Ridge",predictor_policy=if(endpoint_id%in%c("discharge_3cat_final","hlos_trajectory_final"))"Pragmatic clinical + race + ethnicity + payer" else "Pragmatic clinical")];ridge_rows[[length(ridge_rows)+1L]]<-met
  }
 }
 ridge_meta[[length(ridge_meta)+1L]]<-data.table(endpoint_id=endpoint_id,endpoint_label=spec$endpoint_label,N_train=sum(train_idx),N_test=sum(test_idx),encoded_features=ncol(x_train),lambda_min=cvfit$lambda.min,lambda_1se=cvfit$lambda.1se,alpha=0,lambda_selection="lambda.min",internal_folds=5L)
 rm(x_train,x_test,cvfit);gc();endpoint_elapsed<-proc.time()[["elapsed"]]-endpoint_start_time;cat("Completed ",endpoint_id," | elapsed seconds: ",round(endpoint_elapsed,1),"\n",sep="")
}
parallel::stopCluster(cv_cluster);foreach::registerDoSEQ()
ridge_results<-rbindlist(ridge_rows,fill=TRUE);ridge_meta<-rbindlist(ridge_meta,fill=TRUE)
ridge_results<-merge(headline_targets[,.(endpoint_id,target,target_order=.I)],ridge_results,by=c("endpoint_id","target"),all.x=TRUE);if(any(is.na(ridge_results$AUROC)))warning("One or more headline ridge targets did not produce an AUROC. Inspect the output before manuscript use.");setorder(ridge_results,target_order);if(any(!is.finite(ridge_results$AUROC)))stop("Non-finite Ridge AUROC detected after model fitting. Do not use comparator output.",call.=FALSE)
fwrite(ridge_results,file.path(report_dir,"16_2024_RIDGE_HEADLINE_PERFORMANCE.csv"));fwrite(ridge_meta,file.path(report_dir,"17_RIDGE_COMPARATOR_MODEL_METADATA.csv"))

xgb<-fread(xgb_file);xgb<-merge(headline_targets[,.(endpoint_id,target,target_order=.I)],xgb,by=c("endpoint_id","target"),all.x=TRUE);xgb[,model:="XGBoost"]
qc<-merge(ridge_results[,.(endpoint_id,target,Ridge_N=N,Ridge_events=events)],xgb[,.(endpoint_id,target,XGBoost_N=N,XGBoost_events=events)],by=c("endpoint_id","target"),all=TRUE);qc[,`:=`(N_match=Ridge_N==XGBoost_N,event_match=Ridge_events==XGBoost_events)];fwrite(qc,file.path(report_dir,"18_RIDGE_XGBOOST_COHORT_CONCORDANCE_QC.csv"));if(any(qc$N_match==FALSE|qc$event_match==FALSE,na.rm=TRUE))stop(paste("Ridge and canonical XGBoost evaluation cohorts do not match.","Review 18_RIDGE_XGBOOST_COHORT_CONCORDANCE_QC.csv before using","the comparator."),call.=FALSE)

xgb_final<-xgb[,.(endpoint_id,target,target_order,XGBoost_N=N,XGBoost_events=events,XGBoost_prevalence=prevalence,XGBoost_AUROC=AUROC,XGBoost_AUPRC=AUPRC,XGBoost_Brier=Brier,XGBoost_calibration_intercept=calibration_intercept,XGBoost_calibration_slope=calibration_slope)]
ridge_final<-ridge_results[,.(endpoint_id,target,Ridge_N=N,Ridge_events=events,Ridge_prevalence=prevalence,Ridge_AUROC=AUROC,Ridge_AUPRC=AUPRC,Ridge_Brier=Brier,Ridge_calibration_intercept=calibration_intercept,Ridge_calibration_slope=calibration_slope)]
comparison<-merge(xgb_final,ridge_final,by=c("endpoint_id","target"),all=TRUE);comparison[,`:=`(delta_AUROC_Ridge_minus_XGBoost=Ridge_AUROC-XGBoost_AUROC,delta_AUPRC_Ridge_minus_XGBoost=Ridge_AUPRC-XGBoost_AUPRC,delta_Brier_Ridge_minus_XGBoost=Ridge_Brier-XGBoost_Brier)];setorder(comparison,target_order);fwrite(comparison,file.path(report_dir,"19_2024_RIDGE_VS_XGBOOST_CANONICAL_COMPARATOR.csv"))
writeLines(c("TBI-TRACT RIDGE COMPARATOR","","Development cohort: 2020-2023.","Temporal evaluation cohort: 2024.","",paste("The Ridge comparator used the same endpoint-specific predictor policy","as the final XGBoost model."),paste("Numeric missing values were median-imputed using development data only.","Categorical missingness was represented explicitly as __UNKNOWN__, with","previously unseen categories mapped to __OTHER__."),paste("Categorical predictors were one-hot encoded. glmnet Ridge models used","alpha=0, standardized predictors, and lambda.min selected by 5-fold","year/outcome-balanced internal cross-validation in the 2020-2023","development cohort."),paste("The XGBoost values in the final comparator were imported directly from","03_2024_HEADLINE_CLASSIFICATION_BOOTSTRAP_CI.csv and therefore correspond","to the canonical locked 2024 manuscript-facing XGBoost results.")),file.path(report_dir,"20_RIDGE_COMPARATOR_METHODS.txt"))
cat("\n============================================================\n","FAIR RIDGE-vs-XGBOOST COMPARATOR COMPLETE\n","Output: ",report_dir,"\n============================================================\n",sep="")
