# =============================================================================
# 30_build_TBI_TRACT_reporting_QC_outputs.R
# Publication-release reporting/QC completion for TBI-TRACT.
#
# This script DOES NOT refit or retune any locked XGBoost model. It reconciles
# cohort flow, outcome observation, predictor missingness, deployment metadata,
# and forward-temporal reporting terminology from already-finalized artifacts.
# =============================================================================

rm(list = ls()); gc()
if (!requireNamespace("data.table", quietly=TRUE)) stop("Package data.table is required.", call.=FALSE)
suppressPackageStartupMessages(library(data.table))

cfg <- c(file.path(getwd(),"R","00_config.R"),"R/00_config.R")
cfg <- cfg[file.exists(cfg)][1L]
if (!length(cfg) || is.na(cfg)) stop("Could not find R/00_config.R.", call.=FALSE)
source(cfg)

methods_dir <- file.path(output_dir,"METHODS_COMPLETION_TBI_TRACT")
manuscript_dir <- file.path(output_dir,"TBI_TRACT_MANUSCRIPT_OUTPUT_DATA")
deploy_dir <- file.path(output_dir,"TBI_TRACT_FINAL_DEPLOYMENT_CV")
sens_dir <- file.path(output_dir,"TBI_TRACT_FINAL_SENSITIVITY_SUITE")
class_dir <- file.path(output_dir,"TBI_TRACT_PRAGMATIC_FINALIST_RETUNE")
report_dir <- file.path(output_dir,"TBI_TRACT_REPORTING_QC_COMPLETION")
dir.create(report_dir,recursive=TRUE,showWarnings=FALSE)

needed <- c(
  dataset=file.path(methods_dir,"frozen_methods_dataset_retained_2020_2024.rds"),
  types=file.path(methods_dir,"11_FROZEN_PREDICTOR_TYPES.csv"),
  flow=file.path(methods_dir,"01_DIRECT_PRESENTATION_OBSERVATION_STATUS_BY_YEAR.csv")
)
if(!all(file.exists(needed))) stop("Missing prerequisite file(s):\n",paste(needed[!file.exists(needed)],collapse="\n"),call.=FALSE)

dt <- as.data.table(readRDS(needed[["dataset"]])); types<-fread(needed[["types"]]); flow<-fread(needed[["flow"]])
safe_num <- function(x) suppressWarnings(as.numeric(as.character(x)))

# ---- Complete cohort flow ----------------------------------------------------
AGE_ELIGIBLE_TOTAL <- 1161027L; EXPECTED_DIRECT_TOTAL <- 831116L; EXPECTED_RETAINED_TOTAL <- 755880L
flow[,status_label:=fcase(
 observation_status=="Retained complete trajectory","Retained complete trajectory",
 observation_status=="Future transfer-out","Future transfer-out",
 observation_status=="Future AMA","Future AMA",
 observation_status=="Future ED-other/institutional/custody","Future institutional / custody / other",
 default=observation_status)]
ft<-flow[,.(N=sum(N,na.rm=TRUE)),by=status_label]
get_n<-function(x){z<-ft[status_label==x,N];if(length(z))as.integer(z[1L]) else 0L}
direct_total<-sum(ft$N);retained_total<-get_n("Retained complete trajectory");transfer_out<-get_n("Future transfer-out");ama<-get_n("Future AMA");other<-get_n("Future institutional / custody / other")
if(direct_total!=EXPECTED_DIRECT_TOTAL)stop("Direct-presentation count changed.",call.=FALSE)
if(retained_total!=EXPECTED_RETAINED_TOTAL)stop("Retained cohort count changed.",call.=FALSE)
if(transfer_out+ama+other != direct_total-retained_total)stop("Post-direct exclusions do not reconcile.",call.=FALSE)
if(nrow(dt)!=retained_total)stop("Frozen retained dataset does not reconcile to cohort flow.",call.=FALSE)
pre_direct_excluded<-AGE_ELIGIBLE_TOTAL-direct_total
cohort_flow<-data.table(order=1:8,record_type=c("included","excluded","included","excluded","excluded","excluded","included","QC"),stage=c("Age-eligible TBI encounters with qualifying S06 diagnosis","Transfer-in or unknown presentation status excluded","Direct presentations","Subsequent transfer-out","Subsequent discharge against medical advice","Subsequent institutional/custody/other trajectory loss","Retained complete index-hospital trajectory","All post-direct exclusions"),N=c(AGE_ELIGIBLE_TOTAL,pre_direct_excluded,direct_total,transfer_out,ama,other,retained_total,direct_total-retained_total),denominator=c(NA_integer_,AGE_ELIGIBLE_TOTAL,AGE_ELIGIBLE_TOTAL,direct_total,direct_total,direct_total,direct_total,direct_total))
cohort_flow[,percent_of_denominator:=fifelse(is.na(denominator)|denominator==0,NA_real_,100*N/denominator)]
fwrite(cohort_flow,file.path(report_dir,"01_COMPLETE_COHORT_FLOW.csv"))

# ---- Outcome observation ----------------------------------------------------
dt[,hospital_days:=safe_num(hospital_days)];dt[,icu_days:=safe_num(icu_days)];dt[,vent_days:=safe_num(vent_days)]
dt[hospital_days<0,hospital_days:=NA_real_];dt[icu_days<0,icu_days:=NA_real_];dt[vent_days<0,vent_days:=NA_real_]
dt[,hlos_trajectory_reporting:=fcase(is.na(hospital_days),NA_character_,hospital_days<=7,"Hospital LOS <=7 days",hospital_days<=27,"Hospital LOS 8-27 days",hospital_days>=28,"Hospital LOS >=28 days")]
dt[,icu_trajectory_reporting:=fcase(is.na(icu_days),NA_character_,icu_days<=0,"No ICU",icu_days<=7,"ICU 1-7 days",icu_days>=8,"ICU >=8 days")]
dt[,ventilation_trajectory_reporting:=fcase(is.na(vent_days),NA_character_,vent_days<=0,"No ventilation",vent_days<=7,"Ventilation 1-7 days",vent_days>=8,"Ventilation >=8 days")]
binobs<-function(x){z<-safe_num(x);!is.na(z)&z%in%c(0,1)}
outcome_obs<-rbindlist(list(
 data.table(outcome="Disposition",observed_N=sum(!is.na(dt$discharge_3cat_final))),
 data.table(outcome="Hospital LOS trajectory",observed_N=sum(!is.na(dt$hlos_trajectory_reporting))),
 data.table(outcome="ICU trajectory",observed_N=sum(!is.na(dt$icu_trajectory_reporting))),
 data.table(outcome="Mechanical ventilation trajectory",observed_N=sum(!is.na(dt$ventilation_trajectory_reporting))),
 data.table(outcome="Invasive ICP monitoring (EVD/BOLT)",observed_N=sum(binobs(dt$icp_pressure_monitor_final))),
 data.table(outcome="Craniotomy/craniectomy",observed_N=sum(binobs(dt$craniotomy_craniectomy_final)))))
outcome_obs[,`:=`(retained_cohort_N=nrow(dt),missing_outcome_N=nrow(dt)-observed_N,observed_percent=100*observed_N/nrow(dt))]
fwrite(outcome_obs,file.path(report_dir,"02_OUTCOME_OBSERVATION_COUNTS.csv"))

# ---- Predictor missingness --------------------------------------------------
num<-types[category=="numeric",predictor];catv<-types[category=="categorical",predictor]
unknown_coded<-function(x){z<-tolower(trimws(as.character(x)));is.na(x)|z%in%c("","unknown","unknown/not recorded","unknown / not recorded","__unknown__","na","n/a")}
miss_rows<-list()
for(v in unique(c(num,catv,"race_clean","ethnicity_clean","insurance_clean"))){
 if(!v%in%names(dt))next
 if(v%in%num){m<-is.na(safe_num(dt[[v]]))}else{m<-unknown_coded(dt[[v]])}
 miss_rows[[length(miss_rows)+1L]]<-rbindlist(lapply(c(2020:2024,NA_integer_),function(yy){idx<-if(is.na(yy))rep(TRUE,nrow(dt)) else dt$admission_year==yy;data.table(predictor=v,year=if(is.na(yy))"Overall" else as.character(yy),N=sum(idx),missing_N=sum(m[idx],na.rm=TRUE),missing_percent=100*mean(m[idx],na.rm=TRUE))}))
}
missing_long<-rbindlist(miss_rows,fill=TRUE);fwrite(missing_long,file.path(report_dir,"03_PREDICTOR_MISSINGNESS_LONG.csv"))
missing_wide<-dcast(missing_long,predictor~year,value.var="missing_percent");fwrite(missing_wide,file.path(report_dir,"04_PREDICTOR_MISSINGNESS_WIDE.csv"))

# ---- Deployment/reproducibility manifest -----------------------------------
manifest_file<-file.path(deploy_dir,"04_FINAL_DEPLOYMENT_MODEL_MANIFEST.csv")
rounds_file<-file.path(deploy_dir,"03_FULL_DEVELOPMENT_5FOLD_CV_ROUNDS.csv")
if(file.exists(manifest_file)){
 manifest<-fread(manifest_file)
 if(file.exists(rounds_file)){
   rounds<-fread(rounds_file)
   if("selected_rounds_5fold_cv"%in%names(rounds)){
    rr<-rounds[,.(endpoint_id,selected_rounds_internal_5fold_cv=selected_rounds_5fold_cv)]
    manifest<-merge(manifest,rr,by="endpoint_id",all.x=TRUE)
   }
 }
 manifest[,`:=`(
  deployment_training_years="2020-2024",
  structural_selection_rule=paste("Across TEST_2022, TEST_2023, and TEST_2024 rolling-origin temporal tuning folds, select the configuration with the lowest mean relative excess tuning loss; ties use lower mean rank, lower worst rank, shallower depth, then larger lambda."),
  boosting_round_selection_rule=paste("With structural hyperparameters locked, select deployment boosting rounds by year/outcome-stratified 5-fold internal cross-validation in the complete 2020-2024 development cohort; these folds are not manuscript-facing validation."),
  encoding_rule=paste("Numeric missing values are retained for native XGBoost handling. Categorical blank/missing values map to __UNKNOWN__; unseen fitted-time categories map to __OTHER__."),
  calibration_rule=paste("Binary/one-vs-rest flexible calibration uses a binomial GLM with ns(logit predicted probability, df=4), evaluated over the 0.5th-99.5th percentile prediction range with pointwise 95% confidence intervals."))]
 fwrite(manifest,file.path(report_dir,"09_MODEL_REPRODUCIBILITY_MANIFEST.csv"))
 arts<-unique(na.omit(c(as.character(manifest$model_path),as.character(manifest$encoder_path))));arts<-arts[file.exists(arts)]
 if(length(arts))fwrite(data.table(path=arts,md5=unname(tools::md5sum(arts))),file.path(report_dir,"10_MODEL_ARTIFACT_CHECKSUMS.csv"))
}

writeLines(c(
 "TBI-TRACT REPRODUCIBILITY METHODS","",
 "STRUCTURAL HYPERPARAMETER SELECTION",
 "Each candidate configuration was evaluated separately across the 2022, 2023, and 2024 rolling-origin forward-temporal tuning folds. The selected configuration minimized mean relative excess tuning loss; ties were resolved by lower mean rank, lower worst rank, shallower tree depth, then larger L2 regularization (lambda).","",
 "BOOSTING ROUNDS",
 "After structural hyperparameters were locked, year/outcome-stratified 5-fold internal cross-validation in the full 2020-2024 deployment-development cohort selected boosting rounds. These folds were not reported as validation.","",
 "ENCODING",
 "XGBoost retained numeric missing values natively. Categorical missing/blank values mapped to __UNKNOWN__; unseen levels mapped to __OTHER__. The Ridge comparator used development-only median imputation for numeric values.","",
 "TEMPORAL TERMINOLOGY",
 "Manuscript-facing 2022-2024 predictions are rolling-origin forward-temporal evaluation predictions, not conventional out-of-fold cross-validation predictions."),file.path(report_dir,"11_REPRODUCIBILITY_METHODS.txt"))
writeLines(capture.output(sessionInfo()),file.path(report_dir,"12_R_SESSION_INFO.txt"))

# ---- Relabel pooled metric file without changing values ---------------------
pooled_old<-file.path(manuscript_dir,"02_CLASSIFICATION_POOLED_OOF_METRICS.csv")
if(file.exists(pooled_old)){
 p<-fread(pooled_old);if("years"%in%names(p))p[,years:="2022-2024 rolling-origin forward-temporal evaluation"]
 fwrite(p,file.path(report_dir,"13_CLASSIFICATION_POOLED_TEMPORAL_EVALUATION_METRICS.csv"))
}

# ---- Full subgroup values ---------------------------------------------------
subgroup_file<-file.path(sens_dir,"03_FINAL_CLASSIFICATION_SUBGROUP_ROBUSTNESS.csv")
year_file<-file.path(manuscript_dir,"01_CLASSIFICATION_YEAR_SPECIFIC_METRICS.csv")
if(file.exists(subgroup_file)&&file.exists(year_file)){
 s<-fread(subgroup_file)[test_year==2024L];y<-fread(year_file)[year==2024L,.(endpoint_id,target,overall_N=N,overall_events=events,overall_prevalence=prevalence,overall_AUROC=AUROC,overall_AUPRC=AUPRC,overall_Brier=Brier,overall_calibration_intercept=calibration_intercept,overall_calibration_slope=calibration_slope)]
 s<-merge(s,y,by=c("endpoint_id","target"),all.x=TRUE);if(all(c("AUROC","overall_AUROC")%in%names(s)))s[,delta_AUROC_vs_overall:=AUROC-overall_AUROC];if(all(c("AUPRC","overall_AUPRC")%in%names(s)))s[,delta_AUPRC_vs_overall:=AUPRC-overall_AUPRC]
 fwrite(s,file.path(report_dir,"14_2024_SUBGROUP_FULL_ABSOLUTE_PERFORMANCE.csv"))
}
social_file<-file.path(class_dir,"10_SYSTEM_MEDIATED_SUBGROUP_DELTAS_VS_PRAGMATIC.csv")
if(file.exists(social_file)){
 s<-fread(social_file)[test_year==2024L & variant_id=="PRAGMATIC_PLUS_RACE_ETHNICITY_PAYER"]
 fwrite(s,file.path(report_dir,"15_2024_RACE_ETHNICITY_PAYER_FULL_SUBGROUP_PERFORMANCE.csv"))
}

writeLines(c("TBI-TRACT REPORTING/QC COMPLETION FINISHED","",paste0("Age-eligible cohort: ",format(AGE_ELIGIBLE_TOTAL,big.mark=",")),paste0("Direct presentations: ",format(direct_total,big.mark=",")),paste0("Retained cohort: ",format(retained_total,big.mark=",")),"","Primary XGBoost models were NOT refit or retuned."),file.path(report_dir,"REPORTING_QC_SUMMARY.txt"))
cat("\nTBI-TRACT reporting/QC completion finished. Output: ",report_dir,"\n",sep="")
