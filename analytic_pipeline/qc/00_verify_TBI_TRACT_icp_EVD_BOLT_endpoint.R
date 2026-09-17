# =============================================================================
# 00_verify_TBI_TRACT_icp_EVD_BOLT_endpoint.R
# Verify that the frozen TBI-TRACT ICP endpoint is exactly EVD OR parenchymal
# ICP monitor/BOLT. Brain tissue oxygen and jugular venous bulb monitoring are
# not part of this endpoint. Fits no models and changes no data.
# =============================================================================

rm(list = ls()); gc()
if (!requireNamespace("data.table", quietly=TRUE)) stop("Package data.table is required.", call.=FALSE)
suppressPackageStartupMessages(library(data.table))

config_candidates <- c(file.path(getwd(),"R","00_config.R"),"R/00_config.R")
config_file <- config_candidates[file.exists(config_candidates)][1L]
if(length(config_file)==0L||is.na(config_file)) stop("Could not find R/00_config.R.",call.=FALSE)
source(config_file)

safe_num <- function(x) suppressWarnings(as.numeric(as.character(x)))
normalize_id <- function(x) trimws(as.character(x))
first_existing <- function(x){x<-unique(x);x<-x[!is.na(x)&nzchar(x)];hit<-x[file.exists(x)];if(length(hit)==0L)NA_character_ else hit[1L]}

methods_dir <- file.path(output_dir,"METHODS_COMPLETION_TBI_TRACT")
dataset_file <- file.path(methods_dir,"frozen_methods_dataset_retained_2020_2024.rds")
if(!file.exists(dataset_file)) stop("Frozen methods dataset not found: ",dataset_file,call.=FALSE)

candidate_neuro_files <- unique(c(
 file.path(project_dir,"neurosurgical_resource_endpoints_2020_2024","04_neurosurgical_resource_endpoint_flags.csv"),
 file.path(data_raw_dir,"04_neurosurgical_resource_endpoint_flags.csv"),
 file.path("C:/Users/garin/OneDrive/OHSU/Research/TQIP/Griffith_G-selected","neurosurgical_resource_endpoints_2020_2024","04_neurosurgical_resource_endpoint_flags.csv"),
 file.path("C:/OneDrive/garingriffith/OneDrive/OHSU/Research/TQIP/Griffith_G-selected","neurosurgical_resource_endpoints_2020_2024","04_neurosurgical_resource_endpoint_flags.csv")))
recursive_hits <- tryCatch(list.files(project_dir,pattern="^04_neurosurgical_resource_endpoint_flags\\.csv$",recursive=TRUE,full.names=TRUE,ignore.case=TRUE),error=function(e)character())
neuro_file <- first_existing(unique(c(candidate_neuro_files,recursive_hits)))
if(is.na(neuro_file)) stop("Could not find 04_neurosurgical_resource_endpoint_flags.csv.",call.=FALSE)

frozen <- as.data.table(readRDS(dataset_file))
needed_frozen <- c("admission_year","inc_key","icp_pressure_monitor_final")
if(!all(needed_frozen%in%names(frozen)))stop("Frozen dataset missing required fields: ",paste(setdiff(needed_frozen,names(frozen)),collapse=", "),call.=FALSE)
nf <- fread(neuro_file,integer64="character")
needed_nf <- c("year","INC_KEY","icp_evd","icp_parenchymal_bolt")
if(!all(needed_nf%in%names(nf)))stop("Neurosurgical endpoint file missing required EVD/BOLT fields: ",paste(setdiff(needed_nf,names(nf)),collapse=", "),call.=FALSE)

nf[,`:=`(admission_year=as.integer(year),inc_key=normalize_id(INC_KEY))]
nf <- unique(nf[,.(admission_year,inc_key,icp_evd,icp_parenchymal_bolt)],by=c("admission_year","inc_key"))
check <- merge(frozen[,.(admission_year,inc_key=normalize_id(inc_key),icp_pressure_monitor_final)],nf,by=c("admission_year","inc_key"),all.x=TRUE,sort=FALSE)
evd<-safe_num(check$icp_evd);bolt<-safe_num(check$icp_parenchymal_bolt)
check[,icp_expected_evd_bolt_only:=fifelse((!is.na(evd)&evd==1)|(!is.na(bolt)&bolt==1),1L,fifelse((!is.na(evd)&evd==0)&(!is.na(bolt)&bolt==0),0L,NA_integer_))]
check[,exact_match:=(is.na(icp_pressure_monitor_final)&is.na(icp_expected_evd_bolt_only))|(!is.na(icp_pressure_monitor_final)&!is.na(icp_expected_evd_bolt_only)&icp_pressure_monitor_final==icp_expected_evd_bolt_only)]

qc_by_year <- check[,.(N=.N,N_frozen_positive=sum(icp_pressure_monitor_final==1L,na.rm=TRUE),N_expected_EVD_or_BOLT=sum(icp_expected_evd_bolt_only==1L,na.rm=TRUE),N_missing_frozen=sum(is.na(icp_pressure_monitor_final)),N_missing_expected=sum(is.na(icp_expected_evd_bolt_only)),N_mismatch=sum(!exact_match,na.rm=TRUE)),by=admission_year][order(admission_year)]
print(qc_by_year)
total_mismatch <- sum(!check$exact_match,na.rm=TRUE)
if(total_mismatch>0L){mismatch_file<-file.path(methods_dir,"12_ICP_EVD_BOLT_ENDPOINT_MISMATCH_QC.csv");fwrite(check[!exact_match],mismatch_file);stop("FAIL: frozen ICP endpoint differs from EVD-or-BOLT definition in ",total_mismatch," rows. Review ",mismatch_file,call.=FALSE)}
qc_file<-file.path(methods_dir,"12_ICP_EVD_BOLT_ENDPOINT_QC.csv");fwrite(qc_by_year,qc_file)
cat("\nPASS: icp_pressure_monitor_final is exactly EVD OR parenchymal ICP BOLT.\nBrain tissue oxygen and jugular venous bulb monitoring are not included.\nQC written to: ",qc_file,"\n",sep="")
