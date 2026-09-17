# TBI-TRACT local configuration template
#
# Copy this file to R/00_config.R and edit the paths for your environment.
# R/00_config.R should remain local and is excluded from version control.

project_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)

# Local directory containing the ACS TQIP/TQP source files or harmonized
# warehouse used by the analysis. These data are not distributed in this repo.
warehouse_dir <- "PATH/TO/TQIP_HARMONIZED_WAREHOUSE"
data_raw_dir <- "PATH/TO/LOCAL_TQIP_SOURCE_FILES"

# Project-local directories for derived data and analysis outputs.
data_dir <- file.path(project_dir, "data")
data_out_dir <- file.path(project_dir, "data", "derived")
output_dir <- file.path(project_dir, "output")

dir.create(data_out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
