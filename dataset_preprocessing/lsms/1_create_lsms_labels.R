source("utils.R")
source("process_ethiopia.R")
source("process_malawi.R")
source("process_uganda.R")

library(haven)
library(reshape2)
library(tidyverse)

data_dir = "."

# =============================================================================
# STAGE 1: Process household-level data
# ------------------------------------
# Only keep households for which we have (lat,lon) geocoordinates and which
# exist in all surveys of that country. In this stage, we keep households
# that don't have complete asset information. This is to allow for flexibility
# in how we define various indices later (e.g., asset index). Some indices
# might only use some of the columns, in which case it's OK for other columns
# to have NAs.
# =============================================================================

#' Save clean dataframe to RDS file in {data_dir}/clean/
#'
#' @param df Dataframe to save
#' @param name Name of RDS file (without .RDS extension)
save_clean_rds = function(df, name) {
  clean_dir = file.path(data_dir, "clean")
  if (!dir.exists(clean_dir)) {
    dir.create(clean_dir)
  }
  filename = str_glue("{name}.RDS")
  rds_path = file.path(clean_dir, filename)
  print(str_glue("Writing {filename} to ", rds_path))

  write_rds(df, rds_path)
}


loc_cols = c("lat", "lon", "year", "country")

# ETHIOPIA
extract_ethiopia(data_dir)
ethiopia = process_ethiopia(data_dir)
save_clean_rds(ethiopia, "ethiopia")
all_locs = ethiopia[, loc_cols]

# MALAWI
extract_malawi(data_dir)
malawi = process_malawi(data_dir)
save_clean_rds(malawi, "malawi")
all_locs = rbind(all_locs, malawi[, loc_cols])

# NIGERIA

# TANZANIA

# UGANDA
extract_uganda(data_dir)
uganda = process_uganda(data_dir)
save_clean_rds(uganda, "uganda")
all_locs = rbind(all_locs, uganda[, loc_cols])