# Malawi LSMS Surveys Overview
# ----------------------------
# In 1997, Malawi conducted its first Integrated Household Survey (IHS).
# The IHS series is not entirely a panel (i.e., some households are different
# between survey rounds). Starting with the 2010 IHS (aka. IHS3), Malawi began
# a panel survey known as the Integrated Household Panel Survey (IHPS) series.
#
# To date (2021), Malawi has completed 5 rounds of IHS and 3 rounds of IHPS:
#   1997-98: IHS1
#   2004-05: IHS2
#   2010-11: IHS3, includes IHPS 2010 which is comprised of 204 baseline EAs
#   2013   : IHPS 2013, which tracked 204 (all) baseline EAs from IHPS 2010
#   2016-17: IHS4, includes IHPS 2016, which tracked 102 (half) of baseline EAs
#            from IHPS 2010
#   2019-20: IHS5
# We use IHPS 2010 and IHPS 2016, so we only look at the 102 EAs tracked across
# both surveys.


#' Extracts the necessary data files for LSMS Malawi surveys.
#'
#' Depends on extract_and_rename() function in utils.R.
#' Assumes that the following ZIP files have already been downloaded
#' into {data_dir}/raw/Malawi:
#' - MWI_2010-2013-2016_IHPS_v03_M_CSV.zip
#'
#' @param data_dir Path to data folder, with "raw" and "clean" subfolders.
#'   Should not end in a "/".
#' @return Nothing.
#' @export
extract_malawi = function(data_dir) {
  raw_mw = file.path(data_dir, "raw", "Malawi")
  zip_path = file.path(raw_mw, "MWI_2010-2013-2016_IHPS_v03_M_CSV.zip")
  extract_dir = file.path(raw_mw, "new")
  extract_files = c(
      "hh_mod_a_filt_10.csv",  # 2010 Household Questionnaire, Module A: Household Identification
      "hh_mod_a_filt_13.csv",  # 2013 Household Questionnaire, Module A: Household Identification
      "hh_mod_a_filt_16.csv",  # 2016 Household Questionnaire, Module A: Household Identification
      "hh_mod_f_10.csv",  # 2010 Household Questionnaire, Module F: Housing
      "hh_mod_f_16.csv",  # 2016 Household Questionnaire, Module F: Housing
      "hh_mod_l_10.csv",  # 2010 Household Questionnaire, Module L: Durable Goods
      "hh_mod_l_16.csv",  # 2016 Household Questionnaire, Module L: Durable Goods
      "householdgeovariables_ihs3_rerelease_10.csv"  # geo data
  )
  extract_and_rename(zip_path, extract_files, extract_dir)
  return()
}


#' Reads housing data (for mw10 and mw16)
#'
#' I have only manually checked that these variable names are consistent for
#' mw10 and mw16. This should probably also work for mw13, but I have NOT
#' double-checked.
#'
#' hh_f10 (rooms): How many separate rooms do the members of your household
#'   occupy?
#'   - integer, 0-8 (mw10), 1-9 (mw16)
#' hh_f09 (floor): The floor of the main dwelling is predominantly made of what
#    material?
#'   - discrete choices, mapped to integers 1-6
#' hh_f41 (toilet): What kind of toilet facility does your household use?
#'   - discrete choices, mapped to integers 1-6
#' hh_f36 (watsup): What was your main source of drinking water?
#'   - discrete choices, mapped to integers 1-16
#' hh_f19 (electric): Do you have electricity working in your dwelling?
#'   - integer, 1 = Yes, 2 = No
#' hh_f31 (phone): Is there a MTL telephone in working condition in the
#'   dwelling unit?
#'   - integer, 1 = Yes, 2 = No
#'
#' @param hh_csv_path Path to CSV with <hhid_col> and columns listed above
#' @param hhid_col String, name of character column to pass through
#' @param recode_dir Path to directory with Malawi floor/toilet/watsup recode
#'   CSVs
#' @return dataframe with columns <hhid_col>, watsup, toilet, floor, rooms,
#'   electric, phone, floor_qual, toilet_qual, watsup_qual
process_malawi_housing = function(hh_csv_path, hhid_col, recode_dir) {
  # tables to convert floor, toilet, and watsup (water supply) to 1-5 scale
  floor = read_csv(file.path(recode_dir, "floor_recode.csv"))
  toilet = read_csv(file.path(recode_dir, "toilet_recode.csv"))
  watsup = read_csv(file.path(recode_dir, "watsup_recode.csv"))

  df = read_csv(hh_csv_path, col_types = list(
      hh_f10 = col_integer(),
      hh_f09 = col_integer(),
      hh_f41 = col_integer(),
      hh_f36 = col_integer(),
      hh_f19 = col_integer(),
      hh_f31 = col_integer(),
      .default = col_character())
    ) %>%
    select(all_of(hhid_col), hh_f10, hh_f09, hh_f41, hh_f36, hh_f19, hh_f31) %>%
    rename(rooms = hh_f10, floor = hh_f09, toilet = hh_f41, watsup = hh_f36,
           electric = hh_f19, phone = hh_f31) %>%
    mutate(electric = ifelse(electric == 2, 0, 1),
           phone = ifelse(phone == 2, 0, 1)) %>%
    merge_verbose(floor, by = "floor", all.x = TRUE) %>%
    merge_verbose(toilet, by = "toilet", all.x = TRUE) %>%
    merge_verbose(watsup, by = "watsup", all.x = TRUE)
  return(df)
}


#' Reads assets data (for mw10 and mw16)
#'
#' I have only manually checked that these variable names are consistent for
#' mw10 and mw16. This should probably also work for mw13, but I have NOT
#' double-checked.
#'
#' hh_l01: Does your household own a [ITEM]?
#'   - integer, 1 = Yes, 2 = No
#' hh_l02: Durable good item code
#'   - discrete choices, mapped to integers 501-532
#'   - 507 = radio, 509 = TV, 514 = fridge, 518 = car
#'
#' @param assets_csv_path Path to CSV with <hhid_col> and columns listed above
#' @param hhid_col String, name of character column to pass through
#' @return dataframe with columns <hhid_col>, radio, tv, fridge, auto
process_malawi_assets = function(assets_csv_path, hhid_col) {
  df = read_csv(assets_csv_path, col_types = list(
      hh_l01 = col_integer(),
      hh_l02 = col_integer(),
      .default = col_character())
    ) %>%
    select(all_of(hhid_col), hh_l01, hh_l02) %>%
    filter(hh_l02 %in% c(507, 509, 514, 518)) %>%
    pivot_wider(names_from = hh_l02, values_from = hh_l01) %>%
    rename(radio = `507`, tv = `509`, fridge = `514`, auto = `518`) %>%
    mutate(across(-all_of(hhid_col), function(x) {ifelse(x == 1, 1, 0)})) %>%
    replace(is.na(.), 0)
  return(df)
}


#' Main function for processing Malawi LSMS data
#'
#' @param data_dir Path to data folder, with "raw" and "clean" subfolders.
#'   Should not end in a "/".
process_malawi = function(data_dir) {
  raw_mw = file.path(data_dir, "raw", "Malawi")

  # ===========================================================================
  # Match household IDs across surveys
  # ----------------------------------
  #
  # Create match between `case_id` (2010) and `y3_hhid`
  # - `case_id` and `HHID` are 1:1 and are both unique in mw10_hh_info
  # - `case_id` in mw16_hh_info sadly does not match `case_id` in mw10_hh_info
  # - in mw16_hh_info, `y3_hhid` and `HHID` have a 1:many mapping
  #   => `y3_hhid` has the format XXXX-YYYY, and we assume that a lowest number
  #      for YYYY indicates the original household
  # - we use `case_id` (2010) as the "household_id" for Malawi
  # ===========================================================================
  mw10_hh_info = read_csv(
      file.path(raw_mw, "hh_mod_a_filt_10.csv"),
      col_types = list(.default = col_character())
    ) %>%
    select(HHID, case_id, ea_id, hh_a01)  # hh_a01 is the district code
  mw16_hh_info = read_csv(
      file.path(raw_mw, "hh_mod_a_filt_16.csv"),
      col_types = list(.default = col_character())
    ) %>%
    select(HHID, y3_hhid, ea_id, district) %>%
    rename(ea_id16 = ea_id) %>%
    arrange(HHID, y3_hhid) %>%
    filter(!duplicated(HHID))
  match = mw10_hh_info %>% inner_join(mw16_hh_info, by = "HHID")

  # Check that all HHIDs generated above stayed in original EAs and districts
  stopifnot(all(match$ea_id == match$ea_id16))
  stopifnot(all(match$hh_a01 == match$district))
  match = match %>% select(HHID, case_id, y3_hhid)


  # ===========================================================================
  # Read household survey data
  # ===========================================================================

  # read housing data
  recode_dir = file.path(raw_mw, "recode")
  mw10h = process_malawi_housing(
      hh_csv_path = file.path(raw_mw, "hh_mod_f_10.csv"), hhid_col = "case_id",
      recode_dir = recode_dir) %>%
    rename(household_id = case_id)
  mw16h = process_malawi_housing(
      hh_csv_path = file.path(raw_mw, "hh_mod_f_16.csv"), hhid_col = "y3_hhid",
      recode_dir = recode_dir) %>%
    merge_verbose(match, by = "y3_hhid", all = FALSE) %>%
    select(-y3_hhid) %>%
    rename(household_id = case_id)

  # read assets data
  mw10a = process_malawi_assets(
      assets_csv_path = file.path(raw_mw, "hh_mod_l_10.csv"),
      hhid_col = "case_id") %>%
    rename(household_id = case_id)
  mw16a = process_malawi_assets(
      assets_csv_path = file.path(raw_mw, "hh_mod_l_16.csv"),
      hhid_col = "y3_hhid") %>%
    merge_verbose(match, by = "y3_hhid", all = FALSE) %>%
    select(-y3_hhid) %>%
    rename(household_id = case_id)


  # ===========================================================================
  # Combine housing, asset ownership, and geo info
  # ===========================================================================

  geo = read_csv(
      file.path(raw_mw, "householdgeovariables_ihs3_rerelease_10.csv"),
      col_types = list(.default = col_character())
    ) %>%
    select(case_id, ea_id, lat_modified, lon_modified) %>%
    rename(household_id = case_id, lat = lat_modified, lon = lon_modified) %>%
    distinct() %>%
    drop_na()

  final_cols = c(
      "year", "country", "household_id", "ea_id", "lat", "lon",
      "rooms", "electric", "phone", "radio", "tv", "auto",
      "floor_qual", "toilet_qual", "watsup_qual")

  mw10 = mw10h %>%
    merge_verbose(mw10a, by = "household_id", all = TRUE) %>%
    merge_verbose(geo, by = "household_id", all = FALSE) %>%
    mutate(year = 2010, country = "mw") %>%
    select(all_of(final_cols))

  mw16 = mw16h %>%
    merge_verbose(mw16a, by = "household_id", all = TRUE) %>%
    merge_verbose(geo, by = "household_id", all = FALSE) %>%
    mutate(year = 2016, country = "mw") %>%
    select(all_of(final_cols))

  mw10 = mw10 %>% filter(household_id %in% mw16$household_id)
  mw16 = mw16 %>% filter(household_id %in% mw10$household_id)
  malawi = rbind(mw10, mw16)
  return(malawi)
}
