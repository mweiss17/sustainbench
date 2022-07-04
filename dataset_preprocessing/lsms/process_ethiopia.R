# As of 2016, LSMS has conducted 3 survey waves in Ethiopia:
# 1. Ethiopia Rural Socioeconomic Survey (ERSS) 2011/12, aka. ESS1
# 2. Ethiopia Socioeconmic Survey 2013/14, aka. ESS2
# 3. Ethiopia Socioeconmic Survey 2015/16, aka. ESS3

# While ERSS 2011/12 only surveyed rural areas and small town areas, the later
# waves added urban areas. A small town area is defined as having population
# <10,000. An urban area is any tow with population >10,000 people.

# The `household_id` and `ea_id` fields are consistent across all waves.
# Starting with ESS2, due to the addition of new urban areas, the surveys
# added new `household_id2` and `ea_id2` fields.

#' Extracts the necessary data files for LSMS Ethiopia surveys.
#'
#' Depends on extract_and_rename() function in lsms_utils.R.
#' Assumes that the following ZIP files have already been downloaded
#' into {data_dir}/raw/Ethiopia:
#' - ETH_2011_ERSS_v02_M_CSV.zip
#' - ETH_2015_ESS_v03_M_CSV.zip
#'
#' @param data_dir Path to data folder, with "raw" and "clean" subfolders.
#'   Should not end in a "/".
#' @return Nothing.
#' @export
extract_ethiopia = function(data_dir) {
  raw_et = file.path(data_dir, "raw", "Ethiopia")
  zip_path = file.path(raw_et, "ETH_2011_ERSS_v02_M_CSV.zip")
  extract_dir = file.path(raw_et, "ERSS_11.12")
  extract_files = c(
      "ETH_2011_ERSS_v02_M_CSV/sect9_hh_w1.csv",   # Household Questionnaire, Section 9 (Housing)
      "ETH_2011_ERSS_v02_M_CSV/sect10_hh_w1.csv",  # Household Questionnaire, Section 10 (Assets)
      "ETH_2011_ERSS_v02_M_CSV/pub_eth_householdgeovariables_y1.csv"  # geographic data
  )
  extract_and_rename(zip_path, extract_files, extract_dir)

  zip_path = file.path(raw_et, "ETH_2015_ESS_v03_M_CSV.zip")
  extract_dir = file.path(raw_et, "ESS_15.16")
  extract_files = c(
      "Household/sect9_hh_w3.csv",  # Household Questionnaire, Section 9: Housing
      "Household/sect10_hh_w3.csv"  # Household Questionnaire, Section 10: Household assets
  )
  extract_and_rename(zip_path, extract_files, extract_dir)
  return()
}

#' Main function for processing Ethiopia LSMS data
#'
#' @param data_dir Path to data folder, with "raw" and "clean" subfolders.
#'   Should not end in a "/".
process_ethiopia = function(data_dir) {

  raw_et11 = file.path(data_dir, "raw", "Ethiopia", "ERSS_11.12")
  raw_et15 = file.path(data_dir, "raw", "Ethiopia", "ESS_15.16")

  # ===========================================================================
  # Read housing data
  # -----------------
  #
  # hh_s9q02_a: How long has this household been living in this dwelling?
  #   - integer, in years
  # hh_s9q04 (rooms): How many rooms (excluding kitchen, toilet and bath room)
  #   does the household occupy?
  #   - integer 0+
  # hh_s9q07 (floor): The floor of the main dwelling is predominantly made of
  #  what material?
  #   - discrete choices mapped to integers 1-10
  # hh_s9q10 (toilet): What type of toilet facilities does the household use?
  #   - (w1) discrete choices mapped to integers 1-9
  #   - (w3) discrete choices mapped to integers 1-8
  # hh_s9q13 (watsup): What is the main source of drinking water in the rainy
  #   season?
  #   - (w1) discrete choices mapped to integers 1-13 (watsup = "water supply")
  #   - (w3) discrete choices mapped to integers 1-15
  # hh_s9q19_a (electric, w3-only): What is the main Source of light for the
  #   household?
  #   - discrete choices mapped to integers 1-13,
  #     <=4 indicates electricity
  # hh_s9q20 (electric, w1-only): How many times did the household face electric
  #   power failure/interruption lasting for at least one hour during the last
  #   week?
  #   - discrete choices mapped to integers 1-6,
  #     1 = "don't use electricity", 2+ = bins for frequency of power outage
  # ===========================================================================

  w1h = read_csv(file.path(raw_et11, "sect9_hh_w1.csv"), col_types = list(
      household_id = col_character(),
      .default = col_double()
    )) %>%
    select(household_id, hh_s9q04, hh_s9q07, hh_s9q10, hh_s9q13, hh_s9q20) %>%
    rename(rooms = hh_s9q04, floor = hh_s9q07, toilet = hh_s9q10,
           watsup = hh_s9q13, electric = hh_s9q20) %>%
    mutate(electric = ifelse(electric == 1, 0, 1))

  w3h = read_csv(file.path(raw_et15, "sect9_hh_w3.csv"), col_types = list(
      household_id = col_character(),
      household_id2 = col_character(),
      .default = col_double()
    )) %>%
    filter(hh_s9q02_a >= 4) %>%  # drop migrant household, keep if stay >4 years
    select(household_id, household_id2, hh_s9q04, hh_s9q07, hh_s9q10, hh_s9q13,
           hh_s9q19_a) %>%
    drop_na(household_id) %>%  # drop households not in wave 1
    rename(rooms = hh_s9q04, floor = hh_s9q07, toilet = hh_s9q10,
           watsup = hh_s9q13, electric = hh_s9q19_a) %>%
    mutate(electric = ifelse(electric <= 4, 1, 0))

  # 2 households from wave1 each split into 2 households for wave3, all in
  # the same `ea_id`. (`household_id` "13010100303004" and "13010100303032")
  # In order to create a 1-to-1 mapping of households between waves 1 and 3,
  # we choose to drop the wave3 households with larger `household_id2`. This
  # is arbitrary, but we couldn't think of anything better to do.
  w3h = w3h %>%
    arrange(household_id, household_id2) %>%
    filter(!duplicated(household_id))  # drops multiples of old id

  w3h = w3h %>% filter(household_id %in% w1h$household_id)
  w1h = w1h %>% filter(household_id %in% w3h$household_id)
  households = w3h[, c("household_id", "household_id2")]

  # recode floor, toilet, and water data on a 1-5 scale
  recode_dir = file.path(data_dir, "raw", "Ethiopia", "recode")

  floor = read_csv(file.path(recode_dir, "floor_recode.csv"))
  toilet = read_csv(file.path(recode_dir, "toilet_recode.csv"))
  watsup = read_csv(file.path(recode_dir, "watsup_recode.csv"))
  w1h = w1h %>%
    merge_verbose(floor, by.x = "floor", by.y = "floor_code", all.x = TRUE) %>%
    merge_verbose(toilet, by.x = "toilet", by.y = "toilet_code", all.x = TRUE) %>%
    merge_verbose(watsup, by.x = "watsup", by.y = "watsup_code", all.x = TRUE)

  toilet = read_csv(file.path(recode_dir, "toilet_recode_w3.csv"))
  watsup = read_csv(file.path(recode_dir, "watsup_recode_w3.csv"))
  w3h = w3h %>%
    merge_verbose(floor, by.x = "floor", by.y = "floor_code", all.x = TRUE) %>%
    merge_verbose(toilet, by.x = "toilet", by.y = "toilet_code", all.x = TRUE) %>%
    merge_verbose(watsup, by.x = "watsup", by.y = "watsup_code", all.x = TRUE)


  # ===========================================================================
  # Read household asset ownership data
  # -----------------------------------
  #
  # hh_s10q0a: Item Description
  #   - character
  # hh_s10q01: How many of this [item] does your household own?
  #   - integer 0+
  # ===========================================================================

  w1a = read_dta("raw/Ethiopia/ERSS_11.12/sect10_hh_w1.dta")
  w3a = read_dta("raw/Ethiopia/ERSS_15.16/sect10_hh_w3.dta")

  w1a = read_csv(file.path(raw_et11, "sect10_hh_w1.csv"), col_types = list(
      household_id = col_character(),
      hh_s10q0a = col_character(),
      hh_s10q01 = col_integer()
    )) %>%
    filter(hh_s10q0a %in% c("Fixed line telephone", "Radio", "Television",
                            "Refrigerator", "Private car")) %>%
    select(household_id, hh_s10q0a, hh_s10q01) %>%
    reshape2::dcast(household_id ~ hh_s10q0a) %>%
    mutate_at(vars(-("household_id")),
              function(x) {ifelse(x >= 1, 1, 0)}) %>%
    rename(phone = `Fixed line telephone`, auto = `Private car`, radio = Radio,
           fridge = Refrigerator, tv = Television)

  w3a = read_csv(file.path(raw_et15, "sect10_hh_w3.csv"), col_types = list(
      household_id = col_character(),
      household_id2 = col_character(),
      hh_s10q0a = col_character(),
      hh_s10q01 = col_integer()
    )) %>%
    filter(household_id2 %in% households$household_id2,
           hh_s10q0a %in% c("Fixed line telephone", "Radio/tape recorder",
                            "Television", "Refrigerator", "Private car")) %>%
    select(household_id, hh_s10q0a, hh_s10q01) %>%
    reshape2::dcast(household_id ~ hh_s10q0a) %>%
    mutate_at(vars(-("household_id")),
              function(x) {ifelse(x >= 1, 1, 0)}) %>%
    rename(phone = `Fixed line telephone`, auto = `Private car`,
           radio = `Radio/tape recorder`, fridge = Refrigerator,
           tv = Television)


  # ===========================================================================
  # Combine housing, asset ownership, and geo info
  # ===========================================================================

  geo = read_csv(
      file.path(raw_et11, "pub_eth_householdgeovariables_y1.csv"),
      col_types = list(.default = col_character())
    ) %>%
    select(household_id, ea_id, LAT_DD_MOD, LON_DD_MOD) %>%
    rename(lat = LAT_DD_MOD, lon = LON_DD_MOD) %>%
    distinct() %>%
    drop_na()

  final_cols = c(
      "year", "country", "household_id", "ea_id", "lat", "lon",
      "rooms", "electric", "phone", "radio", "tv", "auto",
      "floor_qual", "toilet_qual", "watsup_qual")

  # merge the asset w/ house data
  w1 = w1h %>%
    merge_verbose(w1a, by = "household_id", all = TRUE) %>%
    merge_verbose(geo, by = "household_id", all = FALSE) %>%
    mutate(year = 2011, country = "et") %>%
    select(all_of(final_cols))

  w3 = w3h %>%
    merge_verbose(w3a, by = "household_id", all = TRUE) %>%
    merge_verbose(geo, by = "household_id", all = FALSE) %>%
    mutate(year = 2015, country = "et") %>%
    select(all_of(final_cols))

  w1 = w1 %>% filter(household_id %in% w3$household_id)
  w3 = w3 %>% filter(household_id %in% w1$household_id)
  ethiopia = rbind(w1, w3)
  return(ethiopia)
}
