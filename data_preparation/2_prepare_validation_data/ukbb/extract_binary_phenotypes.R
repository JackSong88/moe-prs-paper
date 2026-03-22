#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse)
  library(bigreadr)
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(PheWAS)
})

##########################
# Map file paths — edit these
##########################

PHECODE_MAP_RDS       <- "data/phewas/phecode_map.rds"        # tibble: vocabulary_id, code, phecode
PHECODE_MAP_ICD10_RDS <- "data/phewas/phecode_map_icd10.rds"  # tibble: vocabulary_id, code, phecode

##########################
# CLI options
##########################

option_list <- list(
  make_option(c("-f","--ukb-file"),  type="character", default=NULL,
              help="Path to UKB wide-format CSV"),
  make_option(c("-p","--phecodes"), type="character", default=NULL,
              help="Comma-separated list of phecodes"),
  make_option(c("-n","--phenotype-names"), type="character", default=NULL,
                help="Comma-separated names for phenotypes (same order as --phecodes)"),
  make_option(c("-o","--outdir"),   type="character", default="data/ukbb-selected-phecodes",
              help="Output directory"),
  make_option(c("--include-cancer"),       action="store_true", default=FALSE,
              help="Include cancer-related ICD fields"),
  make_option(c("--include-selfreported"), action="store_true", default=FALSE,
              help="Include self-reported fields (20002 + coding609)"),
  make_option(c("--apply-phecode-exclusion"), action="store_true", default=FALSE,
            help="Apply phecode exclusions")
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$`ukb-file`) || is.null(opt$phecodes)) {
  stop("Usage: Rscript prepare-selected-phecodes-cli.R -f ukb.csv -p 411.2,274 [--include-cancer] [--include-selfreported]")
}

ukb_csv              <- opt$`ukb-file`
phecodes_requested   <- unlist(strsplit(opt$phecodes, ",\\s*"))
out_dir              <- opt$outdir
include_cancer       <- opt$`include-cancer`
include_selfreported <- opt$`include-selfreported`
apply_phecode_exclusions <- opt$`apply-phecode-exclusion`
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

##########################
# Parse phenotype names (if passed)

if (!is.null(opt$`phenotype-names`)) {
  phenotype_names <- unlist(strsplit(opt$`phenotype-names`, ",\\s*"))
  if (length(phenotype_names) != length(phecodes_requested)) {
    stop("--phenotype-names must have same length as --phecodes")
  }
} else {
  phenotype_names <- phecodes_requested
}

##########################
# Load mapping tables
##########################

# NOTE: Download from: https://phewascatalog.org/phewas/
phecode_map1 <- as_tibble(read.csv("data/phewas/Phecode_map_v1_2_icd9_icd10cm.csv",
  colClasses = c(ICD = "character",
    Flag = "integer",
    ICDString = "character",
    Phecode = "character",
    PhecodeString = "character",
    PhecodeCategory = "character"
)))

# Add the vocabulary ID:
phecode_map1 <- phecode_map1 %>% mutate(vocabulary_id = dplyr::case_when(
  Flag %in% c(9L, 9)  ~ "ICD9CM",
  Flag %in% c(10L,10) ~ "ICD10CM",
  TRUE                ~ NA_character_
))

# Transform the ICD code:
phecode_map1 <- mutate_at(phecode_map1, "ICD", ~ sub("\\.", "", .))

# Keep the three columns:
phecode_map1 <- phecode_map1 %>%
  select(vocabulary_id = vocabulary_id,
         code = ICD,
         phecode = Phecode)


# phecode_map       <- readRDS(PHECODE_MAP_RDS)
# phecode_map_icd10 <- readRDS(PHECODE_MAP_ICD10_RDS)

message("phecode_map rows: ", nrow(phecode_map1),
        " | vocabs: ", paste(unique(phecode_map1$vocabulary_id), collapse=", "))

##########################
# Read UKB fields
##########################

if (!file.exists(ukb_csv)) stop("UKB CSV not found: ", ukb_csv)
sex <- fread2(ukb_csv, select = "22001-0.0")[[1]]
eid <- fread2(ukb_csv, select = "eid")[[1]]

icd10_select <- c(
  # Causes of death
  paste0("40001-", 0:1, ".0"),      # underlying cause
  paste0("40002-0.", 0:13),         # contributory causes
  paste0("40002-1.", 0:13),

  # Hospital inpatient: primary, secondary, external causes
  paste0("41201-0.", 0:21),   # external causes
  paste0("41202-0.", 0:79),   # primary ICD10 diagnoses (array length ~79)
  paste0("41204-0.", 0:209),  # secondary ICD10 diagnoses (array length — adjust if needed)

  # Summary diagnoses (big array of distinct ICD10 codes)
  paste0("41270-0.", 0:258)   # summary ICD10 diagnoses (array length ~259)
)
if (include_cancer) {
  icd10_select <- unique(c(icd10_select, paste0("40006-", 0:16, ".0")))
}

df_ICD10 <- fread2(ukb_csv, colClasses = "character", select = icd10_select)

if (include_selfreported) {
  # NOTE: Download from: https://biobank.ndph.ox.ac.uk/ukb/coding.cgi?id=609
  coding609 <- fread2("data/phewas/coding609.tsv")
  sr_cols   <- c(paste0("20002-0.", 0:33), paste0("20002-1.", 0:33),
                 paste0("20002-2.", 0:33), paste0("20002-3.", 0:33))
  df_sr <- fread2(ukb_csv, colClasses = "character", select = sr_cols)

  df_sr <- df_sr %>% mutate_all(~ as.character(factor(., levels = coding609$coding,
    labels = coding609$meaning)))

  df_ICD10 <- bind_cols(df_ICD10, df_sr)
}

icd9_select <- paste0("41271-0.", 0:46)
if (include_cancer) {
  icd9_select <- unique(c(icd9_select, paste0("40013-", 0:14, ".0")))
}
df_ICD9 <- tryCatch(
  fread2(ukb_csv, colClasses = "character", select = icd9_select),
  error = function(e) data.frame()
)

##########################
# Wide -> long tibbles
##########################

build_long <- function(df, vocab_id) {
  df %>%
    mutate_all(~ ifelse(. == "", NA, .)) %>%
    mutate(id = row_number()) %>%
    pivot_longer(-id, values_to = "code", values_drop_na = TRUE) %>%
    group_by(id, code) %>%
    summarise(count = n(), .groups = "drop") %>%
    ungroup() %>%
    transmute(
      id            = as.integer(id),
      vocabulary_id = vocab_id,
      code          = trimws(as.character(code)),
      count         = as.integer(count)
    ) %>%
    as_tibble()
}

# vocabulary_id values must match exactly what is in phecode_map
id_icd10_count <- build_long(df_ICD10, "ICD10CM")

if (ncol(df_ICD9) > 0) {
  id_icd9_count <- build_long(df_ICD9, "ICD9CM")
} else {
  id_icd9_count <- tibble(id = integer(0), vocabulary_id = character(0),
                          code = character(0), count = integer(0))
}

codes_tab <- bind_rows(id_icd10_count, id_icd9_count)

message("codes_tab rows: ", nrow(codes_tab),
        " | vocabs: ", paste(unique(codes_tab$vocabulary_id), collapse=", "))

##########################
# createPhenotypes
##########################

phen_wide <- createPhenotypes(
  codes_tab,
  id.sex                 = tibble(id = seq_along(sex), sex = c("F","M")[sex + 1L]),
  vocabulary.map         =phecode_map1, #mutate_at(phecode_map1,
                            #   "code", ~ sub("\\.", "", .)),
  min.code.count         = 1,
  add.phecode.exclusions = apply_phecode_exclusions,
  full.population.ids    = seq_along(sex),
  translate              = TRUE
)

phen_wide <- phen_wide[order(phen_wide$id), , drop = FALSE]

##########################
# Select requested phecodes and write output
##########################

present <- intersect(phecodes_requested, colnames(phen_wide))
missing <- setdiff(phecodes_requested, present)
if (length(missing) > 0) warning("Missing phecodes: ", paste(missing, collapse = ", "))

phen_selected <- phen_wide %>%
  select(all_of(c("id", present)))
phen_selected$eid <- eid

for (i in seq_along(phecodes_requested)) {

  pc <- phecodes_requested[i]
  fname <- phenotype_names[i]

  if (!pc %in% names(phen_selected)) next

  out_df <- phen_selected %>%
    select(eid, all_of(pc)) %>%
    filter(!is.na(.data[[pc]])) %>%
    transmute(FID = eid,
              IID = eid,
              PHENO = as.integer(.data[[pc]]))

  write.table(out_df,
              file = file.path(out_dir, paste0(fname, ".txt")),
              sep = "\t",
              quote = FALSE,
              row.names = FALSE,
              col.names = FALSE)
}

message("Done. Output written to: ", out_dir)
