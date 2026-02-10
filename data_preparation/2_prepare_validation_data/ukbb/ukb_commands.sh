#!/bin/bash

set -e

source env/moe/bin/activate

# Extract covariates / phenotype data / QC filters:
python data_preparation/2_prepare_validation_data/ukbb/generate_qc_filters.py ||
python data_preparation/2_prepare_validation_data/ukbb/extract_medication_data.py
python data_preparation/2_prepare_validation_data/ukbb/prepare_phenotype_data.py

# Extract genotype data:
bash data_preparation/2_prepare_validation_data/ukbb/extract_genotype_data.sh
