#!/bin/bash

set -e

mkdir -p data/covariates/cartagene/
mkdir -p data/misc/

source env/moe/bin/activate

python data_preparation/2_prepare_validation_data/cartagene/prepare_covariates.py
python data_preparation/2_prepare_validation_data/cartagene/extract_medication_data.py
python data_preparation/2_prepare_validation_data/cartagene/prepare_phenotype_data.py
