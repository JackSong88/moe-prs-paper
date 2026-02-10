#!/bin/bash
# Helper script to run the entire validation data extraction and preprocessing

source env/moe/bin/activate

python data_preparation/2_prepare_validation_data/1_generate_scoring_snp_set.py

source data_preparation/2_prepare_validation_data/cartagene/cartagene_commands.sh
source data_preparation/2_prepare_validation_data/ukbb/ukb_commands.sh
