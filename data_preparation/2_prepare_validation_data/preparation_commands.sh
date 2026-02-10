#!/bin/bash
# Helper script to run the entire validation data extraction and preprocessing

set -e

source env/moe/bin/activate

python data_preparation/2_prepare_validation_data/1_generate_scoring_snp_set.py

bash data_preparation/2_prepare_validation_data/cartagene/cartagene_commands.sh
bash data_preparation/2_prepare_validation_data/ukbb/ukb_commands.sh
