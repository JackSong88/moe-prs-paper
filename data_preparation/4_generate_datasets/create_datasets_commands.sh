#!/bin/bash

biobanks=("ukbb" "cartagene")
prop_test=0.3  # Proportion of samples to use for testing

source env/moe/bin/activate

# Loop over biobanks:
for biobank in "${biobanks[@]}"
do

    mapfile -t phenotypes < <(
        find "data/phenotypes/${biobank}" -type f -name '*.txt' -printf '%f\n' |
        sed 's/\.txt$//' |
        sort -u
    )

    for phenotype in "${phenotypes[@]}"
    do
        python3 data_preparation/4_generate_datasets/create_datasets.py --biobank "$biobank" --phenotype "$phenotype" --pcs-source "1kghdp" --prop-test "$prop_test"
    done
done
