#!/bin/bash

# Loop over the accuracy metrics in data/evaluation directory
# and invoke the plot_predictive_performance.py script to generate
# the accuracy figures for each one:


biobank=${1:-"ukbb"}
mapfile -t phenotypes < <(
  find data/harmonized_data/ -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -u
)
sex_stratified_phenotypes=("TST" "URT" "CRTN" "WHR")

source env/moe/bin/activate

for phenotype in "${phenotypes[@]}"
do
  for dataset in data/evaluation/"$phenotype"/"$biobank"/*.csv
  do

    if [[ "${sex_stratified_phenotypes[*]}" =~ "$phenotype" ]]; then
      category="Sex"
    else
      category="Ancestry"
    fi

    extracted_biobank=$(echo "$dataset" | grep -oP "(?<=data/evaluation/$phenotype/)[^/]+")

    python3 plotting/plot_predictive_performance.py --metrics-file "$dataset" \
                                                    --aggregate-single-prs \
                                                    --category $category \
                                                    --restrict-to-same-biobank

    #python3 plotting/plot_predictive_performance.py --metrics-file "$dataset" \
    #                                                --aggregate-single-prs \
    #                                                --category "$category" \
    #                                                --restrict-to-same-biobank \
    #                                                --train-dataset "ukbb/train_data_rph"
    #python3 plotting/plot_predictive_performance.py --metrics-file "$dataset" \
    #                                                --aggregate-single-prs \
    #                                                --category "$category" \
    #                                                --restrict-to-same-biobank \
    #                                                --train-dataset "ukbb/train_data_rprs"
    #python3 plotting/plot_predictive_performance.py --metrics-file "$dataset" \
    #                                                --aggregate-single-prs \
    #                                                --category Ancestry UMAP_Cluster Sex
    #python3 plotting/plot_predictive_performance.py --metrics-file "$dataset" \
    #                                                --aggregate-single-prs \
    #                                                --category Ancestry UMAP_Cluster Sex \
    #                                                --restrict-to-same-biobank
  done
done
