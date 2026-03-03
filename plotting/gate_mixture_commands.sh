#!/bin/bash

# Loop over the datasets in data/harmonized_data directory,
# find the relevant MoE models for each one, and then
# invoke the plot_pgs_admixture.py script to generate
# the admixture figures for each one:

source env/moe/bin/activate

train_biobank=${1:-"ukbb"}

mapfile -t phenotypes < <(
  find data/harmonized_data/ -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -u
)
sex_stratified_phenotypes=("TST" "URT" "CRTN" "WHR")

echo "> Processing data for models trained on ${train_biobank}..."

# Loop over the phenotypes:
for phenotype in "${phenotypes[@]}"
do

  if [[ "${sex_stratified_phenotypes[*]}" =~ "$phenotype" ]]; then
      category="Sex"
  else
      category="Ancestry"
  fi

  for dataset in data/harmonized_data/"$phenotype"/"$train_biobank"/test_*.pkl
  do
    for model in data/trained_models/"$phenotype"/"$train_biobank"/*/Mo*.pkl
    do
      # Check that the model exists before invoking the plotting script:
      if [ ! -f "$model" ]; then
        echo "Model not found: $model"
        continue
      fi
      python3 plotting/plot_pgs_admixture.py --model "$model" \
                                             --dataset "$dataset" \
                                             --group-col "$category" \
                                             --subsample
    done
  done
done
