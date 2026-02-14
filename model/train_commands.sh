#!/bin/bash

mkdir -p ./log/model_fit/

mapfile -t phenotypes < <(
  find data/harmonized_data/ -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -u
)

for phenotype in "${phenotypes[@]}"
do
  sbatch -J "$phenotype" model/train_job.sh "$phenotype"
done
