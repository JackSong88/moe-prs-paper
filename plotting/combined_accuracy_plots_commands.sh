#!/bin/bash

source env/moe/bin/activate

models=("MoE-global-int" "MoE-fixed-resid-global-int")

for model in "${models[@]}"; do
    python plotting/combined_accuracy_plots.py --biobank ukbb --aggregate-single-prs --restrict-to-same-biobank --dataset train --moe-model "$model"
    python plotting/combined_accuracy_plots.py --biobank ukbb --aggregate-single-prs --restrict-to-same-biobank --moe-model "$model"
    python plotting/combined_accuracy_plots.py --biobank cartagene --aggregate-single-prs --restrict-to-same-biobank --moe-model "$model"
    python plotting/combined_accuracy_plots.py --biobank cartagene --aggregate-single-prs --restrict-to-same-biobank --dataset train --moe-model "$model"
done
