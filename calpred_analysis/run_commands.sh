#!/bin/bash

module load gcc/12.3 r/4.3.1
export R_LIBS=calpred_analysis/calpred_R_env
source env/moe/bin/activate

mkdir -p figures/calpred

phenotypes=("HEIGHT" "LDL" "HDL" "URT" "CRTN")
biobanks=("ukbb" "cartagene")

for pheno in "${phenotypes[@]}"; do
    for bb in "${biobanks[@]}"; do
        python calpred_analysis/fit_calpred.py --dataset "data/harmonized_data/${pheno}/${bb}/train_data.pkl"
    done
done

# Run calpred on the adjusted LDL phenotype in the UKB:

python calpred_analysis/fit_calpred.py --dataset "data/harmonized_data/LDL_adj/ukbb/train_data.pkl"
