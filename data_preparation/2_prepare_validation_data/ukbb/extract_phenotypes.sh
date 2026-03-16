#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=01:00:00
#SBATCH --output=./log/data_preparation/ukbb_phenotypes/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL
#!/bin/bash

source env/moe/bin/activate

python data_preparation/2_prepare_validation_data/ukbb/prepare_phenotype_data.py

module load gcc/12.3 r/4.5.0
mkdir -p "env/R_phewas_env" || true

export R_LIBS="env/R_phewas_env"

# WORK IN PROGRESS!!!
Rscript data_preparation/2_prepare_validation_data/ukbb/extract_binary_phenotypes.R \
        -f "/project/rpp-aevans-ab/neurohub/UKB/Tabular/current.csv" \
        -p 250.2,250.1,427.2,428,401,433.1,411.4,290.1,296.2,332,274.1,495 \
        -n T2D,T1D,AF,HF,HTN,STR,CAD,DEM,MDD,PD,GOUT,ASTH \
        -o "data/ukbb-selected-phecodes/" \
        --include-selfreported
