#!/bin/bash
# This script creates the python environment for running the scripts
# on the cluster.

mkdir -p env

echo "========================================================"
echo "Setting up environment for MoE project..."

module load python/3.10
python --version

# Create environment with latest version of VIPRS:
rm -rf env/moe/
python -m venv env/moe/
source env/moe/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

deactivate

echo "========================================================"
# Setting up the environment for PheWAS / binary phenotype extraction logic

# Setup the R environment:
module load gcc/12.3 r/4.5.0
mkdir -p "env/R_phewas_env" || true

export R_LIBS="env/R_phewas_env"

R -e 'install.packages(c("optparse", "bigreadr", "dplyr", "data.table",  "tidyr", "stringr", "remotes", "purrr"), repos="https://cloud.r-project.org/")'
R -e 'remotes::install_github("PheWAS/PheWAS", dependencies = TRUE)'

# Download some of the required data:
mkdir -p data/phewas/
R -e 'library(PheWAS); data("phecode_map_icd10"); saveRDS(phecode_map_icd10, file="data/phewas/phecode_map_icd10.rds")'
R -e 'library(PheWAS); data("phecode_map");  saveRDS(phecode_map,  file="data/phewas/phecode_map.rds")'
R -e 'library(PheWAS); data("pheinfo");           saveRDS(pheinfo,           file="data/phewas/pheinfo.rds")'

wget -O data/phewas/phecode_map_icd9_icd10.csv.zip https://phewascatalog.org/phewas/_w_e12980f30d034d7eb7096b31d11424ae/data/Phecode_map_v1_2_icd9_icd10cm.csv.zip
unzip data/phewas/phecode_map_icd9_icd10.csv.zip  -d data/phewas/
rm data/phewas/phecode_map_icd9_icd10.csv.zip

curl 'https://biobank.ndph.ox.ac.uk/ukb/codown.cgi' \
  --data-raw 'id=609' \
  -o data/phewas/coding609.tsv

echo "Done!"
