#!/bin/bash
#SBATCH --account=def-sgravel
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=08:00:00
#SBATCH --output=./log/data_preparation/2_prepare_validation_data/ukbb/%x.out
#SBATCH --mail-user=shadi.zabad@mail.mcgill.ca
#SBATCH --mail-type=FAIL

echo "Job started at: `date`"
echo "Job ID: $SLURM_JOBID"

module load plink

UKBB_GENOTYPE_DIR="/project/rpp-aevans-ab/neurohub/UKB/Bulk/Imputation/UKB_imputation_from_genotype/"

CHR=${1:-22}  # Chromosome number (default 22)
ind_keep_file=${2-"data/keep_files/ukbb_qc_individuals.keep"}
output_dir=${3-"data/ukbb_qc_genotypes"}
snp_keep="data/snp_sets/GRCh37.bed"

mkdir -p "$output_dir"

plink2 --bgen "$UKBB_GENOTYPE_DIR/ukb22828_c${CHR}_b0_v3.bgen" ref-first \
      --sample "$UKBB_GENOTYPE_DIR/ukb22828_c${CHR}_b0_v3.sample" \
      --make-bed \
      --allow-no-sex \
      --max-alleles 2 \
      --keep "$ind_keep_file" \
      --extract range "$snp_keep" \
      --hard-call-threshold "0.1" \
      --out "$output_dir/chr_${CHR}"

echo "Job finished with exit code $? at: `date`"
