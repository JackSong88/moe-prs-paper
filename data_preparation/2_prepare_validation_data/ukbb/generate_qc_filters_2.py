import functools
import os.path as osp
import sys

import pandas as pd
from magenpy.utils.system_utils import makedir

print = functools.partial(print, flush=True)

# ----------------------------------------
# Options / paths / constants:

num_pcs = 10

# Paths/file names:
ukb_dir = "/project/rpp-aevans-ab/neurohub/UKB/"
covar_file = "data/covariates/ukbb/covars_ukbb_pcs.txt"
keep_file = "data/keep_files/ukbb_qc_individuals.keep"

pc_columns = [f"PC{i + 1}" for i in range(num_pcs)]

# -------- Sample quality control --------
# Read the sample QC file from the UKBB archive
print("> Extracting individual data...")

column_subset = ["eid", "22001-0.0", "21003-0.0", "22019-0.0", "22021-0.0"]
column_subset += [f"22009-0.{i + 1}" for i in range(num_pcs)]

df = pd.read_csv(osp.join(ukb_dir, "Tabular/current.csv"), usecols=column_subset)
# Make sure the columns are returned in the correct order:
df = df[column_subset]
# Rename the columns:
df.columns = [
    "IID",
    "Sex",
    "Age",
    "sex_chromosome_aneuploidy",
    "genetic_relatedness",
] + pc_columns
# Add Family ID:
df["FID"] = df["IID"]

# Read the list of withdrawn individuals:
withdrawn_df = pd.read_csv(
    osp.join(ukb_dir, "Withdrawals/w45551_20250818.csv"), names=["IID"]
)

# -------------------------------------------------

# Apply the standard filters:

# Remove withdrawn samples from df:
df = df[~df["IID"].isin(withdrawn_df["IID"])]

df = df.loc[
    (
        df["genetic_relatedness"] == 0 | (df["genetic_relatedness"] == 1)
    )  # Remove samples with excess relatives
    & (df["sex_chromosome_aneuploidy"] != 1),  # Remove samples with sex chr aneuploidy
]

# Write the list of remaining individuals to file:
makedir(osp.dirname(keep_file))
df[["FID", "IID"]].to_csv(keep_file, sep="\t", header=False, index=False)

# -------- Sample covariate file --------
# Create a covariates file to use in downstream analyses:

print("Creating a file with covariates for the selected individuals...")

covar_df = df[["FID", "IID", "Sex"] + pc_columns + ["Age"]]
covar_df.dropna(inplace=True)

for col in pc_columns:
    covar_df[col] = covar_df[col].round(decimals=5)

# Write the file:
makedir(osp.dirname(covar_file))
covar_df.to_csv(covar_file, sep="\t", header=False, index=False)
