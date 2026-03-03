import numpy as np
import pandas as pd
from magenpy.utils.system_utils import makedir

covar_df = pd.read_csv(
    "~/links/projects/def-sgravel/cartagene/flagship_GWAS_covariates/GWAS_ALL_covariates.tsv",
    sep="\t",
)
covar_df = covar_df.drop(columns=["Age2", "Array"]).dropna()
covar_df["Sex"] = np.abs(
    covar_df["Sex"] - 1
)  # Reverse the sex coding to match the coding in the UK Biobank

makedir("data/covariates/cartagene/")

covar_df[["FID", "IID", "Sex"] + [f"PC{i}" for i in range(1, 11)] + ["Age"]].to_csv(
    "data/covariates/cartagene/covars_cartagene_pcs.txt",
    sep="\t",
    index=False,
    header=False,
)

# Extract cluster information from the flagship's pipeline:

cluster_df = pd.read_csv(
    "~/links/projects/def-sgravel/cartagene/clusters/ID_to_clusterID_flagship.tsv",
    sep="\s+",
    dtype={"cluster": int},
)
cluster_df["IID"] = cluster_df["FID"]
cluster_df.rename(columns={"cluster": "Cluster"}, inplace=True)

cluster_df[["FID", "IID", "Cluster"]].to_csv(
    "data/covariates/cartagene/cluster_assignment.txt", sep="\t", index=False
)
