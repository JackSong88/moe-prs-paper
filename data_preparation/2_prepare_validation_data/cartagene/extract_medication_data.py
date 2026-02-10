import os.path as osp

import pandas as pd

cartagene_homedir = "~/links/projects/def-sgravel/cartagene/"

cols_dict = {
    "file111": "IID",
    "HIGH_BLOOD_CHOLESTEROL_CUR_TRE": "chol_med",
    "HIGH_BP_CURRENTLY_TREATED": "bp_med",
}

med_df = pd.read_csv(
    osp.join(cartagene_homedir, "old_metadata/data_Gravel936028_2.zip"),
    usecols=list(cols_dict.keys()),
)

med_df.rename(columns=cols_dict, inplace=True)

# Recode as binary:
med_df["chol_med"] = med_df["chol_med"] == 1
med_df["bp_med"] = med_df["bp_med"] == 1

# Output the medication use table:
med_df[["IID", "chol_med", "bp_med"]].to_csv(
    "data/covariates/cartagene/medication_use.txt", sep="\t", index=False
)

covar_df = pd.read_csv(
    "data/covariates/cartagene/covars_cartagene_pcs.txt",
    sep="\t",
    usecols=[1, 2, 13],
    names=["IID", "Sex", "Age"],
)
df = covar_df.merge(med_df)

# Create age bins:

bins = [0, 50, 60, float("inf")]
labels = ["Age<50", "Age 50–60", "Age>60"]

df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False).astype(str)

res = []

for med_cat in ("chol_med", "bp_med"):
    for group in ("Sex", "AgeGroup"):
        res.append(
            df.groupby(group)[med_cat]
            .mean()
            .reset_index()
            .rename(columns={group: "Group", "chol_med": "Proportion_Using_Medication"})
        )
        res[-1]["Medication"] = {
            "chol_med": "Cholesterol lowering medication",
            "bp_med": "Blood pressure medication",
        }[med_cat]

group_df = pd.concat(res, ignore_index=True)
group_df.to_csv("data/misc/medication_prevalence_cartagene.csv", index=False)
