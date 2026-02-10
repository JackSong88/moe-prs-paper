"""
Extract cholesterol-lowering medication data from the UK Biobank dataset.
Categorize medication use by sex and age
"""

import pandas as pd

df = pd.read_csv("/lustre03/project/6008063/neurohub/UKB/Tabular/current.csv", nrows=0)

# 6177 records medication use in males and 6153 records
# medication use in females:
medication_cols = [c for c in df.columns if "6177-0" in c or "6153-0" in c]

df = pd.read_csv(
    "/lustre03/project/6008063/neurohub/UKB/Tabular/current.csv",
    usecols=["eid", "22001-0.0", "21022-0.0"] + medication_cols,
)

# Rename columns for clarity
df.rename(columns={"eid": "IID", "22001-0.0": "Sex", "21022-0.0": "Age"}, inplace=True)

# Remove rows with missing values for sex or age or LDL:
df.dropna(subset=["Age", "Sex"], inplace=True)
df["Sex"] = df["Sex"].astype(int).map({0: "Female", 1: "Male"})

# Collapse the medication use columns into a single binary column:
# Based on the following data coding:
# https://biobank.ndph.ox.ac.uk/showcase/coding.cgi?id=100625
df["chol_med"] = (df[medication_cols] == 1).any(axis=1)
df["bp_med"] = (df[medication_cols] == 2).any(axis=1)

# Output the medication use table:
df[["IID", "chol_med", "bp_med"]].to_csv(
    "data/covariates/ukbb/medication_use.txt", sep="\t", index=False
)

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
group_df.to_csv("data/misc/medication_prevalence_ukb.csv", index=False)
