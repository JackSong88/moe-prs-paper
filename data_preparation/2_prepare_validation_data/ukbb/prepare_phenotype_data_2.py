
import pandas as pd
import numpy as np
import os.path as osp
import sys
import itertools
from magenpy.utils.system_utils import makedir
sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils import detect_outliers

ukb_homedir = "/project/rpp-aevans-ab/neurohub/UKB/"

pheno_dict = {
    "48-0.0": "WAIST",
    "49-0.0": "HIP",
    "50-0.0": "HEIGHT",
    "21001-0.0": "BMI",
    "30760-0.0": "HDL",
    "30780-0.0": "LDL",
    "30690-0.0": "TC",
    "30870-0.0": "LOG_TG",
    "20151-0.0": "FVC",
    "20150-0.0": "FEV1",
    "20258-0.0": "FEV1_FVC",
    "30700-0.0": "CRTN",
    "30880-0.0": "URT",
    "30850-0.0": "TST",
    "4080-0.0": "SBP",
    "4079-0.0": "DBP",
    "6138-0.0": "EDU"
}

# ------------------------------------------------------
# Helper functions to transform some of the phenotypes:

def transform_education_years(dat):
    """
    Transform educations level from categorical variable
    to education years defined by Okaby et al. 2016.
    Relevant tables are:

        https://biobank.ndph.ox.ac.uk/ukb/coding.cgi?id=100305
        https://elifesciences.org/articles/48376#app1table4
    """

    return dat.map({
        1: 20,
        2: 13,
        3: 10,
        4: 10,
        5: 19,
        6: 15,
        -7: 7,
        -3: np.nan
    })

pheno_transform_func = {
    "6138-0.0": transform_education_years,
    "30870-0.0": np.log,

}

log_before_outlier_detection = [
    '21001-0.0', '30760-0.0', '20151-0.0', '20150-0.0'
]

# ------------------------------------------------------
# Read quantitative phenotypes

pheno_df = pd.read_csv(osp.join(ukb_homedir, "Tabular/current.csv"),
                       usecols=['eid', '22001-0.0'] + list(pheno_dict.keys()))
pheno_df.rename(columns={
    'eid': 'IID',
    '22001-0.0': 'Sex'
}, inplace=True)


# Read the list of withdrawn individuals:
withdrawn_df = pd.read_csv(osp.join(ukb_homedir, "Withdrawals/w45551_20250818.csv"),
                          names=['IID'])


# Remove withdrawn samples from df:
pheno_df = pheno_df[~pheno_df['IID'].isin(withdrawn_df['IID'])]
pheno_df['FID'] = pheno_df['IID']

# Create the phenotype directory:
makedir("data/phenotypes/ukbb/")

# ------------------------------------------------------

# Loop over the phenotypes, process them, and output to file:
for pheno in pheno_dict.keys():

    sub_pheno_df = pheno_df[['FID', 'IID', pheno]].copy()
    sub_pheno_df.columns = ['FID', 'IID', 'phenotype']

    # Apply phenotype-specific transforms:
    if pheno in pheno_transform_func:
        sub_pheno_df['phenotype'] = pheno_transform_func[pheno](sub_pheno_df['phenotype'])

    # Filter outliers in each sex separately:
    # If the phenotype is skewed and positive, apply log transformation before outlier detection.
    if pheno in log_before_outlier_detection:
        od_pheno = np.log(sub_pheno_df['phenotype'])
    else:
        od_pheno = sub_pheno_df['phenotype']

    sub_pheno_df['phenotype'] = np.where(
        detect_outliers(od_pheno, stratify=pheno_df['Sex']),
        np.nan, sub_pheno_df['phenotype']
    )
    # Save the phenotype
    sub_pheno_df.to_csv(f"data/phenotypes/ukbb/{pheno_dict[pheno]}.txt",
        sep="\t", index=False, header=False, na_rep='NA'
    )

# =============================================================================
# Computed phenotypes:

# 1) Waist-to-hip ratio (WHR):

sub_pheno_df = pheno_df[['FID', 'IID', '48-0.0', '49-0.0']].copy()
# Remove outliers in each separately:
sub_pheno_df['48-0.0'] = np.where(
    detect_outliers(sub_pheno_df['48-0.0'], stratify=pheno_df['Sex']),
    np.nan, sub_pheno_df['48-0.0']
)
sub_pheno_df['49-0.0'] = np.where(
    detect_outliers(sub_pheno_df['49-0.0'], stratify=pheno_df['Sex']),
    np.nan, sub_pheno_df['49-0.0']
)
# Compute WHR
sub_pheno_df['phenotype'] = sub_pheno_df['48-0.0'] / sub_pheno_df['49-0.0']
# Remove outliers in WHR:
sub_pheno_df['phenotype'] = np.where(
    detect_outliers(sub_pheno_df['phenotype'], stratify=pheno_df['Sex']),
    np.nan, sub_pheno_df['phenotype']
)
# Save phenotype:
sub_pheno_df[['FID', 'IID', 'phenotype']].to_csv(f"data/phenotypes/ukbb/WHR.txt",
    sep="\t", index=False, header=False, na_rep='NA'
)

# =============================================================================
# Case/control phenotypes

# Add ICD10 cause of death, primary + secondary
icd10_cols = [f"40001-{i}.0" for i in range(2)] + [f"40002-{i}.{j}" for i, j in itertools.product(range(2), range(14))]
# Add ICD10 diagnoses, main + secondary
icd10_cols += [f"41202-0.{i}" for i in range(80)] + [f"41204-0.{i}" for i in range(210)]
general_illness_cols = [f"20002-{i}.{j}" for i, j in itertools.product(range(2), range(34))]

# Read the diagnosis codes:
df_disease = pd.read_csv(osp.join(ukb_homedir, "Tabular/current.csv"),
                         usecols=['eid'] + icd10_cols + general_illness_cols)

# Remove withdrawn individuals:
df_disease = df_disease[~df_disease['eid'].isin(withdrawn_df['IID'])]
df_disease.rename(columns={'eid': 'IID'}, inplace=True)
df_disease['FID'] = df_disease['IID']

# ------------------ Asthma ------------------

# Extract index of individuals who have been diagnosed with asthma
asthma_idx = np.where(np.logical_or(
    (df_disease[general_illness_cols] == 1111).any(axis=1),
    df_disease[icd10_cols].select_dtypes(include="object").apply(
        lambda col: col.str.startswith("J45", na=False)
    ).any(axis=1)
))[0]

# Extract index of individuals who have asthma-related diagnoses (to be excluded)
asthma_like_idx = np.where(np.logical_or(
    df_disease[general_illness_cols].isin(range(1111, 1126)).any(axis=1),
    df_disease[icd10_cols].select_dtypes(include="object").apply(
        lambda col: col.str.startswith("J", na=False)
    ).any(axis=1)
))[0]

asthma_df = df_disease[['FID', 'IID']].copy()
asthma_df['phenotype'] = 0
asthma_df.iloc[asthma_like_idx, -1] = -9
asthma_df.iloc[asthma_idx, -1] = 1

asthma_df = asthma_df.loc[asthma_df['phenotype'] != -9]
asthma_df.to_csv("data/phenotypes/ukbb/ASTHMA.txt",
    sep="\t", index=False, header=False, na_rep='NA')

# Free up some memory:
del(asthma_df)
del(asthma_idx)
del(asthma_like_idx)

# ------------------ T1D & T2D ------------------

# Extract index of individuals who have general diabetes diagnosis
diabetes_like_idx = np.where(np.logical_or(
    df_disease[general_illness_cols].isin(range(1220, 1224)).any(axis=1),
    df_disease[icd10_cols].select_dtypes(include="object").apply(
        lambda col: col.str.startswith(("E10", "E11", "E12", "E13", "E14"), na=False)
    ).any(axis=1)
))[0]

# Extract index of individuals who have T1D diagnosis
t1d_idx = np.where(np.logical_or(
    (df_disease[general_illness_cols] == 1222).any(axis=1),
    df_disease[icd10_cols].select_dtypes(include="object").apply(
        lambda col: col.str.startswith("E10", na=False)
    ).any(axis=1)
))[0]

# Extract index of individuals who have T2D diagnosis
t2d_idx = np.where(np.logical_or(
    (df_disease[general_illness_cols] == 1223).any(axis=1),
    df_disease[icd10_cols].select_dtypes(include="object").apply(
        lambda col: col.str.startswith("E11", na=False)
    ).any(axis=1)
))[0]

# T1D:
t1d_df = df_disease[['FID', 'IID']].copy()
t1d_df['phenotype'] = 0
t1d_df.iloc[diabetes_like_idx, -1] = -9
t1d_df.iloc[t1d_idx, -1] = 1
t1d_df.iloc[t2d_idx, -1] = -9

t1d_df = t1d_df.loc[t1d_df['phenotype'] != -9]
# t1d_df.to_csv("data/phenotypes/ukbb/T1D.txt",
# sep="\t", index=False, header=False, na_rep='NA')

# T2D:
t2d_df = df_disease[['FID', 'IID']].copy()
t2d_df['phenotype'] = 0
t2d_df.iloc[diabetes_like_idx, -1] = -9
t2d_df.iloc[t2d_idx, -1] = 1
t2d_df.iloc[t1d_idx, -1] = -9

t2d_df = t2d_df.loc[t2d_df['phenotype'] != -9]
t2d_df.to_csv("data/phenotypes/ukbb/T2D.txt",
    sep="\t", index=False, header=False, na_rep='NA')
