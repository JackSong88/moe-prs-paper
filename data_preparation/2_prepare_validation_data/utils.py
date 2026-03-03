import numpy as np


def detect_outliers(phenotype, sigma_threshold=3, stratify=None, nan_policy="omit"):
    """
    Detect samples with outlier phenotype values.
    This function takes a vector of quantitative phenotypes,
    computes the z-score for every individual, and returns a
    boolean vector indicating whether individual i has phenotype value
    within the specified standard deviations `sigma_threshold`.
    :param phenotype: A numpy vector of continuous or quantitative phenotypes.
    :param sigma_threshold: The multiple of standard deviations or sigmas after
    which we consider the phenotypic value an outlier.
    :param stratify: A numpy array indicating group membership to stratify the outlier detection.
    :param nan_policy: The policy to use when encountering NaN values in the phenotype vector.
    By default, we compute the z-scores ignoring NaN values.

    :return: A boolean array indicating whether the phenotype value is an outlier (i.e.
    True indicates outlier).
    """

    from scipy.stats import zscore

    if stratify is None:
        stratify = np.ones_like(phenotype)

    mask = np.zeros_like(phenotype, dtype=bool)

    for group in np.unique(stratify):
        mask[stratify == group] = (
            np.abs(zscore(phenotype[stratify == group], nan_policy=nan_policy))
            > sigma_threshold
        )

    return mask


def adjust_ldl_cholesterol_for_medication(dat, med_use_df):
    """
    Based on the GLGC consortium:
        https://pmc.ncbi.nlm.nih.gov/articles/PMC8730582/#S7
    """

    merged_tab = dat.merge(med_use_df, on="IID")
    merged_tab["phenotype"] = np.where(
        merged_tab["chol_med"], merged_tab["phenotype"] / 0.7, merged_tab["phenotype"]
    )

    return merged_tab[["FID", "IID", "phenotype"]].copy()


def adjust_total_cholesterol_for_medication(dat, med_use_df):
    """
    Based on the GLGC consortium:
        https://pmc.ncbi.nlm.nih.gov/articles/PMC8730582/#S7
    """
    merged_tab = dat.merge(med_use_df, on="IID")
    merged_tab["phenotype"] = np.where(
        merged_tab["chol_med"], merged_tab["phenotype"] / 0.8, merged_tab["phenotype"]
    )

    return merged_tab[["FID", "IID", "phenotype"]].copy()


def adjust_systolic_blood_pressure_for_medication(dat, med_use_df):
    """
    Based on: https://www.nature.com/articles/s41588-018-0205-x#Sec12
    """

    merged_tab = dat.merge(med_use_df, on="IID")
    merged_tab["phenotype"] = np.where(
        merged_tab["bp_med"], merged_tab["phenotype"] + 15, merged_tab["phenotype"]
    )

    return merged_tab[["FID", "IID", "phenotype"]].copy()


def adjust_diastolic_blood_pressure_for_medication(dat, med_use_df):
    """
    Based on: https://www.nature.com/articles/s41588-018-0205-x#Sec12
    """
    merged_tab = dat.merge(med_use_df, on="IID")
    merged_tab["phenotype"] = np.where(
        merged_tab["bp_med"], merged_tab["phenotype"] + 10, merged_tab["phenotype"]
    )

    return merged_tab[["FID", "IID", "phenotype"]].copy()
