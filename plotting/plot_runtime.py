import glob
import json
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def extract_runtime_real_phenotypes():
    rt_df = pd.read_csv("tables/model_runtimes.tsv", sep="\t")
    rt_df = rt_df.loc[
        (rt_df["variation"] == "train_data")
        & rt_df["model"].isin(["MoE-global-int", "MultiPRS", "MoE-PyTorch"])
    ]
    rt_df["Model"] = rt_df["model"].map(
        {
            "MoE-PyTorch": "MoEPRS-SGD",
            "MoE-global-int": "MoEPRS-EM",
            "MultiPRS": "MultiPRS",
        }
    )

    return rt_df[["Model", "Runtime_min"]].copy().assign(Dataset="Real Phenotypes")


def extract_runtime_simulations():
    rt_res = []

    for f in glob.glob(
        "data/trained_models_simulations/sim_*/*/ukbb/*/train_data/*.json"
    ):
        model_name = osp.basename(f).replace("_runtime.json", "")
        if model_name in ["MultiPRS", "MoE-global-int"]:
            with open(f) as of:
                rt_res.append(json.load(of))
                rt_res[-1]["Model"] = {
                    "MultiPRS": "MultiPRS",
                    "MoE-global-int": "MoEPRS-EM",
                }[model_name]

    return pd.DataFrame(rt_res).assign(Dataset="Simulated Phenotypes")


# For now, focus on real phenotypes:
runtime_results = extract_runtime_real_phenotypes()

# Set a professional theme
sns.set_context("paper", font_scale=1.75)
sns.set_theme(style="whitegrid")

g = sns.boxplot(
    data=runtime_results,
    x="Model",
    y="Runtime_min",
    hue="Model",
    order=["MoEPRS-EM", "MoEPRS-SGD", "MultiPRS"],
    hue_order=["MoEPRS-EM", "MoEPRS-SGD", "MultiPRS"],
    showmeans=True,
    showfliers=False,
    palette={"MoEPRS-EM": "#375E97", "MoEPRS-SGD": "#3B6F7F", "MultiPRS": "#FFBB00"},
    meanprops={  # Optional: Customize the mean marker look
        "marker": "D",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "6",
    },
)
# Axis labels
plt.xlabel("Model")
plt.ylabel("Runtime (minutes)")
# Title
plt.title("Runtime of Ensemble PRS methods in the UK Biobank")
plt.tight_layout()
plt.savefig("figures/method_runtime.eps")
plt.close()
