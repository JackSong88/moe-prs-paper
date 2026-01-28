import argparse
import glob
import json
import os
import os.path as osp
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from magenpy.utils.system_utils import makedir

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
sys.path.append(osp.join(parent_dir, "score/"))
sys.path.append(osp.join(parent_dir, "evaluation/"))

from baseline_models import AncestryWeightedPRS, MultiPRS
from evaluate_predictive_performance import stratified_evaluation
from moe import MoEPRS
from PRSDataset import PRSDataset


def map_sim_scenario_names(col):
    return col.map(
        {
            "single_model": "Single model",
            "multiprs": "Multiprs",
            "discrete_context (Sex)": "Discrete context (Sex)",
            "discrete_context (Ancestry)": "Discrete context (Ancestry)",
            "continuous_context (Age)": "Continuous context (Age)",
            "moe": "Mixture-of-Experts",
        }
    )


def get_sim_order(sims):
    return [
        s
        for s in [
            "Single model",
            "Multiprs",
            "Mixture-of-Experts",
            "Discrete context (Sex)",
            "Discrete context (Ancestry)",
            "Continuous context (Age)",
        ]
        if s in sims
    ]


def extract_trained_models(dataset_path, model_subset=None):
    """
    For a given dataset path, extract trained models from the specified model subset.
    """

    trained_models_path = osp.dirname(
        dataset_path.replace("harmonized_data", "trained_models")
    )
    trained_models_path = osp.join(trained_models_path, "*", "*.pkl")

    trained_models = {}

    for f in glob.glob(trained_models_path):
        model_name = osp.basename(f).replace(".pkl", "")

        if model_subset is not None:
            if model_name not in model_subset:
                continue

        if "moe" in model_name.lower():
            trained_models[model_name] = MoEPRS.from_saved_model(f)
        else:
            trained_models[model_name] = MultiPRS.from_saved_model(f)

    if len(trained_models) == 0:
        raise FileNotFoundError(f"No trained models found in {trained_models_path}")

    return trained_models


def evaluate_prediction_accuracy_on_dataset(dataset_path):
    print("Evaluating:", dataset_path)
    prs_dataset = PRSDataset.from_pickle(dataset_path)

    # Load the simulation configuration from the pickle file:
    config_path = osp.join(osp.dirname(dataset_path), "config.pkl")
    with open(config_path, "rb") as f:
        simulation_config = pickle.load(f)

    # Load the models that were trained on this dataset:
    trained_models = extract_trained_models(
        dataset_path,
        model_subset=["MoE-global-int", "MultiPRS"],
    )

    eval_result = stratified_evaluation(prs_dataset, trained_models)
    eval_result = eval_result.loc[
        eval_result["PGS"].isin(["MoE-global-int", "MultiPRS"])
    ]
    eval_result["PGS"] = eval_result["PGS"].map(
        {"MoE-global-int": "MoEPRS", "MultiPRS": "MultiPRS"}
    )
    eval_result["Heritability"] = simulation_config["heritability"]

    eval_result["Simulation Scenario"] = simulation_config["simulation_type"]

    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictive performance")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of jobs to launch when performing evaluation",
    )
    parser.add_argument(
        "--phenotype",
        type=str,
        default="HEIGHT",
        help="The phenotype to plot the simulation results for.",
    )
    args = parser.parse_args()

    predictive_perf = []

    dataset_paths = glob.glob(
        f"data/harmonized_data_simulations/sim_*/{args.phenotype}/ukbb/*_h0.*/test_data.pkl"
    )
    predictive_perf = Parallel(n_jobs=args.jobs, backend="multiprocessing")(
        delayed(evaluate_prediction_accuracy_on_dataset)(path) for path in dataset_paths
    )

    predictive_perf = pd.concat(predictive_perf)
    predictive_perf["Simulation Scenario"] = map_sim_scenario_names(
        predictive_perf["Simulation Scenario"]
    )
    predictive_perf.rename(
        columns={"Simulation Scenario": "Scenario", "PGS": "Model"}, inplace=True
    )

    sns.set_context("paper", font_scale=2.25)

    g = sns.catplot(
        data=predictive_perf,
        x="Heritability",
        col="Scenario",
        col_order=get_sim_order(predictive_perf["Scenario"].unique()),
        hue_order=["MoEPRS", "MultiPRS"],
        col_wrap=3,
        y="Incremental_R2",
        kind="box",
        showfliers=False,
        hue="Model",
        palette={"MoEPRS": "#375E97", "MultiPRS": "#FFBB00"},
    )

    for ax in g.axes.flat:
        title = ax.get_title()
        if title.startswith("Scenario = "):
            ax.set_title(title.replace("Scenario = ", ""))

    g.set_ylabels("Incremental $R^2$")

    plt.savefig(f"figures/simulations/predictive_performance_{args.phenotype}.eps")
