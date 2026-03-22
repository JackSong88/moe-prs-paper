import argparse
import glob
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from error_bars import add_error_bars_to_catplot
from magenpy.utils.system_utils import makedir
from plot_predictive_performance import generate_model_colors, postprocess_metrics_df
from plot_utils import read_eval_metrics, sort_groups, transform_eval_metrics
from significance_annotation import add_significance_annotations

# ---------------------------------------------------------------------------


def plot_combined_accuracy_metrics(
    metrics_df,
    output_f=None,
    metric="Incremental_R2",
    palette="Set2",
    hue_order=None,
    col_order=None,
    col_wrap=None,
    test_models=None,  # Test if models are significantly different
    significance_symbols=None,
    sharey=False,
    height=5,
    aspect=1,
):
    # ---------------------------------------------------------------------
    # Sanity checks / preparation

    if test_models is not None:
        if len(test_models) == 2 and isinstance(test_models[0], str):
            test_models = [test_models]

        for tm in test_models:
            assert len(tm) == 2

        assert f"{metric}_err" in metrics_df

    if hue_order is None:
        _, hue_order = generate_model_colors(metrics_df, metric)

    sorted_groups = sort_groups(metrics_df["Evaluation Group"].unique())

    # ---------------------------------------------------------------------

    grid = sns.catplot(
        x="Evaluation Group",
        y=metric,
        col="Phenotype",
        col_wrap=col_wrap,
        col_order=col_order,
        order=sorted_groups,
        hue="Model Name",
        palette=palette,
        hue_order=hue_order,
        kind="bar",
        height=height,
        aspect=aspect,
        sharey=sharey,
        data=metrics_df,
    )

    if f"{metric}_err" in metrics_df.columns:
        add_error_bars_to_catplot(
            grid,
            metrics_df,
            "Evaluation Group",
            metric,
            hue="Model Name",
            hue_order=hue_order,
            col="Phenotype",
        )

        if test_models is not None:
            add_significance_annotations(
                grid,
                metrics_df,
                "Evaluation Group",
                metric,
                f"{metric}_err",
                hue="Model Name",
                hue_order=hue_order,
                test_pairs=test_models,
                x_labels=sorted_groups,
                symbols=significance_symbols,
            )

    grid.set_axis_labels(
        x_var="Evaluation Group",
        y_var={
            "Incremental_R2": "Incremental $R^2$",
            "Liability_R2": "Liability $R^2$",
            "CORR": "Pearson $R$",
        }[metric],
    )

    for ax in grid.axes.flat:
        title = ax.get_title()
        if title.startswith("Phenotype = "):
            ax.set_title(title.replace("Phenotype = ", ""))

    if output_f is None:
        plt.show()
    else:
        plt.savefig(output_f)
        plt.close()

    return grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot predictive performance of PRS models by category."
    )

    parser.add_argument(
        "--biobank",
        dest="biobank",
        type=str,
        required=True,
        choices={"ukbb", "cartagene"},
        help="The name of the biobank to plot the accuracy metrics for.",
    )
    parser.add_argument(
        "--category",
        dest="category",
        type=str,
        default=["Ancestry"],
        nargs="+",
        help="The category (or list of categories) to plot the predictive performance for.",
    )
    parser.add_argument(
        "--aggregate-single-prs",
        dest="aggregate_single_prs",
        action="store_true",
        default=False,
        help="Aggregate the results for SinglePRS models (select best for each category).",
    )
    parser.add_argument(
        "--restrict-to-same-biobank",
        dest="restrict_to_same_biobank",
        action="store_true",
        default=False,
        help="Restrict the analysis to models trained and tested on the same biobank.",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        choices={"train", "test"},
        default="test",
        help="The type of dataset to plot predictive performance on.",
    )
    parser.add_argument(
        "--extension",
        dest="extension",
        type=str,
        default=".png",
        help="The file extension to use for saving the plot.",
    )
    parser.add_argument(
        "--moe-model",
        dest="moe_model",
        type=str,
        default="MoE",
        help="The name of the MoE model to plot as reference.",
    )

    args = parser.parse_args()

    sns.set_context("paper", font_scale=1.5)

    phenotype_cats = {
        "binary": ["ASTHMA", "T2D"],
        "continuous": [
            "HEIGHT",
            "BMI",
            "HDL",
            "LDL",
            "LOG_TG",
            "TC",
            "DBP",
            "DBP_adj",
            "LDL_adj",
            "SBP",
            "SBP_adj",
            "TC_adj",
        ],
        "sex_biased": ["TST", "URT", "CRTN", "WHR"],
    }

    metric = {
        "binary": "Liability_R2",
        "continuous": "Incremental_R2",
        "sex_biased": "Incremental_R2",
    }

    stratification_variable = {
        "binary": ["Ancestry", "Coarse Ancestry"],
        "continuous": ["Ancestry", "Coarse Ancestry"],
        "sex_biased": ["Sex"],
    }

    metrics_dfs = {}

    for f in glob.glob(f"data/evaluation/*/{args.biobank}/{args.dataset}_data.csv"):
        pheno = f.split("/")[-3]
        try:
            pheno_cat = [k for k, v in phenotype_cats.items() if pheno in v][0]
        except IndexError:
            continue

        df = transform_eval_metrics(read_eval_metrics(f))

        keep_models_moe = [f"{args.moe_model} ({args.biobank})"]
        if not args.restrict_to_same_biobank:
            other_biobank = ["cartagene", "ukbb"][args.biobank == "cartagene"]
            keep_models_moe.append(f"{args.moe_model} ({other_biobank})")

        df = df.loc[
            (df["Model Category"] != "MoE") | df["Model Name"].isin(keep_models_moe)
        ]

        if args.restrict_to_same_biobank:
            df = df.loc[df["Training biobank"] == df["Test biobank"]]

        for eval_cat in stratification_variable[pheno_cat]:
            eval_df = postprocess_metrics_df(
                df,
                metric[pheno_cat],
                category=eval_cat,
                aggregate_single_prs=args.aggregate_single_prs,
            )

            metric_cat = f"{pheno_cat}_{eval_cat}"

            if metric_cat not in metrics_dfs:
                metrics_dfs[metric_cat] = [eval_df]
            else:
                metrics_dfs[metric_cat].append(eval_df)

    output_dir = f"figures/accuracy/{args.biobank}/{args.dataset}/"
    output_dir = osp.join(
        output_dir, ["", "same_biobank"][args.restrict_to_same_biobank]
    )

    makedir(output_dir)

    for pheno_eval_cat, dfs in metrics_dfs.items():
        if len(dfs) < 1:
            raise ValueError(
                f"No data to plot after applying filters for {pheno_eval_cat}."
            )

        pheno_cat = "_".join(pheno_eval_cat.split("_")[:-1])

        plot_combined_accuracy_metrics(
            pd.concat(dfs, axis=0).reset_index(drop=True),
            osp.join(
                output_dir, f"combined_metrics_{args.moe_model}_{pheno_eval_cat}.eps"
            ),
            metric=metric[pheno_cat],
            col_order=phenotype_cats[pheno_cat],
            col_wrap=min(5, len(phenotype_cats[pheno_cat])),
            test_models=[
                (f"{args.moe_model} ({args.biobank})", f"MultiPRS ({args.biobank})"),
                (f"{args.moe_model} ({args.biobank})", "Best Single Source PRS"),
            ],
            significance_symbols=("*", "+"),  # "◆"),
        )
