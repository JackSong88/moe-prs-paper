import os.path as osp
import os
import sys
import copy
import pickle
import argparse

import numpy as np
import pandas as pd
import seaborn as sns

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))

from magenpy.utils.system_utils import makedir
from gate_interpretation import plot_expert_weights
from PRSDataset import PRSDataset
from moe import MoEPRS
from plot_utils import sort_groups, GROUP_MAP, MODEL_NAME_MAP

# Torch/Lightning only needed for .pt
import torch
from moe_pytorch import Lit_MoEPRS
from moe_pytorch_inference import load_model_any

def plot_admixture_graphs(prs_dataset,
                          model,
                          title=None,
                          output_file=None,
                          group_col=None,
                          min_group_size=50,
                          subsample_within_groups=False,
                          agg_mechanism='mean',
                          figsize='auto',
                          palette='Set3',
                          sorted_groups=None,
                          drop_legend=False,
                          tick_rotation=90):

    assert agg_mechanism in ['mean', 'sort'], "Aggregation mechanism must be either 'mean' or 'sort'."

    prs_dataset.set_backend("numpy")

    proba = np.asarray(model.predict_proba(prs_dataset))

    # Map the PRS IDs:
    mapped_prs_ids = []
    for prs_id in model.expert_cols:
        mapped_prs_ids.append(MODEL_NAME_MAP.get(prs_id, prs_id))

    proba = pd.DataFrame(proba, columns=mapped_prs_ids)

    if group_col is not None:

        proba[group_col] = prs_dataset.get_data_columns(group_col).flatten()

        # Filter tiny groups:
        if min_group_size is not None and min_group_size > 0:
            group_counts = proba[group_col].value_counts()
            group_counts = group_counts[group_counts >= min_group_size]
            proba = proba[proba[group_col].isin(group_counts.index)]

        # Map the group names:
        if group_col == 'Sex':
            proba[group_col] = proba[group_col].astype(int).astype(str).map(GROUP_MAP).fillna(proba[group_col])

        if sorted_groups is None and group_col in ('Ancestry', 'UMAP_Cluster'):
            sorted_groups = sort_groups(proba[group_col].unique())

        if subsample_within_groups:
            median_group_size = min(int(np.median(proba.groupby(group_col).size())), 1000)

            def cond_subsample_func(x):
                if len(x) > 2 * median_group_size:
                    return x.sample(2 * median_group_size)
                else:
                    return x

            proba = proba.groupby(group_col).apply(cond_subsample_func).reset_index(drop=True)

        if figsize == 'auto':
            if agg_mechanism == 'sort' and sorted_groups is not None:
                figsize = (25, 5)
            else:
                figsize = (12, 6)

        return plot_expert_weights(proba,
                                   agg_col=group_col,
                                   agg_mechanism=agg_mechanism,
                                   agg_order=sorted_groups,
                                   figsize=figsize,
                                   title=title,
                                   palette=palette,
                                   output_file=output_file,
                                   drop_legend=drop_legend,
                                   tick_rotation=tick_rotation)
    else:
        return plot_expert_weights(proba,
                                   title=title,
                                   palette=palette,
                                   agg_order=sorted_groups,
                                   output_file=output_file,
                                   drop_legend=drop_legend,
                                   tick_rotation=tick_rotation)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Plot the admixture graph (gate probabilities) for a given model and dataset.'
    )

    parser.add_argument('--model', dest='model', type=str, required=True,
                        help='Path to trained model: either .pkl (MoEPRS) or .pt (MoE-PyTorch.pt).')
    parser.add_argument('--dataset', dest='dataset', type=str, required=True,
                        help='Path to harmonized PRSDataset .pkl.')
    parser.add_argument('--group-col', dest='group_col', type=str, nargs='+', default=None,
                        help='Column(s) to stratify by (e.g., Ancestry, Sex, UMAP_Cluster).')
    parser.add_argument('--agg-mechanism', dest='agg_mechanism', type=str, default='sort',
                        choices={'mean', 'sort'},
                        help='Aggregation mechanism: mean (group-average) or sort (individual bars).')
    parser.add_argument('--extension', dest='extension', type=str, default='.png',
                        help='File extension for plots.')
    parser.add_argument('--subsample', dest='subsample', action='store_true', default=False,
                        help='Subsample within large groups for cleaner sort plots.')
    parser.add_argument('--torch-batch-size', dest='torch_batch_size', type=int, default=65536,
                        help='Batch size when computing gate probs for .pt models.')

    args = parser.parse_args()

    sns.set_context("paper", font_scale=2)

    p_dataset = PRSDataset.from_pickle(args.dataset)
    moe_like = load_model_any(
        p_dataset,
        args.model,
        gate_batch_size=args.torch_batch_size,
    )

    # mirror your previous output folder logic
    data_path = args.dataset.replace('data/harmonized_data', 'figures/admixture_graphs').replace('.pkl', '')
    model_path = '_'.join(args.model.replace('.pkl', '').replace('.pt', '').split('/')[-3:])

    makedir(data_path)

    if args.group_col is None:
        plot_output_file = osp.join(data_path, model_path + args.extension)
        plot_admixture_graphs(p_dataset, moe_like, output_file=plot_output_file)
    else:
        for gcol in args.group_col:
            plot_admixture_graphs(
                p_dataset,
                moe_like,
                group_col=gcol,
                output_file=osp.join(data_path, model_path + f'_{gcol}{args.extension}'),
                agg_mechanism=args.agg_mechanism,
                subsample_within_groups=args.subsample
            )
