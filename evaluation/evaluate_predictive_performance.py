import numpy as np
import pandas as pd
import argparse
import glob
import os
import os.path as osp
import sys
from magenpy.utils.system_utils import makedir
from viprs.eval.binary_metrics import roc_auc, pr_auc, liability_r2, nagelkerke_r2
from viprs.eval.continuous_metrics import mse, r2, incremental_r2, partial_correlation, pearson_r
from viprs.eval.eval_utils import r2_stats

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
sys.path.append(osp.join(parent_dir, "score/"))

from PRSDataset import PRSDataset
from moe import MoEPRS
from baseline_models import MultiPRS, AncestryWeightedPRS
import torch
from moe_pytorch import Lit_MoEPRS
<<<<<<< Updated upstream
=======
from moe_pytorch_inference import load_model_any
>>>>>>> Stashed changes


from eval_utils import (
    generate_predictions,
    generate_categorical_masks,
    generate_continuous_masks,
    generate_pc_cluster_masks
)

import numpy as np
from sklearn.metrics import average_precision_score

def average_precision_at_top_percentile(y_true, y_pred, percentile=0.05):
    """
    Computes average precision for identifying the top percentile of y_true using y_pred.

    Parameters:
    - y_true: array-like, true continuous target values
    - y_pred: array-like, predicted scores
    - percentile: float, e.g., 0.05 for top 5%

    Returns:
    - average_precision: float, average precision score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Determine the threshold to consider top percentile
    threshold = np.percentile(y_true, 100 * (1 - percentile))  # top X%

    # Binary labels: 1 if in top X%, else 0
    y_top = (y_true >= threshold).astype(int)

    # Average precision score
    ap = average_precision_score(y_top, y_pred)

    return ap

<<<<<<< Updated upstream

class TorchMoEWrapper:
    ''' inference wrapper around a trained Lit_MoEPRS model '''
    def __init__(self, lit_model, scaler_path=None):
        self.lit_model = lit_model
        self.lit_model.eval()
        self.scaler_path = scaler_path

    def predict(self, prs_dataset):
        # expected column groups used during training
        expected = self.lit_model.group_getitem_cols

        # Ensure the dataset exposes the same group_getitem_cols interface
        # required by the trained Lightning model
        if not getattr(prs_dataset, "group_getitem_cols", None):
            prs_dataset.set_group_getitem_cols(expected)
        else:
            # reset if not matching
            need = ("experts","gate_input","phenotype")
            if any(k not in prs_dataset.group_getitem_cols for k in need) or \
            any(prs_dataset.group_getitem_cols[k] != expected[k] for k in ("experts","gate_input")):
                prs_dataset.set_group_getitem_cols(expected)

        # load the saved scaler used during training
        import pickle
        with open(osp.join(self.scaler_path, "MoE-PyTorch.scaler.pkl"), "rb") as f:
            loaded_scaler = pickle.load(f)

        # ensure the dataset applies the same training scaler without refitting
        import copy
        d = copy.deepcopy(prs_dataset)
        d.standardize_data(scaler=loaded_scaler, refit=False)

        #inference only 
        with torch.no_grad():
            return self.lit_model.predict_from_dataset(d)

def load_lit_from_pt(prs_dataset, pt_path):
    """
    Rebuild the Lightning module with the right config and load weights from a .pt file.
    """

    #load checkpoint (weights and training config)
    checkpoint = torch.load(pt_path, map_location="cpu")
    if "config" not in checkpoint:
        raise ValueError(f"Missing 'config' in checkpoint: {pt_path}")

    cfg = checkpoint["config"]

    # --- rebuild group_getitem_cols to match training ---
    group_getitem_cols = {
        "phenotype": [prs_dataset.phenotype_col],
        "gate_input": prs_dataset.covariates_cols,
        "experts": prs_dataset.prs_cols,
        "global_input": prs_dataset.covariates_cols
    }

    # include per-exert covariates if used during training
    if cfg.get("has_expert_covariates", False):
        group_getitem_cols["expert_covariates"] = prs_dataset.covariates_cols

    # --- reconstruct Lightning model from config ---
    lit = Lit_MoEPRS(
        group_getitem_cols=group_getitem_cols,
        gate_model_layers=cfg["gate_model_layers"],
        gate_add_batch_norm=cfg["gate_add_batch_norm"],
        loss=cfg["loss"],
        optimizer=cfg["optimizer"],
        family=cfg["family"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        topk_k=cfg.get("topk_k", None),
        tau_start=cfg.get("tau_start", 1.0),
        tau_end=cfg.get("tau_end", 1.0),
        hard_ste=cfg.get("hard_ste", True),
        lb_coef=cfg.get("lb_coef", 0.0),
        eps=cfg.get("eps", 1e-12),

        use_ard_bias=cfg.get("use_ard_bias", True),
        use_global_head=cfg.get("use_global_head", True),

        ent_coef=cfg.get("ent_coef_start", 0.0),
        ent_coef_end=cfg.get("ent_coef_end", None),
        ent_warm_epochs=cfg.get("ent_warm_epochs", 0),
        ent_decay_epochs=cfg.get("ent_decay_epochs", 0),
    )

    # non-state attrs that affect forward
    if cfg.get("min_sigma2", None) is not None:
        lit.min_sigma2 = float(cfg["min_sigma2"])
    if "expert_bias_scale_floor" in cfg:
        lit.expert_bias_scale_floor = float(cfg["expert_bias_scale_floor"])

    # --- load trained weights ---
    lit.load_state_dict(checkpoint["state_dict"], strict=True)

    return lit

=======
>>>>>>> Stashed changes
def stratified_evaluation(prs_dataset,
                          trained_models=None,
                          cat_group_cols=None,
                          cont_group_cols=None,
                          cont_group_bins=None,
                          pc_clusters=None,
                          min_group_size=30):

    if cont_group_cols is not None:
        assert cont_group_bins is not None, "Bins must be provided for continuous group columns!"

    prs_dataset.set_backend("numpy")

    if trained_models is None or len(trained_models) == 0:
        preds = None
    else:
        preds = generate_predictions(prs_dataset, trained_models)

    # Generate sample masks to stratify the dataset:
    msks = {}
    if cat_group_cols is not None:
        msks.update(generate_categorical_masks(prs_dataset, cat_group_cols, min_group_size))
    if cont_group_cols is not None:
        msks.update(generate_continuous_masks(prs_dataset, cont_group_cols, cont_group_bins))
    if pc_clusters is not None:
        msks.update(generate_pc_cluster_masks(prs_dataset, 'median', pc_clusters))

    dfs = []

    # Evaluate the models across everyone:

    edf = evaluate_prs_models(prs_dataset, other_models=preds)
    edf['EvalCategory'] = 'All'
    edf['EvalGroup'] = 'All'
    edf['N'] = prs_dataset.N

    dfs.append(edf)

    for mg, msk_group in msks.items():
        print("> Evaluation group:", mg)
        for m, msk in msk_group.items():
            print("\t> Subgroup:", m)
            edf = evaluate_prs_models(prs_dataset, other_models=preds, mask=msk, min_group_size=min_group_size)

            if edf is None:
                continue

            edf['EvalCategory'] = mg
            edf['EvalGroup'] = m

            if msk is None:
                edf['N'] = prs_dataset.N
            else:
                edf['N'] = sum(msk)

            dfs.append(edf)

    return pd.concat(dfs)


def evaluate_prs_models(prs_dataset,
                        other_models=None,
                        mask=None,
                        min_group_size=30):

    prs_dataset.set_backend("numpy")

    if mask is None:
        mask = np.ones(prs_dataset.N).astype(bool)

    if mask.sum() < min_group_size:
        print(f"Skipping evaluation due to insufficient sample size ({mask.sum()} < {min_group_size})")
        return None

    if prs_dataset.phenotype_likelihood == 'gaussian':
        metrics = ('CORR', 'MSE', 'Incremental_R2', 'Partial_CORR', 'AVG_PREC_TOP5')
    else:
        metrics = ('ROC_AUC', 'PR_AUC', 'Liability_R2', 'Nagelkerke_R2')

    prs_df = pd.DataFrame(prs_dataset.get_prs_predictions()[mask, :],
                          columns=prs_dataset.prs_ids)

    phenotype = prs_dataset.get_phenotype().flatten()[mask]

    if other_models is not None:
        prs_df = pd.concat([prs_df, other_models.loc[mask, :].reset_index(drop=True)], axis=1)

    if any([m in metrics for m in ('Incremental_R2', 'Partial_CORR', 'Liability_R2', 'Nagelkerke_R2')]):
        covar = pd.DataFrame(prs_dataset.get_covariates()[mask, :])
    else:
        covar = None

    # Remove columns with NaN values:
    #na_cols = prs_df.isna().sum(axis=0) > 0
    #if na_cols.any():
    #    print("Removing the following columns due to NaN values:", prs_df.columns[na_cols])
    #    prs_df.dropna(axis=1, inplace=True)

    # Remove invariant columns from the covariates:
    # Mainly relevant when evaluating on age groups or sex...
    if covar is not None:
        covar = covar.loc[:, covar.var(axis=0) > 0]

    metrics_df = pd.DataFrame({'PGS': prs_df.columns})

    for metric in metrics:

        metrics_df[metric] = np.nan
        pgs_metrics = []
        pgs_err = []

        for pgs in prs_df.columns:

            # Keep only records with valid PGS values:
            keep = ~np.isnan(prs_df[pgs].values)
            n = keep.sum()
            pgs_values = prs_df[pgs].values[keep]
            phenotype_values = phenotype[keep]

            if n < min_group_size:
                pgs_metrics.append(np.nan)
                if metric in ('Incremental_R2', 'Liability_R2', 'Nagelkerke_R2'):
                    pgs_err.append(np.nan)
                continue

            if metric == 'CORR':
                pgs_metrics.append(pearson_r(phenotype_values, pgs_values))
            elif metric == 'MSE':
                pgs_metrics.append(mse(phenotype_values, pgs_values))
            elif metric == 'AVG_PREC_TOP5':
                pgs_metrics.append(average_precision_at_top_percentile(phenotype_values, pgs_values, percentile=0.05))
            elif metric == 'Incremental_R2':
                pgs_metrics.append(incremental_r2(phenotype_values, pgs_values,
                                                  covar.loc[keep, :]))
            elif metric == 'Partial_CORR':
                pgs_metrics.append(partial_correlation(phenotype_values, pgs_values,
                                                       covar.loc[keep, :]))
            elif metric in ('ROC_AUC', 'PR_AUC'):
                if phenotype_values.sum() in (0, phenotype_values.shape[0]):
                    pgs_metrics.append(np.nan)
                else:
                    if metric == 'ROC_AUC':
                        pgs_metrics.append(roc_auc(phenotype_values, pgs_values))
                    else:
                        pgs_metrics.append(pr_auc(phenotype_values, pgs_values))
            elif metric == 'Liability_R2':
                pgs_metrics.append(liability_r2(phenotype_values, pgs_values,
                                                covar.loc[keep, :]))
            elif metric == 'Nagelkerke_R2':
                try:
                    pgs_metrics.append(nagelkerke_r2(phenotype_values, pgs_values,
                                                     covar.loc[keep, :]))
                except Exception:
                    pgs_metrics.append(0.)

            if metric in ('Incremental_R2', 'Liability_R2', 'Nagelkerke_R2'):
                try:
                    pgs_err.append(r2_stats(pgs_metrics[-1], n)['SE'])
                except AssertionError:
                    pgs_err.append(0.)

        metrics_df[metric] = pgs_metrics
        if len(pgs_err) > 0:
            metrics_df[f'{metric}_err'] = pgs_err

    return metrics_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluate PRS models")

    parser.add_argument("--test-data", dest="test_data", type=str, required=True,
                        help="The path to the test data file.")
    parser.add_argument("--cat-group-cols", dest="cat_group_cols", type=str, nargs='+', default=None,
                        help="The columns to use for categorical stratification.")
    parser.add_argument("--cont-group-cols", dest="cont_group_cols", type=str, nargs='+', default=None,
                        help="The columns to use for continuous stratification.")
    parser.add_argument("--cont-group-bins", dest="cont_group_bins", type=int, nargs='+', default=None,
                        help="The number of bins to use for continuous stratification.")
    parser.add_argument("--pc-clusters", dest="pc_clusters", type=int, default=None,
                        help="The number of PC clusters to use for stratification.")
    parser.add_argument('--min-group-size', dest='min_group_size', type=int, default=30,
                        help="The minimum group size to consider for evaluation.")
    parser.add_argument("--model-source", dest="model_source", type=str, default=None,
                    help="Optional substring filter for model directory names (e.g. 'subsampled' or 'train')")


    args = parser.parse_args()

    print(f"Evaluating Meta PRS performance on {args.test_data}")

    prs_dataset = PRSDataset.from_pickle(args.test_data)

    # Obtain path for relevant trained models:

    trained_models_root = osp.dirname(osp.dirname(args.test_data.replace('harmonized_data', 'trained_models')))

    if args.model_source is None:
        # existing behavior (search all subfolders)
        trained_models_path = osp.join(trained_models_root, "*", "*", "*.pkl")
    else:
        # filter subfolder by pattern
        trained_models_path = osp.join(trained_models_root, "*", f"{args.model_source}","*.pkl")


    trained_models = {}

    for f in glob.glob(trained_models_path):
        if "scaler" in f.lower():
            continue  # skip scaler for pt

        model_name = osp.basename(f).replace(".pkl", "")
        split_fname = f.split('/')
        model_prefix = split_fname[-3] + '/' + split_fname[-2] + ':'
        model_name = model_prefix + model_name

        if 'moe' in model_name.lower():
            trained_models[model_name] = MoEPRS.from_saved_model(f)
        elif 'AncestryWeightedPRS' in model_name:
            trained_models[model_name] = AncestryWeightedPRS.from_saved_model(f)
        else:
            trained_models[model_name] = MultiPRS.from_saved_model(f)
    
    if args.model_source is None:
        pt_glob = osp.join(trained_models_root, "*", "*", "MoE-PyTorch.pt")
    else:
        pt_glob = osp.join(trained_models_root, "*", f"{args.model_source}", "MoE-PyTorch.pt")

    for f in glob.glob(pt_glob):
        # mirror the naming pattern used for .pkl models
        split_fname = f.split('/')
        model_prefix = split_fname[-3] + '/' + split_fname[-2] + ':'
        model_name = model_prefix + 'MoE-PyTorch'

<<<<<<< Updated upstream
        # rebuild Lightning module and wrap it to expose .predict(prs_dataset)
        lit = load_lit_from_pt(prs_dataset, f)


        model_dir = osp.dirname(f)

        trained_models[model_name] = TorchMoEWrapper(
            lit_model=lit,
            scaler_path=model_dir
=======
        trained_models[model_name] = load_model_any(
            prs_dataset,
            f,
>>>>>>> Stashed changes
        )

    if len(trained_models) == 0:
        raise FileNotFoundError(f"No trained models found in {trained_models_path}")

    eval_df = stratified_evaluation(prs_dataset,
                                    trained_models,
                                    cat_group_cols=args.cat_group_cols,
                                    cont_group_cols=args.cont_group_cols,
                                    cont_group_bins=args.cont_group_bins,
                                    pc_clusters=args.pc_clusters,
                                    min_group_size=args.min_group_size)

    output_path = args.test_data.replace('harmonized_data', 'evaluation').replace('.pkl', '.csv')

    print("> Saving evaluation metrics to:\n\t", output_path)

    makedir(os.path.dirname(output_path))
    eval_df.to_csv(output_path, index=False)
