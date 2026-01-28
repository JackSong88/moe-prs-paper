import copy
import os.path as osp
import sys

sys.path.append(osp.dirname(osp.dirname(__file__)))
sys.path.append(osp.dirname(osp.dirname(osp.dirname(__file__))))
import argparse
import json

from baseline_models import AncestryWeightedPRS, MultiPRS
from magenpy.utils.system_utils import makedir
from moe import MoEPRS
from PRSDataset import PRSDataset

from moe_pytorch import make_deterministic, Lit_MoEPRS, train_model
from utils import Timer
import copy

def train_baseline_linear_models(
    dataset, penalty_type=None, penalty=0.0, class_weights=None, add_intercept=True
):
    dataset.set_backend("numpy")

    print(
        f"> Training baseline models for {dataset.phenotype_col} with {dataset.N} samples..."
    )

    base_models = dict()
    runtimes = dict()

    base_models["MultiPRS"] = MultiPRS(
        prs_dataset=dataset,
        expert_cols=dataset.prs_cols,
        covariates_cols=dataset.covariates_cols,
        add_intercept=add_intercept,
        class_weights=class_weights,
        penalty_type=penalty_type,
        penalty=penalty,
    )

    with Timer() as timer:
        base_models["MultiPRS"].fit()

    runtimes["MultiPRS"] = timer.minutes

    base_models["Covariates"] = MultiPRS(
        prs_dataset=dataset,
        covariates_cols=dataset.covariates_cols,
        add_intercept=add_intercept,
        class_weights=class_weights,
        penalty_type=penalty_type,
        penalty=penalty,
    )
    with Timer() as timer:
        base_models["Covariates"].fit()

    runtimes["Covariates"] = timer.minutes

    for i, pgs_id in enumerate(dataset.prs_cols):
        base_models[f"{pgs_id}-covariates"] = MultiPRS(
            prs_dataset=dataset,
            expert_cols=pgs_id,
            covariates_cols=dataset.covariates_cols,
            add_intercept=add_intercept,
            class_weights=class_weights,
            penalty_type=penalty_type,
            penalty=penalty,
        )

        with Timer() as timer:
            base_models[f"{pgs_id}-covariates"].fit()

        runtimes[f"{pgs_id}-covariates"] = timer.minutes

    return base_models, runtimes


def train_moe_model_numpy(
    dataset,
    gate_penalty=0.0,
    expert_penalty=0.0,
    gate_add_intercept=True,
    expert_add_intercept=True,
    optimizer="L-BFGS-B",
):
    print(
        f"> Training MoE model for {dataset.phenotype_col} with {dataset.N} samples..."
    )

    dataset.set_backend("numpy")

    moe_models = dict()
    runtimes = dict()

    moe = MoEPRS(prs_dataset=dataset,
                 expert_cols=dataset.prs_cols,
                 gate_input_cols=dataset.covariates_cols,
                 global_covariates_cols=dataset.covariates_cols,
                 optimizer=optimizer,
                 fix_residuals=False,
                 gate_add_intercept=gate_add_intercept,
                 expert_add_intercept=expert_add_intercept,
                 gate_penalty=gate_penalty,
                 expert_penalty=expert_penalty,
                 n_jobs=min(4, dataset.n_prs_models))

    with Timer() as timer:
        moe.fit()
    
    runtimes['MoE'] = timer.minutes

    moe_global_int = MoEPRS(prs_dataset=dataset,
                 expert_cols=dataset.prs_cols,
                 gate_input_cols=dataset.covariates_cols,
                 global_covariates_cols=dataset.covariates_cols,
                 optimizer=optimizer,
                 fix_residuals=False,
                 gate_add_intercept=gate_add_intercept,
                 expert_add_intercept=False,
                 gate_penalty=gate_penalty,
                 expert_penalty=expert_penalty,
                 n_jobs=min(4, dataset.n_prs_models))

    with Timer() as timer:
        moe_global_int.fit()
    
    runtimes['MoE-global-int'] = timer.minutes

    moe_cfg = MoEPRS(prs_dataset=dataset,
                expert_cols=dataset.prs_cols,
                gate_input_cols=None,
                global_covariates_cols=dataset.covariates_cols,
                optimizer=optimizer,
                fix_residuals=False,
                gate_add_intercept=gate_add_intercept,
                expert_add_intercept=False,  ## Check this?
                gate_penalty=gate_penalty,
                expert_penalty=expert_penalty,
                n_jobs=min(4, dataset.n_prs_models))
    
    with Timer() as timer:
        moe_cfg.fit()
    
    runtimes['MoE-CFG'] = timer.minutes
    
    # Try the same but with two-step fitting:
    moe_global_int_two_step = copy.deepcopy(moe_global_int)
    
    with Timer() as timer:
        moe_global_int_two_step.two_step_fit()
    
    runtimes['MoE-global-int-two-step'] = timer.minutes
            
    moe_models = {
        'MoE-CFG': moe_cfg,
        'MoE': moe,
        'MoE-global-int': moe_global_int,
        'MoE-global-int-two-step': moe_global_int_two_step,
    }


    if dataset.phenotype_likelihood != 'binomial':
        moe_fix_resid = MoEPRS(prs_dataset=dataset,
                               expert_cols=dataset.prs_cols,
                               gate_input_cols=dataset.covariates_cols,
                               global_covariates_cols=dataset.covariates_cols,
                               optimizer=optimizer,
                               fix_residuals=True,
                               gate_add_intercept=gate_add_intercept,
                               expert_add_intercept=expert_add_intercept,
                               gate_penalty=gate_penalty,
                               expert_penalty=expert_penalty,
                               n_jobs=min(4, dataset.n_prs_models))
        
        with Timer() as timer:
            moe_fix_resid.fit()
        
        runtimes['MoE-fixed-resid'] = timer.minutes

        moe_fix_resid_global_int = MoEPRS(prs_dataset=dataset,
                               expert_cols=dataset.prs_cols,
                               gate_input_cols=dataset.covariates_cols,
                               global_covariates_cols=dataset.covariates_cols,
                               optimizer=optimizer,
                               fix_residuals=True,
                               gate_add_intercept=gate_add_intercept,
                               expert_add_intercept=False,
                               gate_penalty=gate_penalty,
                               expert_penalty=expert_penalty,
                               n_jobs=min(4, dataset.n_prs_models))
        
        with Timer() as timer:
            moe_fix_resid_global_int.fit()
        
        runtimes['MoE-fixed-resid-global-int'] = timer.minutes

        moe_models['MoE-fixed-resid'] = moe_fix_resid
        moe_models['MoE-fixed-resid-global-int'] = moe_fix_resid_global_int

        # Try two-step fitting for the fixed residuals + global intercept model:
        moe_fix_resid_global_int_two_step = copy.deepcopy(moe_fix_resid_global_int)
        
        with Timer() as timer:
            moe_fix_resid_global_int_two_step.two_step_fit()
        
        runtimes['MoE-fixed-resid-global-int-two-step'] = timer.minutes

        moe_models['MoE-fixed-resid-global-int-two-step'] = moe_fix_resid_global_int_two_step

        """
        moe_huber = MoEPRS(prs_dataset=dataset,
                           expert_cols=dataset.prs_cols,
                           gate_input_cols=dataset.covariates_cols,
                           global_covariates_cols=dataset.covariates_cols,
                           optimizer=optimizer,
                           loss='huber',
                           gate_add_intercept=gate_add_intercept,
                           expert_add_intercept=expert_add_intercept,
                           gate_penalty=gate_penalty,
                           expert_penalty=expert_penalty,
                           n_jobs=min(4, dataset.n_prs_models))
        moe_huber.fit()

        res['MoE-huber'] = moe_huber
        """

    return moe_models, runtimes


def train_moe_models_torch(dataset,
                            gate_model_layers=None,
                            add_covariates_to_experts=False,
                            loss='likelihood_mixture2',
                            optimizer='Adam',
                            gate_add_batch_norm=True,
                            gate_add_layer_norm=False,
                            weight_decay=0.,
                            learning_rate=1e-3,
                            max_epochs=1000,
                            batch_size=2048,
                            weigh_samples=False,
                            seed = 8,
                            topk_k=None,                               
                            tau_start=1.5,                         
                            tau_end=1.0,                           
                            hard_ste=False,                        
                            lb_coef=0.00,                         
                            ent_coef=0.05,
                            ent_coef_end=0.0, 
                            ent_warm_epochs=10,                   
                            ent_decay_epochs=90,                  
                            ancestry_balance_lambda=None,                         
                            use_per_expert_bias=False,
                            use_global_head=True,
                            global_head_bias=True,
                            fix_sigma2=False
                        ):

    with Timer() as timer:
        moe_models["MoE-global-int-two-step"] = moe_global_int_two_step.two_step_fit()

    runtimes["MoE-global-int-two-step"] = timer.minutes

    moe_cfg = MoEPRS(
        prs_dataset=dataset,
        expert_cols=dataset.prs_cols,
        gate_input_cols=None,
        global_covariates_cols=dataset.covariates_cols,
        optimizer=optimizer,
        fix_residuals=False,
        gate_add_intercept=gate_add_intercept,
        expert_add_intercept=False,  ## Check this?
        gate_penalty=gate_penalty,
        expert_penalty=expert_penalty,
        n_jobs=min(4, dataset.n_prs_models),
    )

    with Timer() as timer:
        moe_models["MoE-CFG"] = moe_cfg.fit()

    runtimes["MoE-CFG"] = timer.minutes

    if dataset.phenotype_likelihood != "binomial":
        moe_fix_resid = MoEPRS(
            prs_dataset=dataset,
            expert_cols=dataset.prs_cols,
            gate_input_cols=dataset.covariates_cols,
            global_covariates_cols=dataset.covariates_cols,
            optimizer=optimizer,
            fix_residuals=True,
            gate_add_intercept=gate_add_intercept,
            expert_add_intercept=expert_add_intercept,
            gate_penalty=gate_penalty,
            expert_penalty=expert_penalty,
            n_jobs=min(4, dataset.n_prs_models),
        )

        with Timer() as timer:
            moe_models["MoE-fixed-resid"] = moe_fix_resid.fit()

        runtimes["MoE-fixed-resid"] = timer.minutes

        moe_fix_resid_global_int = MoEPRS(
            prs_dataset=dataset,
            expert_cols=dataset.prs_cols,
            gate_input_cols=dataset.covariates_cols,
            global_covariates_cols=dataset.covariates_cols,
            optimizer=optimizer,
            fix_residuals=True,
            gate_add_intercept=gate_add_intercept,
            expert_add_intercept=False,
            gate_penalty=gate_penalty,
            expert_penalty=expert_penalty,
            n_jobs=min(4, dataset.n_prs_models),
        )
        with Timer() as timer:
            moe_models["MoE-fixed-resid-global-int"] = moe_fix_resid_global_int.fit()

        runtimes["MoE-fixed-resid-global-int"] = timer.minutes

    return moe_models, runtimes


def train_moe_models_torch(
    dataset,
    gate_model_layers=None,
    add_covariates_to_experts=False,
    loss="likelihood_mixture",
    optimizer="Adam",
    penalty=0.0,
    learning_rate=1e-2,
    max_epochs=100,
    batch_size=None,
    weigh_samples=False,
):
    dataset.set_backend("torch")

    # Define which columns to fetch 
    group_getitem_cols = {
        'phenotype': [dataset.phenotype_col],
        'gate_input': dataset.covariates_cols,
        'experts': dataset.prs_cols,
        "global_input": dataset.covariates_cols
    }

    # whether or not to have expert-specific covariates slopes
    if add_covariates_to_experts:
        group_getitem_cols["expert_covariates"] = dataset.covariates_cols

    dataset.set_group_getitem_cols(group_getitem_cols)

    # If gaussian, and sigma2 is estimated, use the corresponding sigma2 included loss
    loss = "likelihood_mixture_sigma" \
        if (dataset.phenotype_likelihood == "gaussian" and not fix_sigma2) else loss
    
    # Initialize the SGD MoE model through pytorch Lightning:
    m = Lit_MoEPRS(
        dataset.group_getitem_cols,
        gate_model_layers=gate_model_layers,
        loss=loss,
        family=dataset.phenotype_likelihood,
        optimizer=optimizer,
        gate_add_batch_norm=gate_add_batch_norm,
        gate_add_layer_norm=gate_add_layer_norm,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        topk_k=topk_k,                               
        tau_start=tau_start,                         
        tau_end=tau_end,                           
        hard_ste=hard_ste,                        
        lb_coef=lb_coef,                          
        ent_coef=ent_coef,
        ent_coef_end=ent_coef_end, 
        ent_warm_epochs=ent_warm_epochs,                   
        ent_decay_epochs=ent_decay_epochs,                         
        use_per_expert_bias=use_per_expert_bias,
        use_global_head=use_global_head,
        global_head_bias=global_head_bias,
    )

    # Train with PyTorch Lightning:
    with Timer() as t:
        _, m = train_model(m,
                        dataset,
                        max_epochs=max_epochs,
                        batch_size=batch_size,
                        weigh_samples=weigh_samples,
                        seed = seed, split_seed=seed,
                        ancestry_balance_lambda=ancestry_balance_lambda)

    m.runtime_minutes = t.minutes  # <-- attach to model for saving later 

    return m


def train_all_models(
    dataset,
    baseline_kwargs,
    moe_kwargs,
    skip_baseline=False,
    skip_moe=False,
    moe_pytorch_kwargs=None,
    pytorch_only=False,
    skip_moe_pytorch=False,
    seed=8,
):
    trained_models = {}
    runtimes = {}

    if not pytorch_only:
        if not skip_baseline:
            bm, br = train_baseline_linear_models(dataset, **baseline_kwargs)
            trained_models.update(bm)
            runtimes.update(br)

        if not skip_moe:
            mm, mr = train_moe_model_numpy(dataset, **moe_kwargs)
            trained_models.update(mm)
            runtimes.update(mr)

    if not skip_moe_pytorch:
        # Use caller-provided kwargs if available; otherwise use your defaults
        if moe_pytorch_kwargs is None:
            moe_pytorch_kwargs = dict(
                loss="likelihood_mixture2",
                fix_sigma2=False,  # Train MoE-SGD with estimated sigma2 (when gaussian)
                optimizer="Adam",
                gate_model_layers=None,
                gate_add_batch_norm=False,
                gate_add_layer_norm=True,
                learning_rate=1e-3,
                weight_decay=0,
                max_epochs=500,
                batch_size=2048,
                seed=seed,
                topk_k=None,           # Top-k routing
                tau_start=1.5,         # Temperature schedule starting value
                tau_end=1.0,           # Ending temperature value
                hard_ste=False,        # irrelevant when no top-k
                lb_coef=0.00,          # Load balancing coefficient
                ent_coef=0.05,         # Entropy regularization coefficient
                ent_coef_end=0.0,
                ent_warm_epochs=10,    # Warm-up epochs for entropy reg
                ent_decay_epochs=90,   # Decay epochs for entropy reg
                ancestry_balance_lambda=None,
                use_per_expert_bias=False,
                add_covariates_to_experts=False,
                use_global_head=True,
                global_head_bias=True,
            )

        m_pt = train_moe_models_torch(dataset, **moe_pytorch_kwargs)

        trained_models["MoE-PyTorch"] = m_pt
        if hasattr(m_pt, "runtime_minutes"):
            runtimes["MoE-PyTorch"] = float(m_pt.runtime_minutes)

    return trained_models, runtimes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline and MoE models.")
    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        type=str,
        required=True,
        help="The path to the dataset file.",
    )
    parser.add_argument(
        "--baseline-kwargs",
        dest="baseline_kwargs",
        type=str,
        default="",
        help="A comma-separated list of key-value pairs with the arguments for the baseline models.",
    )
    parser.add_argument(
        "--moe-kwargs",
        dest="moe_kwargs",
        type=str,
        default="",
        help="A comma-separated list of key-value pairs with the arguments for the MoE model.",
    )
    parser.add_argument(
        "--residualize-phenotype",
        dest="residualize_phenotype",
        action="store_true",
        default=False,
        help="Whether to residualize the phenotype before training the models.",
    )
    parser.add_argument(
        "--residualize-prs",
        dest="residualize_prs",
        action="store_true",
        default=False,
        help="Whether to residualize the PRS before training the models.",
    )
    parser.add_argument(
        "--skip-baseline",
        dest="skip_baseline",
        action="store_true",
        default=False,
        help="Whether to skip training the baseline models.",
    )
    parser.add_argument(
        "--skip-moe",
        dest="skip_moe",
        action="store_true",
        default=False,
        help="Whether to skip training the MoE models.",
    )
    parser.add_argument(
        "--skip-moe-pytorch",
        dest="skip_moe_pytorch",
        action="store_true",
        default=False,
        help="Whether to skip training the MoE models with PyTorch.",
    )
    parser.add_argument(
        "--pytorch-only",
        dest="pytorch_only",
        action="store_true",
        default=False,
        help="Whether to only train the MoE models with PyTorch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=8,
        help="Random seed for reproducibility.",
    )
    
    args = parser.parse_args()

    make_deterministic(args.seed)

    prs_dataset = PRSDataset.from_pickle(args.dataset_path)

    if args.residualize_phenotype:
        try:
            prs_dataset.adjust_phenotype_for_covariates()
        except AssertionError:
            print("Could not residualize the phenotype.")

    if args.residualize_prs:
        prs_dataset.adjust_prs_for_covariates()

    baseline_kwargs = {}
    if len(args.baseline_kwargs) > 0:
        baseline_kwargs = {
            k: v
            for k, v in [kw.split("=") for kw in args.baseline_kwargs.split(",")]
            if v
        }
    moe_kwargs = {}
    if len(args.moe_kwargs) > 0:
        moe_kwargs = {
            k: v for k, v in [kw.split("=") for kw in args.moe_kwargs.split(",")] if v
        }

    trained_models, model_runtimes = train_all_models(
        prs_dataset,
        baseline_kwargs,
        moe_kwargs,
        skip_baseline=args.skip_baseline,
        skip_moe=args.skip_moe,
        pytorch_only=args.pytorch_only,
        skip_moe_pytorch=args.skip_moe_pytorch,
        seed=args.seed,
    )

    output_dir = osp.dirname(args.dataset_path).replace(
        "harmonized_data", "trained_models"
    )
    dataset_name = osp.basename(args.dataset_path).replace(".pkl", "")

    if args.residualize_phenotype:
        dataset_name += "_rph"
    if args.residualize_prs:
        dataset_name += "_rprs"

    output_dir = osp.join(output_dir, dataset_name)

    makedir(output_dir)

    import torch, pickle, json

    print("> Saving trained models to:\n\t", output_dir)

    for model_name, model in trained_models.items():
        runtime_min = model_runtimes.get(model_name, None)

        out = osp.join(output_dir, f'{model_name}.pkl')
        try:
            model.save(out)
            print(f"Saved NumPy model: {model_name}")

            if runtime_min is not None:
                payload = {"Runtime_min": runtime_min}
                with open(osp.join(output_dir, f"{model_name}_runtime.json"), "w") as f:
                    json.dump(payload, f)
        except Exception:
            #pytorch models
            pt_path = osp.join(output_dir, f"{model_name}.pt")

            # ---- Extract and store torch configuration ----
            config = {
                "loss": model.loss,
                "optimizer": model.optimizer,
                "learning_rate": model.lr,
                "weight_decay": model.weight_decay,
                "gate_model_layers": getattr(model, "gate_model_layers", None),
                "gate_add_batch_norm": model.gate_add_batch_norm,
                "family": model.family,

                "topk_k": getattr(model, "topk_k", None),
                "tau_start": getattr(model, "tau_start", 1.0),
                "tau_end": getattr(model, "tau_end", 1.0),
                "hard_ste": getattr(model, "hard_ste", True),
                "lb_coef": getattr(model, "lb_coef", 0.0),
                "eps": getattr(model, "eps", 1e-12),
                "ent_coef_start": getattr(model, "ent_coef_start", getattr(model, "ent_coef", 0.0)),
                "ent_coef_end": getattr(model, "ent_coef_end", getattr(model, "ent_coef", 0.0)),
                "ent_warm_epochs": getattr(model, "ent_warm_epochs", 0),
                "ent_decay_epochs": getattr(model, "ent_decay_epochs", 0),

                "use_per_expert_bias": getattr(model, "use_per_expert_bias", False),
                "use_global_head": getattr(model, "use_global_head", False),
                "min_sigma2": float(getattr(model, "min_sigma2", 0.0)) if hasattr(model, "min_sigma2") else None,
                "expert_bias_scale_floor": float(getattr(model, "expert_bias_scale_floor", 0.0)),
                "has_expert_covariates": ("expert_covariates" in getattr(model, "group_getitem_cols", {})),
                "global_head_bias": (
                    getattr(model, "global_head", None) is not None and getattr(model.global_head, "bias", None) is not None
                ),
            }

            # ---- Save state_dict + config inside one checkpoint ----
            torch.save({
                "state_dict": model.state_dict(),
                "config": config
            }, pt_path)
            print(f"Saved PyTorch model checkpoint: {model_name}")

            if runtime_min is not None:
                payload = {"Runtime_min": runtime_min}
                with open(osp.join(output_dir, f"{model_name}_runtime.json"), "w") as f:
                    json.dump(payload, f)

            # ---- Save scaler used at training time on the training dataset (same scaler will be used later for evaluation)----
            scaler_path = osp.join(output_dir, "MoE-PyTorch.scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(copy.deepcopy(prs_dataset.scaler), f)
            print(f"Saved scaler: {scaler_path}")

            # ---- Optional: save JSON for inspection ----
            json_path = osp.join(output_dir, "inspect_MoE-PyTorch_config.json")
            with open(json_path, "w") as jf:
                json.dump(config, jf, indent=2)
            print(f"Saved config JSON: {json_path}")

            # ---- Save gate covariate weights for inspection ----
            if hasattr(model, "gate_model"):
                try:
                    import pandas as pd
                    import torch.nn as nn

                    cov_names = prs_dataset.covariates_cols
                    expert_names = prs_dataset.prs_cols
                    C = len(cov_names)
                    K = len(expert_names)

                    # Find the last Linear (logits) layer
                    linear = None
                    for layer in reversed(list(model.gate_model.gate)):
                        if isinstance(layer, nn.Linear):
                            linear = layer
                            break
                    if linear is None:
                        raise RuntimeError("Could not find an nn.Linear layer in gate_model.gate")

                    # Only export if this layer maps covariates -> experts
                    if linear.in_features != C or linear.out_features != K:
                        print(
                            f"Skipping gate weight export for {model_name}: "
                            f"logits layer is {linear.out_features}x{linear.in_features}, "
                            f"but expected {K}x{C} (nonlinear/hidden gate)."
                        )
                    else:
                        W = linear.weight.detach().cpu().numpy()    # (K, C)
                        b = linear.bias.detach().cpu().numpy()      # (K,)

                        cov_names = prs_dataset.covariates_cols     # length C
                        expert_names = prs_dataset.prs_cols         # length K

                        rows = []
                        for k, e_name in enumerate(expert_names):
                            for j, c_name in enumerate(cov_names):
                                rows.append({
                                    "expert_idx": k,
                                    "expert": e_name,
                                    "covariate_idx": j,
                                    "covariate": c_name,
                                    "weight": float(W[k, j]),
                                })
                            # store bias as a separate "covariate"
                            rows.append({
                                "expert_idx": k,
                                "expert": e_name,
                                "covariate_idx": -1,
                                "covariate": "bias",
                                "weight": float(b[k]),
                            })

                        df = pd.DataFrame(rows)
                        weights_path = osp.join(output_dir, f"{model_name}_gate_weights.csv")
                        df.to_csv(weights_path, index=False)
                        print(f"Saved gate weights: {weights_path}")
                except Exception as e:
                    print(f"Could not save gate weights for {model_name}: {e}")

