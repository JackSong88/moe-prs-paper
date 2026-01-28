# model/torch_inference.py

import os.path as osp
import copy
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from moe import MoEPRS
from moe_pytorch import Lit_MoEPRS


def apply_saved_scaler(prs_dataset, model_dir, scaler_fname="MoE-PyTorch.scaler.pkl"):
    """
    Returns a deep-copied dataset with the training scaler applied (no refit).
    If no scaler exists, returns a deep-copied dataset unchanged.
    """
    d = copy.deepcopy(prs_dataset)

    scaler_path = osp.join(model_dir, scaler_fname)
    if osp.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        d.standardize_data(scaler=scaler, refit=False)

    return d


def build_group_getitem_cols(prs_dataset, cfg):
    """
    Reconstruct group_getitem_cols exactly like training did (based on cfg knobs).
    """
    group_getitem_cols = {
        "phenotype": [prs_dataset.phenotype_col],
        "gate_input": prs_dataset.covariates_cols,
        "experts": prs_dataset.prs_cols,
        "global_input": prs_dataset.covariates_cols,
    }
    if cfg.get("has_expert_covariates", False):
        group_getitem_cols["expert_covariates"] = prs_dataset.covariates_cols
    return group_getitem_cols


def load_lit_from_pt(prs_dataset, pt_path, map_location="cpu", strict=True):
    """
    Load a Lit_MoEPRS from your custom .pt format:
      {"state_dict": ..., "config": ...}
    """
    checkpoint = torch.load(pt_path, map_location=map_location)
    if "config" not in checkpoint:
        raise ValueError(f"Missing 'config' in checkpoint: {pt_path}")
    if "state_dict" not in checkpoint:
        raise ValueError(f"Missing 'state_dict' in checkpoint: {pt_path}")

    cfg = checkpoint["config"]
    group_getitem_cols = build_group_getitem_cols(prs_dataset, cfg)

    # load in model with all trained settings
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

        use_per_expert_bias=cfg.get("use_per_expert_bias", False),
        use_global_head=cfg.get("use_global_head", True),
        global_head_bias=cfg.get("global_head_bias", False),

        ent_coef=cfg.get("ent_coef_start", 0.0),
        ent_coef_end=cfg.get("ent_coef_end", None),
        ent_warm_epochs=cfg.get("ent_warm_epochs", 0),
        ent_decay_epochs=cfg.get("ent_decay_epochs", 0),
    )

    # Non-state attrs that affect forward
    if cfg.get("min_sigma2", None) is not None:
        lit.min_sigma2 = float(cfg["min_sigma2"])
    if "expert_bias_scale_floor" in cfg:
        lit.expert_bias_scale_floor = float(cfg["expert_bias_scale_floor"])

    lit.load_state_dict(checkpoint["state_dict"], strict=strict)
    lit.eval()
    lit.to(map_location)
    return lit


class IndexSubset(Dataset):
    """
    Returns ORIGINAL row index so prs_dataset.get_batch can vector-fetch.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return int(self.indices[i])


class TorchMoEModel:
    """
    Unified inference wrapper for Lit_MoEPRS that supports:
      - predict(prs_dataset) -> (N,) numpy
      - predict_proba(prs_dataset) -> (N,K) numpy (gate probs)

    Also exposes .expert_cols for plotting parity with MoEPRS.
    """
    def __init__(self, lit_model, model_dir, expert_cols, pred_batch_size=2048, gate_batch_size=65536):
        self.lit_model = lit_model
        self.model_dir = model_dir
        self.expert_cols = list(expert_cols)
        self.pred_batch_size = int(pred_batch_size)
        self.gate_batch_size = int(gate_batch_size)

    def _prepare_dataset(self, prs_dataset):
        # apply scaler (deep copy)
        d = apply_saved_scaler(prs_dataset, self.model_dir)

        # make sure group_getitem_cols matches what the Lightning model expects
        expected = self.lit_model.group_getitem_cols
        d.set_group_getitem_cols(expected)

        # torch backend + cache matrix for fast batched fetching
        d.set_backend("torch")
        d._data_matrix = None
        d.cache_data_matrix()
        return d

    def _batch_loader(self, d, batch_size):
        idx = np.arange(d.N)
        subset = IndexSubset(d, idx)
        return DataLoader(
            subset,
            batch_size=min(int(batch_size), d.N),
            shuffle=False,
            num_workers=0,
            collate_fn=d.get_batch,  # vectorized fetch
        )

    def predict(self, prs_dataset):
        d = self._prepare_dataset(prs_dataset)

        outs = []
        with torch.no_grad():
            for batch in self._batch_loader(d, self.pred_batch_size):
                yhat = self.lit_model.forward(batch)  # (B,)
                outs.append(yhat.detach().cpu().numpy())

        return np.concatenate(outs, axis=0)

    def predict_proba(self, prs_dataset):
        d = self._prepare_dataset(prs_dataset)

        outs = []
        with torch.no_grad():
            for batch in self._batch_loader(d, self.gate_batch_size):
                p = self.lit_model.gate_forward(batch)  # (B,K)
                outs.append(p.detach().cpu().numpy())

        return np.vstack(outs)


def load_model_any(prs_dataset, model_path, pred_batch_size=2048, gate_batch_size=65536):
    """
    Unified loader:
      - .pt -> TorchMoEModel (predict + predict_proba)
      - else -> MoEPRS.from_saved_model (existing numpy model)
    """
    if model_path.endswith(".pt"):
        lit = load_lit_from_pt(prs_dataset, model_path, map_location="cpu", strict=True)
        model_dir = osp.dirname(model_path)
        return TorchMoEModel(
            lit_model=lit,
            model_dir=model_dir,
            expert_cols=prs_dataset.prs_cols,
            pred_batch_size=pred_batch_size,
            gate_batch_size=gate_batch_size,
        )

    return MoEPRS.from_saved_model(model_path)
