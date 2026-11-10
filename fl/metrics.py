"""Shared metric code and the fixed per-round row schema used by every mode (local, fedavg, fedavg_he)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from torch import nn

from models.identity import best_f1_threshold

ROW_KEYS = (
    "mode", "fraud_type", "n_clients", "alpha", "seed", "round",
    # test metrics at threshold 0.5 (local mode: mean across banks)
    "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc", "test_pr_auc", "test_loss",
    # one threshold tuned on pooled validation predictions (never on test), applied to test
    "tuned_threshold", "test_f1_tuned", "test_precision_tuned", "test_recall_tuned",
    # per-bank spread of test metrics; only local mode has per-bank models, null otherwise
    "bank_f1_min", "bank_f1_max", "bank_pr_auc_min", "bank_pr_auc_max",
    "val_loss", "val_f1", "train_loss", "train_s",
    # costs; crypto columns are 0 for fedavg and null for local
    "encrypt_s", "decrypt_s", "aggregate_s", "uplink_bytes_per_client", "downlink_bytes", "round_wall_s",
    # fedavg_he only: max |decrypted global - plaintext weighted average of the same client updates|
    "max_weight_err",
)


def make_row(cfg, rnd: int, **values) -> dict:
    """Row with every ROW_KEY in order; keys not given are None. Unknown keys are an error."""
    unknown = set(values) - set(ROW_KEYS)
    if unknown:
        raise KeyError(f"unknown row keys {sorted(unknown)}")
    base = dict(mode=cfg.mode, fraud_type=cfg.fraud_type, n_clients=cfg.n_clients, alpha=cfg.alpha,
                seed=cfg.seed, round=rnd)
    return {k: base.get(k, values.get(k)) for k in ROW_KEYS}


@torch.no_grad()
def predict_logits(model, X, batch_size: int = 8192) -> torch.Tensor:
    model.eval()
    X = torch.from_numpy(np.array(X, dtype=np.float32, copy=True))
    out = [model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)]
    return torch.cat(out) if out else torch.empty(0)


def binary_metrics(y, p, threshold: float = 0.5, logits: torch.Tensor | None = None) -> dict:
    """Accuracy, precision, recall, F1, ROC-AUC, PR-AUC at ``threshold`` (+ BCE loss when logits are given)."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    two = len(np.unique(y)) == 2
    m = {"accuracy": float(accuracy_score(y, pred)),
         "precision": float(precision_score(y, pred, zero_division=0)),
         "recall": float(recall_score(y, pred, zero_division=0)),
         "f1": float(f1_score(y, pred, zero_division=0)),
         "roc_auc": float(roc_auc_score(y, p)) if two else float("nan"),
         "pr_auc": float(average_precision_score(y, p)) if two else float("nan")}
    if logits is not None:
        m["loss"] = float(nn.functional.binary_cross_entropy_with_logits(
            logits, torch.as_tensor(y, dtype=torch.float32)))
    return m


def tuned_threshold(y_val, p_val) -> float:
    """F1-maximising threshold chosen on validation predictions only (models/identity.best_f1_threshold)."""
    return best_f1_threshold(np.asarray(y_val).astype(int), np.asarray(p_val))


def tuned_test_metrics(threshold: float, y_test, p_test) -> dict:
    m = binary_metrics(y_test, p_test, threshold)
    return {"tuned_threshold": float(threshold), "test_f1_tuned": m["f1"],
            "test_precision_tuned": m["precision"], "test_recall_tuned": m["recall"]}


def score_model(model, X_val, y_val, X_test, y_test) -> dict:
    """All test_* and tuned columns for one model. The threshold uses validation data only."""
    lv, lt = predict_logits(model, X_val), predict_logits(model, X_test)
    m = binary_metrics(y_test, torch.sigmoid(lt).numpy(), 0.5, logits=lt)
    out = {f"test_{k}": v for k, v in m.items()}
    out.update(tuned_test_metrics(tuned_threshold(y_val, torch.sigmoid(lv).numpy()), y_test,
                                  torch.sigmoid(lt).numpy()))
    return out
