"""Identity-fraud model (owner: Likhitha).

Feedforward network on the ``identity_fraud`` feature view (``idn``: ``id_01-id_38``, ``DeviceType``,
``DeviceInfo``, e-mail domains, ``has_identity``, ``email_mismatch``; 109 inputs on the full IEEE-CIS data).

Contract from ``models/README.md``::

    from models.identity import build_model
    model = build_model(input_dim("identity_fraud"))          # torch nn.Module, ~24k parameters

    from common-style weight exchange (lists of float32 numpy arrays, same order every call):
    w = model.get_weights();  model.set_weights(w)            # or get_weights(model) / set_weights(model, w)

Design notes
------------
* 3 hidden layers (128-64-32), ReLU, dropout. One output **logit**: pair it with ``BCEWithLogitsLoss(pos_weight)``
  (see :func:`make_loss`); apply a sigmoid only for probabilities (:func:`predict_proba`).
* No BatchNorm on purpose. Its running mean/var are *buffers*, not parameters, so FedAvg and CKKS encryption
  would have to treat them separately. The data is already standardised by ``data/features.py``. Because of this,
  ``get_weights`` (all parameters) captures the complete state of the model.
* Fraud is ~3.5% of rows, so accuracy is meaningless. :func:`evaluate` reports PR-AUC, ROC-AUC, F1, precision,
  recall and recall at a fixed false-positive rate, as recommended in ``data/DATA.md``.

Baseline (local-only on one bank, or pooled/centralised) from the repository root::

    python -m models.identity --bank 0
    python -m models.identity --bank pooled
"""
from __future__ import annotations

import argparse
import copy
import csv
import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from torch import nn

log = logging.getLogger(__name__)

FRAUD_TYPE = "identity_fraud"
DEFAULT_HIDDEN = (128, 64, 32)


# ----------------------------------------------------------------------------- 1. model
class IdentityFraudNet(nn.Module):
    """MLP that maps ``input_dim`` identity/device features to one fraud logit."""

    def __init__(self, input_dim: int, hidden: Sequence[int] = DEFAULT_HIDDEN, dropout: float = 0.2):
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if not 2 <= len(hidden) <= 3:
            raise ValueError("use 2 to 3 hidden layers (models/README.md)")
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits with shape (N,)."""
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, X, batch_size: int = 8192) -> np.ndarray:
        """Fraud probabilities (N,) for a numpy array or tensor."""
        return predict_proba(self, X, batch_size)

    # weight exchange used by the federated-learning and encryption code
    def get_weights(self) -> list[np.ndarray]:
        return get_weights(self)

    def set_weights(self, weights: Sequence[np.ndarray]) -> None:
        set_weights(self, weights)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(input_dim: int, hidden: Sequence[int] = DEFAULT_HIDDEN, dropout: float = 0.2,
                seed: int | None = None) -> IdentityFraudNet:
    """Interface function required by the project: ``build_model(input_dim) -> nn.Module``.

    ``seed`` makes the initial weights reproducible. In federated runs every client receives the server's weights
    through ``set_weights`` anyway, so the seed only matters for the server's initial global model.
    """
    if seed is not None:
        torch.manual_seed(seed)
    return IdentityFraudNet(input_dim, hidden, dropout)


def make_loss(pos_weight: float = 1.0, device=None) -> nn.BCEWithLogitsLoss:
    """Binary cross-entropy on logits; ``pos_weight`` = negatives / positives (``data.loader.get_pos_weight``)."""
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], dtype=torch.float32, device=device))


# ----------------------------------------------------------------------------- 2. weights in and out
def get_weights(model: nn.Module) -> list[np.ndarray]:
    """All parameters as a list of float32 numpy arrays (copies), in a fixed order."""
    return [p.detach().cpu().numpy().astype(np.float32, copy=True) for p in model.parameters()]


def set_weights(model: nn.Module, weights: Sequence[np.ndarray]) -> None:
    """Load a list produced by :func:`get_weights` (e.g. the aggregated global model) into ``model``."""
    params = list(model.parameters())
    if len(weights) != len(params):
        raise ValueError(f"expected {len(params)} arrays, got {len(weights)}")
    with torch.no_grad():
        for p, w in zip(params, weights):
            w = np.asarray(w, dtype=np.float32)
            if tuple(w.shape) != tuple(p.shape):
                raise ValueError(f"shape mismatch: expected {tuple(p.shape)}, got {tuple(w.shape)}")
            p.copy_(torch.from_numpy(w.copy()).to(p.device))


# ----------------------------------------------------------------------------- 3. training
def _device(device=None) -> torch.device:
    return torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epochs(model: nn.Module, X: np.ndarray, y: np.ndarray, pos_weight: float = 1.0, epochs: int = 1,
                 lr: float = 1e-3, batch_size: int = 512, weight_decay: float = 1e-5, device=None,
                 seed: int | None = None) -> list[float]:
    """Plain local training; returns the mean loss of each epoch.

    This is also what a federated client calls once per round (``epochs`` = local epochs): it builds a fresh Adam
    optimiser each call, which is the usual choice for FedAvg because optimiser state is not shared.
    """
    dev = _device(device)
    model.to(dev).train()
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    n = len(Xt)
    if n == 0:
        return []
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    loss_fn = make_loss(pos_weight, dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        losses.append(total / n)
    return losses


def train_local(model: nn.Module, X_train, y_train, X_val, y_val, pos_weight: float = 1.0, epochs: int = 30,
                patience: int = 5, lr: float = 1e-3, batch_size: int = 512, weight_decay: float = 1e-5,
                device=None, seed: int | None = 0) -> list[dict]:
    """Local-only training with early stopping on validation PR-AUC. Restores the best weights; returns the history."""
    best_ap, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        loss = train_epochs(model, X_train, y_train, pos_weight, 1, lr, batch_size, weight_decay, device,
                            None if seed is None else seed + ep)[0]
        m = evaluate(model, X_val, y_val, device=device)
        history.append({"epoch": ep, "train_loss": loss, "val_pr_auc": m["pr_auc"], "val_roc_auc": m["roc_auc"]})
        log.info("epoch %2d  loss %.4f  val PR-AUC %.4f  val ROC-AUC %.4f", ep, loss, m["pr_auc"], m["roc_auc"])
        ap = m["pr_auc"] if np.isfinite(m["pr_auc"]) else -1.0
        if ap > best_ap:
            best_ap, best_state, bad = ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                log.info("early stopping at epoch %d (best val PR-AUC %.4f)", ep, best_ap)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ----------------------------------------------------------------------------- 4. evaluation
@torch.no_grad()
def predict_proba(model: nn.Module, X, batch_size: int = 8192, device=None) -> np.ndarray:
    dev = _device(device)
    model.to(dev).eval()
    Xt = torch.as_tensor(np.ascontiguousarray(X, dtype=np.float32))
    out = [torch.sigmoid(model(Xt[i:i + batch_size].to(dev))).cpu() for i in range(0, len(Xt), batch_size)]
    return torch.cat(out).numpy() if out else np.empty(0, dtype=np.float32)


def best_f1_threshold(y_true, proba) -> float:
    """Probability threshold that maximises F1 (pick it on validation data, then reuse it on test)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    order = np.argsort(-proba)
    ys, ps = y_true[order], np.asarray(proba)[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    f1 = 2 * tp / (tp + fp + (ys.sum()))  # = 2TP / (2TP + FP + FN) since FN = P - TP
    # only cut where the score actually changes
    last = np.r_[ps[1:] != ps[:-1], True]
    f1 = np.where(last, f1, -1)
    k = int(np.argmax(f1))
    return float(ps[k])


def recall_at_fpr(y_true, proba, fpr_target: float = 0.01) -> float:
    """Highest recall (TPR) achievable with false-positive rate <= ``fpr_target``."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(tpr[fpr <= fpr_target].max())


def evaluate(model: nn.Module, X, y, threshold: float = 0.5, fpr_target: float = 0.01, device=None) -> dict:
    """PR-AUC, ROC-AUC, F1, precision, recall (at ``threshold``) and recall at ``fpr_target``."""
    y = np.asarray(y).astype(int)
    p = predict_proba(model, X, device=device)
    pred = (p >= threshold).astype(int)
    two_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "fraud_rate": float(y.mean()) if len(y) else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if two_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        f"recall_at_{fpr_target:g}_fpr": recall_at_fpr(y, p, fpr_target),
        "threshold": float(threshold),
    }


# ----------------------------------------------------------------------------- 5. baseline run
def run_baseline(bank: str | int = 0, epochs: int = 30, patience: int = 5, lr: float = 1e-3, batch_size: int = 512,
                 seed: int = 0, n_banks: int = 5, alpha: float = 0.5, data_seed: int = 42,
                 partition: str = "label_skew", out_csv: str | Path | None = "results/identity_baseline.csv") -> dict:
    """Train on one bank's private data (or on all banks pooled) and score on the global test set.

    The decision threshold is chosen on that bank's validation data, never on test.
    """
    from data.loader import get_pos_weight, input_dim, load_client_data, load_test  # run from the repo root

    kw = dict(n_banks=n_banks, alpha=alpha, seed=data_seed, partition=partition)
    if str(bank) == "pooled":
        parts = [load_client_data(b, FRAUD_TYPE, **kw) for b in range(n_banks)]
        X_tr, y_tr = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        X_va, y_va = np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts])
        pos = int(y_tr.sum())
        pos_w = float((len(y_tr) - pos) / pos) if pos else 1.0
    else:
        bank = int(bank)
        X_tr, y_tr, X_va, y_va = load_client_data(bank, FRAUD_TYPE, **kw)
        pos_w = get_pos_weight(bank, FRAUD_TYPE, **kw)
    X_te, y_te = load_test(FRAUD_TYPE, **kw)

    model = build_model(input_dim(FRAUD_TYPE, **kw), seed=seed)
    log.info("bank=%s  params=%d  train=%d (%.2f%% fraud)  pos_weight=%.1f", bank, model.n_parameters(),
             len(y_tr), 100 * y_tr.mean(), pos_w)
    t0 = time.time()
    train_local(model, X_tr, y_tr, X_va, y_va, pos_w, epochs, patience, lr, batch_size, seed=seed)
    thr = best_f1_threshold(y_va, predict_proba(model, X_va))
    metrics = evaluate(model, X_te, y_te, threshold=thr)
    metrics.update(bank=str(bank), mode="local" if str(bank) != "pooled" else "centralised",
                   fraud_type=FRAUD_TYPE, seconds=round(time.time() - t0, 1))
    log.info("TEST  PR-AUC %.4f  ROC-AUC %.4f  F1 %.4f  precision %.4f  recall %.4f  recall@1%%FPR %.4f",
             metrics["pr_auc"], metrics["roc_auc"], metrics["f1"], metrics["precision"], metrics["recall"],
             metrics["recall_at_0.01_fpr"])
    if out_csv:
        out = Path(out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        new = not out.exists()
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics))
            if new:
                w.writeheader()
            w.writerow(metrics)
    return metrics


def _main() -> None:
    ap = argparse.ArgumentParser(description="Baseline for the identity-fraud model (local-only or pooled).")
    ap.add_argument("--bank", default="0", help="bank id (0..n_banks-1) or 'pooled' for the centralised baseline")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0, help="model/training seed")
    ap.add_argument("--n-banks", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--data-seed", type=int, default=42)
    ap.add_argument("--partition", default="label_skew", choices=["label_skew", "iid", "quantity_skew"])
    ap.add_argument("--out", default="results/identity_baseline.csv", help="metrics CSV to append to ('' to skip)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_baseline(a.bank, a.epochs, a.patience, a.lr, a.batch_size, a.seed, a.n_banks, a.alpha, a.data_seed,
                 a.partition, a.out or None)


if __name__ == "__main__":
    _main()
