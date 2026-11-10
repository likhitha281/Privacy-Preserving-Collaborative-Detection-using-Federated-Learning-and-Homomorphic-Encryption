"""Flower client for one simulated bank, used by both "fedavg" and "fedavg_he".

The code path is identical in both modes except for two hooks:
  * downlink: if the fit/evaluate config says ``encrypted=True`` the incoming parameters are uint8-wrapped CKKS
    chunks and are decrypted; otherwise they are plaintext weights (always the case in round 1, whose downlink
    is the shared initial model).
  * uplink: in "fedavg_he" the trained weights are scaled by n_i/N, encrypted and wrapped as uint8 arrays.
"""
from __future__ import annotations

import hashlib
from time import perf_counter

import flwr
import numpy as np
import torch
from torch import nn

from common import interfaces as I
from common.config import ExperimentConfig
from common.utils import get_weights, set_seed, set_weights

from .metrics import binary_metrics, predict_logits


# ----------------------------------------------------------------------------- local training / evaluation
def train_model(model, X_train, y_train, pos_weight, epochs, lr, batch_size) -> list[float]:
    """Train on one bank's private data; returns the mean loss of each epoch. Seed before calling."""
    model.train()
    # copy=True avoids PyTorch warnings caused by read-only NumPy arrays loaded from parquet.
    X = torch.from_numpy(np.array(X_train, dtype=np.float32, copy=True))
    y = torch.from_numpy(np.array(y_train, dtype=np.float32, copy=True))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(len(X))  # uses the global torch RNG, seeded per (round, bank) in fit()
        total = 0.0
        for start in range(0, len(X), batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X[idx]), y[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        losses.append(total / max(len(X), 1))
    return losses


def evaluate_model(model, X, y, threshold: float = 0.5) -> dict:
    """Unweighted BCE loss plus accuracy, precision, recall, F1, ROC-AUC and PR-AUC at ``threshold``."""
    logits = predict_logits(model, X)
    return binary_metrics(y, torch.sigmoid(logits).numpy(), threshold, logits=logits)


def fit_seed(seed: int, server_round: int, bank_id: int) -> int:
    """Per-(round, bank) seed so local training (shuffling, dropout) is identical across modes."""
    return seed * 100_003 + server_round * 101 + bank_id


def payload_bytes(arrays) -> int:
    """Raw payload size (ciphertext bytes or plaintext array bytes); excludes Flower's message envelope."""
    return int(sum(np.asarray(a).nbytes for a in arrays))


# ----------------------------------------------------------------------------- Flower client
class FraudClient(flwr.client.NumPyClient):
    """One simulated bank. ``he_ctx`` is a CKKSClient (secret context) in "fedavg_he" mode, else None."""

    def __init__(self, bank_id: int, cfg: ExperimentConfig, model, data, pos_weight: float,
                 he_ctx=None, total_samples: int | None = None):
        if (cfg.mode == "fedavg_he") != (he_ctx is not None):
            raise ValueError("he_ctx is required in fedavg_he mode and must be None otherwise")
        self.bank_id, self.cfg, self.model, self.pos_weight = bank_id, cfg, model, pos_weight
        self.X_train, self.y_train, self.X_val, self.y_val = data
        self.he_ctx = he_ctx
        self.scale = I.sample_scale(len(self.y_train), total_samples) if he_ctx is not None else None
        self._cache = (None, None)  # (sha256 of last downlink, its decrypted weights)
        # Plaintext weights of the last fit, before scaling/encryption. Read only by the trusted diagnostic
        # harness (max_weight_err); never sent anywhere.
        self.last_update = None

    def _load_downlink(self, parameters, config) -> float:
        """Load the global model into self.model; returns the decryption time (0 for plaintext)."""
        if not config["encrypted"]:
            set_weights(self.model, parameters)
            return 0.0
        chunks = I.ndarrays_to_bytes(parameters)
        digest = hashlib.sha256(b"".join(chunks)).hexdigest()
        t, start = 0.0, perf_counter()
        if digest != self._cache[0]:  # evaluate() and the next fit() receive the same aggregate
            self._cache = (digest, I.decrypt(chunks, self.he_ctx))
            t = perf_counter() - start
        set_weights(self.model, self._cache[1])
        return t

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        decrypt_s = self._load_downlink(parameters, config)
        set_seed(fit_seed(self.cfg.seed, int(config["server_round"]), self.bank_id))
        start = perf_counter()
        losses = train_model(self.model, self.X_train, self.y_train, self.pos_weight,
                             self.cfg.local_epochs, self.cfg.lr, self.cfg.batch_size)
        metrics = {"bank_id": self.bank_id, "train_loss": losses[-1], "train_s": perf_counter() - start,
                   "decrypt_s": decrypt_s, "encrypt_s": 0.0}
        weights = self.last_update = get_weights(self.model)
        if self.he_ctx is not None:
            start = perf_counter()
            weights = I.bytes_to_ndarrays(I.encrypt(weights, self.scale, self.he_ctx))
            metrics["encrypt_s"] = perf_counter() - start
        metrics["uplink_bytes"] = payload_bytes(weights)
        # num_examples drives Flower's sample-weighted FedAvg; the HE strategy uses the agreed counts instead.
        return weights, len(self.y_train), metrics

    def evaluate(self, parameters, config):
        decrypt_s = self._load_downlink(parameters, config)
        m = evaluate_model(self.model, self.X_val, self.y_val)
        return m["loss"], len(self.y_val), {"bank_id": self.bank_id, "f1": m["f1"], "decrypt_s": decrypt_s}


def create_client(bank_id: int, cfg: ExperimentConfig, he_ctx=None, total_samples: int | None = None, data=None):
    """Build one bank's client from its private partition (all loader settings come from cfg).
    ``data`` (X_train, y_train, X_val, y_val) can be passed in to avoid reloading it."""
    data = data if data is not None else I.load_client_data(bank_id, cfg.fraud_type, cfg)
    model = I.build_model(cfg.fraud_type, data[0].shape[1], seed=cfg.seed)
    return FraudClient(bank_id, cfg, model, data, I.pos_weight(bank_id, cfg.fraud_type, cfg), he_ctx, total_samples)
