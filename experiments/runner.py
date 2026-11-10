"""run_experiment(cfg): one run of one mode, written to results/<fraud_type>_<mode>_b<n>_a<alpha>_s<seed>.csv.

Modes
-----
* "fedavg", "fedavg_he": fl.run_federated (see fl/server.py).
* "local": no federation. Each bank trains only on its own partition, starting from the same seeded initial
  weights as the federated global model, with the same train_model / fit_seed per (round, bank) and the same
  budget of rounds * local_epochs epochs (local_epochs per "round", fresh Adam each round exactly like a
  federated client). After every round each bank's model is scored on the shared global test set.
  Row semantics for local mode: test metric columns are the mean across banks; bank_*_min/max give the spread of
  F1@0.5 and PR-AUC across banks; val_loss/val_f1 are each bank's model on its own validation split, weighted by
  validation size (as Flower's aggregate_evaluate does). Nothing is transmitted, so byte columns are 0 and
  crypto columns are null.

Tuned threshold (all modes): chosen every round from validation predictions only, never from test. Federated:
the global model's predictions on all banks' validation rows pooled. Local: each bank model's predictions on its
own validation rows, pooled into one set, giving one threshold applied to every bank model's test predictions.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from common import interfaces as I
from common.config import ExperimentConfig
from common.utils import set_seed
from fl import run_federated
from fl.client import fit_seed, train_model
from fl.logging import read_rows_csv, write_rows_csv
from fl.metrics import ROW_KEYS, binary_metrics, make_row, predict_logits, tuned_test_metrics, tuned_threshold

TEST_COLS = ("test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc", "test_pr_auc", "test_loss")


def result_path(cfg: ExperimentConfig, results_dir=None) -> Path:
    d = Path(results_dir if results_dir is not None else cfg.results_dir)
    return d / f"{cfg.fraud_type}_{cfg.mode}_b{cfg.n_clients}_a{cfg.alpha}_s{cfg.seed}.csv"


def run_local(cfg: ExperimentConfig) -> list[dict]:
    """Local-only baseline; see the module docstring for the row semantics."""
    if cfg.mode != "local":
        raise ValueError("run_local needs mode='local'")
    set_seed(cfg.seed)
    banks = range(cfg.n_clients)
    data = {b: I.load_client_data(b, cfg.fraud_type, cfg) for b in banks}
    pos = {b: I.pos_weight(b, cfg.fraud_type, cfg) for b in banks}
    X_test, y_test = I.load_test(cfg.fraud_type, cfg)
    models = {b: I.build_model(cfg.fraud_type, X_test.shape[1], seed=cfg.seed) for b in banks}
    rows = []
    for rnd in range(1, cfg.rounds + 1):
        start, losses, train_s, val = perf_counter(), [], [], {}
        for b, m in models.items():
            set_seed(fit_seed(cfg.seed, rnd, b))
            t = perf_counter()
            losses.append(train_model(m, data[b][0], data[b][1], pos[b], cfg.local_epochs, cfg.lr,
                                      cfg.batch_size)[-1])
            train_s.append(perf_counter() - t)
            lv = predict_logits(m, data[b][2])
            val[b] = (torch.sigmoid(lv).numpy(), binary_metrics(data[b][3], torch.sigmoid(lv).numpy(), logits=lv))
        wall = perf_counter() - start  # training + own-bank validation (cf. federated round_wall_s)

        thr = tuned_threshold(np.concatenate([data[b][3] for b in banks]), np.concatenate([val[b][0] for b in banks]))
        per_bank = []
        for m in models.values():
            lt = predict_logits(m, X_test)
            p = torch.sigmoid(lt).numpy()
            per_bank.append({**{f"test_{k}": v for k, v in binary_metrics(y_test, p, logits=lt).items()},
                             **tuned_test_metrics(thr, y_test, p)})
        n_val = np.array([len(data[b][3]) for b in banks], dtype=float)
        f1s, aps = [r["test_f1"] for r in per_bank], [r["test_pr_auc"] for r in per_bank]
        rows.append(make_row(
            cfg, rnd, **{k: float(np.mean([r[k] for r in per_bank]))
                         for k in (*TEST_COLS, "test_f1_tuned", "test_precision_tuned", "test_recall_tuned")},
            tuned_threshold=thr, bank_f1_min=min(f1s), bank_f1_max=max(f1s),
            bank_pr_auc_min=float(np.nanmin(aps)), bank_pr_auc_max=float(np.nanmax(aps)),
            val_loss=float(np.average([val[b][1]["loss"] for b in banks], weights=n_val)),
            val_f1=float(np.average([val[b][1]["f1"] for b in banks], weights=n_val)),
            train_loss=float(np.mean(losses)), train_s=float(np.mean(train_s)),
            uplink_bytes_per_client=0, downlink_bytes=0, round_wall_s=wall))
    return rows


def run_experiment(cfg: ExperimentConfig, results_dir=None, force: bool = False) -> list[dict]:
    """Run one config and write its CSV (atomically) plus a <name>.json copy of the config.

    If the CSV already exists and ``force`` is False the run is skipped and its rows are read back, so an
    interrupted grid resumes. A skipped file whose stored config differs from ``cfg`` (e.g. other rounds) raises.
    """
    path = result_path(cfg, results_dir)
    meta = path.with_suffix(".json")
    if path.exists() and not force:
        stored = json.loads(meta.read_text()) if meta.exists() else None
        if stored is not None and stored != cfg.to_dict():
            raise ValueError(f"{path} was produced with a different config {stored}; use force=True to rerun")
        rows = read_rows_csv(path)
        if not rows or tuple(rows[0]) != ROW_KEYS:
            raise ValueError(f"{path} does not have the current ROW_KEYS columns; use force=True to rerun")
        return rows
    rows = run_local(cfg) if cfg.mode == "local" else run_federated(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(cfg.to_dict(), indent=2))  # first: the CSV's existence marks completion
    write_rows_csv(rows, path, overwrite=True)
    return rows
