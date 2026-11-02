"""Data access for the whole project (portion: data engineering + federation simulation).

The functions in section 1 are the contract from the Implementation Architecture Plan:

    from data.loader import load_client_data, load_test

    X_train, y_train, X_val, y_val = load_client_data(bank_id=0, fraud_type="credit_card")
    X_test, y_test                 = load_test("credit_card")

`fraud_type` selects a feature view (IEEE-CIS has one `isFraud` label, so each fraud type is a proxy view):

    credit_card       -> txn  transaction and card fields
    account_takeover  -> beh  per-card velocity, new address / e-mail / device flags
    identity_fraud    -> idn  identity-table fields, device info, e-mail domains

The first call builds the federation from the raw CSVs (under a minute) and caches it under
data/processed/; later calls just read parquet files. Raw files go in data/raw/ieee/.
Features are float32; labels are int8 (cast with `.astype("float32")` for BCEWithLogitsLoss).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_CFG = {"raw_dir": HERE / "raw" / "ieee", "processed_dir": HERE / "processed"}
DEFAULTS = dict(n_banks=5, alpha=0.5, seed=42, partition="label_skew")
VIEWS = {"credit_card": ["txn"], "account_takeover": ["beh"], "identity_fraud": ["idn"]}
ALL_SPLITS = ("train", "val", "test")


def configure(raw_dir=None, processed_dir=None) -> None:
    """Override where raw CSVs are read from / processed data is cached (used by tests)."""
    if raw_dir is not None:
        _CFG["raw_dir"] = Path(raw_dir)
    if processed_dir is not None:
        _CFG["processed_dir"] = Path(processed_dir)


def federation_dir(n_banks=5, alpha=0.5, seed=42, partition="label_skew") -> Path:
    return _CFG["processed_dir"] / f"ieee_{partition}_b{n_banks}_a{alpha}_s{seed}"


def ensure_federation(n_banks=5, alpha=0.5, seed=42, partition="label_skew") -> Path:
    """Return the folder of the requested federation, building it from the raw CSVs if it is missing."""
    root = federation_dir(n_banks, alpha, seed, partition)
    if not (root / "manifest.json").exists():
        from .prepare import run  # imported lazily: only needed for the first build
        run("ieee", _CFG["raw_dir"], root, n_banks=n_banks, partition_mode=partition, alpha=alpha, seed=seed)
    return root


def _groups(fraud_type: str) -> list[str]:
    if fraud_type not in VIEWS:
        raise ValueError(f"fraud_type must be one of {sorted(VIEWS)}, got {fraud_type!r}")
    return VIEWS[fraud_type]


# ----------------------------------------------------------------------------- 1. project interface
def load_client_data(bank_id: int, fraud_type: str, n_banks: int = 5, alpha: float = 0.5, seed: int = 42,
                     partition: str = "label_skew"):
    """Private data of one simulated bank: X_train, y_train, X_val, y_val."""
    root = ensure_federation(n_banks, alpha, seed, partition)
    if not 0 <= bank_id < n_banks:
        raise ValueError(f"bank_id must be in 0..{n_banks - 1}, got {bank_id}")
    g = _groups(fraud_type)
    X_train, y_train = load_bank(root, bank_id, "train", g)
    X_val, y_val = load_bank(root, bank_id, "val", g)
    return X_train, y_train, X_val, y_val


def load_test(fraud_type: str, n_banks: int = 5, alpha: float = 0.5, seed: int = 42,
              partition: str = "label_skew"):
    """Global held-out test set (latest 20% of transactions, all banks pooled): X_test, y_test."""
    return load_global(ensure_federation(n_banks, alpha, seed, partition), "test", _groups(fraud_type))


def input_dim(fraud_type: str, n_banks: int = 5, alpha: float = 0.5, seed: int = 42,
              partition: str = "label_skew") -> int:
    """Number of input features of a view, for build_model(input_dim)."""
    return len(feature_columns(ensure_federation(n_banks, alpha, seed, partition), _groups(fraud_type)))


def get_pos_weight(bank_id: int, fraud_type: str, n_banks: int = 5, alpha: float = 0.5, seed: int = 42,
                   partition: str = "label_skew") -> float:
    """negatives / positives in this bank's train split, for BCE pos_weight (1.0 if the bank has no fraud)."""
    _, y, _, _ = load_client_data(bank_id, fraud_type, n_banks, alpha, seed, partition)
    pos = int(y.sum())
    return float((len(y) - pos) / pos) if pos else 1.0


# ----------------------------------------------------------------------------- 2. lower-level access
def _schema(root) -> dict:
    return json.loads((Path(root) / "schema.json").read_text())


def feature_groups(root) -> list[str]:
    return list(_schema(root)["feature_groups"])


def feature_columns(root, groups=None) -> list[str]:
    s = _schema(root)
    groups = list(groups) if groups else list(s["feature_groups"])
    unknown = set(groups) - set(s["feature_groups"])
    if unknown:
        raise ValueError(f"unknown feature groups {sorted(unknown)}; available: {list(s['feature_groups'])}")
    return [c for g in groups for c in s["feature_groups"][g]]


def n_banks(root) -> int:
    return int(_schema(root)["n_banks"])


def load_bank(root, bank: int, split: str = "train", groups=None, as_numpy: bool = True):
    """Features and labels of one bank for one split. Returns (X, y)."""
    if split not in ALL_SPLITS:
        raise ValueError(f"split must be one of {ALL_SPLITS}")
    cols = feature_columns(root, groups)
    df = pd.read_parquet(Path(root) / f"bank_{bank}" / f"{split}.parquet", columns=cols + ["label"])
    X, y = df[cols], df["label"]
    return (X.to_numpy(np.float32), y.to_numpy(np.int8)) if as_numpy else (X, y)


def load_global(root, split: str = "train", groups=None, as_numpy: bool = True):
    """All banks concatenated: the centralised baseline uses exactly the same rows as the federation."""
    parts = [load_bank(root, b, split, groups, as_numpy=False) for b in range(n_banks(root))]
    X = pd.concat([p[0] for p in parts], ignore_index=True)
    y = pd.concat([p[1] for p in parts], ignore_index=True)
    return (X.to_numpy(np.float32), y.to_numpy(np.int8)) if as_numpy else (X, y)
