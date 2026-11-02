"""Assign entities (cards) to simulated banks.

Partitioning is done per *group* (card1 for IEEE-CIS, one row per group for ULB), never per row,
so a card's history cannot be split across banks. Three modes:

    iid            groups are assigned uniformly at random (every bank looks alike)
    label_skew     fraud-groups and legit-groups are each spread over banks with Dirichlet(alpha)
                   proportions, so banks see different fraud rates (small alpha = more skew)
    quantity_skew  bank sizes follow Dirichlet(alpha), independent of the label
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MODES = ("iid", "label_skew", "quantity_skew")


def assign_groups(group_fraud: np.ndarray, n_banks: int, mode: str, alpha: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Return a bank id (0..n_banks-1) for every group. `group_fraud` is a bool array."""
    g = len(group_fraud)
    if mode == "iid":
        return rng.integers(0, n_banks, size=g)
    if mode == "quantity_skew":
        return rng.choice(n_banks, size=g, p=rng.dirichlet(np.full(n_banks, alpha)))
    if mode == "label_skew":
        bank = np.empty(g, dtype=int)
        for flag in (True, False):
            idx = np.flatnonzero(group_fraud == flag)
            bank[idx] = rng.choice(n_banks, size=len(idx), p=rng.dirichlet(np.full(n_banks, alpha)))
        return bank
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def partition(groups: pd.Series, y: pd.Series, is_train: np.ndarray, n_banks: int, mode: str,
              alpha: float, seed: int, min_fraud_train: int = 100,
              min_rows_train: int = 1000, max_tries: int = 500) -> np.ndarray:
    """Row-level bank ids. Retries (with a derived seed) until every bank has enough train data."""
    uniq, inv = np.unique(groups.to_numpy(), return_inverse=True)
    group_fraud = np.zeros(len(uniq), dtype=bool)
    np.logical_or.at(group_fraud, inv, y.to_numpy().astype(bool))

    for attempt in range(max_tries):
        rng = np.random.default_rng([seed, attempt])
        bank_of_group = assign_groups(group_fraud, n_banks, mode, alpha, rng)
        bank = bank_of_group[inv]
        fraud_tr = np.bincount(bank[is_train & (y.to_numpy() == 1)], minlength=n_banks)
        rows_tr = np.bincount(bank[is_train], minlength=n_banks)
        if (fraud_tr >= min_fraud_train).all() and (rows_tr >= min_rows_train).all():
            return bank
    raise RuntimeError(
        f"Could not give every bank >= {min_fraud_train} train frauds and >= {min_rows_train} train rows "
        f"in {max_tries} tries (mode={mode}, alpha={alpha}, n_banks={n_banks}). "
        "Increase alpha, reduce n_banks, or lower --min-fraud-train.")
