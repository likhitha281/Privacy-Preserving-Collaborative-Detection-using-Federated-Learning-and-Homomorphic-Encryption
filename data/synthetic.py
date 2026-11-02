"""Synthetic data in the SHAPE of IEEE-CIS (same file names and columns), for tests and demos only.

Nothing here reflects real fraud behaviour. Fraud is generated as account-takeover-like bursts
(a new address, e-mail and device, larger amounts, many transactions in two hours) so that the
behavioural features have something to find. Use it to check the pipeline runs, never to report results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EMAILS = np.array(["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "icloud.com"])
BROWSERS = np.array(["chrome 63.0", "safari 11.0", "firefox 57.0", "edge 16.0", "samsung browser 6.2"])
ID_CATS = {"id-12": ["Found", "NotFound"], "id-15": ["New", "Found", "Unknown"],
           "id-16": ["Found", "NotFound"], "id-28": ["New", "Found"], "id-29": ["Found", "NotFound"],
           "id-35": ["T", "F"], "id-36": ["T", "F"], "id-37": ["T", "F"], "id-38": ["T", "F"]}


def generate(out_dir, n_txn: int = 30000, n_cards: int = 1500, seed: int = 0, n_bursts: int = 110,
             id_rate: float = 0.25, days: int = 180) -> dict:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    T = days * 86400.0

    # per-card static attributes
    card1 = np.arange(10000, 10000 + n_cards)
    home = rng.integers(100, 500, n_cards).astype(float)
    c_email = rng.choice(EMAILS, n_cards)
    c_dev = rng.choice(np.array([f"Dev{i}" for i in range(80)]), n_cards)
    c_card4 = rng.choice(["visa", "mastercard", "american express", "discover"], n_cards, p=[.6, .3, .05, .05])
    c_card6 = rng.choice(["debit", "credit"], n_cards, p=[.7, .3])
    c_card2 = rng.integers(100, 600, n_cards).astype(float)
    c_card5 = rng.choice([102., 126., 166., 226.], n_cards)
    c_scale = rng.lognormal(0, 0.4, n_cards)
    weight = rng.pareto(1.5, n_cards) + 1
    weight /= weight.sum()

    # fraud bursts (account-takeover-like)
    b_card = rng.integers(0, n_cards, n_bursts)
    b_size = rng.integers(6, 13, n_bursts)
    b_t0 = rng.uniform(0.2 * T, 0.95 * T, n_bursts)
    f_idx = np.repeat(b_card, b_size)
    f_t = np.concatenate([t0 + np.sort(rng.uniform(0, 7200, s)) for t0, s in zip(b_t0, b_size)])
    n_f = len(f_idx)
    n_l = n_txn - n_f

    l_idx = rng.choice(n_cards, n_l, p=weight)
    l_t = rng.uniform(0, T, n_l)

    idx = np.concatenate([l_idx, f_idx])
    t = np.concatenate([l_t, f_t])
    fraud = np.r_[np.zeros(n_l, int), np.ones(n_f, int)]
    order = np.argsort(t, kind="stable")
    idx, t, fraud = idx[order], t[order], fraud[order]
    n = len(idx)
    fr = fraud == 1

    amt = rng.lognormal(np.log(50), 1.0, n) * c_scale[idx] * np.where(fr, rng.uniform(1, 2.5, n), 1.0)
    addr = np.where(fr | (rng.random(n) > 0.96), rng.integers(500, 900, n), home[idx]).astype(float)
    email = np.where(fr | (rng.random(n) > 0.97), rng.choice(EMAILS, n), c_email[idx])
    email = np.where(fr, rng.choice(["protonmail.com", "anonymous.com", "mail.com"], n), email)
    dev = np.where(fr, rng.choice(np.array([f"Dev{i}" for i in range(80, 140)]), n),
                   np.where(rng.random(n) > 0.95, rng.choice(np.array([f"Dev{i}" for i in range(80)]), n),
                            c_dev[idx]))

    def nanify(a, p):
        a = a.astype(float)
        a[rng.random(len(a)) < p] = np.nan
        return a

    tx = pd.DataFrame({"TransactionID": 2987000 + np.arange(n), "isFraud": fraud,
                       "TransactionDT": (t + 86400).astype(int), "TransactionAmt": np.round(amt, 2),
                       "ProductCD": rng.choice(list("WCHSR"), n, p=[.6, .1, .1, .1, .1])})
    for c, v in zip(["card1", "card2", "card3", "card5"],
                    [card1[idx], c_card2[idx], np.full(n, 150.0), c_card5[idx]]):
        tx[c] = v
    tx["card4"], tx["card6"] = c_card4[idx], c_card6[idx]
    tx = tx[["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
             "card1", "card2", "card3", "card4", "card5", "card6"]]
    tx["addr1"], tx["addr2"] = addr, np.where(rng.random(n) < 0.03, np.nan, 87.0)
    tx["dist1"], tx["dist2"] = nanify(rng.exponential(30, n), 0.8), nanify(rng.exponential(60, n), 0.9)
    tx["P_emaildomain"] = email
    tx["R_emaildomain"] = np.where(rng.random(n) < 0.75, None, email)
    for i in range(1, 15):
        tx[f"C{i}"] = rng.poisson(1 + 0.3 * fr)
    for i in range(1, 16):
        tx[f"D{i}"] = nanify(rng.exponential(40, n) * (1 - 0.1 * fr), 0.4 + 0.02 * i)
    for i in range(1, 10):
        vals = np.where(rng.random(n) < 0.5, "T", "F")
        if i == 4:
            vals = rng.choice(["M0", "M1", "M2"], n)
        tx[f"M{i}"] = np.where(rng.random(n) < 0.4, None, vals)

    # identity table: only a subset of transactions, more likely for fraud; hyphenated names
    has_id = rng.random(n) < np.where(fr, 0.8, id_rate)
    m = int(has_id.sum())
    idn = pd.DataFrame({"TransactionID": tx.loc[has_id, "TransactionID"].to_numpy()})
    for i in range(1, 12):
        idn[f"id-{i:02d}"] = np.round(rng.normal(0, 10, m), 1)
    for c, opts in ID_CATS.items():
        idn[c] = rng.choice(opts, m)
    idn["id-31"] = rng.choice(BROWSERS, m)
    idn["DeviceType"] = rng.choice(["desktop", "mobile"], m)
    idn["DeviceInfo"] = dev[has_id]

    tx.to_csv(out / "train_transaction.csv", index=False)
    idn.to_csv(out / "train_identity.csv", index=False)
    return dict(rows=n, fraud=int(fraud.sum()), identity_rows=m, cards=n_cards)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-txn", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(generate(a.out_dir, a.n_txn, seed=a.seed))
