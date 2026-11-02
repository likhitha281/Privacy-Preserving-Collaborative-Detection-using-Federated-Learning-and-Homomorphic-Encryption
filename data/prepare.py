"""Build the simulated federation.

    python -m data.prepare --dataset ieee --data-dir data/raw/ieee --out-dir data/processed/ieee_label_skew_b5_a0.5_s42

Steps: load -> clean -> causal behavioural features -> global time split (train/val/test) ->
fit encoder on global TRAIN only -> assign cards to banks -> write one folder per bank.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from .partition import MODES, partition

META = ["meta__TransactionID", "meta__TransactionDT", "meta__group"]


def _ulb_groups() -> dict:
    v = [f"V{i}" for i in range(1, 29)]
    return {"txn": {"num": v + ["TransactionAmt", "hour_sin", "hour_cos", "dow_sin", "dow_cos"],
                    "flag": [], "cat": []}}


def split_report(m: dict, summary: pd.DataFrame) -> str:
    """Human-readable split report: class balance per bank and split."""
    L = [f"# Split report ({m['dataset']}, {m['n_banks']} banks, {m['partition']}, alpha={m['alpha']}, seed={m['seed']})", "",
         f"Rows: {m['rows']:,}. Frauds: {m['fraud']:,} ({m['fraud_rate']:.2%}). "
         f"Time split: train {1 - m['val_frac'] - m['test_frac']:.0%}, validation {m['val_frac']:.0%}, "
         f"global test {m['test_frac']:.0%} (latest transactions by TransactionDT).", "",
         "| split | rows | frauds | fraud rate |", "|---|---|---|---|"]
    for s in ("train", "val", "test"):
        L.append(f"| {s} | {m['split_rows'][s]:,} | {m['split_fraud'][s]:,} | "
                 f"{m['split_fraud'][s] / m['split_rows'][s]:.2%} |")
    L += ["", "## Class balance per bank", "",
          "| bank | split | rows | frauds | fraud rate | cards |", "|---|---|---|---|---|---|"]
    for r in summary.itertuples():
        L.append(f"| {r.bank} | {r.split} | {r.rows:,} | {r.fraud:,} | {r.fraud_rate:.2%} | {r.groups:,} |")
    L += ["", "## Feature views (columns)", ""] + [f"- {g}: {n}" for g, n in m["n_features"].items()]
    if m["warnings"]:
        L += ["", "## Warnings", ""] + [f"- {w}" for w in m["warnings"]]
    return "\n".join(L) + "\n"


def run(dataset: str, data_dir, out_dir, n_banks: int = 5, partition_mode: str = "label_skew",
        alpha: float = 0.5, seed: int = 42, include_v: bool = False, min_fraud_train: int = 100,
        top_k: int = 12, val_frac: float = 0.10, test_frac: float = 0.20) -> dict:
    t_start = time.time()
    warnings: list[str] = []
    if partition_mode not in MODES:
        raise ValueError(f"partition_mode must be one of {MODES}")

    # ---- load + features ---------------------------------------------------------------
    if dataset == "ieee":
        raw = F.load_ieee(data_dir, include_v)
        v_cols = list(raw.attrs.get("v_cols", []))
        df = F.add_behavioral(F.clean(raw))
        groups = F.make_groups(v_cols)
        df["card1"] = df["card1"].fillna(-1).astype("int64")
        group_col = df["card1"]
        id_cov = float(df["has_identity"].mean())
    elif dataset == "ulb":
        df = F.clean(F.load_ulb(Path(data_dir) / "creditcard.csv"))
        groups, v_cols, id_cov = _ulb_groups(), [], None
        group_col = df["TransactionID"]  # no entity id in ULB: every transaction is its own group
        warnings.append("ULB has no card/account id: no behavioural or identity groups; rows are "
                        "partitioned individually, so histories are not meaningful.")
    else:
        raise ValueError("dataset must be 'ieee' or 'ulb'")

    # ---- global time split ---------------------------------------------------------------
    dt = df["TransactionDT"].to_numpy(float)
    t1, t2 = np.quantile(dt, [1 - val_frac - test_frac, 1 - test_frac])
    is_train, is_val = dt <= t1, (dt > t1) & (dt <= t2)
    is_test = dt > t2
    split = np.where(is_train, "train", np.where(is_val, "val", "test"))
    y = df["isFraud"].to_numpy()
    for name, m in (("val", is_val), ("test", is_test)):
        if y[m].sum() < 20:
            warnings.append(f"{name} split has only {int(y[m].sum())} fraud rows: metrics will be noisy.")

    # ---- encode (train-only fit) -----------------------------------------------------------
    enc = F.FeatureEncoder(groups, top_k=top_k).fit(df[is_train])
    X = enc.transform(df)

    # ---- federation ------------------------------------------------------------------------
    bank = partition(group_col, df["isFraud"], is_train, n_banks, partition_mode, alpha, seed,
                     min_fraud_train=min_fraud_train)

    # ---- write -----------------------------------------------------------------------------
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("bank_*"):
        shutil.rmtree(old)
    meta = pd.DataFrame({"meta__TransactionID": df["TransactionID"].to_numpy(),
                         "meta__TransactionDT": dt, "meta__group": group_col.to_numpy(),
                         "label": y.astype("int8")})
    rows = []
    for b in range(n_banks):
        (out / f"bank_{b}").mkdir()
        for s in ("train", "val", "test"):
            idx = np.flatnonzero((bank == b) & (split == s))
            part = pd.concat([meta.iloc[idx].reset_index(drop=True),
                              X.iloc[idx].reset_index(drop=True)], axis=1)
            part.to_parquet(out / f"bank_{b}" / f"{s}.parquet", index=False)
            n_f = int(part["label"].sum())
            rows.append(dict(bank=b, split=s, rows=len(part), fraud=n_f,
                             fraud_rate=n_f / len(part) if len(part) else float("nan"),
                             groups=int(part["meta__group"].nunique())))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "partition_summary.csv", index=False)

    fg = {g: [c for c in enc.columns if enc.column_group[c] == g] for g in groups}
    schema = dict(dataset=dataset, n_banks=n_banks, label="label", meta_columns=META,
                  dtype="float32", feature_groups=fg)
    (out / "schema.json").write_text(json.dumps(schema, indent=2))
    (out / "encoder.json").write_text(json.dumps(enc.to_dict(), indent=2))

    tr = summary[summary.split == "train"]
    manifest = dict(
        dataset=dataset, seed=seed, n_banks=n_banks, partition=partition_mode, alpha=alpha,
        include_v=include_v, val_frac=val_frac, test_frac=test_frac, rows=int(len(df)), fraud=int(y.sum()), fraud_rate=float(y.mean()),
        split_rows={s: int((split == s).sum()) for s in ("train", "val", "test")},
        split_fraud={s: int(y[split == s].sum()) for s in ("train", "val", "test")},
        time_cutoffs=dict(train_end=float(t1), val_end=float(t2)),
        identity_coverage=id_cov, encoder_fit_rows=int(is_train.sum()),
        n_features={g: len(c) for g, c in fg.items()}, n_v_columns=len(v_cols),
        train_fraud_rate_per_bank=[float(r) for r in tr["fraud_rate"]],
        warnings=warnings, seconds=round(time.time() - t_start, 1),
        notes=["Encoder fit on the pooled train split (simulation shortcut; a real federation "
               "would share only per-bank statistics).",
               "Behavioural features use only earlier rows of the same card (causal)."])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "split_report.md").write_text(split_report(manifest, summary))
    return manifest


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["ieee", "ulb"], default="ieee")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-banks", type=int, default=5)
    p.add_argument("--partition", choices=MODES, default="label_skew")
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration; smaller = more skew")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-v", action="store_true", help="also keep Vesta's V1-V339 columns")
    p.add_argument("--min-fraud-train", type=int, default=100)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--val-frac", type=float, default=0.10, help="share of the latest-in-time rows used for validation")
    p.add_argument("--test-frac", type=float, default=0.20, help="share of the latest-in-time rows held out as the global test set")
    a = p.parse_args(argv)
    m = run(a.dataset, a.data_dir, a.out_dir, a.n_banks, a.partition, a.alpha, a.seed,
            a.include_v, a.min_fraud_train, a.top_k, a.val_frac, a.test_frac)
    print(json.dumps({k: m[k] for k in ("rows", "fraud", "fraud_rate", "split_rows", "split_fraud",
                                        "identity_coverage", "n_features", "train_fraud_rate_per_bank",
                                        "warnings", "seconds")}, indent=2))
    return m


if __name__ == "__main__":
    main()
