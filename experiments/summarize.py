"""Summaries of results/*.csv (one file per run, written by experiments.runner).

    python -m experiments.summarize [--results-dir results]

Writes summary_main.csv (alpha 0.5; fraud_type x mode x n_clients; mean/std over seeds of the final round),
summary_cost.csv (fedavg_he vs fedavg per-round costs) and summary_noniid.csv (credit_card alpha ablation).
slowdown = fedavg_he / fedavg round wall time (noisy: includes training-time variance between runs);
crypto_overhead = fedavg_he (encrypt + decrypt + aggregate seconds) / fedavg round wall time.
Cost columns are averaged over rounds 2..R of each run (round 1 includes one-off PyTorch warm-up; if a run has
only one round, that round is used), then over seeds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fl.metrics import ROW_KEYS

RUN_KEYS = ["fraud_type", "mode", "n_clients", "alpha", "seed"]
METRICS = ["test_f1", "test_f1_tuned", "test_precision", "test_recall", "test_roc_auc", "test_pr_auc"]
COSTS = ["encrypt_s", "decrypt_s", "aggregate_s", "uplink_bytes_per_client", "downlink_bytes", "round_wall_s"]
MAIN_ALPHA = 0.5


def load_rows(results_dir) -> pd.DataFrame:
    frames = []
    for f in sorted(Path(results_dir).glob("*.csv")):
        if f.name.startswith("summary_") or f.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if list(df.columns) == list(ROW_KEYS):  # skips unrelated files
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(ROW_KEYS))


def final_round(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.groupby(RUN_KEYS)["round"].idxmax()].reset_index(drop=True)


def _mean_std(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by)
    out = g[METRICS].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out.insert(0, "n_seeds", g["seed"].nunique())
    return out.reset_index()


def cost_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["mode"].isin(["fedavg", "fedavg_he"]) & (df["alpha"] == MAIN_ALPHA)]
    if df.empty:
        return pd.DataFrame()
    last = df.groupby(RUN_KEYS)["round"].transform("max")
    df = df[(df["round"] > 1) | (last == 1)]
    per_run = df.groupby(RUN_KEYS)[COSTS + ["max_weight_err"]].agg(
        {**{c: "mean" for c in COSTS}, "max_weight_err": "max"}).reset_index()
    per_cfg = per_run.groupby(["fraud_type", "n_clients", "mode"]).agg(
        {**{c: "mean" for c in COSTS}, "max_weight_err": "max"})
    wide = per_cfg.unstack("mode")
    wide.columns = [f"{mode}_{c}" for c, mode in wide.columns]
    wide = wide.drop(columns=[c for c in wide.columns if c == "fedavg_max_weight_err"])
    if {"fedavg_he_round_wall_s", "fedavg_round_wall_s"} <= set(wide.columns):
        wide["slowdown"] = wide["fedavg_he_round_wall_s"] / wide["fedavg_round_wall_s"]
        # Wall-time slowdown includes run-to-run training-time noise; crypto_overhead isolates HE's own cost.
        crypto = sum(wide[f"fedavg_he_{c}"] for c in ("encrypt_s", "decrypt_s", "aggregate_s"))
        wide["crypto_overhead"] = crypto / wide["fedavg_round_wall_s"]
    return wide.reset_index()


def summarize(results_dir="results", out_dir=None) -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir if out_dir is not None else results_dir)
    rows = load_rows(results_dir)
    tables = {"main": pd.DataFrame(), "cost": pd.DataFrame(), "noniid": pd.DataFrame()}
    if not rows.empty:
        final = final_round(rows)
        tables["main"] = _mean_std(final[final["alpha"] == MAIN_ALPHA], ["fraud_type", "mode", "n_clients"])
        cc = final[final["fraud_type"] == "credit_card"]
        if cc["alpha"].nunique() > 1:
            tables["noniid"] = _mean_std(cc, ["alpha", "mode", "n_clients"])
        tables["cost"] = cost_table(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(out_dir / f"summary_{name}.csv", index=False)
    return tables


def _fmt(t: pd.DataFrame, keys: list[str]) -> str:
    if t.empty:
        return "(no runs)"
    shown = t[keys + ["n_seeds"]].copy()
    for m in METRICS:
        std = t[f"{m}_std"].map(lambda s: "" if pd.isna(s) else f" ± {s:.3f}")
        shown[m.replace("test_", "")] = t[f"{m}_mean"].map("{:.3f}".format) + std
    return shown.to_string(index=False)


def print_tables(tables: dict) -> None:
    print("\n== Main results (final round, alpha 0.5; mean ± std over seeds) ==")
    print(_fmt(tables["main"], ["fraud_type", "mode", "n_clients"]))
    if not tables["noniid"].empty:
        print("\n== Non-IID ablation (credit_card) ==")
        print(_fmt(tables["noniid"], ["alpha", "mode", "n_clients"]))
    print("\n== Cost per round: fedavg_he vs fedavg ==")
    print("(no federated runs)" if tables["cost"].empty else
          tables["cost"].to_string(index=False, float_format=lambda x: f"{x:.4g}"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args(argv)
    print_tables(summarize(a.results_dir))


if __name__ == "__main__":
    main()
