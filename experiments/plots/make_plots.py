"""Figures for the report: python -m experiments.plots.make_plots [--results results] [--out results/figures]

Reads the per-run CSVs in --results (experiments/runner.py; summaries are recomputed here, never read, so a grid
that is still running is fine) and the attack CSVs in --results/attacks. Every figure whose inputs are missing is
skipped with a printed note; nothing here raises on partial results. Each figure is saved as PNG (200 dpi) and
PDF, and --out/index.md lists the figures, their source files and a one-line takeaway computed from the data.

Figures: main_comparison, convergence_b3, convergence_b5, he_cost, noniid, label_flip, gradient_leakage.
"""
from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments import summarize  # noqa: E402
from fl.metrics import ROW_KEYS  # noqa: E402

MODES = ("local", "fedavg", "fedavg_he")
MODE_LABEL = {"local": "Local only", "fedavg": "FedAvg", "fedavg_he": "FedAvg + CKKS"}
# Okabe-Ito colour-blind-safe palette; one colour per mode in every figure.
COLOR = {"local": "#E69F00", "fedavg": "#0072B2", "fedavg_he": "#009E73", "baseline": "#7F7F7F"}
FRAUD_TYPES = ("credit_card", "account_takeover", "identity_fraud")
FRAUD_LABEL = {"credit_card": "Credit card", "account_takeover": "Account takeover\n(behavioural proxy)",
               "identity_fraud": "Identity\n(device proxy)"}
MAIN_ALPHA = summarize.MAIN_ALPHA
BANKS = (3, 5)
DPI = 200


def style() -> None:
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 12,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.3, "grid.linewidth": 0.8,
        "legend.frameon": False, "figure.dpi": 100, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


# ----------------------------------------------------------------------------- loading (never raises)
def note(msg: str) -> None:
    print(f"  note: {msg}", flush=True)


def load_runs(results) -> tuple[pd.DataFrame, list[str]]:
    """All per-run metric CSVs in ``results`` (same rule as summarize.load_rows, but a file that cannot be read,
    e.g. one being written, is skipped with a note instead of raising). Returns (rows with an extra ``source``
    column naming the file, source file names)."""
    frames, names = [], []
    for f in sorted(Path(results).glob("*.csv")):
        if f.name.startswith("summary_"):
            continue
        try:
            df = pd.read_csv(f)
        except Exception as exc:  # empty, partial or foreign file
            note(f"skipping unreadable {f.name} ({type(exc).__name__})")
            continue
        if list(df.columns) == list(ROW_KEYS) and len(df):
            frames.append(df.assign(source=f.name))
            names.append(f.name)
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[*ROW_KEYS, "source"])
    return rows, names


def read_attack_csv(results, stem: str) -> tuple[pd.DataFrame | None, list[str]]:
    """<results>/attacks/<stem>*.csv (excluding *_summary / *_checks files), concatenated; None if absent."""
    d = Path(results) / "attacks"
    files = sorted(f for f in d.glob(f"{stem}*.csv")
                   if not f.stem.endswith(("_summary", "_checks"))) if d.is_dir() else []
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as exc:
            note(f"skipping unreadable {f.name} ({type(exc).__name__})")
    frames = [f for f in frames if len(f)]
    if not frames:
        return None, []
    return pd.concat(frames, ignore_index=True), [f"attacks/{f.name}" for f in files]


def final_stats(rows: pd.DataFrame, metrics, by) -> pd.DataFrame:
    """Final round of every run, then mean/std/count over seeds for each group in ``by``."""
    final = summarize.final_round(rows)
    g = final.groupby(by)
    out = g[list(metrics)].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_seeds"] = g["seed"].nunique()
    return out.reset_index()


# ----------------------------------------------------------------------------- figure plumbing
@dataclass
class Figure:
    name: str
    files: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    takeaway: str = ""
    skipped: str | None = None


def used(df: pd.DataFrame) -> list[str]:
    return sorted(df["source"].unique()) if "source" in df.columns else []


def save(fig, out: Path, name: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for ext, kw in (("png", {"dpi": DPI}), ("pdf", {})):
        fig.savefig(out / f"{name}.{ext}", **kw)
        files.append(f"{name}.{ext}")
    plt.close(fig)
    return files


def mode_legend(ax_or_fig, modes, **kw):
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[m]) for m in modes]
    return ax_or_fig.legend(handles, [MODE_LABEL[m] for m in modes], **kw)


def fmt(x, nd=3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# ----------------------------------------------------------------------------- a. main comparison
def main_comparison(rows, sources, out) -> Figure:
    main = rows[rows["alpha"] == MAIN_ALPHA]
    f = Figure("main_comparison", sources=used(main))
    if main.empty:
        f.skipped = f"no runs at alpha {MAIN_ALPHA}"
        return f
    st = final_stats(main, ["test_pr_auc", "test_f1_tuned"], ["fraud_type", "mode", "n_clients"])
    banks = [b for b in BANKS if (st["n_clients"] == b).any()]
    fts = [ft for ft in FRAUD_TYPES if (st["fraud_type"] == ft).any()]
    metrics = (("test_pr_auc", "PR-AUC"), ("test_f1_tuned", "F1 at tuned threshold"))
    fig, axes = plt.subplots(2, len(banks), figsize=(6.5 * len(banks), 9), squeeze=False, sharey="row")
    width = 0.8 / len(MODES)
    for c, b in enumerate(banks):
        for r, (m, label) in enumerate(metrics):
            ax = axes[r][c]
            for i, mode in enumerate(MODES):
                for j, ft in enumerate(fts):
                    s = st[(st["n_clients"] == b) & (st["mode"] == mode) & (st["fraud_type"] == ft)]
                    if s.empty or pd.isna(s.iloc[0][f"{m}_mean"]):
                        ax.text(j + (i - 1) * width, 0.005, "n/a", ha="center", va="bottom", rotation=90,
                                fontsize=10, color="#555555")
                        continue
                    s = s.iloc[0]
                    err = s[f"{m}_std"] if s["n_seeds"] > 1 else None
                    ax.bar(j + (i - 1) * width, s[f"{m}_mean"], width, color=COLOR[mode], yerr=err,
                           capsize=4 if err is not None else 0, error_kw={"elinewidth": 1.2})
            ax.set_xticks(range(len(fts)), [FRAUD_LABEL[ft] for ft in fts])
            ax.set_ylim(0, None)
            if c == 0:
                ax.set_ylabel(f"{label} (test, final round)")
            if r == 0:
                ax.set_title(f"{b} banks")
    n_seeds = int(st["n_seeds"].max())
    mode_legend(fig, MODES, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.text(0.5, -0.01, f"alpha = {MAIN_ALPHA}; " + (f"mean ± std over up to {n_seeds} seeds" if n_seeds > 1
                                                     else "single seed, no error bars"),
             ha="center", fontsize=11, color="#555555")
    fig.tight_layout()
    f.files = save(fig, out, f.name)

    parts = []
    for b in banks:
        s = st[st["n_clients"] == b].pivot(index="fraud_type", columns="mode", values="test_pr_auc_mean")
        wins = [ft for ft in s.index if {"local", "fedavg"} <= set(s.columns)
                and not s.loc[ft, ["local", "fedavg"]].isna().any() and s.loc[ft, "fedavg"] > s.loc[ft, "local"]]
        both = [ft for ft in s.index if {"local", "fedavg"} <= set(s.columns)
                and not s.loc[ft, ["local", "fedavg"]].isna().any()]
        gap = (s["fedavg_he"] - s["fedavg"]).abs().max() if {"fedavg", "fedavg_he"} <= set(s.columns) else np.nan
        parts.append(f"{b} banks: FedAvg beats local-only PR-AUC on {len(wins)}/{len(both)} fraud types; "
                     f"max |FedAvg+CKKS - FedAvg| PR-AUC gap {fmt(gap)}")
    f.takeaway = "; ".join(parts) + "."
    return f


# ----------------------------------------------------------------------------- b. convergence
def round_stats(rows, metrics):
    g = rows.groupby(["fraud_type", "mode", "n_clients", "round"])
    out = g[list(metrics)].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_seeds"] = g["seed"].nunique()
    return out.reset_index()


def convergence(rows, sources, out, banks: int) -> Figure:
    sub = rows[(rows["alpha"] == MAIN_ALPHA) & (rows["n_clients"] == banks)]
    f = Figure(f"convergence_b{banks}", sources=used(sub))
    if sub.empty:
        f.skipped = f"no {banks}-bank runs at alpha {MAIN_ALPHA}"
        return f
    st = round_stats(sub, ["test_pr_auc", "test_f1_tuned"])
    fts = [ft for ft in FRAUD_TYPES if (st["fraud_type"] == ft).any()]
    metrics = (("test_pr_auc", "PR-AUC"), ("test_f1_tuned", "F1 at tuned threshold"))
    fig, axes = plt.subplots(2, len(fts), figsize=(5.2 * len(fts), 8), squeeze=False, sharex=True)
    modes = [m for m in MODES if (st["mode"] == m).any()]
    for c, ft in enumerate(fts):
        for r, (m, label) in enumerate(metrics):
            ax = axes[r][c]
            for mode in modes:
                s = st[(st["fraud_type"] == ft) & (st["mode"] == mode)].sort_values("round")
                if s.empty:
                    continue
                ax.plot(s["round"], s[f"{m}_mean"], color=COLOR[mode], lw=2.2,
                        ls="--" if mode == "fedavg_he" else "-", label=MODE_LABEL[mode])
                if s["n_seeds"].max() > 1:
                    ax.fill_between(s["round"], s[f"{m}_mean"] - s[f"{m}_std"].fillna(0),
                                    s[f"{m}_mean"] + s[f"{m}_std"].fillna(0), color=COLOR[mode], alpha=0.15, lw=0)
            if r == 0:
                ax.set_title(FRAUD_LABEL[ft].replace("\n", " "))
            if r == 1:
                ax.set_xlabel("Federated round")
            if c == 0:
                ax.set_ylabel(f"{label} (test)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(modes), bbox_to_anchor=(0.5, 1.04))
    fig.text(0.5, -0.01, f"{banks} banks, alpha = {MAIN_ALPHA}; FedAvg + CKKS dashed (it overlaps FedAvg); "
             "local only = mean over banks; band = ± std over seeds when >1 seed", ha="center", fontsize=11,
             color="#555555")
    fig.tight_layout()
    f.files = save(fig, out, f.name)
    p = st.pivot_table(index=["fraud_type", "round"], columns="mode", values="test_pr_auc_mean")
    if {"fedavg", "fedavg_he"} <= set(p.columns) and p[["fedavg", "fedavg_he"]].notna().all(axis=1).any():
        gap = (p["fedavg_he"] - p["fedavg"]).abs().max()
        f.takeaway = (f"{banks} banks: FedAvg+CKKS tracks FedAvg round by round (max PR-AUC difference {fmt(gap, 4)} "
                      f"over all rounds and fraud types).")
    else:
        f.takeaway = f"{banks} banks: FedAvg and FedAvg+CKKS are not both available yet for any fraud type."
    return f


# ----------------------------------------------------------------------------- c. HE cost
def he_cost(rows, sources, out) -> Figure:
    f = Figure("he_cost", sources=used(rows[rows["mode"].isin(["fedavg", "fedavg_he"])
                                           & (rows["alpha"] == MAIN_ALPHA)]))
    cost = summarize.cost_table(rows.drop(columns="source", errors="ignore"))
    if cost.empty or "fedavg_he_encrypt_s" not in cost.columns:
        f.skipped = f"no fedavg_he runs at alpha {MAIN_ALPHA}"
        return f
    cost = cost[cost["fedavg_he_encrypt_s"].notna()].copy()
    # cost_table has no train_s; add it (same averaging: rounds 2..R per run, then mean over runs)
    he = rows[(rows["mode"] == "fedavg_he") & (rows["alpha"] == MAIN_ALPHA)]
    last = he.groupby(summarize.RUN_KEYS)["round"].transform("max")
    he = he[(he["round"] > 1) | (last == 1)]
    tr = he.groupby(["fraud_type", "n_clients"])["train_s"].mean().rename("he_train_s").reset_index()
    cost = cost.merge(tr, on=["fraud_type", "n_clients"], how="left")
    cost["order"] = cost["fraud_type"].map({ft: i for i, ft in enumerate(FRAUD_TYPES)})
    cost = cost.sort_values(["order", "n_clients"]).reset_index(drop=True)
    short = {"credit_card": "Card", "account_takeover": "ATO", "identity_fraud": "Identity"}
    labels = [f"{short[ft]}\n{n} banks" for ft, n in zip(cost["fraud_type"], cost["n_clients"])]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2))
    # grouped (not stacked) on a log axis: crypto steps are ~100x shorter than training and vanish when stacked
    parts = (("he_train_s", "Local training", "#56B4E9"), ("fedavg_he_encrypt_s", "Encrypt", "#F0E442"),
             ("fedavg_he_decrypt_s", "Decrypt", "#CC79A7"), ("fedavg_he_aggregate_s", "Aggregate (server)", "#D55E00"))
    x = np.arange(len(cost))
    bw = 0.8 / len(parts)
    for k, (col, lab, colr) in enumerate(parts):
        a1.bar(x + (k - 1.5) * bw, cost[col].to_numpy(dtype=float), bw, color=colr, label=lab)
    a1.set_yscale("log")
    a1.set_xticks(x, labels)
    a1.set_ylabel("Time per round (s, log scale)")
    a1.set_title("FedAvg + CKKS: time per round")
    a1.legend(loc="upper left", ncol=2, fontsize=11)
    a1.set_ylim(None, a1.get_ylim()[1] * 8)
    a1.text(0, -0.25, "ATO = account takeover. train, encrypt, decrypt: mean per bank; aggregate: server", transform=a1.transAxes,
            fontsize=10, color="#555555")

    w = 0.38
    has_plain = "fedavg_uplink_bytes_per_client" in cost.columns
    if has_plain:
        a2.bar(x - w / 2, cost["fedavg_uplink_bytes_per_client"] / 1e6, w, color=COLOR["fedavg"],
               label=MODE_LABEL["fedavg"])
    a2.bar(x + (w / 2 if has_plain else 0), cost["fedavg_he_uplink_bytes_per_client"] / 1e6, w,
           color=COLOR["fedavg_he"], label=MODE_LABEL["fedavg_he"])
    a2.set_yscale("log")
    a2.set_xticks(x, labels)
    a2.set_ylabel("Uplink per bank per round (MB, log scale)")
    a2.set_title("Upload size")
    a2.legend(loc="upper left")
    a2.grid(axis="y", which="minor", alpha=0.1)
    ratios, overheads = [], []
    for i, r in cost.iterrows():
        ratio = (r["fedavg_he_uplink_bytes_per_client"] / r["fedavg_uplink_bytes_per_client"]
                 if has_plain and pd.notna(r.get("fedavg_uplink_bytes_per_client")) else np.nan)
        ov = r.get("crypto_overhead", np.nan)
        ratios.append(ratio)
        overheads.append(ov)
        txt = (f"×{ratio:.0f}" if pd.notna(ratio) else "") + (f"\ncrypto {100 * ov:.1f}%" if pd.notna(ov) else "")
        if txt:
            a2.annotate(txt, (i + (w / 2 if has_plain else 0), r["fedavg_he_uplink_bytes_per_client"] / 1e6),
                        textcoords="offset points", xytext=(0, 6), ha="center", fontsize=10)
    a2.set_ylim(None, a2.get_ylim()[1] * 4)
    a2.text(0, -0.25, "×: ciphertext / plaintext size; crypto: (encrypt+decrypt+aggregate) / FedAvg round time",
            transform=a2.transAxes, fontsize=10, color="#555555")
    fig.tight_layout()
    f.files = save(fig, out, f.name)
    r_ok = [r for r in ratios if pd.notna(r)]
    o_ok = [o for o in overheads if pd.notna(o)]
    he_time = cost[["fedavg_he_encrypt_s", "fedavg_he_decrypt_s", "fedavg_he_aggregate_s"]].sum(axis=1)
    f.takeaway = (f"Over {len(cost)} configurations, encrypt+decrypt+aggregate takes {fmt(he_time.min(), 3)}-"
                  f"{fmt(he_time.max(), 3)} s per round"
                  + (f"; ciphertext uploads are {min(r_ok):.0f}-{max(r_ok):.0f}x the plaintext size" if r_ok else "")
                  + (f"; crypto adds {100 * min(o_ok):.1f}-{100 * max(o_ok):.1f}% of a FedAvg round" if o_ok else "")
                  + ".")
    return f


# ----------------------------------------------------------------------------- d. non-IID
def noniid(rows, sources, out) -> Figure:
    cc = rows[rows["fraud_type"] == "credit_card"]
    f = Figure("noniid", sources=used(cc))
    if cc["alpha"].nunique() < 2:
        f.skipped = f"credit_card has runs at {cc['alpha'].nunique()} alpha value(s); need at least 2"
        return f
    st = final_stats(cc, ["test_pr_auc"], ["alpha", "mode", "n_clients"])
    banks = [b for b in BANKS if (st["n_clients"] == b).any()]
    fig, axes = plt.subplots(1, len(banks), figsize=(6.2 * len(banks), 5), squeeze=False, sharey=True)
    for c, b in enumerate(banks):
        ax = axes[0][c]
        for mode in MODES:
            s = st[(st["n_clients"] == b) & (st["mode"] == mode)].sort_values("alpha")
            if s.empty:
                continue
            err = s["test_pr_auc_std"].where(s["n_seeds"] > 1)
            ax.errorbar(s["alpha"], s["test_pr_auc_mean"], yerr=err.fillna(0) if err.notna().any() else None,
                        color=COLOR[mode], marker="o", ms=7, lw=2.2, capsize=4,
                        ls="--" if mode == "fedavg_he" else "-", label=MODE_LABEL[mode])
        ax.set_xscale("log")
        alphas = sorted(st["alpha"].unique())
        ax.set_xticks(alphas, [f"{a:g}" for a in alphas])
        ax.minorticks_off()
        ax.set_xlabel("Dirichlet alpha (smaller = more skewed banks)")
        ax.set_title(f"Credit card, {b} banks")
        if c == 0:
            ax.set_ylabel("PR-AUC (test, final round)")
    axes[0][0].legend(loc="lower right")
    fig.tight_layout()
    f.files = save(fig, out, f.name)
    parts = []
    for mode in MODES:
        s = st[st["mode"] == mode]
        if len(s):
            parts.append(f"{MODE_LABEL[mode]} PR-AUC {fmt(s['test_pr_auc_mean'].min())}-"
                         f"{fmt(s['test_pr_auc_mean'].max())}")
    f.takeaway = (f"Across alpha {', '.join(f'{a:g}' for a in sorted(st['alpha'].unique()))}: "
                  + "; ".join(parts) + ".")
    return f


# ----------------------------------------------------------------------------- e. label flip
def label_flip(results, out) -> Figure:
    f = Figure("label_flip")
    df, f.sources = read_attack_csv(results, "label_flip")
    need = {"mode", "n_clients", "seed", "flip_frac", "round", "test_pr_auc", "test_f1_tuned", "test_f1"}
    if df is None:
        f.skipped = "no attacks/label_flip*.csv"
        return f
    if not need <= set(df.columns):
        f.skipped = f"label_flip CSV lacks columns {sorted(need - set(df.columns))}"
        return f
    final = df.loc[df.groupby(["mode", "n_clients", "seed", "flip_frac"])["round"].idxmax()]
    st = final.groupby(["mode", "n_clients", "flip_frac"])[["test_pr_auc", "test_f1_tuned", "test_f1"]].mean()
    st = st.reset_index()
    banks = sorted(st["n_clients"].unique())
    modes = [m for m in ("fedavg", "fedavg_he") if (st["mode"] == m).any()]
    fig, axes = plt.subplots(len(banks), 2, figsize=(12, 4.6 * len(banks)), squeeze=False, sharex=True)
    for r, b in enumerate(banks):
        for mode in modes:
            s = st[(st["n_clients"] == b) & (st["mode"] == mode)].sort_values("flip_frac")
            ls = "--" if mode == "fedavg_he" else "-"
            axes[r][0].plot(s["flip_frac"], s["test_pr_auc"], color=COLOR[mode], marker="o", lw=2.2, ls=ls,
                            label=MODE_LABEL[mode])
            axes[r][1].plot(s["flip_frac"], s["test_f1_tuned"], color=COLOR[mode], marker="o", lw=2.2, ls=ls,
                            label=f"{MODE_LABEL[mode]}, tuned threshold")
            axes[r][1].plot(s["flip_frac"], s["test_f1"], color=COLOR[mode], marker=".", lw=1.2, ls=ls,
                            alpha=0.35, label=f"{MODE_LABEL[mode]}, threshold 0.5")
        axes[r][0].set_ylabel(f"{b} banks\nPR-AUC (test, final round)")
        axes[r][1].set_ylabel("F1 (test, final round)")
        for ax in axes[r]:
            ax.set_ylim(0, None)
    for ax in axes[-1]:
        ax.set_xlabel("Fraction of bank 0's training labels flipped")
        ax.set_xticks(sorted(st["flip_frac"].unique()))
    axes[0][0].set_title("Ranking quality")
    axes[0][1].set_title("F1: tuned threshold (bold) vs 0.5 (faint)")
    axes[0][0].legend(loc="lower left")
    axes[0][1].legend(loc="lower left", fontsize=10)

    # threshold effect at the smallest nonzero f, first bank count, fedavg (fall back to the first mode)
    fs = sorted(x for x in st["flip_frac"].unique() if x > 0)
    mode0 = modes[0]
    if fs and (st["flip_frac"] == 0).any():
        s = st[(st["mode"] == mode0) & (st["n_clients"] == banks[0])].set_index("flip_frac")
        if 0.0 in s.index and fs[0] in s.index:
            d05 = s.loc[0.0, "test_f1"] - s.loc[fs[0], "test_f1"]
            dtu = s.loc[0.0, "test_f1_tuned"] - s.loc[fs[0], "test_f1_tuned"]
            dpr = s.loc[0.0, "test_pr_auc"] - s.loc[fs[0], "test_pr_auc"]
            msg = (f"At f={fs[0]:g} ({banks[0]} banks, {MODE_LABEL[mode0]}): F1@0.5 drops {d05:.3f}, tuned F1 "
                   f"drops {dtu:.3f}, PR-AUC drops {dpr:.3f}.")
            if d05 > 0 and dtu < 0.5 * d05:
                msg += (" Most of the F1@0.5 loss is a threshold shift (scores pushed above 0.5), which the "
                        "validation-tuned threshold undoes.")
            fig.text(0.5, -0.02, msg, ha="center", fontsize=10.5, color="#555555", wrap=True)
            f.takeaway = msg
    fmax = st["flip_frac"].max()
    s0 = st[st["flip_frac"] == 0].set_index(["mode", "n_clients"])["test_pr_auc"]
    s1 = st[st["flip_frac"] == fmax].set_index(["mode", "n_clients"])["test_pr_auc"]
    drop = (s0 - s1).dropna()
    if len(drop) and fmax > 0:
        f.takeaway += f" At f={fmax:g} PR-AUC drops by {drop.min():.3f}-{drop.max():.3f}."
        both = [b for b in banks if ("fedavg", b) in drop.index and ("fedavg_he", b) in drop.index]
        if both:
            gap = max(abs(drop[("fedavg", b)] - drop[("fedavg_he", b)]) for b in both)
            f.takeaway += f" FedAvg vs FedAvg+CKKS drops differ by at most {gap:.3f}"
            f.takeaway += (" (encryption neither helps nor hides the damage from the evaluator)." if gap < 0.02
                           else ".")
    fig.tight_layout()
    f.files = save(fig, out, f.name)
    f.takeaway = f.takeaway.strip() or "Only one flip fraction available; no drop can be computed."
    return f


# ----------------------------------------------------------------------------- f. gradient leakage
def gradient_leakage(results, out) -> Figure:
    f = Figure("gradient_leakage")
    df, f.sources = read_attack_csv(results, "gradient_leakage")
    need = {"mode", "batch_size", "local_steps", "recon_corr", "baseline_corr"}
    if df is None:
        f.skipped = "no attacks/gradient_leakage*.csv"
        return f
    if not need <= set(df.columns):
        f.skipped = f"gradient_leakage CSV lacks columns {sorted(need - set(df.columns))}"
        return f
    ctl = df[df["local_steps"] == 1]
    if ctl.empty:
        f.skipped = "no controlled (one local step) gradient-leakage rows"
        return f
    agg = ctl.groupby(["mode", "batch_size"]).agg(r=("recon_corr", "mean"), r_sd=("recon_corr", "std"),
                                                  b=("baseline_corr", "mean"), n=("recon_corr", "size"))
    agg = agg.reset_index()
    base = ctl.groupby("batch_size")["baseline_corr"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5.4))
    series = (("fedavg", "Attack on plaintext update (FedAvg)", COLOR["fedavg"], "-", "o"),
              ("fedavg_he", "Attack on ciphertext (FedAvg + CKKS)", COLOR["fedavg_he"], "--", "s"))
    for mode, lab, colr, ls, mk in series:
        s = agg[agg["mode"] == mode].sort_values("batch_size")
        if s.empty:
            continue
        ax.errorbar(s["batch_size"], s["r"], yerr=s["r_sd"].where(s["n"] > 1).fillna(0), color=colr, ls=ls,
                    marker=mk, ms=8, lw=2.2, capsize=4, label=lab)
    ax.errorbar(base["batch_size"], base["mean"], yerr=base["std"].fillna(0), color=COLOR["baseline"], ls=":",
                marker="^", ms=7, lw=2, capsize=4, label="Baseline: guess population mean")
    ax.set_xscale("log", base=2)
    bs = sorted(ctl["batch_size"].unique())
    ax.set_xticks(bs, [str(int(b)) for b in bs])
    ax.minorticks_off()
    ax.set_xlabel("Client batch size (rows in the one local step)")
    ax.set_ylabel("Pearson r, reconstructed vs true row")
    ax.set_title("Gradient inversion by the server (one local step)")
    ax.legend(loc="lower left", fontsize=11)
    n = int(agg["n"].max())
    real = df[df["local_steps"] > 1]
    foot = f"mean ± std over {n} trial(s) per point"
    if len(real):
        rr = real.groupby("mode")[["recon_corr", "baseline_corr"]].mean()
        foot += "; realistic (full local epochs): " + ", ".join(
            f"{MODE_LABEL.get(m, m)} r={rr.loc[m, 'recon_corr']:.2f} vs baseline {rr.loc[m, 'baseline_corr']:.2f}"
            for m in rr.index)
    fig.text(0.5, -0.02, foot, ha="center", fontsize=10, color="#555555", wrap=True)
    fig.tight_layout()
    f.files = save(fig, out, f.name)
    b1 = agg[agg["batch_size"] == min(bs)].set_index("mode")
    parts = [f"{MODE_LABEL[m]} r={b1.loc[m, 'r']:.3f}" for m in ("fedavg", "fedavg_he") if m in b1.index]
    bb = base.set_index("batch_size").loc[min(bs), "mean"]
    f.takeaway = (f"Batch size {int(min(bs))}: " + ", ".join(parts) + f" vs mean-guess baseline r={bb:.3f}"
                  + (f"; at batch size {int(max(bs))} plaintext r="
                     f"{agg[(agg['mode'] == 'fedavg') & (agg['batch_size'] == max(bs))]['r'].mean():.3f}"
                     f" vs baseline {base.set_index('batch_size').loc[max(bs), 'mean']:.3f}"
                     if max(bs) != min(bs) and (agg["mode"] == "fedavg").any() else "") + ".")
    return f


# ----------------------------------------------------------------------------- driver
TITLES = {"main_comparison": "Main comparison: local vs FedAvg vs FedAvg + CKKS",
          "convergence_b3": "Convergence, 3 banks", "convergence_b5": "Convergence, 5 banks",
          "he_cost": "Cost of encryption", "noniid": "Non-IID ablation (credit card)",
          "label_flip": "Label flipping by one bank", "gradient_leakage": "Gradient leakage to the server"}


def write_index(figs: list[Figure], out: Path, results) -> Path:
    lines = ["# Figures", "", f"Generated by `python -m experiments.plots.make_plots --results {results}` from the "
             "result files listed under each figure. Takeaways are computed from the same data.", ""]
    for fg in figs:
        lines.append(f"## {TITLES.get(fg.name, fg.name)} (`{fg.name}`)")
        lines.append("")
        if fg.skipped:
            lines += [f"Skipped: {fg.skipped}.", ""]
            continue
        lines.append("Files: " + ", ".join(f"[{x}]({x})" for x in fg.files))
        lines.append("")
        src = fg.sources if len(fg.sources) <= 6 else fg.sources[:6] + [f"... ({len(fg.sources)} files)"]
        lines.append("Sources: " + ", ".join(f"`{s}`" for s in src))
        lines.append("")
        lines.append(f"Takeaway: {fg.takeaway}")
        lines.append("")
    path = out / "index.md"
    out.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def make_plots(results="results", out=None) -> list[Figure]:
    """Render every figure whose inputs exist; returns one Figure record per figure (rendered or skipped)."""
    results = Path(results)
    out = Path(out) if out is not None else results / "figures"
    style()
    rows, names = load_runs(results)
    if rows.empty:
        note(f"no per-run CSVs in {results}/; skipping the federated figures")
    builders = [("main_comparison", lambda: main_comparison(rows, names, out)),
                ("convergence_b3", lambda: convergence(rows, names, out, 3)),
                ("convergence_b5", lambda: convergence(rows, names, out, 5)),
                ("he_cost", lambda: he_cost(rows, names, out)),
                ("noniid", lambda: noniid(rows, names, out)),
                ("label_flip", lambda: label_flip(results, out)),
                ("gradient_leakage", lambda: gradient_leakage(results, out))]
    figs = []
    for name, build in builders:
        if rows.empty and name not in ("label_flip", "gradient_leakage"):
            figs.append(Figure(name, skipped=f"no per-run CSVs in {results}"))
        else:
            try:
                figs.append(build())
            except Exception as exc:  # never let one figure stop the others
                traceback.print_exc()
                plt.close("all")
                figs.append(Figure(name, skipped=f"error while plotting ({type(exc).__name__}: {exc})"))
        fg = figs[-1]
        print(f"  {name:18s} " + (f"skipped: {fg.skipped}" if fg.skipped else f"-> {out}/{fg.files[0]}"),
              flush=True)
    print(f"  index -> {write_index(figs, out, results)}")
    return figs


def main(argv=None) -> list[Figure]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None, help="default: <results>/figures")
    a = ap.parse_args(argv)
    return make_plots(a.results, a.out)


if __name__ == "__main__":
    main()
