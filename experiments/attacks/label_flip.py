"""Label flipping by one malicious bank, and what the aggregation server can (fedavg) or cannot (fedavg_he) see.

    python -m experiments.attacks.label_flip [--quick] [--seeds 42 ...] [--n-clients 3 5] [--modes ...]

Threat model: bank ATTACKER (0) flips a fraction f of its own training labels (a random subset, fixed seed) and
otherwise runs the honest client code; the server is honest-but-curious. For credit_card, each n_clients and seed,
both modes are run for every f in FRACS through fl.run_federated with the two hooks:
  * client_label_transform: bank 0's y_train -> flip_labels(y_train, f) (validation/test labels untouched;
    pos_weight still comes from the loader, i.e. from the bank's true labels);
  * server_observer: records exactly what the server receives each round.

Effect: drop = metric(f=0) - metric(f) for test F1@0.5, F1 at the tuned threshold, PR-AUC and ROC-AUC, per round
and at the final round (positive = the attack hurt). The attacker's sample share n_0/N (real partition sizes) is
reported because FedAvg weights its update by that share.

Detection demo (server side, fedavg only): from the observed uploads, delta_i = upload_i - broadcast, take the
coordinate-wise median delta M, d_i = ||delta_i - M||_2 and score_i = d_i / median_j d_j; a bank is flagged if
score_i > SCREEN_RATIO (fixed in advance, not tuned on these results). In fedavg_he the server only receives
CKKS ciphertext (uint8 chunks), so this screen cannot be computed: those rows say so and carry no screen values.

Outputs (in --results-dir, default results/attacks, or results/attacks/quick with --quick):
  label_flip.csv          one row per run and round
  label_flip_summary.csv  one row per run (final round) + one documented limitation row for fedavg_he
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Iterable

import numpy as np
import pandas as pd

from common.config import ExperimentConfig
from fl import run_federated
from fl.logging import write_rows_csv

from .common import Recorder, flat, is_plaintext, out_dir

FRAUD_TYPE = "credit_card"
FRACS = (0.0, 0.25, 0.5, 1.0)
ATTACKER = 0
SCREEN_RATIO = 2.0
METRICS = {"test_f1": "drop_f1", "test_f1_tuned": "drop_f1_tuned", "test_pr_auc": "drop_pr_auc",
           "test_roc_auc": "drop_roc_auc"}
HE_NOTE = ("not computable: in fedavg_he the server receives only CKKS ciphertext (uint8 chunks) and holds no "
           "secret key, so it cannot compare client updates; encryption hides poisoning from the server")


# ----------------------------------------------------------------------------- attack
def flip_labels(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Copy of binary labels y with round(frac * len(y)) randomly chosen entries flipped (0 <-> 1)."""
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"frac must be in [0, 1], got {frac}")
    y = np.array(y, copy=True)
    idx = np.random.default_rng(seed).choice(len(y), int(round(frac * len(y))), replace=False)
    y[idx] = 1 - y[idx]
    return y


def label_transform(frac: float, seed: int, attacker: int = ATTACKER):
    """client_label_transform hook: the attacker's labels are flipped (also at f=0, where it is the identity)."""
    return lambda bank: (lambda y: flip_labels(y, frac, seed)) if bank == attacker else None


# ----------------------------------------------------------------------------- server-side screen
def distance_screen(obs, ratio: float = SCREEN_RATIO) -> dict | None:
    """{bank_id: (distance, score, flagged)} from one RoundObservation, or None if the server only has ciphertext.

    Uses only what the server holds: the parameters it broadcast and the arrays it received.
    """
    if not is_plaintext(obs.broadcast) or not all(is_plaintext(u.arrays) for u in obs.uploads):
        return None
    base = flat(obs.broadcast)
    deltas = np.stack([flat(u.arrays) - base for u in obs.uploads])
    d = np.linalg.norm(deltas - np.median(deltas, axis=0), axis=1)
    score = d / max(float(np.median(d)), 1e-12)
    return {u.bank_id: (float(d[i]), float(score[i]), bool(score[i] > ratio)) for i, u in enumerate(obs.uploads)}


# ----------------------------------------------------------------------------- runs
def run_one(cfg: ExperimentConfig, frac: float, attacker: int = ATTACKER) -> list[dict]:
    """One federated run with the attacker flipping ``frac`` of its labels; one row per round (no drop columns)."""
    rec = Recorder()
    rows = run_federated(cfg, client_label_transform=label_transform(frac, cfg.seed, attacker),
                         server_observer=rec)
    counts = {u.bank_id: u.num_examples for u in rec.rounds[0].uploads}
    out = []
    for row, obs in zip(rows, rec.rounds):
        screen = distance_screen(obs)
        r = {"mode": cfg.mode, "fraud_type": cfg.fraud_type, "n_clients": cfg.n_clients, "alpha": cfg.alpha,
             "seed": cfg.seed, "flip_frac": frac, "attacker_bank": attacker, "attacker_n": counts[attacker],
             "attacker_share": counts[attacker] / sum(counts.values()), "round": row["round"],
             **{k: row[k] for k in METRICS}, **{v: None for v in METRICS.values()},
             "upload_dtype": str(np.asarray(obs.uploads[0].arrays[0]).dtype),
             "screen_available": screen is not None}
        if screen is None:
            r.update(attacker_dist=None, attacker_score=None, attacker_flagged=None, honest_flagged=None,
                     honest_max_score=None, screen_note=HE_NOTE)
        else:
            honest = [v for b, v in screen.items() if b != attacker]
            r.update(attacker_dist=screen[attacker][0], attacker_score=screen[attacker][1],
                     attacker_flagged=screen[attacker][2], honest_flagged=sum(v[2] for v in honest),
                     honest_max_score=max(v[1] for v in honest), screen_note=f"score > {SCREEN_RATIO}")
        out.append(r)
    return out


def add_drops(rows: list[dict]) -> list[dict]:
    """Fill drop_* = clean (f=0, same mode/n_clients/seed/round) minus attacked; positive = worse."""
    clean = {(r["mode"], r["n_clients"], r["seed"], r["round"]): r for r in rows if r["flip_frac"] == 0.0}
    for r in rows:
        c = clean.get((r["mode"], r["n_clients"], r["seed"], r["round"]))
        if c is not None:
            for m, d in METRICS.items():
                r[d] = c[m] - r[m]
    return rows


def summarize(rows: list[dict]) -> pd.DataFrame:
    """Final-round effect per run, the screen's flag rates over rounds, and the fedavg_he limitation row."""
    df = pd.DataFrame(rows)
    # rows may come back from label_flip.csv (fl.logging.read_rows_csv), where booleans are strings
    for c in ("screen_available", "attacker_flagged"):
        df[c] = df[c].map({True: True, False: False, "True": True, "False": False})
    keys = ["mode", "n_clients", "seed", "flip_frac"]
    out = []
    for _, g in df.groupby(keys, sort=True):
        last = g.loc[g["round"].idxmax()]
        avail = bool(g["screen_available"].all())
        n_honest = int(last["n_clients"]) - 1
        out.append({"row_type": "result", **{k: last[k] for k in keys},
                    "attacker_n": last["attacker_n"], "attacker_share": last["attacker_share"],
                    "final_round": last["round"], **{m: last[m] for m in METRICS},
                    **{d: last[d] for d in METRICS.values()},
                    "mean_drop_f1_over_rounds": g["drop_f1"].mean(),
                    "screen_available": avail,
                    "attacker_flag_rate": g["attacker_flagged"].astype(float).mean() if avail else None,
                    "honest_flag_rate": g["honest_flagged"].sum() / (n_honest * len(g)) if avail else None,
                    "attacker_score_mean": g["attacker_score"].mean() if avail else None,
                    "note": last["screen_note"]})
    if (df["mode"] == "fedavg_he").any():
        out.append({"row_type": "limitation", "mode": "fedavg_he", "screen_available": False, "note": HE_NOTE})
    # nullable ints: the limitation row has no run keys, which would otherwise turn these columns into floats
    return pd.DataFrame(out).astype({"n_clients": "Int64", "seed": "Int64", "attacker_n": "Int64",
                                     "final_round": "Int64"})


def run(n_clients: Iterable[int] = (3, 5), seeds: Iterable[int] = (42,), modes=("fedavg", "fedavg_he"),
        fracs=FRACS, rounds: int = 10, local_epochs: int = 2, results_dir=None, quick: bool = False,
        alpha: float = 0.5):
    """Run the grid, write label_flip.csv and label_flip_summary.csv, return (rows, summary DataFrame)."""
    if 0.0 not in fracs:
        raise ValueError("fracs must include 0.0 (the clean reference for the drops)")
    d = out_dir(results_dir, quick)
    rows, t0 = [], time.perf_counter()
    for seed in seeds:
        for n in n_clients:
            for mode in modes:
                cfg = ExperimentConfig(mode=mode, fraud_type=FRAUD_TYPE, n_clients=n, alpha=alpha, rounds=rounds,
                                       local_epochs=local_epochs, seed=seed)
                for f in sorted(fracs):
                    t = time.perf_counter()
                    rows += run_one(cfg, f)
                    r = rows[-1]
                    print(f"  label_flip {mode:9s} n={n} seed={seed} f={f:<4} final F1 {r['test_f1']:.3f} "
                          f"PR-AUC {r['test_pr_auc']:.3f} attacker_share {r['attacker_share']:.3f} "
                          f"flagged={r['attacker_flagged']}  {time.perf_counter() - t:.0f}s "
                          f"(elapsed {time.perf_counter() - t0:.0f}s)", flush=True)
    add_drops(rows)
    summary = summarize(rows)
    write_rows_csv(rows, d / "label_flip.csv", overwrite=True)
    d.mkdir(parents=True, exist_ok=True)
    summary.to_csv(d / "label_flip_summary.csv", index=False)
    return rows, summary


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Label-flipping attack by bank 0 (see module docstring).")
    ap.add_argument("--quick", action="store_true", help="3 banks, 2 rounds, 1 local epoch -> results/attacks/quick")
    ap.add_argument("--n-clients", nargs="+", type=int, default=[3, 5])
    ap.add_argument("--seeds", nargs="+", type=int, default=[ExperimentConfig.seed])
    ap.add_argument("--modes", nargs="+", choices=["fedavg", "fedavg_he"], default=["fedavg", "fedavg_he"])
    ap.add_argument("--fracs", nargs="+", type=float, default=list(FRACS))
    ap.add_argument("--rounds", type=int, default=ExperimentConfig.rounds)
    ap.add_argument("--local-epochs", type=int, default=ExperimentConfig.local_epochs)
    ap.add_argument("--results-dir", default=None)
    a = ap.parse_args(argv)
    if a.quick:
        a.n_clients, a.rounds, a.local_epochs = [3], 2, 1
    return a


def main(argv=None) -> pd.DataFrame:
    a = parse_args(argv)
    _, summary = run(a.n_clients, a.seeds, a.modes, a.fracs, a.rounds, a.local_epochs, a.results_dir, a.quick)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.drop(columns=["note"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("fedavg_he screen:", HE_NOTE)
    return summary


if __name__ == "__main__":
    main()
