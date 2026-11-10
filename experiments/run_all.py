"""Run the full experiment grid and summarise it: python -m experiments.run_all

Grid (defaults): fraud_type {credit_card, account_takeover, identity_fraud} x mode {local, fedavg, fedavg_he}
x n_clients {3, 5} at alpha 0.5, plus the non-IID ablation alpha {0.1, 5.0} on credit_card only, for each seed
(--seeds, default 42). Finished runs are skipped (one CSV per run), so an interrupted grid resumes.

    --quick     3 banks, 2 rounds, 1 local epoch, one fraud type, no ablation; writes to results/quick/
    --dry-run   print the grid and an estimated runtime from timed 1-round probes; writes nothing
    --synthetic use generated IEEE-CIS-shaped data instead of data/raw/ieee (pipeline checks only)
    --attacks   also run experiments/attacks (label flipping, gradient leakage) -> <results-dir>/attacks/;
                off by default. Attack runs are not resumable and are rerun every time.
"""
from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from common.config import FRAUD_TYPES, MODES, ExperimentConfig
from data import loader
from fl import run_federated

from . import summarize
from .runner import result_path, run_experiment, run_local

NONIID_ALPHAS = (0.1, 5.0)
MAIN_ALPHA = 0.5
FEDERATION_BUILD_S = 60.0  # data/loader.py: "builds the federation from the raw CSVs (under a minute)"


def build_grid(fraud_types, modes, n_clients, seeds, rounds, local_epochs, results_dir, noniid=True):
    def cfg(ft, mode, n, alpha, seed):
        return ExperimentConfig(mode=mode, fraud_type=ft, n_clients=n, alpha=alpha, rounds=rounds,
                                local_epochs=local_epochs, seed=seed, results_dir=results_dir)

    grid = [cfg(ft, m, n, MAIN_ALPHA, s) for s in seeds for ft in fraud_types for n in n_clients for m in modes]
    if noniid and "credit_card" in fraud_types:
        grid += [cfg("credit_card", m, n, a, s) for s in seeds for a in NONIID_ALPHAS for n in n_clients
                 for m in modes]
    return grid


def _hms(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def estimate(grid, force: bool) -> float:
    """Time one 1-round run per mode (credit_card, smallest n_clients, first seed) and extrapolate.

    Per run: (probe_total - probe_round) + rounds * probe_round, where probe_round is round_wall_s (training,
    crypto, aggregation, validation) scaled by local_epochs; the HE-specific part is scaled by n_clients / probe
    n_clients because clients encrypt sequentially. Test scoring per round is counted once (small next to
    training). Missing federations add FEDERATION_BUILD_S each. Other fraud types are assumed to cost the same
    (identical rows, different feature counts).
    """
    todo = [c for c in grid if force or not result_path(c).exists()]
    if not todo:
        return 0.0
    base = min(todo, key=lambda c: (c.fraud_type != "credit_card", c.n_clients, c.seed))
    probe = {}
    for mode in sorted({c.mode for c in todo}):
        c = replace(base, mode=mode, rounds=1)
        t = time.perf_counter()
        row = (run_local(c) if mode == "local" else run_federated(c))[0]
        total = time.perf_counter() - t
        crypto = (row["encrypt_s"] or 0) + (row["decrypt_s"] or 0) + (row["aggregate_s"] or 0)
        probe[mode] = (total - row["round_wall_s"], row["round_wall_s"] - crypto, crypto)
        print(f"  probe {mode:10s} 1 round, {base.n_clients} banks: {total:6.1f}s "
              f"(per-round {row['round_wall_s']:.1f}s, crypto {crypto:.2f}s)")
    est = 0.0
    for c in todo:
        setup, train, crypto = probe[c.mode]
        per_round = train * c.local_epochs / base.local_epochs + crypto * c.n_clients / base.n_clients
        est += setup + c.rounds * per_round
    feds = {tuple(sorted(c.loader_kwargs().items())) for c in todo}
    missing = [f for f in feds if not (loader.federation_dir(**dict(f)) / "manifest.json").exists()]
    return est + FEDERATION_BUILD_S * len(missing)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Run the experiment grid (see module docstring).")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--fraud-types", nargs="+", choices=FRAUD_TYPES, default=list(FRAUD_TYPES))
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--n-clients", nargs="+", type=int, default=[3, 5])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--no-noniid", action="store_true", help="skip the credit_card alpha ablation")
    ap.add_argument("--results-dir", default=None, help="default: results (results/quick with --quick)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun runs whose CSV already exists")
    ap.add_argument("--attacks", action="store_true", help="also run the attack experiments (off by default)")
    a = ap.parse_args(argv)
    if a.quick:
        a.n_clients, a.rounds, a.local_epochs, a.fraud_types, a.no_noniid = [3], 2, 1, a.fraud_types[:1], True
    a.results_dir = a.results_dir or ("results/quick" if a.quick else "results")
    if a.synthetic and a.results_dir.rstrip("/") == "results":
        a.results_dir = "results/synthetic"  # never mix synthetic runs into the reported results
    return a


def attack_configs(a) -> list[ExperimentConfig]:
    """Federations the attack scripts load (credit_card, alpha 0.5; gradient leakage always uses 3 banks)."""
    from .attacks import gradient_leakage
    ns = sorted(set(a.n_clients) | {gradient_leakage.N_CLIENTS})
    return [ExperimentConfig(mode="fedavg", fraud_type="credit_card", n_clients=n, alpha=MAIN_ALPHA, seed=s)
            for s in a.seeds for n in ns]


def run_attacks(a) -> None:
    from .attacks import gradient_leakage, label_flip
    out = Path(a.results_dir) / "attacks"
    print(f"\n== Attacks -> {out}/ ==")
    label_flip.run(n_clients=a.n_clients, seeds=a.seeds, rounds=a.rounds, local_epochs=a.local_epochs,
                   results_dir=out)
    trials, iters = (1, 200) if a.quick else (5, 600)
    rows, _ = gradient_leakage.run(seeds=a.seeds, trials=trials, iters=iters, local_epochs=a.local_epochs,
                                   results_dir=out)
    print(gradient_leakage.summary_table(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main(argv=None) -> None:
    a = parse_args(argv)
    grid = build_grid(a.fraud_types, a.modes, a.n_clients, a.seeds, a.rounds, a.local_epochs, a.results_dir,
                      noniid=not a.no_noniid)
    tmp = None
    if a.synthetic:
        from .synthetic import use_synthetic
        tmp = tempfile.TemporaryDirectory()
        use_synthetic(tmp.name, grid + (attack_configs(a) if a.attacks else []))
    print(f"{len(grid)} runs -> {a.results_dir}/  ({'synthetic' if a.synthetic else 'IEEE-CIS'} data)")
    for i, c in enumerate(grid, 1):
        done = "done" if result_path(c).exists() and not a.force else "todo"
        print(f"  [{i:3d}] {result_path(c).name:55s} rounds={c.rounds} epochs={c.local_epochs}  {done}")
    if a.attacks:
        print(f"  + attacks: label_flip (credit_card, n_clients {a.n_clients}, 2 modes x 4 flip fractions, "
              f"{a.rounds} rounds) and gradient_leakage -> {a.results_dir}/attacks/ (not in the estimate below)")
    if a.dry_run:
        print("Probing...")
        print(f"Estimated runtime for remaining runs: {_hms(estimate(grid, a.force))}")
        return
    t0 = time.perf_counter()
    for i, c in enumerate(grid, 1):
        skip = result_path(c).exists() and not a.force
        t = time.perf_counter()
        rows = run_experiment(c, force=a.force)
        print(f"[{i}/{len(grid)}] {result_path(c).name}: {'skipped (exists)' if skip else 'done'} "
              f"in {time.perf_counter() - t:.1f}s, final F1@0.5 {rows[-1]['test_f1']:.3f}, "
              f"PR-AUC {rows[-1]['test_pr_auc']:.3f}  | elapsed {_hms(time.perf_counter() - t0)}", flush=True)
    summarize.print_tables(summarize.summarize(a.results_dir))
    if a.attacks:
        run_attacks(a)
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
