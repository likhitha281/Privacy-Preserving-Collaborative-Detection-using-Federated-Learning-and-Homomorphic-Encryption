"""Reproduce every result and figure in order: python -m experiments.reproduce [--quick] [--dry-run]

Stages (elapsed time printed per stage):
  1. grid       experiments.run_all --rounds 15 --seeds 42 43 44 (--quick: run_all's quick profile). Finished runs
                are skipped by run_all's resume logic (one CSV per run).
  2. attacks    experiments.attacks.label_flip (credit_card, 3 and 5 banks, same rounds/seeds) and
                experiments.attacks.gradient_leakage (5 trials, 600 iterations per seed; --quick: 1 trial, 200).
                The attack scripts are not resumable, so each one is skipped when its output CSV already exists
                (pass --force-attacks to rerun them).
  3. summarize  experiments.summarize -> <results>/summary_*.csv
  4. plots      experiments.plots.make_plots -> <results>/figures/ (+ index.md)

Outputs: results/ (full) or results/quick/ (--quick); attacks in <results>/attacks/.
--dry-run lists the stages, their settings and which grid runs / attack outputs already exist, without running
anything. For a timed estimate of the grid itself use: python -m experiments.run_all --rounds 15 --seeds 42 43 44
--dry-run.

Expected runtime on the development machine (WSL2, CPU only), from `run_all --dry-run` probes taken while another
grid was running on the same machine, so treat them as upper-end figures:
  grid      about 10h40m for the 88 runs not yet in results/ (run_all --dry-run, 2026-11-10: probes of
            30-43 s per 3-bank round with 2 local epochs; the per-round time is noisy under load)
  attacks   about 8-9 h: label_flip is 48 federated runs of 15 rounds (~10 min each at the probe speed),
            gradient_leakage ~13 min per seed (30 inversions of 600 iterations + the realistic setting)
  Without a competing job, earlier runs on the same machine took ~7.5 s per credit_card round, i.e.
  roughly 4x faster than these figures.
  summarize + plots: under a minute.
  --quick: about 15 minutes in total (grid ~3 min, label_flip ~7 min, gradient_leakage ~2 min).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

ROUNDS, SEEDS, N_CLIENTS, LOCAL_EPOCHS = 15, (42, 43, 44), (3, 5), 2


@dataclass
class Profile:
    results: Path
    grid_argv: list[str]
    flip: dict
    leak: dict


def profile(quick: bool, results: str | None) -> Profile:
    if quick:
        res = Path(results or "results/quick")
        return Profile(res, ["--quick", "--results-dir", str(res)],
                       dict(n_clients=[3], seeds=[42], rounds=2, local_epochs=1),
                       dict(seeds=[42], trials=1, iters=200, local_epochs=1))
    res = Path(results or "results")
    return Profile(res, ["--rounds", str(ROUNDS), "--seeds", *map(str, SEEDS), "--results-dir", str(res)],
                   dict(n_clients=list(N_CLIENTS), seeds=list(SEEDS), rounds=ROUNDS, local_epochs=LOCAL_EPOCHS),
                   dict(seeds=list(SEEDS), trials=5, iters=600, local_epochs=LOCAL_EPOCHS))


# ----------------------------------------------------------------------------- stages
def stage_grid(p: Profile, args) -> None:
    from . import run_all
    run_all.main(p.grid_argv)


def stage_attacks(p: Profile, args) -> None:
    from .attacks import gradient_leakage, label_flip
    out = p.results / "attacks"
    if (out / "label_flip.csv").exists() and not args.force_attacks:
        print(f"  label_flip: {out / 'label_flip.csv'} exists, skipped (--force-attacks to rerun)")
    else:
        label_flip.run(results_dir=out, **p.flip)
    if (out / "gradient_leakage.csv").exists() and not args.force_attacks:
        print(f"  gradient_leakage: {out / 'gradient_leakage.csv'} exists, skipped (--force-attacks to rerun)")
    else:
        gradient_leakage.run(results_dir=out, **p.leak)


def stage_summarize(p: Profile, args) -> None:
    from . import summarize
    summarize.print_tables(summarize.summarize(p.results))


def stage_plots(p: Profile, args) -> None:
    from .plots import make_plots
    make_plots.make_plots(p.results, p.results / "figures")


STAGES = (("grid", stage_grid), ("attacks", stage_attacks), ("summarize", stage_summarize),
          ("plots", stage_plots))


def describe(p: Profile, args) -> list[str]:
    """One line per stage (used by --dry-run); only looks at which files exist."""
    from .run_all import build_grid, parse_args
    from .runner import result_path
    a = parse_args(p.grid_argv)
    grid = build_grid(a.fraud_types, a.modes, a.n_clients, a.seeds, a.rounds, a.local_epochs, a.results_dir,
                      noniid=not a.no_noniid)
    done = sum(result_path(c).exists() for c in grid)
    out = p.results / "attacks"
    flip_runs = len(p.flip["n_clients"]) * len(p.flip["seeds"]) * 2 * 4

    def state(name):
        return "exists, will skip" if (out / name).exists() and not args.force_attacks else "will run"
    return [
        f"1. grid       python -m experiments.run_all {' '.join(p.grid_argv)}: {len(grid)} runs, {done} done, "
        f"{len(grid) - done} to run",
        f"2. attacks    label_flip {p.flip} ({flip_runs} runs; {state('label_flip.csv')}); "
        f"gradient_leakage {p.leak} ({state('gradient_leakage.csv')}) -> {out}/",
        f"3. summarize  {p.results}/summary_*.csv",
        f"4. plots      {p.results / 'figures'}/ (+ index.md)",
    ]


def _hms(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="quick profile -> results/quick/")
    ap.add_argument("--dry-run", action="store_true", help="list the stages without running them")
    ap.add_argument("--results-dir", default=None, help="default: results (results/quick with --quick)")
    ap.add_argument("--force-attacks", action="store_true", help="rerun attacks even if their CSVs exist")
    ap.add_argument("--only", nargs="+", choices=[n for n, _ in STAGES], help="run only these stages")
    args = ap.parse_args(argv)
    p = profile(args.quick, args.results_dir)
    print(f"Reproduce ({'quick' if args.quick else 'full'} profile) -> {p.results}/")
    for line in describe(p, args):
        print("  " + line)
    if args.dry_run:
        print("Dry run: nothing executed.")
        return
    t0 = time.perf_counter()
    for name, fn in STAGES:
        if args.only and name not in args.only:
            continue
        print(f"\n=== stage {name} ===", flush=True)
        t = time.perf_counter()
        fn(p, args)
        print(f"=== stage {name} done in {_hms(time.perf_counter() - t)} (total {_hms(time.perf_counter() - t0)})",
              flush=True)


if __name__ == "__main__":
    main()
