# experiments/ (owner: Pratham)

Runs every experiment in the project and turns the outputs into tables and figures. See the
[root README](../README.md) for the project overview and setup.

| file | purpose |
|---|---|
| `runner.py` | `run_experiment(cfg)`: one run of one mode (`local`, `fedavg`, `fedavg_he`), written to `results/` as a CSV plus a `.json` copy of the config; finished runs are skipped (resume). Also the `local` baseline. |
| `run_all.py` | `python -m experiments.run_all [--quick] [--dry-run] [--attacks] [--synthetic]`: the full grid (3 fraud types x 3 modes x 3/5 banks at alpha 0.5, plus the credit_card alpha 0.1/5.0 ablation) for each seed. `--dry-run` prints the grid and a runtime estimate. |
| `summarize.py` | `python -m experiments.summarize --results-dir results`: `summary_main.csv`, `summary_cost.csv`, `summary_noniid.csv`. |
| `attacks/` | label flipping and gradient leakage against an honest-but-curious server ([README](attacks/README.md)). |
| `plots/make_plots.py` | `python -m experiments.plots.make_plots --results results --out results/figures`: PNG and PDF figures plus `index.md`; works on partial results. |
| `reproduce.py` | `python -m experiments.reproduce [--quick] [--dry-run] [--results-dir DIR] [--only STAGE ...] [--force-attacks]`: grid, attacks, summaries and plots in order, with per-stage timing. The full grid profile is 15 rounds, seeds 42 43 44 (`run_all`'s own default is 10 rounds, seed 42). |
| `synthetic.py` | generated IEEE-CIS-shaped data for tests and `--synthetic` runs (never for reported results). |

Every run is described by one `common.config.ExperimentConfig`. Federated runs call `fl.run_federated`.
