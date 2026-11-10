# results/ (owner: Pratham)

Outputs of the experiments. Produced by `python -m experiments.reproduce` (see the [root README](../README.md));
figures and summaries can be regenerated at any time from the per-run files.

| path | contents |
|---|---|
| `<fraud_type>_<mode>_b<banks>_a<alpha>_s<seed>.csv` | one run (for example `credit_card_fedavg_he_b3_a0.5_s42.csv`): one row per round with test metrics at 0.5 and at the tuned threshold, validation metrics, timing and byte costs (columns: `fl/metrics.py:ROW_KEYS`). Written by `experiments/runner.py`. |
| `<same name>.json` | the `ExperimentConfig` of that run; a rerun with a different config for the same file name is refused. |
| `summary_main.csv` | final round, mean and std over seeds, per fraud type x mode x bank count (alpha 0.5). The committed grid is 15 rounds, seeds 42, 43 and 44. |
| `summary_cost.csv` | per-round cost of `fedavg_he` vs `fedavg` (encrypt, decrypt, aggregate, bytes, wall time, slowdown, crypto overhead). |
| `summary_noniid.csv` | credit_card alpha ablation (0.1, 0.5, 5.0). |
| `attacks/` | `label_flip.csv`, `label_flip_summary.csv`, `gradient_leakage.csv`, `gradient_leakage_ciphertext_checks.csv` ([what they measure](../experiments/attacks/README.md)). The committed files are seed 42 only (label flipping: 10 rounds; gradient leakage: 5 trials per batch size plus one realistic-setting trial). |
| `figures/` | PNG (200 dpi) and PDF figures from `experiments/plots/make_plots.py`, plus `index.md` with each figure's sources and a computed takeaway. |
| `quick/`, `synthetic/` | the same layout for `--quick` and `--synthetic` runs; not committed. |

Other files (`he_benchmark*`, `he_identity_*`, `credit_card_local_baseline.csv`) are standalone checks from the HE
and model portions, not part of the experiment grid.
