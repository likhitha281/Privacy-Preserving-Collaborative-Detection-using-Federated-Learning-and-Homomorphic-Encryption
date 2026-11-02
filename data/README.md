# `data/`: data engineering + federation simulation

Turns the IEEE-CIS Fraud Detection data into a simulated federation of 3 to 5 non-IID banks and serves it to the
other portions. Preprocessing only: no models, federated training or encryption live here.

**Read [DATA.md](DATA.md) first** (what the data is, how it is split, what the features mean).

## Use it from other code

```python
from data.loader import load_client_data, load_test
X_train, y_train, X_val, y_val = load_client_data(bank_id=0, fraud_type="credit_card")
X_test, y_test = load_test("credit_card")
```

Run Python from the repository root so `data` is importable.

## One-time setup

1. Download `train_transaction.csv` and `train_identity.csv` from the Kaggle competition
   [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data) (join the competition and
   accept its rules first) and put them in `data/raw/ieee/`. Never commit them.
2. `pip install -r data/requirements.txt`
3. Optional: build explicitly instead of on first use:
   ```
   python -m data.prepare --dataset ieee --data-dir data/raw/ieee --out-dir data/processed/ieee_label_skew_b5_a0.5_s42 --n-banks 5 --alpha 0.5 --seed 42
   ```
   Other options: `--partition iid|quantity_skew|label_skew`, `--n-banks 3`, `--include-v`, `--val-frac`, `--test-frac`.
   The folder name must follow `ieee_<partition>_b<n_banks>_a<alpha>_s<seed>` for `load_client_data` to find it.

Everyone who uses the same settings and seed gets identical banks, so no processed data has to be shared.

## Check your build

For the defaults (5 banks, `label_skew`, alpha 0.5, seed 42) `manifest.json` must show 590,540 rows, 20,663 frauds,
split rows 413,378 / 59,054 / 118,108 and 95 / 10 / 109 features (txn / beh / idn). Compare with
`reports/split_report_b5_alpha0.5_seed42.md`.

## Files

| file | purpose |
|---|---|
| `loader.py` | functions the other portions call (`load_client_data`, `load_test`, `input_dim`, `get_pos_weight`) |
| `prepare.py` | builds the federation from the raw CSVs |
| `features.py` | cleaning, behavioural features, encoder |
| `partition.py` | assigns cards to banks (iid, label skew, quantity skew) |
| `synthetic.py` | fake IEEE-shaped data for tests only |
| `tests/` | 18 tests: `python -m pytest data/tests -q` (synthetic data, about 15 seconds) |
| `reports/` | split reports with class balance per bank (3 and 5 banks) |
| `DATA.md` | data card for teammates |

## Output of one build (`data/processed/<name>/`, not committed)

`bank_i/{train,val,test}.parquet`, `schema.json`, `encoder.json`, `manifest.json`, `partition_summary.csv`, `split_report.md`.
