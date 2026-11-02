# Federated fraud dataset: data card

What the other portions need to know about the data produced by the data-engineering portion. All numbers come
from a real run on the full IEEE-CIS training data with the defaults (5 banks, `label_skew`, alpha 0.5, seed 42).
The same settings and seed always give identical output. Per-bank tables for 3 and 5 banks are in
`data/reports/`.

## 1. Quick start for teammates

```python
from data.loader import load_client_data, load_test, input_dim, get_pos_weight

X_train, y_train, X_val, y_val = load_client_data(bank_id=0, fraud_type="credit_card")  # one bank's private data
X_test, y_test                 = load_test("credit_card")                                # global held-out test set
model = build_model(input_dim("credit_card"))                                            # 95 for credit_card
pos_w = get_pos_weight(0, "credit_card")                                                 # negatives / positives, for BCE
```

- `fraud_type` is `"credit_card"`, `"account_takeover"` or `"identity_fraud"` (see section 6).
- Optional arguments on every function: `n_banks` (3 to 5, default 5), `alpha` (default 0.5), `seed` (default 42),
  `partition` (`"label_skew"` default, `"iid"`, `"quantity_skew"`). These match the experiment config fields.
- The first call builds the data from the raw CSVs (under a minute) and caches it in `data/processed/`. Later calls
  read parquet files in a fraction of a second. **The raw CSVs and the processed data are not in Git:** download
  the Kaggle files into `data/raw/ieee/` (see `data/README.md`).
- Features are float32 and standardised with no NaN or inf. Labels are int8: cast with `.astype("float32")` for
  `BCEWithLogitsLoss`.
- `load_test` returns the same rows regardless of `n_banks`, `alpha` or `seed`, so results are comparable.

## 2. Source

IEEE-CIS Fraud Detection (Kaggle competition, Vesta transactions). Only the two labelled files are used, joined on
`TransactionID`:

| file | rows | notes |
|---|---|---|
| `train_transaction.csv` | 590,540 | target `isFraud`: 20,663 frauds (3.50%) |
| `train_identity.csv` | 144,233 | device / browser / network identity, present for **24.4%** of transactions only |

`TransactionDT` is a time offset in seconds from an unknown reference date, not a calendar timestamp. Most columns
are masked by the provider (`C`, `D`, `M`, `V` and `id_xx` have no public meaning).

## 3. How the data is produced (`python -m data.prepare`)

1. **Load and join** transaction + identity (left join; `has_identity` records whether a row had identity data).
2. **Clean:** drop duplicate `TransactionID`s, sort by time, add hour and day-of-week (sin/cos) from `TransactionDT`.
3. **Behavioural features** (view `beh`), computed per account from *earlier* transactions only (section 6).
4. **Time split (global):** train = first 70% of transactions by `TransactionDT`, validation = next 10%,
   **test = latest 20%.** The same cut applies to every bank, so no future data leaks into training.
5. **Fit the encoder on the pooled train split only**, then encode every row (section 7).
6. **Partition into simulated banks by card** (section 5) and write one folder per bank.

## 4. Splits

| split | rows | frauds | fraud rate | TransactionDT range |
|---|---|---|---|---|
| train | 413,378 | 14,538 | 3.52% | up to day 120.8 |
| val | 59,054 | 2,061 | 3.49% | day 120.8 to 141.1 |
| test | 118,108 | 4,064 | 3.44% | after day 141.1 |

Use **train** to fit, **val** for tuning and thresholds, and **test** only for the final reported numbers.

## 5. The federation (5 simulated banks)

Banks are simulated: one real dataset is split so each bank only holds its own cards.

- **A card never appears in two banks** (partition is by `card1`), so a card's history stays inside one bank.
- **Non-IID on purpose:** cards that ever had a fraud and cards that never did are each assigned to banks with
  Dirichlet(alpha) proportions, so banks differ in fraud rate and size. The assignment is re-drawn (deterministically
  from the seed) until every bank has at least 100 train frauds and 1,000 train rows.
- Smaller `alpha` means more skew. With 3 banks the skew is milder (train fraud rates 2.6% to 4.4%).

| bank | train rows | train frauds | train fraud rate | val rows | test rows | test fraud rate |
|---|---|---|---|---|---|---|
| 0 | 120,421 | 4,921 | 4.09% | 17,367 | 34,450 | 3.80% |
| 1 | 38,607 | 763 | 1.98% | 5,524 | 11,061 | 1.57% |
| 2 | 124,196 | 5,487 | 4.42% | 17,460 | 33,861 | 4.92% |
| 3 | 64,303 | 2,566 | 3.99% | 9,529 | 20,218 | 3.67% |
| 4 | 65,851 | 801 | 1.22% | 9,174 | 18,518 | 0.93% |

The suggested `pos_weight` per bank (negatives / positives in train) is 23.5, 49.6, 21.6, 24.1 and 81.2. Banks 1 and 4
have very few frauds, so local-only models there should do worst, which is what federation should help with.

## 6. Feature views (what each fraud type loads)

IEEE-CIS has **one** `isFraud` label and no fraud-type labels. Each fraud type is therefore a *feature view* used as
a proxy, as the architecture plan says.

| `fraud_type` | view | input dim | owner | content |
|---|---|---|---|---|
| `credit_card` | `txn` | 95 | Uttam | `TransactionAmt`, `dist1/2`, Vesta `C1-C14` (counts), `D1-D15` (time deltas), hour/day sin-cos, one-hot `ProductCD`, `card4`, `card6`, `M1-M9` |
| `account_takeover` | `beh` | 10 | Harshini | per-card history, below |
| `identity_fraud` | `idn` | 109 | Likhitha | `id_01-id_11`, `id_12-id_38`, `DeviceType`, `DeviceInfo`, `P_/R_emaildomain`, `has_identity`, `email_mismatch` |

**`beh` features.** An *account* is the full tuple `card1..card6`. Every value at row *t* uses only that account's rows
before *t*, so there is no look-ahead (a test perturbs future rows and checks nothing earlier changes).

| column | meaning |
|---|---|
| `beh__beh_prior_count` | earlier transactions on this account |
| `beh__beh_secs_since_prev` | seconds since the account's previous transaction (0 for the first) |
| `beh__beh_amt_logratio` | log of (amount / average of the account's earlier amounts); 0 for the first |
| `beh__beh_cnt_1h`, `_24h`, `_7d` | earlier transactions on this account in the last 1 hour / 24 hours / 7 days |
| `beh__beh_is_first` | 1 if this is the account's first transaction |
| `beh__beh_new_addr`, `_new_email`, `_new_device` | 1 if this `addr1` / `P_emaildomain` / `DeviceInfo` was never seen on the account before (needs at least one earlier observed value) |

**Account-takeover and identity labels do not exist** in this dataset. Call `beh` "ATO-style behavioural signals" and
`idn` "identity and device signals" in the report.

**Not included on purpose:** `TransactionID`, `card1-card3`, `card5` and `addr1/2` as raw ids (they identify entities
and would let a model memorise cards), and Vesta's `V1-V339` (add with `--include-v` if wanted). Columns named
`meta__*` identify rows and cards and must **never** be used as model input. `load_*` functions never return them.

Column names inside a view look like `txn__card4=visa`; the full lists are in `schema.json` of each processed folder.

## 7. Encoding (the same for every bank)

- Numeric: heavy-tailed columns (`TransactionAmt`, `dist*`, `C*`, `D*`, behavioural counts and seconds) get signed
  `log1p`, then median imputation, then standardisation with the **train** mean and std, clipped to ±10.
- Missing values: a `<col>__missing` indicator is added to numeric columns that are 5% to 95% missing in train
  (25 columns). Columns that are almost always missing (`id_07`, `id_08`, 99%) get none.
- Categorical: one-hot of the 12 most frequent train values, plus `=<other>` and `=<missing>` columns where needed.
- 0/1 flags (`beh_is_first`, `has_identity`, ...) are passed through unchanged.

## 8. Reference numbers (quick sanity check, not a project result)

Plain logistic regression with balanced class weights on the pooled train data, scored on the global test set
(test fraud rate 3.44%, so a random model has PR-AUC of about 0.034):

| features | PR-AUC | ROC-AUC |
|---|---|---|
| `credit_card` view (`txn`) | 0.397 | 0.857 |
| `identity_fraud` view (`idn`) | 0.153 | 0.716 |
| `account_takeover` view (`beh`) | 0.062 | 0.623 |
| `txn + beh` | 0.400 | 0.861 |
| all three views | 0.409 | 0.864 |

**Heads-up for the account-takeover owner:** the `beh` view alone has only 10 features and a linear model barely
beats chance on it (PR-AUC 0.062). A neural network may do somewhat better, but expect this view to be the weakest.
If the team wants a stronger account-takeover model, the simplest change is to give that view extra columns
(for example `txn` amount and hour features): edit the one `VIEWS` dictionary at the top of `data/loader.py`.
Discuss with Pratham before changing it, because the architecture plan defines the views.

Report **PR-AUC, recall at a fixed false-positive rate and F1**. Accuracy is meaningless at a 3.5% fraud rate.

## 9. Known limitations (state these in the report)

- Banks are **simulated** from one dataset, and `card1` is only a proxy for a customer or institution.
- The encoder statistics come from the **pooled** train split. A real federation could not compute them centrally;
  this is a simulation shortcut.
- The partition uses the fraud labels of all splits to decide which banks receive fraud-heavy cards.
- The split is by time, not by card: test frauds can come from cards that also appear in a bank's train data.
- Only 24.4% of rows have identity data, so `idn` is mostly "missing" for the rest.
- `beh` features depend on the account key `card1..card6`, which may merge or split real accounts.
- The fraud types are feature-view proxies of one `isFraud` label, not separately labelled tasks.
