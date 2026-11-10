# Anjan's Contribution: Homomorphic Encryption and Privacy

**Owner:** Anjan Ranga Prasad  
**Assigned portion:** Homomorphic encryption and privacy, as specified in the Implementation Architecture Plan.  
**Updated:** November 3, 2026  
**Status:** Standalone HE functions validated with synthetic weights and one-round real-data checks for 3 and 5 banks. Flower integration and multi-round real-data training remain pending.

All file paths below are relative to the repository root unless stated otherwise.

## Assigned tasks and current status


| Assigned deliverable | Work completed | Status |
| --- | --- | --- |
| TenSEAL context and key setup | Implemented one-time shared CKKS setup with separate exported client secret context and server public context. | Implemented and tested |
| Chunked encryption | Implemented sample-weighted encryption of model parameters in chunks of up to 4,096 values. | Implemented and tested |
| Encrypted aggregation | Implemented server-side addition of corresponding ciphertext chunks without decryption. | Validated on trained identity-model weights for 3 and 5 banks; Flower connection pending |
| Decryption | Implemented client-side decryption, reconstruction of parameter arrays, and loading into the identity model for validation. | Validated for 3 and 5 banks; Flower client connection pending |
| Optional clipping and DP noise | Deferred, consistent with the plan's optional ablation. No `dp.py` implementation or DP guarantee is included. | Optional; not implemented |
| Cost measurements | Recorded encryption, aggregation and decryption times, plus serialized ciphertext sizes, for 3 and 5 banks. | Completed on synthetic weights and real-data-trained identity-model weights |

The required standalone HE functions and their correctness/cost checks are implemented. Both real-data configurations used all 413,378 training rows and 59,054 validation rows, with one local epoch per bank and one aggregation round. Connecting these functions to the Flower training workflow is the next shared step with Jahnavi.

## Implementation corresponding to the plan

### 1. TenSEAL context and key setup

Implemented in `he/keys.py` through `create_contexts()`.

The implementation uses the starting parameters specified in the architecture plan:

| Parameter | Value |
| --- | --- |
| Encryption scheme | CKKS through TenSEAL |
| Polynomial modulus degree | 8,192 |
| Coefficient modulus bit sizes | `[60, 40, 40, 60]` |
| CKKS encoding scale | `2**40` |
| Values per chunk | Up to 4,096 |

One shared key setup is generated per experiment. Clients receive the secret context, and the aggregation server receives only the public context. The server constructor rejects a context containing a secret key. The starting configuration has been validated in the synthetic and one-round real-data checks below; no parameter-tuning study is claimed.

### 2. Chunked encryption

Implemented in `he/ckks.py` through `CKKSClient.encrypt(weights, scale)`.

As required by the plan, bank `i` scales its local model weights by `n_i / N` before encryption, where `n_i` is its training sample count and `N` is the total across all configured banks. The weights are flattened and encrypted in ordered chunks. The returned value is a `list[bytes]` containing serialized CKKS ciphertexts.

`ModelSpec` stores parameter names, order, shapes and dtypes so the decrypted vector can later be reconstructed. The CKKS encoding scale (`2**40`) and the sample-count weight (`n_i / N`) serve different purposes.

### 3. Encrypted aggregation

Implemented in `he/ckks.py` through `CKKSAggregator.aggregate(encrypted_updates)`.

The server adds corresponding ciphertext chunks from the banks. Because clients have already applied the sample-count weights, the server performs no additional weighting or division. The decrypted result approximates:

`global_weights = sum((n_i / N) * local_weights_i)`

The method accepts a bank-ID-to-ciphertext-list mapping and returns the summed ciphertext chunks as `list[bytes]`. All configured banks must be present; missing or unexpected banks cause an error so an incomplete round is not averaged using the original denominator.

### 4. Decryption

Implemented in `he/ckks.py` through `CKKSClient.decrypt(encrypted_chunks)`.

Clients decrypt the aggregated ciphertext and reconstruct a list of NumPy arrays using the shared model specification. Original parameter shapes and `float32`/`float64` dtypes are preserved. The real-data check passes these arrays to the identity model's `set_weights` function and successfully evaluates the restored model. Connecting this same operation to the Flower client remains a shared step with Jahnavi.

## Shared interfaces from the original document

| Planned interface | Implemented interface | Intended user |
| --- | --- | --- |
| `encrypt(weights, scale)` → list of ciphertext bytes | `CKKSClient.encrypt(weights, scale)` | Jahnavi's Flower client |
| `aggregate(encrypted updates)` → list of summed ciphertext bytes | `CKKSAggregator.aggregate(encrypted_updates)` | Jahnavi's server strategy |
| `decrypt(list of bytes)` → list of NumPy arrays | `CKKSClient.decrypt(encrypted_chunks)` | Jahnavi's Flower client |

These are instance methods so the context and model specification can be retained between calls. Their final wiring into `common/interfaces.py` should be coordinated with Pratham and Jahnavi. Additional integration details are in `INTEGRATION.md`.

## Verification against the assigned completion criteria

The architecture plan defines this portion as done when the decrypted aggregate matches the plaintext average within a small tolerance and time and ciphertext size per round are reported.

### Correctness checks

All nine automated tests passed on my Mac. They check encryption/decryption, reconstruction of shapes and dtypes, unequal sample-count weighting, absence of server decryption capability, rejection of secret server contexts and invalid inputs, participation checks, metadata serialization, and chunk boundaries.

The 20-round synthetic benchmark produced:

| Model parameters | Banks | Rounds | Maximum absolute error | Result |
| ---: | ---: | ---: | ---: | --- |
| 10,000 | 3 | 20 | `1.96e-08` | PASS |
| 10,000 | 5 | 20 | `2.46e-08` | PASS |
| 50,000 | 3 | 20 | `2.33e-08` | PASS |
| 50,000 | 5 | 20 | `2.63e-08` | PASS |

All 80 round checks passed against the initial maximum absolute error tolerance of `1e-5`. These errors are taken from the Mac terminal output. The parameter counts, bank counts, and 20-round checks correspond to the ranges proposed in the architecture plan. These are synthetic aggregation rounds, not 20 rounds of fraud-model training.

### Time and ciphertext size

The following are mean per-round measurements from the earlier three-round Mac benchmark, recorded in `results/he_benchmark.csv`:

| Parameters | Banks | Encryption (s) | Aggregation (s) | Decryption (s) | Ciphertext upload (MB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 3 | 0.0225 | 0.0051 | 0.0073 | 2.98 |
| 10,000 | 5 | 0.0360 | 0.0067 | 0.0120 | 4.97 |
| 50,000 | 3 | 0.0961 | 0.0209 | 0.0327 | 12.93 |
| 50,000 | 5 | 0.1583 | 0.0295 | 0.0549 | 21.54 |

Encryption and decryption times are totals across all clients executed sequentially. Upload sizes are total serialized ciphertext bytes across the banks, expressed in decimal MB. Training, network transfer, transport overhead and one-time setup are excluded from these per-round figures. The CSV also records individual client means, aggregate payload size and total broadcast size.

### One-round real-data validation: identity model

Implemented in `he_real_data_check.py`, using the team's `data/loader.py` and `models/identity.py`. The data pipeline contains 590,540 IEEE-CIS transactions, divided into 413,378 training rows, 59,054 validation rows and 118,108 held-out test rows. The test set was not evaluated by these checks.

Both runs used the full training and validation splits (`--max-train-per-bank 0 --max-val-per-bank 0`), `label_skew` partitioning, alpha 0.5, and data/model seed 42. The identity model has 109 input features, 24,449 parameters and 6 ciphertext chunks per bank.

Each run performs the following:

1. Initializes each bank's model with identical starting weights.
2. Trains each bank for one local epoch on its private partition.
3. Extracts the trained weights once and uses those same updates for both comparisons.
4. Computes the plaintext average using actual training sample counts.
5. Encrypts the weighted updates, adds their ciphertexts, and decrypts the result on each client.
6. Checks reconstructed weights against the plaintext reference with maximum absolute error tolerance `1e-5`.
7. Loads the plaintext and decrypted averages into the identity model and evaluates each on the same bank validation rows at threshold 0.5.

| Banks | Training rows used | Validation rows used | Maximum weight error | Maximum probability difference | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 413,378 | 59,054 | `3.025839279e-08` | `1.788139343e-07` | PASS |
| 5 | 413,378 | 59,054 | `2.989892911e-08` | `1.788139343e-07` | PASS |

All reported validation metrics matched exactly between plaintext and encrypted averaging within each run: PR-AUC, ROC-AUC, F1, precision, recall, and recall at 1% false-positive rate. The small nonzero probability differences are consistent with approximate CKKS arithmetic.

The sample-count checks confirm that all bank training rows were used and the averaging weights sum to 1.

| Bank ID | Training rows: 3-bank run | Training rows: 5-bank run |
| ---: | ---: | ---: |
| 0 | 208,462 | 120,421 |
| 1 | 42,389 | 38,607 |
| 2 | 162,527 | 124,196 |
| 3 | — | 64,303 |
| 4 | — | 65,851 |
| **Total** | **413,378** | **413,378** |

The 3-bank and 5-bank configurations are separate partitions. Bank IDs should not be treated as identifying the same partition across configurations.

### Real-data cryptography costs

These are measured costs for the single aggregation round in each real-data run, rather than averages over repeated real-data rounds.

| Banks | Encryption total (s) | Aggregation (s) | Decryption total (s) | Sequential crypto total (s) | Ciphertext upload (MB) | Aggregate payload (MB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.047035 | 0.010665 | 0.015761 | 0.073461 | 5.966 | 1.989 |
| 5 | 0.083055 | 0.014013 | 0.025413 | 0.122481 | 9.940 | 1.987 |

Encryption and decryption totals include all clients executing sequentially. Serialized ciphertext upload is summed across banks. MB means 1,000,000 bytes. The all-client broadcast sizes were 5.966 MB and 9.936 MB, respectively. Training, network transfer, transport envelopes and one-time setup are excluded from these cryptography totals.

Evidence for each run is recorded in its `summary.json`, `bank_costs.csv` and `validation_metrics.csv`. The summaries record Python 3.13.5 on macOS ARM64, 4 PyTorch CPU threads, NumPy 2.2.6, TenSEAL 0.3.18, PyTorch 2.15.0, pandas 3.1.0, PyArrow 26.0.1 and scikit-learn 1.9.1. The team repository source commit was `8c1c56aff0b67cb59ba06fd1607b77e0094c6cdd`; the summaries also record hashes of the loader, identity model and HE modules.

### Interpretation of completion

The original plan's standalone correctness and cost criteria have now been demonstrated on synthetic weights and on real-data-trained identity-model weights for both bank counts. These real-data runs validate **one aggregation round**, not a completed Flower simulation, multi-round convergence, generalization on the held-out test set, or privacy-attack resistance. They do not yet establish equivalence of independently trained plaintext and encrypted FedAvg trajectories. Real-data checks for the other feature-view models have not been performed.

## Files to hand over

| File | Purpose |
| --- | --- |
| `he/keys.py` | Context and key setup |
| `he/ckks.py` | Chunked encryption, encrypted aggregation, decryption and model metadata |
| `he/__init__.py` | Public module imports |
| `tests/test_he.py` | Nine automated tests |
| `benchmark.py` | Synthetic correctness and cost measurements |
| `he_real_data_check.py` | One-round identity-model training, plaintext/CKKS comparison, validation and cost logging |
| `requirements.txt` | `numpy==2.2.6` and `tenseal==0.3.18` |
| `INTEGRATION.md` | Instructions for the Flower and shared-interface owners |
| `results/he_benchmark.csv` | Three-round correctness and cost results |
| `results/he_benchmark_20rounds.csv` | Extended benchmark generated on my Mac |
| `results/he_identity_full_3banks/` | Three-bank real-data `summary.json`, `bank_costs.csv` and `validation_metrics.csv` |
| `results/he_identity_full_5banks/` | Five-bank real-data files with the same names |

The result CSVs should accompany the code handover. They are generated by the commands below; this README itself does not bundle them.

## Running my portion

### Original standalone checks

From the original `he_fraud` package folder, using Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: -r requirements.txt

python -m unittest discover -s tests -v
python benchmark.py
python benchmark.py --rounds 20 --output results/he_benchmark_20rounds.csv
```

No dataset or GPU is needed for these standalone checks. Cryptographic randomness and timings vary between runs.

### Real-data checks in the team repository

Place `he_real_data_check.py` at the repository root and the implemented `ckks.py`, `keys.py` and `__init__.py` under `he/`. The shared data pipeline and identity model must also be present. Install their dependencies in the active environment:

```bash
python -m pip install --only-binary=:all: \
  -r data/requirements.txt \
  "numpy==2.2.6" "tenseal==0.3.18" torch
```

Place the original `train_transaction.csv` and `train_identity.csv` under `data/raw/ieee/`. The loader builds and caches each bank configuration on first use. Run from the repository root:

```bash
python he_real_data_check.py \
  --banks 3 --local-epochs 1 \
  --max-train-per-bank 0 --max-val-per-bank 0 \
  --output-dir results/he_identity_full_3banks

python he_real_data_check.py \
  --banks 5 --local-epochs 1 \
  --max-train-per-bank 0 --max-val-per-bank 0 \
  --output-dir results/he_identity_full_5banks
```

The script preserves prior evidence by rejecting an existing output directory. For a rerun, choose a new `--output-dir` name. Each directory contains:

- `summary.json`: configuration, environment, numerical comparison, pass/fail and overall crypto costs.
- `bank_costs.csv`: training sample counts, averaging weights, training loss and per-bank costs.
- `validation_metrics.csv`: paired plaintext and CKKS validation metrics for each bank.

The two training CSVs, processed row-level data, secret contexts and virtual environments must remain outside the commit. Keep the 3-bank and 5-bank result files in separate folders when sharing aggregate evidence. If `results/` is ignored locally, stage only the intended result files explicitly.

## Coordination with the other assigned portions

| Owner | Connection to my work |
| --- | --- |
| Jahnavi — Federated learning | Connect encryption/decryption to the Flower client, ciphertext aggregation to the server strategy, and broadcast the encrypted result. Loading global weights is shared with my portion. |
| Pratham — Integration, evaluation and attacks | Coordinate shared interfaces and experiment settings, consume HE cost logs, and run the comparison modes and attack experiments. |
| Model and data owners | Supply consistent model parameter layouts and training sample counts for each federation. |

Data preprocessing, fraud-model development, local-only baselines, the full experiment runner, cross-mode evaluation and attack scripts belong to the other portions identified in the plan. The HE module is intended to support each fraud-view experiment once connected to those components.

## Remaining integration work

1. Coordinate the context, model metadata and method signatures with Jahnavi and Pratham.
2. Connect client encryption/decryption and server ciphertext addition to Flower. Apply sample-count weighting exactly once.
3. Add experiment/round/context/model identifiers and duplicate/stale-reply checks at the application boundary, as described in `INTEGRATION.md`.
4. Confirm the full Flower path over two training rounds, then run the planned 10–20 rounds with matched plaintext and encrypted experiment settings.
5. Validate the remaining fraud-view models as they are connected and supply real-training cost logs to the evaluation owner.

Optional clipping and DP noise remain deferred until the required integration works.

## Privacy assumptions

The implementation follows the plan's shared-secret-key design. It assumes trusted clients and an honest-but-curious server without the secret key or collusion with a client. A client possessing the shared key could decrypt another client's ciphertext if it obtained it. Encryption alone does not prevent poisoning, label flipping, or exposure through intentionally reported metrics.

The synthetic benchmark and real-data check place all roles in one process. They verify that the server object operates with a public context, but do not demonstrate process isolation. Optional differential privacy remains unimplemented, as allowed by the original plan.
