# common/ (owner: Pratham)

> Placeholder created by the data owner so the folder structure matches the Implementation Architecture Plan. The owner can replace or extend this file.

**Contents:** `interfaces.py`, `config.py`, shared utilities.

**Shared interfaces** (exact signatures go in `common/interfaces.py`):

| function | meaning | provided by |
|---|---|---|
| `load_client_data(bank_id, fraud_type)` | returns X_train, y_train, X_val, y_val (float32 arrays) | Dhriti, `data/loader.py` (done) |
| `load_test(fraud_type)` | returns X_test, y_test | Dhriti, `data/loader.py` (done) |
| `build_model(input_dim)` | returns a torch `nn.Module` | Uttam, Harshini, Likhitha |
| `get_weights`, `set_weights` | weights in and out as a list of numpy arrays | common |
| `encrypt(weights, scale)` | list of bytes (CKKS ciphertext chunks) | Anjan |
| `aggregate(encrypted updates)` | list of bytes (the ciphertext sum) | Anjan |
| `decrypt(list of bytes)` | list of numpy arrays | Anjan |
| `run_experiment(config)` | writes metrics to `results/*.csv`; config: mode (local, fedavg, fedavg_he), fraud_type, n_clients, alpha, rounds, seed | Pratham |
