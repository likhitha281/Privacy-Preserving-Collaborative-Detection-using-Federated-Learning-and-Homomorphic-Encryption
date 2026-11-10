# fl/ (owner: Jahnavi)

Federated training with Flower for the modes `fedavg` (plaintext FedAvg) and `fedavg_he` (CKKS-encrypted FedAvg),
on the real bank partitions from `data/`. See the [root README](../README.md) for the round protocol and setup.

| file | purpose |
|---|---|
| `server.py` | `run_federated(cfg, csv_path=None)`: runs `cfg.rounds` rounds and returns one metrics row per round. `simulate(cfg)` also returns the global weights and the strategy. Flower's strategy, client manager and `NumPyClient` serialisation are driven in one process (no Ray). In `fedavg_he` the harness does the trusted key setup and acts as the evaluator that decrypts the global model and scores the test set. Optional hooks `client_label_transform` and `server_observer` are used only by `experiments/attacks`. |
| `client.py` | `FraudClient`: one bank. Trains locally (Adam, BCE with the bank's `pos_weight`); in `fedavg_he` it scales its weights by n_i/N, encrypts them and decrypts the aggregate it receives. |
| `strategy.py` | `FraudFedAvg` (sample-weighted FedAvg) and `EncryptedFedAvg` (holds only the public CKKS context and adds ciphertexts). Both require every bank to reply each round and abort otherwise. |
| `metrics.py` | metrics at threshold 0.5 and at a validation-tuned threshold, and the per-round row schema `ROW_KEYS` shared by all modes. |
| `logging.py` | CSV writing and reading for metric rows. |

Tests: `tests/test_fl.py` and `tests/test_attacks.py` (synthetic data).
