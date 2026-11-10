# experiments/attacks

Two attack experiments on the credit_card federation (real IEEE-CIS partitions), run in both `fedavg` and
`fedavg_he` through the same `fl` code as the main results, using two optional hooks in `fl.run_federated` /
`fl.simulate` (both default to `None`, which leaves runs unchanged; see `tests/test_attacks.py`):

* `client_label_transform`: `bank_id -> None | (y_train -> y_train')`, applied once to that bank's training
  labels before local training.
* `server_observer`: called once per round with an `fl.strategy.RoundObservation`: the parameters the server
  broadcast and, for every client, exactly the arrays the server received (float32 weights in `fedavg`, 1-D uint8
  CKKS ciphertext chunks in `fedavg_he`), its bank id and sample count.

## Threat model

* Aggregation server: **honest-but-curious**. It runs the protocol correctly but inspects everything it receives
  and knows everything public about the protocol (architecture, learning rate, local step count, the weights it
  broadcast). In `fedavg_he` it holds only the CKKS public/evaluation context, never the secret key.
* Label flipping additionally has **one malicious bank** (bank 0) that flips a fraction of its own training
  labels and otherwise follows the protocol.

## What each script measures

### `label_flip.py`

For n_clients in {3, 5}, each seed, both modes, and flip fraction f in {0, 0.25, 0.5, 1.0}: one full federated
run where bank 0 flips a random f of its training labels.

* Effect: `drop_* = metric(f=0) - metric(f)` (positive = worse) for test F1 at 0.5, F1 at the tuned threshold,
  PR-AUC and ROC-AUC, per round and at the final round. The f=0 run goes through the same hook (identity flip).
* Why impact varies: `attacker_n` and `attacker_share` = n_0 / N from the real partition sizes. FedAvg weights
  bank 0's update by that share (about 0.50 of the data with 3 banks, 0.29 with 5).
* Detection demo (`fedavg` only): from the observer data the server computes each client's update
  (upload - broadcast), the coordinate-wise median update, each client's L2 distance from it, and
  `score = distance / median distance`; a client is flagged when `score > 2.0` (fixed beforehand, not tuned).
  The summary reports how often bank 0 was flagged and the false-flag rate of honest banks. In the committed
  run (seed 42) the screen flagged nobody: bank 0's score stayed at or below 1.70 in every round and flip
  fraction, so it did not detect the attack even in plaintext.
* In `fedavg_he` the same screen cannot be computed: the server only has ciphertext. Those rows have
  `screen_available = False` and no screen values, and the summary has an explicit `limitation` row. No number is
  produced for it.

Outputs: `results/attacks/label_flip.csv` (one row per run and round) and `label_flip_summary.csv` (final round
per run, plus the limitation row).

Note: bank 0's `pos_weight` is the loader's value from its true labels; only `y_train` is changed.

### `gradient_leakage.py`

The server tries to reconstruct bank 0's training rows from bank 0's upload (DLG / inverting-gradients style,
iDLG label inference). Dummy rows and soft labels are optimised so that the update the client would have
produced matches the observed one (per-layer cosine distance, Adam, fixed seed, bounded iterations; small L2
prior; rows clamped to the population feature range). Because the clients train with Adam, a single step is
roughly `lr * sign(gradient)`, so the attack matches a smooth sign surrogate rather than the raw gradient. The
server is assumed to know the client's `pos_weight` (attacker-favourable).

* Controlled setting: each bank trains on B real rows with batch size B for one epoch, i.e. exactly one local
  step, B in {1, 4, 16}. This is where gradient inversion is known to work.
* Realistic setting: bank 0's full partition, `local_epochs` epochs at batch size 512 (hundreds of Adam steps).
  16 rows are reconstructed and each is matched to its closest real row; expected to fail. In the committed
  single trial (816 local steps) the plaintext reconstruction scored r = 0.577 against 0.766 for the mean-row
  baseline, i.e. worse than the baseline.
* Metrics: per-row NMSE (squared error per feature divided by the pooled training variance, floored at 0.01) and
  Pearson r between matched reconstructed and true feature vectors (Hungarian matching), against a baseline
  that guesses the population mean row with the same matching.
* `fedavg_he`: the identical pipeline runs on what the server has: the ciphertext bytes, numerically cast to
  float32 and reshaped to the model's parameter shapes (the only way they type-check as weights). A trusted
  harness also records, in `gradient_leakage_ciphertext_checks.csv`: correlation between ciphertext bytes and the
  true plaintext update/weights, byte entropy, a chi-square test of byte uniformity and the byte agreement between
  two encryptions of the same update. The bytes are a compressed SEAL serialisation with headers, so they are not
  uniform bytes (the chi-square test rejects uniformity); what matters is that they carry no usable correlation
  with the update and that re-encryption gives different bytes (fresh randomness).

What the `fedavg_he` result does **not** show: it is not a proof of security. It only shows that this server,
with this attack, gets nothing usable from ciphertext. Confidentiality comes from CKKS (RLWE, IND-CPA) and from
the server never holding the secret key.

Output: `results/attacks/gradient_leakage.csv` with columns `mode, batch_size, local_steps, recon_nmse,
recon_corr, baseline_nmse, baseline_corr, attack_iters, seed` (one row per trial).

## How to run

```bash
source .venv/bin/activate
python -m experiments.attacks.label_flip --quick        # 3 banks, 2 rounds, 1 epoch -> results/attacks/quick/
python -m experiments.attacks.gradient_leakage --quick  # 1 trial, 200 iterations    -> results/attacks/quick/
python -m experiments.attacks.label_flip                # full: n_clients 3 and 5, 10 rounds, 2 epochs
python -m experiments.attacks.gradient_leakage          # full: 5 trials per batch size, 600 iterations
python -m experiments.run_all --attacks                 # main grid, then both attacks -> results/attacks/
```

The committed `results/attacks/` files come from the two full commands above (their defaults: seed 42, 10
rounds). `python -m experiments.reproduce` runs the attacks with 15 rounds and seeds 42 43 44, but skips each
attack whose output CSV exists unless `--force-attacks` is passed.

Options: `--seeds`, `--modes`, `--results-dir`; label_flip also `--n-clients --fracs --rounds --local-epochs`;
gradient_leakage also `--trials --iters --batch-sizes --local-epochs --no-realistic`.

## Limitations

* Encryption stops the server from inspecting updates, so server-side poisoning defences (distance/median
  screens, robust aggregation, norm clipping) are not available in `fedavg_he`. In the committed run the
  encrypted federation lost about as much as plaintext FedAvg under label flipping (final-round PR-AUC drops
  differ by at most 0.017 between modes, seed 42 only), and the server cannot screen for it.
* All clients share one CKKS secret key (one key pair from trusted setup). A bank, or anyone who obtains a
  bank's key, can decrypt every other bank's upload if it sees them; the protection is only against the server.
  The decrypted global model is also visible to every bank.
* In-process simulation: all parties run in one Python process, and transport is Flower serialisation without a
  network, so there is no channel security, timing or traffic analysis.
* Honest-but-curious server only: a malicious server (dropping, replaying, or substituting ciphertexts, or
  sending crafted models to a target bank) is out of scope. Only one malicious bank and one attack type
  (label flipping) are studied.
* Gradient inversion: one attack family, attacker-favourable assumptions (known pos_weight, learning rate and
  step count; Hungarian/nearest-row matching), and hyperparameters chosen in a small pilot on bank 0. A failure
  in the realistic setting does not prove that no stronger attack exists.
