# Connecting the HE Module to Federated Learning

**HE owner:** Anjan Ranga Prasad  
**Flower owner:** Jahnavi  
**Shared interfaces and evaluation:** Pratham

Place this file at the repository root. It describes how to connect the implemented `he/` module to the team's Flower workflow. It is an integration contract, not a claim that Flower integration has already been completed.

## Current status

The HE module implements key/context setup, chunked encryption, encrypted weighted aggregation, and decryption with model-shape reconstruction. Nine automated tests and the 20-round synthetic benchmarks passed. One-round checks using real-data-trained identity-model weights passed for both 3 and 5 banks; the reported validation metrics matched the plaintext average in each check.

The next step is to connect these functions to the Flower client and server strategy and validate repeated training rounds. Differential privacy remains optional and unimplemented.

## 1. Set up one shared context and model specification

The trusted setup coordinator calls `create_contexts()` once per experiment:

```python
from he import create_contexts, ModelSpec, CKKSClient, CKKSAggregator

keys = create_contexts()
spec = ModelSpec.from_weights(initial_weights, names=parameter_names)
```

`initial_weights` is the shared starting model expressed as an ordered list of NumPy arrays. `parameter_names` must follow the same order. For the identity model, use its `get_weights()` method and the corresponding `named_parameters()` names.

Distribute the following through the appropriate setup paths:

| Recipient | Setup material |
| --- | --- |
| Each bank client | `keys.secret_context`, shared model specification, shared initial weights, and agreed sample counts |
| Aggregation server | `keys.public_context`, shared model specification, and agreed sample counts |

The server must never receive the `keys` object or `keys.secret_context`. Its aggregator constructor rejects a secret-bearing context. For a demonstration with isolated roles, run trusted setup outside the server process.

All banks in an experiment must use the **same actual key setup**, not independently generated keys with matching CKKS parameters. Retain client contexts across rounds. Treat context bytes as serialized binary data, not strings or NumPy arrays.

Use `spec.to_dict()` and `ModelSpec.from_dict(...)` to exchange the public model metadata. Compare `spec.fingerprint` across participants. The specification preserves shapes and dtypes, but cannot identify semantically swapped equal-shaped parameters or inconsistent feature columns; stable parameter ordering and shared preprocessing are required.

## 2. Agree on sample counts and model layout

`sample_counts` maps string bank IDs to positive integer training sample counts:

If `bank_data` maps each bank ID to the loader's `(X_train, y_train, X_val, y_val)` tuple, construct the counts as follows:

```python
sample_counts = {
    str(bank_id): len(parts[1])
    for bank_id, parts in bank_data.items()
}
```

Count examples, not batches or validation records. If a check trains on subsets, construct this mapping after selecting the subsets so it counts the examples actually used.

The HE implementation accepts float32 and float64 arrays. Handle integer buffers separately rather than casting and averaging them. The current identity model has no BatchNorm buffers and exposes `get_weights()` and `set_weights()` directly.

Each fraud-view federation must use one common model architecture and feature ordering. Run separate federations for different feature-view models; do not average their parameters together.

## 3. Client hooks: encrypt after training, decrypt before the next round

Construct the client-side HE object once in each client worker:

```python
client_he = CKKSClient(secret_context_bytes, spec)
```

At round 0, load the shared initial model weights. At subsequent rounds, decrypt and load the previous global model before local training:

```python
global_weights = client_he.decrypt(global_chunks)
model.set_weights(global_weights)  # Identity-model interface.
```

Then train for the configured local epochs, extract the parameters, and encrypt:

```python
local_weights = model.get_weights()
N = sum(sample_counts.values())
fraction = sample_counts[str(bank_id)] / N
encrypted_chunks = client_he.encrypt(local_weights, scale=fraction)
```

`encrypted_chunks` is an ordered `list[bytes]`. The sample-count weight is applied before encryption. It is distinct from the CKKS encoding scale of `2**40`.

This protocol aggregates **full local model weights**. A future change to model deltas requires an explicit agreement on how the previous global weights are added back.

## 4. Server hook: sum ciphertexts exactly once

Construct the server aggregator using only the public context:

```python
server_he = CKKSAggregator(public_context_bytes, spec, sample_counts)
```

After validating one reply from every configured bank, collect the replies into a mapping and aggregate:

```python
# Mapping: string bank ID -> ordered list of ciphertext bytes.
global_chunks = server_he.aggregate(encrypted_updates_by_bank)
```

Broadcast `global_chunks` to the clients without further weighting or division. Ordinary numerical FedAvg must not process these ciphertext bytes. The Flower transport and custom strategy must preserve chunk ordering and binary contents using the API version chosen by the FL owner.

The current method rejects missing or unexpected banks. Abort an incomplete round; application code decides whether to retry. Do not accept a partial sum with the original denominator. Partial participation and dropout recovery require a revised protocol.

## 5. Validate the message envelope in the Flower layer

The HE methods return raw ciphertext bytes and do not implement transport authentication or round tracking. Agree on and validate these fields before calling the aggregator:

- Protocol version and experiment ID.
- Round ID and authenticated bank ID.
- Model-spec fingerprint.
- Context ID derived from the exact shared public-context bytes, such as SHA-256.
- Agreed training sample count and ordered ciphertext chunks.

Reject duplicate bank replies before constructing the mapping, because duplicate dictionary keys can otherwise overwrite earlier replies. Reject stale rounds, wrong experiments and mismatched context/model identifiers.

These checks do not prove that a malicious bank supplied honest weights or used the correct sample scale. The shared-key design assumes trusted clients and an honest-but-curious server that does not collude with a client. A client holding the shared secret key could decrypt another client's ciphertext if it obtained it.

## 6. Log costs and evaluate on the client side

Record encryption, aggregation and decryption times separately. Report whether client costs are individual, averaged or sequential totals. Record serialized ciphertext upload bytes, one aggregate payload size and total broadcast bytes. Keep one-time setup, training time and network transport time separate.

The server cannot directly evaluate its encrypted global model. Clients can decrypt, load and evaluate it, then return explicitly agreed metrics. A separate trusted evaluator may also hold the secret key; the aggregation server must remain keyless.

Use validation data for debugging and threshold selection. Leave final held-out test evaluation to the coordinated experiment workflow. Do not interpret the one-round check as a converged fraud-model result.

## 7. Verification sequence

1. Confirm that a Flower round transports ciphertext bytes without applying numerical FedAvg to them.
2. For a diagnostic round, compare decrypted aggregation with plaintext averaging of the **same trained local updates**, using the existing `1e-5` maximum absolute weight-error tolerance. Keep plaintext references inside a trusted diagnostic harness, not the normal aggregation server.
3. Confirm that decrypted weights load into every client's model and evaluation succeeds.
4. Run two training rounds and confirm that round 2 starts from the decrypted aggregate of round 1.
5. Run the planned 10–20 rounds for 3 and 5 banks, comparing matched plaintext and encrypted experiment settings. Approximate arithmetic can cause trajectories to diverge slightly, so final metric equality is not guaranteed.
6. Connect the other fraud-view models and supply their HE cost logs to the evaluation owner.

The existing `he_real_data_check.py` is the reference for trained-weight extraction, sample-count weighting, decryption/reloading, and cost logging. It runs all roles sequentially in one process; it does not demonstrate process isolation or a completed Flower implementation. Keep generated secret contexts, raw/processed transaction data, and local environments outside Git.
