"""One trained-weight CKKS aggregation check using the team's identity model.

Copy this script to the team repository root, alongside data/, models/ and he/.
This is a one-round integration check, not a Flower run or final fraud baseline.
No raw records, weights, ciphertexts or secret keys are written to the outputs.
"""
from __future__ import annotations
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter


def subset(X, y, limit, seed):
    import numpy as np
    from sklearn.model_selection import train_test_split
    if limit == 0 or limit >= len(y):
        return X, y
    labels, counts = np.unique(y, return_counts=True)
    stratify = y if len(labels) == 2 and counts.min() >= 2 and len(y) - limit >= 2 else None
    indices, _ = train_test_split(np.arange(len(y)), train_size=limit,
                                  random_state=seed, stratify=stratify)
    return X[indices], y[indices]


def json_safe(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--banks', type=int, choices=[3, 5], default=3)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--max-train-per-bank', type=int, default=2000,
                        help='0 uses all training examples; positive values use a seeded subset')
    parser.add_argument('--max-val-per-bank', type=int, default=1000,
                        help='0 uses all validation examples')
    parser.add_argument('--seed', type=int, default=42, help='model/training/subsample seed')
    parser.add_argument('--data-seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--partition', choices=['label_skew', 'iid', 'quantity_skew'], default='label_skew')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--threads', type=int, default=4, help='PyTorch CPU threads; HE contexts use one thread')
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--raw-dir', type=Path, help='Optional override for the two IEEE-CIS CSVs')
    parser.add_argument('--processed-dir', type=Path, help='Optional override for the cache parent')
    parser.add_argument('--data-kind', choices=['ieee', 'synthetic-test'], default='ieee',
                        help='Output label only; synthetic-test never generates or downloads data')
    parser.add_argument('--output-dir', type=Path, default=Path('results/he_identity_smoke'))
    args = parser.parse_args()
    if args.local_epochs < 1 or args.batch_size < 1 or args.threads < 1:
        parser.error('epochs, batch size and threads must be positive')
    if any(v < 0 or v == 1 for v in [args.max_train_per_bank, args.max_val_per_bank]):
        parser.error('subset limits must be 0 (all) or at least 2')
    if not 0 < args.atol < float('inf') or not 0 < args.alpha < float('inf'):
        parser.error('atol and alpha must be finite and positive')
    root = Path(__file__).resolve().parent
    if Path.cwd().resolve() != root:
        parser.error('Run from the repository root containing this script')
    required = ['data/loader.py', 'models/identity.py', 'he/ckks.py', 'he/keys.py', 'he/__init__.py']
    missing = [p for p in required if not (root / p).is_file()]
    if missing:
        parser.error('Missing required files: ' + ', '.join(missing))
    # Require a fresh output directory to preserve evidence from earlier runs.
    if args.output_dir.exists():
        parser.error('Output directory already exists. Choose a new --output-dir to preserve prior results.')
    try:
        import numpy as np
        import torch
        from data.loader import configure, ensure_federation, load_client_data
        from models.identity import build_model, train_epochs, get_weights, set_weights, evaluate, predict_proba
        from he import create_contexts, ModelSpec, CKKSClient, CKKSAggregator
    except ImportError as error:
        parser.error(f'{error}. Install the dependencies and confirm your original he/__init__.py was copied.')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    configure(raw_dir=args.raw_dir, processed_dir=args.processed_dir)
    cfg = dict(n_banks=args.banks, alpha=args.alpha, seed=args.data_seed, partition=args.partition)
    print('Loading the identity_fraud feature view. The first call may preprocess the raw CSVs.', flush=True)
    federation = ensure_federation(**cfg)
    prepared = []
    for bank in range(args.banks):
        Xt, yt, Xv, yv = load_client_data(bank, 'identity_fraud', **cfg)
        full_n = len(yt)
        Xt, yt = subset(Xt, yt, args.max_train_per_bank, args.seed + bank)
        Xv, yv = subset(Xv, yv, args.max_val_per_bank, args.seed + 100 + bank)
        if not len(yt) or not len(yv):
            raise ValueError(f'Bank {bank} has an empty training or validation split')
        if np.unique(yt).size != 2:
            raise ValueError(f'Bank {bank} training subset needs both classes; use a larger subset')
        if not np.isfinite(Xt).all() or not np.isfinite(Xv).all():
            raise ValueError(f'Bank {bank} contains nonfinite features')
        prepared.append((bank, Xt, yt, Xv, yv, full_n))
        print(f'Bank {bank}: using {len(yt)}/{full_n} train rows, {len(yv)} validation rows', flush=True)
    dim = prepared[0][1].shape[1]
    if any(item[1].shape[1] != dim or item[3].shape[1] != dim for item in prepared):
        raise ValueError('Banks have different feature dimensions')
    template = build_model(dim, seed=args.seed)
    initial_state = copy.deepcopy(template.state_dict())
    names = [name for name, _ in template.named_parameters()]
    spec = ModelSpec.from_weights(get_weights(template), names=names)
    counts = {str(bank): int(len(yt)) for bank, _, yt, _, _, _ in prepared}
    setup_start = perf_counter()
    keys = create_contexts()
    server = CKKSAggregator(keys.public_context, spec, counts)
    clients = {str(bank): CKKSClient(keys.secret_context, spec) for bank in range(args.banks)}
    setup_s = perf_counter() - setup_start
    updates, reference = {}, np.zeros(spec.n_values, dtype=np.float64)
    bank_rows = []
    for bank, Xt, yt, Xv, yv, full_n in prepared:
        bank_id = str(bank)
        model = build_model(dim)
        model.load_state_dict(initial_state)
        if any(not torch.equal(value, initial_state[name]) for name, value in model.state_dict().items()):
            raise AssertionError('Local models did not start from identical weights')
        positives = int(yt.sum())
        pos_weight = (len(yt) - positives) / positives
        training_seed = args.seed + 1000 + bank
        torch.manual_seed(training_seed)  # Seeds dropout as well as the shuffle seed below.
        start = perf_counter()
        losses = train_epochs(model, Xt, yt, pos_weight=pos_weight, epochs=args.local_epochs,
                              batch_size=args.batch_size, device='cpu', seed=training_seed)
        train_s = perf_counter() - start
        weights = get_weights(model)
        # Train ONCE. Both plaintext and HE paths consume these identical local updates.
        fraction = server.scale_for(bank_id)
        reference += spec.flatten(weights) * fraction
        start = perf_counter()
        updates[bank_id] = clients[bank_id].encrypt(weights, scale=fraction)
        encrypt_s = perf_counter() - start
        bank_rows.append(dict(bank_id=bank, available_train_rows=full_n,
                              used_train_rows=len(yt), used_train_frauds=positives,
                              used_val_rows=len(yv), sample_weight=fraction, pos_weight=pos_weight,
                              final_local_loss=losses[-1], train_s=train_s, encrypt_s=encrypt_s,
                              ciphertext_upload_bytes=sum(map(len, updates[bank_id]))))
        print(f'Bank {bank}: trained {args.local_epochs} epoch(s), encrypted {len(updates[bank_id])} chunks', flush=True)
    start = perf_counter()
    aggregated = server.aggregate(updates)
    aggregate_s = perf_counter() - start
    decrypted_weights, max_error = [], 0.0
    for row in bank_rows:
        start = perf_counter()
        weights = clients[str(row['bank_id'])].decrypt(aggregated)
        row['decrypt_s'] = perf_counter() - start
        error = float(np.abs(spec.flatten(weights) - reference).max())
        row['max_abs_weight_error'] = error
        max_error = max(max_error, error)
        decrypted_weights.append(weights)
    # The plaintext model is restored through exactly the same float32 load path.
    plain_model, encrypted_model = build_model(dim), build_model(dim)
    set_weights(plain_model, spec.unflatten(reference))
    validation_rows = []
    max_prediction_difference = 0.0
    for item, weights in zip(prepared, decrypted_weights):
        bank, _, _, Xv, yv, _ = item
        set_weights(encrypted_model, weights)
        probabilities_plain = predict_proba(plain_model, Xv, device='cpu')
        probabilities_he = predict_proba(encrypted_model, Xv, device='cpu')
        prediction_difference = float(np.abs(probabilities_plain - probabilities_he).max())
        max_prediction_difference = max(max_prediction_difference, prediction_difference)
        for mode, model in [('plaintext_average', plain_model), ('ckks_average', encrypted_model)]:
            metrics = evaluate(model, Xv, yv, threshold=0.5, device='cpu')
            validation_rows.append(dict(bank_id=bank, mode=mode,
                                        max_abs_probability_difference=prediction_difference, **metrics))
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / 'bank_costs.csv', bank_rows)
    write_csv(args.output_dir / 'validation_metrics.csv', validation_rows)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = json.loads((federation / 'manifest.json').read_text())
    report = dict(
        status='PASS' if max_error <= args.atol else 'FAIL',
        check_type='one_round_trained_weight_aggregation_not_flower', data_kind=args.data_kind,
        fraud_type='identity_fraud', rounds=1, local_epochs=args.local_epochs,
        banks=args.banks, input_dim=dim, n_parameters=spec.n_values,
        chunks_per_client=len(spec.chunk_lengths), used_sample_counts=counts,
        max_train_per_bank=args.max_train_per_bank, max_val_per_bank=args.max_val_per_bank,
        model_and_subsample_seed=args.seed, data_config=cfg,
        max_abs_weight_error=max_error, weight_error_tolerance=args.atol,
        max_abs_probability_difference=max_prediction_difference,
        setup_s=setup_s, aggregate_s=aggregate_s,
        encrypt_total_s=sum(r['encrypt_s'] for r in bank_rows),
        decrypt_total_s=sum(r['decrypt_s'] for r in bank_rows),
        ciphertext_upload_bytes=sum(r['ciphertext_upload_bytes'] for r in bank_rows),
        aggregate_ciphertext_bytes=sum(map(len, aggregated)),
        total_broadcast_ciphertext_bytes=args.banks * sum(map(len, aggregated)),
        model_spec_fingerprint=spec.fingerprint,
        data_manifest_rows=manifest.get('rows'), data_manifest_split_rows=manifest.get('split_rows'),
        source_commit=commit, source_sha256={p: sha256(root / p) for p in required},
        python=platform.python_version(), platform=platform.platform(), cpu_threads=args.threads,
        package_versions={name: importlib.metadata.version(name) for name in
                          ['numpy', 'torch', 'tenseal', 'pandas', 'pyarrow', 'scikit-learn']},
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        notes=['All roles execute sequentially in one process; this is not process isolation.',
               'Sample weights use training rows actually used, including subset limits.',
               'Metrics use validation data with fixed threshold 0.5; test set is not evaluated.',
               'The two averages use identical trained updates, not separately retrained FL trajectories.',
               'Training and network latency are excluded from reported crypto times.',
               'Crypto setup/contexts use the original HE module; no DP is applied.'])
    (args.output_dir / 'summary.json').write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + '\n')
    print(f"{report['status']}: max_abs_weight_error={max_error:.3g}; tolerance={args.atol:g}", flush=True)
    print(f'Maximum validation probability difference: {max_prediction_difference:.3g}', flush=True)
    print(f'Results written to {args.output_dir}', flush=True)
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
