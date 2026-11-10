"""CPU-only synthetic aggregation benchmark; no dataset or training required."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
from he import CKKSClient, CKKSAggregator, ModelSpec, create_contexts


def run_case(n_parameters, n_clients, rounds, seed, atol):
    rng = np.random.default_rng(seed)
    # Mixed shapes check layer reconstruction as well as cross-layer chunking.
    template = [np.zeros((100, n_parameters // 100 - 1), dtype=np.float32),
                np.zeros(100, dtype=np.float32)]
    spec = ModelSpec.from_weights(template, ["dense.weight", "dense.bias"])
    counts = {f"bank_{i}": 100 + 37 * i for i in range(n_clients)}
    start = perf_counter()
    keys = create_contexts()
    server = CKKSAggregator(keys.public_context, spec, counts)
    clients = {bank: CKKSClient(keys.secret_context, spec) for bank in counts}
    setup_s = perf_counter() - start
    rows = []
    for round_id in range(1, rounds + 1):
        updates, local_models = {}, {}
        encryption_s = []
        for bank, client in clients.items():
            # This is synthetic input, not a trained model or a fraud result.
            weights = [(w + rng.normal(0, 0.01, w.shape)).astype(np.float32) for w in template]
            local_models[bank] = weights
            start = perf_counter()
            updates[bank] = client.encrypt(weights, server.scale_for(bank))
            encryption_s.append(perf_counter() - start)
        start = perf_counter()
        aggregate = server.aggregate(updates)
        aggregation_s = perf_counter() - start
        decryption_s, global_models = [], []
        for client in clients.values():
            start = perf_counter()
            global_models.append(client.decrypt(aggregate))
            decryption_s.append(perf_counter() - start)
        reference = sum((spec.flatten(w) * server.scale_for(bank)
                         for bank, w in local_models.items()), np.zeros(spec.n_values))
        errors = [np.abs(spec.flatten(w) - reference) for w in global_models]
        max_error = max(float(e.max()) for e in errors)
        if max_error > atol:
            raise AssertionError(f"Round {round_id}: error {max_error:g} exceeds {atol:g}")
        template = global_models[0]
        payload = sum(len(b) for b in aggregate)
        rows.append({
            "n_parameters": spec.n_values, "n_clients": n_clients, "round": round_id,
            "seed": seed, "chunks_per_client": len(spec.chunk_lengths),
            "setup_s": setup_s, "encrypt_total_s": sum(encryption_s),
            "encrypt_mean_client_s": float(np.mean(encryption_s)),
            "aggregate_s": aggregation_s, "decrypt_total_s": sum(decryption_s),
            "decrypt_mean_client_s": float(np.mean(decryption_s)),
            "sequential_crypto_total_s": sum(encryption_s) + aggregation_s + sum(decryption_s),
            "uplink_ciphertext_bytes": sum(len(b) for chunks in updates.values() for b in chunks),
            "aggregate_ciphertext_bytes": payload,
            "broadcast_ciphertext_bytes": payload * n_clients,
            "plaintext_bytes_per_client": sum(w.nbytes for w in template),
            "max_abs_error": max_error, "tolerance": atol, "passed": True,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=int, nargs="+", default=[10000, 50000])
    parser.add_argument("--clients", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=Path("results/he_benchmark.csv"))
    args = parser.parse_args()
    if any(n < 200 or n % 100 for n in args.parameters):
        parser.error("Parameter counts must be multiples of 100 and at least 200")
    if any(n < 1 for n in args.clients) or args.rounds < 1:
        parser.error("Client and round counts must be positive")
    if not np.isfinite(args.atol) or args.atol <= 0:
        parser.error("atol must be finite and positive")
    rows = []
    for n in args.parameters:
        for c in args.clients:
            case = run_case(n, c, args.rounds, args.seed, args.atol)
            rows.extend(case)
            print(f"PASS parameters={n} banks={c} rounds={args.rounds} "
                  f"max_error={max(r['max_abs_error'] for r in case):.3g}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "numpy": np.__version__,
        "tenseal": importlib.metadata.version("tenseal"), "threads_per_context": 1,
        "poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 40, 40, 60],
        "global_scale": 2**40, "public_context_transfer": "once per experiment; excluded from round payload",
        "benchmark_type": "sequential synthetic crypto only; excludes training and network transport",
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
