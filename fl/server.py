"""Federated simulation harness: run_federated(cfg) for modes "fedavg" and "fedavg_he".

Flower's simulation entry points (``start_simulation`` and ``run_simulation``, both deprecated in flwr 1.43)
require Ray, which is not in requirements.txt. Instead the harness drives Flower's own objects in-process: the
FedAvg-based strategy, a SimpleClientManager, and each NumPyClient through ``to_client()``, so parameters are
serialized to and from Flower ``Parameters`` exactly as on the wire. Rounds are sequential and deterministic.

Roles in "fedavg_he":
  * trusted setup (this module, ``_trusted_setup``) creates the keys once, gives each client a CKKSClient built
    from the secret context as a process-level object, and gives the strategy only a CKKSAggregator built from
    the public context. No key material is put in a Flower config or message.
  * the harness also acts as the trusted evaluator: it holds its own CKKSClient so it can decrypt the global
    model and score the global test set. The aggregation server cannot do this: it must stay keyless (see
    INTEGRATION.md section 6), so test evaluation is deliberately outside the strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from flwr.common import Code, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy

from common import interfaces as I
from common.config import ExperimentConfig
from common.utils import get_weights, set_seed, set_weights
from he import create_contexts

from .client import create_client
from .logging import write_rows_csv
from .metrics import ROW_KEYS, make_row, score_model  # noqa: F401  (ROW_KEYS re-exported)
from .strategy import EncryptedFedAvg, FraudFedAvg

MODES = ("fedavg", "fedavg_he")


class InProcessClientProxy(ClientProxy):
    """Calls a NumPyClient through Flower's Client wrapper (same (de)serialization as a real transport)."""

    def __init__(self, cid: str, numpy_client):
        super().__init__(cid)
        self.client = numpy_client.to_client()

    def fit(self, ins, timeout=None, group_id=None):
        return self.client.fit(ins)

    def evaluate(self, ins, timeout=None, group_id=None):
        return self.client.evaluate(ins)

    def get_parameters(self, ins, timeout=None, group_id=None):
        return self.client.get_parameters(ins)

    def get_properties(self, ins, timeout=None, group_id=None):
        return self.client.get_properties(ins)

    def reconnect(self, ins, timeout=None, group_id=None):
        raise NotImplementedError


@dataclass
class FederatedResult:
    rows: list[dict]
    global_weights: list[list[np.ndarray]] = field(repr=False)  # plaintext global model after each round
    strategy: FraudFedAvg = field(repr=False)


def _call_all(pairs, method):
    results, failures = [], []
    for proxy, ins in pairs:
        try:
            res = getattr(proxy, method)(ins)
            (results if res.status.code == Code.OK else failures).append((proxy, res))
        except Exception as exc:  # recorded as a Flower failure; the strategy decides (it aborts)
            failures.append(exc)
    return results, failures


def _trusted_setup(cfg, init_model, counts):
    """Create keys once; return (client CKKS object per bank, server aggregator, evaluator CKKS object)."""
    keys = create_contexts()
    spec = I.make_spec(init_model)
    clients = {int(b): I.make_client(keys.secret_context, spec) for b in counts}
    return clients, I.make_aggregator(keys.public_context, spec, counts), I.make_client(keys.secret_context, spec)


def _mean(values):
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def max_weight_error(weights, clients, counts) -> float:
    """Trusted diagnostic: max |global - sum_i (n_i/N) w_i| over the plaintext updates the clients encrypted."""
    N = sum(counts.values())
    plain = [sum(counts[str(b)] / N * c.last_update[i].astype(np.float64) for b, c in clients.items())
             for i in range(len(weights))]
    return float(max(np.abs(np.asarray(w, np.float64) - p).max() for w, p in zip(weights, plain)))


def _apply_label_transforms(data, client_label_transform):
    """Replace y_train of each bank whose client_label_transform(bank_id) returns a function."""
    if client_label_transform is None:
        return data
    out = {}
    for b, (X_tr, y_tr, X_va, y_va) in data.items():
        fn = client_label_transform(b)
        if fn is not None:
            y_new = np.asarray(fn(y_tr.copy()), dtype=np.float32)
            if y_new.shape != y_tr.shape:
                raise ValueError(f"label transform for bank {b} changed y_train shape {y_tr.shape} -> {y_new.shape}")
            y_tr = y_new
        out[b] = (X_tr, y_tr, X_va, y_va)
    return out


def simulate(cfg: ExperimentConfig, *, client_label_transform=None, server_observer=None) -> FederatedResult:
    """Run cfg.rounds rounds of cfg.mode; see run_federated for the run and the two optional hooks."""
    if cfg.mode not in MODES:
        raise ValueError(f"run_federated supports modes {MODES}, got {cfg.mode!r}")
    set_seed(cfg.seed)
    he = cfg.mode == "fedavg_he"
    data = {b: I.load_client_data(b, cfg.fraud_type, cfg) for b in range(cfg.n_clients)}
    data = _apply_label_transforms(data, client_label_transform)  # y_train only; validation labels untouched
    counts = {str(b): len(d[1]) for b, d in data.items()}
    N = sum(counts.values())
    X_val = np.concatenate([d[2] for d in data.values()])
    y_val = np.concatenate([d[3] for d in data.values()])
    X_test, y_test = I.load_test(cfg.fraud_type, cfg)
    global_model = I.build_model(cfg.fraud_type, X_test.shape[1], seed=cfg.seed)
    initial = ndarrays_to_parameters(get_weights(global_model))  # round-1 downlink is plaintext in both modes

    he_clients, aggregator, evaluator = _trusted_setup(cfg, global_model, counts) if he else ({}, None, None)
    manager, cid_to_bank, clients = SimpleClientManager(), {}, {}
    for b in range(cfg.n_clients):
        cid = f"client-{b}"  # the Flower identity; the strategy maps it to a bank id
        clients[b] = create_client(b, cfg, he_clients.get(b), N if he else None, data=data[b])
        manager.register(InProcessClientProxy(cid, clients[b]))
        cid_to_bank[cid] = b
    strategy = (EncryptedFedAvg(cfg.n_clients, initial, aggregator, cid_to_bank, server_observer=server_observer)
                if he else FraudFedAvg(cfg.n_clients, initial, server_observer=server_observer))

    parameters = strategy.initialize_parameters(manager)
    rows, history = [], []
    for rnd in range(1, cfg.rounds + 1):
        start = perf_counter()
        fit_res, fit_fail = _call_all(strategy.configure_fit(rnd, parameters, manager), "fit")
        parameters, _ = strategy.aggregate_fit(rnd, fit_res, fit_fail)
        if parameters is None:
            raise RuntimeError(f"round {rnd}: aggregation returned no parameters")
        ev_res, ev_fail = _call_all(strategy.configure_evaluate(rnd, parameters, manager), "evaluate")
        if ev_fail:
            raise RuntimeError(f"round {rnd}: {len(ev_fail)} client evaluation failure(s)")
        val_loss, val_metrics = strategy.aggregate_evaluate(rnd, ev_res, ev_fail)
        wall = perf_counter() - start  # training + crypto + aggregation + client validation; excludes test eval

        # Trusted evaluator (not the server): decrypt the global model, tune the threshold on the pooled
        # validation predictions and score the held-out test set.
        arrays = parameters_to_ndarrays(parameters)
        weights = I.decrypt(I.ndarrays_to_bytes(arrays), evaluator) if he else arrays
        set_weights(global_model, weights)
        history.append(get_weights(global_model))

        fm = [r.metrics for _, r in fit_res]
        em = {r.metrics["bank_id"]: r.metrics for _, r in ev_res}
        rows.append(make_row(
            cfg, rnd, **score_model(global_model, X_val, y_val, X_test, y_test),
            val_loss=float(val_loss), val_f1=float(val_metrics["f1"]),
            train_loss=_mean(m["train_loss"] for m in fm), train_s=_mean(m["train_s"] for m in fm),
            encrypt_s=_mean(m["encrypt_s"] for m in fm),
            # per-client mean time to decrypt this round's aggregate (cached for the next fit)
            decrypt_s=_mean(m["decrypt_s"] + em[m["bank_id"]]["decrypt_s"] for m in fm),
            aggregate_s=strategy.last_cost["aggregate_s"],
            uplink_bytes_per_client=_mean(m["uplink_bytes"] for m in fm),
            downlink_bytes=strategy.last_cost["downlink_bytes"], round_wall_s=wall,
            max_weight_err=max_weight_error(weights, clients, counts) if he else None))
    return FederatedResult(rows, history, strategy)


def run_federated(cfg: ExperimentConfig, csv_path=None, *, client_label_transform=None,
                  server_observer=None) -> list[dict]:
    """Run FedAvg ("fedavg") or CKKS-encrypted FedAvg ("fedavg_he") and return one metrics row per round.

    Every bank must take part in every round. If any client fails (fit or evaluate) the run is aborted with
    RuntimeError: encrypted aggregation needs all banks' ciphertexts, and plaintext FedAvg is held to the same
    rule so the two modes stay comparable. Rows are appended to ``csv_path`` only if a path is given.
    Test metrics are reported at threshold 0.5 and at one threshold tuned each round on the global model's
    pooled validation predictions (fl.metrics.score_model).

    Optional hooks for experiments/attacks (both default to None, which leaves the run unchanged):
      * client_label_transform: bank_id -> None or a function y_train -> y_train', applied once to that bank's
        training labels before any local training (validation and test labels are never changed). pos_weight
        still comes from the loader, i.e. from the bank's true labels.
      * server_observer: callable receiving one fl.strategy.RoundObservation per round: exactly the arrays the
        server receives from each client (float32 weights in "fedavg", uint8 ciphertext chunks in
        "fedavg_he"), with bank ids and sample counts, plus the parameters the server broadcast for the round.
    """
    rows = simulate(cfg, client_label_transform=client_label_transform, server_observer=server_observer).rows
    if csv_path is not None:
        write_rows_csv(rows, csv_path)
    return rows
