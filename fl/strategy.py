"""Server strategies: plaintext sample-weighted FedAvg and ciphertext-only FedAvg for CKKS."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import flwr
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from common import interfaces as I
from he import CKKSAggregator


@dataclass(frozen=True)
class Upload:
    """One client's fit reply exactly as the server receives it (after Flower deserialisation).

    ``arrays`` are float32 weight arrays in "fedavg" and 1-D uint8 CKKS ciphertext chunks in "fedavg_he".
    ``bank_id`` is the id the reply carries (verified against the Flower identity in "fedavg_he").
    """
    bank_id: int
    num_examples: int
    arrays: list = field(repr=False)


@dataclass(frozen=True)
class RoundObservation:
    """Everything an honest-but-curious server sees in one fit round: the parameters it broadcast for this
    round (its own state: plaintext in round 1 and in "fedavg", ciphertext afterwards in "fedavg_he") and every
    client's upload, sorted by bank id."""
    server_round: int
    broadcast: list = field(repr=False)
    uploads: list


def _weighted_mean(metrics, key):
    total = sum(n for n, _ in metrics)
    return sum(n * m[key] for n, m in metrics) / total if total else float("nan")


def _evaluate_metrics(metrics):
    return {"f1": _weighted_mean(metrics, "f1")}


class FraudFedAvg(flwr.server.strategy.FedAvg):
    """Standard FedAvg (weighted by each client's num_examples) that also times aggregation and tells clients
    whether the downlink is encrypted. All banks must reply; a failed client aborts the round.

    ``server_observer`` (optional, default None): called once per fit round with a RoundObservation, before
    aggregation. Used only by experiments/attacks to model an honest-but-curious server; it gets copies, so it
    cannot change what is aggregated. With None nothing is recorded or copied.
    """

    encrypted_downlink = False

    def __init__(self, n_clients: int, initial_parameters, server_observer=None, **kwargs):
        super().__init__(fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=n_clients,
                         min_evaluate_clients=n_clients, min_available_clients=n_clients,
                         accept_failures=False, initial_parameters=initial_parameters,
                         on_fit_config_fn=self._config, on_evaluate_config_fn=self._config,
                         evaluate_metrics_aggregation_fn=_evaluate_metrics,
                         fit_metrics_aggregation_fn=lambda _: {}, **kwargs)
        self.n_clients = n_clients
        self.last_cost: dict = {}
        self.server_observer = server_observer
        self._broadcast = None

    def _config(self, server_round: int) -> dict:
        # Explicit flag: round 1's fit receives the plaintext initial model; everything after is the aggregate.
        later = self.encrypted_downlink and server_round > 1
        return {"server_round": server_round, "encrypted": bool(later)}

    def configure_fit(self, server_round, parameters, client_manager):
        if self.server_observer is not None:
            self._broadcast = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def _observe(self, server_round, results):
        if self.server_observer is None:
            return
        uploads = sorted((Upload(int(res.metrics["bank_id"]), int(res.num_examples),
                                 [np.array(a, copy=True) for a in parameters_to_ndarrays(res.parameters)])
                          for _, res in results), key=lambda u: u.bank_id)
        self.server_observer(RoundObservation(server_round, self._broadcast, uploads))

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Evaluation always receives this round's aggregate, so it is encrypted whenever aggregates are.
        pairs = super().configure_evaluate(server_round, parameters, client_manager)
        for _, ins in pairs:
            ins.config["encrypted"] = self.encrypted_downlink
        return pairs

    def _check(self, results, failures):
        if failures or len(results) != self.n_clients:
            raise RuntimeError(f"aggregation requires all {self.n_clients} banks: got {len(results)} results "
                               f"and {len(failures)} failures; aborting the run")

    def aggregate_fit(self, server_round, results, failures):
        self._check(results, failures)
        self._observe(server_round, results)
        start = perf_counter()
        params, metrics = super().aggregate_fit(server_round, results, failures)
        self.last_cost = {"aggregate_s": perf_counter() - start,
                          "downlink_bytes": sum(a.nbytes for a in parameters_to_ndarrays(params))}
        return params, metrics


class EncryptedFedAvg(FraudFedAvg):
    """Adds CKKS ciphertexts; Flower's numerical FedAvg never touches them.

    Holds only a CKKSAggregator (public context). Clients have already scaled by n_i/N, so the sum is the
    weighted average. Every bank must reply exactly once each round: a missing, failed, duplicate or unknown
    bank raises RuntimeError/ValueError and aborts the run, because the ciphertext sum has no valid partial form.
    """

    encrypted_downlink = True

    def __init__(self, n_clients: int, initial_parameters, aggregator: CKKSAggregator, cid_to_bank: dict,
                 server_observer=None):
        if not isinstance(aggregator, CKKSAggregator):
            raise TypeError("EncryptedFedAvg takes a CKKSAggregator (public context only)")
        if sorted(map(str, cid_to_bank.values())) != sorted(aggregator.sample_counts):
            raise ValueError("cid_to_bank must cover exactly the aggregator's banks")
        super().__init__(n_clients, initial_parameters, server_observer=server_observer)
        self.aggregator = aggregator
        self.cid_to_bank = {str(c): str(b) for c, b in cid_to_bank.items()}

    def aggregate_fit(self, server_round, results, failures):
        self._check(results, failures)
        updates = {}
        for proxy, res in results:
            bank = self.cid_to_bank.get(proxy.cid)
            if bank is None:
                raise ValueError(f"unknown Flower client id {proxy.cid!r}")
            if bank in updates:
                raise ValueError(f"duplicate reply from bank {bank}")
            if str(res.metrics.get("bank_id")) != bank:
                raise ValueError(f"client {proxy.cid} claims bank {res.metrics.get('bank_id')}, expected {bank}")
            updates[bank] = I.ndarrays_to_bytes(parameters_to_ndarrays(res.parameters))
        self._observe(server_round, results)  # after validation: only well-formed rounds are observed
        start = perf_counter()
        summed = I.aggregate(updates, self.aggregator)
        self.last_cost = {"aggregate_s": perf_counter() - start, "downlink_bytes": sum(len(c) for c in summed)}
        return ndarrays_to_parameters(I.bytes_to_ndarrays(summed)), {}
