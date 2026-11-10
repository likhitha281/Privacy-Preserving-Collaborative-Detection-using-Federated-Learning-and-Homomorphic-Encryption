"""Project interface from the Implementation Architecture Plan, as thin wrappers over data/, models/ and he/.

HE objects: ``ctx`` is an ``he.CKKSClient`` (secret context, client side) and ``agg`` an ``he.CKKSAggregator``
(public context only, server side). Build them once per experiment with make_client / make_aggregator.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from data import loader
from he import CKKSAggregator, CKKSClient, ModelSpec

from . import utils
from .config import ExperimentConfig


# ----------------------------------------------------------------------------- data
def load_client_data(bank_id: int, fraud_type: str, cfg: ExperimentConfig):
    """X_train, y_train, X_val, y_val as float32 (the loader returns int8 labels)."""
    X_tr, y_tr, X_va, y_va = loader.load_client_data(bank_id, fraud_type, **cfg.loader_kwargs())
    return X_tr.astype(np.float32, copy=False), y_tr.astype(np.float32), \
        X_va.astype(np.float32, copy=False), y_va.astype(np.float32)


def load_test(fraud_type: str, cfg: ExperimentConfig):
    X, y = loader.load_test(fraud_type, **cfg.loader_kwargs())
    return X.astype(np.float32, copy=False), y.astype(np.float32)


def input_dim(fraud_type: str, cfg: ExperimentConfig) -> int:
    return loader.input_dim(fraud_type, **cfg.loader_kwargs())


def pos_weight(bank_id: int, fraud_type: str, cfg: ExperimentConfig) -> float:
    return loader.get_pos_weight(bank_id, fraud_type, **cfg.loader_kwargs())


def sample_counts(fraud_type: str, cfg: ExperimentConfig) -> dict[str, int]:
    """Training rows per bank, keyed by string bank id (the key type CKKSAggregator expects)."""
    return {str(b): len(load_client_data(b, fraud_type, cfg)[1]) for b in range(cfg.n_clients)}


# ----------------------------------------------------------------------------- model
def build_model(fraud_type: str, input_dim: int, seed: int | None = None):
    return utils.build_model(fraud_type, input_dim, seed)


# ----------------------------------------------------------------------------- HE
def make_spec(model) -> ModelSpec:
    return ModelSpec.from_weights(utils.get_weights(model), names=utils.param_names(model))


def make_client(secret_context: bytes, spec: ModelSpec) -> CKKSClient:
    return CKKSClient(secret_context, spec)


def make_aggregator(public_context: bytes, spec: ModelSpec, counts: Mapping[str, int]) -> CKKSAggregator:
    return CKKSAggregator(public_context, spec, counts)


def sample_scale(n_i: int, N: int) -> float:
    """FedAvg weight n_i / N, applied by the client before encryption (not the CKKS 2**40 scale)."""
    for name, v in (("n_i", n_i), ("N", N)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
            raise ValueError(f"{name} must be a positive int, got {v!r}")
    if n_i > N:
        raise ValueError(f"n_i ({n_i}) cannot exceed N ({N})")
    return n_i / N


def encrypt(weights: Sequence[np.ndarray], scale: float, ctx: CKKSClient) -> list[bytes]:
    """Cast to the spec dtypes (float32 from get_weights) so he/ does not reject e.g. float64 copies."""
    if len(weights) != len(ctx.spec.dtypes):
        raise ValueError(f"expected {len(ctx.spec.dtypes)} arrays, got {len(weights)}")
    return ctx.encrypt([np.asarray(w, dtype=d) for w, d in zip(weights, ctx.spec.dtypes)], scale)


def aggregate(updates: Mapping, agg: CKKSAggregator) -> list[bytes]:
    """Sum preweighted ciphertexts. Keys are normalised to str; the aggregator aborts on missing/extra banks."""
    keyed = {str(k): list(v) for k, v in updates.items()}
    if len(keyed) != len(updates):
        raise ValueError("duplicate bank ids after str() normalisation")
    return agg.aggregate(keyed)


def decrypt(chunks: Sequence[bytes], ctx: CKKSClient) -> list[np.ndarray]:
    return ctx.decrypt(list(chunks))


# ----------------------------------------------------------------------------- Flower transport
def bytes_to_ndarrays(chunks: Sequence[bytes]) -> list[np.ndarray]:
    """Wrap ciphertext bytes as 1-D uint8 arrays so a NumPyClient can return them as parameters."""
    if any(not isinstance(c, bytes) for c in chunks):
        raise TypeError("every chunk must be bytes")
    return [np.frombuffer(c, dtype=np.uint8) for c in chunks]


def ndarrays_to_bytes(arrays: Sequence[np.ndarray]) -> list[bytes]:
    """Inverse of bytes_to_ndarrays; refuses anything that is not a 1-D uint8 array (no silent casts)."""
    out = []
    for a in arrays:
        a = np.asarray(a)
        if a.dtype != np.uint8 or a.ndim != 1:
            raise ValueError(f"expected 1-D uint8 array, got dtype={a.dtype} ndim={a.ndim}")
        out.append(a.tobytes())
    return out
