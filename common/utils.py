"""Weight exchange, seeding and model construction shared by fl/, he/ glue and experiments/."""
from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
from torch import nn


# get_weights / set_weights: copied from models/identity.py:114-129 (the most defensive of the three copies).
def get_weights(model: nn.Module) -> list[np.ndarray]:
    """All parameters as a list of float32 numpy arrays (copies), in a fixed order."""
    return [p.detach().cpu().numpy().astype(np.float32, copy=True) for p in model.parameters()]


def set_weights(model: nn.Module, weights: Sequence[np.ndarray]) -> None:
    """Load a list produced by :func:`get_weights` (e.g. the aggregated global model) into ``model``."""
    params = list(model.parameters())
    if len(weights) != len(params):
        raise ValueError(f"expected {len(params)} arrays, got {len(weights)}")
    with torch.no_grad():
        for p, w in zip(params, weights):
            w = np.asarray(w, dtype=np.float32)
            if tuple(w.shape) != tuple(p.shape):
                raise ValueError(f"shape mismatch: expected {tuple(p.shape)}, got {tuple(w.shape)}")
            p.copy_(torch.from_numpy(w.copy()).to(p.device))


def param_names(model: nn.Module) -> list[str]:
    """Parameter names in the same order as get_weights (for he.ModelSpec)."""
    return [name for name, _ in model.named_parameters()]


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA devices


def _builders() -> dict:
    # Imported lazily: models/account_takeover.py imports this module.
    from models import account_takeover, credit_card, identity
    return {"credit_card": credit_card.build_model,
            "account_takeover": account_takeover.build_model,
            "identity_fraud": identity.build_model}


def build_model(fraud_type: str, input_dim: int, seed: int | None = None) -> nn.Module:
    """The model for a fraud type. All three model files expose build_model(input_dim); only identity takes a
    seed, so seeding is done here for all of them to give identical initial weights for a given seed."""
    builders = _builders()
    if fraud_type not in builders:
        raise ValueError(f"fraud_type must be one of {sorted(builders)}, got {fraud_type!r}")
    if seed is not None:
        torch.manual_seed(seed)
    return builders[fraud_type](input_dim)
