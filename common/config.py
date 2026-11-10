"""One frozen config object per experiment; every caller derives loader arguments from it."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from data.loader import VIEWS
from data.partition import MODES as PARTITIONS

MODES = ("local", "fedavg", "fedavg_he")
FRAUD_TYPES = tuple(VIEWS)  # ("credit_card", "account_takeover", "identity_fraud")
MIN_CLIENTS, MAX_CLIENTS = 3, 5


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    fraud_type: str
    n_clients: int = 5
    alpha: float = 0.5
    rounds: int = 10
    local_epochs: int = 2
    seed: int = 42
    lr: float = 1e-3
    batch_size: int = 512
    results_dir: str = "results"
    partition: str = "label_skew"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.fraud_type not in FRAUD_TYPES:
            raise ValueError(f"fraud_type must be one of {FRAUD_TYPES}, got {self.fraud_type!r}")
        if self.partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {PARTITIONS}, got {self.partition!r}")
        if not _is_int(self.n_clients) or not MIN_CLIENTS <= self.n_clients <= MAX_CLIENTS:
            raise ValueError(f"n_clients must be an int in {MIN_CLIENTS}..{MAX_CLIENTS}, got {self.n_clients!r}")
        for name in ("rounds", "local_epochs", "batch_size"):
            v = getattr(self, name)
            if not _is_int(v) or v < 1:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        if not _is_int(self.seed) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative int, got {self.seed!r}")
        for name in ("alpha", "lr"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0 or v == float("inf"):
                raise ValueError(f"{name} must be a finite positive number, got {v!r}")
            # Normalise to float: the loader's cache folder name embeds alpha (a1 vs a1.0 differ).
            object.__setattr__(self, name, float(v))
        if not isinstance(self.results_dir, str) or not self.results_dir:
            raise ValueError("results_dir must be a nonempty string")

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    def loader_kwargs(self) -> dict:
        """The only place loader arguments are built; pass as **cfg.loader_kwargs() to data.loader."""
        return dict(n_banks=self.n_clients, alpha=self.alpha, seed=self.seed, partition=self.partition)
