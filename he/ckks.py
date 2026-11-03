"""Model-shaped CKKS encryption and addition-only server aggregation.

The model spec is public metadata. All banks in an experiment must use the same
spec, context and parameter semantics. CKKS scale (2**40) is distinct from the
FedAvg sample weight passed as encrypt(..., scale=n_i/N).
"""
from dataclasses import dataclass
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
import numpy as np
import tenseal as ts

SLOTS = 4096


@dataclass(frozen=True)
class ModelSpec:
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    names: tuple[str, ...]

    def __post_init__(self):
        if not self.shapes or not (len(self.shapes) == len(self.dtypes) == len(self.names)):
            raise ValueError("Spec must have equally many nonempty shapes, dtypes and names")
        if len(set(self.names)) != len(self.names) or any(not n for n in self.names):
            raise ValueError("Layer names must be unique and nonempty")
        for shape, dtype in zip(self.shapes, self.dtypes):
            if any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in shape):
                raise ValueError("Layer dimensions must be positive integers")
            if np.dtype(dtype).name not in ("float32", "float64"):
                raise ValueError("Only float32/float64 parameters are supported")

    @classmethod
    def from_weights(cls, weights, names=None):
        arrays = [np.asarray(w) for w in weights]
        return cls(tuple(a.shape for a in arrays), tuple(a.dtype.name for a in arrays),
                   tuple(names) if names is not None else tuple(f"param_{i}" for i in range(len(arrays))))

    @property
    def sizes(self):
        return tuple(math.prod(shape) for shape in self.shapes)

    @property
    def n_values(self):
        return sum(self.sizes)

    @property
    def chunk_lengths(self):
        return tuple(min(SLOTS, self.n_values - i) for i in range(0, self.n_values, SLOTS))

    def to_dict(self):
        return {"shapes": [list(s) for s in self.shapes], "dtypes": list(self.dtypes),
                "names": list(self.names)}

    @classmethod
    def from_dict(cls, data):
        return cls(tuple(tuple(s) for s in data["shapes"]),
                   tuple(data["dtypes"]), tuple(data["names"]))

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def flatten(self, weights):
        if len(weights) != len(self.shapes):
            raise ValueError("Wrong number of model arrays")
        flattened = []
        for name, w, shape, dtype in zip(self.names, weights, self.shapes, self.dtypes):
            a = np.asarray(w)
            if a.shape != shape or a.dtype.name != dtype:
                raise ValueError(f"Shape or dtype mismatch for {name}")
            if not np.isfinite(a).all():
                raise ValueError(f"Nonfinite model weights in {name}")
            flattened.append(a.astype(np.float64).reshape(-1))
        return np.concatenate(flattened)

    def unflatten(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.n_values,) or not np.isfinite(values).all():
            raise ValueError("Invalid decrypted vector")
        result, offset = [], 0
        for shape, dtype, size in zip(self.shapes, self.dtypes, self.sizes):
            with np.errstate(over="ignore"):
                a = values[offset:offset + size].astype(dtype).reshape(shape)
            if not np.isfinite(a).all():
                raise ValueError("Decrypted weights overflow their target dtype")
            result.append(a.copy())
            offset += size
        return result


def _read_vectors(context, chunks, spec):
    if not isinstance(chunks, (list, tuple)) or len(chunks) != len(spec.chunk_lengths):
        raise ValueError("Ciphertext chunk count does not match model spec")
    vectors = []
    for blob, length in zip(chunks, spec.chunk_lengths):
        if not isinstance(blob, bytes) or not blob:
            raise ValueError("Every ciphertext chunk must be nonempty bytes")
        v = ts.ckks_vector_from(context, blob)
        if v.size() != length:
            raise ValueError("Ciphertext vector length does not match model spec")
        vectors.append(v)
    return vectors


class CKKSClient:
    def __init__(self, secret_context: bytes, spec: ModelSpec, *, n_threads: int = 1):
        self._context = ts.context_from(secret_context, n_threads=n_threads)
        if not self._context.has_secret_key():
            raise ValueError("Client decryption requires a secret context")
        self.spec = spec

    def encrypt(self, weights, scale: float) -> list[bytes]:
        """Encrypt full local weights preweighted by n_i/N, exactly once."""
        if isinstance(scale, bool) or not np.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("FedAvg scale must be finite and in (0, 1]")
        flat = self.spec.flatten(weights) * float(scale)
        return [ts.ckks_vector(self._context, flat[i:i + SLOTS].tolist()).serialize()
                for i in range(0, flat.size, SLOTS)]

    def decrypt(self, chunks: list[bytes]) -> list[np.ndarray]:
        vectors = _read_vectors(self._context, chunks, self.spec)
        values = np.concatenate([np.asarray(v.decrypt(), dtype=np.float64) for v in vectors])
        return self.spec.unflatten(values)


class CKKSAggregator:
    def __init__(self, public_context: bytes, spec: ModelSpec, sample_counts: Mapping[str, int],
                 *, n_threads: int = 1):
        self._context = ts.context_from(public_context, n_threads=n_threads)
        if self._context.has_secret_key():
            raise ValueError("Server must never receive a secret context")
        self.spec = spec
        self.sample_counts = dict(sample_counts)
        if not self.sample_counts:
            raise ValueError("At least one client is required")
        if any(not isinstance(k, str) or not k for k in self.sample_counts):
            raise ValueError("Bank IDs must be nonempty strings")
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1
               for n in self.sample_counts.values()):
            raise ValueError("Sample counts must be positive integers")

    @property
    def total_samples(self):
        return sum(self.sample_counts.values())

    def scale_for(self, bank_id: str):
        return self.sample_counts[bank_id] / self.total_samples

    def aggregate(self, encrypted_updates: Mapping[str, Sequence[bytes]]) -> list[bytes]:
        """Abort missing/extra-bank rounds; sum preweighted chunks without dividing."""
        if set(encrypted_updates) != set(self.sample_counts):
            raise ValueError("Incomplete or unexpected bank set: abort and retry the round")
        summed = None
        for bank_id in self.sample_counts:
            vectors = _read_vectors(self._context, encrypted_updates[bank_id], self.spec)
            if summed is None:
                summed = vectors
            else:
                for i, vector in enumerate(vectors):
                    summed[i] += vector
        return [v.serialize() for v in summed]
