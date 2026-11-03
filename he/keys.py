"""One-time trusted setup; the server must receive only public_context bytes."""
from dataclasses import dataclass
import tenseal as ts


@dataclass(frozen=True)
class KeyMaterial:
    # Never put this object or secret_context in a server config or client message.
    secret_context: bytes
    public_context: bytes


def create_contexts(*, n_threads: int = 1) -> KeyMaterial:
    """Generate one shared key pair for an experiment (not one per client/round)."""
    if isinstance(n_threads, bool) or not isinstance(n_threads, int) or n_threads < 1:
        raise ValueError("n_threads must be a positive integer")
    context = ts.context(
        ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60], n_threads=n_threads,
    )
    context.global_scale = 2**40
    # Addition needs neither rotations nor ciphertext multiplication.
    secret = context.serialize(
        save_public_key=True, save_secret_key=True,
        save_galois_keys=False, save_relin_keys=False,
    )
    public = context.serialize(
        save_public_key=True, save_secret_key=False,
        save_galois_keys=False, save_relin_keys=False,
    )
    server_context = ts.context_from(public, n_threads=n_threads)
    if server_context.has_secret_key():
        raise RuntimeError("Public context unexpectedly contains a secret key")
    return KeyMaterial(secret, public)
