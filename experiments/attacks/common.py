"""Helpers shared by the attack scripts: the observer that models the curious server, and output paths."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fl.strategy import RoundObservation

DEFAULT_DIR = Path("results") / "attacks"


class Recorder:
    """A fl server_observer that keeps every RoundObservation (what the server received, per round)."""

    def __init__(self):
        self.rounds: list[RoundObservation] = []

    def __call__(self, obs: RoundObservation) -> None:
        self.rounds.append(obs)


def is_plaintext(arrays) -> bool:
    """True if the arrays are interpretable as model weights (float), False for uint8 ciphertext chunks."""
    return all(np.issubdtype(np.asarray(a).dtype, np.floating) for a in arrays)


def flat(arrays) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])


def out_dir(results_dir, quick: bool) -> Path:
    """results/attacks by default; quick runs go to <dir>/quick so they never mix with reported results."""
    d = Path(results_dir) if results_dir is not None else DEFAULT_DIR
    return d / "quick" if quick and results_dir is None else d
