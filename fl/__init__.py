"""Federated learning (Flower): run_federated(cfg) for "fedavg" and "fedavg_he"."""
from .server import run_federated, simulate

__all__ = ["run_federated", "simulate"]
