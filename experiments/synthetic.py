"""Synthetic IEEE-CIS-shaped data for tests and --synthetic runs (never for reported results)."""
from __future__ import annotations

from pathlib import Path

from data import loader
from data.prepare import run
from data.synthetic import generate


def use_synthetic(root, cfgs, n_txn: int = 30000) -> dict:
    """Generate raw synthetic CSVs under ``root``, point data.loader at them and pre-build every federation the
    configs need (with the relaxed minimums data/tests uses, since synthetic data has few frauds).
    Returns the previous loader settings; pass them to ``restore`` when done."""
    saved = dict(loader._CFG)
    root = Path(root)
    generate(root / "raw", n_txn=n_txn, n_cards=1500, seed=0)
    loader.configure(raw_dir=root / "raw", processed_dir=root / "processed")
    for kw in {tuple(sorted(c.loader_kwargs().items())) for c in cfgs}:
        kw = dict(kw)
        out = loader.federation_dir(**kw)
        if not (out / "manifest.json").exists():
            run("ieee", root / "raw", out, n_banks=kw["n_banks"], partition_mode=kw["partition"],
                alpha=kw["alpha"], seed=kw["seed"], min_fraud_train=20, top_k=8)
    return saved


def restore(saved: dict) -> None:
    loader._CFG.update(saved)
