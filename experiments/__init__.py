"""Experiment runner, grid driver and summaries (owner: Pratham).

    python -m experiments.run_all            # full grid -> results/*.csv + results/summary_*.csv
    python -m experiments.run_all --quick    # 3 banks, 2 rounds, 1 epoch, one fraud type -> results/quick/
"""
from .runner import result_path, run_experiment

__all__ = ["run_experiment", "result_path"]
