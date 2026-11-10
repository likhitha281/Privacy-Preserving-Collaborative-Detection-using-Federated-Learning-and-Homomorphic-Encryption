import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch import nn

from common.config import ExperimentConfig
from experiments import runner, summarize
from experiments.runner import result_path, run_experiment
from experiments.synthetic import restore, use_synthetic
from fl.logging import write_rows_csv
from fl.metrics import ROW_KEYS, make_row, score_model, tuned_threshold

TIMING = {"train_s", "round_wall_s", "encrypt_s", "decrypt_s", "aggregate_s"}


def tiny(mode, **kw):
    return ExperimentConfig(**{"mode": mode, "fraud_type": "credit_card", "n_clients": 3, "rounds": 2,
                               "local_epochs": 1, "seed": 7, **kw})


class TestRunner(unittest.TestCase):
    """Synthetic 3-bank federation; no IEEE-CIS files needed."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls._saved = use_synthetic(cls._tmp.name, [tiny("local")])
        cls.out = Path(cls._tmp.name) / "results"
        cls.local = run_experiment(tiny("local"), cls.out)
        cls.fedavg = run_experiment(tiny("fedavg"), cls.out)

    @classmethod
    def tearDownClass(cls):
        restore(cls._saved)
        cls._tmp.cleanup()

    def test_local_has_same_columns_as_fedavg(self):
        self.assertEqual([tuple(r) for r in self.local], [ROW_KEYS] * 2)
        self.assertEqual([tuple(r) for r in self.fedavg], [ROW_KEYS] * 2)
        for r in self.local:
            self.assertLessEqual(r["bank_f1_min"], r["test_f1"])
            self.assertLessEqual(r["test_f1"], r["bank_f1_max"])
            self.assertIsNone(r["encrypt_s"])
            self.assertIsNone(r["max_weight_err"])
            self.assertEqual(r["uplink_bytes_per_client"], 0)
        self.assertTrue(result_path(tiny("local"), self.out).exists())

    def test_resume_skips_existing_run(self):
        with mock.patch.object(runner, "run_local", side_effect=AssertionError("should not rerun")):
            again = run_experiment(tiny("local"), self.out)
        for a, b in zip(again, self.local):
            for k in ROW_KEYS:
                if isinstance(b[k], float):
                    self.assertAlmostEqual(a[k], b[k], places=12, msg=k)
                else:
                    self.assertEqual(a[k], b[k], k)
        with self.assertRaises(ValueError):  # same file name, different rounds: refuse silently reusing it
            run_experiment(tiny("local", rounds=3), self.out)

    def test_deterministic_for_equal_seed(self):
        with tempfile.TemporaryDirectory() as d:
            for mode, ref in (("local", self.local), ("fedavg", self.fedavg)):
                rows = run_experiment(tiny(mode), d)
                for a, b in zip(rows, ref):
                    for k in set(ROW_KEYS) - TIMING:
                        if isinstance(b[k], float):
                            # Same seed gives the same run up to float32 rounding in PyTorch CPU kernels:
                            # occasionally ~1e-8 relative differences in train_loss were observed between
                            # runs in one process (no difference at all in most runs).
                            np.testing.assert_allclose(a[k], b[k], rtol=1e-6, atol=1e-7, err_msg=f"{mode} {k}")
                        else:
                            self.assertEqual(a[k], b[k], f"{mode} {k}")


class TestTunedThreshold(unittest.TestCase):
    class Identity(nn.Module):  # logit = x[:, 0], so probability = sigmoid(x)
        def forward(self, x):
            return x[:, 0]

    def test_threshold_uses_validation_only(self):
        logit = lambda p: np.log(p / (1 - p))  # noqa: E731
        # validation: positives score >= 0.8, so the F1-optimal threshold is 0.8
        p_val = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 0.9])
        y_val = np.array([0, 0, 0, 0, 1, 1])
        # test: positives score 0.3-0.5; tuning on test would pick 0.3
        p_test = np.array([0.05, 0.1, 0.3, 0.4, 0.5, 0.9])
        y_test = np.array([0, 0, 1, 1, 1, 0])
        model = self.Identity()
        X_val, X_test = logit(p_val)[:, None], logit(p_test)[:, None]
        self.assertAlmostEqual(tuned_threshold(y_test, p_test), 0.3, places=5)
        m = score_model(model, X_val, y_val, X_test, y_test)
        self.assertAlmostEqual(m["tuned_threshold"], 0.8, places=5)
        self.assertEqual(m["test_f1_tuned"], 0.0)  # val threshold misses every test positive
        flipped = score_model(model, X_val, y_val, X_test, 1 - y_test)  # other test labels, same threshold
        self.assertAlmostEqual(flipped["tuned_threshold"], m["tuned_threshold"], places=12)


class TestSummarize(unittest.TestCase):
    def test_summaries_from_two_runs(self):
        def rows(mode, f1, enc, wall):
            c = tiny(mode)
            return [make_row(c, r, test_f1=f1 - 0.1 * (2 - r), test_f1_tuned=f1 + 0.05, test_precision=0.4,
                             test_recall=0.6, test_roc_auc=0.9, test_pr_auc=0.5, encrypt_s=enc, decrypt_s=enc,
                             aggregate_s=enc, uplink_bytes_per_client=100, downlink_bytes=100, round_wall_s=wall,
                             max_weight_err=1e-8 if mode == "fedavg_he" else None) for r in (1, 2)]

        with tempfile.TemporaryDirectory() as d:
            write_rows_csv(rows("fedavg", 0.3, 0.0, 1.0), result_path(tiny("fedavg"), d))
            write_rows_csv(rows("fedavg_he", 0.32, 0.5, 2.5), result_path(tiny("fedavg_he"), d))
            t = summarize.summarize(d)
            self.assertTrue(all((Path(d) / f"summary_{n}.csv").exists() for n in ("main", "cost", "noniid")))
            main = t["main"].set_index("mode")
            self.assertAlmostEqual(main.loc["fedavg", "test_f1_mean"], 0.3)  # final round, not round 1
            self.assertAlmostEqual(main.loc["fedavg_he", "test_f1_tuned_mean"], 0.37)
            cost = t["cost"].iloc[0]
            self.assertAlmostEqual(cost["slowdown"], 2.5)
            self.assertAlmostEqual(cost["crypto_overhead"], 1.5)
            self.assertAlmostEqual(cost["fedavg_he_encrypt_s"], 0.5)
            self.assertTrue(t["noniid"].empty)
            self.assertEqual(len(pd.read_csv(Path(d) / "summary_main.csv")), 2)
            again = summarize.summarize(d)  # rerun with the (partly empty) summary files present
            self.assertEqual(len(again["main"]), 2)


if __name__ == "__main__":
    unittest.main()
