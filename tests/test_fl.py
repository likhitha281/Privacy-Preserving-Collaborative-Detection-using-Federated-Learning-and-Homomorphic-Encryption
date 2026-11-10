import csv
import tempfile
import unittest
from unittest import mock

import numpy as np

from common.config import ExperimentConfig
from data import loader
from data.prepare import run
from data.synthetic import generate
from fl import run_federated, simulate
from fl.client import FraudClient
from fl.logging import write_rows_csv
from fl.server import ROW_KEYS
from fl.strategy import EncryptedFedAvg
from he import CKKSAggregator, CKKSClient


def tiny(mode, **kw):
    return ExperimentConfig(**{"mode": mode, "fraud_type": "credit_card", "n_clients": 3, "rounds": 2,
                               "local_epochs": 1, "seed": 7, **kw})


class TestFederated(unittest.TestCase):
    """Synthetic 3-bank federation (data/synthetic.py); no IEEE-CIS files needed."""

    @classmethod
    def setUpClass(cls):
        cls._saved = dict(loader._CFG)
        cls._tmp = tempfile.TemporaryDirectory()
        raw = f"{cls._tmp.name}/raw"
        generate(raw, n_txn=30000, n_cards=1500, seed=0)
        loader.configure(raw_dir=raw, processed_dir=f"{cls._tmp.name}/processed")
        kw = tiny("fedavg").loader_kwargs()
        run("ieee", raw, loader.federation_dir(**kw), n_banks=kw["n_banks"], partition_mode=kw["partition"],
            alpha=kw["alpha"], seed=kw["seed"], min_fraud_train=20, top_k=8)
        cls.plain = simulate(tiny("fedavg"))
        cls.enc = simulate(tiny("fedavg_he"))

    @classmethod
    def tearDownClass(cls):
        loader._CFG.update(cls._saved)
        cls._tmp.cleanup()

    def test_fedavg_completes_with_metric_columns(self):
        rows = self.plain.rows
        self.assertEqual([r["round"] for r in rows], [1, 2])
        for r in rows:
            self.assertEqual(tuple(r), ROW_KEYS)
            for k in ("test_f1", "test_roc_auc", "test_pr_auc", "test_precision", "test_recall", "test_accuracy"):
                self.assertTrue(0.0 <= r[k] <= 1.0, k)
            self.assertEqual((r["encrypt_s"], r["decrypt_s"]), (0.0, 0.0))
            self.assertIsNone(r["max_weight_err"])
            self.assertIsNone(r["bank_f1_min"])  # one global model: no per-bank spread
            self.assertTrue(0.0 < r["tuned_threshold"] < 1.0)
            self.assertGreater(r["uplink_bytes_per_client"], 0)

    def test_fedavg_he_completes_with_costs(self):
        rows = self.enc.rows
        self.assertEqual([r["round"] for r in rows], [1, 2])
        for r in rows:
            self.assertEqual(tuple(r), ROW_KEYS)
            for k in ("encrypt_s", "decrypt_s", "aggregate_s", "round_wall_s"):
                self.assertGreater(r[k], 0.0, k)
            self.assertLess(r["max_weight_err"], 1e-5)  # CKKS sum vs plaintext average of the same updates
            # ciphertext is far larger than the float32 plaintext model
            self.assertGreater(r["uplink_bytes_per_client"], 10 * self.plain.rows[0]["uplink_bytes_per_client"])

    def test_he_global_weights_match_fedavg_after_round_1(self):
        diff = max(float(np.abs(a - b).max())
                   for a, b in zip(self.plain.global_weights[0], self.enc.global_weights[0]))
        self.assertLess(diff, 1e-4)
        # sanity: training actually moved the weights, so the match is not trivial
        self.assertFalse(all(np.array_equal(a, b) for a, b in
                             zip(self.plain.global_weights[0], self.plain.global_weights[1])))

    def test_strategy_never_holds_secret_context(self):
        s = self.enc.strategy
        self.assertIsInstance(s, EncryptedFedAvg)
        self.assertIsInstance(s.aggregator, CKKSAggregator)
        self.assertFalse(s.aggregator._context.has_secret_key())
        for name, value in vars(s).items():
            self.assertNotIsInstance(value, CKKSClient, name)
        for value in vars(s.aggregator).values():
            self.assertNotIsInstance(value, CKKSClient)
        with self.assertRaises(TypeError):
            EncryptedFedAvg(3, None, object(), {})

    def test_encrypted_flag_is_explicit(self):
        s = self.enc.strategy
        self.assertEqual([s._config(r)["encrypted"] for r in (1, 2, 3)], [False, True, True])
        self.assertFalse(any(self.plain.strategy._config(r)["encrypted"] for r in (1, 2, 3)))

    def test_client_failure_aborts_he_run(self):
        original = FraudClient.fit

        def flaky(self, parameters, config):
            if self.bank_id == 1:
                raise ConnectionError("bank 1 dropped")
            return original(self, parameters, config)

        with mock.patch.object(FraudClient, "fit", flaky), self.assertRaisesRegex(RuntimeError, "all 3 banks"):
            simulate(tiny("fedavg_he", rounds=1))

    def test_local_mode_rejected(self):
        with self.assertRaises(ValueError):
            run_federated(tiny("local"))


class TestCsv(unittest.TestCase):
    def test_round_trip_and_append(self):
        rows = [{"mode": "fedavg", "round": 1, "f1": 0.5}, {"mode": "fedavg", "round": 2, "f1": 0.25}]
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/nested/dir/rows.csv"
            write_rows_csv(rows[:1], path)
            write_rows_csv(rows[1:], path)  # append: header not repeated
            with open(path, newline="") as f:
                back = list(csv.DictReader(f))
            self.assertEqual(back, [{k: str(v) for k, v in r.items()} for r in rows])
            with self.assertRaises(ValueError):
                write_rows_csv([{"other": 1}], path)


if __name__ == "__main__":
    unittest.main()
