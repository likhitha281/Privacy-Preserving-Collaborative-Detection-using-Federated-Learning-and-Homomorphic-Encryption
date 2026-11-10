"""Attack hooks and experiments on a tiny synthetic federation (no IEEE-CIS files needed)."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import interfaces as I
from common.config import ExperimentConfig
from common.utils import get_weights, set_seed
from experiments.attacks import gradient_leakage as gl
from experiments.attacks import label_flip as lf
from experiments.attacks.common import Recorder
from experiments.synthetic import restore, use_synthetic
from fl import simulate
from fl.client import train_model

TIMING = {"train_s", "round_wall_s", "encrypt_s", "decrypt_s", "aggregate_s"}
GOLDEN = Path(__file__).parent / "fixtures" / "hooks_golden.json"


def tiny(mode, **kw):
    return ExperimentConfig(**{"mode": mode, "fraud_type": "credit_card", "n_clients": 3, "rounds": 2,
                               "local_epochs": 1, "seed": 7, **kw})


def strip(rows, drop=TIMING):
    return [{k: v for k, v in r.items() if k not in drop} for r in rows]


class SingleThread:
    """PyTorch CPU kernels are bitwise reproducible only single-threaded (multi-threaded reductions differ at
    ~1e-10 between identical runs), so exact-equality comparisons run with one thread."""

    def __enter__(self):
        self.n = torch.get_num_threads()
        torch.set_num_threads(1)

    def __exit__(self, *exc):
        torch.set_num_threads(self.n)


class TestAttackHooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls._saved = use_synthetic(cls._tmp.name, [tiny("fedavg")])
        with SingleThread():
            cls.plain = simulate(tiny("fedavg"))
            cls.plain_rec = Recorder()
            cls.plain_hooked = simulate(tiny("fedavg"), client_label_transform=lambda b: None,
                                        server_observer=cls.plain_rec)
            cls.enc_rec = Recorder()
            cls.enc = simulate(tiny("fedavg_he"), server_observer=cls.enc_rec)

    @classmethod
    def tearDownClass(cls):
        restore(cls._saved)
        cls._tmp.cleanup()

    def test_hooks_none_reproduces_pre_hook_output(self):
        """tests/fixtures/hooks_golden.json was produced by the code before the hooks were added (same synthetic
        data, seed 7, one thread). fedavg must match it exactly, including the global weights."""
        golden = json.loads(GOLDEN.read_text())
        self.assertEqual(strip(self.plain.rows), golden["fedavg"]["rows"])
        sha = [hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in w)).hexdigest()
               for w in self.plain.global_weights]
        self.assertEqual(sha, golden["fedavg"]["weights_sha256"])
        # fedavg_he cannot be bitwise reproducible: every encryption uses fresh randomness, so the decrypted
        # global model differs from the golden run by CKKS noise (~1e-8). Continuous metrics must agree closely;
        # thresholded ones (F1, precision, recall, accuracy, tuned threshold) can flip on a single borderline
        # prediction and are only checked for structure.
        continuous = ("test_roc_auc", "test_pr_auc", "test_loss", "val_loss", "train_loss")
        he_golden = golden["fedavg_he"]["rows"]
        self.assertEqual(len(self.enc.rows), len(he_golden))
        for a, b in zip(self.enc.rows, he_golden):
            self.assertEqual(set(b) | TIMING, set(a))
            for k in continuous:
                np.testing.assert_allclose(a[k], b[k], rtol=1e-4, err_msg=k)
            for k in ("mode", "round", "n_clients", "seed"):
                self.assertEqual(a[k], b[k], k)

    def test_noop_hooks_identical_to_no_hooks(self):
        self.assertEqual(strip(self.plain_hooked.rows), strip(self.plain.rows))
        for wa, wb in zip(self.plain_hooked.global_weights, self.plain.global_weights):
            for a, b in zip(wa, wb):
                np.testing.assert_array_equal(a, b)

    def test_observer_sees_float_in_plain_and_uint8_ciphertext_in_he(self):
        for rec, rounds in ((self.plain_rec, 2), (self.enc_rec, 2)):
            self.assertEqual([o.server_round for o in rec.rounds], list(range(1, rounds + 1)))
            for o in rec.rounds:
                self.assertEqual([u.bank_id for u in o.uploads], [0, 1, 2])
                self.assertTrue(all(u.num_examples > 0 for u in o.uploads))
        for o in self.plain_rec.rounds:
            for u in o.uploads:
                self.assertTrue(all(a.dtype == np.float32 for a in u.arrays))
        for o in self.enc_rec.rounds:
            for u in o.uploads:
                self.assertTrue(all(a.dtype == np.uint8 and a.ndim == 1 for a in u.arrays))
        self.assertEqual(self.enc_rec.rounds[0].broadcast[0].dtype, np.float32)  # round 1: plaintext init
        self.assertEqual(self.enc_rec.rounds[1].broadcast[0].dtype, np.uint8)    # then: encrypted aggregate
        self.assertIsNotNone(lf.distance_screen(self.plain_rec.rounds[0]))
        self.assertIsNone(lf.distance_screen(self.enc_rec.rounds[0]))

    def test_plain_observed_weights_average_to_global(self):
        """fedavg uploads are exactly the clients' trained weights, i.e. what FedAvg averages."""
        o = self.plain_rec.rounds[-1]
        N = sum(u.num_examples for u in o.uploads)
        avg = [sum(u.num_examples / N * u.arrays[i].astype(np.float64) for u in o.uploads)
               for i in range(len(o.uploads[0].arrays))]
        for a, g in zip(avg, self.plain.global_weights[-1]):
            np.testing.assert_allclose(a, g, atol=1e-6)

    def test_label_transform_must_keep_shape(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            simulate(tiny("fedavg", rounds=1), client_label_transform=lambda b: (lambda y: y[:-1]))


class TestLabelFlip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls._saved = use_synthetic(cls._tmp.name, [tiny("fedavg")])

    @classmethod
    def tearDownClass(cls):
        restore(cls._saved)
        cls._tmp.cleanup()

    def test_flip_labels(self):
        y = np.array([0, 0, 1, 1, 0, 0, 0, 1], dtype=np.float32)
        np.testing.assert_array_equal(lf.flip_labels(y, 0.0, 1), y)
        np.testing.assert_array_equal(lf.flip_labels(y, 1.0, 1), 1 - y)
        self.assertEqual(int((lf.flip_labels(y, 0.5, 1) != y).sum()), 4)
        np.testing.assert_array_equal(lf.flip_labels(y, 0.5, 1), lf.flip_labels(y, 0.5, 1))

    def test_f0_equals_clean_and_full_flip_lowers_f1(self):
        cfg = tiny("fedavg", rounds=3)
        with SingleThread():
            clean = simulate(cfg).rows
            f0 = lf.run_one(cfg, 0.0)
            f1 = lf.run_one(cfg, 1.0)
        for a, b in zip(f0, clean):
            for k in lf.METRICS:
                self.assertEqual(a[k], b[k], k)
        rows = lf.add_drops(f0 + f1)
        self.assertEqual([r["drop_f1"] for r in rows if r["flip_frac"] == 0.0], [0.0] * 3)
        self.assertLess(f1[-1]["test_f1"], clean[-1]["test_f1"])
        self.assertGreater(rows[-1]["drop_f1"], 0)
        self.assertTrue(all(r["screen_available"] for r in f1))
        counts = I.sample_counts("credit_card", cfg)  # real partition sizes
        self.assertEqual(f1[0]["attacker_n"], counts["0"])
        self.assertAlmostEqual(f1[0]["attacker_share"], counts["0"] / sum(counts.values()))

    def test_he_rows_record_limitation(self):
        rows = lf.run_one(tiny("fedavg_he", rounds=1), 0.5)
        self.assertFalse(rows[0]["screen_available"])
        self.assertIsNone(rows[0]["attacker_flagged"])
        self.assertEqual(rows[0]["upload_dtype"], "uint8")
        summary = lf.summarize(lf.add_drops(rows))
        self.assertIn("limitation", set(summary["row_type"]))


class TinyMLP(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TestGradientInversion(unittest.TestCase):
    def test_recovers_batch_size_1_input_better_than_mean_baseline(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(2000, 8)).astype(np.float32)
        prior = gl.Prior.from_data(X)
        wins = 0
        for trial in range(3):
            set_seed(trial)
            model = TinyMLP()
            w0 = get_weights(model)
            x, y = X[[trial]], np.array([0.0], dtype=np.float32)
            train_model(model, x, y, 1.0, 1, 1e-3, 1)  # the real client step: one Adam step on one row
            X_rec, _ = gl.invert_update(TinyMLP(), w0, get_weights(model), n_rec=1, lr=1e-3, pos_weight=1.0,
                                        prior=prior, iters=300, seed=trial)
            rn, rc = gl.match_metrics(X_rec, x, prior)
            bn, bc = gl.baseline_metrics(1, x, prior)
            wins += rn < bn and rc > bc
        self.assertGreaterEqual(wins, 2)

    def test_ciphertext_bytes_type_check_only_by_cast_and_reshape(self):
        shapes = [(3, 2), (3,)]
        cipher = [np.arange(5, dtype=np.uint8), np.arange(7, dtype=np.uint8)]
        out = gl.as_model_arrays(cipher, shapes)
        self.assertEqual([a.shape for a in out], shapes)
        self.assertTrue(all(a.dtype == np.float32 for a in out))
        w = [np.ones((3, 2), np.float32), np.zeros(3, np.float32)]
        for a, b in zip(gl.as_model_arrays(w, shapes), w):
            np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            gl.as_model_arrays([np.arange(4, dtype=np.uint8)], shapes)

    def test_metrics_perfect_reconstruction(self):
        X = np.random.default_rng(1).normal(size=(50, 6))
        prior = gl.Prior.from_data(X)
        n, r = gl.match_metrics(X[[3, 1]], X[:4], prior)
        self.assertAlmostEqual(n, 0.0, places=9)
        self.assertAlmostEqual(r, 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
