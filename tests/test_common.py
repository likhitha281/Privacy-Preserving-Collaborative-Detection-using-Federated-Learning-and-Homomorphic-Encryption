import tempfile
import unittest

import numpy as np

import models.account_takeover as account_takeover
from common import interfaces as I
from common import utils
from common.config import ExperimentConfig
from data import loader
from data.prepare import run
from data.synthetic import generate
from he import create_contexts

FRAUD_TYPES = ("credit_card", "account_takeover", "identity_fraud")


class TestConfig(unittest.TestCase):
    def test_defaults_and_loader_kwargs(self):
        cfg = ExperimentConfig(mode="fedavg", fraud_type="credit_card")
        self.assertEqual((cfg.alpha, cfg.rounds, cfg.local_epochs, cfg.seed), (0.5, 10, 2, 42))
        self.assertEqual(cfg.loader_kwargs(), dict(n_banks=5, alpha=0.5, seed=42, partition="label_skew"))

    def test_rejects_invalid_values(self):
        bad = [dict(mode="sgd"), dict(fraud_type="phishing"), dict(n_clients=2), dict(n_clients=6),
               dict(n_clients=True), dict(alpha=0), dict(alpha=float("nan")), dict(rounds=0),
               dict(local_epochs=1.5), dict(lr=-1), dict(batch_size=0), dict(seed=-1),
               dict(partition="random"), dict(results_dir="")]
        for override in bad:
            with self.subTest(**override), self.assertRaises(ValueError):
                ExperimentConfig(**{"mode": "local", "fraud_type": "credit_card", **override})

    def test_dict_round_trip_and_alpha_normalised(self):
        cfg = ExperimentConfig(mode="fedavg_he", fraud_type="identity_fraud", n_clients=3, alpha=1)
        self.assertIsInstance(cfg.alpha, float)
        self.assertEqual(ExperimentConfig.from_dict(cfg.to_dict()), cfg)
        with self.assertRaises(ValueError):
            ExperimentConfig.from_dict({**cfg.to_dict(), "typo": 1})

    def test_frozen(self):
        cfg = ExperimentConfig(mode="local", fraud_type="credit_card")
        with self.assertRaises(Exception):
            cfg.rounds = 3


class TestWeights(unittest.TestCase):
    def test_get_set_round_trip_all_models(self):
        for ft in FRAUD_TYPES:
            with self.subTest(fraud_type=ft):
                model = utils.build_model(ft, 12, seed=0)
                w = utils.get_weights(model)
                self.assertTrue(all(a.dtype == np.float32 for a in w))
                self.assertEqual(sum(a.size for a in w), utils.param_count(model))
                new = [(a * 0.5 + 1).astype(np.float64) for a in w]  # float64 input is cast on load
                utils.set_weights(model, new)
                for exp, got in zip(new, utils.get_weights(model)):
                    np.testing.assert_allclose(got, exp, rtol=0, atol=1e-6)
                with self.assertRaises(ValueError):
                    utils.set_weights(model, new[:-1])

    def test_seed_gives_identical_initial_weights(self):
        for ft in FRAUD_TYPES:
            a = utils.get_weights(utils.build_model(ft, 12, seed=3))
            b = utils.get_weights(utils.build_model(ft, 12, seed=3))
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)

    def test_account_takeover_exposes_shared_helpers(self):
        self.assertIs(account_takeover.get_weights, utils.get_weights)
        self.assertIs(account_takeover.set_weights, utils.set_weights)

    def test_unknown_fraud_type(self):
        with self.assertRaises(ValueError):
            utils.build_model("phishing", 12)


class TestTransport(unittest.TestCase):
    def test_uint8_round_trip(self):
        chunks = [b"\x00\x01\xff", bytes(range(256)) * 3, b"x"]
        arrays = I.bytes_to_ndarrays(chunks)
        self.assertTrue(all(a.dtype == np.uint8 and a.ndim == 1 for a in arrays))
        self.assertEqual(I.ndarrays_to_bytes(arrays), chunks)

    def test_rejects_non_uint8(self):
        with self.assertRaises(ValueError):
            I.ndarrays_to_bytes([np.zeros(3, np.float32)])
        with self.assertRaises(TypeError):
            I.bytes_to_ndarrays(["not bytes"])

    def test_sample_scale(self):
        self.assertAlmostEqual(I.sample_scale(25, 100), 0.25)
        for n_i, N in [(0, 10), (11, 10), (1.5, 10), (True, 10)]:
            with self.assertRaises(ValueError):
                I.sample_scale(n_i, N)


class TestSyntheticFederation(unittest.TestCase):
    """Builds a 3-bank federation from data/synthetic.py so no IEEE-CIS files are needed."""

    @classmethod
    def setUpClass(cls):
        cls._saved = dict(loader._CFG)
        cls._tmp = tempfile.TemporaryDirectory()
        raw = f"{cls._tmp.name}/raw"
        generate(raw, n_txn=30000, n_cards=1500, seed=0)
        loader.configure(raw_dir=raw, processed_dir=f"{cls._tmp.name}/processed")
        cls.cfg = ExperimentConfig(mode="fedavg_he", fraud_type="credit_card", n_clients=3, seed=7)
        kw = cls.cfg.loader_kwargs()
        run("ieee", raw, loader.federation_dir(**kw), n_banks=kw["n_banks"], partition_mode=kw["partition"],
            alpha=kw["alpha"], seed=kw["seed"], min_fraud_train=20, top_k=8)
        cls.keys = create_contexts()

    @classmethod
    def tearDownClass(cls):
        loader._CFG.update(cls._saved)
        cls._tmp.cleanup()

    def test_client_data_is_float32(self):
        for ft in FRAUD_TYPES:
            parts = I.load_client_data(0, ft, self.cfg)
            self.assertTrue(all(p.dtype == np.float32 for p in parts))
            X_te, y_te = I.load_test(ft, self.cfg)
            self.assertEqual((X_te.dtype, y_te.dtype), (np.float32, np.float32))
            self.assertEqual(X_te.shape[1], I.input_dim(ft, self.cfg))

    def test_encrypt_aggregate_decrypt_matches_weighted_average(self):
        ft = self.cfg.fraud_type
        counts = I.sample_counts(ft, self.cfg)
        self.assertEqual(len(set(counts.values())), 3, "test needs unequal sample counts")
        N = sum(counts.values())
        dim = I.input_dim(ft, self.cfg)
        locals_ = {b: utils.get_weights(I.build_model(ft, dim, seed=int(b))) for b in counts}
        spec = I.make_spec(I.build_model(ft, dim, seed=0))
        ctx = I.make_client(self.keys.secret_context, spec)
        agg = I.make_aggregator(self.keys.public_context, spec, counts)

        # Client -> Flower (uint8) -> server -> Flower (uint8) -> client.
        wire = {b: I.bytes_to_ndarrays(I.encrypt(w, I.sample_scale(counts[b], N), ctx)) for b, w in locals_.items()}
        summed = I.aggregate({int(b): I.ndarrays_to_bytes(a) for b, a in wire.items()}, agg)
        result = I.decrypt(I.ndarrays_to_bytes(I.bytes_to_ndarrays(summed)), ctx)

        for i, got in enumerate(result):
            expected = sum(counts[b] / N * locals_[b][i].astype(np.float64) for b in counts)
            self.assertEqual(got.dtype, np.float32)
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-5)
        model = I.build_model(ft, dim)
        utils.set_weights(model, result)

    def test_aggregate_aborts_on_missing_bank(self):
        ft = self.cfg.fraud_type
        counts = I.sample_counts(ft, self.cfg)
        model = I.build_model(ft, I.input_dim(ft, self.cfg), seed=0)
        spec = I.make_spec(model)
        ctx = I.make_client(self.keys.secret_context, spec)
        agg = I.make_aggregator(self.keys.public_context, spec, counts)
        chunks = I.encrypt(utils.get_weights(model), 0.5, ctx)
        with self.assertRaises(ValueError):
            I.aggregate({"0": chunks, "1": chunks}, agg)


if __name__ == "__main__":
    unittest.main()
