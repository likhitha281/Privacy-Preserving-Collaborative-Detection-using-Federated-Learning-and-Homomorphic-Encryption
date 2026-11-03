import unittest
import numpy as np
import tenseal as ts
from he import CKKSClient, CKKSAggregator, ModelSpec, create_contexts


class TestEncryptedAggregation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.keys = create_contexts()

    def setUp(self):
        rng = np.random.default_rng(1)
        # Crosses the 4096-slot boundary, includes bias and scalar parameters.
        self.weights = [rng.normal(size=(65, 70)).astype(np.float32),
                        np.array([-2.0, 0, 3.0], np.float32), np.array(0.5, np.float64)]
        self.spec = ModelSpec.from_weights(self.weights, ["layer.weight", "layer.bias", "scalar"])
        self.client = CKKSClient(self.keys.secret_context, self.spec)
        self.server = CKKSAggregator(self.keys.public_context, self.spec, {"a": 1, "b": 3})

    def test_roundtrip_preserves_shapes_dtypes_values(self):
        result = self.client.decrypt(self.client.encrypt(self.weights, 1.0))
        for expected, actual in zip(self.weights, result):
            self.assertEqual(expected.shape, actual.shape)
            self.assertEqual(expected.dtype, actual.dtype)
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)

    def test_unequal_sample_weighted_aggregate(self):
        second = [(w * -0.4).astype(w.dtype) for w in self.weights]
        chunks = self.server.aggregate({
            "a": self.client.encrypt(self.weights, 0.25),
            "b": self.client.encrypt(second, 0.75),
        })
        result = self.spec.flatten(self.client.decrypt(chunks))
        expected = self.spec.flatten(self.weights) * .25 + self.spec.flatten(second) * .75
        np.testing.assert_allclose(result, expected, atol=1e-5, rtol=0)

    def test_public_server_cannot_decrypt(self):
        public = ts.context_from(self.keys.public_context, n_threads=1)
        self.assertFalse(public.has_secret_key())
        blob = self.client.encrypt(self.weights, 1.0)[0]
        with self.assertRaises(Exception):
            ts.ckks_vector_from(public, blob).decrypt()

    def test_reject_secret_key_on_server(self):
        with self.assertRaises(ValueError):
            CKKSAggregator(self.keys.secret_context, self.spec, {"a": 1})

    def test_abort_missing_or_extra_bank(self):
        chunks = self.client.encrypt(self.weights, 0.25)
        for updates in [{"a": chunks}, {"a": chunks, "b": chunks, "extra": chunks}]:
            with self.assertRaises(ValueError):
                self.server.aggregate(updates)

    def test_reject_invalid_weights_and_scales(self):
        for scale in [0, -1, 2, np.nan, np.inf, True]:
            with self.assertRaises(ValueError):
                self.client.encrypt(self.weights, scale)
        wrong = [self.weights[0].astype(np.float64), *self.weights[1:]]
        with self.assertRaises(ValueError):
            self.client.encrypt(wrong, 1.0)
        bad = [w.copy() for w in self.weights]
        bad[0][0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.client.encrypt(bad, 1.0)

    def test_reject_wrong_chunk_layout(self):
        chunks = self.client.encrypt(self.weights, 1.0)
        with self.assertRaises(ValueError):
            self.client.decrypt(chunks[:-1])
        with self.assertRaises(ValueError):
            self.client.decrypt(list(reversed(chunks)))

    def test_spec_serialization_and_integer_buffer_rejection(self):
        self.assertEqual(self.spec, ModelSpec.from_dict(self.spec.to_dict()))
        self.assertEqual(self.spec.fingerprint, ModelSpec.from_dict(self.spec.to_dict()).fingerprint)
        with self.assertRaises(ValueError):
            ModelSpec.from_weights([np.zeros(3, dtype=np.int64)])

    def test_slot_boundary_sizes(self):
        for size in [4095, 4096, 4097, 8192]:
            weights = [np.full(size, -0.125, np.float32)]
            client = CKKSClient(self.keys.secret_context, ModelSpec.from_weights(weights))
            chunks = client.encrypt(weights, 1.0)
            self.assertEqual(len(chunks), (size + 4095) // 4096)
            np.testing.assert_allclose(client.decrypt(chunks)[0], weights[0], atol=1e-5, rtol=0)


if __name__ == "__main__":
    unittest.main()
