"""Focused tests for gallery-centering and part-wise silhouette normalization."""

import unittest

import numpy as np

from silhouette_embedding import center_and_normalize_embedding, center_and_normalize_gallery


class SilhouetteEmbeddingTests(unittest.TestCase):
    def test_gallery_and_probe_use_the_same_partwise_centering(self):
        gallery = np.zeros((2, 4096), dtype=np.float32)
        gallery[0, :256] = 1.0
        gallery[1, :256] = 3.0
        gallery[0, 256:] = 2.0
        gallery[1, 256:] = 6.0

        normalized, mean = center_and_normalize_gallery(gallery)
        probe = center_and_normalize_embedding(gallery[0], mean)

        parts = normalized.reshape(2, 16, 256)
        probe_parts = probe.reshape(16, 256)
        np.testing.assert_allclose(np.linalg.norm(parts[0, 0]), 1.0)
        np.testing.assert_allclose(np.linalg.norm(parts[0, 1]), 1.0)
        np.testing.assert_allclose(probe_parts, parts[0])

    def test_constant_parts_become_zero_without_nan(self):
        gallery = np.ones((2, 4096), dtype=np.float32)
        normalized, mean = center_and_normalize_gallery(gallery)

        self.assertTrue(np.isfinite(normalized).all())
        np.testing.assert_array_equal(normalized, 0.0)
        np.testing.assert_array_equal(center_and_normalize_embedding(gallery[0], mean), 0.0)


if __name__ == "__main__":
    unittest.main()
