"""Analytic checks of published mathematics; no research data or training."""

import unittest

import numpy as np

from paper_cognition import (
    normalize_age,
    observed_c,
    progressive_prediction,
    pseudo_cognitive_sample,
)


class PaperCognitionTests(unittest.TestCase):
    def test_published_age_scale_without_clipping(self):
        self.assertEqual(normalize_age(65), 0.0)
        self.assertEqual(normalize_age(80), 0.5)
        self.assertEqual(normalize_age(95), 1.0)
        self.assertAlmostEqual(normalize_age(64), -1 / 30)

    def test_mci_is_excluded_and_unknown_diagnoses_require_mapping(self):
        self.assertEqual(observed_c('CN'), 0.0)
        self.assertEqual(observed_c('NC'), 0.0)
        self.assertEqual(observed_c('AD'), 1.0)
        self.assertIsNone(observed_c('MCI'))
        with self.assertRaises(ValueError):
            observed_c('uncertain dementia')

    def test_equation4_equal_shape_weights_and_fractional_c(self):
        calls = []

        def integrate(shape, start, end, c):
            calls.append((start, end, c))
            # Analytically solvable, condition-specific constant velocities.
            velocity = np.array([1.0, 2.0]) if c == 0.0 else np.array([3.0, 4.0])
            return shape + (end - start) * velocity

        shape, c = pseudo_cognitive_sample(
            np.array([0.0, 1.0]), np.array([8.0, 9.0]), 0.0, 1.0, 0.25,
            integrate=integrate,
        )
        # Forward=[.25,1.5], backward=[5.75,6]; their unweighted mean.
        np.testing.assert_allclose(shape, [3.0, 3.75])
        self.assertEqual(c, 0.25)
        self.assertEqual(calls, [(0.0, 0.25, 0.0), (1.0, 0.25, 1.0)])

    def test_c_fraction_is_invariant_under_age_normalization(self):
        _, c = pseudo_cognitive_sample(
            np.zeros(2), np.ones(2), normalize_age(70), normalize_age(80),
            normalize_age(72.5), integrate=lambda shape, start, end, c: shape,
        )
        self.assertAlmostEqual(c, 0.25)

    def test_progressive_c_is_estimated_from_each_new_voxel_shape(self):
        conditions, converted, times = [], [], []

        def voxelize(shape):
            converted.append(shape.copy())
            return shape * 10

        def integrate(shape, start, end, c):
            conditions.append(c)
            times.append((start, end))
            return shape + (end - start) * 3 * c

        initial = np.array([0.1])
        result = progressive_prediction(
            initial, 0.0, 1.0, steps=3, integrate=integrate,
            shape_to_voxels=voxelize, estimate_c=lambda voxels: voxels[0] / 10,
        )
        # Each segment doubles the scalar shape; holding initial c gives .4.
        np.testing.assert_allclose(result, [0.8])
        np.testing.assert_allclose(conditions, [0.1, 0.2, 0.4])
        np.testing.assert_allclose(np.concatenate(converted), [0.1, 0.2, 0.4])
        np.testing.assert_allclose(times, [[0, 1/3], [1/3, 2/3], [2/3, 1]])
        np.testing.assert_allclose(initial, [0.1])

    def test_invalid_conversion_intervals_are_not_silently_relabelled(self):
        for start, end, sample in [(1, 1, 1), (1, 0, 0.5), (0, 1, 1.5)]:
            with self.subTest(times=(start, end, sample)):
                with self.assertRaises(ValueError):
                    pseudo_cognitive_sample(
                        np.zeros(2), np.ones(2), start, end, sample,
                        integrate=lambda shape, start, end, c: shape,
                    )

    def test_invalid_estimator_outputs_are_not_clipped(self):
        for c in [-0.1, 1.1, float('nan')]:
            with self.subTest(c=c):
                with self.assertRaises(ValueError):
                    progressive_prediction(
                        np.zeros(2), 0, 1, steps=1,
                        integrate=lambda shape, start, end, c: shape,
                        shape_to_voxels=lambda shape: shape,
                        estimate_c=lambda voxels: c,
                    )


if __name__ == '__main__':
    unittest.main()
