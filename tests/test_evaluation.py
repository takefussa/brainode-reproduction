import unittest

import numpy as np

from evaluation import evaluate_pca_prediction, shape_errors


class OneCoordinateProjection:
    """Analytic PCA-like map that discards two coordinates for one vertex."""

    def transform(self, vertices):
        return vertices[:, :1]

    def inverse_transform(self, coefficients):
        return np.column_stack([coefficients[:, 0], np.zeros((len(coefficients), 2))])


class EvaluationTests(unittest.TestCase):
    def test_euclidean_distance_is_not_coordinate_mae(self):
        observed = np.array([[0., 0., 0.], [1., 1., 1.]])
        errors = shape_errors(observed + [3., 4., 0.], observed)
        self.assertEqual(errors['mean_vertex_euclidean'], 5.)
        self.assertAlmostEqual(errors['coordinate_mae'], 7 / 3)
        np.testing.assert_array_equal(errors['per_vertex_euclidean'], [5., 5.])

    def test_pca_target_can_hide_error_to_the_actual_observation(self):
        results = evaluate_pca_prediction(
            OneCoordinateProjection(), np.array([2.]),
            np.array([[2., 3., 4.]]), np.array([[1., 3., 4.]]))
        self.assertEqual(results['pca_target_mean_vertex_euclidean'], 0.)
        self.assertEqual(results['public_coordinate_mae_to_pca_target'], 0.)
        self.assertEqual(results['prediction_mean_vertex_euclidean'], 5.)
        self.assertEqual(results['pca_reconstruction_mean_vertex_euclidean'], 5.)
        self.assertEqual(results['static_source_mean_vertex_euclidean'], 1.)

    def test_integer_coordinates_do_not_overflow_in_subtraction(self):
        results = shape_errors(np.array([[0, 0, 0]], dtype=np.uint8),
                               np.array([[255, 0, 0]], dtype=np.uint8))
        self.assertEqual(results['mean_vertex_euclidean'], 255.)

    def test_invalid_geometry_is_not_silently_scored(self):
        for left, right in [(np.zeros((1, 3)), np.zeros((2, 3))),
                            (np.zeros((0, 3)), np.zeros((0, 3))),
                            (np.array([[np.nan, 0., 0.]]), np.zeros((1, 3)))]:
            with self.subTest(shapes=(left.shape, right.shape)):
                with self.assertRaises(ValueError):
                    shape_errors(left, right)


if __name__ == '__main__':
    unittest.main()
