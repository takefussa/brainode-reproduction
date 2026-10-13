"""Analytic boundary and geometry checks, without research-data fitting."""

import itertools
import unittest

import numpy as np

from mesh_input import boundary_geometry, boundary_points


class MeshInputTests(unittest.TestCase):
    def test_single_voxel_has_eight_half_grid_boundary_corners(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True
        expected = np.array(list(itertools.product([1.5, 2.5], repeat=3)))
        np.testing.assert_array_equal(boundary_points(mask), expected)

    def test_world_geometry_preserves_rotation_spacing_and_translation(self):
        mask = np.zeros((6, 6, 6), dtype=bool)
        mask[2:4, 2:4, 2:4] = True
        affine = np.array([[0, -1, 0, -10], [2, 0, 0, 3], [0, 0, 1, 5], [0, 0, 0, 1]])
        geometry = boundary_geometry(mask, affine)
        self.assertEqual(len(geometry['boundary_voxel']), 26)
        np.testing.assert_allclose(geometry['centroid_voxel'], [2.5, 2.5, 2.5])
        np.testing.assert_allclose(geometry['centroid_header_world'], [-12.5, 8, 7.5])
        np.testing.assert_allclose(np.ptp(geometry['boundary_header_world'], axis=0), [2, 4, 2])
        shifted = affine.copy()
        shifted[:3, 3] += [20, -13, 8]
        translated = boundary_geometry(mask, shifted)
        np.testing.assert_allclose(
            geometry['target_centered_header_world_units'],
            translated['target_centered_header_world_units'],
        )

    def test_invalid_masks_and_affines_are_rejected(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        with self.assertRaises(ValueError):
            boundary_points(mask)
        mask[2, 2, 2] = True
        with self.assertRaises(ValueError):
            boundary_points(mask.astype(np.uint8))
        with self.assertRaises(ValueError):
            boundary_geometry(mask, np.zeros((4, 4)))


if __name__ == '__main__':
    unittest.main()
