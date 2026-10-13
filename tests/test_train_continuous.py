"""Leakage and endpoint-relative labeling checks for continuous-c training."""

import unittest
import numpy as np
from train_continuous import augmented_trajectories, conversion_pairs


class ContinuousTrainingTests(unittest.TestCase):
    def data(self):
        return dict(pids=np.array(['a', 'a', 'a', 'a', 'b', 'b']),
                    splits=np.array(['train']*4+['test']*2),
                    times=np.array([0., 0., .5, 1., 0., 1.]),
                    conditions=np.array([0., 0., 0., 1., 0., 1.]),
                    coefficients=np.arange(12).reshape(6, 2))

    def test_all_train_endpoint_pairs_and_same_time_repeats_retained(self):
        self.assertEqual(conversion_pairs(self.data()), [('a', 0, 3), ('a', 1, 3), ('a', 2, 3)])

    def test_overlapping_endpoint_relative_labels_are_not_merged(self):
        data = self.data()
        pseudo = dict(pair_indices=np.array([0, 1]), coefficients=np.array([[1., 2.], [3., 4.]]),
                      times=np.array([.75, .75]), conditions=np.array([.75, .5]))
        manifest = dict(records=[dict(pid='a', source_index=0, target_index=3),
                                 dict(pid='a', source_index=2, target_index=3)])
        trajectories = augmented_trajectories(data, pseudo, manifest)['a']
        self.assertEqual(len(trajectories), 2)
        np.testing.assert_allclose(trajectories[0][2], [0, .75, 1])
        np.testing.assert_allclose(trajectories[1][2], [0, .5, 1])
        np.testing.assert_allclose(trajectories[0][0][[0, 2]], data['coefficients'][[0, 3]])

    def test_held_out_endpoint_is_rejected(self):
        data = self.data()
        pseudo = dict(pair_indices=np.array([0]), coefficients=np.array([[1., 2.]]),
                      times=np.array([.5]), conditions=np.array([.5]))
        with self.assertRaises(ValueError):
            augmented_trajectories(data, pseudo, dict(records=[dict(pid='b', source_index=4, target_index=5)]))


if __name__ == '__main__':
    unittest.main()
