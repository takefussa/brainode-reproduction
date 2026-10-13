import unittest

import numpy as np
import torch
from torch import nn

from paper_training import optimizer_batch_step, scale_pca_shapes, trajectory_loss


class ConstantVelocity(nn.Module):
    def __init__(self, value=2.):
        super().__init__()
        self.velocity = nn.Parameter(torch.tensor(value))

    def forward(self, t, shape, condition):
        return torch.ones_like(shape) * self.velocity


class PaperTrainingTests(unittest.TestCase):
    def test_all_forward_backward_pairs_keep_same_age_observations(self):
        times = torch.tensor([.2, .4, .4, .6])
        shapes = (times * 2.)[:, None]
        loss, pairs = trajectory_loss(ConstantVelocity(), shapes, times, torch.zeros(4))
        self.assertEqual(pairs, 10)  # Same-age pair is excluded, all other observations remain.
        self.assertLess(float(loss), 1e-12)

    def test_vertex_scaling_includes_pca_mean(self):
        mean = np.array([2., -3., 5.])
        components = np.eye(3)[:2]
        vertex = np.array([4., 7., 5.])
        coefficients = torch.tensor((vertex - mean) @ components.T)
        predicted = scale_pca_shapes(coefficients, mean, components, 1.02)
        np.testing.assert_allclose(predicted.numpy(), (vertex * 1.02 - mean) @ components.T)

    def test_both_subjects_and_last_short_batch_update_parameters(self):
        parameter = nn.Parameter(torch.tensor(0.))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        optimizer_batch_step(optimizer, [(parameter - 1.).square(), (parameter - 3.).square()])
        self.assertAlmostEqual(float(parameter), .4, places=6)
        optimizer_batch_step(optimizer, [(parameter - 1.).square()])
        self.assertAlmostEqual(float(parameter), .52, places=6)


if __name__ == '__main__':
    unittest.main()
