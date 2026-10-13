"""Verify soft labels, physical geometry and connection to the original ODE."""

import unittest

import numpy as np
import torch
from torch import nn
import trimesh

from cognition import (
    CognitionEstimator3D,
    PCAShapeVoxelizer,
    RK4ShapeIntegrator,
    cognition_loss,
    train_cognition_estimator,
)
from model import ODEFuncWithAttention
from paper_cognition import progressive_prediction, pseudo_cognitive_sample


class IdentityPCA:
    def inverse_transform(self, coefficients):
        return coefficients


class ConstantVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, t, shape, c):
        return torch.ones_like(shape) * (2 + c[0] + self.anchor * 0)


class CognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_soft_target_gradient_and_training_accept_fractional_labels(self):
        torch.manual_seed(42)
        estimator = CognitionEstimator3D()
        for parameter in estimator.parameters():
            nn.init.zeros_(parameter)
        voxels = torch.zeros(1, 1, 8, 8, 8)
        target = torch.tensor([0.25])
        loss = cognition_loss(estimator, voxels, target)
        loss.backward()
        # BCE derivative wrt bias is sigmoid(0)-.25=.25, not a binary label.
        torch.testing.assert_close(estimator.head.bias.grad, torch.tensor([0.25]))
        optimizer = torch.optim.SGD(estimator.parameters(), lr=0.1)
        history = train_cognition_estimator(
            estimator, [(voxels, target)], epochs=2, optimizer=optimizer,
        )
        self.assertLess(history[1], history[0])
        self.assertTrue(0 < estimator.estimate_c(voxels) < 0.5)

    def test_cnn_output_range_and_real_backpropagation(self):
        torch.manual_seed(42)
        estimator = CognitionEstimator3D()
        voxels = torch.rand(3, 1, 8, 8, 8)
        prediction = estimator(voxels)
        self.assertEqual(tuple(prediction.shape), (3,))
        self.assertTrue(torch.all((prediction > 0) & (prediction < 1)))
        cognition_loss(estimator, voxels, [0.0, 0.35, 1.0]).backward()
        for parameter in estimator.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_voxelization_preserves_fixed_mm_lattice_and_fills_interior(self):
        mesh = trimesh.creation.box(extents=[4, 4, 4])
        voxelize = PCAShapeVoxelizer(
            IdentityPCA(), mesh.faces, grid_shape=(12, 12, 12),
            origin_mm=(-6, -6, -6), pitch_mm=1,
        )
        centered = voxelize(mesh.vertices.reshape(-1))
        shifted = voxelize((mesh.vertices + [1, 0, 0]).reshape(-1))
        self.assertEqual(centered.shape, (1, 1, 12, 12, 12))
        self.assertEqual(centered[0, 0, 6, 6, 6], 1)
        self.assertEqual(centered.sum(), shifted.sum())
        np.testing.assert_array_equal(shifted, np.roll(centered, 1, axis=2))
        self.assertEqual(set(np.unique(centered)), {0, 1})

    def test_voxelization_refuses_crop_and_open_surface(self):
        mesh = trimesh.creation.box(extents=[4, 4, 4])
        for faces, origin in [(mesh.faces, (0, 0, 0)), (mesh.faces[:-1], (-6, -6, -6))]:
            with self.subTest(origin=origin, faces=len(faces)):
                voxelize = PCAShapeVoxelizer(
                    IdentityPCA(), faces, grid_shape=(12, 12, 12),
                    origin_mm=origin, pitch_mm=1,
                )
                with self.assertRaises(ValueError):
                    voxelize(mesh.vertices.reshape(-1))

    def test_rk4_backward_and_pseudo_c_are_continuous(self):
        integrate = RK4ShapeIntegrator(ConstantVelocity())
        result = integrate(torch.zeros(2), 1, 0, 0.25)
        torch.testing.assert_close(result, torch.full((2,), -2.25))
        shape, c = pseudo_cognitive_sample(
            torch.zeros(2), torch.full((2,), 4.0), 0, 1, 0.25,
            integrate=integrate,
        )
        torch.testing.assert_close(shape, torch.full((2,), 1.125))
        self.assertEqual(c, 0.25)

    def test_center_occupancy_matches_analytic_box_without_surface_dilation(self):
        # Faces deliberately lie between voxel centers, avoiding boundary ties.
        mesh=trimesh.creation.box(extents=[2,2,2]);mesh.apply_translation([1.25,1.25,1.25])
        voxelize=PCAShapeVoxelizer(IdentityPCA(),mesh.faces,grid_shape=(8,8,8),
            origin_mm=(-2,-2,-2),pitch_mm=1,method='center-occupancy')
        occupied=voxelize(mesh.vertices.ravel())[0,0]
        expected=np.zeros((8,8,8),dtype=np.float32);expected[3:5,3:5,3:5]=1
        np.testing.assert_array_equal(occupied,expected)
        shifted=voxelize((mesh.vertices+[1,0,0]).ravel())[0,0]
        np.testing.assert_array_equal(shifted,np.roll(expected,1,axis=0))
        with self.assertRaises(ValueError):
            voxelize((mesh.vertices+[20,0,0]).ravel())

    def test_progressive_prediction_connects_voxel_cnn_and_actual_rk4(self):
        estimator = CognitionEstimator3D()
        for parameter in estimator.parameters():
            nn.init.zeros_(parameter)
        result = progressive_prediction(
            torch.zeros(2), 0, 1, steps=3,
            integrate=RK4ShapeIntegrator(ConstantVelocity()),
            shape_to_voxels=lambda shape: np.zeros((1, 1, 8, 8, 8), np.float32),
            estimate_c=estimator.estimate_c,
        )
        # Untrained zero-logit CNN returns .5; analytic velocity is 2.5.
        torch.testing.assert_close(result, torch.full((2,), 2.5))

    def test_original_brainode_attention_model_accepts_fractional_c(self):
        torch.manual_seed(42)
        model = ODEFuncWithAttention(150, 1)
        result = RK4ShapeIntegrator(model)(torch.zeros(150), 0.2, 0.3, 0.35)
        self.assertEqual(tuple(result.shape), (150,))
        result.square().sum().backward()
        self.assertTrue(torch.isfinite(result).all())
        self.assertIsNotNone(model.fc2.weight.grad)


if __name__ == '__main__':
    unittest.main()
