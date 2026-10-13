"""Voxel-input continuous cognition estimator for BrainODE section 3.2.

Appendix A.3 specifies a simple 3D CNN, voxel input, and scalar c in [0, 1].
Its layers and loss are unpublished. Channels (16, 32, 64), ReLU, pooling,
sigmoid and soft-label BCE below are explicit implementation supplements
to the paper's description.
This is not a claim to recover the authors' unpublished estimator.
"""

from math import isfinite
from numbers import Integral

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint
import trimesh


class CognitionEstimator3D(nn.Module):
    """Binary occupancy [B,1,X,Y,Z] -> continuous cognition [B]."""

    def __init__(self, channels=(16, 32, 64)):
        super().__init__()
        if not channels or any(
            isinstance(c, bool) or not isinstance(c, Integral) or c <= 0
            for c in channels
        ):
            raise ValueError("channels must contain positive integers")
        layers, previous = [], 1
        for width in channels:
            layers.extend([
                nn.Conv3d(previous, width, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(2),
            ])
            previous = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(previous, 1)
        self.minimum_extent = 2 ** len(channels)

    def logits(self, voxels):
        if voxels.ndim != 5 or voxels.shape[1] != 1:
            raise ValueError("voxel input must have shape [B,1,X,Y,Z]")
        if min(voxels.shape[2:]) < self.minimum_extent:
            raise ValueError("voxel grid is too small for the pooling stages")
        return self.head(self.pool(self.features(voxels)).flatten(1)).squeeze(1)

    def forward(self, voxels):
        return torch.sigmoid(self.logits(voxels))

    def estimate_c(self, voxels):
        """Single-shape inference adapter for Algorithm 2."""
        self.eval()
        with torch.inference_mode():
            voxels = torch.as_tensor(
                voxels, dtype=self.head.weight.dtype, device=self.head.weight.device,
            )
            prediction = self(voxels)
            if prediction.numel() != 1:
                raise ValueError("progressive inference requires one shape")
            return prediction.item()


def cognition_loss(estimator, voxels, labels):
    """Supplement: BCE accepts both observed 0/1 and fractional pseudo labels."""
    logits = estimator.logits(voxels)
    labels = torch.as_tensor(labels, dtype=logits.dtype, device=logits.device).reshape(-1)
    if labels.shape != logits.shape:
        raise ValueError("one cognition label is required per voxel shape")
    if not torch.isfinite(labels).all() or torch.any((labels < 0) | (labels > 1)):
        raise ValueError("cognition labels must be finite and in [0,1]")
    return F.binary_cross_entropy_with_logits(logits, labels)


def train_cognition_estimator(estimator, batches, *, epochs, optimizer):
    """Train observed+pseudo soft labels; epoch count/optimizer are explicit.

    The caller constructs a repeatable training-only loader, including pseudo
    shapes sampled only from training subjects. No split or mixing ratio is
    chosen here. This function is not run on research data during setup.
    """
    if isinstance(epochs, bool) or not isinstance(epochs, Integral) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    history = []
    for _ in range(epochs):
        estimator.train()
        total, count = 0.0, 0
        for voxels, labels in batches:
            voxels = voxels.to(estimator.head.weight)
            optimizer.zero_grad()
            loss = cognition_loss(estimator, voxels, labels)
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(voxels)
            count += len(voxels)
        if count == 0:
            raise ValueError("training loader is empty or not repeatable")
        history.append(total / count)
    return history


class PCAShapeVoxelizer:
    """PCA -> shared-topology mesh in mm -> solid occupancy on a fixed grid.

    All shapes use the caller's explicit origin, pitch and grid shape in the
    canonical mesh frame. There is no per-shape scaling, centering or crop.
    Grid settings are supplementary, not specified by the BrainODE paper.
    The caller must confirm the mm geometry before using research meshes.
    surface-fill preserves the original diagnostic implementation;
    center-occupancy classifies fixed voxel centers by ray parity. Neither
    method repairs self-intersections or proves anatomical surface validity.
    """

    def __init__(self, pca, faces, *, grid_shape, origin_mm, pitch_mm,
                 method='surface-fill', occupancy_rays=3):
        self.pca = pca
        self.faces = np.asarray(faces)
        if self.faces.ndim != 2 or self.faces.shape[1] != 3 or not len(self.faces):
            raise ValueError("shared triangular faces must have shape [F,3]")
        if not np.issubdtype(self.faces.dtype, np.integer) or np.any(self.faces < 0):
            raise ValueError("faces must be nonnegative integer vertex indices")
        if len(grid_shape) != 3 or any(
            isinstance(n, bool) or not isinstance(n, Integral) or n <= 0
            for n in grid_shape
        ):
            raise ValueError("grid_shape must contain three positive integers")
        self.grid_shape = tuple(int(n) for n in grid_shape)
        self.origin_mm = np.asarray(origin_mm, dtype=np.float64)
        self.pitch_mm = float(pitch_mm)
        if method not in ('surface-fill', 'center-occupancy'):
            raise ValueError('unknown voxelization method')
        if isinstance(occupancy_rays, bool) or not isinstance(occupancy_rays, Integral) or occupancy_rays < 1 or occupancy_rays % 2 != 1:
            raise ValueError('occupancy_rays must be a positive odd integer')
        self.method, self.occupancy_rays = method, int(occupancy_rays)
        if self.origin_mm.shape != (3,) or not np.isfinite(self.origin_mm).all():
            raise ValueError("origin_mm must contain three finite coordinates")
        if not isfinite(self.pitch_mm) or self.pitch_mm <= 0:
            raise ValueError("pitch_mm must be positive and finite")
        lattice_origin = self.origin_mm / self.pitch_mm
        if not np.allclose(lattice_origin, np.rint(lattice_origin), rtol=0, atol=1e-8):
            raise ValueError("origin must align with the voxelizer's global pitch lattice")

    def __call__(self, coefficients):
        if isinstance(coefficients, torch.Tensor):
            coefficients = coefficients.detach().cpu().numpy()
        coefficients = np.asarray(coefficients)
        if coefficients.ndim == 2 and coefficients.shape[0] == 1:
            coefficients = coefficients[0]
        if coefficients.ndim != 1 or not np.isfinite(coefficients).all():
            raise ValueError("one finite PCA coefficient vector is required")
        vertices = np.asarray(self.pca.inverse_transform(coefficients[None])).reshape(-1, 3)
        if not np.isfinite(vertices).all() or self.faces.max() >= len(vertices):
            raise ValueError("invalid reconstructed vertices or face indices")
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces, process=False)
        if not mesh.is_watertight:
            raise ValueError("solid voxelization requires a watertight mesh")
        if self.method == 'center-occupancy':
            # Classify voxel centers instead of filling intersected surface cells.
            # Ray parity on self-intersecting surfaces remains a documented
            # interpretation, not proof of a valid anatomical interior.
            import open3d as o3d
            lower = self.origin_mm - self.pitch_mm / 2
            upper = self.origin_mm + (np.asarray(self.grid_shape) - .5) * self.pitch_mm
            if np.any(vertices.min(0) < lower) or np.any(vertices.max(0) > upper):
                raise ValueError('shape does not fit the fixed voxel grid; do not crop it')
            start = np.maximum(0, np.floor((vertices.min(0)-self.origin_mm)/self.pitch_mm).astype(int)-1)
            end = np.minimum(self.grid_shape, np.ceil((vertices.max(0)-self.origin_mm)/self.pitch_mm).astype(int)+2)
            axes = [np.arange(a,b) for a,b in zip(start,end)]
            coordinates = np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1)
            centers = (self.origin_mm + coordinates * self.pitch_mm).astype(np.float32)
            scene = o3d.t.geometry.RaycastingScene(nthreads=4)
            scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)),
                                o3d.core.Tensor(self.faces.astype(np.uint32)))
            occupancy = scene.compute_occupancy(o3d.core.Tensor(centers), nthreads=4,
                                                nsamples=self.occupancy_rays).numpy()
            if not np.isfinite(occupancy).all() or not occupancy.any():
                raise ValueError('voxel-center occupancy is empty or invalid')
            grid = np.zeros(self.grid_shape, dtype=np.float32)
            grid[tuple(slice(int(a),int(b)) for a,b in zip(start,end))] = occupancy
            return grid[None,None]
        occupied = mesh.voxelized(pitch=self.pitch_mm, method='subdivide').fill().points
        indices = np.rint((occupied - self.origin_mm) / self.pitch_mm).astype(np.int64)
        if not len(indices) or np.any(indices < 0) or np.any(indices >= self.grid_shape):
            raise ValueError("shape does not fit the fixed voxel grid; do not crop it")
        grid = np.zeros(self.grid_shape, dtype=np.float32)
        grid[tuple(indices.T)] = 1.0
        return grid[None, None]  # [B=1,C=1,X,Y,Z]


class RK4ShapeIntegrator:
    """Connect Eq. (4)/Algorithm 2 to the existing ODE model, holding c fixed."""

    def __init__(self, model):
        self.model = model

    def __call__(self, shape, start_time, end_time, c):
        start_time, end_time, c = map(float, (start_time, end_time, c))
        if not all(isfinite(x) for x in (start_time, end_time, c)) or not 0 <= c <= 1:
            raise ValueError("times must be finite and c must be in [0,1]")
        parameter = next(self.model.parameters())
        initial = torch.as_tensor(shape, dtype=parameter.dtype, device=parameter.device)
        unbatched = initial.ndim == 1
        if unbatched:
            initial = initial.unsqueeze(0)
        if initial.ndim != 2 or initial.shape[0] != 1:
            raise ValueError("one PCA shape [K] or [1,K] is required")
        if start_time == end_time:
            result = initial.clone()
        else:
            times = initial.new_tensor([start_time, end_time])
            condition = initial.new_tensor([c])
            result = odeint(
                lambda t, coefficients: self.model(t, coefficients, condition),
                initial, times, method='rk4',
            )[-1]
        return result.squeeze(0) if unbatched else result
