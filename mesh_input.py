"""Boundary point extraction matching the official shape-generation utility.

No independent surface mesh, image resampling, smoothing or scale adjustment
is added here. Header-world coordinates retain the header's numerical units;
this module does not infer millimetres when NIfTI spatial units are unknown.
"""

import numpy as np
from scipy import ndimage


def boundary_points(mask):
    """Official 2x2x2 convolution rule: 0<count<8, argwhere + .5."""
    mask = np.asarray(mask)
    if mask.ndim != 3 or mask.dtype != np.bool_:
        raise ValueError("a 3D boolean region mask is required")
    if not mask.any():
        raise ValueError("region mask is empty")
    counts = ndimage.convolve(
        mask.astype(np.uint8), np.ones((2, 2, 2), dtype=np.uint8),
        mode='constant', cval=0,
    )
    return np.argwhere((counts > 0) & (counts < 8)).astype(np.float64) + 0.5


def boundary_geometry(mask, affine):
    """Keep both voxel and header-world coordinates and their centroids.

    This prepares the target PCD before the official rigid ICP. The returned
    arrays are not fitted meshes or final BrainODE inputs. Translation can be
    restored using the saved centroid; ICP matrices must be saved separately.
    """
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("affine must be a finite 4x4 matrix")
    if not np.allclose(affine[3], [0, 0, 0, 1]) or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("affine must be an invertible spatial transform")
    voxel = boundary_points(mask)
    world = voxel @ affine[:3, :3].T + affine[:3, 3]
    voxel_centroid = voxel.mean(0)
    world_centroid = world.mean(0)
    return {
        'boundary_voxel': voxel,
        'boundary_header_world': world,
        'target_centered_voxel_units': voxel - voxel_centroid,
        'target_centered_header_world_units': world - world_centroid,
        'centroid_voxel': voxel_centroid,
        'centroid_header_world': world_centroid,
        'affine': affine,
    }
