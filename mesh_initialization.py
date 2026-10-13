"""Explicit template-only initialization for the lab's physical mask sizes.

Mask-volume initialization is a documented supplement to the paper. It is
applied only to the first observation's template, not to masks, subsequent
fitted meshes or final volumes. Faces and vertex order remain unchanged.
"""

import math

import numpy as np
import trimesh


def initialize_template(vertices, faces, *, mode, mask_volume=None):
    vertices=np.asarray(vertices,dtype=np.float64)
    faces=np.asarray(faces)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices)
        or not np.isfinite(vertices).all()):
        raise ValueError('finite [N,3] template vertices required')
    if (faces.ndim!=2 or faces.shape[1]!=3 or not len(faces)
        or not np.issubdtype(faces.dtype,np.integer)
        or faces.min()<0 or faces.max()>=len(vertices)):
        raise ValueError('valid integer triangle indices required')
    if mode not in ('unscaled','mask-volume'):
        raise ValueError('unknown template initialization mode')
    volume=float(trimesh.Trimesh(vertices,faces,process=False).volume)
    if not math.isfinite(volume) or volume<=0:
        raise ValueError('template signed volume must be positive')
    if mask_volume is not None and (not math.isfinite(mask_volume) or mask_volume<=0):
        raise ValueError('mask volume must be finite and positive')
    if mode=='mask-volume' and mask_volume is None:
        raise ValueError('mask volume is required for size initialization')
    factor=float((mask_volume/volume)**(1/3)) if mode=='mask-volume' else 1.
    pivot=vertices.mean(0)
    initialized=(vertices-pivot)*factor+pivot if mode=='mask-volume' else vertices.copy()
    return initialized,dict(mode=mode,factor=factor,pivot=pivot.tolist(),
        unscaled_template_volume=volume,target_mask_volume=mask_volume,
        expected_initial_volume=volume*factor**3,
        applies_to='first observation template only',
        final_volume_constraint=False,mask_rescaling=False)
