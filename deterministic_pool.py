"""Nonoverlapping 2x2x2 maximum pooling with deterministic CUDA gradients.

PyTorch 2.4.1 rejects MaxPool3D CUDA backward in strict deterministic mode.
Each disjoint group below selects the first maximum in the same z/y/x order
as kernel=2, stride=2 MaxPool3D. Reduction backward writes to unique elements.
"""

import torch
from torch import nn


class DisjointMaxPool3D(nn.Module):
    def forward(self, value):
        if value.ndim != 5 or any(n % 2 for n in value.shape[2:]):
            raise ValueError('disjoint pooling requires even [B,C,X,Y,Z] extents')
        b, c, x, y, z = value.shape
        groups = value.reshape(b, c, x//2, 2, y//2, 2, z//2, 2)
        groups = groups.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(b, c, x//2, y//2, z//2, 8)
        return torch.max(groups, dim=-1).values
