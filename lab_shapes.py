"""Numeric, patient-separated PCA data for the laboratory shape experiment."""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class ShapePCA:
    mean_: np.ndarray
    components_: np.ndarray

    def transform(self, shapes):
        return (np.asarray(shapes, dtype=np.float64) - self.mean_) @ self.components_.T

    def inverse_transform(self, coefficients):
        return np.asarray(coefficients, dtype=np.float64) @ self.components_ + self.mean_

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data['mean'].copy(), data['components'].copy())


def patient_indices(pids, splits):
    """Preserve observation order and require one split per merged patient."""
    if len(pids) != len(splits) or not len(pids):
        raise ValueError('nonempty matching patient and split arrays required')
    groups = {}
    for index, (pid, split) in enumerate(zip(pids, splits)):
        if split not in ('train', 'test'):
            raise ValueError('unknown split')
        groups.setdefault(str(pid), []).append(index)
    for indices in groups.values():
        if len({str(splits[i]) for i in indices}) != 1:
            raise ValueError('a merged patient occurs in multiple splits')
    return {pid: np.asarray(indices, dtype=np.int64) for pid, indices in groups.items()}


def fit_train_pca(vertices, pids, splits, dimensions):
    """Fit all training observations; held-out geometry never enters fit."""
    vertices = np.asarray(vertices, dtype=np.float64)
    patient_indices(pids, splits)
    if vertices.ndim != 3 or vertices.shape[2] != 3 or len(vertices) != len(pids):
        raise ValueError('expected corresponding vertices[N,V,3]')
    if not np.isfinite(vertices).all():
        raise ValueError('nonfinite mesh')
    train = np.asarray(splits) == 'train'
    flat = vertices.reshape(len(vertices), -1)
    if dimensions < 1 or dimensions > min(train.sum() - 1, flat.shape[1]):
        raise ValueError('insufficient training observations for PCA dimension')
    fitted = PCA(n_components=dimensions, svd_solver='full', whiten=False).fit(flat[train])
    pca = ShapePCA(fitted.mean_.copy(), fitted.components_.copy())
    coefficients = pca.transform(flat)
    return pca, coefficients, fitted
