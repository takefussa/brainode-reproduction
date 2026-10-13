"""Inspect the PCA asset without executing repository-selected Python globals."""
import json
import pickle
from pathlib import Path
import numpy as np
from numpy.core import multiarray


class PCARecord:
    """Data-only replacement for sklearn's PCA; never fits or transforms data."""


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ('sklearn.decomposition._pca', 'PCA'): PCARecord,
            ('numpy.core.multiarray', '_reconstruct'): multiarray._reconstruct,
            ('numpy.core.multiarray', 'scalar'): multiarray.scalar,
            ('numpy._core.multiarray', '_reconstruct'): multiarray._reconstruct,
            ('numpy._core.multiarray', 'scalar'): multiarray.scalar,
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Forbidden global: {module}.{name}')
        return allowed[module, name]

    def persistent_load(self, pid):
        raise pickle.UnpicklingError('Persistent objects forbidden')


root = Path('/home/take/brainode')
with (root / 'dataset/PCA/pca.pickle').open('rb') as f:
    content = DataOnlyUnpickler(f).load()
pca = content['pca']
summary = {'keys': list(content), 'n_components': int(pca.n_components_),
           'n_features': int(pca.n_features_in_), 'n_samples': int(pca.n_samples_),
           'whiten': pca.whiten, 'sklearn_version_stored': pca._sklearn_version,
           'components_shape': list(pca.components_.shape), 'mean_shape': list(pca.mean_.shape),
           'mean_centroid': pca.mean_.reshape(-1, 3).mean(0).tolist(),
           'explained_variance_ratio_sum': float(pca.explained_variance_ratio_.sum())}
(root / 'audit/official_pca_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
