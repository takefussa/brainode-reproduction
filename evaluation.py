"""BrainODE shape errors against the original observed mesh.

No split, observation selection, or subject aggregation is chosen here.
Distances retain the coordinate units supplied by the caller; the lab's mm
interpretation remains an explicit assumption in the experiment metadata.
"""

import numpy as np


def shape_errors(predicted, observed):
    predicted, observed = np.asarray(predicted, dtype=np.float64), np.asarray(observed, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.ndim != 2 or predicted.shape[1] != 3:
        raise ValueError('expected matching corresponding vertex arrays [V,3]')
    if not len(predicted) or not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError('vertices must be nonempty and finite')
    difference = predicted - observed
    distances = np.linalg.norm(difference, axis=1)
    return dict(mean_vertex_euclidean=float(distances.mean()),
                coordinate_mae=float(np.abs(difference).mean()),
                per_vertex_euclidean=distances)


def evaluate_pca_prediction(pca, predicted_coefficients, observed_vertices, source_vertices):
    """Separate prediction error, PCA reconstruction error, and a static source.

    The primary target is observed_vertices before PCA. The public-code MAE
    against a PCA-reconstructed target is provided separately for comparison.
    No future diagnosis is required. Vertex order is assumed to be shared and
    must be verified in the input mesh manifests by the dataset builder.
    """
    coefficients = np.asarray(predicted_coefficients, dtype=np.float64)
    if coefficients.ndim != 1 or not len(coefficients) or not np.isfinite(coefficients).all():
        raise ValueError('one finite PCA coefficient vector is required')
    observed, source = np.asarray(observed_vertices, dtype=np.float64), np.asarray(source_vertices, dtype=np.float64)
    # Validate real geometry before passing flattened arrays into PCA.
    shape_errors(source, observed)
    predicted = pca.inverse_transform(coefficients[None]).reshape(-1, 3)
    reconstructed = pca.inverse_transform(pca.transform(observed.reshape(1, -1))).reshape(-1, 3)
    real = shape_errors(predicted, observed)
    compressed = shape_errors(predicted, reconstructed)
    reconstruction = shape_errors(reconstructed, observed)
    static = shape_errors(source, observed)
    return dict(prediction_mean_vertex_euclidean=real['mean_vertex_euclidean'],
                pca_target_mean_vertex_euclidean=compressed['mean_vertex_euclidean'],
                public_coordinate_mae_to_pca_target=compressed['coordinate_mae'],
                pca_reconstruction_mean_vertex_euclidean=reconstruction['mean_vertex_euclidean'],
                static_source_mean_vertex_euclidean=static['mean_vertex_euclidean'],
                per_vertex_euclidean=real['per_vertex_euclidean'],
                predicted_vertices=predicted)
