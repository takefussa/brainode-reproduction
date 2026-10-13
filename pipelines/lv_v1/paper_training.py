"""BrainODE Algorithm 1 loss with RK4 and all temporal observation pairs.

Same-age observations are retained as separate sources/targets, while the
solver receives unique times. Instantaneous replicate-to-replicate prediction
is not a future/past trajectory. No artificial time jitter is added.
"""

import torch
from torchdiffeq import odeint


def scale_pca_shapes(coefficients, pca_mean, pca_components, factor):
    """Scale actual vertex coordinates, including the PCA centering offset.

    PCA(factor * vertices) is not generally factor * PCA(vertices).
    """
    mean = torch.as_tensor(pca_mean, dtype=coefficients.dtype, device=coefficients.device)
    components = torch.as_tensor(pca_components, dtype=coefficients.dtype, device=coefficients.device)
    return factor * coefficients + (factor - 1.) * (mean @ components.T)


def trajectory_loss(model, shapes, times, conditions):
    """Mean squared Euclidean error over every ordered, distinct-age pair.

    Each source uses its own observed c, held constant for the corresponding
    forward/backward trajectory as in Algorithm 1. Target diagnosis is not
    supplied to the ODE. Return (None, 0) for a subject without temporal pairs;
    this does not exclude their observations from PCA/cognition data.
    """
    if shapes.ndim != 2 or times.ndim != 1 or conditions.ndim != 1:
        raise ValueError('expected shapes[N,D], times[N], conditions[N]')
    if len(shapes) != len(times) or len(times) != len(conditions) or not len(times):
        raise ValueError('observations must have matching nonempty lengths')
    if not all(torch.isfinite(x).all() for x in (shapes, times, conditions)):
        raise ValueError('all observations must be finite')
    losses, count = [], 0
    for source_index in range(len(times)):
        for direction in (1, -1):
            targets = torch.nonzero((times - times[source_index]) * direction > 0).flatten()
            if not len(targets):
                continue
            unique_times, mapping = torch.unique(times[targets], sorted=True, return_inverse=True)
            if direction == -1:
                unique_times = unique_times.flip(0)
                mapping = len(unique_times) - 1 - mapping
            solver_times = torch.cat([times[source_index:source_index + 1], unique_times])
            condition = conditions[source_index].reshape(1)
            predicted = odeint(lambda t, value: model(t, value, condition),
                                shapes[source_index:source_index + 1], solver_times,
                                method='rk4')[1:, 0, :]
            losses.append((predicted[mapping] - shapes[targets]).square().sum())
            count += len(targets)
    return (sum(losses) / count, count) if count else (None, 0)


def optimizer_batch_step(optimizer, losses):
    """Use every subject's gradient, including the final short batch."""
    losses = [loss for loss in losses if loss is not None]
    if not losses:
        return None
    optimizer.zero_grad(set_to_none=True)
    loss = torch.stack(losses).mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach())
