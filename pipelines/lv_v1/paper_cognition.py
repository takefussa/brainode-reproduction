"""BrainODE paper Eq. (4) and Algorithm 2, without undocumented defaults.

Source: NeurIPS 2025, section 3.2, Appendix A.3/A.4.
https://proceedings.neurips.cc/paper_files/paper/2025/file/
abdc8c031aa6c6917c3b593166e5e340-Paper-Conference.pdf

The caller supplies an ODE integrator, a trained voxel-input cognition
estimator, voxel conversion, and the number of progressive steps. This module
does not choose subjects, train models, or invent the unpublished 3D CNN.
Shapes are corresponding PCA coefficient arrays (NumPy or torch tensors).
"""

from collections.abc import Callable
from math import isfinite
from typing import Any

Integrator = Callable[[Any, float, float, float], Any]


def normalize_age(age: float) -> float:
    """Appendix A.4: t = (age - 65) / 30, without clipping or filtering.

    Reference formula only. The lab pipeline uses lab_dataset.AgeTimeScale
    fitted to eligible training acquisition ages. Cohort eligibility is
    separate. Values outside 65..95 remain visible in this reference formula.
    """
    age = float(age)
    if not isfinite(age):
        raise ValueError("age must be finite")
    return (age - 65.0) / 30.0


def observed_c(diagnosis: str) -> float | None:
    """Observed endpoint labels; MCI is excluded, not assigned intermediate c.

    Continuous intermediate supervision comes from Eq. (4). Unknown diagnoses
    require an explicit mapping decision instead of silent exclusion. This
    function alone does not discard a subject who has an MCI observation.
    """
    if diagnosis in ("CN", "NC"):
        return 0.0
    if diagnosis == "AD":
        return 1.0
    if diagnosis == "MCI":
        return None
    raise ValueError(f"diagnosis requires an explicit mapping: {diagnosis!r}")


def pseudo_cognitive_sample(
    nc_shape: Any,
    ad_shape: Any,
    nc_time: float,
    ad_time: float,
    sample_time: float,
    *,
    integrate: Integrator,
) -> tuple[Any, float]:
    """Eq. (4): average two ODE predictions and assign a continuous label.

    Endpoints must come from the same NC-to-AD subject in a shared PCA space.
    ``integrate(shape, start_time, end_time, c)`` holds c fixed for this solve
    and supports backward integration and zero-length intervals. Sampling
    times and eligible endpoint pairs must be specified by the caller.

    The shape weight is exactly 1/2 on each trajectory, even at the endpoints;
    c is the elapsed-time fraction. Real endpoint observations remain separate
    training examples. There is no linear interpolation of observed vertices.
    """
    nc_time, ad_time, sample_time = map(float, (nc_time, ad_time, sample_time))
    if not all(isfinite(t) for t in (nc_time, ad_time, sample_time)):
        raise ValueError("times must be finite")
    if ad_time <= nc_time:
        raise ValueError("AD endpoint must be later than NC endpoint")
    if not nc_time <= sample_time <= ad_time:
        raise ValueError("sample time must lie between the two endpoints")

    c = (sample_time - nc_time) / (ad_time - nc_time)
    forward = integrate(nc_shape, nc_time, sample_time, 0.0)
    backward = integrate(ad_shape, ad_time, sample_time, 1.0)
    return (forward + backward) * 0.5, c


def progressive_prediction(
    initial_shape: Any,
    initial_time: float,
    target_time: float,
    *,
    steps: int,
    integrate: Integrator,
    shape_to_voxels: Callable[[Any], Any],
    estimate_c: Callable[[Any], float],
) -> Any:
    """Algorithm 2: estimate c anew before each equal-duration ODE segment.

    Appendix A.3 requires voxel input to the 3D CNN. ``shape_to_voxels`` must
    therefore reconstruct the mesh from PCA and voxelize it under an explicitly specified
    specification. ``estimate_c`` consumes that voxel representation and
    returns the trained estimator's scalar in [0, 1]. No future diagnosis is
    supplied. The paper's general step count N has no default here.
    """
    initial_time, target_time = map(float, (initial_time, target_time))
    if not all(isfinite(t) for t in (initial_time, target_time)):
        raise ValueError("times must be finite")
    if target_time <= initial_time:
        raise ValueError("target time must be later than initial time")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")

    interval = (target_time - initial_time) / steps
    current = initial_shape
    for i in range(steps):
        c = float(estimate_c(shape_to_voxels(current)))
        if not isfinite(c) or not 0.0 <= c <= 1.0:
            raise ValueError("cognition estimator must return a finite c in [0, 1]")
        start = initial_time + i * interval
        end = target_time if i == steps - 1 else initial_time + (i + 1) * interval
        current = integrate(current, start, end, c)
    return current
