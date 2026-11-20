"""Small public API used by the four paper notebooks."""

from .artifacts import load_trajectory_run
from .trajectory import (
    GaussianStatistics,
    align_regularity_coordinates,
    analytic_gaussian_vp_from_start,
    controlled_same_start_rmse,
    load_gaussian_statistics,
    regularity_local_coordinates,
    summarize_curves,
    validate_statistics_compatibility,
)
from .hyperbolic import (
    ambient_start_procrustes_projection,
    aligned_gaussian_hyperbolic_projection,
    select_hyperbolic_eigenpairs,
)
from .independent_start import load_result as load_independent_start_result
__all__ = [
    "GaussianStatistics",
    "ambient_start_procrustes_projection",
    "aligned_gaussian_hyperbolic_projection",
    "align_regularity_coordinates",
    "analytic_gaussian_vp_from_start",
    "controlled_same_start_rmse",
    "load_trajectory_run",
    "load_gaussian_statistics",
    "load_independent_start_result",
    "regularity_local_coordinates",
    "select_hyperbolic_eigenpairs",
    "summarize_curves",
    "validate_statistics_compatibility",
]
