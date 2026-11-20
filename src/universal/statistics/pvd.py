"""Exact PVD covariance after averaging over training-time point sampling."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .core import validate_statistics


@torch.no_grad()
def fit_pvd_resampling_limit(source, *, rank: int, batch_size: int) -> dict[str, Any]:
    """Fit the K -> infinity limit of PVD's with-replacement point sampling.

    Conditional on a training shape, the 2,048 point slots are independent
    draws from its 10,000-point pool.  Consequently the full covariance is
    determined by a 3x3 within-shape covariance and a 3x3 covariance of shape
    centroids.  This routine computes those moments directly and expands their
    eigensystem to the model's coordinate-major ``[xyz, point]`` layout.
    """
    dimension = int(source.dimension)
    npoints = int(source.npoints)
    if not 1 <= rank <= dimension:
        raise ValueError(f"rank must be in [1, {dimension}]")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    pools = np.asarray(source.dataset_object.train_points[: source.num_shapes])
    num_shapes, pool_size, spatial_dimension = pools.shape
    if spatial_dimension != 3 or dimension != spatial_dimension * npoints:
        raise ValueError("unexpected PVD point-pool shape")

    within_scatter = torch.zeros(3, 3, dtype=torch.float64)
    centroid_sum = torch.zeros(3, dtype=torch.float64)
    centroid_outer_sum = torch.zeros(3, 3, dtype=torch.float64)
    for begin in range(0, num_shapes, batch_size):
        points = torch.from_numpy(pools[begin : begin + batch_size]).to(torch.float64)
        centroids = points.mean(dim=1)
        residuals = points - centroids[:, None]
        within_scatter += torch.einsum("bpi,bpj->ij", residuals, residuals)
        centroid_sum += centroids.sum(dim=0)
        centroid_outer_sum += centroids.T @ centroids

    spatial_mean = centroid_sum / num_shapes
    within = within_scatter / (num_shapes * pool_size)
    between = centroid_outer_sum / num_shapes - torch.outer(spatial_mean, spatial_mean)
    within = (within + within.T) / 2
    between = (between + between.T) / 2

    # Coordinate-major covariance: within (x) I + between (x) 11^T.
    common_values, common_vectors = torch.linalg.eigh(within + npoints * between)
    residual_values, residual_vectors = torch.linalg.eigh(within)

    point_common = torch.ones(npoints, dtype=torch.float64) / npoints**0.5
    point_residual = torch.zeros(npoints, npoints - 1, dtype=torch.float64)
    indices = torch.arange(npoints - 1, dtype=torch.float64)
    normalizers = torch.sqrt((indices + 1) * (indices + 2))
    for column in range(npoints - 1):
        point_residual[: column + 1, column] = 1 / normalizers[column]
        point_residual[column + 1, column] = -(column + 1) / normalizers[column]

    groups = []
    for index in range(3):
        vector = torch.kron(common_vectors[:, index], point_common).unsqueeze(1)
        groups.append((float(common_values[index]), vector))
        vectors = (
            residual_vectors[:, index, None, None]
            * point_residual[None]
        ).reshape(dimension, npoints - 1)
        groups.append((float(residual_values[index]), vectors))
    groups.sort(key=lambda item: item[0], reverse=True)

    # Spectrally truncate only after constructing the exact K -> infinity
    # covariance. Equal-eigenvalue residual groups have no preferred basis;
    # full rank therefore remains the canonical PVD setting.
    value_parts = []
    vector_parts = []
    remaining = rank
    for value, vectors in groups:
        count = min(remaining, vectors.shape[1])
        if count:
            value_parts.append(torch.full((count,), value, dtype=torch.float64))
            vector_parts.append(vectors[:, :count])
            remaining -= count
        if remaining == 0:
            break
    eigenvalues = torch.cat(value_parts).clamp_min(0).float()
    eigenvectors = torch.cat(vector_parts, dim=1).float()
    mean = spatial_mean.repeat_interleave(npoints).float()
    total_variance = npoints * torch.trace(within + between)
    result = {
        "mean": mean,
        "eigenvectors": eigenvectors,
        "eigenvalues": eigenvalues,
        "num_samples": num_shapes,
        "rank": rank,
        "requested_rank": rank,
        "dimension": dimension,
        "total_variance": float(total_variance),
        "explained_variance_ratio": eigenvalues / float(total_variance),
        "preprocessing_id": source.preprocessing_id,
        "flatten_layout": source.flatten_layout,
        "source_split": source.source_split,
        "effective_rank": rank,
        "within_spatial_covariance": within.float(),
        "between_centroid_covariance": between.float(),
        "point_pool_size": pool_size,
        "num_shapes": num_shapes,
    }
    validate_statistics(result)
    return result
