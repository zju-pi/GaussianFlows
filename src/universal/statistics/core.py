"""Gaussian statistics via exact full covariance or randomized low-rank PCA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


FORMAT_VERSION = 1


def add_isotropic_variance(
    statistics: dict[str, Any],
    variance: float,
    *,
    preprocessing_suffix: str,
) -> dict[str, Any]:
    """Add an exact isotropic covariance term to full-rank statistics.

    For a random vector ``y = x + eps`` with independent
    ``eps ~ N(0, variance I)``, the mean and eigenvectors are unchanged and
    every covariance eigenvalue increases by ``variance``.  Requiring a
    full-rank eigendecomposition avoids silently treating the unretained
    subspace as zero variance.
    """
    validate_statistics(statistics)
    variance = float(variance)
    if not torch.isfinite(torch.tensor(variance)) or variance < 0:
        raise ValueError("isotropic variance must be finite and nonnegative")
    if int(statistics["rank"]) != int(statistics["dimension"]):
        raise ValueError("exact isotropic correction requires full-rank statistics")

    corrected = dict(statistics)
    values = torch.as_tensor(statistics["eigenvalues"]).clone() + variance
    total_variance = float(statistics["total_variance"]) + (
        int(statistics["dimension"]) * variance
    )
    corrected.update({
        "eigenvalues": values,
        "total_variance": total_variance,
        "explained_variance_ratio": values / max(total_variance, 1e-30),
        "preprocessing_id": (
            f"{statistics['preprocessing_id']}+{preprocessing_suffix}"
        ),
        "isotropic_variance_added": variance,
    })
    validate_statistics(corrected)
    return corrected


def _empty_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _checked_batches(source, batch_size: int, label: str):
    count = 0
    for batch in source.iter_batches(batch_size):
        batch = torch.as_tensor(batch, dtype=torch.float32)
        if batch.ndim != 2 or batch.shape[1] != source.dimension:
            raise ValueError(
                f"{label}: expected [B,{source.dimension}], got {tuple(batch.shape)}"
            )
        if not torch.isfinite(batch).all():
            raise ValueError(f"{label}: source contains non-finite values")
        count += len(batch)
        yield batch
    if count != source.num_samples:
        raise RuntimeError(
            f"{label}: iterator produced {count} samples, expected {source.num_samples}"
        )


@torch.no_grad()
def fit_gaussian_statistics(
    source,
    *,
    rank: int,
    batch_size: int,
    device: torch.device | str,
    oversample: int = 0,
    power_iterations: int = 2,
    seed: int = 42,
    allow_rank_truncation: bool = False,
) -> dict[str, Any]:
    """Fit a centered low-rank covariance without collecting source samples.

    The source must expose stable ``iter_batches(batch_size)`` passes plus
    ``num_samples``, ``dimension``, and canonical metadata attributes.
    """
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if batch_size < 1 or rank < 1 or oversample < 0 or power_iterations < 0:
        raise ValueError("invalid fit parameters")
    num_samples, dimension = int(source.num_samples), int(source.dimension)
    max_rank = min(num_samples - 1, dimension)
    if max_rank < 1:
        raise ValueError("at least two samples are required")
    if rank > max_rank and not allow_rank_truncation:
        raise ValueError(
            f"requested rank {rank} exceeds the maximum available rank {max_rank}; "
            "set allow_rank_truncation=True to opt in to truncation"
        )
    effective_rank = min(rank, max_rank)
    projected_rank = min(effective_rank + oversample, max_rank)

    sum_vector = torch.zeros(dimension, dtype=torch.float64, device=device)
    for batch in tqdm(
        _checked_batches(source, batch_size, "mean"),
        total=(num_samples + batch_size - 1) // batch_size,
        desc="Gaussian mean",
    ):
        sum_vector += batch.to(device=device, dtype=torch.float64).sum(dim=0)
    mean = (sum_vector / num_samples).float()
    del sum_vector

    # A randomized range finder is unnecessary when every ambient direction
    # is retained.  Accumulating the covariance directly is both exact (up to
    # floating-point arithmetic) and avoids an N x d QR decomposition.
    if effective_rank == dimension:
        covariance = torch.zeros(
            dimension, dimension, dtype=torch.float32, device=device
        )
        total_squared = torch.zeros((), dtype=torch.float64, device=device)
        for batch in tqdm(
            _checked_batches(source, batch_size, "full covariance"),
            total=(num_samples + batch_size - 1) // batch_size,
            desc="Gaussian full covariance",
        ):
            centered = batch.to(device) - mean
            covariance += centered.T @ centered
            total_squared += centered.double().square().sum()
        covariance /= num_samples - 1
        values, vectors = torch.linalg.eigh(covariance)
        order = torch.argsort(values, descending=True)
        eigenvalues = values[order].clamp_min(0.0)
        eigenvectors = vectors[:, order]
        total_variance = total_squared / (num_samples - 1)
        result = {
            "mean": mean.cpu(),
            "eigenvectors": eigenvectors.cpu(),
            "eigenvalues": eigenvalues.cpu(),
            "num_samples": num_samples,
            "rank": effective_rank,
            "requested_rank": rank,
            "dimension": dimension,
            "total_variance": float(total_variance.cpu()),
            "explained_variance_ratio": (
                eigenvalues / total_variance.float().clamp_min(1e-30)
            ).cpu(),
            "preprocessing_id": source.preprocessing_id,
            "flatten_layout": source.flatten_layout,
            "source_split": source.source_split,
        }
        validate_statistics(result)
        return result

    generator = torch.Generator(device=device).manual_seed(seed)
    omega = torch.randn(
        dimension, projected_rank, dtype=torch.float32,
        device=device, generator=generator,
    )
    projected = torch.empty(
        num_samples, projected_rank, dtype=torch.float32, device=device
    )
    cursor = 0
    for batch in tqdm(
        _checked_batches(source, batch_size, "range finder"),
        total=(num_samples + batch_size - 1) // batch_size,
        desc="Gaussian range finder",
    ):
        centered = batch.to(device) - mean
        projected[cursor : cursor + len(batch)] = centered @ omega
        cursor += len(batch)
    sample_basis, _ = torch.linalg.qr(projected, mode="reduced")
    del projected, omega
    _empty_cuda_cache(device)

    small_matrix = torch.zeros(
        projected_rank, dimension, dtype=torch.float32, device=device
    )
    cursor = 0
    for batch in tqdm(
        _checked_batches(source, batch_size, "projected matrix"),
        total=(num_samples + batch_size - 1) // batch_size,
        desc="Gaussian projected matrix",
    ):
        centered = batch.to(device) - mean
        small_matrix += sample_basis[cursor : cursor + len(batch)].T @ centered
        cursor += len(batch)
    del sample_basis

    gram = small_matrix @ small_matrix.T
    gram_values, gram_vectors = torch.linalg.eigh(gram)
    order = torch.argsort(gram_values, descending=True)[:effective_rank]
    singular = torch.sqrt(gram_values[order].clamp_min(0.0))
    left = gram_vectors[:, order]
    safe_singular = singular.clamp_min(torch.finfo(torch.float32).eps)
    eigenvectors = (small_matrix.T @ left) / safe_singular
    eigenvectors, _ = torch.linalg.qr(eigenvectors, mode="reduced")
    del gram, gram_values, gram_vectors, singular, left, small_matrix
    _empty_cuda_cache(device)

    for iteration in range(power_iterations):
        covariance_times_vectors = torch.zeros_like(eigenvectors)
        for batch in tqdm(
            _checked_batches(source, batch_size, f"power iteration {iteration + 1}"),
            total=(num_samples + batch_size - 1) // batch_size,
            desc=f"Gaussian power iteration {iteration + 1}",
        ):
            centered = batch.to(device) - mean
            covariance_times_vectors += centered.T @ (centered @ eigenvectors)
        eigenvectors, _ = torch.linalg.qr(
            covariance_times_vectors, mode="reduced"
        )

    projected_covariance = torch.zeros(
        effective_rank, effective_rank, dtype=torch.float32, device=device
    )
    total_squared = torch.zeros((), dtype=torch.float64, device=device)
    for batch in tqdm(
        _checked_batches(source, batch_size, "Rayleigh quotient"),
        total=(num_samples + batch_size - 1) // batch_size,
        desc="Gaussian Rayleigh quotient",
    ):
        centered = batch.to(device) - mean
        coordinates = centered @ eigenvectors
        projected_covariance += coordinates.T @ coordinates
        total_squared += centered.double().square().sum()
    projected_covariance /= num_samples - 1
    values, rotation = torch.linalg.eigh(projected_covariance)
    order = torch.argsort(values, descending=True)
    eigenvalues = values[order].clamp_min(0.0)
    eigenvectors = eigenvectors @ rotation[:, order]
    total_variance = total_squared / (num_samples - 1)

    result = {
        "mean": mean.cpu(),
        "eigenvectors": eigenvectors.cpu(),
        "eigenvalues": eigenvalues.cpu(),
        "num_samples": num_samples,
        "rank": effective_rank,
        "requested_rank": rank,
        "dimension": dimension,
        "total_variance": float(total_variance.cpu()),
        "explained_variance_ratio": (
            eigenvalues / total_variance.float().clamp_min(1e-30)
        ).cpu(),
        "preprocessing_id": source.preprocessing_id,
        "flatten_layout": source.flatten_layout,
        "source_split": source.source_split,
    }
    validate_statistics(result)
    return result


def validate_statistics(statistics: dict[str, Any]) -> None:
    mean = torch.as_tensor(statistics["mean"], dtype=torch.float64)
    vectors = torch.as_tensor(statistics["eigenvectors"], dtype=torch.float64)
    values = torch.as_tensor(statistics["eigenvalues"], dtype=torch.float64)
    dimension, rank = int(statistics["dimension"]), int(statistics["rank"])
    if mean.shape != (dimension,) or vectors.shape != (dimension, rank):
        raise ValueError("mean/eigenvector shapes do not match dimension and rank")
    if not torch.isfinite(mean).all():
        raise ValueError("mean contains non-finite values")
    if not torch.isfinite(vectors).all():
        raise ValueError("eigenvectors contain non-finite values")
    if values.shape != (rank,) or not torch.isfinite(values).all():
        raise ValueError("invalid eigenvalues")
    if torch.any(values < -1e-8) or torch.any(torch.diff(values) > 1e-6):
        raise ValueError("eigenvalues must be nonnegative and descending")
    identity = torch.eye(rank, dtype=torch.float64)
    orthogonality_error = torch.linalg.matrix_norm(
        vectors.T @ vectors - identity, ord="fro"
    ) / max(rank, 1) ** 0.5
    if float(orthogonality_error) > 2e-3:
        raise ValueError(
            f"eigenvectors are not orthonormal: {float(orthogonality_error)}"
        )
    total_variance = float(statistics["total_variance"])
    if not torch.isfinite(torch.tensor(total_variance)) or total_variance < 0:
        raise ValueError("total_variance must be finite and nonnegative")
    if float(values.sum()) > total_variance * (1.0 + 2e-4) + 1e-7:
        raise ValueError("captured eigenvalues exceed total variance")
    for key in ("preprocessing_id", "flatten_layout", "source_split"):
        if not statistics.get(key):
            raise ValueError(f"missing statistics metadata: {key}")


def save_statistics(
    path: str | Path,
    *,
    dataset: str,
    statistics: dict[str, Any],
    source_metadata: dict[str, Any],
    fit_config: dict[str, Any],
) -> Path:
    validate_statistics(statistics)
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"statistics artifact already exists: {path}")
    payload = {
        "format_version": FORMAT_VERSION,
        "artifact": "gaussian_statistics",
        "dataset": dataset,
        "global_stats": statistics,
        "source": source_metadata,
        "fit": fit_config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_statistics(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"invalid statistics artifact: {path}")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported statistics format version: {payload.get('format_version')!r}"
        )
    if payload.get("artifact") != "gaussian_statistics":
        raise ValueError(f"not a Gaussian-statistics artifact: {path}")
    if "global_stats" not in payload:
        raise ValueError("statistics artifact is missing global_stats")
    validate_statistics(payload["global_stats"])
    return payload
