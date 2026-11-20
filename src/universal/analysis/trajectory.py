"""Readable Gaussian baselines for diffusion sampling trajectories.

All trajectories enter this module in VP coordinates.  Conversion to VE is
explicit, and every comparison starts from the same sampled noise state.
Only the four paper candidates are implemented here:

* the analytic fitted-Gaussian trajectory;
* its isotropic-covariance ablation with the same mean and total variance;
* the straight chord ending at the Gaussian endpoint; and
* an amplitude-matched Brownian bridge around that chord.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


STATISTICS_FORMAT_VERSION = 1

@dataclass(frozen=True)
class GaussianStatistics:
    mean: torch.Tensor
    eigenvectors: torch.Tensor
    eigenvalues: torch.Tensor
    num_samples: int
    rank: int
    total_variance: float | None = None
    dataset: str | None = None
    preprocessing_id: str | None = None
    flatten_layout: str | None = None
    source_split: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    fit_config: dict[str, Any] = field(default_factory=dict)


def load_gaussian_statistics(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    rank: int | None = None,
    allow_rank_truncation: bool = False,
) -> GaussianStatistics:
    """Load a unified Gaussian-statistics artifact."""
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"invalid statistics artifact: {path}")
    if payload.get("format_version") != STATISTICS_FORMAT_VERSION:
        raise ValueError(
            f"unsupported statistics format version: {payload.get('format_version')!r}"
        )
    if payload.get("artifact") != "gaussian_statistics":
        raise ValueError(f"not a Gaussian-statistics artifact: {path}")
    if "global_stats" not in payload:
        raise ValueError("statistics artifact is missing global_stats")
    values = payload["global_stats"]
    available = int(values.get("rank", len(values["eigenvalues"])))
    requested = available if rank is None else int(rank)
    if requested > available and not allow_rank_truncation:
        raise ValueError(
            f"requested rank {requested} exceeds artifact rank {available}; "
            "set allow_rank_truncation=True to opt in to truncation"
        )
    selected = min(requested, available)
    if selected < 1:
        raise ValueError("rank must be positive")
    mean = torch.as_tensor(values["mean"], dtype=torch.float32, device=device)
    vectors = torch.as_tensor(
        values["eigenvectors"][:, :selected], dtype=torch.float32, device=device
    )
    eigenvalues = torch.as_tensor(
        values["eigenvalues"][:selected], dtype=torch.float32, device=device
    )
    if vectors.shape != (mean.numel(), selected):
        raise ValueError("statistics dimensions are inconsistent")
    if eigenvalues.shape != (selected,):
        raise ValueError("statistics eigenvalue count is inconsistent with rank")
    if not torch.isfinite(mean).all():
        raise ValueError("statistics mean contains non-finite values")
    if not torch.isfinite(vectors).all():
        raise ValueError("statistics eigenvectors contain non-finite values")
    if torch.any(eigenvalues < 0) or not torch.isfinite(eigenvalues).all():
        raise ValueError("eigenvalues must be finite and nonnegative")
    if torch.any(torch.diff(eigenvalues) > 1e-6):
        raise ValueError("eigenvalues must be in descending order")
    num_samples = int(values["num_samples"])
    if num_samples < 2:
        raise ValueError("statistics require at least two fitting samples")
    total_variance = float(values.get("total_variance", eigenvalues.sum().cpu()))
    if not math.isfinite(total_variance) or total_variance < 0:
        raise ValueError("total_variance must be finite and nonnegative")
    return GaussianStatistics(
        mean=mean,
        eigenvectors=vectors,
        eigenvalues=eigenvalues,
        num_samples=num_samples,
        rank=selected,
        total_variance=total_variance,
        dataset=payload.get("dataset"),
        preprocessing_id=values.get("preprocessing_id"),
        flatten_layout=values.get("flatten_layout"),
        source_split=values.get("source_split"),
        source_metadata=dict(payload.get("source", {})),
        fit_config=dict(payload.get("fit", {})),
    )


_COMPATIBLE_PROTOCOLS = {
    "cifar10": {
        "trajectory": {"cifar10_edm_model_space_v1"},
        "statistics": {"cifar10_train_rgb_chw_minus1_1_v1"},
        "source_split": {"train"},
    },
    "diffusion_lm": {
        "trajectory": {"diffusion_lm_e2e_tgt_random_embedding_v1"},
        "statistics": {
            "diffusion_lm_e2e_train_checkpoint_embedding_v2+source_xstart_t0_jitter",
        },
        "source_split": {"train_clean_embeddings_checkpoint_matched"},
    },
    "diffwave": {
        "trajectory": {"sc09_audio_16khz_model_space_v1"},
        # The first entry is the corrected name.  The second identifies existing
        # artifacts produced by the same all-digit source protocol.
        "statistics": {
            "sc09_source_all_digits_audio_16000hz_mono_pad_or_trim_16000_v1",
            "sc09_train_audio_16000hz_mono_pad_or_trim_16000_v1",
        },
        "source_split": {"source_protocol_all_digits", "train"},
    },
    "pvd": {
        "trajectory": {"pvd_chair_model_normalized_raw_v1"},
        "statistics": {
            "pvd_chair_train_model_space_uncentered_flattened_exact_resampling_limit_v1"
        },
        "source_split": {"train"},
    },
}


def validate_statistics_compatibility(
    manifest: dict[str, Any],
    statistics: GaussianStatistics,
) -> None:
    """Validate semantic compatibility between trajectories and statistics.

    Trajectory and training-data preprocessing IDs are intentionally different,
    so compatibility is defined by explicit dataset protocols rather than by
    filename, ambient dimension alone, or literal ID equality.
    """
    dataset = str(manifest.get("dataset", ""))
    errors = []
    if statistics.dataset != dataset:
        errors.append(
            f"dataset mismatch: trajectory={dataset!r}, statistics={statistics.dataset!r}"
        )
    dimension = int(manifest.get("latent_dimension", -1))
    if dimension != statistics.mean.numel():
        errors.append(
            f"dimension mismatch: trajectory={dimension}, statistics={statistics.mean.numel()}"
        )
    layout = manifest.get("flatten_layout")
    if layout != statistics.flatten_layout:
        errors.append(
            f"flatten-layout mismatch: trajectory={layout!r}, "
            f"statistics={statistics.flatten_layout!r}"
        )

    trajectory_preprocessing = manifest.get("preprocessing_id")
    if dataset in _COMPATIBLE_PROTOCOLS:
        protocol = _COMPATIBLE_PROTOCOLS[dataset]
        if trajectory_preprocessing not in protocol["trajectory"]:
            errors.append(f"unsupported trajectory preprocessing: {trajectory_preprocessing!r}")
        if statistics.preprocessing_id not in protocol["statistics"]:
            errors.append(
                f"unsupported statistics preprocessing: {statistics.preprocessing_id!r}"
            )
        if statistics.source_split not in protocol["source_split"]:
            errors.append(f"unsupported statistics source split: {statistics.source_split!r}")
    else:
        errors.append(f"no compatibility protocol is registered for dataset {dataset!r}")

    if dataset == "diffusion_lm":
        trajectory_checkpoint = manifest.get("source", {}).get("checkpoint", {})
        statistics_checkpoint = statistics.source_metadata.get("checkpoint_path")
        if not statistics_checkpoint:
            errors.append("Diffusion-LM statistics do not record their checkpoint")
        else:
            statistics_name = Path(statistics_checkpoint).name
            trajectory_name = Path(trajectory_checkpoint.get("path", "")).name
            if statistics_name != trajectory_name:
                errors.append("Diffusion-LM trajectory/statistics checkpoint mismatch")
        recorded_hash = statistics.source_metadata.get("checkpoint_first_mib_sha256")
        trajectory_hash = trajectory_checkpoint.get("first_mib_sha256")
        if not recorded_hash or not trajectory_hash:
            errors.append("Diffusion-LM checkpoint fingerprint is missing")
        elif recorded_hash != trajectory_hash:
            errors.append("Diffusion-LM trajectory/statistics checkpoint hash mismatch")
        jitter = statistics.source_metadata.get("xstart_jitter")
        correction = statistics.fit_config.get("post_fit_covariance_correction")
        if not isinstance(jitter, dict) or not isinstance(correction, dict):
            errors.append("Diffusion-LM statistics omit source timestep-zero jitter")
        else:
            jitter_variance = jitter.get("variance")
            correction_variance = correction.get("variance")
            valid_variance = (
                isinstance(jitter_variance, (int, float))
                and math.isfinite(float(jitter_variance))
                and float(jitter_variance) >= 0
            )
            if (
                not valid_variance
                or correction.get("type") != "isotropic"
                or not isinstance(correction_variance, (int, float))
                or not math.isclose(
                    float(jitter_variance), float(correction_variance),
                    rel_tol=1e-7, abs_tol=1e-12,
                )
            ):
                errors.append("Diffusion-LM timestep-zero jitter metadata is inconsistent")

    if dataset == "pvd":
        if manifest.get("saved_layout") != "point_xyz":
            errors.append("PVD saved layout must be point_xyz")
        if statistics.source_metadata.get("category") != manifest.get("category"):
            errors.append("PVD category mismatch")
        native_shape = manifest.get("native_shape", [])
        expected_points = int(native_shape[0]) if native_shape else None
        if statistics.source_metadata.get("npoints") != expected_points:
            errors.append("PVD point-count mismatch")

    if errors:
        raise ValueError("incompatible trajectory/statistics artifacts:\n- " + "\n- ".join(errors))


@torch.no_grad()
def analytic_gaussian_vp_from_start(
    starts_vp: torch.Tensor,
    schedule: dict[str, torch.Tensor],
    statistics: GaussianStatistics,
) -> torch.Tensor:
    """Evaluate the analytic Gaussian probability-flow path.

    The calculation is performed in VE coordinates.  Retained covariance
    modes use their fitted eigenvalues; the orthogonal complement evolves
    isotropically.  The result is returned in VP coordinates.
    """
    starts = torch.as_tensor(
        starts_vp, dtype=torch.float32, device=statistics.eigenvectors.device
    )
    alpha = torch.as_tensor(schedule["alpha_bar"], dtype=torch.float32, device=starts.device)
    sigma = torch.as_tensor(schedule["sigma_ve"], dtype=torch.float32, device=starts.device)
    mean = statistics.mean

    start_ve = starts / torch.sqrt(alpha[0])
    centered = start_ve - mean
    coefficients = centered @ statistics.eigenvectors
    retained = coefficients @ statistics.eigenvectors.T
    null_velocity = (centered - retained) / sigma[0]
    eigenvalues = statistics.eigenvalues
    scale = torch.sqrt(
        (eigenvalues[None] + sigma[:, None].square())
        / (eigenvalues[None] + sigma[0].square())
    )
    batch, states, rank = len(starts), len(sigma), len(eigenvalues)
    signal = (
        (coefficients[:, None] * scale[None]).reshape(-1, rank)
        @ statistics.eigenvectors.T
    ).reshape(batch, states, -1)
    path_ve = mean + signal + sigma[None, :, None] * null_velocity[:, None]
    path_vp = path_ve * torch.sqrt(alpha)[None, :, None]
    path_vp[:, 0] = starts
    return path_vp


def _start_aligned(path: torch.Tensor) -> torch.Tensor:
    """Translate every path to its first state in float64."""
    path = path.to(dtype=torch.float64)
    return path - path[:, :1]


def _r_factor(path: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compact factor for start-aligned, uniformly weighted Procrustes."""
    weighted = path[:, 1:].to(torch.float64) * torch.sqrt(
        weights[1:].to(torch.float64)
    )[None, :, None]
    _q, r = torch.linalg.qr(weighted.transpose(1, 2), mode="r")
    return r


def _procrustes_l2(
    reference_r: torch.Tensor,
    candidate_r: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Exact ambient-space Procrustes residual without constructing a D x D matrix."""
    left, _, right_h = torch.linalg.svd(
        reference_r @ candidate_r.transpose(-1, -2), full_matrices=False
    )
    residual = reference_r - (left @ right_h) @ candidate_r
    l2 = torch.linalg.vector_norm(residual, dim=-2)
    l2 = l2 / torch.sqrt(weights[1:])[None]
    l2 = torch.cat([
        torch.zeros(len(l2), 1, device=l2.device, dtype=l2.dtype), l2
    ], dim=1)
    return l2


@torch.no_grad()
def controlled_same_start_rmse(
    real_vp: torch.Tensor,
    schedule: dict[str, torch.Tensor],
    statistics: GaussianStatistics,
    *,
    device: torch.device | str,
    batch_size: int = 8,
    space: str = "ve",
) -> dict[str, object]:
    """Compare real paths with Gaussian, isotropic, chord, and bridge baselines.

    Raw errors use same-start paths.  Aligned errors additionally fit one
    orthogonal transformation per real/candidate pair over the complete path.
    The Brownian amplitude is fitted only to the scalar deviation energy of
    the real path from the Gaussian chord; its random shape remains a null.

    The paper protocol is fixed: saved states have equal weight, alignment is
    start-centered, chord progress is linear in the selected noise coordinate,
    the fitted empirical mean/covariance are unscaled, and bridge variance is
    set entirely by amplitude matching. ``space`` is the only analysis choice.
    """
    real_vp = torch.as_tensor(real_vp, dtype=torch.float32).cpu()
    if real_vp.ndim != 3 or real_vp.shape[1] != len(schedule["alpha_bar"]):
        raise ValueError("real_vp must have shape [sample, time, dimension]")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    space = space.lower()
    if space not in {"vp", "ve"}:
        raise ValueError("space must be vp or ve")

    alpha = torch.as_tensor(schedule["alpha_bar"], dtype=torch.float32)
    noise = torch.as_tensor(
        schedule["sigma_ve" if space == "ve" else "noise_std_vp"], dtype=torch.float64
    )
    progress = (noise[0] - noise) / (noise[0] - noise[-1]).clamp_min(1e-12)
    dimension = real_vp.shape[-1]
    if statistics.total_variance is None:
        raise ValueError("isotropic Gaussian requires total_variance")
    isotropic_variance = float(statistics.total_variance) / dimension
    if not math.isfinite(isotropic_variance) or isotropic_variance < 0:
        raise ValueError("isotropic variance must be finite and nonnegative")
    weights = torch.full_like(progress, 1.0 / len(progress), dtype=torch.float64)
    generator = torch.Generator().manual_seed(0)

    names = ("gaussian", "isotropic_gaussian", "gaussian_chord", "brownian")
    curves = {
        f"{name}_{alignment}_{metric}": []
        for name in names
        for alignment in ("raw", "aligned")
        for metric in ("l2", "per_coordinate_rmse", "ratio_to_gaussian_chord")
    }
    bridge_std_parts = []
    for begin in tqdm(range(0, len(real_vp), batch_size), desc=f"RMSE ({space.upper()})"):
        batch_vp = real_vp[begin : begin + batch_size].to(device)
        gaussian_vp = analytic_gaussian_vp_from_start(
            batch_vp[:, 0], schedule, statistics
        )
        # For Sigma = c I, every centered direction has the same analytic
        # scale.  Compute it directly so the ablation is exact even when the
        # fitted Gaussian stores only a low-rank eigensystem.
        alpha_device = alpha.to(device)
        sigma_device = torch.as_tensor(
            schedule["sigma_ve"], dtype=batch_vp.dtype, device=device
        )
        mean = statistics.mean.to(device=device, dtype=batch_vp.dtype)
        start_ve = batch_vp[:, 0] / torch.sqrt(alpha_device[0])
        isotropic_scale = torch.sqrt(
            (isotropic_variance + sigma_device.square())
            / (isotropic_variance + sigma_device[0].square()).clamp_min(1e-30)
        )
        isotropic_ve = mean + isotropic_scale[None, :, None] * (
            start_ve[:, None] - mean
        )
        isotropic_vp = isotropic_ve * torch.sqrt(alpha_device)[None, :, None]
        isotropic_vp[:, 0] = batch_vp[:, 0]
        if space == "ve":
            scale = torch.rsqrt(alpha).to(device)[None, :, None]
            real, gaussian = batch_vp * scale, gaussian_vp * scale
            isotropic_gaussian = isotropic_vp * scale
        else:
            real, gaussian = batch_vp, gaussian_vp
            isotropic_gaussian = isotropic_vp
        real_delta = real - real[:, :1]
        gaussian_delta = gaussian - gaussian[:, :1]
        isotropic_delta = isotropic_gaussian - isotropic_gaussian[:, :1]

        chord_u = progress.float()[None].expand(len(real), -1)
        gaussian_chord = chord_u.to(
            device=device, dtype=gaussian_delta.dtype
        )[:, :, None] * gaussian_delta[:, -1:, :]

        # Match E[||sigma B(u)||^2] to the weighted real/chord deviation energy.
        deviation = (real_delta - gaussian_chord).square().sum(-1).cpu()
        deviation_energy = (weights[None] * deviation).sum(1)
        bridge_u = chord_u
        bridge_profile = bridge_u * (1 - bridge_u)
        expected_unit_energy = dimension * (weights[None] * bridge_profile).sum(1)
        bridge_std = torch.sqrt(deviation_energy / expected_unit_energy.clamp_min(1e-30))
        bridge_std_parts.append(bridge_std)
        intervals = torch.diff(bridge_u).clamp_min(0)
        random = torch.randn(len(real), len(progress) - 1, dimension, generator=generator)
        motion = torch.cat([
            torch.zeros(len(real), 1, dimension),
            (random * torch.sqrt(intervals)[:, :, None]).cumsum(1),
        ], dim=1)
        standard_bridge = motion - bridge_u[:, :, None] * motion[:, -1:, :]
        brownian = gaussian_chord.cpu() + (
            bridge_std.float()[:, None, None]
            * standard_bridge
        )
        brownian[:, 0] = gaussian_chord[:, 0].cpu()
        brownian[:, -1] = gaussian_chord[:, -1].cpu()
        brownian = brownian.to(device)

        device_weights = weights.to(device)
        reference = _start_aligned(real_delta)
        reference_r = _r_factor(reference, device_weights)
        integrated_errors = {}
        for name, candidate in {
            "gaussian": gaussian_delta,
            "isotropic_gaussian": isotropic_delta,
            "gaussian_chord": gaussian_chord,
            "brownian": brownian,
        }.items():
            raw_l2 = torch.linalg.vector_norm(
                real_delta.to(torch.float64) - candidate.to(torch.float64), dim=-1
            )
            centered = _start_aligned(candidate)
            aligned_l2 = _procrustes_l2(
                reference_r, _r_factor(centered, device_weights), device_weights,
            )
            for alignment, l2 in (("raw", raw_l2), ("aligned", aligned_l2)):
                curves[f"{name}_{alignment}_l2"].append(l2.cpu())
                curves[f"{name}_{alignment}_per_coordinate_rmse"].append(
                    (l2 / math.sqrt(dimension)).cpu()
                )
                integrated_errors[name, alignment] = torch.sqrt(
                    (device_weights[None] * l2.square()).sum(1)
                )
        # Use one common reference for all eight reported values.  In
        # particular, alignment is allowed to improve the Gaussian chord
        # instead of renormalizing that improvement back to one.
        denominator = integrated_errors["gaussian_chord", "raw"].clamp_min(1e-12)
        for alignment in ("raw", "aligned"):
            for name in names:
                curves[f"{name}_{alignment}_ratio_to_gaussian_chord"].append(
                    (integrated_errors[name, alignment] / denominator).cpu()
                )

    result: dict[str, object] = {key: torch.cat(parts) for key, parts in curves.items()}
    result.update({
        "bridge_std_per_sample": torch.cat(bridge_std_parts),
        "progress": progress,
        "weights": weights,
        "noise_levels": noise,
        "space": space,
        "pair_count": len(real_vp),
        "isotropic_variance": isotropic_variance,
    })
    return result


@torch.no_grad()
def regularity_local_coordinates(
    trajectories: torch.Tensor,
    *,
    components: int = 3,
    device: torch.device | str = "cpu",
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit an exact independent PCA basis to every start-aligned trajectory.

    PCA is obtained from the time-by-time Gram matrix, whose size depends on
    the number of saved states rather than the ambient data dimension.  The
    reported fractions are therefore the leading covariance eigenvalues
    divided by the sum of all covariance eigenvalues.  Each axis uses the
    deterministic sign convention that its largest absolute ambient loading
    is positive; this fixes PCA's sign ambiguity without aligning trajectories
    to one another.
    """
    # Small temporal components are easily corrupted when a float32 Gram
    # eigenvalue is used to reconstruct an ambient loading.  Keep the complete
    # PCA path in float64; only the compact [sample,time,component] result is held.
    trajectories = torch.as_tensor(trajectories)
    if trajectories.ndim != 3 or not 1 <= components <= min(trajectories.shape[1:]):
        raise ValueError("trajectories must have shape [sample,time,dimension]")
    if not trajectories.is_floating_point() or not torch.isfinite(trajectories).all():
        raise ValueError("trajectories must be finite floating-point values")
    normalized_weights = None
    if weights is not None:
        normalized_weights = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
        if (
            len(normalized_weights) != trajectories.shape[1]
            or not torch.isfinite(normalized_weights).all()
            or torch.any(normalized_weights < 0)
            or float(normalized_weights.sum()) <= 0
        ):
            raise ValueError("weights must be finite, nonnegative, and match time")
        normalized_weights = normalized_weights / normalized_weights.sum()
    coordinates, explained = [], []
    for source_path in tqdm(trajectories, desc="Exact trajectory PCA"):
        # Convert one path at a time so high-dimensional collections (notably
        # DiffWave) do not require a second full float64 host copy.
        path = source_path.to(device=device, dtype=torch.float64)
        delta = path - path[:1]
        if normalized_weights is None:
            centered = path - path.mean(0, keepdim=True)
        else:
            w = normalized_weights.to(device)
            mean = (w[:, None] * path).sum(0, keepdim=True)
            centered = torch.sqrt(w)[:, None] * (path - mean)

        # X X^T has the same nonzero eigenvalues as the ambient covariance
        # X^T X, but is at most [number of states, number of states].
        gram = centered @ centered.T
        eigenvalues, left_vectors = torch.linalg.eigh(gram)
        top_values = eigenvalues[-components:].flip(0).clamp_min(0)
        top_left_vectors = left_vectors[:, -components:].flip(1)

        # Recover only the requested ambient eigenvectors: v_k = X^T u_k / sqrt(lambda_k).
        norms = torch.sqrt(top_values)
        tolerance = torch.finfo(path.dtype).eps * norms.max().clamp_min(1)
        valid = norms > tolerance
        vectors = centered.T @ top_left_vectors
        vectors[:, valid] /= norms[valid]
        vectors[:, ~valid] = 0

        # Fix each PCA axis independently using only that trajectory.  Without
        # this convention, eigensolver-specific signs can make independent
        # bases appear accidentally aligned or arbitrarily reflected.
        largest_loading_rows = vectors.abs().argmax(dim=0)
        axis_signs = torch.sign(
            vectors[largest_loading_rows, torch.arange(components, device=path.device)]
        )
        axis_signs[axis_signs == 0] = 1
        vectors *= axis_signs

        total_variance = gram.diagonal().sum().clamp_min(1e-12)
        coordinates.append((delta @ vectors).cpu())
        explained.append((top_values / total_variance).cpu())
    return torch.stack(coordinates), torch.stack(explained)


@torch.no_grad()
def align_regularity_coordinates(
    coordinates: torch.Tensor,
    *,
    reference_index: int = 0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Orthogonally align local-PCA paths without scaling or translation.

    A general orthogonal map is intentional here because independently fitted
    PCA axes have arbitrary signs as well as arbitrary orientations.
    """
    coordinates = torch.as_tensor(coordinates, dtype=torch.float32).cpu()
    if not 0 <= reference_index < len(coordinates):
        raise IndexError(reference_index)
    if weights is None:
        weights = torch.full((coordinates.shape[1],), 1 / coordinates.shape[1])
    else:
        weights = torch.as_tensor(weights, dtype=torch.float32).reshape(-1)
        weights = weights / weights.sum()
    reference = coordinates[reference_index]
    result = []
    for source in coordinates:
        left, _, right_h = torch.linalg.svd(source.T @ (weights[:, None] * reference))
        result.append(source @ (left @ right_h))
    return torch.stack(result)


def summarize_curves(curves: torch.Tensor) -> dict[str, torch.Tensor]:
    curves = torch.as_tensor(curves, dtype=torch.float32)
    return {
        "mean": curves.mean(0),
        "q25": torch.quantile(curves, 0.25, dim=0),
        "q75": torch.quantile(curves, 0.75, dim=0),
    }
