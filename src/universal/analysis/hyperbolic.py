"""Two-eigenmode hyperbolic projection used by the analysis notebooks."""

from __future__ import annotations

import torch

from .trajectory import GaussianStatistics, analytic_gaussian_vp_from_start


def select_hyperbolic_eigenpairs(
    eigenvalues: torch.Tensor,
    *,
    first_index: int = 0,
    reference_index: int | None = None,
    gap_fractions: tuple[float, ...] = (1.0, 0.6, 0.3, 0.1),
) -> torch.Tensor:
    """Select pairs between explicit first and reference spectral indices.

    ``first_index`` is the lowest candidate index and defines the largest
    spectral gap.  ``reference_index`` is the fixed second index.  Intermediate
    first indices are selected to approximate ``gap_fractions`` of that largest
    gap.
    """
    values = torch.as_tensor(eigenvalues, dtype=torch.float64).reshape(-1).cpu()
    reference = min(1000, len(values) - 1) if reference_index is None else reference_index
    if not 0 <= first_index < reference < len(values):
        raise ValueError("indices must satisfy 0 <= first_index < reference_index < rank")
    candidate_indices = torch.arange(first_index, reference)
    if len(gap_fractions) > len(candidate_indices):
        raise ValueError("more gap fractions were requested than available first indices")
    gaps = values[candidate_indices] - values[reference]
    if gaps[0] <= 0:
        raise ValueError("the selected first/reference indices have no positive spectral gap")
    available = torch.ones(len(candidate_indices), dtype=torch.bool)
    selected = []
    for fraction in gap_fractions:
        distance = (gaps - float(fraction) * gaps[0]).abs()
        distance[~available] = torch.inf
        offset = int(distance.argmin())
        index = int(candidate_indices[offset])
        selected.append(index)
        available[offset] = False
    return torch.tensor([[index, reference] for index in selected], dtype=torch.long)


@torch.no_grad()
def gaussian_hyperbolic_projection(
    trajectories: torch.Tensor,
    sigma_ve: torch.Tensor,
    statistics: GaussianStatistics,
    eigenpairs: torch.Tensor,
    *,
    signal_scale: torch.Tensor | None = None,
    sample_count: int = 10,
    min_abs_initial_weight: float = 0.25,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor | float | str]:
    """Project General-SDE paths and the analytic Gaussian onto eigenmodes.

    For ``x_t = s(t) x_0 + s(t) sigma(t) epsilon``, the plotted coordinate is
    ``<x_t-s(t)mu,v_k> / (s(t) w_k)``, where
    ``w_k=<x_T-s(T)mu,v_k>/(s(T)sqrt(lambda_k+sigma(T)^2))``.
    The analytic Gaussian coordinate is therefore
    ``sqrt(lambda_k+sigma(t)^2)``.  VE uses ``s(t)=1`` and VP uses
    ``s(t)=sqrt(alpha_bar(t))``.
    """
    paths = torch.as_tensor(trajectories, dtype=torch.float32).cpu()
    sigma = torch.as_tensor(sigma_ve, dtype=torch.float32).reshape(-1).cpu()
    scale = (
        torch.ones_like(sigma)
        if signal_scale is None
        else torch.as_tensor(signal_scale, dtype=torch.float32).reshape(-1).cpu()
    )
    pairs = torch.as_tensor(eigenpairs, dtype=torch.long).reshape(-1, 2).cpu()
    if paths.ndim != 3 or paths.shape[1] != len(sigma) or len(scale) != len(sigma):
        raise ValueError("trajectories, sigma, and signal_scale must share time")
    if sample_count < 1 or min_abs_initial_weight < 0:
        raise ValueError("sample_count must be positive and conditioning threshold nonnegative")
    if torch.any(scale <= 0) or not torch.isfinite(scale).all():
        raise ValueError("signal_scale must be positive and finite")
    dimensions = torch.unique(pairs.flatten(), sorted=True)
    work_device = statistics.eigenvectors.device if device is None else torch.device(device)
    vectors = statistics.eigenvectors[:, dimensions].to(work_device)
    values = statistics.eigenvalues[dimensions].to(work_device)
    mean = statistics.mean.to(work_device)

    device_scale = scale.to(work_device)
    device_sigma = sigma.to(work_device)
    start_coefficients = (
        paths[:, 0].to(work_device) - device_scale[0] * mean
    ) @ vectors
    initial_weights = start_coefficients / (
        device_scale[0] * torch.sqrt(values + device_sigma[0].square())
    )
    conditioning = initial_weights.abs().amin(1).cpu()
    finite = torch.isfinite(initial_weights).all(1).cpu()
    # A zero denominator is invalid even when the caller requests threshold 0.
    effective_threshold = max(
        float(min_abs_initial_weight), torch.finfo(initial_weights.dtype).eps
    )
    eligible = torch.nonzero(finite & (conditioning >= effective_threshold)).flatten()
    selected = eligible[:sample_count]
    if len(selected) == 0:
        raise ValueError(
            "no trajectory has finite, nonzero initial weights above the "
            f"conditioning threshold {min_abs_initial_weight}"
        )
    column = {int(index): offset for offset, index in enumerate(dimensions.tolist())}
    pair_columns = torch.tensor([[column[int(i)], column[int(j)]] for i, j in pairs])
    normalized = []
    for sample in selected.tolist():
        centered = paths[sample].to(work_device) - device_scale[:, None] * mean
        projected = centered @ vectors
        normalized.append(
            (projected / (device_scale[:, None] * initial_weights[sample][None])).cpu()
        )
    normalized = torch.stack(normalized)
    real_coordinates = torch.stack(
        [normalized[:, :, columns] for columns in pair_columns], dim=1
    )
    analytic = torch.sqrt(
        values.double().cpu()[None] + sigma.double()[:, None].square()
    )
    gaussian_coordinates = torch.stack(
        [analytic[:, columns] for columns in pair_columns], dim=0
    )
    all_values = statistics.eigenvalues.cpu()
    gaps = torch.tensor([float(all_values[i] - all_values[j]) for i, j in pairs])
    return {
        "pairs": pairs,
        "eigenvalue_gaps": gaps,
        "sample_indices": selected,
        "conditioning_scores": conditioning[selected],
        "real_coordinates": real_coordinates,
        "gaussian_coordinates": gaussian_coordinates,
        "sigma_ve": sigma,
        "signal_scale": scale,
    }


@torch.no_grad()
def ambient_start_procrustes_projection(
    source: torch.Tensor,
    target: torch.Tensor,
    projection_vectors: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Align in ambient space, then return only requested projected coordinates."""
    source = torch.as_tensor(source)
    target = torch.as_tensor(target, device=source.device)
    vectors = torch.as_tensor(projection_vectors, device=source.device)
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError("source and target must match [sample,time,dimension]")
    source = source.to(torch.float64)
    target = target.to(torch.float64)
    vectors = vectors.to(torch.float64)
    source, target = source - source[:, :1], target - target[:, :1]
    if weights is not None:
        weights = torch.as_tensor(weights, dtype=torch.float64, device=source.device)
        weights = weights / weights.sum()
    result = []
    for source_path, target_path in zip(source, target):
        source_fit, target_fit = source_path[1:], target_path[1:]
        if weights is not None:
            root = torch.sqrt(weights[1:])[:, None]
            source_fit, target_fit = source_fit * root, target_fit * root
        source_q, source_r = torch.linalg.qr(source_fit.T, mode="reduced")
        target_q, target_r = torch.linalg.qr(target_fit.T, mode="reduced")
        left, _, right_h = torch.linalg.svd(source_r @ target_r.T, full_matrices=False)
        result.append(source_path @ source_q @ left @ right_h @ (target_q.T @ vectors))
    return torch.stack(result)


def _plane_aware_start_procrustes_projection(
    source: torch.Tensor,
    target: torch.Tensor,
    output_vectors: torch.Tensor,
    fit_vectors: torch.Tensor,
    fit_normalizers: torch.Tensor,
    *,
    rmse_slack: float,
    optimization_steps: int = 30,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Choose a reproducible near-optimal Procrustes map for 2D projections.

    Standard Procrustes can have many equally (or nearly equally) optimal maps
    when the two trajectories occupy low-dimensional subspaces.  This routine
    uses the normalized error in all requested eigenmode pairs as a joint
    tie-breaker, while constraining the full-trajectory RMSE to at most
    ``1 + rmse_slack`` times its Procrustes optimum.  A single map is used for
    every requested pair; no panel is fitted separately.
    """
    if rmse_slack < 0:
        raise ValueError("rmse_slack must be nonnegative")
    source = torch.as_tensor(source, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64, device=source.device)
    output_vectors = torch.as_tensor(
        output_vectors, dtype=torch.float64, device=source.device
    )
    fit_vectors = torch.as_tensor(fit_vectors, dtype=torch.float64, device=source.device)
    normalizers = torch.as_tensor(
        fit_normalizers, dtype=torch.float64, device=source.device
    )
    source = source - source[:, :1]
    target = target - target[:, :1]

    projected_paths = []
    optimal_rmse = []
    selected_rmse = []
    plane_before = []
    plane_after = []
    dimension = source.shape[-1]

    for sample, (source_path, target_path) in enumerate(zip(source, target)):
        source_q, source_r = torch.linalg.qr(source_path[1:].T, mode="reduced")
        target_q, target_r = torch.linalg.qr(target_path[1:].T, mode="reduced")
        left, cross_singular_values, right_h = torch.linalg.svd(
            source_r @ target_r.T, full_matrices=False
        )
        image0 = target_q @ right_h.T @ left.T
        source_coordinates = source_path @ source_q

        # All useful alternatives lie in the span of the standard solution and
        # the displayed covariance directions.  Optimizing there keeps the
        # numerical problem small even when the ambient dimension is large.
        residual_vectors = fit_vectors - image0 @ (image0.T @ fit_vectors)
        residual_basis, residual_size, _ = torch.linalg.svd(
            residual_vectors, full_matrices=False
        )
        tolerance = (
            torch.finfo(torch.float64).eps
            * max(residual_vectors.shape)
            * residual_size.max().clamp_min(1)
        )
        residual_basis = residual_basis[:, residual_size > tolerance]
        basis = torch.cat([image0, residual_basis], dim=1)
        rank = image0.shape[1]
        z0 = torch.cat(
            [
                torch.eye(rank, dtype=torch.float64, device=source.device),
                torch.zeros(
                    len(residual_basis.T), rank, dtype=torch.float64, device=source.device
                ),
            ],
            dim=0,
        )

        target_in_basis = target_path @ basis
        output_in_basis = basis.T @ output_vectors
        fit_in_basis = basis.T @ fit_vectors
        fit_target = target_path @ fit_vectors
        denominator = normalizers[sample]
        source_norm = source_path.square().sum()
        target_norm = target_path.square().sum()

        def scores(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            predicted_fit = source_coordinates @ z.T @ fit_in_basis
            plane_loss = ((predicted_fit - fit_target) / denominator).square().mean()
            cross = (source_coordinates * (target_in_basis @ z)).sum()
            full_squared = (source_norm + target_norm - 2 * cross).clamp_min(0)
            return plane_loss, full_squared

        def retract(z: torch.Tensor) -> torch.Tensor:
            q, r = torch.linalg.qr(z, mode="reduced")
            signs = torch.where(
                torch.diagonal(r) < 0,
                -torch.ones((), dtype=q.dtype, device=q.device),
                torch.ones((), dtype=q.dtype, device=q.device),
            )
            return q * signs

        initial_plane, optimum_squared = scores(z0)
        cap_squared = optimum_squared * (1.0 + float(rmse_slack)) ** 2
        best_z = z0.detach().clone()
        best_plane = initial_plane.detach().clone()
        # Constrained Riemannian descent.  Away from the RMSE boundary, descend
        # the plane objective directly.  Near the boundary, remove the outward
        # component along the full-error gradient and continue approximately
        # tangentially.  Retraction and backtracking keep every accepted iterate
        # orthogonal and feasible, without an arbitrary penalty coefficient.
        step_size = 1.0
        for _ in range(max(int(optimization_steps), 1)):
            with torch.enable_grad():
                variable = best_z.detach().clone().requires_grad_(True)
                plane_objective, full_objective = scores(variable)
                plane_gradient = torch.autograd.grad(
                    plane_objective, variable, retain_graph=True
                )[0]
                full_gradient = torch.autograd.grad(full_objective, variable)[0]
                # In the singular basis of the Procrustes cross-covariance,
                # small singular values identify directions that barely affect
                # whole-path RMSE.  Inverse-curvature preconditioning lets the
                # tie-breaker use those directions instead of spending its
                # budget on already well-determined ones.
                relative_singular_values = (
                    cross_singular_values
                    / cross_singular_values.max().clamp_min(1e-20)
                )
                inverse_curvature = 1 / (relative_singular_values + 1e-6)
                inverse_curvature = inverse_curvature.clamp_max(1e4)
                plane_gradient = (
                    plane_gradient @ left * inverse_curvature
                ) @ left.T

                def tangent_projection(gradient: torch.Tensor) -> torch.Tensor:
                    symmetric = (variable.T @ gradient + gradient.T @ variable) / 2
                    return gradient - variable @ symmetric

                plane_tangent = tangent_projection(plane_gradient)
                full_tangent = tangent_projection(full_gradient)
                direction = -plane_tangent

                # Once the constraint is active, use the component tangent to
                # its level set.  This is what lets the optimizer keep improving
                # the displayed planes after it has spent the permitted budget.
                boundary_active = full_objective >= cap_squared * (1 - 1e-5)
                outward = (full_tangent * direction).sum()
                full_norm_squared = full_tangent.square().sum()
                if boundary_active and outward > 0 and full_norm_squared > 0:
                    direction = direction - outward / full_norm_squared * full_tangent
                direction_norm = torch.linalg.vector_norm(direction)
            if direction_norm <= torch.finfo(torch.float64).eps:
                break
            direction = direction.detach() / direction_norm
            accepted = False
            with torch.no_grad():
                for _ in range(18):
                    step = step_size
                    candidate = retract(best_z + step * direction)
                    candidate_plane, candidate_squared = scores(candidate)
                    feasible = candidate_squared <= cap_squared * (1.0 + 1e-10) + 1e-14
                    if feasible and candidate_plane < best_plane:
                        best_z = candidate
                        best_plane = candidate_plane
                        step_size = min(step_size * 1.25, 4.0)
                        accepted = True
                        break
                    step_size *= 0.5
                if not accepted:
                    break

        final_plane, final_squared = scores(best_z)
        projected_paths.append(source_coordinates @ best_z.T @ output_in_basis)
        optimal_rmse.append(torch.sqrt(optimum_squared / (source_path.numel())))
        selected_rmse.append(torch.sqrt(final_squared / (source_path.numel())))
        plane_before.append(torch.sqrt(initial_plane))
        plane_after.append(torch.sqrt(final_plane))

    optimal_rmse = torch.stack(optimal_rmse)
    selected_rmse = torch.stack(selected_rmse)
    return torch.stack(projected_paths), {
        "optimal_full_rmse": optimal_rmse,
        "selected_full_rmse": selected_rmse,
        "full_rmse_ratio": selected_rmse / optimal_rmse.clamp_min(1e-20),
        "plane_rmse_before": torch.stack(plane_before),
        "plane_rmse_after": torch.stack(plane_after),
    }


@torch.no_grad()
def aligned_gaussian_hyperbolic_projection(
    real_vp: torch.Tensor,
    schedule: dict[str, torch.Tensor],
    statistics: GaussianStatistics,
    eigenpairs: torch.Tensor,
    *,
    space: str = "ve",
    sample_count: int = 10,
    min_abs_initial_weight: float = 0.25,
    alignment_rmse_slack: float | None = None,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor | float | str]:
    """Project real paths after same-start ambient Procrustes alignment.

    The caller supplies trajectories in VP coordinates.  Alignment is fitted
    with equal weight per saved state in the requested VP or VE space.  The
    When ``alignment_rmse_slack`` is set, the displayed planes jointly break
    near-ties among maps within that relative RMSE budget. The subsequent
    normalization follows the General-SDE scale ``s(t)`` so that
    the analytic Gaussian curve is ``sqrt(lambda_k + sigma(t)^2)`` in either
    space.
    """
    space = space.lower()
    if space not in {"vp", "ve"}:
        raise ValueError("space must be vp or ve")
    paths_vp = torch.as_tensor(real_vp, dtype=torch.float32).cpu()
    alpha = torch.as_tensor(schedule["alpha_bar"], dtype=torch.float32).reshape(-1).cpu()
    sigma = torch.as_tensor(schedule["sigma_ve"], dtype=torch.float32).reshape(-1).cpu()
    if paths_vp.ndim != 3 or paths_vp.shape[1] != len(alpha) or len(alpha) != len(sigma):
        raise ValueError("real_vp and schedule must share the time dimension")

    signal_scale = torch.sqrt(alpha) if space == "vp" else torch.ones_like(alpha)
    real_paths = paths_vp if space == "vp" else paths_vp * torch.rsqrt(alpha)[None, :, None]
    result = gaussian_hyperbolic_projection(
        real_paths,
        sigma,
        statistics,
        eigenpairs,
        signal_scale=signal_scale,
        sample_count=sample_count,
        min_abs_initial_weight=min_abs_initial_weight,
        device=device,
    )

    work_device = statistics.eigenvectors.device if device is None else torch.device(device)
    selected = result["sample_indices"]
    gaussian_vp = analytic_gaussian_vp_from_start(
        paths_vp[selected, 0].to(work_device),
        schedule,
        statistics,
    )
    if space == "vp":
        gaussian_paths = gaussian_vp
    else:
        gaussian_paths = gaussian_vp * torch.rsqrt(alpha.to(work_device))[None, :, None]

    pairs = torch.as_tensor(eigenpairs, dtype=torch.long).reshape(-1, 2).cpu()
    dimensions = torch.unique(pairs.flatten(), sorted=True)
    vectors = statistics.eigenvectors[:, dimensions].to(work_device)
    column = {int(index): offset for offset, index in enumerate(dimensions.tolist())}
    pair_columns = torch.tensor(
        [[column[int(i)], column[int(j)]] for i, j in pairs], dtype=torch.long
    )
    mean = statistics.mean.to(work_device)
    device_scale = signal_scale.to(work_device)
    start_projection = real_paths[selected, 0].to(work_device) @ vectors
    mean_projection = mean @ vectors
    centered_start = start_projection - device_scale[0] * mean_projection
    values = statistics.eigenvalues[dimensions].to(work_device)
    device_sigma = sigma.to(work_device)
    initial_weights = centered_start / (
        device_scale[0] * torch.sqrt(values + device_sigma[0].square())
    )
    if alignment_rmse_slack is None:
        aligned_displacements = ambient_start_procrustes_projection(
            real_paths[selected].to(work_device),
            gaussian_paths,
            vectors,
            weights=None,
        )
        result["alignment"] = "start_procrustes"
    else:
        # Repeat shared modes according to the displayed pairs, so every panel
        # contributes equally to the joint tie-break objective.
        fit_columns = pair_columns.flatten().to(work_device)
        fit_vectors = vectors[:, fit_columns]
        fit_weights = initial_weights[:, fit_columns]
        fit_normalizers = (
            device_scale[None, :, None] * fit_weights[:, None, :]
        )
        aligned_displacements, diagnostics = _plane_aware_start_procrustes_projection(
            real_paths[selected].to(work_device),
            gaussian_paths,
            vectors,
            fit_vectors,
            fit_normalizers,
            rmse_slack=float(alignment_rmse_slack),
        )
        result.update({key: value.cpu() for key, value in diagnostics.items()})
        result["alignment"] = "constrained_plane_start_procrustes"
        result["alignment_rmse_slack"] = float(alignment_rmse_slack)
    centered_aligned = (
        start_projection[:, None]
        + aligned_displacements
        - device_scale[None, :, None] * mean_projection
    )
    normalized = centered_aligned / (
        device_scale[None, :, None] * initial_weights[:, None]
    )

    result["real_coordinates"] = torch.stack(
        [normalized[:, :, columns].cpu() for columns in pair_columns], dim=1
    )
    result["space"] = space
    return result
