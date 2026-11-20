"""Offline independent-start trajectory-RMSE analysis.

The target real trajectory ``i`` is compared with candidates initialized from
another sample ``j``.  Every path is translated to its own start, and one
orthogonal Procrustes map is fitted over all saved states.  Expensive pairwise
errors are saved once; notebooks only load the resulting matrices and plot.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from tqdm import tqdm

from universal.artifacts.trajectory import save_manifest

from .artifacts import load_trajectory_run
from .trajectory import (
    analytic_gaussian_vp_from_start,
    load_gaussian_statistics,
    validate_statistics_compatibility,
)


FORMAT_VERSION = 1
METHODS = (
    "real",
    "gaussian",
    "isotropic_gaussian",
    "gaussian_chord",
    "brownian_bridge",
)


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _torch_load(path: Path):
    """Load trusted local analysis artifacts across supported PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        return torch.load(path, map_location="cpu")


def _prepare_real_vp(
    manifest: dict, schedule: dict, trajectories: torch.Tensor
) -> torch.Tensor:
    """Convert stored trajectories to flattened VP coordinates."""
    alpha = torch.as_tensor(schedule["alpha_bar"], dtype=torch.float32)
    stored_space = str(manifest["stored_space"]).lower()
    if stored_space == "ve":
        trajectories = trajectories.float() * torch.sqrt(alpha)[None, :, None]
    elif stored_space == "vp":
        trajectories = trajectories.float()
    else:
        raise ValueError(f"unsupported stored space: {stored_space!r}")

    # PVD shards use [point, xyz], whereas its covariance artifact uses the
    # coordinate-major [xyz, point] layout.
    if manifest["dataset"] == "pvd":
        points = int(manifest["native_shape"][0])
        trajectories = trajectories.reshape(
            len(trajectories), len(alpha), points, 3
        )
        trajectories = trajectories.permute(0, 1, 3, 2).flatten(2)
    return trajectories.contiguous()


def compact_factor(paths: torch.Tensor) -> torch.Tensor:
    """Return an exact compact factor for start-aligned, equally weighted paths."""
    _, num_states, _ = paths.shape
    centered = paths.double() - paths[:, :1].double()
    weighted = centered[:, 1:] / math.sqrt(num_states)
    _unused_q, factor = torch.linalg.qr(weighted.transpose(1, 2), mode="r")
    return factor


def brownian_paths(
    real_delta: torch.Tensor,
    chord: torch.Tensor,
    progress: torch.Tensor,
    *,
    first_index: int,
    seed: int,
) -> torch.Tensor:
    """Construct bridges whose amplitude uses only real donor path ``j``."""
    batch_size, num_states, dimension = real_delta.shape
    deviation_energy = (real_delta - chord).square().sum(-1).mean(1)
    expected_energy = dimension * (progress * (1 - progress)).mean()
    bridge_std = torch.sqrt(deviation_energy / expected_energy.clamp_min(1e-30))
    intervals = torch.diff(progress).clamp_min(0).cpu().float()

    bridges = []
    for local_index in range(batch_size):
        generator = torch.Generator().manual_seed(seed + first_index + local_index)
        increments = torch.randn(
            num_states - 1,
            dimension,
            generator=generator,
            dtype=torch.float32,
        ) * torch.sqrt(intervals)[:, None]
        motion = torch.cat(
            [torch.zeros(1, dimension), torch.cumsum(increments, dim=0)], dim=0
        )
        standard_bridge = motion - progress.cpu().float()[:, None] * motion[-1:]
        bridge = (
            chord[local_index].cpu()
            + bridge_std[local_index].cpu().float() * standard_bridge
        )
        bridge[0] = chord[local_index, 0].cpu()
        bridge[-1] = chord[local_index, -1].cpu()
        bridges.append(bridge)
    return torch.stack(bridges).to(real_delta.device)


@torch.no_grad()
def build_factors(
    real_vp: torch.Tensor,
    schedule: dict,
    statistics,
    *,
    device: torch.device,
    batch_size: int,
    brownian_seed: int,
) -> dict[str, torch.Tensor]:
    """Construct all comparison paths once and retain compact exact factors."""
    if statistics.total_variance is None:
        raise ValueError("isotropic Gaussian requires total_variance")
    num_trajectories, num_states, dimension = real_vp.shape
    alpha = torch.as_tensor(schedule["alpha_bar"], dtype=torch.float32)
    sigma = torch.as_tensor(schedule["sigma_ve"], dtype=torch.float32)
    progress = (sigma[0].double() - sigma.double()) / (
        sigma[0].double() - sigma[-1].double()
    ).clamp_min(1e-12)
    ve_scale = torch.rsqrt(alpha).to(device)[None, :, None]
    alpha_device = alpha.to(device)
    sigma_device = sigma.to(device)
    mean = statistics.mean.to(device=device, dtype=torch.float32)
    isotropic_variance = float(statistics.total_variance) / dimension
    isotropic_scale = torch.sqrt(
        (isotropic_variance + sigma_device.square())
        / (isotropic_variance + sigma_device[0].square()).clamp_min(1e-30)
    )
    parts = {method: [] for method in METHODS}

    for begin in tqdm(
        range(0, num_trajectories, batch_size), desc="trajectory factors"
    ):
        batch_vp = real_vp[begin : begin + batch_size].to(device)
        real_ve = batch_vp * ve_scale
        parts["real"].append(compact_factor(real_ve).cpu())

        gaussian_vp = analytic_gaussian_vp_from_start(
            batch_vp[:, 0], schedule, statistics
        )
        gaussian_ve = gaussian_vp * ve_scale
        parts["gaussian"].append(compact_factor(gaussian_ve).cpu())

        start_ve = batch_vp[:, 0] / torch.sqrt(alpha_device[0])
        isotropic_ve = mean + isotropic_scale[None, :, None] * (
            start_ve[:, None] - mean
        )
        parts["isotropic_gaussian"].append(compact_factor(isotropic_ve).cpu())

        gaussian_delta = gaussian_ve - gaussian_ve[:, :1]
        chord = progress.to(device=device, dtype=gaussian_delta.dtype)[None, :, None]
        chord = chord * gaussian_delta[:, -1:, :]
        parts["gaussian_chord"].append(compact_factor(chord).cpu())

        real_delta = real_ve - real_ve[:, :1]
        bridge = brownian_paths(
            real_delta,
            chord,
            progress,
            first_index=begin,
            seed=brownian_seed,
        )
        parts["brownian_bridge"].append(compact_factor(bridge).cpu())

    factors = {method: torch.cat(values) for method, values in parts.items()}
    expected = (num_trajectories, num_states - 1, num_states - 1)
    for method, factor in factors.items():
        if factor.shape != expected or not torch.isfinite(factor).all():
            raise RuntimeError(f"invalid {method} factors: {tuple(factor.shape)}")
    return factors


@torch.no_grad()
def factor_distances(
    reference: torch.Tensor,
    comparisons: torch.Tensor,
    *,
    dimension: int,
) -> torch.Tensor:
    """Compute exact per-coordinate RMSE after optimal orthogonal alignment."""
    cross = reference @ comparisons.transpose(-1, -2)
    nuclear_norm = torch.linalg.svdvals(cross).sum(-1)
    squared = (
        reference.square().sum((-2, -1))
        + comparisons.square().sum((-2, -1))
        - 2 * nuclear_norm
    )
    return squared.clamp_min(0).sqrt() / math.sqrt(dimension)


def _empty_checkpoint(num_trajectories: int, provenance: dict) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "artifact": "independent_start_rmse_checkpoint",
        "provenance": provenance,
        "errors": {
            method: torch.full(
                (num_trajectories, num_trajectories),
                float("nan"),
                dtype=torch.float64,
            )
            for method in METHODS
        },
        "completed_targets": torch.zeros(num_trajectories, dtype=torch.bool),
    }


def _selected_targets(argument: str | None, count: int) -> list[int]:
    if argument is None:
        return list(range(count))
    targets = [int(value.strip()) for value in argument.split(",") if value.strip()]
    if not targets or any(value < 0 or value >= count for value in targets):
        raise ValueError(f"targets must lie in [0, {count - 1}]")
    return list(dict.fromkeys(targets))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trajectory-run", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to TRAJECTORY_RUN/analyses/independent_start_rmse",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trajectory-batch-size", type=int, default=4)
    parser.add_argument("--target-batch-size", type=int, default=8)
    parser.add_argument("--pair-batch-size", type=int, default=64)
    parser.add_argument("--brownian-seed", type=int, default=0)
    parser.add_argument(
        "--targets",
        default=None,
        help="optional comma-separated target indices; omitted means all targets",
    )
    parser.add_argument("--rebuild-factors", action="store_true")
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="retain the large compact-factor cache after successful completion",
    )
    parser.set_defaults(handler=run, result_kind="independent-start RMSE")


def run(args: argparse.Namespace) -> Path:
    for name in ("trajectory_batch_size", "target_batch_size", "pair_batch_size"):
        if getattr(args, name) < 1:
            raise ValueError("batch sizes must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    run_path = args.trajectory_run.expanduser().resolve()
    statistics_path = args.statistics.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_path / "analyses" / "independent_start_rmse"
    )
    cache_dir = output_dir / ".cache"
    factors_path = cache_dir / "factors.pt"
    checkpoint_path = cache_dir / "checkpoint.pt"
    result_path = output_dir / "pairwise_errors.pt"
    manifest_path = output_dir / "manifest.json"

    if result_path.exists():
        existing = load_result(result_path)
        recorded = existing["provenance"]
        if (
            Path(recorded["trajectory_run"]).resolve() != run_path
            or Path(recorded["statistics"]).resolve() != statistics_path
            or int(recorded.get("brownian_seed", -1)) != args.brownian_seed
        ):
            raise ValueError(
                "completed result provenance does not match the requested inputs "
                "or Brownian seed"
            )
        print(f"Complete result already exists: {result_path}", flush=True)
        return manifest_path

    manifest, schedule, real, sample_ids = load_trajectory_run(run_path)
    statistics = load_gaussian_statistics(statistics_path, device=device)
    validate_statistics_compatibility(manifest, statistics)
    real_vp = _prepare_real_vp(manifest, schedule, real)
    num_trajectories, num_states, dimension = real_vp.shape
    provenance = {
        "dataset": manifest["dataset"],
        "space": "ve",
        "alignment": "start_centered_orthogonal_procrustes_reflection_allowed",
        "temporal_weighting": "saved_states",
        "num_trajectories": num_trajectories,
        "num_states": num_states,
        "dimension": dimension,
        "trajectory_run": str(run_path),
        "statistics": str(statistics_path),
        "statistics_rank": int(statistics.rank),
        "brownian_seed": args.brownian_seed,
        "brownian_amplitude": "matched_to_X_j_minus_Gaussian_chord_j",
        "sample_ids": sample_ids.tolist(),
        "sampler": manifest.get("sampler", {}),
    }

    if factors_path.exists() and not args.rebuild_factors:
        cached = _torch_load(factors_path)
        if cached.get("provenance") != provenance:
            raise ValueError("cached factor provenance does not match this analysis")
        factors = cached["factors"]
        print(f"Loaded cached factors: {factors_path}", flush=True)
    else:
        factors = build_factors(
            real_vp,
            schedule,
            statistics,
            device=device,
            batch_size=args.trajectory_batch_size,
            brownian_seed=args.brownian_seed,
        )
        _atomic_torch_save({"provenance": provenance, "factors": factors}, factors_path)
        print(f"Saved factor cache: {factors_path}", flush=True)

    if checkpoint_path.exists():
        checkpoint = _torch_load(checkpoint_path)
        if checkpoint.get("provenance") != provenance:
            raise ValueError("checkpoint provenance does not match this analysis")
        print(f"Resuming checkpoint: {checkpoint_path}", flush=True)
    else:
        checkpoint = _empty_checkpoint(num_trajectories, provenance)
    errors = checkpoint["errors"]
    completed = checkpoint["completed_targets"]
    target_order = [
        index
        for index in _selected_targets(args.targets, num_trajectories)
        if not completed[index]
    ]
    all_indices = torch.arange(num_trajectories)

    for target_begin in range(0, len(target_order), args.target_batch_size):
        target_indices = torch.tensor(
            target_order[target_begin : target_begin + args.target_batch_size]
        )
        references = factors["real"][target_indices].to(device)[:, None]
        block_errors = {
            method: torch.empty(
                len(target_indices), num_trajectories, dtype=torch.float64
            )
            for method in METHODS
        }
        for method in METHODS:
            candidates = factors[method]
            for begin in range(0, num_trajectories, args.pair_batch_size):
                indices = all_indices[begin : begin + args.pair_batch_size]
                block_errors[method][:, indices] = factor_distances(
                    references,
                    candidates[indices].to(device)[None],
                    dimension=dimension,
                ).cpu()
        for row, target_index in enumerate(target_indices.tolist()):
            for method in METHODS:
                block_errors[method][row, target_index] = float("nan")
                errors[method][target_index] = block_errors[method][row]
            real_values = block_errors["real"][row]
            mask = all_indices != target_index
            errors["real"][mask, target_index] = real_values[mask]
            completed[target_index] = True
        checkpoint = {
            "format_version": FORMAT_VERSION,
            "artifact": "independent_start_rmse_checkpoint",
            "provenance": provenance,
            "errors": errors,
            "completed_targets": completed,
        }
        _atomic_torch_save(checkpoint, checkpoint_path)
        print(
            f"Completed {int(completed.sum())}/{num_trajectories} targets",
            flush=True,
        )

    if completed.all():
        artifact = {
            "format_version": FORMAT_VERSION,
            "artifact": "independent_start_rmse",
            "provenance": provenance,
            "errors": errors,
            "completed_targets": completed,
        }
        _atomic_torch_save(artifact, result_path)
        save_manifest(
            manifest_path,
            {
                "format_version": FORMAT_VERSION,
                "artifact": "independent_start_rmse",
                "provenance": provenance,
                "files": {
                    "pairwise_errors": result_path.name,
                },
            },
        )
        if not args.keep_cache:
            checkpoint_path.unlink(missing_ok=True)
            factors_path.unlink(missing_ok=True)
            try:
                cache_dir.rmdir()
            except OSError:
                pass
        print(f"Complete result: {result_path}", flush=True)
    else:
        save_manifest(
            manifest_path,
            {
                "format_version": FORMAT_VERSION,
                "artifact": "independent_start_rmse_partial",
                "provenance": provenance,
                "completed_targets": int(completed.sum()),
                "checkpoint": str(checkpoint_path),
            },
        )
        print(
            f"Partial result: {int(completed.sum())}/{num_trajectories} targets",
            flush=True,
        )
    return manifest_path


def load_result(path: str | Path) -> dict:
    """Load and structurally validate a completed pairwise-error artifact."""
    path = Path(path)
    payload = _torch_load(path)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported independent-start artifact format")
    if payload.get("artifact") != "independent_start_rmse":
        raise ValueError(f"not an independent-start RMSE artifact: {path}")
    errors = payload.get("errors", {})
    if set(errors) != set(METHODS):
        raise ValueError("independent-start artifact has unexpected methods")
    shape = errors["real"].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("pairwise errors must be square matrices")
    for method, values in errors.items():
        if values.shape != shape:
            raise ValueError(f"inconsistent pairwise matrix for {method}")
    return payload

