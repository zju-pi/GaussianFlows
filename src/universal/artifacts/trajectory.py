"""Canonical on-disk format for sampled trajectories."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


FORMAT_VERSION = 1


def build_schedule(sigma_ve: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build the shared VE/VP schedule metadata from descending VE sigmas."""
    sigma_ve = torch.as_tensor(sigma_ve, dtype=torch.float32).detach().cpu().reshape(-1)
    alpha_bar = 1.0 / (1.0 + sigma_ve.square())
    schedule = {
        "sigma_ve": sigma_ve,
        "alpha_bar": alpha_bar,
        "noise_std_vp": torch.sqrt((1.0 - alpha_bar).clamp_min(0.0)),
    }
    validate_schedule(schedule)
    return schedule


def validate_schedule(schedule: dict[str, torch.Tensor]) -> None:
    sigma = torch.as_tensor(schedule["sigma_ve"], dtype=torch.float64).reshape(-1)
    alpha = torch.as_tensor(schedule["alpha_bar"], dtype=torch.float64).reshape(-1)
    if len(sigma) < 2 or len(alpha) != len(sigma):
        raise ValueError("schedule must contain matching sigma_ve and alpha_bar vectors")
    if not torch.isfinite(sigma).all() or not torch.isfinite(alpha).all():
        raise ValueError("schedule contains non-finite values")
    if torch.any(sigma < 0) or torch.any(torch.diff(sigma) > 1e-7):
        raise ValueError("sigma_ve must be nonnegative and monotonically nonincreasing")
    expected = 1.0 / (1.0 + sigma.square())
    if not torch.allclose(alpha, expected, rtol=2e-6, atol=1e-8):
        raise ValueError("alpha_bar is inconsistent with sigma_ve")


def validate_trajectory(
    trajectory: torch.Tensor,
    seeds: torch.Tensor,
    num_states: int,
) -> None:
    if trajectory.ndim < 3:
        raise ValueError("trajectory must have shape [batch, time, ...]")
    if trajectory.shape[1] != num_states:
        raise ValueError(
            f"trajectory has {trajectory.shape[1]} states, expected {num_states}"
        )
    seeds = torch.as_tensor(seeds).reshape(-1)
    if trajectory.shape[0] != len(seeds):
        raise ValueError("trajectory batch size and seed count differ")
    if not trajectory.is_floating_point() or not torch.isfinite(trajectory).all():
        raise ValueError("trajectory must be a finite floating-point tensor")


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_trajectory_shard(
    path: str | Path,
    trajectory: torch.Tensor,
    seeds: torch.Tensor,
    *,
    num_states: int,
    class_ids: torch.Tensor | None = None,
    sample_ids: torch.Tensor | None = None,
) -> None:
    path = Path(path)
    trajectory = trajectory.detach().to(device="cpu", dtype=torch.float32).contiguous()
    seeds = torch.as_tensor(seeds, dtype=torch.int64).detach().cpu().reshape(-1)
    validate_trajectory(trajectory, seeds, num_states)
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "trajectory": trajectory,
        "seeds": seeds,
    }
    if class_ids is not None:
        class_ids = torch.as_tensor(class_ids, dtype=torch.int64).detach().cpu().reshape(-1)
        if len(class_ids) != len(seeds):
            raise ValueError("class_ids and seeds must have the same length")
        payload["class_ids"] = class_ids
    if sample_ids is not None:
        sample_ids = torch.as_tensor(sample_ids, dtype=torch.int64).detach().cpu().reshape(-1)
        if len(sample_ids) != len(seeds):
            raise ValueError("sample_ids and seeds must have the same length")
        payload["sample_ids"] = sample_ids
    _atomic_torch_save(payload, path)


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)

