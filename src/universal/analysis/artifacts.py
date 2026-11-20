"""Read universal trajectory artifacts with structural validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from universal.artifacts.trajectory import load_manifest, validate_schedule


def _load_tensor_payload(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0, retained for older dataset environments.
        return torch.load(path, map_location="cpu")


def load_trajectory_run(
    run_dir: str | Path,
    *,
    limit: int | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Load a run as ``manifest, schedule, [N,T,D], sample_ids``.

    Native dimensions are flattened only in memory. Shards remain unchanged.
    ``limit`` is useful for interactive notebooks with high-dimensional audio.
    The time dimension is read from the artifact; no sampling-step count is
    assumed here.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    manifest = load_manifest(run_dir / "manifest.json")
    schedule = _load_tensor_payload(run_dir / manifest["schedule"])
    validate_schedule(schedule)
    num_states = len(schedule["sigma_ve"])
    declared_states = int(manifest["num_states"])
    if num_states != declared_states:
        raise ValueError("schedule length does not match manifest num_states")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None")

    trajectories, identifiers = [], []
    loaded = 0
    for record in manifest["shards"]:
        payload = _load_tensor_payload(run_dir / record["path"])
        batch = torch.as_tensor(payload["trajectory"], dtype=torch.float32)
        if batch.ndim < 3 or batch.shape[1] != num_states:
            raise ValueError(f"invalid trajectory shape in {record['path']}: {batch.shape}")
        ids = payload.get("sample_ids", payload["seeds"])
        ids = torch.as_tensor(ids, dtype=torch.int64).reshape(-1)
        if len(ids) != len(batch):
            raise ValueError(f"identifier count mismatch in {record['path']}")
        if int(record.get("count", -1)) != len(batch):
            raise ValueError(f"shard count mismatch in {record['path']}")
        take = len(batch) if limit is None else min(len(batch), limit - loaded)
        if take:
            trajectories.append(batch[:take].flatten(start_dim=2))
            identifiers.append(ids[:take])
            loaded += take
        if limit is not None and loaded >= limit:
            break
    if not trajectories:
        raise ValueError(f"no trajectories found in {run_dir}")
    trajectory = torch.cat(trajectories)
    sample_ids = torch.cat(identifiers)
    if len(torch.unique(sample_ids)) != len(sample_ids):
        raise ValueError("trajectory identifiers must be unique")
    declared_ids = manifest.get("sample_ids", manifest.get("seeds"))
    if declared_ids is not None:
        expected_ids = torch.as_tensor(declared_ids, dtype=torch.int64)[:len(sample_ids)]
        if not torch.equal(sample_ids, expected_ids):
            raise ValueError("loaded trajectory identifiers do not match manifest")
    if limit is None and len(trajectory) != int(manifest["num_trajectories"]):
        raise ValueError("loaded trajectory count does not match manifest")
    if not torch.isfinite(trajectory).all():
        raise ValueError("trajectory contains non-finite values")
    return manifest, schedule, trajectory, sample_ids
