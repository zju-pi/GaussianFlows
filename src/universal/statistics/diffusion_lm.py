"""Source-faithful covariance corrections for Diffusion-LM."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _first_beta(schedule: str, steps: int) -> float:
    if steps < 1:
        raise ValueError("diffusion_steps must be positive")
    if schedule == "linear":
        return 1e-4
    if schedule == "sqrt":
        alpha_bar_0 = 1.0 - math.sqrt(0.0001)
        alpha_bar_1 = 1.0 - math.sqrt(1.0 / steps + 0.0001)
        return min(1.0 - alpha_bar_1 / alpha_bar_0, 0.999)
    if schedule == "cosine":
        def alpha_bar(t: float) -> float:
            return math.cos((t + 0.008) / 1.008 * math.pi / 2.0) ** 2
        return min(1.0 - alpha_bar(1.0 / steps) / alpha_bar(0.0), 0.999)
    raise ValueError(
        f"unsupported Diffusion-LM noise schedule for x_start jitter: {schedule}"
    )


def source_xstart_jitter(training_args_path: str | Path) -> dict[str, Any]:
    """Return the pre-forward x_start jitter used by Diffusion-LM training."""
    path = Path(training_args_path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("training_mode") != "e2e":
        raise ValueError(
            "automatic x_start jitter correction is currently verified only "
            "for Diffusion-LM training_mode='e2e'"
        )
    steps = int(config["diffusion_steps"])
    schedule = str(config["noise_schedule"])
    variance = _first_beta(schedule, steps)
    return {
        "variance": variance,
        "standard_deviation": math.sqrt(variance),
        "diffusion_steps": steps,
        "noise_schedule": schedule,
        "training_args_path": str(path),
        "dataset_noise_level": float(config.get("noise_level", 0.0)),
    }
