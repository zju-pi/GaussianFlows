"""Schedule selection and trajectory transforms for PVD sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch


SCHEDULE_MODES = ("native", "uniform_vp_noise_std")
SAMPLERS = ("edm", "ddim")


@dataclass(frozen=True)
class PvdSchedule:
    timesteps: torch.Tensor
    alpha_bar: torch.Tensor
    sigma_ve: torch.Tensor
    noise_std_vp: torch.Tensor


def select_native_ddim_timesteps(
    num_train_timesteps: int, num_steps: int
) -> torch.Tensor:
    """Match the source DDIM grid, including the one-update edge case."""
    if not 1 <= num_steps <= num_train_timesteps:
        raise ValueError("num_steps must be in [1, num_train_timesteps]")
    if num_steps == 1:
        return torch.tensor([num_train_timesteps - 1], dtype=torch.int64)
    ascending = torch.linspace(
        0, num_train_timesteps - 1, num_steps, dtype=torch.float64
    )
    timesteps = torch.round(ascending).to(torch.int64).flip(0)
    if len(torch.unique(timesteps)) != num_steps:
        raise ValueError("native DDIM timestep schedule contains duplicates")
    return timesteps


def schedule_from_alpha(
    alpha_bar_train: torch.Tensor,
    timesteps: torch.Tensor,
) -> PvdSchedule:
    alpha_train = torch.as_tensor(alpha_bar_train, dtype=torch.float32).cpu()
    steps = torch.as_tensor(timesteps, dtype=torch.int64).cpu()
    selected = alpha_train[steps]
    alpha = torch.cat([selected, torch.ones(1, dtype=torch.float32)])
    noise = torch.sqrt((1.0 - alpha).clamp_min(0.0))
    sigma = noise / torch.sqrt(alpha)
    return PvdSchedule(steps, alpha, sigma, noise)


def select_uniform_vp_timesteps(
    alpha_bar_train: torch.Tensor, num_states: int = 101
) -> torch.Tensor:
    alpha = torch.as_tensor(alpha_bar_train, dtype=torch.float64).reshape(-1)
    count = num_states - 1
    if count < 1 or count > len(alpha):
        raise ValueError("num_states must be in [2, num_train_timesteps + 1]")
    if not torch.all(torch.diff(alpha) < 0):
        raise ValueError("training alpha_bar must be strictly decreasing")
    coordinate = torch.sqrt((1.0 - alpha).clamp_min(0.0))
    targets = torch.linspace(
        float(coordinate[-1]), float(coordinate[0]), count, dtype=torch.float64
    )
    chosen: list[int] = []
    upper = len(alpha) - 1
    for position, target in enumerate(targets):
        remaining = count - position - 1
        candidates = torch.arange(remaining, upper + 1)
        index = int(candidates[torch.argmin(torch.abs(coordinate[candidates] - target))])
        chosen.append(index)
        upper = index - 1
    result = torch.tensor(chosen, dtype=torch.int64)
    if len(result) > 1 and not torch.all(torch.diff(result) < 0):
        raise RuntimeError("selected PVD timesteps are not strictly decreasing")
    return result


@torch.no_grad()
def uniform_ddim_trajectory(
    diffusion,
    denoise_fn,
    noise: torch.Tensor,
    schedule: PvdSchedule,
) -> torch.Tensor:
    """Run deterministic DDIM over explicitly selected training timesteps."""
    current = noise
    states = [current]
    steps = schedule.timesteps.tolist()
    for index, timestep in enumerate(steps):
        previous = steps[index + 1] if index + 1 < len(steps) else -1
        t = torch.full(
            (len(noise),), timestep, dtype=torch.int64, device=noise.device
        )
        current = diffusion.ddim_sample(
            denoise_fn, current, t, previous, clip_denoised=False, eta=0.0
        ).detach()
        states.append(current)
    return torch.stack(states, dim=1)


@torch.no_grad()
def uniform_edm_trajectory(
    diffusion,
    denoise_fn,
    noise: torch.Tensor,
    schedule: PvdSchedule,
    *,
    solver: str = "heun",
) -> torch.Tensor:
    """Run the source VE ODE solver at uniform-VP-selected sigmas."""
    if solver not in ("euler", "heun"):
        raise ValueError("EDM solver must be euler or heun")
    sigma = schedule.sigma_ve.to(device=noise.device, dtype=noise.dtype)
    current = noise * sigma[0]
    states = [current]
    for index in range(len(sigma) - 1):
        sigma_current, sigma_next = sigma[index], sigma[index + 1]
        derivative, _, _ = diffusion._edm_derivative(
            denoise_fn, current, sigma_current, clip_denoised=False
        )
        step = sigma_next - sigma_current
        proposal = current + step * derivative
        if solver == "heun" and float(sigma_next) > 0:
            next_derivative, _, _ = diffusion._edm_derivative(
                denoise_fn, proposal, sigma_next, clip_denoised=False
            )
            current = current + step * (0.5 * derivative + 0.5 * next_derivative)
        else:
            current = proposal
        current = current.detach()
        states.append(current)
    return torch.stack(states, dim=1)
