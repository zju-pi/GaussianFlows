"""Deterministic DDIM schedule and update used by Diffusion-LM sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch


SCHEDULE_MODES = ("ddim200", "ddim1000", "native")


@dataclass(frozen=True)
class SelectedSchedule:
    timesteps: torch.Tensor
    alpha_bar: torch.Tensor
    sigma_ve: torch.Tensor
    noise_std_vp: torch.Tensor


def schedule_from_alpha(alpha_bar: torch.Tensor, timesteps: torch.Tensor) -> SelectedSchedule:
    """Attach equivalent VP and VE noise coordinates to stored states."""
    alpha = torch.as_tensor(alpha_bar, dtype=torch.float64).cpu().reshape(-1)
    steps = torch.as_tensor(timesteps, dtype=torch.int64).cpu().reshape(-1)
    if len(alpha) != len(steps) + 1 or not torch.all(torch.diff(alpha) > 0):
        raise ValueError("alpha_bar must contain one increasing value per state")
    sigma = torch.sqrt((1 - alpha).clamp_min(0) / alpha)
    noise = torch.sqrt((1 - alpha).clamp_min(0))
    return SelectedSchedule(steps, alpha, sigma, noise)


def select_ddim_stride_timesteps(
    num_train_steps: int, num_sampling_steps: int
) -> torch.Tensor:
    """Return the descending fixed-stride grid used by ``ddimN`` respacing."""
    for stride in range(1, num_train_steps):
        ascending = torch.arange(0, num_train_steps, stride, dtype=torch.int64)
        if len(ascending) == num_sampling_steps:
            return ascending.flip(0)
    raise ValueError(
        f"cannot construct ddim{num_sampling_steps} from {num_train_steps} training steps"
    )


@torch.no_grad()
def deterministic_ddim_trajectory(
    diffusion,
    model,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    denoised_fn=None,
    clip_denoised: bool = False,
    model_kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run eta=0 DDIM and retain the initial noise and every updated state."""
    model_kwargs = {} if model_kwargs is None else model_kwargs
    steps = torch.as_tensor(timesteps, dtype=torch.int64).cpu()
    if len(steps) < 1 or (len(steps) > 1 and not torch.all(torch.diff(steps) < 0)):
        raise ValueError("timesteps must be nonempty and strictly decreasing")
    alpha_train = torch.as_tensor(diffusion.alphas_cumprod, dtype=torch.float64)
    alpha_states = torch.cat([alpha_train[steps], torch.ones(1, dtype=torch.float64)])

    current = noise
    states = [current]
    for position, timestep in enumerate(steps.tolist()):
        t = torch.full((len(noise),), timestep, device=noise.device, dtype=torch.long)
        output = diffusion.p_mean_variance(
            model, current, t, clip_denoised=clip_denoised,
            denoised_fn=denoised_fn, model_kwargs=model_kwargs,
        )
        alpha_now = alpha_states[position].to(current)
        alpha_next = alpha_states[position + 1].to(current)
        predicted_clean = output["pred_xstart"]
        epsilon = (current - torch.sqrt(alpha_now) * predicted_clean) / torch.sqrt(1 - alpha_now)
        current = torch.sqrt(alpha_next) * predicted_clean + torch.sqrt(1 - alpha_next) * epsilon
        states.append(current)
    return torch.stack(states, dim=1), alpha_states
