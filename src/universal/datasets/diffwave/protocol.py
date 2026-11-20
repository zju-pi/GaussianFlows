"""DiffWave's deterministic VE-space sampling protocol and schedules."""

from __future__ import annotations

from dataclasses import dataclass

import torch


SCHEDULE_MODES = ("native", "uniform_vp_noise_std", "uniform_ve")
STORED_SPACES = ("ve", "vp")
INITIALIZATION_MODES = (
    "standard_normal_vp",
    "terminal_noise_matched",
    "terminal_variance_matched",
)


@dataclass(frozen=True)
class DiscreteSchedule:
    """Selected training timesteps followed by the sampler's clean state."""

    timesteps: torch.Tensor
    alpha_bar: torch.Tensor
    sigma_ve: torch.Tensor
    noise_std_vp: torch.Tensor


def training_alpha_bar(
    num_train_steps: int,
    beta_start: float,
    beta_end: float,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if num_train_steps < 1:
        raise ValueError("num_train_steps must be positive")
    if not 0.0 < beta_start <= beta_end < 1.0:
        raise ValueError("expected 0 < beta_start <= beta_end < 1")
    beta = torch.linspace(
        beta_start, beta_end, num_train_steps, dtype=torch.float32, device=device
    )
    # Match source utils.calc_diffusion_hyperparams operation-for-operation.
    alpha_bar = 1.0 - beta
    for timestep in range(1, num_train_steps):
        alpha_bar[timestep] *= alpha_bar[timestep - 1]
    return alpha_bar


def _ordered_nearest_indices(values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Select unique descending indices nearest to descending targets.

    The constrained greedy bounds explicitly prevent duplicate timesteps while
    retaining enough lower indices for all remaining targets.
    """
    values = values.reshape(-1)
    targets = targets.reshape(-1)
    if len(targets) > len(values):
        raise ValueError("cannot select more unique timesteps than training steps")
    chosen: list[int] = []
    upper = len(values) - 1
    for position, target in enumerate(targets):
        remaining = len(targets) - position - 1
        candidates = torch.arange(remaining, upper + 1, device=values.device)
        offset = torch.argmin(torch.abs(values[candidates] - target))
        index = int(candidates[offset].item())
        chosen.append(index)
        upper = index - 1
    return torch.tensor(chosen, dtype=torch.int64, device=values.device)


def build_discrete_schedule(
    alpha_bar_train: torch.Tensor,
    *,
    num_states: int = 101,
    mode: str = "native",
) -> DiscreteSchedule:
    """Build a reverse-time discrete schedule plus the final clean state."""
    alpha_bar_train = torch.as_tensor(alpha_bar_train, dtype=torch.float32).reshape(-1)
    if mode not in SCHEDULE_MODES:
        raise ValueError(f"unknown schedule mode {mode!r}; choose from {SCHEDULE_MODES}")
    if num_states < 2 or num_states - 1 > len(alpha_bar_train):
        raise ValueError("num_states must be in [2, num_train_steps + 1]")
    if not torch.all(torch.diff(alpha_bar_train) < 0):
        raise ValueError("training alpha_bar must be strictly decreasing")

    num_noise_states = num_states - 1
    if mode == "native":
        # This exactly matches generate.py, including float32 linspace rounding.
        timesteps = torch.linspace(
            len(alpha_bar_train) - 1,
            0,
            num_noise_states,
            dtype=torch.int64,
            device=alpha_bar_train.device,
        )
        if len(torch.unique(timesteps)) != len(timesteps):
            raise ValueError("native linspace produced duplicate timesteps")
    else:
        noise_std = torch.sqrt((1.0 - alpha_bar_train).clamp_min(0.0))
        sigma = noise_std / torch.sqrt(alpha_bar_train)
        coordinate = noise_std if mode == "uniform_vp_noise_std" else sigma
        targets = torch.linspace(
            coordinate[-1], coordinate[0], num_noise_states,
            dtype=coordinate.dtype, device=coordinate.device,
        )
        timesteps = _ordered_nearest_indices(coordinate, targets)

    if len(timesteps) > 1 and not torch.all(torch.diff(timesteps) < 0):
        raise ValueError("selected timesteps must be unique and strictly decreasing")
    selected_alpha = alpha_bar_train[timesteps]
    selected_noise = torch.sqrt((1.0 - selected_alpha).clamp_min(0.0))
    selected_sigma = selected_noise / torch.sqrt(selected_alpha)
    return DiscreteSchedule(
        timesteps=timesteps,
        alpha_bar=torch.cat([selected_alpha, torch.ones_like(selected_alpha[:1])]),
        sigma_ve=torch.cat([selected_sigma, torch.zeros_like(selected_sigma[:1])]),
        noise_std_vp=torch.cat([selected_noise, torch.zeros_like(selected_noise[:1])]),
    )


def seeded_latents(
    rng_seeds: list[int] | torch.Tensor,
    shape: tuple[int, ...],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw one latent per RNG seed, invariant to caller batch partitioning."""
    if not shape:
        raise ValueError("latent shape cannot be empty")
    tensors = []
    for seed in rng_seeds:
        # The source calls torch.manual_seed immediately before torch.randn.
        # Resetting once per sample preserves that global CUDA RNG state as well
        # as making each latent independent of batch partitioning.
        torch.manual_seed(int(seed) % (1 << 32))
        tensors.append(torch.randn(shape, device=device, dtype=dtype))
    if not tensors:
        raise ValueError("at least one RNG seed is required")
    return torch.stack(tensors)


def terminal_latent_scale(
    schedule: DiscreteSchedule,
    *,
    mode: str,
    data_variance: float | None = None,
) -> float:
    """Return the isotropic VP scale used for the initial latent.

    ``standard_normal_vp`` reproduces the released sampler's N(0, I) prior.
    ``terminal_noise_matched`` uses the terminal forward-noise standard
    deviation. ``terminal_variance_matched`` additionally includes the mean
    per-coordinate variance of the training data, matching the average
    variance of q(x_T) without injecting covariance eigen-directions.
    """
    if mode not in INITIALIZATION_MODES:
        raise ValueError(
            f"unknown initialization mode {mode!r}; choose from {INITIALIZATION_MODES}"
        )
    alpha_terminal = float(schedule.alpha_bar[0])
    if mode == "standard_normal_vp":
        return 1.0
    if mode == "terminal_noise_matched":
        return float((1.0 - alpha_terminal) ** 0.5)
    if data_variance is None or not torch.isfinite(torch.tensor(data_variance)):
        raise ValueError("terminal_variance_matched requires finite data_variance")
    if data_variance < 0:
        raise ValueError("data_variance must be nonnegative")
    return float(
        ((1.0 - alpha_terminal) + alpha_terminal * data_variance) ** 0.5
    )


@torch.no_grad()
def ddim_ve_trajectory(
    net,
    latents: torch.Tensor,
    schedule: DiscreteSchedule,
    *,
    condition=None,
) -> torch.Tensor:
    """Return deterministic source-protocol VE states as ``[B,T,C,L]``."""
    sigma = schedule.sigma_ve.to(device=latents.device, dtype=torch.float32)
    timesteps = schedule.timesteps.to(device=latents.device)
    if latents.ndim != 3:
        raise ValueError("latents must have shape [batch, channels, length]")
    if len(sigma) != len(timesteps) + 1:
        raise ValueError("schedule must contain one clean state after its timesteps")

    current = latents * sigma[0]
    states = [current]
    for index, timestep in enumerate(timesteps):
        sigma_current, sigma_next = sigma[index], sigma[index + 1]
        # Preserve the source operation order rather than substituting alpha[index].
        alpha_bar_target = 1.0 / (1.0 + sigma_current.square())
        timestep_index = int(timestep.item())
        diffusion_steps = timestep_index * torch.ones(
            (len(latents), 1), device=latents.device
        )
        epsilon = net(
            (current * torch.sqrt(alpha_bar_target), diffusion_steps),
            mel_spec=condition,
        )
        # Kept to mirror the source CUDA execution path; it is not the update.
        _ = current - sigma_current * epsilon
        current = current + (sigma_next - sigma_current) * epsilon
        states.append(current)
    return torch.stack(states, dim=1)


@torch.no_grad()
def ddim_vp_trajectory(
    net,
    latents: torch.Tensor,
    schedule: DiscreteSchedule,
    *,
    condition=None,
) -> torch.Tensor:
    """Run source-faithful deterministic DDIM in VP coordinates.

    The released DiffWave sampler initializes ``x_T`` as standard Gaussian in
    VP space.  We retain that prior and the source training timesteps, changing
    only the stochastic DDPM transition to its ``eta=0`` DDIM counterpart.
    """
    alpha = schedule.alpha_bar.to(device=latents.device, dtype=torch.float32)
    timesteps = schedule.timesteps.to(device=latents.device)
    if latents.ndim != 3:
        raise ValueError("latents must have shape [batch, channels, length]")
    if len(alpha) != len(timesteps) + 1:
        raise ValueError("schedule must contain one clean state after its timesteps")

    current = latents
    states = [current]
    for index, timestep in enumerate(timesteps):
        alpha_current, alpha_next = alpha[index], alpha[index + 1]
        diffusion_steps = int(timestep.item()) * torch.ones(
            (len(latents), 1), device=latents.device
        )
        epsilon = net((current, diffusion_steps), mel_spec=condition)
        pred_xstart = (
            current - torch.sqrt(1.0 - alpha_current) * epsilon
        ) / torch.sqrt(alpha_current)
        current = (
            torch.sqrt(alpha_next) * pred_xstart
            + torch.sqrt(1.0 - alpha_next) * epsilon
        )
        states.append(current)
    return torch.stack(states, dim=1)


def ve_to_vp_trajectory(
    trajectory: torch.Tensor, schedule: DiscreteSchedule
) -> torch.Tensor:
    """Convert ``[B,T,...]`` VE states to the corresponding VP states."""
    if trajectory.ndim < 3 or trajectory.shape[1] != len(schedule.alpha_bar):
        raise ValueError("trajectory time dimension must match the schedule")
    alpha_sqrt = torch.sqrt(schedule.alpha_bar).to(
        device=trajectory.device, dtype=trajectory.dtype
    )
    shape = [1] * trajectory.ndim
    shape[1] = len(alpha_sqrt)
    return trajectory * alpha_sqrt.reshape(shape)


def vp_to_ve_trajectory(
    trajectory: torch.Tensor, schedule: DiscreteSchedule
) -> torch.Tensor:
    """Convert ``[B,T,...]`` VP states to the corresponding VE states."""
    if trajectory.ndim < 3 or trajectory.shape[1] != len(schedule.alpha_bar):
        raise ValueError("trajectory time dimension must match the schedule")
    alpha_rsqrt = torch.rsqrt(schedule.alpha_bar).to(
        device=trajectory.device, dtype=trajectory.dtype
    )
    shape = [1] * trajectory.ndim
    shape[1] = len(alpha_rsqrt)
    return trajectory * alpha_rsqrt.reshape(shape)
