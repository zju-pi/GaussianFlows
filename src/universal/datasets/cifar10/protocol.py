"""CIFAR-10 sampling protocol adapted from diff-sampler/gits-main.

The reference ``test.sh`` uses the Euler sampler with 101 polynomial/Karras
noise levels (rho=7), AFS disabled, and no final denoise-to-zero step.  This
module preserves those updates but removes the source solver's hard-coded
``./data/traj_*`` write side effect.
"""

from __future__ import annotations

import torch


class StackedRandomGenerator:
    """One torch.Generator per sample so results do not depend on batching."""

    def __init__(self, device: torch.device | str, seeds: list[int] | torch.Tensor):
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32))
            for seed in seeds
        ]

    def randn(self, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError("leading tensor dimension must equal the seed count")
        return torch.stack(
            [torch.randn(size[1:], generator=generator, **kwargs)
             for generator in self.generators]
        )

    def randint(self, *args, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError("leading tensor dimension must equal the seed count")
        return torch.stack(
            [torch.randint(*args, size=size[1:], generator=generator, **kwargs)
             for generator in self.generators]
        )


def polynomial_schedule(
    num_states: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Exact ``solver_utils.get_schedule(..., schedule_type='polynomial')``."""
    if num_states < 2:
        raise ValueError("num_states must be at least two")
    if not 0 < sigma_min < sigma_max:
        raise ValueError("expected 0 < sigma_min < sigma_max")
    if rho <= 0:
        raise ValueError("rho must be positive")
    indices = torch.arange(num_states, device=device, dtype=torch.float32)
    return (
        sigma_max ** (1.0 / rho)
        + indices / (num_states - 1)
        * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho


@torch.no_grad()
def euler_trajectory(
    net,
    latents: torch.Tensor,
    sigma_ve: torch.Tensor,
    *,
    class_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return source-protocol VE states as ``[batch, time, C, H, W]``."""
    sigma_ve = torch.as_tensor(
        sigma_ve, device=latents.device, dtype=torch.float32
    ).reshape(-1)
    if len(sigma_ve) < 2 or torch.any(torch.diff(sigma_ve) >= 0):
        raise ValueError("Euler schedule must be strictly decreasing")

    current = latents * sigma_ve[0]
    states = [current]
    for sigma_current, sigma_next in zip(sigma_ve[:-1], sigma_ve[1:]):
        denoised = net(current, sigma_current, class_labels=class_labels)
        derivative = (current - denoised) / sigma_current
        current = current + (sigma_next - sigma_current) * derivative
        states.append(current)
    return torch.stack(states, dim=1)

