import torch

from universal.artifacts.trajectory import build_schedule, validate_trajectory
from universal.datasets.cifar10.protocol import (
    StackedRandomGenerator,
    euler_trajectory,
    polynomial_schedule,
)


class ZeroDenoiser:
    def __call__(self, x, sigma, class_labels=None):
        return torch.zeros_like(x)


def test_per_seed_rng_is_batch_invariant():
    together = StackedRandomGenerator("cpu", [3, 7]).randn([2, 1, 2, 2])
    separate = torch.cat([
        StackedRandomGenerator("cpu", [seed]).randn([1, 1, 2, 2])
        for seed in (3, 7)
    ])
    assert torch.equal(together, separate)


def test_euler_protocol_and_artifact_shapes():
    sigma = polynomial_schedule(5, 0.1, 2.0, 3.0, device="cpu")
    latents = torch.randn(2, 3, 4, 4)
    trajectory = euler_trajectory(ZeroDenoiser(), latents, sigma)
    assert trajectory.shape == (2, 5, 3, 4, 4)
    assert torch.allclose(trajectory[:, 0], latents * sigma[0])
    # With denoised=0, Euler preserves x/sigma exactly.
    expected = latents[:, None] * sigma[None, :, None, None, None]
    assert torch.allclose(trajectory, expected, rtol=1e-5, atol=1e-6)
    schedule = build_schedule(sigma)
    validate_trajectory(trajectory, torch.tensor([0, 1]), len(schedule["sigma_ve"]))

