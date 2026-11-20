import torch

from universal.datasets.diffusion_lm.protocol import (
    deterministic_ddim_trajectory,
    schedule_from_alpha,
    select_ddim_stride_timesteps,
)


def test_ddim200_is_stride_ten():
    timesteps = select_ddim_stride_timesteps(2000, 200)
    assert torch.equal(timesteps, torch.arange(1990, -1, -10))


def test_schedule_ve_vp_identity():
    alpha = torch.tensor([0.01, 0.25, 0.75, 1.0], dtype=torch.float64)
    schedule = schedule_from_alpha(alpha, torch.tensor([9, 5, 1]))
    assert torch.allclose(schedule.alpha_bar, 1 / (1 + schedule.sigma_ve.square()))


def test_deterministic_ddim_retains_initial_and_clean_states():
    class DummyDiffusion:
        alphas_cumprod = torch.tensor([0.9, 0.6, 0.3, 0.1]).numpy()

        def p_mean_variance(self, model, x, t, **kwargs):
            return {"pred_xstart": torch.zeros_like(x)}

    noise = torch.randn(2, 4, 3)
    trajectory, alpha = deterministic_ddim_trajectory(
        DummyDiffusion(), object(), noise, torch.tensor([3, 1])
    )
    assert trajectory.shape == (2, 3, 4, 3)
    assert torch.equal(trajectory[:, 0], noise)
    assert alpha[-1] == 1
