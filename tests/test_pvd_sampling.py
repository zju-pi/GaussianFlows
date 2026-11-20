import torch

from universal.datasets.pvd.sample import build_parser
from universal.datasets.pvd.protocol import (
    schedule_from_alpha,
    select_native_ddim_timesteps,
    select_uniform_vp_timesteps,
)


def test_sampling_defaults_to_raw_model_states():
    args = build_parser().parse_args(["--output-dir", "unused"])
    assert args.sampler == "ddim"
    assert args.num_states == 201
    assert args.schedule_mode == "native"


def test_uniform_vp_schedule_is_unique_and_consistent():
    alpha = torch.linspace(0.9999, 0.001, 1000)
    timesteps = select_uniform_vp_timesteps(alpha, num_states=101)
    schedule = schedule_from_alpha(alpha, timesteps)
    assert len(torch.unique(timesteps)) == 100
    assert torch.all(torch.diff(timesteps) < 0)
    assert timesteps[0] == 999 and timesteps[-1] == 0
    assert schedule.alpha_bar[-1] == 1
    assert schedule.sigma_ve[-1] == 0
    assert torch.allclose(
        schedule.alpha_bar,
        1.0 / (1.0 + schedule.sigma_ve.square()),
        rtol=2e-5, atol=1e-7,
    )


def test_single_update_native_ddim_starts_at_terminal_timestep():
    assert select_native_ddim_timesteps(1000, 1).tolist() == [999]
    assert select_native_ddim_timesteps(1000, 200)[0] == 999
    assert select_native_ddim_timesteps(1000, 200)[-1] == 0

