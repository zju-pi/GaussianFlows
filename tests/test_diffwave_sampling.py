import json
import tempfile
from pathlib import Path

import torch

from universal.artifacts.trajectory import (
    save_manifest,
    save_trajectory_shard,
    validate_schedule,
)
from universal.datasets.diffwave.protocol import (
    build_discrete_schedule,
    ddim_vp_trajectory,
    ddim_ve_trajectory,
    seeded_latents,
    terminal_latent_scale,
    training_alpha_bar,
    ve_to_vp_trajectory,
    vp_to_ve_trajectory,
)


class ZeroEpsilon:
    def __call__(self, inputs, mel_spec=None):
        x, _timesteps = inputs
        return torch.zeros_like(x)


def _training_schedule():
    return training_alpha_bar(200, 1e-4, 2e-2)


def test_source_schedule_matches_upstream_selection():
    schedule = build_discrete_schedule(_training_schedule(), num_states=201, mode="native")
    expected = torch.arange(199, -1, -1, dtype=torch.long)
    assert torch.equal(schedule.timesteps, expected)
    assert len(torch.unique(schedule.timesteps)) == 200
    assert schedule.alpha_bar[-1] == 1
    assert schedule.sigma_ve[-1] == 0
    validate_schedule({
        "sigma_ve": schedule.sigma_ve,
        "alpha_bar": schedule.alpha_bar,
        "noise_std_vp": schedule.noise_std_vp,
    })


def test_uniform_vp_schedule_is_unique_and_ordered():
    schedule = build_discrete_schedule(
        _training_schedule(), num_states=101, mode="uniform_vp_noise_std"
    )
    assert torch.all(torch.diff(schedule.timesteps) < 0)
    assert len(torch.unique(schedule.timesteps)) == 100
    assert schedule.timesteps[0] == 199
    assert schedule.timesteps[-1] == 0
    assert torch.all(torch.diff(schedule.noise_std_vp) <= 0)


def test_per_seed_latents_are_batch_invariant():
    together = seeded_latents([40, 41], (1, 8), device="cpu")
    separate = torch.cat([
        seeded_latents([seed], (1, 8), device="cpu") for seed in (40, 41)
    ])
    assert torch.equal(together, separate)


def test_trajectory_shape_first_state_and_finiteness():
    schedule = build_discrete_schedule(_training_schedule(), num_states=6, mode="native")
    latents = seeded_latents([40, 41], (1, 16), device="cpu")
    trajectory = ddim_ve_trajectory(ZeroEpsilon(), latents, schedule)
    assert trajectory.shape == (2, 6, 1, 16)
    assert torch.equal(trajectory[:, 0], latents * schedule.sigma_ve[0])
    assert torch.isfinite(trajectory).all()
    assert torch.equal(trajectory[:, -1], trajectory[:, 0])


def test_vp_ddim_uses_standard_normal_initial_state():
    schedule = build_discrete_schedule(_training_schedule(), num_states=6, mode="native")
    latents = seeded_latents([40, 41], (1, 16), device="cpu")
    trajectory = ddim_vp_trajectory(ZeroEpsilon(), latents, schedule)
    assert trajectory.shape == (2, 6, 1, 16)
    assert torch.equal(trajectory[:, 0], latents)
    assert torch.isfinite(trajectory).all()


def test_terminal_latent_scales_match_forward_variance():
    schedule = build_discrete_schedule(_training_schedule(), num_states=201, mode="native")
    alpha = float(schedule.alpha_bar[0])
    assert terminal_latent_scale(schedule, mode="standard_normal_vp") == 1.0
    noise_scale = terminal_latent_scale(schedule, mode="terminal_noise_matched")
    assert abs(noise_scale**2 - (1.0 - alpha)) < 1e-7
    data_variance = 0.0072
    matched_scale = terminal_latent_scale(
        schedule, mode="terminal_variance_matched", data_variance=data_variance
    )
    assert abs(matched_scale**2 - ((1.0 - alpha) + alpha * data_variance)) < 1e-7
    assert noise_scale < matched_scale < 1.0


def test_ve_to_vp_trajectory_matches_schedule_identity():
    schedule = build_discrete_schedule(
        _training_schedule(), num_states=6, mode="uniform_vp_noise_std"
    )
    trajectory_ve = torch.ones(2, 6, 1, 8)
    trajectory_vp = ve_to_vp_trajectory(trajectory_ve, schedule)
    expected = torch.sqrt(schedule.alpha_bar).reshape(1, 6, 1, 1)
    assert torch.allclose(trajectory_vp, expected.expand_as(trajectory_vp))
    assert torch.equal(trajectory_vp[:, -1], trajectory_ve[:, -1])
    assert torch.allclose(
        vp_to_ve_trajectory(trajectory_vp, schedule), trajectory_ve
    )


def test_diffwave_shard_and_manifest_round_trip():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        trajectory = torch.randn(2, 3, 1, 16)
        shard_path = root / "real/shard_00000.pt"
        save_trajectory_shard(
            shard_path,
            trajectory,
            torch.tensor([40, 41]),
            num_states=3,
            sample_ids=torch.tensor([0, 1]),
        )
        payload = torch.load(shard_path, map_location="cpu")
        assert torch.equal(payload["trajectory"], trajectory)
        assert payload["seeds"].tolist() == [40, 41]
        assert payload["sample_ids"].tolist() == [0, 1]

        manifest_path = root / "manifest.json"
        save_manifest(manifest_path, {"sample_ids": [0, 1], "rng_seeds": [40, 41]})
        with manifest_path.open(encoding="utf-8") as handle:
            assert json.load(handle)["rng_seeds"] == [40, 41]
