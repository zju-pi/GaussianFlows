"""Sample SC09 DiffWave/SaShiMi trajectories into universal artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from universal.artifacts.trajectory import (
    FORMAT_VERSION,
    save_manifest,
    save_trajectory_shard,
    validate_schedule,
)
from universal.datasets.diffwave.protocol import (
    INITIALIZATION_MODES,
    SCHEDULE_MODES,
    STORED_SPACES,
    build_discrete_schedule,
    ddim_ve_trajectory,
    ddim_vp_trajectory,
    seeded_latents,
    terminal_latent_scale,
    training_alpha_bar,
    ve_to_vp_trajectory,
    vp_to_ve_trajectory,
)
from universal.runtime import file_fingerprint, parse_seed_list, source_revision


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE_REPO = Path(
    os.environ.get(
        "UNIVERSAL_DIFFWAVE_REPO", PROJECT_ROOT / "third_party/diffwave-sashimi"
    )
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "assets/checkpoints/diffwave/800000.pkl"
)
DEFAULT_STATISTICS = (
    PROJECT_ROOT / "runs/diffwave/statistics/rank4096.pt"
)


def load_average_data_variance(path: Path) -> float:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    values = payload["global_stats"]
    dimension = int(values.get("dimension", torch.as_tensor(values["mean"]).numel()))
    total_variance = float(values["total_variance"])
    if dimension < 1 or total_variance < 0:
        raise ValueError("invalid dimension or total_variance in statistics")
    return total_variance / dimension


def default_checkpoint_dir(source_repo: Path) -> Path:
    return source_repo / "exp/unet_d128_n6_pool_2_expand2_ff2_T200_betaT0.02_uncond/checkpoint"


def resolve_checkpoint(source_repo: Path, checkpoint: Path | None, ckpt_iter: str) -> Path:
    if checkpoint is not None:
        resolved = checkpoint.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved
    directory = default_checkpoint_dir(source_repo)
    if not directory.is_dir():
        raise FileNotFoundError(f"DiffWave checkpoint directory does not exist: {directory}")
    if ckpt_iter == "max":
        iterations = []
        for path in directory.glob("*.pkl"):
            try:
                iterations.append(int(path.stem))
            except ValueError:
                pass
        if not iterations:
            raise FileNotFoundError(f"no numeric .pkl checkpoints in {directory}")
        ckpt_iter = str(max(iterations))
    resolved = (directory / f"{int(ckpt_iter)}.pkl").resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _resolved_source_config(source_repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compose the same Hydra SC09 + SaShiMi config used by generate.py."""
    source_text = str(source_repo.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError as error:
        raise ImportError(
            "DiffWave's native environment must provide hydra-core and omegaconf"
        ) from error
    with initialize_config_dir(version_base=None, config_dir=str(source_repo / "configs")):
        cfg = compose(config_name="config", overrides=["experiment=sc09", "model=sashimi"])
    OmegaConf.resolve(cfg)
    return tuple(
        OmegaConf.to_container(section, resolve=True)
        for section in (cfg.diffusion, cfg.model, cfg.dataset)
    )


def load_model(source_repo: Path, checkpoint: Path, device: torch.device):
    diffusion_cfg, model_cfg, dataset_cfg = _resolved_source_config(source_repo)
    if diffusion_cfg != {"T": 200, "beta_0": 0.0001, "beta_T": 0.02, "beta": None}:
        raise ValueError(f"unexpected resolved SC09 diffusion config: {diffusion_cfg}")
    source_text = str(source_repo.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from models import construct_model

    network = construct_model(model_cfg).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    network.load_state_dict(payload["model_state_dict"])
    network.eval().requires_grad_(False)
    return network, diffusion_cfg, model_cfg, dataset_cfg


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-ids", "--seeds", dest="sample_ids", type=parse_seed_list,
        default=parse_seed_list("0-199"),
        help="sample IDs; source-compatible RNG seeds are sample_id + --seed-offset",
    )
    parser.add_argument("--seed-offset", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-states", type=int, default=201)
    parser.add_argument("--schedule-mode", choices=SCHEDULE_MODES, default="native")
    parser.add_argument(
        "--sampling-space", choices=STORED_SPACES, default="vp",
        help=(
            "coordinate system used for the DDIM recursion; VE uses the pure "
            "noise prior x_T=sigma_T*epsilon"
        ),
    )
    parser.add_argument(
        "--initialization", choices=INITIALIZATION_MODES,
        default="standard_normal_vp",
        help=(
            "initial VP latent: source N(0,I), terminal forward-noise scale, "
            "or terminal average q(x_T) variance"
        ),
    )
    parser.add_argument(
        "--statistics-path", type=Path, default=DEFAULT_STATISTICS,
        help="statistics providing total_variance for terminal_variance_matched",
    )
    parser.add_argument(
        "--stored-space", choices=STORED_SPACES, default="vp",
        help="coordinate space written to shards; sampling is performed in VP",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-repo", type=Path, default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--ckpt-iter", default="max")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample DiffWave/SaShiMi SC09 trajectories in universal format"
    )
    return add_arguments(parser)


@torch.no_grad()
def run(args: argparse.Namespace) -> Path:
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.seed_offset < 0:
        raise ValueError("seed_offset must be nonnegative")
    if args.sampling_space == "ve" and args.initialization != "terminal_noise_matched":
        raise ValueError(
            "VE sampling requires --initialization terminal_noise_matched "
            "(x_T^VE = sigma_T * epsilon)"
        )
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} already exists; choose a new run directory")

    source_repo = args.source_repo.resolve()
    checkpoint = resolve_checkpoint(source_repo, args.checkpoint, args.ckpt_iter)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        # Upstream models/utils.py constructs the timestep embedding with a bare
        # ``.cuda()``. Make that source operation honor the requested device.
        torch.cuda.set_device(device)
    network, diffusion_cfg, model_cfg, dataset_cfg = load_model(source_repo, checkpoint, device)
    train_alpha = training_alpha_bar(
        int(diffusion_cfg["T"]), float(diffusion_cfg["beta_0"]),
        # Upstream constructs this schedule on CPU and only then calls .cuda().
        float(diffusion_cfg["beta_T"]), device="cpu",
    )
    # Keep the native checkpoint schedule on the sampling device.
    schedule_alpha = train_alpha.to(device) if args.schedule_mode == "native" else train_alpha
    discrete = build_discrete_schedule(
        schedule_alpha, num_states=args.num_states, mode=args.schedule_mode
    )
    schedule = {
        "sigma_ve": discrete.sigma_ve.detach().cpu(),
        "alpha_bar": discrete.alpha_bar.detach().cpu(),
        "noise_std_vp": discrete.noise_std_vp.detach().cpu(),
        "timesteps": discrete.timesteps.detach().cpu(),
    }
    validate_schedule(schedule)

    data_variance = None
    statistics_fingerprint = None
    if args.initialization == "terminal_variance_matched":
        statistics_path = args.statistics_path.expanduser().resolve()
        if not statistics_path.is_file():
            raise FileNotFoundError(statistics_path)
        data_variance = load_average_data_variance(statistics_path)
        statistics_fingerprint = file_fingerprint(statistics_path)
    latent_scale = terminal_latent_scale(
        discrete, mode=args.initialization, data_variance=data_variance
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(schedule, output_dir / "schedule.pt")
    real_dir = output_dir / "real"
    real_dir.mkdir()
    rng_seeds = [sample_id + args.seed_offset for sample_id in args.sample_ids]
    audio_length = int(dataset_cfg["segment_length"])
    shard_records = []
    for shard_index, begin in enumerate(
        tqdm(range(0, len(args.sample_ids), args.batch_size), desc="DiffWave trajectories")
    ):
        batch_ids = args.sample_ids[begin : begin + args.batch_size]
        batch_rng_seeds = rng_seeds[begin : begin + args.batch_size]
        base_latents = seeded_latents(
            batch_rng_seeds, (1, audio_length), device=device
        )
        if args.sampling_space == "ve":
            trajectory = ddim_ve_trajectory(network, base_latents, discrete)
            if args.stored_space == "vp":
                trajectory = ve_to_vp_trajectory(trajectory, discrete)
        else:
            trajectory = ddim_vp_trajectory(
                network, base_latents * latent_scale, discrete
            )
            if args.stored_space == "ve":
                trajectory = vp_to_ve_trajectory(trajectory, discrete)
        shard_name = f"shard_{shard_index:05d}.pt"
        save_trajectory_shard(
            real_dir / shard_name,
            trajectory,
            torch.tensor(batch_rng_seeds),
            num_states=args.num_states,
            sample_ids=torch.tensor(batch_ids),
        )
        shard_records.append({
            "path": f"real/{shard_name}",
            "count": len(batch_ids),
            "sample_ids": batch_ids,
            "rng_seeds": batch_rng_seeds,
        })
        del trajectory, base_latents

    manifest = {
        "format_version": FORMAT_VERSION,
        "dataset": "diffwave",
        "artifact": "real_trajectories",
        "num_trajectories": len(args.sample_ids),
        "num_states": args.num_states,
        "native_shape": [1, audio_length],
        "latent_dimension": audio_length,
        "stored_space": args.stored_space,
        "flatten_layout": "channel_time",
        "preprocessing_id": "sc09_audio_16khz_model_space_v1",
        "schedule": "schedule.pt",
        "sampler": {
            "name": "deterministic_ddim",
            "protocol": "diffwave-sashimi/checkpoints:generate.py:sampling schedule; eta=0 DDIM update",
            "schedule_mode": args.schedule_mode,
            "num_states": args.num_states,
            "num_updates": args.num_states - 1,
            "training_steps": int(diffusion_cfg["T"]),
            "beta_start": float(diffusion_cfg["beta_0"]),
            "beta_end": float(diffusion_cfg["beta_T"]),
            "selected_timesteps": discrete.timesteps.detach().cpu().tolist(),
            "final_clean_update": True,
            "stochastic": False,
            "eta": 0.0,
            "conditioning": None,
            "initialization": args.initialization,
            "initial_latent_scale_vp": latent_scale,
            "terminal_average_data_variance": data_variance,
            "initialization_statistics": statistics_fingerprint,
            "integration_space": args.sampling_space,
        },
        "sample_ids": args.sample_ids,
        "rng_seeds": rng_seeds,
        "rng_rule": "rng_seed = sample_id + seed_offset",
        "seed_offset": args.seed_offset,
        "shards": shard_records,
        "source": {
            "repository": str(source_repo),
            "commit": source_revision(source_repo),
            "checkpoint": file_fingerprint(checkpoint),
            "resolved_model": model_cfg,
            "resolved_dataset": dataset_cfg,
        },
        "software": {"torch": torch.__version__, "cuda": torch.version.cuda},
    }
    save_manifest(manifest_path, manifest)
    return manifest_path


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = run(args)
    print(f"Saved universal DiffWave trajectory run: {manifest_path}")


if __name__ == "__main__":
    main()
