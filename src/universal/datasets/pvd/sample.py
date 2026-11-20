"""Sample PVD chair trajectories into universal artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import types
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from universal.artifacts.trajectory import (
    FORMAT_VERSION,
    save_manifest,
    save_trajectory_shard,
    validate_schedule,
)
from universal.datasets.pvd.protocol import (
    SAMPLERS,
    SCHEDULE_MODES,
    schedule_from_alpha,
    select_native_ddim_timesteps,
    select_uniform_vp_timesteps,
    uniform_ddim_trajectory,
    uniform_edm_trajectory,
)
from universal.runtime import file_fingerprint, parse_seed_list, source_revision


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE_REPO = Path(
    os.environ.get("UNIVERSAL_PVD_REPO", PROJECT_ROOT / "third_party/PVD")
)
DEFAULT_CHECKPOINT = PROJECT_ROOT / "assets/checkpoints/pvd/chair_1799.pth"
DEFAULT_PVCNN_BACKEND = (
    Path(os.environ["UNIVERSAL_PVCNN_BACKEND"])
    if "UNIVERSAL_PVCNN_BACKEND" in os.environ else None
)
_LOADED_SOURCE_REPO: Path | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # PyTorch 1.10 does not expose warn_only.
        torch.use_deterministic_algorithms(True)


def load_cached_pvcnn_backend(path: Path) -> None:
    """Optionally load a prebuilt PVCNN extension instead of compiling it."""
    if not path.is_file():
        raise FileNotFoundError(
            f"PVCNN extension does not exist: {path}."
        )
    spec = importlib.util.spec_from_file_location("_pvcnn_backend", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load PVCNN extension: {path}")
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    wrapper = types.ModuleType("modules.functional.backend")
    wrapper._backend = extension
    sys.modules["modules.functional.backend"] = wrapper


def load_model(source_repo: Path, checkpoint: Path, device: torch.device, args):
    global _LOADED_SOURCE_REPO
    source_repo = source_repo.resolve()
    if _LOADED_SOURCE_REPO is not None and source_repo != _LOADED_SOURCE_REPO:
        raise RuntimeError(
            "PVD model modules are already loaded from a different source repository; "
            "use a fresh Python process before switching --source-repo"
        )
    cached = sys.modules.get("test_generation")
    if cached is not None:
        cached_path = Path(cached.__file__).resolve()
        try:
            cached_path.relative_to(source_repo)
        except ValueError as error:
            raise RuntimeError(
                "test_generation is cached from a different PVD repository; "
                "use a fresh Python process before switching --source-repo"
            ) from error
    source_text = str(source_repo.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    if args.pvcnn_backend is not None:
        load_cached_pvcnn_backend(args.pvcnn_backend.expanduser().resolve())
    from test_generation import Model, get_betas
    _LOADED_SOURCE_REPO = source_repo

    model_args = argparse.Namespace(
        nc=3, embed_dim=64, attention=True, dropout=0.1
    )
    betas = get_betas("linear", 0.0001, 0.02, 1000)
    model = Model(model_args, betas, "mse", "eps", "fixedsmall")
    payload = torch.load(checkpoint, map_location="cpu")
    state = OrderedDict(
        (key.replace("module.", ""), value)
        for key, value in payload["model_state"].items()
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seed_list, default=parse_seed_list("0-199"))
    parser.add_argument("--batch-size", type=int, default=1, help="trajectories per shard")
    parser.add_argument("--num-states", type=int, default=201)
    parser.add_argument("--schedule-mode", choices=SCHEDULE_MODES, default="native")
    parser.add_argument("--sampler", choices=SAMPLERS, default="ddim")
    parser.add_argument("--edm-solver", choices=("euler", "heun"), default="heun")
    parser.add_argument(
        "--edm-schedule", choices=("karras", "exponential", "vp"), default="karras"
    )
    parser.add_argument("--edm-rho", type=float, default=7.0)
    parser.add_argument("--npoints", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-repo", type=Path, default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--pvcnn-backend", type=Path, default=DEFAULT_PVCNN_BACKEND,
        help=(
            "optional prebuilt _pvcnn_backend.so; by default the upstream PVD "
            "code builds/loads its extension normally"
        ),
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample PVD trajectories")
    return add_arguments(parser)


@torch.no_grad()
def run(args: argparse.Namespace) -> Path:
    if args.batch_size < 1 or args.num_states < 2:
        raise ValueError("batch_size must be positive and num_states at least two")
    if args.sampler == "edm" and args.num_states < 3:
        raise ValueError("EDM requires at least three states (two denoising updates)")
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} already exists; choose a new run directory")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PVD sampling requires a CUDA device")
    torch.cuda.set_device(device)
    source_repo = args.source_repo.resolve()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = load_model(source_repo, checkpoint, device, args)

    uniform_schedule = None
    if args.schedule_mode == "uniform_vp_noise_std":
        selected = select_uniform_vp_timesteps(
            model.diffusion.alphas_cumprod, args.num_states
        )
        uniform_schedule = schedule_from_alpha(
            model.diffusion.alphas_cumprod, selected
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    real_dir = output_dir / "real"
    real_dir.mkdir()
    shard_records = []
    canonical_schedule = None
    shape = (1, 3, args.npoints)
    for shard_index, begin in enumerate(
        tqdm(range(0, len(args.seeds), args.batch_size), desc="PVD trajectories")
    ):
        batch_seeds = args.seeds[begin : begin + args.batch_size]
        trajectories = []
        for seed in batch_seeds:
            set_seed(seed)
            if args.schedule_mode == "native":
                _final, trace = model.gen_samples(
                    shape, device, sampler=args.sampler, num_steps=args.num_states - 1,
                    eta=0.0, return_trajectory=True,
                    edm_solver=args.edm_solver, edm_schedule=args.edm_schedule,
                    edm_rho=args.edm_rho, clip_denoised=False,
                )
                key = "trajectory_ve" if args.sampler == "edm" else "trajectory_vp"
                trajectory = trace[key].transpose(-1, -2).contiguous()
                current_schedule = {
                    "sigma_ve": trace["ve_schedule"].float().cpu(),
                    "alpha_bar": trace["alpha_bar"].float().cpu(),
                    "noise_std_vp": torch.sqrt(
                        (1.0 - trace["alpha_bar"].float().cpu()).clamp_min(0.0)
                    ),
                    "timesteps": trace["timesteps"].long().cpu(),
                }
                if args.sampler == "ddim":
                    expected_timesteps = select_native_ddim_timesteps(
                        len(model.diffusion.alphas_cumprod), args.num_states - 1
                    )
                    expected_states = torch.cat(
                        [expected_timesteps, torch.tensor([-1], dtype=torch.int64)]
                    )
                    if not torch.equal(current_schedule["timesteps"], expected_states):
                        raise RuntimeError(
                            "upstream PVD DDIM returned an unexpected timestep grid"
                        )
            else:
                noise = torch.randn(shape, device=device)
                if args.sampler == "ddim":
                    model_layout = uniform_ddim_trajectory(
                        model.diffusion, model._denoise, noise, uniform_schedule
                    )
                else:
                    model_layout = uniform_edm_trajectory(
                        model.diffusion, model._denoise, noise, uniform_schedule,
                        solver=args.edm_solver,
                    )
                trajectory = model_layout.transpose(-1, -2).cpu().contiguous()
                current_schedule = {
                    "sigma_ve": uniform_schedule.sigma_ve.float(),
                    "alpha_bar": uniform_schedule.alpha_bar.float(),
                    "noise_std_vp": uniform_schedule.noise_std_vp.float(),
                    "timesteps": uniform_schedule.timesteps.long(),
                }
            validate_schedule(current_schedule)
            trajectories.append(trajectory)
            if canonical_schedule is None:
                canonical_schedule = current_schedule
            else:
                for key in ("sigma_ve", "alpha_bar", "noise_std_vp", "timesteps"):
                    if not torch.equal(current_schedule[key], canonical_schedule[key]):
                        raise RuntimeError(f"PVD {key} schedule differs between samples")

        batch = torch.cat(trajectories, dim=0)
        shard_name = f"shard_{shard_index:05d}.pt"
        save_trajectory_shard(
            real_dir / shard_name, batch, torch.tensor(batch_seeds),
            num_states=args.num_states,
        )
        shard_records.append({
            "path": f"real/{shard_name}", "count": len(batch_seeds),
            "seeds": batch_seeds,
        })
        del batch, trajectories

    assert canonical_schedule is not None
    torch.save(canonical_schedule, output_dir / "schedule.pt")
    stored_space = "ve" if args.sampler == "edm" else "vp"
    manifest = {
        "format_version": FORMAT_VERSION,
        "dataset": "pvd",
        "category": "chair",
        "artifact": "real_trajectories",
        "num_trajectories": len(args.seeds), "num_states": args.num_states,
        "native_shape": [args.npoints, 3],
        "latent_dimension": args.npoints * 3,
        "stored_space": stored_space,
        "flatten_layout": "xyz_point",
        "saved_layout": "point_xyz",
        "preprocessing_id": "pvd_chair_model_normalized_raw_v1",
        "preprocessing": {
            "coordinate_space": "upstream_model_normalized",
            "permutation_invariant": False,
        },
        "schedule": "schedule.pt",
        "sampler": {
            "name": args.sampler, "schedule_mode": args.schedule_mode,
            "protocol": (
                "deterministic DDIM over evenly spaced native training timesteps; "
                "eta=0; terminal clean state"
                if args.sampler == "ddim"
                else "EDM-style VE probability-flow integration"
            ),
            "num_states": args.num_states, "num_updates": args.num_states - 1,
            "shard_batch_size": args.batch_size,
            "model_batch_size": 1,
            "training_steps": 1000, "eta": 0.0,
            "edm_solver": args.edm_solver if args.sampler == "edm" else None,
            "edm_schedule": args.edm_schedule if args.sampler == "edm" else None,
            "edm_rho": args.edm_rho if args.sampler == "edm" else None,
            "selected_timesteps": canonical_schedule["timesteps"].tolist(),
        },
        "seeds": args.seeds, "shards": shard_records,
        "source": {
            "repository": str(source_repo), "commit": source_revision(source_repo),
            "checkpoint": file_fingerprint(checkpoint),
            "pvcnn_backend": (
                file_fingerprint(args.pvcnn_backend.expanduser().resolve())
                if args.pvcnn_backend is not None else "upstream_jit"
            ),
        },
        "software": {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "reproducibility": {
            "initial_noise_replayable_from_seed": True,
            "bitwise_full_path_replay_guaranteed": False,
            "note": (
                "PVD's native PVCNN CUDA operations may vary across replays or "
                "toolchains even when deterministic controls and seeds are fixed."
            ),
        },
    }
    save_manifest(manifest_path, manifest)
    return manifest_path


def main() -> None:
    args = build_parser().parse_args()
    path = run(args)
    print(f"Saved universal PVD trajectory run: {path}")


if __name__ == "__main__":
    main()
