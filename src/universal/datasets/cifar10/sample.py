"""Sample CIFAR-10 real trajectories into the universal artifact format."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from universal.artifacts.trajectory import (
    FORMAT_VERSION,
    build_schedule,
    save_manifest,
    save_trajectory_shard,
)
from universal.datasets.cifar10.protocol import (
    StackedRandomGenerator,
    euler_trajectory,
    polynomial_schedule,
)
from universal.runtime import file_fingerprint, parse_seed_list, source_revision


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE_REPO = Path(
    os.environ.get("UNIVERSAL_EDM_REPO", PROJECT_ROOT / "third_party/edm")
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "assets/checkpoints/cifar10/edm-cifar10-32x32-uncond-vp.pkl"
)


def load_edm_network(checkpoint: Path, source_repo: Path, device: torch.device):
    """Load the official EDM pickle using its bundled persistence runtime."""
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    source_text = str(source_repo.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    import dnnlib  # imported after adding the reference repository

    with dnnlib.util.open_url(str(checkpoint), verbose=True) as handle:
        network = pickle.load(handle)["ema"].to(device)
    network.eval().requires_grad_(False)
    if not hasattr(network, "sigma_min"):
        network.sigma_min = 0.002
    if not hasattr(network, "sigma_max"):
        network.sigma_max = 80.0
    return network


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seed_list, default=parse_seed_list("0-199"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-states", type=int, default=201)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-repo", type=Path, default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample CIFAR-10 EDM trajectories in the universal format"
    )
    return add_arguments(parser)


@torch.no_grad()
def run(args: argparse.Namespace) -> Path:
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"{manifest_path} already exists; choose a new run directory"
        )

    source_repo = args.source_repo.resolve()
    checkpoint = args.checkpoint.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    network = load_edm_network(checkpoint, source_repo, device)
    sigma_min = max(float(network.sigma_min), 0.002)
    sigma_max = min(float(network.sigma_max), 80.0)
    sigma_ve = polynomial_schedule(
        args.num_states, sigma_min, sigma_max, args.rho, device=device
    )
    schedule = build_schedule(sigma_ve)
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(schedule, output_dir / "schedule.pt")
    real_dir = output_dir / "real"
    real_dir.mkdir()

    shard_records = []
    for shard_index, begin in enumerate(
        tqdm(range(0, len(args.seeds), args.batch_size), desc="CIFAR-10 trajectories")
    ):
        batch_seeds = args.seeds[begin : begin + args.batch_size]
        random = StackedRandomGenerator(device, batch_seeds)
        latents = random.randn(
            [len(batch_seeds), network.img_channels, network.img_resolution, network.img_resolution],
            device=device,
            dtype=torch.float32,
        )
        class_labels = None
        class_ids = None
        if int(network.label_dim) > 0:
            class_ids = random.randint(
                int(network.label_dim), size=[len(batch_seeds)], device=device
            )
            class_labels = torch.eye(int(network.label_dim), device=device)[class_ids]
        trajectory = euler_trajectory(
            network, latents, sigma_ve, class_labels=class_labels
        )
        shard_name = f"shard_{shard_index:05d}.pt"
        save_trajectory_shard(
            real_dir / shard_name,
            trajectory,
            torch.tensor(batch_seeds),
            num_states=args.num_states,
            class_ids=class_ids,
        )
        shard_records.append({
            "path": f"real/{shard_name}",
            "count": len(batch_seeds),
            "seeds": batch_seeds,
        })
        del trajectory, latents, class_labels

    manifest = {
        "format_version": FORMAT_VERSION,
        "dataset": "cifar10",
        "artifact": "real_trajectories",
        "num_trajectories": len(args.seeds),
        "num_states": args.num_states,
        "native_shape": [int(network.img_channels), int(network.img_resolution), int(network.img_resolution)],
        "latent_dimension": int(network.img_channels * network.img_resolution ** 2),
        "stored_space": "ve",
        "flatten_layout": "channel_height_width",
        "preprocessing_id": "cifar10_edm_model_space_v1",
        "schedule": "schedule.pt",
        "sampler": {
            "name": "euler",
            "protocol": "diff-sampler/gits-main/sample.py:test.sh",
            "num_states": args.num_states,
            "num_updates": args.num_states - 1,
            "schedule_type": "polynomial",
            "schedule_rho": args.rho,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "afs": False,
            "denoise_to_zero": False,
        },
        "seeds": args.seeds,
        "shards": shard_records,
        "source": {
            "repository": str(source_repo),
            "commit": source_revision(source_repo),
            "checkpoint": file_fingerprint(checkpoint),
        },
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    save_manifest(manifest_path, manifest)
    return manifest_path


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = run(args)
    print(f"Saved universal CIFAR-10 trajectory run: {manifest_path}")


if __name__ == "__main__":
    main()

