"""Sample Diffusion-LM trajectories into universal artifacts."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from functools import partial
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
from universal.datasets.diffusion_lm.protocol import (
    SCHEDULE_MODES,
    deterministic_ddim_trajectory,
    schedule_from_alpha,
    select_ddim_stride_timesteps,
)
from universal.runtime import file_fingerprint, parse_seed_list, source_revision


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE_REPO = Path(
    os.environ.get(
        "UNIVERSAL_DIFFUSION_LM_REPO",
        PROJECT_ROOT / "third_party/Diffusion-LM/improved-diffusion",
    )
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "assets/checkpoints/diffusion_lm/e2e/ema_0.9999_200000.pt"
)
DEFAULT_BERT_CONFIG = (
    PROJECT_ROOT / "configs/models/bert-base-uncased-config.json"
)


def set_seed(seed: int) -> None:
    """Equivalent RNG setup to transformers.set_seed without extra coupling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_runtime(
    source_repo: Path,
    checkpoint: Path,
    device: torch.device,
    bert_config: Path,
):
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config_path = checkpoint.parent / "training_args.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open(encoding="utf-8") as handle:
        training = json.load(handle)
    source_text = str(source_repo.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from improved_diffusion.script_util import (
        create_model_and_diffusion,
        model_and_diffusion_defaults,
    )
    from improved_diffusion.test_util import denoised_fn_round

    runtime = dict(training)
    if not bert_config.is_file():
        raise FileNotFoundError(f"local BERT config does not exist: {bert_config}")
    runtime["config_name"] = str(bert_config.resolve())
    runtime["sigma_small"] = True
    runtime["diffusion_steps"] = int(training["diffusion_steps"])
    runtime["timestep_respacing"] = ""
    keys = model_and_diffusion_defaults().keys()
    model, diffusion = create_model_and_diffusion(**{key: runtime[key] for key in keys})
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device).eval().requires_grad_(False)

    embedding = torch.nn.Embedding.from_pretrained(
        model.word_embedding.weight.detach().cpu().clone(), freeze=True
    ).to(device)
    args = argparse.Namespace(**runtime)
    denoised_fn = partial(denoised_fn_round, args, embedding)
    return model, diffusion, denoised_fn, training, runtime


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=parse_seed_list, default=parse_seed_list("0-199")
    )
    parser.add_argument("--batch-size", type=int, default=16, help="trajectories per shard")
    parser.add_argument("--schedule-mode", choices=SCHEDULE_MODES, default="ddim200")
    parser.add_argument(
        "--num-states", type=int, default=201,
        help="states including the initial noise and final clean state",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-repo", type=Path, default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--bert-config", type=Path, default=DEFAULT_BERT_CONFIG)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample Diffusion-LM trajectories in universal format"
    )
    return add_arguments(parser)


@torch.no_grad()
def run(args: argparse.Namespace) -> Path:
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} already exists; choose a new run directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    source_repo = args.source_repo.resolve()
    checkpoint = args.checkpoint.resolve()
    model, diffusion, denoised_fn, training, runtime = load_runtime(
        source_repo, checkpoint, device, args.bert_config.resolve()
    )

    training_steps = int(training["diffusion_steps"])
    if args.schedule_mode in ("ddim200", "ddim1000"):
        sampling_steps = 200 if args.schedule_mode == "ddim200" else 1000
        expected_states = sampling_steps + 1
        if args.num_states != expected_states:
            raise ValueError(
                f"{args.schedule_mode} requires exactly {expected_states} stored states"
            )
        selected_timesteps = select_ddim_stride_timesteps(
            training_steps, sampling_steps
        )
    elif args.schedule_mode == "native":
        if args.num_states != training_steps + 1:
            raise ValueError(
                f"native mode requires {training_steps + 1} stored states"
            )
        selected_timesteps = torch.arange(
            training_steps - 1, -1, -1, dtype=torch.int64
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    real_dir = output_dir / "real"
    real_dir.mkdir()
    shard_records = []
    canonical_schedule = None
    num_states = None
    shape = (1, int(runtime["image_size"]) ** 2, int(runtime["in_channel"]))
    for shard_index, begin in enumerate(
        tqdm(range(0, len(args.seeds), args.batch_size), desc="Diffusion-LM trajectories")
    ):
        batch_seeds = args.seeds[begin : begin + args.batch_size]
        noises = []
        for seed in batch_seeds:
            set_seed(seed)
            noises.append(torch.randn(shape[1:], device=device))
        noise = torch.stack(noises)
        batch, alpha_states = deterministic_ddim_trajectory(
            diffusion, model, noise, selected_timesteps,
            denoised_fn=denoised_fn, clip_denoised=False,
        )
        schedule = schedule_from_alpha(alpha_states, selected_timesteps)
        current = {
            "sigma_ve": schedule.sigma_ve.float(),
            "alpha_bar": schedule.alpha_bar.float(),
            "noise_std_vp": schedule.noise_std_vp.float(),
            "timesteps": schedule.timesteps,
        }
        validate_schedule(current)
        if canonical_schedule is None:
            canonical_schedule = current
            num_states = batch.shape[1]
        else:
            if not torch.equal(current["timesteps"], canonical_schedule["timesteps"]):
                raise RuntimeError("trajectory timestep schedules differ")
            if not torch.allclose(current["alpha_bar"], canonical_schedule["alpha_bar"]):
                raise RuntimeError("trajectory alpha schedules differ")
        shard_name = f"shard_{shard_index:05d}.pt"
        save_trajectory_shard(
            real_dir / shard_name, batch, torch.tensor(batch_seeds),
            num_states=int(num_states),
        )
        shard_records.append({
            "path": f"real/{shard_name}", "count": len(batch_seeds),
            "seeds": batch_seeds,
        })
        del batch, noise, noises

    assert canonical_schedule is not None and num_states is not None
    torch.save(canonical_schedule, output_dir / "schedule.pt")
    manifest = {
        "format_version": FORMAT_VERSION,
        "dataset": "diffusion_lm",
        "artifact": "real_trajectories",
        "num_trajectories": len(args.seeds),
        "num_states": int(num_states),
        "native_shape": [int(runtime["image_size"]) ** 2, int(runtime["in_channel"])],
        "latent_dimension": int(runtime["image_size"]) ** 2 * int(runtime["in_channel"]),
        "stored_space": "vp",
        "flatten_layout": "token_embedding",
        "preprocessing_id": "diffusion_lm_e2e_tgt_random_embedding_v1",
        "schedule": "schedule.pt",
        "sampler": {
            "name": "deterministic_ddim",
            "protocol": (
                "Diffusion-LM Appendix C alpha_bar'[k]=alpha_bar[10k]; "
                "improved_diffusion.gaussian_diffusion DDIM eta=0"
                if args.schedule_mode == "ddim200"
                else "fixed-stride training-timestep schedule; deterministic DDIM eta=0"
            ),
            "schedule_mode": args.schedule_mode,
            "num_states": int(num_states),
            "num_updates": int(num_states) - 1,
            "training_steps": int(training["diffusion_steps"]),
            "runtime_diffusion_steps": int(runtime["diffusion_steps"]),
            "noise_schedule": runtime["noise_schedule"],
            "selected_timesteps": canonical_schedule["timesteps"].tolist(),
            "eta": 0.0,
            "clip_denoised": False,
            "clamp": "nearest_embedding",
            "timestep_respacing": (
                "ddim200" if args.schedule_mode == "ddim200"
                else "ddim1000" if args.schedule_mode == "ddim1000"
                else None
            ),
            "initialization": "standard_normal_vp",
            "dense_first": {"enabled": False},
        },
        "seeds": args.seeds,
        "shards": shard_records,
        "source": {
            "repository": str(source_repo), "commit": source_revision(source_repo),
            "checkpoint": file_fingerprint(checkpoint),
            "training_args": str((checkpoint.parent / "training_args.json").resolve()),
            "bert_config": str(args.bert_config.resolve()),
        },
        "software": {"torch": torch.__version__, "cuda": torch.version.cuda},
    }
    save_manifest(manifest_path, manifest)
    return manifest_path


def main() -> None:
    args = build_parser().parse_args()
    path = run(args)
    print(f"Saved universal Diffusion-LM trajectory run: {path}")


if __name__ == "__main__":
    main()
