"""CLI integration for all canonical Gaussian statistics sources."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .core import add_isotropic_variance, fit_gaussian_statistics, save_statistics
from .diffusion_lm import source_xstart_jitter
from .pvd import fit_pvd_resampling_limit
from .sources import (
    Cifar10Source,
    DiffusionLmCheckpointTextSource,
    DiffWaveSource,
    PvdSource,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


DEFAULTS = {
    "cifar10": {"rank": 1024, "batch_size": 256},
    "diffusion-lm": {"rank": 1024, "batch_size": 256},
    "diffwave": {"rank": 4096, "batch_size": 32},
    "pvd": {"rank": 6144, "batch_size": 32},
}


def _common(
    parser: argparse.ArgumentParser, dataset: str, *, exact: bool = False
) -> None:
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=DEFAULTS[dataset]["rank"])
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULTS[dataset]["batch_size"]
    )
    parser.add_argument("--max-samples", type=int, default=None)
    if not exact:
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--oversample", type=int, default=0)
        parser.add_argument("--power-iterations", type=int, default=2)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--allow-rank-truncation", action="store_true",
            help="explicitly permit rank=min(requested, dimension, samples-1)",
        )
    parser.set_defaults(handler=run, result_kind="Gaussian statistics")


def add_subparsers(parent: argparse._SubParsersAction) -> None:
    cifar = parent.add_parser("cifar10", help="fit CIFAR-10 train pixels")
    _common(cifar, "cifar10")
    cifar.add_argument(
        "--data-root", type=Path,
        default=PROJECT_ROOT / "assets/datasets/cifar10/cifar-10-batches-py",
    )

    lm = parent.add_parser(
        "diffusion-lm", help="fit clean Diffusion-LM training embeddings"
    )
    _common(lm, "diffusion-lm")
    lm.add_argument(
        "--text-path", type=Path,
        default=PROJECT_ROOT / "assets/datasets/diffusion_lm/e2e/src1_train.txt",
    )
    lm.add_argument(
        "--vocab-path", type=Path,
        default=PROJECT_ROOT / "assets/checkpoints/diffusion_lm/e2e/vocab.json",
    )
    lm.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "assets/checkpoints/diffusion_lm/e2e/ema_0.9999_200000.pt",
    )
    lm.add_argument("--sequence-length", type=int, default=64)
    lm.add_argument(
        "--xstart-jitter", choices=("source", "none"), default="source",
        help="include the pre-forward timestep-0 noise used by E2E training",
    )
    lm.add_argument(
        "--training-args", type=Path, default=None,
        help="defaults to training_args.json beside the checkpoint",
    )

    wave = parent.add_parser("diffwave", help="fit SC09 training audio")
    _common(wave, "diffwave")
    wave.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "assets/datasets/sc09"
    )
    wave.add_argument("--sample-rate", type=int, default=16_000)
    wave.add_argument("--sample-length", type=int, default=16_000)

    pvd = parent.add_parser(
        "pvd", help="fit the exact PVD training-time point-resampling limit"
    )
    _common(pvd, "pvd", exact=True)
    pvd.add_argument(
        "--data-root", type=Path,
        default=PROJECT_ROOT / "assets/datasets/pvd/ShapeNetCore.v2.PC15k",
    )
    pvd.add_argument(
        "--source-repo", type=Path,
        default=Path(os.environ.get("UNIVERSAL_PVD_REPO", PROJECT_ROOT / "third_party/PVD")),
    )
    pvd.add_argument("--category", default="chair")
    pvd.add_argument("--npoints", type=int, default=2048)


def build_source(args: argparse.Namespace):
    if args.dataset == "cifar10":
        return Cifar10Source(args.data_root, args.max_samples)
    if args.dataset == "diffusion-lm":
        return DiffusionLmCheckpointTextSource(
            args.text_path, args.vocab_path, args.checkpoint,
            args.max_samples, args.sequence_length,
        )
    if args.dataset == "diffwave":
        return DiffWaveSource(
            args.data_root, args.max_samples, args.sample_rate, args.sample_length
        )
    if args.dataset == "pvd":
        return PvdSource(
            args.data_root, args.source_repo, category=args.category,
            npoints=args.npoints, max_samples=args.max_samples,
        )
    raise ValueError(f"unsupported statistics dataset: {args.dataset}")


def run(args: argparse.Namespace) -> Path:
    source = build_source(args)
    if args.dataset == "pvd":
        statistics = fit_pvd_resampling_limit(
            source, rank=args.rank, batch_size=args.batch_size
        )
    else:
        statistics = fit_gaussian_statistics(
            source, rank=args.rank, batch_size=args.batch_size, device=args.device,
            oversample=args.oversample, power_iterations=args.power_iterations,
            seed=args.seed, allow_rank_truncation=args.allow_rank_truncation,
        )
    source_metadata = source.metadata()
    if args.dataset == "pvd":
        fit_config = {
            "algorithm": "exact_pvd_training_resampling_limit",
            "shape_batch_size": args.batch_size,
            "rank": statistics["rank"],
            "effective_rank": statistics["rank"],
            "device": "cpu_float64_moments",
        }
    else:
        fit_config = {
            "algorithm": (
                "full_covariance_eigh"
                if statistics["rank"] == source.dimension
                else "streaming_randomized_centered_pca"
            ),
            "rank": statistics["rank"],
            "requested_rank": args.rank,
            "batch_size": args.batch_size,
            "oversample": args.oversample,
            "power_iterations": args.power_iterations,
            "seed": args.seed,
            "device": args.device,
            "allow_rank_truncation": args.allow_rank_truncation,
        }
    if (
        args.dataset == "diffusion-lm"
        and args.xstart_jitter == "source"
    ):
        training_args = args.training_args or args.checkpoint.parent / "training_args.json"
        jitter = source_xstart_jitter(training_args)
        statistics = add_isotropic_variance(
            statistics,
            jitter["variance"],
            preprocessing_suffix="source_xstart_t0_jitter",
        )
        source_metadata["xstart_jitter"] = jitter
        fit_config["post_fit_covariance_correction"] = {
            "type": "isotropic",
            **jitter,
        }
    return save_statistics(
        args.output_path, dataset=source.dataset, statistics=statistics,
        source_metadata=source_metadata,
        fit_config=fit_config,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="universal-stats")
    datasets = parser.add_subparsers(dest="dataset", required=True)
    add_subparsers(datasets)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = args.handler(args)
    print(f"Saved universal {args.dataset} Gaussian statistics: {path}")


if __name__ == "__main__":
    main()
