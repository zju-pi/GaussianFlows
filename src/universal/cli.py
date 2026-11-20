"""Top-level command line interface shared by all dataset adapters."""

from __future__ import annotations

import argparse

from universal.datasets.cifar10 import sample as cifar10_sample
from universal.datasets.diffwave import sample as diffwave_sample
from universal.datasets.diffusion_lm import sample as diffusion_lm_sample
from universal.datasets.pvd import sample as pvd_sample
from universal.analysis import independent_start
from universal.statistics import cli as statistics_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="universal")
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="sample real trajectories")
    datasets = sample.add_subparsers(dest="dataset", required=True)

    cifar10 = datasets.add_parser(
        "cifar10", help="sample CIFAR-10 EDM trajectories"
    )
    cifar10_sample.add_arguments(cifar10)
    cifar10.set_defaults(handler=cifar10_sample.run)
    diffwave = datasets.add_parser(
        "diffwave", help="sample DiffWave/SaShiMi SC09 trajectories"
    )
    diffwave_sample.add_arguments(diffwave)
    diffwave.set_defaults(handler=diffwave_sample.run)
    diffusion_lm = datasets.add_parser(
        "diffusion-lm", help="sample Diffusion-LM text-embedding trajectories"
    )
    diffusion_lm_sample.add_arguments(diffusion_lm)
    diffusion_lm.set_defaults(handler=diffusion_lm_sample.run)
    pvd = datasets.add_parser("pvd", help="sample PVD point-cloud trajectories")
    pvd_sample.add_arguments(pvd)
    pvd.set_defaults(handler=pvd_sample.run)

    stats = commands.add_parser("stats", help="fit Gaussian statistics")
    stats_datasets = stats.add_subparsers(dest="dataset", required=True)
    statistics_cli.add_subparsers(stats_datasets)

    analyze = commands.add_parser("analyze", help="run offline trajectory analyses")
    analyses = analyze.add_subparsers(dest="analysis", required=True)
    independent = analyses.add_parser(
        "independent-start-rmse",
        help="compute all-pairs aligned RMSE for independent initializations",
    )
    independent_start.add_arguments(independent)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = args.handler(args)
    if args.command == "analyze":
        print(f"Saved universal {args.analysis}: {manifest_path}")
        return
    result_kind = getattr(args, "result_kind", "trajectory run")
    print(f"Saved universal {args.dataset} {result_kind}: {manifest_path}")


if __name__ == "__main__":
    main()

