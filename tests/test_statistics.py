import pytest
import torch

from universal.statistics import (
    add_isotropic_variance,
    fit_gaussian_statistics,
    load_statistics,
)
from universal.statistics.cli import build_parser


class MatrixSource:
    dataset = "fixture"
    preprocessing_id = "fixture_v1"
    flatten_layout = "feature"
    source_split = "train"

    def __init__(self, matrix):
        self.matrix = matrix
        self.num_samples, self.dimension = matrix.shape

    def iter_batches(self, batch_size):
        for begin in range(0, self.num_samples, batch_size):
            yield self.matrix[begin: begin + batch_size]


def test_pvd_statistics_defaults_to_full_rank_protocol():
    args = build_parser().parse_args([
        "pvd", "--output-path", "unused.pt"
    ])
    assert args.rank == 6144
    assert args.batch_size == 32


def test_full_rank_fit_matches_empirical_covariance():
    matrix = torch.tensor([
        [1.0, 2.0, 0.0], [2.0, -1.0, 1.0], [0.0, 3.0, 2.0], [4.0, 1.0, -2.0]
    ])
    result = fit_gaussian_statistics(
        MatrixSource(matrix), rank=3, batch_size=2, device="cpu"
    )
    covariance = torch.cov(matrix.T)
    expected = torch.linalg.eigvalsh(covariance).flip(0)
    assert torch.allclose(result["mean"], matrix.mean(0))
    assert torch.allclose(result["eigenvalues"], expected, atol=1e-5)
    assert abs(result["total_variance"] - float(torch.trace(covariance))) < 1e-6


def test_isotropic_correction_updates_every_full_rank_eigenvalue():
    matrix = torch.randn(8, 3)
    result = fit_gaussian_statistics(
        MatrixSource(matrix), rank=3, batch_size=4, device="cpu"
    )
    corrected = add_isotropic_variance(
        result, 0.2, preprocessing_suffix="noise"
    )
    assert torch.allclose(corrected["eigenvalues"], result["eigenvalues"] + 0.2)
    assert abs(corrected["total_variance"] - result["total_variance"] - 0.6) < 1e-6


def test_requested_fit_rank_is_strict_unless_explicitly_truncated():
    matrix = torch.randn(4, 3)
    with pytest.raises(ValueError, match="requested rank 4"):
        fit_gaussian_statistics(
            MatrixSource(matrix), rank=4, batch_size=2, device="cpu"
        )
    result = fit_gaussian_statistics(
        MatrixSource(matrix), rank=4, batch_size=2, device="cpu",
        allow_rank_truncation=True,
    )
    assert result["rank"] == 3
    assert result["requested_rank"] == 4


@pytest.mark.parametrize("field", ["mean", "eigenvectors"])
def test_statistics_loader_rejects_nonfinite_tensors(tmp_path, field):
    statistics = fit_gaussian_statistics(
        MatrixSource(torch.randn(6, 3)), rank=3, batch_size=2, device="cpu"
    )
    statistics[field] = statistics[field].clone()
    statistics[field].reshape(-1)[0] = torch.nan
    path = tmp_path / "bad.pt"
    torch.save({
        "format_version": 1,
        "artifact": "gaussian_statistics",
        "dataset": "fixture",
        "global_stats": statistics,
        "source": {},
        "fit": {},
    }, path)
    with pytest.raises(ValueError, match="non-finite"):
        load_statistics(path)


def test_statistics_loader_rejects_unknown_artifact_version(tmp_path):
    statistics = fit_gaussian_statistics(
        MatrixSource(torch.randn(6, 3)), rank=3, batch_size=2, device="cpu"
    )
    path = tmp_path / "future.pt"
    torch.save({
        "format_version": 999,
        "artifact": "gaussian_statistics",
        "dataset": "fixture",
        "global_stats": statistics,
        "source": {},
        "fit": {},
    }, path)
    with pytest.raises(ValueError, match="format version"):
        load_statistics(path)
