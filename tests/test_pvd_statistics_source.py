import sys
import types

import pytest
import torch

from universal.datasets.pvd import sample as pvd_sample
from universal.statistics.pvd import fit_pvd_resampling_limit
from universal.statistics.sources import PvdSource


def test_pvd_exact_resampling_fit_matches_explicit_covariance():
    source = PvdSource.__new__(PvdSource)
    source.dataset = "pvd"
    source.source_split = "train"
    source.flatten_layout = "xyz_point"
    source.preprocessing_id = "pvd_fixture_exact_resampling_limit_v1"
    source.npoints = 2
    source.dimension = 6
    source.num_shapes = 2
    source.dataset_object = types.SimpleNamespace(train_points=torch.tensor([
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[0.0, 2.0, 0.0], [0.0, 4.0, 0.0]],
    ]).numpy())

    result = fit_pvd_resampling_limit(source, rank=6, batch_size=1)
    covariance = (
        result["eigenvectors"]
        @ torch.diag(result["eigenvalues"])
        @ result["eigenvectors"].T
    )
    within = result["within_spatial_covariance"]
    between = result["between_centroid_covariance"]
    expected = torch.kron(within, torch.eye(2)) + torch.kron(
        between, torch.ones(2, 2)
    )
    assert torch.allclose(covariance, expected, atol=1e-5)
    assert result["num_samples"] == 2

    truncated = fit_pvd_resampling_limit(source, rank=4, batch_size=1)
    assert truncated["eigenvectors"].shape == (6, 4)
    assert torch.allclose(truncated["eigenvalues"], result["eigenvalues"][:4])
    assert truncated["total_variance"] == result["total_variance"]


def test_pvd_sampler_rejects_switching_source_repo_in_one_process(
    tmp_path, monkeypatch
):
    first = (tmp_path / "first").resolve()
    second = (tmp_path / "second").resolve()
    first.mkdir()
    second.mkdir()
    monkeypatch.setattr(pvd_sample, "_LOADED_SOURCE_REPO", first)
    with pytest.raises(RuntimeError, match="fresh Python process"):
        pvd_sample.load_model(second, tmp_path / "checkpoint", torch.device("cpu"), None)


def test_pvd_statistics_reject_cached_dataset_from_other_repo(
    tmp_path, monkeypatch
):
    source_repo = (tmp_path / "source").resolve()
    source_repo.mkdir()
    cached = types.ModuleType("datasets.shapenet_data_pc")
    cached.__file__ = str(tmp_path / "other" / "datasets" / "shapenet_data_pc.py")
    monkeypatch.setitem(sys.modules, "datasets.shapenet_data_pc", cached)
    with pytest.raises(RuntimeError, match="fresh Python process"):
        PvdSource(tmp_path / "data", source_repo)
