import inspect

import pytest
import torch

from universal.analysis import (
    GaussianStatistics,
    aligned_gaussian_hyperbolic_projection,
    controlled_same_start_rmse,
    load_gaussian_statistics,
    load_trajectory_run,
    regularity_local_coordinates,
    validate_statistics_compatibility,
)
from universal.analysis.hyperbolic import (
    ambient_start_procrustes_projection,
    gaussian_hyperbolic_projection,
    select_hyperbolic_eigenpairs,
)
from universal.analysis.independent_start import compact_factor, factor_distances
from universal.analysis.trajectory import (
    _procrustes_l2,
    _r_factor,
    _start_aligned,
    analytic_gaussian_vp_from_start,
)
from universal.artifacts import build_schedule, save_manifest, save_trajectory_shard


def test_independent_start_factor_distance_matches_direct_procrustes_rmse():
    torch.manual_seed(13)
    states, dimension = 6, 9
    reference = torch.randn(1, states, dimension, dtype=torch.float64)
    candidate = torch.randn(1, states, dimension, dtype=torch.float64)
    reference_factor = compact_factor(reference)
    candidate_factor = compact_factor(candidate)
    factored = factor_distances(
        reference_factor, candidate_factor, dimension=dimension
    ).squeeze()

    reference_centered = reference - reference[:, :1]
    candidate_centered = candidate - candidate[:, :1]
    cross = candidate_centered.squeeze(0).T @ reference_centered.squeeze(0)
    left, _singular, right_t = torch.linalg.svd(cross, full_matrices=False)
    rotation = left @ right_t
    direct = torch.sqrt(
        ((reference_centered - candidate_centered @ rotation).square()).mean()
    )
    assert torch.allclose(factored, direct, atol=1e-10)


def test_trajectory_loader_accepts_arbitrary_state_count(tmp_path):
    num_states = 7
    schedule = build_schedule(torch.linspace(3.0, 0.0, num_states))
    torch.save(schedule, tmp_path / "schedule.pt")
    trajectory = torch.randn(2, num_states, 2, 3)
    save_trajectory_shard(
        tmp_path / "real/shard_00000.pt", trajectory, torch.tensor([4, 9]),
        num_states=num_states,
    )
    save_manifest(tmp_path / "manifest.json", {
        "dataset": "synthetic",
        "num_trajectories": 2,
        "num_states": num_states,
        "schedule": "schedule.pt",
        "shards": [{"path": "real/shard_00000.pt", "count": 2}],
    })

    manifest, loaded_schedule, loaded, sample_ids = load_trajectory_run(tmp_path)
    assert manifest["num_states"] == num_states
    assert len(loaded_schedule["sigma_ve"]) == num_states
    assert loaded.shape == (2, num_states, 6)
    assert sample_ids.tolist() == [4, 9]


def test_trajectory_loader_rejects_duplicate_identifiers(tmp_path):
    schedule = build_schedule(torch.linspace(2.0, 0.0, 4))
    torch.save(schedule, tmp_path / "schedule.pt")
    save_trajectory_shard(
        tmp_path / "real/shard_00000.pt",
        torch.randn(2, 4, 2),
        torch.tensor([7, 7]),
        num_states=4,
    )
    save_manifest(tmp_path / "manifest.json", {
        "dataset": "synthetic", "num_trajectories": 2, "num_states": 4,
        "schedule": "schedule.pt", "seeds": [7, 8],
        "shards": [{"path": "real/shard_00000.pt", "count": 2}],
    })
    with pytest.raises(ValueError, match="unique"):
        load_trajectory_run(tmp_path)


def test_controlled_comparison_exposes_four_candidates():
    torch.manual_seed(1)
    samples, states, dimension = 3, 5, 8
    alpha = torch.linspace(0.2, 1.0, states)
    schedule = {
        "alpha_bar": alpha,
        "sigma_ve": torch.sqrt((1 - alpha) / alpha),
        "noise_std_vp": torch.sqrt(1 - alpha),
    }
    vectors = torch.eye(dimension)[:, :3]
    statistics = GaussianStatistics(
        mean=torch.zeros(dimension), eigenvectors=vectors,
        eigenvalues=torch.tensor([2.0, 1.0, 0.5]),
        num_samples=20, rank=3, total_variance=8.0,
    )
    real = torch.randn(samples, states, dimension)
    result = controlled_same_start_rmse(
        real, schedule, statistics, device="cpu", batch_size=2,
        space="ve",
    )
    for candidate in (
        "gaussian", "isotropic_gaussian", "gaussian_chord", "brownian"
    ):
        assert result[f"{candidate}_raw_l2"].shape == (samples, states)
        assert result[f"{candidate}_aligned_l2"].shape == (samples, states)
    assert not any("endpoint_matched" in key for key in result)
    assert "chord_raw_l2" not in result
    assert torch.allclose(
        result["gaussian_chord_raw_l2"][:, -1],
        result["brownian_raw_l2"][:, -1],
    )
    assert torch.equal(result["gaussian_raw_l2"][:, 0], torch.zeros(samples))
    assert torch.equal(
        result["isotropic_gaussian_raw_l2"][:, 0], torch.zeros(samples)
    )
    assert result["isotropic_variance"] == 1.0
    assert torch.allclose(
        result["gaussian_chord_raw_ratio_to_gaussian_chord"],
        torch.ones_like(result["gaussian_chord_raw_ratio_to_gaussian_chord"]),
    )
    assert torch.allclose(
        result["gaussian_chord_aligned_ratio_to_gaussian_chord"],
        result["gaussian_chord_aligned_l2"].square().mean(1).sqrt()
        / result["gaussian_chord_raw_l2"].square().mean(1).sqrt().clamp_min(1e-12),
    )
    assert result["gaussian_aligned_l2"].dtype == torch.float64
    assert result["gaussian_raw_l2"].dtype == torch.float64
    assert torch.equal(
        result["weights"], torch.full((states,), 1 / states, dtype=torch.float64)
    )


def test_controlled_comparison_exposes_only_scientific_coordinate_choice():
    parameters = inspect.signature(controlled_same_start_rmse).parameters
    removed = {
        "weighting", "alignment_center", "chord_mode", "mean_mode",
        "covariance_scale", "bridge_variance_scale", "bridge_seed",
    }
    assert removed.isdisjoint(parameters)
    assert "space" in parameters


def test_regularity_explained_variance_uses_total_centered_energy():
    trajectory = torch.tensor([
        [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [-2.0, 0.0]]
    ])
    coordinates, explained = regularity_local_coordinates(
        trajectory, components=2
    )
    centered = trajectory[0] - trajectory[0].mean(0, keepdim=True)
    expected = torch.linalg.svdvals(centered.double()).square()
    expected = expected / expected.sum()
    assert coordinates.shape == (1, 4, 2)
    assert torch.allclose(explained[0], expected[:2], atol=1e-5)


def test_regularity_small_components_match_float64_svd():
    torch.manual_seed(19)
    states, dimension = 32, 8
    projector = torch.eye(states, dtype=torch.float64) - torch.full(
        (states, states), 1 / states, dtype=torch.float64
    )
    left, _ = torch.linalg.qr(projector @ torch.randn(states, dimension).double())
    right, _ = torch.linalg.qr(torch.randn(dimension, dimension).double())
    singular = torch.tensor([1.0, 0.2, 0.015, 0.004, 0.001, 3e-4, 1e-4, 3e-5])
    trajectory = ((left * singular) @ right.T).float().unsqueeze(0)

    coordinates, explained = regularity_local_coordinates(
        trajectory, components=3
    )
    centered = trajectory[0].double() - trajectory[0].double().mean(0, keepdim=True)
    expected = torch.linalg.svdvals(centered).square()
    expected = expected / expected.sum()
    relative_error = (explained[0] - expected[:3]).abs() / expected[:3]
    assert coordinates.dtype == torch.float64
    assert float(relative_error.max()) < 1e-8


def test_analysis_statistics_loader_rejects_insufficient_rank(tmp_path):
    path = tmp_path / "statistics.pt"
    torch.save({
        "format_version": 1,
        "artifact": "gaussian_statistics",
        "dataset": "cifar10",
        "global_stats": {
            "mean": torch.zeros(2), "eigenvectors": torch.eye(2),
            "eigenvalues": torch.tensor([2.0, 1.0]), "num_samples": 5,
            "rank": 2, "dimension": 2, "total_variance": 3.0,
            "preprocessing_id": "cifar10_train_rgb_chw_minus1_1_v1",
            "flatten_layout": "channel_height_width", "source_split": "train",
        },
        "source": {}, "fit": {},
    }, path)
    with pytest.raises(ValueError, match="requested rank 3"):
        load_gaussian_statistics(path, rank=3)
    loaded = load_gaussian_statistics(
        path, rank=3, allow_rank_truncation=True
    )
    assert loaded.rank == 2


def test_analysis_statistics_loader_rejects_bad_version_and_nonfinite(tmp_path):
    base = {
        "format_version": 999,
        "artifact": "gaussian_statistics",
        "dataset": "cifar10",
        "global_stats": {
            "mean": torch.zeros(2), "eigenvectors": torch.eye(2),
            "eigenvalues": torch.tensor([2.0, 1.0]), "num_samples": 5,
            "rank": 2, "dimension": 2, "total_variance": 3.0,
        },
        "source": {}, "fit": {},
    }
    path = tmp_path / "statistics.pt"
    torch.save(base, path)
    with pytest.raises(ValueError, match="format version"):
        load_gaussian_statistics(path)
    base["format_version"] = 1
    base["global_stats"]["mean"][0] = torch.inf
    torch.save(base, path)
    with pytest.raises(ValueError, match="mean contains non-finite"):
        load_gaussian_statistics(path)


def test_hyperbolic_projection_recovers_analytic_gaussian_curve():
    sigma = torch.tensor([3.0, 2.0, 1.0, 0.0])
    alpha = 1 / (1 + sigma.square())
    schedule = {
        "alpha_bar": alpha,
        "sigma_ve": sigma,
        "noise_std_vp": torch.sqrt(1 - alpha),
    }
    statistics = GaussianStatistics(
        mean=torch.tensor([0.5, -0.25, 0.75]),
        eigenvectors=torch.eye(3),
        eigenvalues=torch.tensor([4.0, 1.0, 0.25]),
        num_samples=10,
        rank=3,
        total_variance=5.25,
    )
    starts_ve = torch.tensor([[2.0, 3.0, 4.0], [-2.0, -3.0, -4.0]])
    starts_vp = starts_ve * torch.sqrt(alpha[0])
    gaussian_vp = analytic_gaussian_vp_from_start(starts_vp, schedule, statistics)

    for space in ("ve", "vp"):
        result = aligned_gaussian_hyperbolic_projection(
            gaussian_vp,
            schedule,
            statistics,
            torch.tensor([[0, 1]]),
            space=space,
            sample_count=2,
            min_abs_initial_weight=0,
        )
        expected = result["gaussian_coordinates"][0]
        assert result["space"] == space
        assert result["alignment"] == "start_procrustes"
        assert torch.allclose(result["real_coordinates"][:, 0], expected, atol=2e-6)


def test_plane_aware_hyperbolic_alignment_respects_full_rmse_cap():
    torch.manual_seed(4)
    dimension = 20
    sigma = torch.linspace(3.0, 0.0, 8)
    alpha = 1 / (1 + sigma.square())
    schedule = {
        "alpha_bar": alpha,
        "sigma_ve": sigma,
        "noise_std_vp": torch.sqrt(1 - alpha),
    }
    statistics = GaussianStatistics(
        mean=torch.zeros(dimension),
        eigenvectors=torch.eye(dimension),
        eigenvalues=torch.linspace(4.0, 0.2, dimension),
        num_samples=10,
        rank=dimension,
        total_variance=42.0,
    )
    starts = torch.randn(3, dimension) * torch.sqrt(alpha[0])
    gaussian = analytic_gaussian_vp_from_start(starts, schedule, statistics)
    real = gaussian + 0.08 * torch.randn_like(gaussian)
    real[:, 0] = gaussian[:, 0]

    result = aligned_gaussian_hyperbolic_projection(
        real,
        schedule,
        statistics,
        torch.tensor([[0, 15], [1, 15]]),
        space="ve",
        sample_count=3,
        min_abs_initial_weight=0,
        alignment_rmse_slack=0.01,
    )
    assert result["alignment"] == "constrained_plane_start_procrustes"
    assert torch.all(result["full_rmse_ratio"] <= 1.010001)
    assert torch.all(result["plane_rmse_after"] <= result["plane_rmse_before"])


def test_ambient_procrustes_projection_recovers_rotated_target():
    torch.manual_seed(2)
    source = torch.randn(2, 6, 4)
    source = source - source[:, :1]
    rotation, _ = torch.linalg.qr(torch.randn(4, 4))
    target = source @ rotation
    projection_vectors = torch.eye(4)[:, :2]

    projected = ambient_start_procrustes_projection(
        source, target, projection_vectors
    )
    assert projected.dtype == torch.float64
    assert torch.allclose(projected, target[:, :, :2].double(), atol=2e-6)


def test_alignment_retains_ill_conditioned_residuals_in_float64():
    torch.manual_seed(7)
    samples, states, dimension = 1, 12, 8
    left, _ = torch.linalg.qr(torch.randn(states - 1, dimension))
    right, _ = torch.linalg.qr(torch.randn(dimension, dimension))
    singular = torch.logspace(0, -6, dimension)
    body = (left * singular) @ right.T
    reference = torch.cat([torch.zeros(1, dimension), body]).unsqueeze(0).float()
    rotation, _ = torch.linalg.qr(torch.randn(dimension, dimension))
    candidate = reference @ rotation.float()
    weights = torch.full((states,), 1 / states, dtype=torch.float64)

    reference_centered = _start_aligned(reference)
    candidate_centered = _start_aligned(candidate)
    residual = _procrustes_l2(
        _r_factor(reference_centered, weights),
        _r_factor(candidate_centered, weights),
        weights,
    )
    assert residual.dtype == torch.float64
    assert torch.isfinite(residual).all()
    assert float(residual.max()) < 2e-6


def test_hyperbolic_selection_never_backfills_below_threshold():
    sigma = torch.tensor([2.0, 1.0, 0.0])
    statistics = GaussianStatistics(
        mean=torch.zeros(2), eigenvectors=torch.eye(2),
        eigenvalues=torch.tensor([2.0, 1.0]), num_samples=10, rank=2,
        total_variance=3.0,
    )
    paths = torch.tensor([
        [[2.0, 2.0], [1.5, 1.5], [1.0, 1.0]],
        [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
    ])
    result = gaussian_hyperbolic_projection(
        paths, sigma, statistics, torch.tensor([[0, 1]]),
        sample_count=2, min_abs_initial_weight=0.25,
    )
    assert result["sample_indices"].tolist() == [0]
    assert torch.isfinite(result["real_coordinates"]).all()

    with pytest.raises(ValueError, match="no trajectory"):
        gaussian_hyperbolic_projection(
            paths[1:], sigma, statistics, torch.tensor([[0, 1]]),
            sample_count=1, min_abs_initial_weight=0.25,
        )


def test_hyperbolic_eigenpair_selection_respects_first_index():
    pairs = select_hyperbolic_eigenpairs(
        torch.tensor([10.0, 9.0, 7.0, 4.0, 0.0]),
        first_index=1,
        reference_index=4,
        gap_fractions=(1.0, 0.5),
    )
    assert pairs.tolist() == [[1, 4], [3, 4]]

    with pytest.raises(ValueError, match="first_index"):
        select_hyperbolic_eigenpairs(
            torch.tensor([3.0, 2.0, 1.0]), first_index=2, reference_index=2
        )


def _statistics_fixture(dataset="cifar10", **changes):
    values = dict(
        mean=torch.zeros(6), eigenvectors=torch.eye(6),
        eigenvalues=torch.ones(6), num_samples=10, rank=6,
        total_variance=6.0, dataset=dataset,
        preprocessing_id="cifar10_train_rgb_chw_minus1_1_v1",
        flatten_layout="channel_height_width", source_split="train",
    )
    values.update(changes)
    return GaussianStatistics(**values)


def test_statistics_compatibility_rejects_same_dimension_wrong_dataset():
    manifest = {
        "dataset": "cifar10", "latent_dimension": 6,
        "flatten_layout": "channel_height_width",
        "preprocessing_id": "cifar10_edm_model_space_v1",
    }
    validate_statistics_compatibility(manifest, _statistics_fixture())
    with pytest.raises(ValueError, match="dataset mismatch"):
        validate_statistics_compatibility(
            manifest, _statistics_fixture(dataset="diffwave")
        )


def test_statistics_loader_preserves_provenance(tmp_path):
    path = tmp_path / "statistics.pt"
    torch.save({
        "format_version": 1,
        "artifact": "gaussian_statistics",
        "dataset": "cifar10",
        "global_stats": {
            "mean": torch.zeros(2), "eigenvectors": torch.eye(2),
            "eigenvalues": torch.tensor([2.0, 1.0]), "num_samples": 5,
            "rank": 2, "dimension": 2, "total_variance": 3.0,
            "preprocessing_id": "cifar10_train_rgb_chw_minus1_1_v1",
            "flatten_layout": "channel_height_width", "source_split": "train",
        },
        "source": {"normalization": "uint8 / 127.5 - 1"},
        "fit": {"algorithm": "fixture"},
    }, path)
    statistics = load_gaussian_statistics(path)
    assert statistics.dataset == "cifar10"
    assert statistics.preprocessing_id == "cifar10_train_rgb_chw_minus1_1_v1"
    assert statistics.flatten_layout == "channel_height_width"
    assert statistics.source_split == "train"
    assert statistics.source_metadata["normalization"] == "uint8 / 127.5 - 1"
    assert statistics.fit_config["algorithm"] == "fixture"


def test_pvd_compatibility_rejects_unregistered_preprocessing():
    statistics = _statistics_fixture(
        dataset="pvd",
        preprocessing_id="pvd_chair_train_unknown_transform_v1",
        flatten_layout="xyz_point",
        source_metadata={"category": "chair", "npoints": 2},
    )
    manifest = {
        "dataset": "pvd", "category": "chair", "latent_dimension": 6,
        "native_shape": [2, 3], "flatten_layout": "xyz_point",
        "saved_layout": "point_xyz",
        "preprocessing_id": "pvd_chair_model_unknown_transform_v1",
    }
    with pytest.raises(ValueError, match="unsupported trajectory preprocessing"):
        validate_statistics_compatibility(manifest, statistics)


def test_pvd_compatibility_rejects_wrong_saved_layout():
    statistics = _statistics_fixture(
        dataset="pvd",
        preprocessing_id=(
            "pvd_chair_train_model_space_uncentered_"
            "flattened_exact_resampling_limit_v1"
        ),
        flatten_layout="xyz_point",
        source_metadata={"category": "chair", "npoints": 2},
    )
    manifest = {
        "dataset": "pvd", "category": "chair", "latent_dimension": 6,
        "native_shape": [2, 3], "flatten_layout": "xyz_point",
        "saved_layout": "xyz_point",
        "preprocessing_id": "pvd_chair_model_normalized_raw_v1",
    }
    with pytest.raises(ValueError, match="saved layout"):
        validate_statistics_compatibility(manifest, statistics)
