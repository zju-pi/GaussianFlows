# Universal diffusion trajectories

This repository now has three supported tasks:

1. sample complete diffusion trajectories;
2. fit Gaussian statistics to the corresponding training data; and
3. reproduce the trajectory figures and errors in readable notebooks.

The active datasets are CIFAR-10, DiffWave/SC09, Diffusion-LM/E2E, and
PVD/Chair.

## License

Original GaussianFlows software is available under the [MIT License](LICENSE).
The bundled model runtimes keep their own [licenses and notices](third_party/README.md),
including the noncommercial terms for NVIDIA EDM. Datasets and checkpoints are
subject to their respective upstream terms.

## Installation

Python 3.10 or newer is supported. Install a CUDA-compatible PyTorch build for
the target machine first; do not rely on pip to choose the CUDA build. Then
install the complete development environment with:

```bash
python -m pip install -r requirements.txt
```

Dependencies are defined in `pyproject.toml`; `requirements.txt` is only a
convenient editable-install entry point. A smaller installation can select
only the required extras, for example:

```bash
# Analysis, statistics, and notebooks, without model-specific samplers.
python -m pip install -e ".[statistics,notebooks]"

# Add the DiffWave sampler.
python -m pip install -e ".[statistics,notebooks,diffwave]"
```

Available extras are `statistics`, `notebooks`, `cifar10`, `diffwave`,
`diffusion-lm`, `pvd`, `download`, and `test`.

The complete four-model workflow is intended to run from a source checkout
installed in editable mode, because sampler runtimes, model configurations, and
notebooks live beside the Python package. A standalone wheel contains the core
Python package only and is not a complete experiment bundle.

The `diffwave` extra includes both the Lightning utility imported by the
released S4 implementation and PyKeOps, which supplies its required Cauchy
kernel when the optional compiled Cauchy extension is absent.

PVD sampling additionally requires a CUDA toolchain compatible with the
installed PyTorch build. Its bundled source compiles the PVCNN extension on
first use. A compatible prebuilt extension can instead be supplied with
`--pvcnn-backend` or `UNIVERSAL_PVCNN_BACKEND`. PVD statistics fitting does not
require this extension.

## Download datasets and checkpoints

Large assets are not committed. Download each modality independently:

```bash
./scripts/download_cifar10_assets.sh
./scripts/download_diffwave_assets.sh
./scripts/download_diffusion_lm_assets.sh
./scripts/download_pvd_assets.sh
```

The scripts install files under `assets/`, resume interrupted downloads, and
verify the known checkpoint SHA-256 hashes. Diffusion-LM and PVD downloads need
`gdown>=6.4.2`, supplied by the `download` extra. PVD's official PointFlow archive
contains all ShapeNet categories; the script retains only Chair (`03001627`).
See [`assets/README.md`](assets/README.md) for the resulting layout.

Download provenance is explicit in each script: CIFAR-10 uses the Toronto
archive and NVIDIA EDM weight; DiffWave uses Speech Commands v0.02 and the
official `checkpoints` branch; Diffusion-LM uses its GitHub E2E files and the
specified Google Drive release; PVD uses the PointFlow ShapeNet archive and
the PVD pretrained-model Drive release.

The small portions of the original model repositories required to load the
checkpoints are included under `third_party/`; no additional source checkout is
needed. Their provenance and retained compatibility changes are documented in
[`third_party/README.md`](third_party/README.md). To test another source tree,
pass `--source-repo` or set `UNIVERSAL_EDM_REPO`, `UNIVERSAL_DIFFWAVE_REPO`,
`UNIVERSAL_DIFFUSION_LM_REPO`, or `UNIVERSAL_PVD_REPO`.

## Layout

```text
src/universal/
  artifacts/       common manifest and shard format
  datasets/        one sampler adapter per active dataset
  statistics/      training-data adapters and exact/randomized covariance fitters
  analysis/        loading, Gaussian baselines, PCA, alignment, projection
scripts/            thin sampling/statistics launchers
notebooks/          four modality notebooks plus one cross-modality RMSE notebook
third_party/         minimal checkpoint-loading runtimes from source projects
configs/models/      Diffusion-LM model configuration
tests/               lightweight protocol and numerical checks
```

## 1. Sample trajectories

All samplers store the initial state and every numerical update. The standard
paper runs contain 200 updates and therefore 201 states.

```bash
./scripts/sample_cifar10.sh \
  --output-dir runs/cifar10/trajectories/euler_karras_201_seed0_199 \
  --seeds 0-199 --batch-size 8 --num-states 201 --device cuda:0

./scripts/sample_diffwave.sh \
  --output-dir runs/diffwave/trajectories/native_ddim_ve_201_sample0_199 \
  --sample-ids 0-199 --batch-size 8 --num-states 201 \
  --schedule-mode native --sampling-space ve \
  --initialization terminal_noise_matched --stored-space ve --device cuda:0

./scripts/sample_diffusion_lm.sh \
  --output-dir runs/diffusion_lm/trajectories/ddim200_seed0_199 \
  --seeds 0-199 --batch-size 16 --schedule-mode ddim200 \
  --num-states 201 --device cuda:0

# Run this command from an environment with PVD's CUDA dependencies installed.
./scripts/sample_pvd.sh \
  --output-dir runs/pvd/trajectories/chair_ddim_201_raw_seed0_199 \
  --seeds 0-199 --batch-size 1 --num-states 201 \
  --sampler ddim --schedule-mode native --device cuda:0
```

PVD trajectories and statistics use the model-normalized coordinates produced
by the released training-data loader.

Use `python -m universal sample DATASET --help` for all options.
Diffusion-LM intentionally exposes only the native 2000-step grid and its
fixed-stride DDIM200/DDIM1000 subsets.

Every run records its actual number of states, coordinate space, schedule,
preprocessing, checkpoint fingerprint, and source revision in `manifest.json`.
The analysis code reads these values and does not assume a fixed state count.

## 2. Fit Gaussian statistics

Every dataset goes through the same centered-covariance interface. For
CIFAR-10, DiffWave, and Diffusion-LM, truncated fits use
`streaming_randomized_centered_pca`, which avoids materializing the ambient
covariance; a full-rank request instead accumulates the covariance and applies
`torch.linalg.eigh`. PVD uses the structured exact calculation described below.
The saved `fit.algorithm` identifies the method, and requested ranks are never
silently reduced.

```bash
./scripts/calculate_stats.sh cifar10 \
  --output-path runs/cifar10/statistics/rank1024.pt --rank 1024

./scripts/calculate_stats.sh diffusion-lm \
  --output-path runs/diffusion_lm/statistics/rank1024.pt \
  --rank 1024 --xstart-jitter source

./scripts/calculate_stats.sh diffwave \
  --output-path runs/diffwave/statistics/rank4096.pt --rank 4096

# PVD statistics do not require its CUDA sampling extension.
./scripts/calculate_stats.sh pvd \
  --output-path runs/pvd/statistics/rank6144.pt --rank 6144
```

The adapters reproduce the model-space preprocessing used during training.
Diffusion-LM additionally includes the source training implementation's
timestep-zero embedding jitter as isotropic covariance when
`--xstart-jitter source` is selected.
Following released diffwave-sashimi commit
`9bd78f8c894cad0952a5692450f2145e24466b29`, DiffWave uses every extracted
zero--nine WAV file. Its `SpeechCommands` training loader recursively enumerates
`data/sc09/**/*.wav` and never reads the broader Speech Commands
`validation_list.txt` or `testing_list.txt`; the repository README instructs
users to place the ten complete digit directories under `data/sc09`. The
statistics metadata records this source training protocol as
`source_protocol_all_digits`.
For PVD, the fit analytically averages over the training loader's uniform
with-replacement selection of 2,048 points from each shape's 10,000-point
training pool. Let $W\in\mathbb{R}^{3\times3}$ be the mean within-shape
spatial covariance and $B\in\mathbb{R}^{3\times3}$ the covariance of shape
centroids. In coordinate-major layout, the exact covariance is

```math
\Sigma = W\otimes I_{2048} + B\otimes \mathbf{1}\mathbf{1}^{\mathsf T}.
```

The implementation diagonalizes this structured covariance analytically and
saves its leading eigenpairs. The canonical setting uses all 6,144 directions
and therefore represents the exact infinite-resampling covariance; a smaller
`--rank` requests a spectral approximation of that same covariance, not a
finite point-resampling estimate. Dataset-global normalization and point-pool
provenance are recorded in the artifact metadata.
PVD shape filenames are sorted before the upstream deterministic shuffle, and
the resulting order hash is saved in newly fitted statistics.

PVD sampling fixes the seed, file order, model batch size, and available PyTorch
determinism controls.  These make the initial noise reproducible, but the native
PVCNN CUDA path is not claimed to reproduce every trajectory state bit for bit
across replays or CUDA toolchains; this qualification is recorded in each new
trajectory manifest.

All wrappers invoke the current environment's `python`; they never activate or
name a Conda environment. PVD sampling normally lets the upstream code build
its PVCNN extension. To reuse a compatible prebuilt extension, pass
`--pvcnn-backend /path/to/_pvcnn_backend.so` or set
`UNIVERSAL_PVCNN_BACKEND`.

## 3. Compute independent-start trajectory RMSE

This offline analysis compares every real trajectory with candidates from all
other initializations, after start centering and one optimal orthogonal
Procrustes transformation.  The default output is stored inside the trajectory
run under `analyses/independent_start_rmse/`; notebooks only load the saved
pairwise error matrices and plot them.

```bash
python -m universal analyze independent-start-rmse \
  --trajectory-run runs/cifar10/trajectories/euler_karras_201_seed0_199 \
  --statistics runs/cifar10/statistics/rank1024.pt \
  --device cuda:0

python -m universal analyze independent-start-rmse \
  --trajectory-run runs/diffwave/trajectories/native_ddim_ve_201_sample0_199 \
  --statistics runs/diffwave/statistics/rank4096.pt \
  --device cuda:0

python -m universal analyze independent-start-rmse \
  --trajectory-run runs/diffusion_lm/trajectories/ddim200_seed0_199 \
  --statistics runs/diffusion_lm/statistics/rank1024.pt \
  --device cuda:0

python -m universal analyze independent-start-rmse \
  --trajectory-run runs/pvd/trajectories/chair_ddim_201_raw_seed0_199 \
  --statistics runs/pvd/statistics/rank6144.pt \
  --device cuda:0
```

The completed directory contains `pairwise_errors.pt` and `manifest.json`.
Large compact factors and the resumable checkpoint live under
`.cache/` during computation and are removed after successful completion unless
`--keep-cache` is specified.

## 4. Run the notebooks

Open one of:

- `notebooks/cifar10.ipynb`
- `notebooks/diffwave.ipynb`
- `notebooks/diffusion_lm.ipynb`
- `notebooks/pvd.ipynb`
- `notebooks/trajectory_rmse_distributions.ipynb` for the precomputed all-pairs trajectory
  RMSE histogram and empirical CDF

Each modality notebook has the same order:

1. paths plus structural and semantic artifact validation;
2. unweighted local-PCA trajectory regularity;
3. controlled same-start RMSE;
4. two-eigenmode projection.

For trajectory regularity, each path is translated so that its initial state is
the origin, and its temporal PCA basis is estimated independently. Thirty-two
paths are selected with a fixed random seed; a single orthogonal Procrustes map
then aligns each local three-dimensional path to a user-selected reference.
The plotting space, displayed subset, reference path, and camera angle are the
only controls in this section.

The RMSE section contains the analytic fitted Gaussian, an isotropic Gaussian
with covariance $[\mathrm{tr}(\Sigma)/d]I$, the fitted Gaussian's endpoint
chord, and an amplitude-matched Brownian bridge around that chord. Analysis space
(`vp`/`ve`) is the only exposed RMSE choice. Saved states receive equal weight,
the common starting point is translated to the origin before fitting one
orthogonal Procrustes transformation over the complete trajectory, and chord
progress is linear in the selected noise coordinate. The Gaussian uses the
empirical mean and fitted covariance, while the Brownian variance is fixed by
matching the real trajectory's mean squared deviation from the chord.
All integrated errors are reported relative to the raw Gaussian-chord error.
Thus only the raw chord is exactly 1; the aligned chord reports the improvement
due to orthogonal alignment under the same common scale.

Before analysis, the notebooks validate dataset identity, ambient dimension,
flattening layout, and the registered trajectory/statistics preprocessing
protocol. Diffusion-LM additionally checks checkpoint and timestep-zero-jitter
provenance.

The two-eigenmode section first computes the minimum-RMSE, same-start
ambient-space Procrustes alignment in VE coordinates. It then uses the four
displayed eigenmode pairs jointly to select a single orthogonal map whose
full-path RMSE remains within 1% of that optimum; no panel is aligned
independently. CIFAR-10, DiffWave, and Diffusion-LM select
four pairs spanning progressively smaller spectral gaps and omit the largest
eigenmode. PVD follows the same convention with four fixed pairs adapted to its
six-block structured spectrum: the two remaining common-point modes and one
representative from each of the first two residual eigenspaces are compared
with a reference from the third residual eigenspace.
Each panel overlays the analytic Gaussian curves, pointwise means of the real
trajectories, and ten real trajectories selected near the median
trajectory-wide deviation from the mean.

