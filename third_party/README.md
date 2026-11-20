# Vendored checkpoint runtimes

These are selected upstream source files and configurations needed to load the
four supported checkpoints and sample trajectories. Datasets, checkpoints, and
generated samples are not included. The root MIT license does not replace the
licenses in this directory. Downloaded assets have their own upstream terms.

| Directory | Upstream project | Source revision | Included runtime | License |
| --- | --- | --- | --- | --- |
| `edm/` | [NVlabs/edm](https://github.com/NVlabs/edm) | `85516ab5aac006d2b6e74e5a5f2d89630cb126bd` (local checkout, based on NVlabs/edm) | `dnnlib`, `torch_utils.misc`, `torch_utils.persistence`, and `training.networks` | [CC BY-NC-SA 4.0](edm/LICENSE.txt) |
| `diffwave-sashimi/` | [albertfgu/diffwave-sashimi](https://github.com/albertfgu/diffwave-sashimi), `checkpoints` branch | `9bd78f8c894cad0952a5692450f2145e24466b29` | SaShiMi model and SC09 Hydra configuration | [MIT](diffwave-sashimi/LICENSE) |
| `Diffusion-LM/` | [XiangLi1999/Diffusion-LM](https://github.com/XiangLi1999/Diffusion-LM) | `759889d58ef38e2eed41a8c34db8032e072826f4` | model and diffusion definitions and token rounding | [Apache-2.0](Diffusion-LM/LICENSE) |
| `PVD/` | [alexzhou907/PVD](https://github.com/alexzhou907/PVD) | `9747265a5f141e5546fd4f862bfa66aa59f1bd33` | point-cloud model, PVCNN operators, and the locally extended sampling runtime | [MIT](PVD/LICENSE) |

EDM's retained files match the recorded checkout. The CIFAR-10 pickle relies on
EDM's persistence machinery, not its training or distillation drivers. The
bundled EDM runtime is subject to NVIDIA's noncommercial terms. The retained
DiffWave files are unchanged; its `generate.py` experiments and unused Hydra
configurations are excluded. The model factory still imports WaveNet even when
loading the SaShiMi checkpoint.

The Diffusion-LM package is copied from the working repository and is executed
with its verified legacy dependency `transformers==4.20.1`. Its training loop
and unused dataset loaders are excluded. The
sampler in `src/universal/datasets/diffusion_lm/` implements deterministic DDIM
directly and uses the package only for the checkpoint-compatible model and
diffusion definitions.

The PVD runtime omits the completion model, unused dataset adapters, and
standalone evaluation utilities. It retains compatibility changes:
Open3D is not imported by the dataset module during model loading,
the CUDA extension is not tied to `/usr/bin/gcc-8`, and `test_generation.py`
provides deterministic DDIM/EDM trajectory-returning methods used by the
universal adapter. The PVCNN binding also includes `<cstdint>` before the
PyTorch/pybind11 headers, which is required by some modern compiler and older
PyTorch-header combinations. Shape filenames are sorted before the upstream
seeded shuffle so that dataset order does not depend on filesystem enumeration.
The DDIM schedule also handles the one-update edge case by starting at the
terminal training timestep; standard 200-update trajectories are unchanged.
These are runtime changes, not dataset or checkpoint copies.
