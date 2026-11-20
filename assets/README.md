# Downloaded assets

Run the dataset-specific scripts in `scripts/` to populate this directory.
Large data and checkpoints are intentionally ignored by Git.

```text
assets/
  datasets/
    cifar10/cifar-10-batches-py/
    sc09/{zero,one,...,nine}/
    diffusion_lm/e2e/{src1_train,src1_valid,src1_test}.txt
    pvd/ShapeNetCore.v2.PC15k/03001627/
  checkpoints/
    cifar10/edm-cifar10-32x32-uncond-vp.pkl
    diffwave/800000.pkl
    diffusion_lm/e2e/{ema_0.9999_200000.pt,training_args.json,vocab.json}
    pvd/chair_1799.pth
```

Checkpoint-loading source is bundled under `third_party/`; no additional
repository checkout is required. PVD sampling additionally requires a working
CUDA environment for its PVCNN extension. The Diffusion-LM and PVD download
scripts require `gdown>=6.4.2`.

Interrupted network transfers are kept under `assets/downloads/` and resume on
the next run.
