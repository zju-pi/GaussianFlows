#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${ROOT}/assets/datasets/cifar10"
WEIGHTS="${ROOT}/assets/checkpoints/cifar10"
CACHE="${ROOT}/assets/downloads"
mkdir -p "${DATA}" "${WEIGHTS}" "${CACHE}"

fetch() {
  wget -c --show-progress -O "$2.part" "$1"
  mv "$2.part" "$2"
}

if [[ ! -f "${DATA}/cifar-10-batches-py/data_batch_5" ]]; then
  archive="${CACHE}/cifar-10-python.tar.gz"
  fetch "${CIFAR10_URL:-https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz}" "${archive}"
  tar -xzf "${archive}" -C "${DATA}"
fi

checkpoint="${WEIGHTS}/edm-cifar10-32x32-uncond-vp.pkl"
sha="4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da"
if ! echo "${sha}  ${checkpoint}" | sha256sum -c --status 2>/dev/null; then
  fetch https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl "${checkpoint}"
fi
echo "${sha}  ${checkpoint}" | sha256sum --check
echo "CIFAR-10 assets ready."
