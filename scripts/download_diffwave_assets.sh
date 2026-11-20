#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${ROOT}/assets/datasets/sc09"
WEIGHTS="${ROOT}/assets/checkpoints/diffwave"
CACHE="${ROOT}/assets/downloads"
DIGITS=(zero one two three four five six seven eight nine)
ARCHIVE_DIGITS=(./zero ./one ./two ./three ./four ./five ./six ./seven ./eight ./nine)
mkdir -p "${DATA}" "${WEIGHTS}" "${CACHE}"

fetch() {
  wget -c --show-progress -O "$2.part" "$1"
  mv "$2.part" "$2"
}

if [[ "$(find "${DATA}" -name '*.wav' | wc -l)" -ne 38908 ]]; then
  archive="${CACHE}/speech_commands_v0.02.tar.gz"
  fetch https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz "${archive}"
  # The official archive records digit directories with a leading "./".
  tar -xzf "${archive}" -C "${DATA}" "${ARCHIVE_DIGITS[@]}"
fi
[[ "$(find "${DATA}" -name '*.wav' | wc -l)" -eq 38908 ]]

checkpoint="${WEIGHTS}/800000.pkl"
sha="53f80d98662d5bb0fe3c3984a6df94680bb7de4378a43bd0a4c0600e2fd42fcd"
if ! echo "${sha}  ${checkpoint}" | sha256sum -c --status 2>/dev/null; then
  fetch https://media.githubusercontent.com/media/albertfgu/diffwave-sashimi/checkpoints/exp/unet_d128_n6_pool_2_expand2_ff2_T200_betaT0.02_uncond/checkpoint/800000.pkl "${checkpoint}"
fi
echo "${sha}  ${checkpoint}" | sha256sum --check
echo "DiffWave/SC09 assets ready."
