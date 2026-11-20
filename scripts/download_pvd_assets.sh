#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${ROOT}/assets/datasets/pvd/ShapeNetCore.v2.PC15k"
WEIGHTS="${ROOT}/assets/checkpoints/pvd"
CACHE="${ROOT}/assets/downloads"
mkdir -p "${DATA}" "${WEIGHTS}" "${CACHE}"

if [[ ! -d "${DATA}/03001627/train" ]]; then
  archive="${CACHE}/ShapeNetCore.v2.PC15k.zip"
  python -m gdown --continue --retries 10 --timeout 60 1sw9gdk_igiyyt7MqALyxZhRrtPvAn0sX -O "${archive}"
  unzip -q "${archive}" 'ShapeNetCore.v2.PC15k/03001627/*' -d "${ROOT}/assets/datasets/pvd"
fi

checkpoint="${WEIGHTS}/chair_1799.pth"
sha="1dd3b2c4a2f1c2491c95f3c6e2aefacd047711158f83a48cfc25a1f33ae4eea3"
if ! echo "${sha}  ${checkpoint}" | sha256sum -c --status 2>/dev/null; then
  python -m gdown --continue --retries 10 --timeout 60 1-HIOD-3C3VtsEo9HVK3XsGEtsmn-TeuZ -O "${checkpoint}.part"
  mv "${checkpoint}.part" "${checkpoint}"
fi
echo "${sha}  ${checkpoint}" | sha256sum --check
echo "PVD Chair assets ready."
