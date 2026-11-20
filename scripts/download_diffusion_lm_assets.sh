#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${ROOT}/assets/datasets/diffusion_lm/e2e"
WEIGHTS="${ROOT}/assets/checkpoints/diffusion_lm/e2e"
RAW="https://raw.githubusercontent.com/XiangLi1999/Diffusion-LM/759889d58ef38e2eed41a8c34db8032e072826f4/datasets/e2e_data"
mkdir -p "${DATA}" "${WEIGHTS}"

fetch() {
  wget -c --show-progress -O "$2.part" "$1"
  mv "$2.part" "$2"
}

for split in train valid test; do
  target="${DATA}/src1_${split}.txt"
  if [[ ! -s "${target}" ]]; then
    fetch "${RAW}/src1_${split}.txt" "${target}"
  fi
done

checkpoint="${WEIGHTS}/ema_0.9999_200000.pt"
sha="ed842cb391bf5c3568e2bf39ec52a605bda46cee8abf79a8389b54f792a09bab"
if ! echo "${sha}  ${checkpoint}" | sha256sum -c --status 2>/dev/null; then
  python -m gdown --continue --retries 10 --timeout 60 1vVtJYlKIEFTbDT5ZQGS5r7jMdqwWb8wz -O "${checkpoint}.part"
  mv "${checkpoint}.part" "${checkpoint}"
fi

cp "${ROOT}/configs/models/diffusion-lm-e2e-training-args.json" "${WEIGHTS}/training_args.json"
python "${ROOT}/scripts/build_diffusion_lm_vocab.py" "${DATA}/src1_train.txt" "${WEIGHTS}/vocab.json"
echo "${sha}  ${checkpoint}" | sha256sum --check
echo "Diffusion-LM/E2E assets ready."
