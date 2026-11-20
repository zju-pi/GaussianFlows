"""Repeatable adapters from training data to flattened model-space samples."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class Cifar10Source:
    dataset = "cifar10"
    dimension = 3 * 32 * 32
    preprocessing_id = "cifar10_train_rgb_chw_minus1_1_v1"
    source_split = "train"
    flatten_layout = "channel_height_width"

    def __init__(self, root: Path, max_samples: int | None = None):
        self.root = root.expanduser().resolve()
        self.files = [self.root / f"data_batch_{index}" for index in range(1, 6)]
        for path in self.files:
            if not path.is_file():
                raise FileNotFoundError(path)
        self.num_samples = min(50_000, max_samples or 50_000)

    def iter_batches(self, batch_size: int) -> Iterator[torch.Tensor]:
        remaining = self.num_samples
        for path in self.files:
            with path.open("rb") as handle:
                array = pickle.load(handle, encoding="bytes")[b"data"][:remaining]
            matrix = torch.from_numpy(array).float() / 127.5 - 1.0
            for begin in range(0, len(matrix), batch_size):
                yield matrix[begin: begin + batch_size]
            remaining -= len(matrix)
            if remaining <= 0:
                break

    def metadata(self) -> dict:
        return {"data_root": str(self.root), "sample_shape": [3, 32, 32],
                "normalization": "uint8 / 127.5 - 1", "flatten_layout": "channel_height_width"}


class DiffusionLmCheckpointTextSource:
    """Rebuild the clean E2E embeddings seen by the pretrained checkpoint."""

    dataset = "diffusion_lm"
    source_split = "train_clean_embeddings_checkpoint_matched"
    flatten_layout = "token_embedding"

    def __init__(
        self, text_path: Path, vocab_path: Path, checkpoint_path: Path,
        max_samples: int | None = None, sequence_length: int = 64,
    ):
        self.text_path = text_path.expanduser().resolve()
        self.vocab_path = vocab_path.expanduser().resolve()
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        with self.vocab_path.open(encoding="utf-8") as handle:
            self.vocab = json.load(handle)
        checkpoint = _torch_load(self.checkpoint_path)
        self.embedding = torch.as_tensor(checkpoint["word_embedding.weight"], dtype=torch.float32)
        self.sequence_length = int(sequence_length)
        self.dimension = self.sequence_length * self.embedding.shape[1]
        self.preprocessing_id = "diffusion_lm_e2e_train_checkpoint_embedding_v2"

        from spacy.lang.en import English
        tokenizer = English().tokenizer
        rows = []
        with self.text_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.split("||")
                if len(fields) < 2:
                    raise ValueError(f"{self.text_path}:{line_number}: missing ||")
                tokens = [token.text for token in tokenizer(fields[1])]
                ids = [self.vocab["START"]]
                ids += [self.vocab.get(token, self.vocab["UNK"]) for token in tokens]
                ids += [self.vocab["END"]]
                ids = ids[:self.sequence_length]
                ids += [self.vocab["PAD"]] * (self.sequence_length - len(ids))
                rows.append(ids)
                if max_samples is not None and len(rows) >= max_samples:
                    break
        self.token_ids = torch.tensor(rows, dtype=torch.long)
        self.num_samples = len(rows)
        if self.num_samples < 2:
            raise ValueError("Diffusion-LM statistics require at least two samples")

    def iter_batches(self, batch_size: int) -> Iterator[torch.Tensor]:
        for begin in range(0, self.num_samples, batch_size):
            ids = self.token_ids[begin: begin + batch_size]
            yield self.embedding[ids].reshape(len(ids), self.dimension)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def metadata(self) -> dict:
        with self.checkpoint_path.open("rb") as handle:
            checkpoint_first_mib_sha256 = hashlib.sha256(
                handle.read(1024 * 1024)
            ).hexdigest()
        return {
            "text_path": str(self.text_path), "text_sha256": self._sha256(self.text_path),
            "vocab_path": str(self.vocab_path), "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_size_bytes": self.checkpoint_path.stat().st_size,
            "checkpoint_first_mib_sha256": checkpoint_first_mib_sha256,
            "sequence_length": self.sequence_length,
            "embedding_dimension": self.embedding.shape[1],
            "tokenizer": "spacy.lang.en.English().tokenizer",
            "construction": "target text -> START/END -> truncate/right-pad -> checkpoint embedding",
        }


class DiffWaveSource:
    dataset = "diffwave"
    # The released SC09 protocol consumes every extracted digit WAV and does
    # not apply Speech Commands validation/testing-list filtering.
    source_split = "source_protocol_all_digits"
    flatten_layout = "channel_time"

    def __init__(
        self, root: Path, max_samples: int | None = None,
        sample_rate: int = 16_000, length: int = 16_000,
    ):
        self.root = root.expanduser().resolve()
        self.sample_rate, self.length, self.dimension = sample_rate, length, length
        # Match the released SpeechCommands loader: recursive WAV discovery,
        # speaker-style filenames only, and no validation/testing-list filter.
        self.files = sorted(
            path for path in self.root.rglob("*.wav")
            if "_nohash_" in path.name and "_background_noise_" not in path.parts
        )
        self.files = self.files[:min(len(self.files), max_samples or len(self.files))]
        self.num_samples = len(self.files)
        self.preprocessing_id = (
            f"sc09_source_all_digits_audio_{sample_rate}hz_"
            f"mono_pad_or_trim_{length}_v1"
        )
        if self.num_samples < 2:
            raise FileNotFoundError(f"fewer than two wav files in {self.root}")

    def _audio(self, path: Path) -> torch.Tensor:
        import soundfile as sf
        values, source_rate = sf.read(path, dtype="float32", always_2d=True)
        values = values.mean(1)
        if source_rate != self.sample_rate:
            from scipy.signal import resample_poly
            divisor = math.gcd(source_rate, self.sample_rate)
            values = resample_poly(values, self.sample_rate // divisor, source_rate // divisor)
        result = torch.zeros(self.length)
        copied = min(len(values), self.length)
        result[:copied] = torch.from_numpy(np.asarray(values[:copied], dtype=np.float32).copy())
        return result

    def iter_batches(self, batch_size: int) -> Iterator[torch.Tensor]:
        for begin in range(0, self.num_samples, batch_size):
            yield torch.stack([self._audio(path) for path in self.files[begin: begin + batch_size]])

    def metadata(self) -> dict:
        return {"data_root": str(self.root), "sample_rate": self.sample_rate,
                "sample_length": self.length, "channels": 1,
                "flatten_layout": "channel_time",
                "corpus_selection": "all extracted zero--nine WAV files, matching source protocol",
                "corpus_selection_evidence": (
                    "released commit 9bd78f8c894cad0952a5692450f2145e24466b29: "
                    "SpeechCommands recursively loads data/sc09/**/*.wav and does not "
                    "read validation_list.txt or testing_list.txt"
                ),
                "file_order": "lexicographic absolute path"}


class PvdSource:
    """PVD Chair training shapes and their 10,000-point training pools.

    Statistics analytically average over the loader's with-replacement draw of
    2,048 points.  No finite collection of sampled point sets is materialized.
    """

    dataset = "pvd"
    source_split = "train"
    flatten_layout = "xyz_point"

    def __init__(
        self, root: Path, source_repo: Path, *, category: str = "chair",
        npoints: int = 2048, max_samples: int | None = None,
    ):
        self.root = root.expanduser().resolve()
        self.source_repo = source_repo.expanduser().resolve()
        cached = sys.modules.get("datasets.shapenet_data_pc")
        if cached is not None:
            cached_path = Path(cached.__file__).resolve()
            try:
                cached_path.relative_to(self.source_repo)
            except ValueError as error:
                raise RuntimeError(
                    "PVD dataset modules are already loaded from a different source "
                    "repository; use a fresh Python process before switching --source-repo"
                ) from error
        sys.path.insert(0, str(self.source_repo))
        from datasets.shapenet_data_pc import ShapeNet15kPointClouds
        self.category, self.npoints = category, npoints
        self.dataset_object = ShapeNet15kPointClouds(
            root_dir=str(self.root), categories=[category], split="train",
            tr_sample_size=npoints, te_sample_size=npoints, scale=1.0,
            normalize_per_shape=False, normalize_std_per_axis=False,
        )
        self.num_shapes = min(len(self.dataset_object), max_samples or len(self.dataset_object))
        self.num_samples = self.num_shapes
        self.dimension = 3 * npoints
        self.preprocessing_id = (
            f"pvd_{category}_train_model_space_uncentered_"
            "flattened_exact_resampling_limit_v1"
        )
        if self.num_shapes < 2:
            raise ValueError("PVD statistics require at least two shapes")

        ordered_shapes = self.dataset_object.all_cate_mids[:self.num_shapes]
        encoded_order = "\n".join(f"{category}/{name}" for category, name in ordered_shapes)
        self.shape_order_sha256 = hashlib.sha256(encoded_order.encode()).hexdigest()

    def metadata(self) -> dict:
        return {
            "data_root": str(self.root), "source_repo": str(self.source_repo),
            "category": self.category, "npoints": self.npoints,
            "num_shapes": self.num_shapes,
            "point_pool_size": int(self.dataset_object.train_points.shape[1]),
            "resampling_expectation": "analytic_with_replacement",
            "shape_order": "sorted filenames followed by upstream seed-38383 shuffle",
            "shape_order_sha256": self.shape_order_sha256,
            "coordinate_space": "upstream_model_normalized",
            "normalization": {
                "scope": "dataset_global",
                "mean_per_axis": np.asarray(
                    self.dataset_object.all_points_mean
                ).reshape(-1).tolist(),
                "std": np.asarray(self.dataset_object.all_points_std).reshape(-1).tolist(),
                "normalize_per_shape": False,
                "normalize_std_per_axis": False,
                "scale": 1.0,
                "training_point_pool": "first 10000 of each normalized 15000-point shape",
            },
            "statistics_layout": "xyz_point",
            "point_slots": "exchangeable under permutations",
        }
