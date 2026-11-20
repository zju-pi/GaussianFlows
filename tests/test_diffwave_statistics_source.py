from pathlib import Path

from universal.statistics.sources import DiffWaveSource


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path.resolve()


def test_diffwave_source_matches_released_all_digit_loader(tmp_path):
    """The released SC09 loader enumerates digit WAVs without split lists."""
    first = _touch(tmp_path / "zero" / "alice_nohash_0.wav")
    second = _touch(tmp_path / "nine" / "nested" / "bob_nohash_1.wav")
    _touch(tmp_path / "_background_noise_" / "noise_nohash_0.wav")
    _touch(tmp_path / "zero" / "not_a_speech_commands_name.wav")
    (tmp_path / "validation_list.txt").write_text(
        "zero/alice_nohash_0.wav\n", encoding="utf-8"
    )
    (tmp_path / "testing_list.txt").write_text(
        "nine/nested/bob_nohash_1.wav\n", encoding="utf-8"
    )

    source = DiffWaveSource(tmp_path)

    assert source.files == sorted((first, second))
    assert source.num_samples == 2
    assert source.source_split == "source_protocol_all_digits"
    assert "does not read validation_list.txt" in (
        source.metadata()["corpus_selection_evidence"]
    )
