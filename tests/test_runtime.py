import argparse

import pytest

from universal.runtime import parse_seed_list, source_revision


def test_parse_seed_list_supports_ranges_and_singletons():
    assert parse_seed_list("0-2,5") == [0, 1, 2, 5]


@pytest.mark.parametrize("value", ["", "2-1", "1,1", "-1"])
def test_parse_seed_list_rejects_invalid_values(value):
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        parse_seed_list(value)


def test_source_revision_reads_vendored_marker(tmp_path, monkeypatch):
    revision = "0123456789abcdef"
    (tmp_path / "SOURCE_REVISION").write_text(revision + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "universal.runtime.subprocess.run",
        lambda *args, **kwargs: pytest.fail("Git must not override a vendored marker"),
    )
    assert source_revision(tmp_path) == revision
