import pytest
from tests.helpers import normalise


def test_normalise_replaces_absolute_path():
    assert normalise({"model_path": "/tmp/edgemason/model.onnx"}) == {"model_path": "<PATH>"}


def test_normalise_replaces_uuid():
    assert normalise({"id": "a3f8b2c1-4d5e-6f78-9a0b-c1d2e3f4a5b6"}) == {"id": "<UUID>"}


def test_normalise_replaces_system_info():
    assert normalise({"system_info": {"os": "Linux", "cpu": "x86_64"}}) == {
        "system_info": "<SYSTEM_INFO>"
    }


def test_normalise_replaces_version():
    assert normalise("onnxruntime==1.27.0") == "<VERSION>"


def test_normalise_replaces_timestamp():
    assert normalise("2026-10-14T23:05:12.123456") == "<TIMESTAMP>"


def test_normalise_nested_list_replaces_all_and_does_not_mutate():
    original = [
        {"path": "/tmp/model.onnx", "id": "a3f8b2c1-4d5e-6f78-9a0b-c1d2e3f4a5b6"},
        {
            "system_info": {"os": "Linux"},
            "version": "onnx==1.16.0",
            "time": "2026-10-14T23:05:12.123456",
        },
    ]
    expected = [
        {"path": "<PATH>", "id": "<UUID>"},
        {
            "system_info": "<SYSTEM_INFO>",
            "version": "<VERSION>",
            "time": "<TIMESTAMP>",
        },
    ]

    result = normalise(original)

    assert result == expected
    assert original[0]["path"] == "/tmp/model.onnx"
    assert original[1]["version"] == "onnx==1.16.0"

import os

from tests.helpers import GOLDEN_DIR, assert_matches_golden


def test_assert_matches_golden_creates_missing_golden(tmp_path, monkeypatch):
    monkeypatch.setattr("tests.helpers.GOLDEN_DIR", tmp_path / "golden")
    with pytest.raises(AssertionError, match="created"):
        assert_matches_golden("missing_case", {"a": 1})
    created = tmp_path / "golden" / "missing_case.json"
    assert created.is_file()
    assert "a" in created.read_text()


def test_assert_matches_golden_passes_on_match(tmp_path, monkeypatch):
    monkeypatch.setattr("tests.helpers.GOLDEN_DIR", tmp_path / "golden")
    gdir = tmp_path / "golden"
    gdir.mkdir()
    (gdir / "match_case.json").write_text(__import__("json").dumps(__import__("tests.helpers", fromlist=["normalise"]).normalise({"a": 1}), indent=2, sort_keys=True))
    assert_matches_golden("match_case", {"a": 1}) is None


def test_assert_matches_golden_fails_with_diff_on_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr("tests.helpers.GOLDEN_DIR", tmp_path / "golden")
    gdir = tmp_path / "golden"
    gdir.mkdir()
    (gdir / "mismatch_case.json").write_text('{"a": 2}\n')
    with pytest.raises(AssertionError) as exc:
        assert_matches_golden("mismatch_case", {"a": 1})
    assert "---" in str(exc.value) or "@@" in str(exc.value)


def test_assert_matches_golden_update_env_rewrites(tmp_path, monkeypatch):
    monkeypatch.setattr("tests.helpers.GOLDEN_DIR", tmp_path / "golden")
    gdir = tmp_path / "golden"
    gdir.mkdir()
    (gdir / "update_case.json").write_text('{"a": 2}')
    monkeypatch.setenv("UPDATE_GOLDEN", "1")
    assert_matches_golden("update_case", {"a": 1}) is None
    assert '"a": 1' in (gdir / "update_case.json").read_text()
