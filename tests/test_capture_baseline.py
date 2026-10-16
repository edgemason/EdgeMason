"""Tests for scripts/capture_baseline.py.

Everything here runs on a tiny ONNX model and a 20-sample npz written into a
temporary assets root: no network, no provisioned assets, no real models.
"""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

import scripts.capture_baseline as capture_baseline

CLASS_COUNT = 8
SAMPLE_COUNT = 20
INPUT_FEATURES = 4
# The six files the capture writes; environment.json is informational.
BASELINE_FILES = (
    "environment.json",
    "analysis.json",
    "plans.json",
    "quantized.json",
    "accuracy.json",
    "reproduction.json",
)
DETERMINISTIC_FILES = tuple(name for name in BASELINE_FILES if name != "environment.json")


def _tiny_model(path: Path) -> Path:
    """MatMul(4x8) + Add + Relu: fixed batch 1, eight output classes."""
    weights = np.arange(INPUT_FEATURES * CLASS_COUNT, dtype=np.float32).reshape(
        INPUT_FEATURES, CLASS_COUNT
    )
    bias = np.linspace(0.0, 0.7, CLASS_COUNT).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["X", "W"], ["h"]),
            helper.make_node("Add", ["h", "B"], ["a"]),
            helper.make_node("Relu", ["a"], ["Y"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, INPUT_FEATURES])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, CLASS_COUNT])],
        initializer=[
            helper.make_tensor("W", TensorProto.FLOAT, weights.shape, weights.tobytes(), raw=True),
            helper.make_tensor("B", TensorProto.FLOAT, bias.shape, bias.tobytes(), raw=True),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def _tiny_dataset(path: Path) -> Path:
    rng = np.random.default_rng(0)
    np.savez(
        path,
        inputs=rng.random((SAMPLE_COUNT, INPUT_FEATURES)).astype(np.float32),
        labels=rng.integers(0, CLASS_COUNT, SAMPLE_COUNT).astype(np.int64),
    )
    return path


def _manifest(*model_names: str) -> dict:
    assets = [{"path": f"models/{name}.onnx"} for name in model_names]
    assets.append({"path": capture_baseline.DATASET_PATH})
    return {"version": 1, "assets": assets}


def _assets_root(tmp_path: Path, model_names: tuple[str, ...] = ("tiny",)) -> Path:
    root = tmp_path / "assets"
    (root / "models" / "samples").mkdir(parents=True)
    for name in model_names:
        _tiny_model(root / "models" / f"{name}.onnx")
    _tiny_dataset(root / capture_baseline.DATASET_PATH)
    (root / capture_baseline.ASSETS_MANIFEST).write_text(
        json.dumps(_manifest(*model_names)), encoding="utf-8"
    )
    return root


def _all_keys(value: object):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _all_keys(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_capture_writes_every_baseline_file_without_a_filename_key(tmp_path):
    root = _assets_root(tmp_path)
    out_dir = tmp_path / "baseline"

    exit_code = capture_baseline.main(
        ["--out", str(out_dir), "--assets-root", str(root), "--models", "tiny"]
    )

    assert exit_code == 0
    assert sorted(path.name for path in out_dir.iterdir()) == sorted(BASELINE_FILES)

    analysis = json.loads((out_dir / "analysis.json").read_text(encoding="utf-8"))
    assert list(analysis) == ["tiny"]
    assert "filename" not in set(_all_keys(analysis))
    assert analysis["tiny"]["total_parameters"] == INPUT_FEATURES * CLASS_COUNT + CLASS_COUNT

    quantized = json.loads((out_dir / "quantized.json").read_text(encoding="utf-8"))
    assert sorted(quantized["tiny"]) == sorted(capture_baseline.QUANTIZED_CANDIDATES)
    assert all(len(digest) == 64 for digest in quantized["tiny"].values())
    assert len(set(quantized["tiny"].values())) == len(quantized["tiny"])

    accuracy = json.loads((out_dir / "accuracy.json").read_text(encoding="utf-8"))
    assert sorted(accuracy["tiny"]) == sorted(capture_baseline.QUANTIZED_CANDIDATES)
    for scores in accuracy["tiny"].values():
        assert scores["sample_count"] == SAMPLE_COUNT - 6  # 30% calibrates, the rest tests
        assert 0.0 <= scores["top1_accuracy"] <= 1.0

    plans = json.loads((out_dir / "plans.json").read_text(encoding="utf-8"))
    assert sorted(plans["tiny"]) == ["false", "true"]
    assert sorted(plans["tiny"]["false"]) == sorted(plans["tiny"]["true"])
    assert all(plan["model_family"] == "matmul" for plan in plans["tiny"].values())


def test_two_captures_produce_identical_files_except_the_environment(tmp_path):
    root = _assets_root(tmp_path)

    for name in ("first", "second"):
        exit_code = capture_baseline.main(
            ["--out", str(tmp_path / name), "--assets-root", str(root), "--models", "tiny"]
        )
        assert exit_code == 0

    for filename in DETERMINISTIC_FILES:
        first = (tmp_path / "first" / filename).read_bytes()
        second = (tmp_path / "second" / filename).read_bytes()
        assert first == second, f"{filename} differs between two captures"


def test_quick_fails_and_writes_nothing_when_an_asset_is_missing(tmp_path, capsys):
    missing = capture_baseline.QUICK_MODELS[1]
    root = _assets_root(tmp_path, model_names=(capture_baseline.QUICK_MODELS[0],))
    out_dir = tmp_path / "baseline"

    exit_code = capture_baseline.main(
        ["--out", str(out_dir), "--assets-root", str(root), "--quick"]
    )

    assert exit_code != 0
    assert f"models/{missing}.onnx" in capsys.readouterr().err
    assert not out_dir.exists()
