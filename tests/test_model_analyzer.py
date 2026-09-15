import json

import pytest

from api.analyzer.model_analyzer import (
    count_parameters,
    is_file_valid,
    list_operator_types,
    load_model,
    model_report,
    model_size_bytes,
    model_size_mb,
)


def test_is_file_valid():
    assert is_file_valid("model.onnx")
    assert not is_file_valid("model.h5")
    assert not is_file_valid("model.ONNX.tmp")


def test_load_model_rejects_non_onnx():
    with pytest.raises(ValueError):
        load_model("not_onnx.txt")


def test_count_parameters(tiny_onnx):
    model = load_model(str(tiny_onnx))
    # W: 4*8, B: 8
    assert count_parameters(model) == 4 * 8 + 8


def test_list_operator_types(tiny_onnx):
    model = load_model(str(tiny_onnx))
    ops = list_operator_types(model)
    assert ops["MatMul"] == 1
    assert ops["Add"] == 1
    assert ops["Relu"] == 1


def test_model_size_bytes(tiny_onnx):
    assert model_size_bytes(str(tiny_onnx)) == tiny_onnx.stat().st_size


def test_model_size_mb(tiny_onnx):
    assert model_size_mb(str(tiny_onnx)) == tiny_onnx.stat().st_size / (1024 * 1024)


def test_model_report(tiny_onnx):
    report = json.loads(model_report(str(tiny_onnx)))
    assert report["filename"] == str(tiny_onnx)
    assert report["model_size_mb"] > 0
    assert report["total_parameters"] == 40
    assert report["num_nodes"] == 3
    assert report["operator_types"]["MatMul"] == 1
    assert report["ir_version"] > 0
    assert len(report["opsets"]) >= 1
