import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from api.optimizer.fetcher import fetch_profile, main

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fetcher"

FP32_MODEL = FIXTURES / "squeezenet1.1-13.fp32.onnx"
FP32_TRACE = FIXTURES / "squeezenet1.1-13.fp32.trace.json"
FP32_MEASURED = FIXTURES / "squeezenet1.1-13.fp32.measured.json"

INT8_MODEL = FIXTURES / "squeezenet1.1-13.int8.onnx"
INT8_TRACE = FIXTURES / "squeezenet1.1-13.int8.trace.json"
INT8_MEASURED = FIXTURES / "squeezenet1.1-13.int8.measured.json"

QDQ_OPS = {"QuantizeLinear", "DequantizeLinear"}


def _fetch(trace, model, measured):
    return fetch_profile(profile=str(trace), model=str(model), measured=str(measured))


def _measurement(tmp_path, warmup_runs, name="measured.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"warmup_runs": warmup_runs}), encoding="utf-8")
    return path


def _event(name, op_type, duration):
    return {
        "cat": "Node",
        "name": f"{name}_kernel_time",
        "dur": duration,
        "args": {"op_name": op_type},
    }


def _write_trace(tmp_path, events, name="trace.json"):
    path = tmp_path / name
    path.write_text(json.dumps(events), encoding="utf-8")
    return path


def _onnx_node_names(model_path):
    return {node.name for node in onnx.load(str(model_path)).graph.node}


def test_fp32_compute_table_matches_fixture_with_no_boundary():
    table = _fetch(FP32_TRACE, FP32_MODEL, FP32_MEASURED)

    assert table["precision"] == "fp32"
    assert table["boundary"] == {}
    assert set(table["compute"]) <= _onnx_node_names(FP32_MODEL)
    assert table["compute"]["squeezenet0_conv0_fwd"] == {
        "op_type": "Conv",
        "avg_us": pytest.approx(631.3333333333334),
        "n": 3,
    }
    assert table["compute"]["squeezenet0_relu0_fwd"]["avg_us"] == pytest.approx(
        127.33333333333333
    )
    assert all(entry["n"] == 3 for entry in table["compute"].values())
    assert table["meta"]["total_kernel_us"] == pytest.approx(8674.666666666666)


def test_int8_boundary_has_every_qdq_node_with_resolved_layer_and_tensor():
    table = _fetch(INT8_TRACE, INT8_MODEL, INT8_MEASURED)

    model = onnx.load(str(INT8_MODEL))
    qdq_nodes = {node.name for node in model.graph.node if node.op_type in QDQ_OPS}
    compute_nodes = {node.name for node in model.graph.node if node.op_type not in QDQ_OPS}

    assert table["precision"] == "int8"
    assert set(table["boundary"]) == qdq_nodes
    assert set(table["compute"]) <= compute_nodes
    assert "squeezenet0_conv0_fwd" in table["compute"]
    assert table["compute"]["squeezenet0_conv0_fwd"]["op_type"] == "Conv"

    for name, entry in table["boundary"].items():
        assert entry["op_type"] in QDQ_OPS
        assert entry["serves_layer"] in table["compute"]
        assert entry["tensor"]
        assert entry["avg_us"] >= 0.0

    assert table["boundary"]["data_QuantizeLinear"]["serves_layer"] == "squeezenet0_conv0_fwd"
    assert (
        table["boundary"]["squeezenet0_flatten0_reshape0_DequantizeLinear"]["serves_layer"]
        == "squeezenet0_flatten0_reshape0"
    )
    # Runtime duplicates are merged under the real Q/DQ node name.
    assert not any("/" in name for name in table["boundary"])


def test_warmup_occurrences_are_dropped_before_averaging(tmp_path):
    trace = _write_trace(
        tmp_path,
        [
            _event("squeezenet0_conv0_fwd", "Conv", 1000),
            _event("squeezenet0_conv0_fwd", "Conv", 900),
            _event("squeezenet0_conv0_fwd", "Conv", 4),
            _event("squeezenet0_conv0_fwd", "Conv", 6),
        ],
    )
    measured = _measurement(tmp_path, warmup_runs=2)

    table = _fetch(trace, FP32_MODEL, measured)

    assert table["compute"]["squeezenet0_conv0_fwd"] == {
        "op_type": "Conv",
        "avg_us": pytest.approx(5.0),
        "n": 2,
    }


def test_compute_node_without_onnx_counterpart_raises(tmp_path):
    trace = _write_trace(tmp_path, [_event("not_a_real_layer", "Conv", 10)])
    measured = _measurement(tmp_path, warmup_runs=0)

    with pytest.raises(ValueError, match="no counterpart"):
        _fetch(trace, FP32_MODEL, measured)


def test_output_makes_no_end_to_end_equality_claim():
    table = _fetch(INT8_TRACE, INT8_MODEL, INT8_MEASURED)

    expected_total = sum(entry["avg_us"] for entry in table["compute"].values())
    expected_total += sum(entry["avg_us"] for entry in table["boundary"].values())

    assert set(table) == {"precision", "compute", "boundary", "meta"}
    assert set(table["meta"]) == {"total_kernel_us", "note"}
    assert table["meta"]["total_kernel_us"] == pytest.approx(expected_total)
    assert "NOT end-to-end latency" in table["meta"]["note"]
    assert not any("end_to_end" in key for key in table["meta"])


def test_malformed_trace_raises_clear_error(tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    measured = _measurement(tmp_path, warmup_runs=0)

    with pytest.raises(ValueError, match="JSON"):
        _fetch(bad_json, FP32_MODEL, measured)

    not_a_list = tmp_path / "object.json"
    not_a_list.write_text(json.dumps({"events": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        _fetch(not_a_list, FP32_MODEL, measured)


def test_trace_without_kernel_events_raises_profiling_hint(tmp_path):
    trace = _write_trace(
        tmp_path, [{"cat": "Session", "name": "model_loading_uri", "dur": 1, "args": {}}]
    )
    measured = _measurement(tmp_path, warmup_runs=0)

    with pytest.raises(ValueError, match=r"no node events; was --profile set\?"):
        _fetch(trace, FP32_MODEL, measured)


def test_unattributable_qdq_tensor_raises(tmp_path):
    model_path = tmp_path / "qdq_only.onnx"
    scale = numpy_helper.from_array(np.array(0.5, dtype=np.float32), "scale")
    zero_point = numpy_helper.from_array(np.array(0, dtype=np.int8), "zero_point")
    graph = helper.make_graph(
        [helper.make_node("QuantizeLinear", ["X", "scale", "zero_point"], ["Y"], name="Q")],
        "qdq_only",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.INT8, [1, 4])],
        initializer=[scale, zero_point],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, model_path)

    trace = _write_trace(tmp_path, [_event("Q", "QuantizeLinear", 10)])
    measured = _measurement(tmp_path, warmup_runs=0)

    with pytest.raises(ValueError, match="attribute"):
        _fetch(trace, model_path, measured)


def test_missing_warmup_count_raises(tmp_path):
    trace = _write_trace(tmp_path, [_event("squeezenet0_conv0_fwd", "Conv", 10)])
    measured = tmp_path / "measured.json"
    measured.write_text(json.dumps({"benchmark_runs": 3}), encoding="utf-8")

    with pytest.raises(ValueError, match="warmup_runs"):
        _fetch(trace, FP32_MODEL, measured)

    with pytest.raises(ValueError, match="warmup_runs"):
        _fetch(trace, FP32_MODEL, tmp_path / "missing.json")


def test_too_many_warmups_leaves_no_events(tmp_path):
    trace = _write_trace(tmp_path, [_event("squeezenet0_conv0_fwd", "Conv", 10)])
    measured = _measurement(tmp_path, warmup_runs=1)

    with pytest.raises(ValueError, match="warm-up"):
        _fetch(trace, FP32_MODEL, measured)


def test_two_trace_nodes_colliding_on_a_layer_raise(tmp_path):
    trace = _write_trace(
        tmp_path,
        [
            _event("squeezenet0_conv0_fwd", "Conv", 10),
            _event("squeezenet0_conv0_fwd_nchwc", "Conv", 10),
        ],
    )
    measured = _measurement(tmp_path, warmup_runs=0)

    with pytest.raises(ValueError, match="collide"):
        _fetch(trace, FP32_MODEL, measured)


def test_cli_writes_the_table(tmp_path):
    out = tmp_path / "table.json"

    assert (
        main(
            [
                "--profile",
                str(FP32_TRACE),
                "--model",
                str(FP32_MODEL),
                "--measured",
                str(FP32_MEASURED),
                "--out",
                str(out),
            ]
        )
        == 0
    )

    table = json.loads(out.read_text(encoding="utf-8"))
    assert table["precision"] == "fp32"
    assert table["compute"]["squeezenet0_conv0_fwd"]["n"] == 3
