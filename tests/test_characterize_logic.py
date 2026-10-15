import math

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from api.optimizer.validation import (
    MAX_VALIDATION_BYTES,
    MAX_VALIDATION_SAMPLES,
    ValidationData,
    data_warnings,
    load_validation_data,
    split_calibration_and_test,
)
from api.planner.rule_based import (
    LATENCY_THRESHOLD_MS,
    SIZE_THRESHOLD_MB,
    PlannerInput,
    generate_candidates,
)

CNN_OPS = {"Conv": 52, "Clip": 35, "Add": 10, "Gemm": 1}
MATMUL_OPS = {"MatMul": 96, "Add": 120, "Softmax": 12, "Gemm": 2}
from api.services.hardware import estimate_on_target, select_execution_provider
from api.services.ranking import rank_results

from tests.helpers import assert_matches_golden


def tiny_model(path, input_shape, elem_type=TensorProto.FLOAT, outputs_shape=None,
               num_inputs=1, num_nodes=1, initializer_as_input=False):
    if outputs_shape is None:
        outputs_shape = list(input_shape)
    # Build graph. If initializer_as_input, use same name for an input.
    initializers = []
    nodes = []
    if num_inputs == 1:
        if initializer_as_input:
            shape_val = list(input_shape)
            w = np.zeros(shape_val, dtype=np.float32) if len(shape_val) > 0 else np.array(0.0, dtype=np.float32)
            initializers.append(helper.make_tensor("X", TensorProto.FLOAT, w.shape, w.tobytes(), raw=True))
            graph_inputs = [helper.make_tensor_value_info("X", elem_type, input_shape)]
        else:
            graph_inputs = [helper.make_tensor_value_info("X", elem_type, input_shape)]
    else:
        graph_inputs = []
        for i in range(num_inputs):
            graph_inputs.append(helper.make_tensor_value_info(f"X{i}", elem_type, input_shape))
    graph_outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, outputs_shape)]
    # nodes
    for i in range(num_nodes):
        if num_nodes == 1:
            if num_inputs == 1:
                nodes.append(helper.make_node("Identity", ["X"], ["Y"]))
            else:
                nodes.append(helper.make_node("Identity", ["X0"], ["Y"]))
        else:
            if i == 0:
                nodes.append(helper.make_node("Identity", ["X"], ["n0"]))
            elif i == num_nodes - 1:
                nodes.append(helper.make_node("Identity", [f"n{i-1}"], ["Y"]))
            else:
                nodes.append(helper.make_node("Identity", [f"n{i-1}"], [f"n{i}"]))
    graph = helper.make_graph(nodes, "m", graph_inputs, graph_outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)


# ---------------------------------------------------------------------------
# T1: Planner matrix
# ---------------------------------------------------------------------------


def test_planner_matrix():
    operator_variants = [
        {},
        CNN_OPS,
        MATMUL_OPS,
        {"Conv": 1, "MatMul": 2},
    ]
    model_sizes = [1.0, 5.0, 10.0]
    has_calibs = [False, True]
    include_tvms = [False, True]
    max_latencies = [None, 50.0, 100.0, 200.0]

    plans = []
    for ops in operator_variants:
        for ms in model_sizes:
            for has_cal in has_calibs:
                for inc_tvm in include_tvms:
                    for ml in max_latencies:
                        plan = generate_candidates(
                            PlannerInput(
                                model_size_mb=ms,
                                max_latency_ms=ml,
                                has_calibration_data=has_cal,
                                include_tvm=inc_tvm,
                                operator_counts=ops,
                            )
                        )
                        plans.append(plan)
    assert len(plans) == 4 * 3 * 2 * 2 * 4
    assert_matches_golden("planner_matrix", plans)


# ---------------------------------------------------------------------------
# T1: Ranking matrix
# ---------------------------------------------------------------------------


def _rank_result(name, quantization, host_p50_ms, memory_delta_mb, top1):
    from api.services.hardware import estimate_on_target

    return {
        "candidate": {"name": name, "quantization": quantization},
        "benchmark": {
            "success": True,
            "latency_ms": {"p50": host_p50_ms},
            "memory_mb": {"delta": memory_delta_mb},
        },
        "estimate": estimate_on_target("rpi4", quantization, host_p50_ms, memory_delta_mb),
        "accuracy": {"top1_accuracy": top1, "top5_accuracy": min(1.0, top1 + 0.1)},
    }


def test_ranking_matrix():
    results = [
        _rank_result("alpha", "fp32", 5.0, 50.0, 0.6),
        _rank_result("beta", "static_int8", 8.0, 20.0, 0.75),
        _rank_result("gamma", "dynamic_int8", 12.0, 5.0, 0.7),
        _rank_result("delta", "fp32", 3.0, 80.0, 0.5),
        _rank_result("epsilon", "static_int8", 10.0, 30.0, 0.8),
    ]
    priorities = ["latency", "memory", "accuracy", "balanced", "bogus"]
    constraints = {
        "none": (None, None),
        "max_latency": (100.0, None),
        "max_memory": (None, 50.0),
        "both": (100.0, 50.0),
    }

    matrix = {}
    for priority in priorities:
        matrix[priority] = {}
        for cname, (ml, mm) in constraints.items():
            ranked = rank_results(results, priority, ml, mm)
            matrix[priority][cname] = [r["candidate"]["name"] for r in ranked]

    assert_matches_golden("ranking_matrix", matrix)

    # when every result fails constraints -> []
    impossible = rank_results(results, "latency", 0.1, 0.1)
    assert impossible == []


# ---------------------------------------------------------------------------
# T2: Target estimate matrix
# ---------------------------------------------------------------------------


def test_estimate_matrix():
    hardware_ids = ["rpi3", "rpi4", "rpi5", "jetson_nano", "x86_64"]
    quants = ["fp32", "dynamic_int8", "static_int8", "unknown_q"]
    matrix = {}
    for hw in hardware_ids:
        matrix[hw] = {}
        for q in quants:
            matrix[hw][q] = estimate_on_target(hw, q, 10.0, 100.0)
    assert_matches_golden("estimate_matrix", matrix)


# ---------------------------------------------------------------------------
# T2: Provider selection matrix
# ---------------------------------------------------------------------------


def test_provider_matrix():
    import api.services.hardware as hardware

    hardware_ids = ["rpi3", "rpi4", "rpi5", "jetson_nano", "x86_64"]
    requesteds = ["auto", "cpu", "cuda"]
    include_tvms = [False, True]
    installed_opts = {
        "cpu_only": ["CPUExecutionProvider"],
        "cpu_cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }

    matrix = {}
    for hw in hardware_ids:
        matrix[hw] = {}
        for req in requesteds:
            matrix[hw][req] = {}
            for inc in include_tvms:
                matrix[hw][req][str(inc)] = {}
                for label, providers in installed_opts.items():
                    old = hardware.ort.get_available_providers
                    hardware.ort.get_available_providers = lambda p=list(providers): list(p)
                    try:
                        try:
                            res = select_execution_provider(hw, req, inc)
                            matrix[hw][req][str(inc)][label] = {"result": res}
                        except ValueError as exc:
                            matrix[hw][req][str(inc)][label] = {"error": str(exc)}
                    finally:
                        hardware.ort.get_available_providers = old
    assert_matches_golden("provider_matrix", matrix)


# ---------------------------------------------------------------------------
# T2: Data split and warnings matrices
# ---------------------------------------------------------------------------


def test_split_matrix():
    counts = [2, 3, 10, 100, 333, 1000]
    matrix = []
    for count in counts:
        data = ValidationData(
            inputs=np.arange(count, dtype=np.float32).reshape(count, 1, 1),
            labels=np.arange(count, dtype=np.int64),
        )
        split = split_calibration_and_test(data)
        matrix.append(
            {
                "sample_count": count,
                "calibration_count": len(split.calibration),
                "calibration": split.calibration.tolist(),
                "test_inputs": split.test.inputs.tolist(),
                "test_labels": split.test.labels.tolist(),
            }
        )
    assert_matches_golden("split_matrix", matrix)


def test_data_warnings_matrix():
    pairs = [
        (300, 700),
        (77, 179),
        (50, 35),
        (15, 5),
        (1, 99),
        (10, 10),
        (100, 200),
        (99, 1),
        (200, 1000),
    ]
    matrix = []
    for cal, test in pairs:
        matrix.append(
            {
                "calibration_count": cal,
                "test_count": test,
                "warnings": data_warnings(cal, test),
            }
        )
    assert_matches_golden("data_warnings_matrix", matrix)


def test_validation_limit_errors():
    import api.optimizer.validation as validation

    import tempfile
    from pathlib import Path

    # too many samples
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "many.npz"
        inputs = np.zeros((MAX_VALIDATION_SAMPLES + 1, 2), dtype=np.float32)
        labels = np.arange(MAX_VALIDATION_SAMPLES + 1, dtype=np.int64)
        np.savez(p, inputs=inputs, labels=labels)
        with pytest.raises(ValueError, match="must contain 1-1000 samples"):
            load_validation_data(p)

    # extra keys
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "extra.npz"
        np.savez(p, inputs=np.zeros((2, 2), dtype=np.float32), labels=np.array([0, 1], dtype=np.int64), extra=np.array([1]))
        with pytest.raises(ValueError, match="only inputs and labels"):
            load_validation_data(p)

    # bad dtypes
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "badinputs.npz"
        np.savez(p, inputs=np.zeros((2, 2), dtype=np.float64), labels=np.array([0, 1], dtype=np.int64))
        with pytest.raises(ValueError, match="float32 batch"):
            load_validation_data(p)
        p2 = Path(d) / "badlabels.npz"
        np.savez(p2, inputs=np.zeros((2, 2), dtype=np.float32), labels=np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="one-dimensional integer"):
            load_validation_data(p2)

    # NaN
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "nan.npz"
        arr = np.zeros((2, 2), dtype=np.float32)
        arr[0, 0] = float("nan")
        np.savez(p, inputs=arr, labels=np.array([0, 1], dtype=np.int64))
        with pytest.raises(ValueError, match="finite values"):
            load_validation_data(p)

    # too large (file size)
    old = validation.MAX_VALIDATION_BYTES
    validation.MAX_VALIDATION_BYTES = 1
    try:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "large.npz"
            np.savez(p, inputs=np.zeros((2, 2), dtype=np.float32), labels=np.array([0, 1], dtype=np.int64))
            with pytest.raises(ValueError, match="validation data is too large"):
                load_validation_data(p)
    finally:
        validation.MAX_VALIDATION_BYTES = old

    # expanded too large via _validate_inputs
    old = validation.MAX_VALIDATION_BYTES
    validation.MAX_VALIDATION_BYTES = 1
    try:
        with pytest.raises(ValueError, match="expanded validation data is too large"):
            validation._validate_inputs(np.zeros((2, 2), dtype=np.float32))
    finally:
        validation.MAX_VALIDATION_BYTES = old

    # calibration .npy too many samples
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "calib.npy"
        np.save(p, np.zeros((MAX_VALIDATION_SAMPLES + 1, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="must contain 1-1000 samples"):
            validation.load_calibration_inputs(p)


# ---------------------------------------------------------------------------
# T2: Model validation
# ---------------------------------------------------------------------------


def test_model_validation_cases(tmp_path):
    from api.services.model_validation import validate_model

    cases = []

    # 1 valid
    p = tmp_path / "valid.onnx"
    tiny_model(p, [1, 4])
    cases.append((p, None))

    # 2 dynamic batch (valid)
    p = tmp_path / "dynbatch.onnx"
    tiny_model(p, ["N", 4])
    cases.append((p, None))

    # 3 two inputs
    p = tmp_path / "two.onnx"
    tiny_model(p, [1, 4], num_inputs=2)
    cases.append((p, "the pipeline supports single-input models only"))

    # 4 int64
    p = tmp_path / "int64.onnx"
    tiny_model(p, [1, 4], elem_type=TensorProto.INT64)
    cases.append((p, "the pipeline supports float32 model inputs only"))

    # 5 dynamic non-batch dim
    p = tmp_path / "dynnonbatch.onnx"
    tiny_model(p, [1, "M"])
    cases.append((p, "only the model batch dimension may be dynamic"))

    # 6 dim too large
    p = tmp_path / "toobigdim.onnx"
    tiny_model(p, [1, 5000])
    cases.append((p, "model input dimension is too large"))

    # 7 too many elements
    p = tmp_path / "toomany.onnx"
    tiny_model(p, [1, 100, 100, 100, 11])
    cases.append((p, "model input tensor is too large"))

    # 8 too many nodes
    p = tmp_path / "toomany_nodes.onnx"
    tiny_model(p, [1, 4], num_nodes=10001)
    cases.append((p, "model has too many nodes"))

    # 9 initializer as input
    p = tmp_path / "init_as_input.onnx"
    tiny_model(p, [1, 4], initializer_as_input=True)
    cases.append((p, "the pipeline supports single-input models only"))

    for path, expected in cases:
        if expected is None:
            validate_model(path)  # should not raise
        else:
            with pytest.raises(ValueError) as exc:
                validate_model(path)
            assert str(exc.value) == expected
