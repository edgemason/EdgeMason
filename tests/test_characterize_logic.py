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
