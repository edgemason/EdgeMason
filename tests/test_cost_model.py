"""P2.7: graph-aware cost model.

The fast tests use hand-built graphs with known activation sizes and synthetic
registers, so feature extraction, the OLS fit and the cross-validation report
are exercised without timing a real model.
"""

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from api.optimizer import cost_model
from api.optimizer.fetcher import BoundaryEntry


def _mixed_graph(tmp_path) -> str:
    """A synthetic mixed graph: int8 conv -> fp32 conv -> int8 conv.

    Tensors are named so precision is visible: quantized tensors end in
    ``_quantized``. The graph is structural only; it is never executed.
    """
    nodes = [
        helper.make_node("QuantizeLinear", ["X"], ["X_quantized"], name="qin"),
        helper.make_node("QLinearConv", ["X_quantized"], ["C0_quantized"], name="conv0_quant"),
        helper.make_node(
            "DequantizeLinear", ["C0_quantized"], ["C0"], name="C0_DequantizeLinear"
        ),
        helper.make_node("Conv", ["C0"], ["C1"], name="conv1"),
        helper.make_node("QuantizeLinear", ["C1"], ["C1_quantized"], name="q1"),
        helper.make_node(
            "QLinearConv", ["C1_quantized"], ["OUT_quantized"], name="conv2_quant"
        ),
        helper.make_node(
            "DequantizeLinear", ["OUT_quantized"], ["OUT"], name="OUT_DequantizeLinear"
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "mixed",
        [helper.make_tensor_value_info("X", TensorProto.UINT8, [1, 2, 4, 4])],
        [helper.make_tensor_value_info("OUT", TensorProto.FLOAT, [1, 5, 4, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    path = tmp_path / "mixed.onnx"
    onnx.save(model, path)
    return str(path)


def _boundary(op_type: str, tensor: str, serves_layer: str) -> BoundaryEntry:
    return {
        "op_type": op_type,
        "avg_us": 1.0,
        "serves_layer": serves_layer,
        "tensor": tensor,
    }


def _geometry():
    return cost_model.GraphGeometry(
        shape_map={
            "X": (1, 2, 4, 4),
            "C0": (1, 3, 4, 4),
            "C1": (1, 4, 4, 4),
            "OUT": (1, 5, 4, 4),
        },
        layer_outputs={"conv0": "C0", "conv1": "C1", "conv2": "OUT"},
    )


def test_features_equal_hand_computed_element_counts(tmp_path):
    """Spec #1: seam and stranded features are the hand-summed element counts."""
    mixed = _mixed_graph(tmp_path)
    mixed_boundary = {
        "qin": _boundary("QuantizeLinear", "X", "conv0"),
        "d0": _boundary("DequantizeLinear", "C0", "conv0"),
        "q1": _boundary("QuantizeLinear", "C1", "conv1"),
        "dout": _boundary("DequantizeLinear", "OUT", "conv2"),
    }
    # The pure-int8 model also converts the model input and output; those are
    # not fp32<->int8 seams introduced by the config, so they cancel out.
    baseline_boundary = {
        "qin": _boundary("QuantizeLinear", "X", "conv0"),
        "dout": _boundary("DequantizeLinear", "OUT", "conv2"),
    }

    features = cost_model.extract_features(
        _geometry(),
        ["conv1"],
        mixed_boundary,
        baseline_boundary,
        mixed,
    )
    # C0 is 1*3*4*4 = 48 elements, C1 is 1*4*4*4 = 64.
    assert features.seam_bytes == 48 + 64
    assert features.stranded_fp32_bytes == 64


def test_stranded_layer_expands_through_its_fp32_region(tmp_path):
    """`conv0` stays stranded even when relu/pool between it and int8 are fp32."""
    nodes = [
        helper.make_node("QLinearConv", ["X_quantized"], ["A_quantized"], name="convA_quant"),
        helper.make_node("DequantizeLinear", ["A_quantized"], ["A"], name="A_dq"),
        helper.make_node("Conv", ["A"], ["B"], name="convB"),
        helper.make_node("Relu", ["B"], ["C"], name="reluB"),
        helper.make_node("MaxPool", ["C"], ["D"], name="poolB"),
        helper.make_node("QuantizeLinear", ["D"], ["D_quantized"], name="q_out"),
        helper.make_node("QLinearConv", ["D_quantized"], ["E_quantized"], name="convC_quant"),
    ]
    graph = helper.make_graph(
        nodes,
        "mixed_region",
        [helper.make_tensor_value_info("X_quantized", TensorProto.UINT8, [1, 2, 4, 4])],
        [helper.make_tensor_value_info("E_quantized", TensorProto.UINT8, [1, 2, 4, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    path = tmp_path / "region.onnx"
    onnx.save(model, path)

    stranded = cost_model.stranded_fp32_layers(str(path), ["convB"])
    assert stranded == ("convB",)

    geometry = cost_model.GraphGeometry(
        shape_map={
            "A": (1, 3, 4, 4),
            "B": (1, 3, 4, 4),
            "C": (1, 3, 4, 4),
            "D": (1, 3, 4, 4),
        },
        layer_outputs={"convB": "B", "reluB": "C", "poolB": "D"},
    )
    assert cost_model.stranded_elements(geometry, stranded) == 48


def test_missing_shape_raises_instead_of_fabricating(tmp_path):
    """Spec #6: a missing activation shape stops the feature extraction."""
    mixed = _mixed_graph(tmp_path)
    geometry = cost_model.GraphGeometry(
        shape_map={"X": (1, 2, 4, 4), "C1": (1, 4, 4, 4)},
        layer_outputs={"conv1": "C1"},
    )
    with pytest.raises(cost_model.ShapeError, match="C0"):
        cost_model.extract_features(
            geometry,
            ["conv1"],
            {"d0": _boundary("DequantizeLinear", "C0", "conv0")},
            {},
            mixed,
        )
    with pytest.raises(cost_model.ShapeError):
        cost_model.element_count((1, -1, 4, 4))


def test_fit_recovers_synthetic_coefficients():
    """Spec #2: measured = additive + 2*seam + 3*stranded recovers alpha/beta."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(40):
        additive = float(rng.uniform(2000.0, 6000.0))
        seam = int(rng.integers(0, 1_000_000))
        stranded = int(rng.integers(0, 900_000))
        rows.append(
            {
                "additive_pred_us": additive,
                "seam_bytes": seam,
                "stranded_fp32_bytes": stranded,
                "measured_p50_us": additive + 2.0 * seam + 3.0 * stranded,
                "status": "ok",
            }
        )

    model = cost_model.fit_cost_model(rows)
    assert model.alpha == pytest.approx(2.0, rel=1e-9)
    assert model.beta == pytest.approx(3.0, rel=1e-9)


def test_zero_features_predict_the_additive_sum():
    """Spec #3: a config with no seams and no stranded layers is additive-only."""
    model = cost_model.CostModel(alpha=0.004, beta=0.002)
    features = cost_model.Features(
        additive_us=2100.0, seam_bytes=0, stranded_fp32_bytes=0
    )
    assert model.predict(features) == pytest.approx(2100.0)


def test_fit_is_deterministic():
    """Spec #5: the same rows and seed produce identical coefficients."""
    rng = np.random.default_rng(1)
    rows = [
        {
            "additive_pred_us": 3000.0,
            "seam_bytes": int(seam),
            "stranded_fp32_bytes": int(stranded),
            "measured_p50_us": 3000.0 + 0.5 * seam + 1.5 * stranded,
            "status": "ok",
        }
        for seam, stranded in zip(
            rng.integers(0, 500_000, 30), rng.integers(0, 500_000, 30), strict=True
        )
    ]
    first = cost_model.fit_cost_model(rows)
    second = cost_model.fit_cost_model(rows)
    assert (first.alpha, first.beta) == (second.alpha, second.beta)

    first_cv = cost_model.cross_validate(rows, k=5, seed=3)
    second_cv = cost_model.cross_validate(rows, k=5, seed=3)
    assert first_cv == second_cv


def test_cross_validation_reports_both_mape_and_stability(tmp_path):
    """Spec #4: k-fold CV produces out-of-sample MAPE for both models."""
    rng = np.random.default_rng(2)
    rows = []
    for _ in range(40):
        additive = float(rng.uniform(2000.0, 6000.0))
        seam = int(rng.integers(0, 1_000_000))
        stranded = int(rng.integers(0, 900_000))
        measured = additive + 0.002 * seam + 0.003 * stranded
        rows.append(
            {
                "additive_pred_us": additive,
                "seam_bytes": seam,
                "stranded_fp32_bytes": stranded,
                "measured_p50_us": measured,
                "status": "ok",
            }
        )

    report = cost_model.cross_validate(rows, k=5, seed=0)
    assert report["k"] == 5
    assert report["seed"] == 0
    assert len(report["folds"]) == 5
    assert report["out_of_sample"]["additive_mape"] >= 0.0
    assert report["out_of_sample"]["cost_model_mape"] < report["out_of_sample"]["additive_mape"]
    for fold in report["folds"]:
        assert {"alpha", "beta", "additive_mape", "cost_model_mape"} <= set(fold)
    assert report["stability"]["alpha_min"] <= report["stability"]["alpha_max"]
    assert report["stability"]["beta_min"] <= report["stability"]["beta_max"]

    path = tmp_path / "report.json"
    cost_model.write_report(report, path)
    assert json.loads(path.read_text())["out_of_sample"]["cost_model_mape"] == pytest.approx(
        report["out_of_sample"]["cost_model_mape"]
    )


def test_fold_too_small_raises():
    """Spec #6: a fold with too few training points stops instead of fitting."""
    rows = [
        {
            "additive_pred_us": 1000.0,
            "seam_bytes": index,
            "stranded_fp32_bytes": 0,
            "measured_p50_us": 1000.0 + index,
            "status": "ok",
        }
        for index in range(3)
    ]
    with pytest.raises(ValueError, match="three points"):
        cost_model.cross_validate(rows, k=3, seed=0)


@pytest.mark.slow
def test_real_tiny_model_features_are_nonzero(tmp_path):
    """End-to-end: a real quantized model yields both feature terms."""
    from api.optimizer.quantizer import Quantizer

    input_shape = (1, 3, 8, 8)
    nodes = []
    initializers = []
    source = "X"
    channels = [3, 4, 4]
    rng = np.random.default_rng(0)
    for index in range(2):
        weights = rng.random((channels[index + 1], channels[index], 3, 3)).astype(np.float32)
        bias = rng.random((channels[index + 1],)).astype(np.float32)
        nodes.append(
            helper.make_node(
                "Conv",
                [source, f"W{index}", f"B{index}"],
                [f"C{index}"],
                name=f"conv{index}",
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            )
        )
        initializers.append(
            helper.make_tensor(
                f"W{index}", TensorProto.FLOAT, weights.shape, weights.tobytes(), raw=True
            )
        )
        initializers.append(
            helper.make_tensor(
                f"B{index}", TensorProto.FLOAT, bias.shape, bias.tobytes(), raw=True
            )
        )
        source = f"C{index}"
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, list(input_shape))],
        [helper.make_tensor_value_info(source, TensorProto.FLOAT, [1, 4, 8, 8])],
        initializer=initializers,
    )
    base = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    base.ir_version = 9
    base_path = tmp_path / "tiny.onnx"
    onnx.save(base, base_path)

    quantizer = Quantizer(str(tmp_path / "models"))
    mixed_path = quantizer.create_mixed_int8(
        str(base_path),
        np.random.default_rng(0).random((2, 3, 8, 8), dtype=np.float32),
        ["conv0"],
    )
    geometry = cost_model.build_geometry(str(base_path))
    boundary = cost_model.fetch_raw_boundary(str(mixed_path), tmp_path / "raw")
    features = cost_model.extract_features(
        geometry, ["conv0"], boundary, {}, str(mixed_path)
    )
    assert features.seam_bytes > 0
    assert features.stranded_fp32_bytes > 0


@pytest.mark.slow
def test_real_sweep_register_produces_a_cross_validation_report(tmp_path):
    """Spec #4: a real sweep register feeds the CV report (win or no-win)."""
    from api.optimizer import sweep_runner

    model = Path(__file__).parent / "fixtures" / "fetcher" / "squeezenet1.1-13.fp32.onnx"
    calibration = np.random.default_rng(0).random((8, 3, 224, 224), dtype=np.float32)
    rows = sweep_runner.run_sweep(
        str(model),
        calibration,
        register_path=tmp_path / "register.jsonl",
        seed=0,
        target=7,
        work_dir=tmp_path / "work",
        warmup_runs=1,
        benchmark_runs=3,
        time_budget_seconds=0.5,
        isolated=False,
    )
    assert len(rows) == 7
    ok = [row for row in rows if row["status"] == "ok"]
    assert len(ok) >= 6
    for row in rows:
        assert row["seam_bytes"] is not None
        assert row["stranded_fp32_bytes"] is not None

    report = cost_model.cross_validate(rows, k=2, seed=0)
    assert report["out_of_sample"]["additive_mape"] >= 0.0
    assert report["out_of_sample"]["cost_model_mape"] >= 0.0
    assert {"alpha_us_per_element", "beta_us_per_element"} <= set(report["coefficients"])
    assert len(report["folds"]) == 2
