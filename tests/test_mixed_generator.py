"""P2.5: mixed-precision generator (per-layer fp32/int8).

The generator takes an explicit list of pre-processed ONNX node names to keep in
fp32 and quantizes the rest to INT8 with ORT's QOperator format. These tests use
the committed `squeezenet1.1-13` FP32 fixture, so they need no provisioned
assets.
"""

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnxruntime.quantization.shape_inference import quant_pre_process

from api.optimizer.fetcher import fetch_profile
from api.optimizer.quantizer import Quantizer

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fetcher"
FP32_MODEL = FIXTURES / "squeezenet1.1-13.fp32.onnx"

INTEGER_CONV_OPS = frozenset({"QLinearConv", "ConvInteger"})
BOUNDARY_OPS = frozenset({"QuantizeLinear", "DequantizeLinear"})

# Calibration count is small on purpose: these tests check structure, not
# accuracy (P2.4 owns the accuracy gate).
CALIBRATION_SHAPE = (4, 3, 224, 224)


def _calibration_npy(tmp_path: Path) -> str:
    path = tmp_path / "calibration.npy"
    np.save(path, np.random.default_rng(0).random(CALIBRATION_SHAPE, dtype=np.float32))
    return str(path)


def _prepared_conv_names(tmp_path: Path) -> list[str]:
    """Conv node names in the graph `quantize_static` actually sees."""
    prepped = tmp_path / "prepped_vocabulary.onnx"
    quant_pre_process(str(FP32_MODEL), str(prepped), skip_symbolic_shape=True)
    return [
        node.name
        for node in onnx.load(str(prepped)).graph.node
        if node.op_type == "Conv"
    ]


def _conv_ops(model_path: str) -> dict[str, str]:
    return {
        node.name: node.op_type
        for node in onnx.load(model_path).graph.node
        if "Conv" in node.op_type
    }


def _profile(model_path: Path, workdir: Path, tag: str, warmup_runs: int = 1, runs: int = 3):
    """Profile one model and return the P2.3 fetcher's per-layer tables."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # Keep every ONNX node visible so trace names map back to the graph.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_profiling = True
    options.profile_file_prefix = str(workdir / f"{tag}.profile")
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_meta = session.get_inputs()[0]
    shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in input_meta.shape]
    feed = {input_meta.name: np.random.default_rng(0).random(shape, dtype=np.float32)}
    for _ in range(warmup_runs + runs):
        session.run(None, feed)
    trace = Path(session.end_profiling())

    measured = workdir / f"{tag}.measured.json"
    measured.write_text(json.dumps({"warmup_runs": warmup_runs}), encoding="utf-8")
    return fetch_profile(str(trace), str(model_path), str(measured))


def _conv_entries(table):
    return {
        name: entry
        for name, entry in table["compute"].items()
        if entry["op_type"] in INTEGER_CONV_OPS or entry["op_type"] == "Conv"
    }


def _consumers_of(model_path: str, layer_name: str) -> set[str]:
    graph = onnx.load(model_path).graph
    produced = {
        output
        for node in graph.node
        if node.name == layer_name
        for output in node.output
    }
    return {
        node.name
        for node in graph.node
        if node.name and set(node.input) & produced
    }


def test_empty_fp32_list_matches_static_int8(tmp_path):
    """Spec #1: an empty exclude list is plain all-INT8."""
    calibration = _calibration_npy(tmp_path)
    quantizer = Quantizer(str(tmp_path / "out"))

    mixed_path = quantizer.create_mixed_int8(str(FP32_MODEL), calibration, [])
    static_path = quantizer.create_static_int8(str(FP32_MODEL), calibration)

    assert Path(mixed_path).name == "model_mixed_int8.onnx"
    mixed_ops = _conv_ops(mixed_path)
    static_ops = _conv_ops(static_path)
    assert mixed_ops, "fixture has no convolution layers"
    assert set(mixed_ops.values()) <= INTEGER_CONV_OPS
    assert mixed_ops == static_ops

    # The fetcher must see the same all-integer picture in both models.
    mixed_table = _profile(Path(mixed_path), tmp_path, "mixed_empty")
    assert all(
        entry["op_type"] in INTEGER_CONV_OPS
        for entry in _conv_entries(mixed_table).values()
    )


def test_excluding_every_quantizable_node_leaves_only_fp32_conv(tmp_path):
    """Spec #2: excluding every conv (here: every quantizable node) keeps them fp32."""
    calibration = _calibration_npy(tmp_path)
    quantizer = Quantizer(str(tmp_path / "out"))

    excludable = quantizer.excludable_node_names(str(FP32_MODEL))
    assert excludable, "pre-processed fixture has no quantizable nodes"

    mixed_path = quantizer.create_mixed_int8(str(FP32_MODEL), calibration, excludable)

    ops = _conv_ops(mixed_path)
    assert ops, "fixture has no convolution layers"
    assert set(ops.values()) == {"Conv"}
    assert not any(
        node.op_type in INTEGER_CONV_OPS
        for node in onnx.load(mixed_path).graph.node
    )


def test_subset_keeps_exactly_the_named_conv_fp32(tmp_path):
    """Spec #3: N named convs stay fp32, the rest are QLinearConv, with Q/DQ at the edges."""
    conv_names = _prepared_conv_names(tmp_path)
    assert len(conv_names) == 26
    # An interior conv so the fp32 layer has int8 neighbours on both sides.
    fp32_name = conv_names[13]

    calibration = _calibration_npy(tmp_path)
    quantizer = Quantizer(str(tmp_path / "out"))
    mixed_path = Path(
        quantizer.create_mixed_int8(str(FP32_MODEL), calibration, [fp32_name])
    )

    ops = _conv_ops(str(mixed_path))
    assert ops[fp32_name] == "Conv"
    assert all(
        op in INTEGER_CONV_OPS for name, op in ops.items() if name != fp32_name
    )

    table = _profile(mixed_path, tmp_path, "mixed_subset")
    traced = {
        name: entry["op_type"]
        for name, entry in _conv_entries(table).items()
    }
    assert traced[fp32_name] == "Conv"
    assert all(
        op in INTEGER_CONV_OPS for name, op in traced.items() if name != fp32_name
    )

    # Boundary nodes must sit on the fp32 layer's edges, never inside the INT8
    # run: apart from the model-input QuantizeLinear (which ORT attributes to
    # the first conv), the only convolution the fetcher attributes Q/DQ to is
    # the fp32 one.
    boundary = table["boundary"]
    assert boundary, "mixed model should have Q/DQ boundary nodes"
    served = {entry["serves_layer"] for entry in boundary.values()}
    graph_conv_order = [
        node.name
        for node in onnx.load(str(mixed_path)).graph.node
        if "Conv" in node.op_type
    ]
    served_convs = {layer for layer in served if layer in ops}
    assert fp32_name in served_convs
    assert served_convs - {fp32_name} <= {graph_conv_order[0]}
    assert any(
        entry["op_type"] == "DequantizeLinear" and entry["serves_layer"] == fp32_name
        for entry in boundary.values()
    )

    # The output edge is converted too: a direct consumer of the fp32 conv
    # carries a QuantizeLinear before the next integer conv.
    consumers = _consumers_of(str(mixed_path), fp32_name)
    assert any(
        entry["op_type"] == "QuantizeLinear" and entry["serves_layer"] in consumers
        for entry in boundary.values()
    )


def test_unknown_fp32_node_is_rejected_by_name(tmp_path):
    """Spec #4: a name absent from the pre-processed graph raises, not silently ignored."""
    quantizer = Quantizer(str(tmp_path / "out"))
    with pytest.raises(ValueError, match="definitely_not_a_node"):
        quantizer.create_mixed_int8(
            str(FP32_MODEL),
            _calibration_npy(tmp_path),
            ["definitely_not_a_node"],
        )


def test_non_quantizable_fp32_node_is_rejected(tmp_path):
    """Spec #5 (failures): a real but non-quantizable node raises; the caller's model is wrong."""
    quantizer = Quantizer(str(tmp_path / "out"))
    # Dropout is present in the pre-processed fixture but has no QLinear form.
    with pytest.raises(ValueError, match="squeezenet0_dropout0_fwd"):
        quantizer.create_mixed_int8(
            str(FP32_MODEL),
            _calibration_npy(tmp_path),
            ["squeezenet0_dropout0_fwd"],
        )


def test_same_inputs_produce_byte_identical_models(tmp_path):
    """Spec #5: generation is deterministic."""
    calibration = _calibration_npy(tmp_path)
    conv_names = _prepared_conv_names(tmp_path)
    fp32_names = [conv_names[3], conv_names[13]]

    first = Path(
        Quantizer(str(tmp_path / "out_a")).create_mixed_int8(
            str(FP32_MODEL), calibration, fp32_names
        )
    )
    second = Path(
        Quantizer(str(tmp_path / "out_b")).create_mixed_int8(
            str(FP32_MODEL), calibration, fp32_names
        )
    )

    assert first.read_bytes() == second.read_bytes()
