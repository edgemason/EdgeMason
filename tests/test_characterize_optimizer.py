import hashlib
import io
import json
import tempfile
import zipfile

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from api.optimizer.benchmark_engine import BenchmarkEngine, benchmark
from api.optimizer.quantizer import Quantizer
from api.packager.generator import create_deployment_package
from tests.helpers import assert_matches_golden


@pytest.fixture
def tiny_conv(tmp_path):
    seeded_rng = np.random.default_rng(42)
    weights = seeded_rng.random((4, 3, 3, 3)).astype(np.float32)
    model_path = tmp_path / "conv.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["X", "W"], ["H"]),
            helper.make_node("Relu", ["H"], ["Y"]),
        ],
        "conv",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 6, 6])],
        initializer=[
            helper.make_tensor("W", TensorProto.FLOAT, weights.shape, weights.tobytes(), raw=True)
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, model_path)

    calib = seeded_rng.random((8, 3, 8, 8)).astype(np.float32)
    calib_path = tmp_path / "calib.npy"
    np.save(calib_path, calib)
    return model_path, calib_path


def _op_type_list(model_path):
    model = onnx.load(str(model_path))
    return sorted(n.op_type for n in model.graph.node)


def _initializer_info(model_path):
    model = onnx.load(str(model_path))
    return [
        {"name": i.name, "dtype": onnx.TensorProto.DataType.Name(i.data_type), "shape": list(i.dims)}
        for i in model.graph.initializer
    ]


# ---------------------------------------------------------------------------
# T1: Quantizer
# ---------------------------------------------------------------------------


def test_static_int8_conv(tiny_conv):
    model_path, calib_path = tiny_conv
    q = Quantizer(str(tmp_path if False else model_path.parent / "out"))
    out = q.create_static_int8(str(model_path), str(calib_path))

    loaded = onnx.load(out)
    # weight dequant zero-point is int8
    weight_zp = None
    weight_scale = None
    weight_axis = None
    for node in loaded.graph.node:
        if node.op_type == "DequantizeLinear":
            # identify weight dequant by presence of axis attr
            axis_attr = [a for a in node.attribute if a.name == "axis"]
            if axis_attr:
                weight_zp = node.input[2]
                weight_scale = node.input[1]
                weight_axis = axis_attr[0].i
    assert weight_zp is not None
    zp_tensor = [i for i in loaded.graph.initializer if i.name == weight_zp][0]
    assert zp_tensor.data_type == onnx.TensorProto.INT8
    scale_tensor = [i for i in loaded.graph.initializer if i.name == weight_scale][0]
    assert list(scale_tensor.dims) == [4]
    assert weight_axis == 0
    # activation zero-point int8
    # find X dequant (no axis attr) zero point dtype
    for node in loaded.graph.node:
        if node.op_type == "DequantizeLinear":
            axis_attr = [a for a in node.attribute if a.name == "axis"]
            if not axis_attr:
                act_zp_name = node.input[2]
                act_zp = [i for i in loaded.graph.initializer if i.name == act_zp_name][0]
                assert act_zp.data_type == onnx.TensorProto.INT8
                break

    assert_matches_golden(
        "static_int8_tiny_conv",
        {
            "op_types": _op_type_list(out),
            "initializers": _initializer_info(out),
        },
    )


def test_quant_pre_process_called_once(tiny_conv, monkeypatch):
    import api.optimizer.quantizer as qmod

    model_path, calib_path = tiny_conv
    real = qmod.quant_pre_process
    calls = []

    def wrapper(model_input, model_output, skip_symbolic_shape=False):
        calls.append(bool(skip_symbolic_shape))
        return real(model_input, model_output, skip_symbolic_shape=skip_symbolic_shape)

    monkeypatch.setattr(qmod, "quant_pre_process", wrapper)
    out_dir = model_path.parent / "out2"
    q = Quantizer(str(out_dir))
    q.create_static_int8(str(model_path), str(calib_path))
    assert len(calls) == 1
    assert calls == [True]


def test_quant_pre_process_failure_fallback(tiny_conv, monkeypatch):
    import api.optimizer.quantizer as qmod

    model_path, calib_path = tiny_conv

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(qmod, "quant_pre_process", boom)
    q = Quantizer(str(model_path.parent / "out3"))
    out = q.create_static_int8(str(model_path), str(calib_path))
    # # QUIRK: silent fallback
    loaded = onnx.load(out)
    assert loaded.graph.node


def test_dynamic_int8_tiny(tiny_onnx):
    from tests.conftest import tiny_onnx as _  # noqa

    q = Quantizer(str(tiny_onnx.parent / "outdyn"))
    out = q.create_dynamic_int8(str(tiny_onnx))
    loaded = onnx.load(out)
    # find weight zero point initializer
    zp_names = [i.name for i in loaded.graph.initializer if "zero_point" in i.name]
    # uint8 check
    found_uint8 = False
    for i in loaded.graph.initializer:
        if "zero_point" in i.name and i.name.startswith("W"):
            assert i.data_type == onnx.TensorProto.UINT8
            found_uint8 = True
    assert found_uint8
    assert_matches_golden(
        "dynamic_int8_tiny",
        {
            "op_types": _op_type_list(out),
            "initializers": _initializer_info(out),
        },
    )


def test_create_fp32_bytes_equal(tiny_onnx):
    q = Quantizer(str(tiny_onnx.parent / "outfp"))
    out = q.create_fp32(str(tiny_onnx))
    assert tiny_onnx.read_bytes() == (tiny_onnx.parent / "outfp" / "model_fp32.onnx").read_bytes() or (tiny_onnx.read_bytes() == __import__("pathlib").Path(out).read_bytes())


def test_quantize_for_candidate(tiny_onnx, tiny_conv):
    conv_model, calib_path = tiny_conv
    q = Quantizer(str(tiny_onnx.parent / "outq"))
    # fp32
    p = q.quantize_for_candidate(str(tiny_onnx), "onnx_fp32_baseline")
    assert str(p).endswith(".onnx")
    # dynamic
    p = q.quantize_for_candidate(str(tiny_onnx), "onnx_dynamic_int8")
    assert str(p).endswith(".onnx")
    # static with calibration
    p = q.quantize_for_candidate(str(tiny_conv[0]), "onnx_static_int8", calibration_data=str(calib_path))
    assert str(p).endswith(".onnx")
    # static without
    with pytest.raises(ValueError, match="Static INT8 requires calibration data"):
        q.quantize_for_candidate(str(tiny_onnx), "onnx_static_int8")
    # bogus
    with pytest.raises(ValueError, match="Unknown candidate"):
        q.quantize_for_candidate(str(tiny_onnx), "bogus")


def test_static_int8_determinism(tiny_conv):
    model_path, calib_path = tiny_conv
    d1 = model_path.parent / "det1"
    d2 = model_path.parent / "det2"
    q1 = Quantizer(str(d1))
    q2 = Quantizer(str(d2))
    p1 = q1.create_static_int8(str(model_path), str(calib_path))
    p2 = q2.create_static_int8(str(model_path), str(calib_path))
    assert __import__("pathlib").Path(p1).read_bytes() == __import__("pathlib").Path(p2).read_bytes()


# ---------------------------------------------------------------------------
# T2: Benchmark + packaging
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_benchmark_isolated_format(tiny_onnx):
    result = json.loads(benchmark(str(tiny_onnx), isolated=True, warmup_runs=2, benchmark_runs=10))
    # # QUIRK: renamed
    assert set(result["memory_mb"].keys()) == {"baseline", "peak", "delta"}
    assert_matches_golden("benchmark_isolated_format", result)


@pytest.mark.slow
def test_benchmark_inprocess_format(tiny_onnx):
    engine = BenchmarkEngine(hardware_id="rpi4", execution_provider="cpu")
    result = engine.benchmark(str(tiny_onnx), warmup_runs=2, benchmark_runs=10).to_dict()
    # # QUIRK: differs from isolated
    assert set(result["memory_mb"].keys()) == {"baseline_mb", "peak_mb", "delta_mb"}
    assert result["hardware_estimate"] is not None
    # # QUIRK: dead estimate, overwritten by pipeline (uses latency_scaling_factor not per-quant)
    assert result["hardware_estimate"]["estimated_latency_p50_ms"] == round(
        result["latency_ms"]["p50"] * 11.8, 2
    )
    assert_matches_golden("benchmark_inprocess_format", result)


def test_benchmark_failure_format():
    result = json.loads(benchmark("missing.onnx", isolated=False))
    assert_matches_golden("benchmark_failure_format", result)


def test_package_tiny_rpi4(tiny_onnx):
    report = {
        "latency_ms": {"p50": 12.3},
        "hardware_estimate": {"latency_p50_ms": 145.2, "memory_mb": 25.9},
        "accuracy": {"top1_accuracy": 0.575},
    }
    candidate = {"name": "onnx_fp32_baseline", "framework": "onnxruntime", "quantization": "fp32"}
    out_zip = tiny_onnx.parent / "pkg_rpi4.zip"
    create_deployment_package(
        model_path=str(tiny_onnx),
        benchmark_report=report,
        hardware_id="rpi4",
        output_path=str(out_zip),
        candidate=candidate,
    )
    contents = {}
    with zipfile.ZipFile(out_zip) as zf:
        for info in zf.infolist():
            data = zf.read(info.filename)
            try:
                contents[info.filename] = data.decode("utf-8")
            except UnicodeDecodeError:
                contents[info.filename] = "sha256:" + hashlib.sha256(data).hexdigest()
    assert_matches_golden("package_tiny_rpi4", contents)


def test_package_tiny_x86_64(tiny_onnx):
    report = {
        "latency_ms": {"p50": 12.3},
        "hardware_estimate": {"latency_p50_ms": 145.2, "memory_mb": 25.9},
        "accuracy": {"top1_accuracy": 0.575},
    }
    candidate = {"name": "onnx_fp32_baseline", "framework": "onnxruntime", "quantization": "fp32"}
    out_zip = tiny_onnx.parent / "pkg_x86.zip"
    create_deployment_package(
        model_path=str(tiny_onnx),
        benchmark_report=report,
        hardware_id="x86_64",
        output_path=str(out_zip),
        candidate=candidate,
    )
    contents = {}
    with zipfile.ZipFile(out_zip) as zf:
        for info in zf.infolist():
            data = zf.read(info.filename)
            try:
                contents[info.filename] = data.decode("utf-8")
            except UnicodeDecodeError:
                contents[info.filename] = "sha256:" + hashlib.sha256(data).hexdigest()
    assert_matches_golden("package_tiny_x86_64", contents)
