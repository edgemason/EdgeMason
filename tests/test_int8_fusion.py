"""P2.4: static INT8 must execute real integer Conv kernels, not fp32 Conv + Q/DQ.

The fusion check runs on the committed, real `squeezenet1.1-13` FP32 fixture, so
it needs no provisioned assets. It quantises that model with the shipped
`create_static_int8` config, profiles both models with ONNX Runtime, and asks the
P2.3 fetcher what kernels actually ran.

The accuracy check needs the provisioned model and labelled sample (see
`scripts/provision_assets.py`); it skips with a clear reason when they are absent
rather than silently passing.
"""

import json
import platform
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from api.optimizer.fetcher import fetch_profile
from api.optimizer.quantizer import Quantizer
from api.optimizer.validation import (
    evaluate_onnx,
    load_validation_data,
    split_calibration_and_test,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fetcher"

FP32_MODEL = FIXTURES / "squeezenet1.1-13.fp32.onnx"

INTEGER_CONV_OPS = frozenset({"QLinearConv", "ConvInteger"})

# Reported in reports/EdgeMason_Report.md ("within 2.3 points"), sourced by
# FINDINGS F-09 from reports/data/quantization_analysis.json. P2.4 must not
# trade more accuracy than this for speed.
RECORDED_TOP1_TOLERANCE = 0.023


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
    feed = {
        input_meta.name: np.random.default_rng(0).random(shape, dtype=np.float32)
    }
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
        if "Conv" in entry["op_type"] or entry["op_type"] in INTEGER_CONV_OPS
    }


def _has_int8_dot_product_instructions() -> bool:
    """True where int8 Conv can genuinely beat fp32 Conv on this host."""
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        # Cortex-A76 (Pi 5) and newer have the dot-product extension.
        return True
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return False
    flags = cpuinfo.read_text(encoding="utf-8", errors="ignore")
    return "avx_vnni" in flags or "avx512_vnni" in flags


def test_static_int8_emits_integer_conv_kernels_not_float_conv(tmp_path):
    """Spec #1: the quantized graph contains integer Conv kernels, not float Conv."""
    calibration = np.random.default_rng(0).random(
        (4, 3, 224, 224), dtype=np.float32
    )
    calibration_path = tmp_path / "calibration.npy"
    np.save(calibration_path, calibration)

    quantizer = Quantizer(str(tmp_path / "out"))
    int8_path = Path(quantizer.create_static_int8(str(FP32_MODEL), str(calibration_path)))

    graph = onnx.load(str(int8_path)).graph
    conv_ops = [
        node.op_type for node in graph.node if "Conv" in node.op_type
    ]
    assert conv_ops, "fixture has no convolution layers"
    assert set(conv_ops) <= INTEGER_CONV_OPS, (
        f"expected integer Conv kernels only, found {sorted(set(conv_ops))}"
    )


def test_fetched_trace_shows_integer_conv_and_lower_conv_time(tmp_path):
    """Spec #1/#2: the fetcher sees integer kernels and int8 Conv is faster."""
    calibration = np.random.default_rng(0).random(
        (4, 3, 224, 224), dtype=np.float32
    )
    calibration_path = tmp_path / "calibration.npy"
    np.save(calibration_path, calibration)

    quantizer = Quantizer(str(tmp_path / "out"))
    int8_path = Path(quantizer.create_static_int8(str(FP32_MODEL), str(calibration_path)))

    fp32_table = _profile(FP32_MODEL, tmp_path, "fp32")
    int8_table = _profile(int8_path, tmp_path, "int8")

    fp32_conv = _conv_entries(fp32_table)
    int8_conv = _conv_entries(int8_table)

    assert fp32_conv, "FP32 trace has no Conv kernels"
    assert int8_conv, "INT8 trace has no Conv kernels"
    assert all(entry["op_type"] == "Conv" for entry in fp32_conv.values())
    assert all(entry["op_type"] in INTEGER_CONV_OPS for entry in int8_conv.values()), (
        "INT8 trace still contains fp32 Conv kernels: "
        f"{sorted({entry['op_type'] for entry in int8_conv.values()})}"
    )

    if not _has_int8_dot_product_instructions():
        pytest.skip(
            "host has no int8 dot-product instructions (VNNI/dotprod); "
            "the speed gate is deferred to the Pi 5"
        )

    fp32_us = sum(entry["avg_us"] for entry in fp32_conv.values())
    int8_us = sum(entry["avg_us"] for entry in int8_conv.values())
    assert int8_us < fp32_us, (
        f"int8 Conv time {int8_us:.1f} us is not lower than fp32 {fp32_us:.1f} us"
    )


@pytest.mark.slow
def test_static_int8_accuracy_within_recorded_tolerance(tmp_path):
    """Spec #3: top-1 stays within the recorded 2.3-point tolerance."""
    model_path = PROJECT_ROOT / "models" / "squeezenet1.1-13.onnx"
    sample_path = PROJECT_ROOT / "models" / "samples" / "imagenetv2_mobilenetv2_256.npz"
    if not model_path.exists() or not sample_path.exists():
        pytest.skip(
            "squeezenet1.1-13 or its labelled sample is not provisioned; "
            "run scripts/provision_assets.py"
        )

    split = split_calibration_and_test(load_validation_data(sample_path))
    quantizer = Quantizer(str(tmp_path / "out"))
    int8_path = quantizer.create_static_int8(str(model_path), split.calibration)

    fp32_accuracy = evaluate_onnx(model_path, split.test)
    int8_accuracy = evaluate_onnx(int8_path, split.test)

    top1_drop = fp32_accuracy["top1_accuracy"] - int8_accuracy["top1_accuracy"]
    assert top1_drop <= RECORDED_TOP1_TOLERANCE, (
        f"INT8 top-1 {int8_accuracy['top1_accuracy']:.4f} is "
        f"{top1_drop:.4f} below FP32 {fp32_accuracy['top1_accuracy']:.4f}; "
        f"recorded tolerance is {RECORDED_TOP1_TOLERANCE:.3f}"
    )
