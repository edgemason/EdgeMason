"""P2.6: pilot sweep runner.

These tests use a tiny four-conv model and stub the benchmark step, so the fast
gate exercises config generation, lookup building, seam measurement, the
additive prediction and the failed-row path without timing real models.
"""

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from api.optimizer import sweep_runner
from api.optimizer.quantizer import Quantizer

CONV_CHANNELS = [3, 4, 4, 4, 4]
INPUT_SHAPE = (1, 3, 8, 8)


def _tiny_conv_model(tmp_path: Path) -> Path:
    nodes = []
    initializers = []
    source = "X"
    rng = np.random.default_rng(0)
    for index in range(4):
        weights = rng.random((CONV_CHANNELS[index + 1], CONV_CHANNELS[index], 3, 3)).astype(
            np.float32
        )
        bias = rng.random((CONV_CHANNELS[index + 1],)).astype(np.float32)
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
        "tiny_conv",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, list(INPUT_SHAPE))],
        [helper.make_tensor_value_info(source, TensorProto.FLOAT, [1, 4, 8, 8])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    path = tmp_path / "tiny_conv.onnx"
    onnx.save(model, path)
    return path


def _calibration() -> np.ndarray:
    return np.random.default_rng(0).random((4, 3, 8, 8), dtype=np.float32)


def _ok_benchmark(*args, **kwargs) -> dict:
    return {"success": True, "error": None, "latency_ms": {"p50": 1.5, "p90": 1.8}}


def _sweep(model_path: Path, tmp_path: Path, monkeypatch, target: int = 8):
    monkeypatch.setattr(sweep_runner, "_benchmark_model", _ok_benchmark)
    register = tmp_path / "register.jsonl"
    rows = sweep_runner.run_sweep(
        str(model_path),
        _calibration(),
        register_path=str(register),
        seed=7,
        target=target,
        work_dir=str(tmp_path / "work"),
        isolated=False,
    )
    return rows, register


def test_configs_are_reproducible_and_span_seam_counts(tmp_path, monkeypatch):
    """Spec #1: same seed -> same configs; the sweep's seam counts vary."""
    layers = ["conv0", "conv1", "conv2", "conv3"]
    first = sweep_runner.generate_configs(layers, seed=7, target=12)
    second = sweep_runner.generate_configs(layers, seed=7, target=12)
    assert [(c.config_id, c.fp32_nodes) for c in first] == [
        (c.config_id, c.fp32_nodes) for c in second
    ]
    assert {len(c.fp32_nodes) for c in first} >= {0, 1, len(layers)}

    rows, _ = _sweep(_tiny_conv_model(tmp_path), tmp_path, monkeypatch, target=12)
    assert len({row["seam_count"] for row in rows}) > 1


def test_one_config_row_has_measurements_prediction_and_seam_count(tmp_path, monkeypatch):
    """Spec #2: a successful row carries measured p50/p90, additive_pred and seam count."""
    model_path = _tiny_conv_model(tmp_path)
    rows, _ = _sweep(model_path, tmp_path, monkeypatch, target=4)

    row = rows[0]
    assert row["status"] == "ok"
    assert row["error"] is None
    assert row["measured_p50_us"] == pytest.approx(1500.0)
    assert row["measured_p90_us"] == pytest.approx(1800.0)
    assert row["additive_pred_us"] > 0
    assert isinstance(row["seam_count"], int)
    assert row["fp32_nodes"] == []


def test_failed_config_is_recorded_and_sweep_continues(tmp_path, monkeypatch):
    """Spec #3 / B-59: a raising config becomes a failed row; later rows still run."""
    model_path = _tiny_conv_model(tmp_path)
    monkeypatch.setattr(sweep_runner, "_benchmark_model", _ok_benchmark)

    original = Quantizer.create_mixed_int8
    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated quantizer failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Quantizer, "create_mixed_int8", flaky)

    register = tmp_path / "register.jsonl"
    rows = sweep_runner.run_sweep(
        str(model_path),
        _calibration(),
        register_path=str(register),
        seed=7,
        target=8,
        work_dir=str(tmp_path / "work"),
        isolated=False,
    )

    assert len(rows) >= 3
    assert len({row["config_id"] for row in rows}) == len(rows)
    assert [row["status"] for row in rows].count("failed") == 1
    failed = next(row for row in rows if row["status"] == "failed")
    assert "simulated quantizer failure" in failed["error"]
    assert failed["measured_p50_us"] is None

    written = [json.loads(line) for line in register.read_text().splitlines()]
    assert len(written) == len(rows)
    assert any(row["status"] == "failed" for row in written)


def test_benchmark_failure_is_recorded_as_failed_row(tmp_path, monkeypatch):
    """Spec #3 / B-59: a run that reports failure is kept as a failed row."""
    model_path = _tiny_conv_model(tmp_path)

    def failed_benchmark(model_path_arg, *args, **kwargs) -> dict:
        # The lookup's pure-model measurements must still succeed; only the
        # per-config runs fail, which is what this test is about.
        if "model_mixed_int8" not in str(model_path_arg):
            return {"success": True, "error": None, "latency_ms": {"p50": 1.5, "p90": 1.8}}
        return {
            "success": False,
            "error": "simulated OOM",
            "latency_ms": {"p50": 0.0, "p90": 0.0},
        }

    monkeypatch.setattr(sweep_runner, "_benchmark_model", failed_benchmark)
    rows = sweep_runner.run_sweep(
        str(model_path),
        _calibration(),
        register_path=str(tmp_path / "register.jsonl"),
        seed=7,
        target=4,
        work_dir=str(tmp_path / "work"),
        isolated=False,
    )

    assert rows
    assert all(row["status"] == "failed" for row in rows)
    assert all("simulated OOM" in row["error"] for row in rows)


@pytest.mark.slow
def test_real_benchmark_records_measured_timing(tmp_path):
    """The shared bench protocol produces real p50/p90 values when not stubbed."""
    model_path = _tiny_conv_model(tmp_path)
    lookup = sweep_runner.build_lookup_table(
        str(model_path),
        _calibration(),
        quantizer=Quantizer(str(tmp_path / "lookup_models")),
        work_dir=tmp_path / "lookup_work",
        warmup_runs=1,
        benchmark_runs=3,
        time_budget_seconds=1.0,
        isolated=False,
    )
    assert lookup.regime in {"optimized", "calibrated"}
    assert {item["precision"] for item in lookup.raw_sanity} == {"fp32", "int8"}

    rows = sweep_runner.run_sweep(
        str(model_path),
        _calibration(),
        register_path=str(tmp_path / "register.jsonl"),
        seed=7,
        target=4,
        work_dir=str(tmp_path / "work"),
        warmup_runs=1,
        benchmark_runs=3,
        time_budget_seconds=1.0,
        isolated=False,
    )

    ok_rows = [row for row in rows if row["status"] == "ok"]
    assert ok_rows
    assert all(row["measured_p50_us"] > 0 for row in ok_rows)
    assert all(row["measured_p90_us"] >= row["measured_p50_us"] for row in ok_rows)


def test_additive_prediction_is_an_exact_hand_sum():
    """Spec #4: additive prediction is the plain sum of per-layer lookup costs."""
    lookup = {
        "a": sweep_runner.LayerTiming(fp32_us=1.5, int8_us=4.0),
        "b": sweep_runner.LayerTiming(fp32_us=2.0, int8_us=5.5),
        "c": sweep_runner.LayerTiming(fp32_us=10.0, int8_us=0.5),
    }
    assert sweep_runner.compute_additive_prediction(lookup, []) == 10.0
    assert sweep_runner.compute_additive_prediction(lookup, ["b"]) == 6.5
    assert sweep_runner.compute_additive_prediction(lookup, ["a", "c"]) == 17.0

    with pytest.raises(KeyError, match="missing"):
        sweep_runner.compute_additive_prediction(lookup, ["not_a_layer"])


def test_sanity_gate_accepts_matching_lookup_and_rejects_mismatch():
    """Amendment A #3: the gate passes a consistent lookup and raises otherwise."""
    layers = {
        "a": sweep_runner.LayerTiming(fp32_us=10.0, int8_us=20.0),
        "b": sweep_runner.LayerTiming(fp32_us=30.0, int8_us=40.0),
    }
    report = sweep_runner.check_lookup_sanity(
        layers, fp32_measured_us=40.0, int8_measured_us=60.0
    )
    assert [item["precision"] for item in report] == ["fp32", "int8"]
    assert all(item["rel_error"] == pytest.approx(0.0) for item in report)

    # Within tolerance (10% off) is accepted.
    sweep_runner.check_lookup_sanity(
        layers, fp32_measured_us=44.0, int8_measured_us=60.0
    )
    # A lookup from the wrong execution regime is rejected.
    with pytest.raises(sweep_runner.LookupSanityError, match="pure fp32"):
        sweep_runner.check_lookup_sanity(
            layers, fp32_measured_us=80.0, int8_measured_us=60.0
        )
    with pytest.raises(sweep_runner.LookupSanityError, match="pure int8"):
        sweep_runner.check_lookup_sanity(
            layers, fp32_measured_us=40.0, int8_measured_us=120.0
        )
    with pytest.raises(sweep_runner.LookupSanityError, match="must be positive"):
        sweep_runner.check_lookup_sanity(
            layers, fp32_measured_us=0.0, int8_measured_us=60.0
        )


def test_calibration_rescales_each_precision_to_its_measurement():
    """Amendment A #4: calibration makes both pure predictions exact."""
    layers = {
        "a": sweep_runner.LayerTiming(fp32_us=10.0, int8_us=20.0),
        "b": sweep_runner.LayerTiming(fp32_us=30.0, int8_us=40.0),
    }
    calibrated = sweep_runner.calibrate_lookup(
        layers, fp32_measured_us=50.0, int8_measured_us=90.0
    )
    assert sweep_runner.lookup_sums(calibrated) == pytest.approx((50.0, 90.0))
    report = sweep_runner.check_lookup_sanity(
        calibrated, fp32_measured_us=50.0, int8_measured_us=90.0
    )
    assert all(item["rel_error"] == pytest.approx(0.0) for item in report)

    with pytest.raises(sweep_runner.LookupSanityError, match="non-positive"):
        sweep_runner.calibrate_lookup(
            {"zero": sweep_runner.LayerTiming(fp32_us=0.0, int8_us=0.0)},
            fp32_measured_us=10.0,
            int8_measured_us=10.0,
        )


def test_lookup_is_profiled_in_the_benchmark_optimization_regime():
    """Amendment A #1: the lookup must not profile with ORT_DISABLE_ALL."""
    import onnxruntime as ort

    assert (
        sweep_runner.BENCHMARK_OPTIMIZATION_LEVEL
        is ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )


def test_empty_fp32_list_has_zero_seams_and_stays_all_int8(tmp_path, monkeypatch):
    """Spec #5: the pure-int8 config has no seams and no fp32 conv."""
    model_path = _tiny_conv_model(tmp_path)
    rows, _ = _sweep(model_path, tmp_path, monkeypatch, target=4)

    pure = next(row for row in rows if row["fp32_nodes"] == [])
    assert pure["status"] == "ok"
    assert pure["seam_count"] == 0

    table = sweep_runner.build_lookup_table(
        str(model_path),
        _calibration(),
        quantizer=Quantizer(str(tmp_path / "lookup_models")),
        work_dir=tmp_path / "lookup_work",
        warmup_runs=1,
        benchmark_runs=3,
        isolated=False,
    )
    assert table.layers
    assert table.regime in {"optimized", "calibrated"}
    assert table.int8_boundary  # baseline boundary table is available


def test_decorrelated_sampler_spans_placement_independently_of_seams():
    """P2.7 §3: sample many-seam/low-res and few-seam/high-res configs."""
    layers = [f"L{i}" for i in range(12)]
    sizes = {name: 100_000 for name in layers}
    sizes["L0"] = 1_000_000  # the high-resolution head
    sizes["L1"] = 900_000

    first = sweep_runner.generate_decorrelated_configs(layers, sizes, seed=0, target=40)
    second = sweep_runner.generate_decorrelated_configs(layers, sizes, seed=0, target=40)
    assert [(c.config_id, c.fp32_nodes) for c in first] == [
        (c.config_id, c.fp32_nodes) for c in second
    ]
    assert first[0].fp32_nodes == ()

    subsets = [set(config.fp32_nodes) for config in first]
    assert set() in subsets and set(layers) in subsets
    # A small, high-resolution-only fp32 island (the conv0 effect).
    assert {"L0"} in subsets
    # Many low-resolution layers kept fp32: seams without high-res placement.
    low_only = [
        subset
        for subset in subsets
        if subset and not (subset & {"L0", "L1"}) and len(subset) >= 6
    ]
    assert low_only
