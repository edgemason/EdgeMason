import json
import os
import tempfile
import numpy as np
import onnx
from onnx import helper, TensorProto
import pytest

from api.optimizer.benchmark_engine import BenchmarkEngine, benchmark, load_hardware_profiles


def create_dummy_onnx_model(filepath: str):
    """Create a minimal ONNX model for testing."""
    # Create simple node: Y = Relu(X)
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])
    node = helper.make_node("Relu", ["input"], ["output"])

    graph = helper.make_graph([node], "dummy_graph", [X], [Y])
    model = helper.make_model(graph, producer_name="test_engine")
    onnx.save(model, filepath)
    return filepath


def test_hardware_profiles_loaded():
    """Test that hardware target profiles are properly loaded."""
    profiles = load_hardware_profiles()
    assert "rpi4" in profiles
    assert "rpi5" in profiles
    assert "x86_64" in profiles
    assert profiles["rpi4"]["latency_scaling_factor"] > 1.0


def test_benchmark_engine_runs_and_measures_metrics():
    """
    Acceptance Criteria Tests:
    1. Runs repeatable benchmarks with warmup + measured iterations
    2. Captures latency percentiles and throughput
    3. Captures model size and memory usage
    4. Outputs machine-readable results (JSON)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.onnx")
        create_dummy_onnx_model(model_path)

        engine = BenchmarkEngine(hardware_id="rpi4")
        result = engine.benchmark(
            model_path=model_path,
            warmup_runs=5,
            benchmark_runs=20,
            batch_size=1,
        )

        # 1. Success check
        assert result.success is True, f"Benchmark failed with error: {result.error}"

        # 2. Latency percentiles check
        lat = result.latency_ms
        assert "avg" in lat and "min" in lat and "max" in lat and "std" in lat
        assert "p25" in lat and "p50" in lat and "p75" in lat and "p90" in lat and "p95" in lat and "p99" in lat
        assert lat["min"] <= lat["p50"] <= lat["max"]
        assert lat["p50"] > 0

        # 3. Throughput check
        tp = result.throughput
        assert "fps" in tp and "total_time_ms" in tp
        assert tp["fps"] > 0
        assert tp["total_inferences"] == 20

        # 4. Model size check
        ms = result.model_size
        assert "bytes" in ms and "kb" in ms and "mb" in ms
        assert ms["bytes"] > 0
        assert ms["kb"] > 0

        # 5. Memory metrics check
        mem = result.memory_mb
        assert "before_mb" in mem and "after_mb" in mem and "delta_mb" in mem and "peak_rss_mb" in mem
        assert mem["before_mb"] > 0

        # 6. Target hardware scaling estimate check
        assert result.hardware_estimate is not None
        assert result.hardware_estimate["target_hardware"] == "rpi4"
        assert result.hardware_estimate["estimated_latency_p50_ms"] > lat["p50"]
        assert result.hardware_estimate["estimated_throughput_fps"] < tp["fps"]

        # 7. Machine-readable JSON output check
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["framework"] == "onnxruntime"
        assert parsed["target_hardware"] == "rpi4"


def test_functional_benchmark_interface():
    """Test the drop-in functional benchmark() wrapper."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.onnx")
        create_dummy_onnx_model(model_path)

        json_out = benchmark(
            filename=model_path,
            warmup_runs=3,
            benchmark_runs=10,
            hardware_id="x86_64",
        )
        parsed = json.loads(json_out)

        assert parsed["success"] is True
        assert "latency_ms" in parsed
        assert "throughput" in parsed
        assert "memory_mb" in parsed
        assert "model_size" in parsed
        assert parsed["provider"] == "CPUExecutionProvider"


def test_batch_size_throughput_scaling():
    """Test throughput calculation with batch_size > 1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.onnx")
        create_dummy_onnx_model(model_path)

        engine = BenchmarkEngine(hardware_id="x86_64")
        result = engine.benchmark(
            model_path=model_path,
            warmup_runs=2,
            benchmark_runs=10,
            batch_size=4,
        )

        assert result.success is True
        assert result.throughput["batch_size"] == 4
        assert result.throughput["fps"] > 0


def test_missing_model_file_error_handling():
    """Test error handling when a non-existent file path is provided."""
    engine = BenchmarkEngine()
    result = engine.benchmark("non_existent_model.onnx")
    assert result.success is False
    assert "not found" in result.error.lower()


def test_hardware_profiles_scaling_differences():
    """Test that different edge targets apply their respective scaling factors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.onnx")
        create_dummy_onnx_model(model_path)

        rpi3_res = BenchmarkEngine(hardware_id="rpi3").benchmark(model_path, warmup_runs=2, benchmark_runs=5)
        rpi5_res = BenchmarkEngine(hardware_id="rpi5").benchmark(model_path, warmup_runs=2, benchmark_runs=5)

        assert rpi3_res.hardware_estimate["scaling_factor"] > rpi5_res.hardware_estimate["scaling_factor"]
        assert rpi3_res.hardware_estimate["estimated_latency_p50_ms"] > rpi5_res.hardware_estimate["estimated_latency_p50_ms"]

