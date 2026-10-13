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
        assert set(mem) == {"baseline_mb", "peak_mb", "delta_mb"}
        assert mem["baseline_mb"] > 0
        assert mem["peak_mb"] >= mem["baseline_mb"]
        assert mem["delta_mb"] == pytest.approx(mem["peak_mb"] - mem["baseline_mb"], abs=0.01)

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
            execution_provider="cpu",
        )
        parsed = json.loads(json_out)

        assert parsed["success"] is True
        assert "latency_ms" in parsed
        assert "throughput" in parsed
        assert "memory_mb" in parsed
        assert "model_size" in parsed
        assert parsed["provider"] == "CPUExecutionProvider"


def test_cuda_benchmark_fails_clearly_when_provider_is_unavailable(
    tiny_onnx,
    monkeypatch,
):
    from api.optimizer import benchmark_engine

    monkeypatch.setattr(
        benchmark_engine.rt,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )

    parsed = json.loads(
        benchmark(
            filename=str(tiny_onnx),
            warmup_runs=1,
            benchmark_runs=1,
            execution_provider="cuda",
            isolated=False,
        )
    )

    assert parsed["success"] is False
    assert parsed["provider"] == "CUDAExecutionProvider"
    assert parsed["error"] == "CUDAExecutionProvider is not available"


def test_auto_benchmark_reports_cpu_when_cuda_cannot_run_model(
    tiny_onnx,
    monkeypatch,
):
    from api.optimizer import benchmark_engine

    real_session = benchmark_engine.rt.InferenceSession
    requested_providers = []

    def session_with_unsupported_cuda(*args, providers, **kwargs):
        requested_providers.append(providers)
        if providers == ["CUDAExecutionProvider"]:
            raise RuntimeError("model has operators assigned to CPU")
        return real_session(*args, providers=providers, **kwargs)

    monkeypatch.setattr(
        benchmark_engine.rt,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        benchmark_engine.rt,
        "InferenceSession",
        session_with_unsupported_cuda,
    )

    parsed = json.loads(
        benchmark(
            filename=str(tiny_onnx),
            warmup_runs=1,
            benchmark_runs=1,
            execution_provider="auto",
            isolated=False,
        )
    )

    assert parsed["success"] is True
    assert parsed["provider"] == "CPUExecutionProvider"
    assert requested_providers == [
        ["CUDAExecutionProvider"],
        ["CPUExecutionProvider"],
    ]


def test_cuda_benchmark_rejects_runtime_provider_fallback(tiny_onnx, monkeypatch):
    from api.optimizer import benchmark_engine

    class CpuFallbackSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(
        benchmark_engine.rt,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        benchmark_engine.rt,
        "InferenceSession",
        lambda *args, **kwargs: CpuFallbackSession(),
    )

    parsed = json.loads(
        benchmark(
            filename=str(tiny_onnx),
            warmup_runs=1,
            benchmark_runs=1,
            execution_provider="cuda",
            isolated=False,
        )
    )

    assert parsed["success"] is False
    assert parsed["provider"] == "CUDAExecutionProvider"
    assert parsed["error"] == "CUDAExecutionProvider failed to initialize"


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


def test_benchmark_rejects_oversized_input_dimensions(tmp_path):
    model_path = tmp_path / "oversized.onnx"
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 5000]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 5000]
    )
    graph = helper.make_graph(
        [helper.make_node("Relu", ["input"], ["output"])],
        "oversized",
        [model_input],
        [model_output],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, model_path)

    result = BenchmarkEngine().benchmark(
        str(model_path),
        warmup_runs=1,
        benchmark_runs=1,
    )

    assert result.success is False
    assert "dimension is too large" in result.error


def test_hardware_profiles_scaling_differences():
    """Test that different edge targets apply their respective scaling factors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.onnx")
        create_dummy_onnx_model(model_path)

        rpi3_res = BenchmarkEngine(hardware_id="rpi3").benchmark(model_path, warmup_runs=2, benchmark_runs=5)
        rpi5_res = BenchmarkEngine(hardware_id="rpi5").benchmark(model_path, warmup_runs=2, benchmark_runs=5)

        assert rpi3_res.hardware_estimate["scaling_factor"] > rpi5_res.hardware_estimate["scaling_factor"]
        assert rpi3_res.hardware_estimate["estimated_latency_p50_ms"] > rpi5_res.hardware_estimate["estimated_latency_p50_ms"]


def _matmul_model(path, columns):
    weights = np.random.rand(1000, columns).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["X", "W"], ["Y"])],
        "matmul",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1000])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, columns])],
        initializer=[
            helper.make_tensor("W", TensorProto.FLOAT, weights.shape, weights.tobytes(), raw=True)
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    return str(path)


def test_isolated_benchmark_measures_peak_memory_from_before_model_load(tmp_path):
    small = _matmul_model(tmp_path / "small.onnx", 1_000)   # ~4 MB of weights
    big = _matmul_model(tmp_path / "big.onnx", 10_000)      # ~40 MB of weights

    def memory(model_path):
        parsed = json.loads(
            benchmark(model_path, warmup_runs=1, benchmark_runs=3, execution_provider="cpu")
        )
        assert parsed["success"] is True, parsed["error"]
        return parsed["memory_mb"]

    small_memory, big_memory = memory(small), memory(big)

    assert set(big_memory) == {"baseline", "peak", "delta"}
    # Loading the weights happens after the baseline, so it is counted.
    assert big_memory["delta"] >= 30
    assert big_memory["delta"] > small_memory["delta"] + 20


def test_isolated_benchmark_reports_timeout(tiny_onnx):
    parsed = json.loads(
        benchmark(str(tiny_onnx), warmup_runs=1, benchmark_runs=1, timeout_seconds=0.01)
    )

    assert parsed["success"] is False
    assert "timed out" in parsed["error"]


def test_without_budget_runs_exact_counts(tiny_onnx):
    result = BenchmarkEngine().benchmark(str(tiny_onnx), warmup_runs=3, benchmark_runs=25)

    assert result.success is True
    assert result.warmup_runs == 3
    assert result.benchmark_runs == 25
    assert result.throughput["total_inferences"] == 25


def test_time_budget_stops_early_but_keeps_minimum_runs(tiny_onnx, monkeypatch):
    from api.optimizer import benchmark_engine

    clock = iter(range(0, 10_000))  # every perf_counter call advances 1 "second"
    monkeypatch.setattr(benchmark_engine.time, "perf_counter", lambda: float(next(clock)))

    engine = BenchmarkEngine(time_budget_seconds=4.0)
    result = engine.benchmark(str(tiny_onnx), warmup_runs=10, benchmark_runs=100)

    assert result.success is True
    # Warm-up may use 25% of the budget (1 s), so it stops after 1 run.
    assert result.warmup_runs == 1
    # Budget is used up quickly, but the minimum of 10 timed runs is kept.
    assert result.benchmark_runs == benchmark_engine.MIN_TIMED_RUNS
    assert len(result.latency_ms) > 0


def test_time_budget_allows_all_runs_for_fast_models(tiny_onnx):
    result = BenchmarkEngine(time_budget_seconds=30.0).benchmark(
        str(tiny_onnx), warmup_runs=10, benchmark_runs=100
    )

    assert result.warmup_runs == 10
    assert result.benchmark_runs == 100


def test_failed_benchmark_reports_zero_runs():
    result = BenchmarkEngine().benchmark("non_existent_model.onnx", warmup_runs=10, benchmark_runs=100)

    assert result.success is False
    assert result.warmup_runs == 0
    assert result.benchmark_runs == 0
