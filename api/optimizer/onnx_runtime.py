import json
import time

import numpy as np
import onnxruntime as rt
import psutil

try:
    from api.analyzer.model_analyzer import is_file_valid
except ImportError:
    from analyzer.model_analyzer import is_file_valid

PROVIDER = "CPUExecutionProvider"

# ---------------------------------------------------------------------------
# TVM benchmark
# ---------------------------------------------------------------------------

def benchmark_tvm(
    filename: str,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
) -> str:
    """Benchmark a TVM compiled module (.tar) and return a JSON report.

    The report schema is identical to the ONNX benchmark() output so the
    ranking logic in main.py works without any changes.

    Args:
        filename       : Path to the TVM exported .tar module.
        warmup_runs    : Number of ignored warm-up inferences.
        benchmark_runs : Number of measured inferences.

    Returns:
        str : JSON string with latency and memory statistics.
    """
    from api.optimizer.tvm_runtime import load_input_specs, load_vm, run_vm

    try:
        # 1. Load the compiled Relax VM module
        specs = load_input_specs(filename)
        order = [spec["name"] for spec in specs]
        vm, dev = load_vm(filename)

        # 2. Build a random input matching the compiled shapes
        inputs = {}
        for spec in specs:
            shape = [s if s > 0 else 1 for s in spec["shape"]]
            inputs[spec["name"]] = np.random.rand(*shape).astype(np.float32)

        def _run():
            run_vm(vm, dev, inputs, order)

        # 3. Warm-up
        for _ in range(warmup_runs):
            _run()

        # 4. Measure latency and memory
        latencies = []
        process = psutil.Process()
        memory_before = process.memory_info().rss

        for _ in range(benchmark_runs):
            start = time.perf_counter()
            _run()
            latencies.append((time.perf_counter() - start) * 1000)

        memory_after = process.memory_info().rss

        report = {
            "model_path": filename,
            "provider": "TVM/CPUDevice",
            "warmup_runs": warmup_runs,
            "benchmark_runs": benchmark_runs,
            "latency_ms": {
                "avg": float(np.mean(latencies)),
                "p50": float(np.percentile(latencies, 50)),
                "p95": float(np.percentile(latencies, 95)),
                "p99": float(np.percentile(latencies, 99)),
            },
            "memory_mb": {
                "before": memory_before / (1024 * 1024),
                "after":  memory_after  / (1024 * 1024),
                "delta":  (memory_after - memory_before) / (1024 * 1024),
            },
            "success": True,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        report = {
            "model_path": filename,
            "provider": "TVM/CPUDevice",
            "warmup_runs": warmup_runs,
            "benchmark_runs": benchmark_runs,
            "latency_ms": {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0},
            "memory_mb": {"before": 0.0, "after": 0.0, "delta": 0.0},
            "success": False,
            "error": str(exc),
        }

    return json.dumps(report, indent=2)


def inference_session(filename: str) -> rt.InferenceSession:
    """Create an inference session for the given ONNX model.

    Args:
      filename : Path to the .onnx file

    Output:
      onnxruntime.InferenceSession : Ready-to-use session
    """
    if not is_file_valid(filename):
        raise ValueError(
            "The model is not in onnx format. Please convert it into onnx format"
        )
    return rt.InferenceSession(filename, providers=[PROVIDER])


def make_input(session: rt.InferenceSession, batch_size: int = 1, sample_input: np.ndarray | None = None) -> dict:
    """Build input matching the session's declared shapes.

    If sample_input is provided, use it (tiled to batch_size).
    Else, build random float32 inputs.
    """
    inputs = {}
    for i in session.get_inputs():
        shape = list(i.shape)
        if shape[0] == "N" or shape[0] is None or shape[0] == -1:
            shape[0] = batch_size

        if sample_input is not None:
            # Validate shape/dtype (simple check)
            if sample_input.dtype != np.float32:
                sample_input = sample_input.astype(np.float32)
            # Tile to batch size if needed
            if shape[0] > 1:
                val = np.tile(sample_input, (shape[0], 1, 1, 1))
            else:
                val = sample_input
            inputs[i.name] = val
        else:
            inputs[i.name] = np.random.rand(*shape).astype(np.float32)
    return inputs


def run_inference(
    session: rt.InferenceSession, inputs: dict
) -> list[np.ndarray]:
    """Run a single inference and return the raw outputs.

    Args:
      session : The loaded inference session
      inputs : Mapping of input name -> numpy array

    Output:
      list : Raw outputs returned by the session
    """
    label_name = session.get_outputs()[0].name
    return session.run([label_name], inputs)


try:
    from api.optimizer.benchmark_engine import benchmark as engine_benchmark
except ImportError:
    from optimizer.benchmark_engine import benchmark as engine_benchmark


def benchmark(
    filename: str,
    input_data: dict | None = None,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    hardware_id: str = "x86_64",
    batch_size: int = 1,
    sample_path: str | None = None,
    execution_provider: str = "auto",
    time_budget_seconds: float | None = None,
) -> str:
    """Benchmark an ONNX or TVM model."""
    if input_data is None and sample_path and filename.endswith(".onnx"):
        session = inference_session(filename)
        sample_input = np.load(sample_path, allow_pickle=False)
        input_data = make_input(
            session,
            batch_size=batch_size,
            sample_input=sample_input,
        )

    return engine_benchmark(
        filename=filename,
        input_data=input_data,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        hardware_id=hardware_id,
        batch_size=batch_size,
        execution_provider=execution_provider,
        time_budget_seconds=time_budget_seconds,
    )
