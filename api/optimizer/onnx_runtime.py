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
    try:
        import tvm
        from tvm.contrib import graph_executor
    except ImportError as exc:
        raise ImportError(
            "Apache TVM is not installed. Run: pip install apache-tvm"
        ) from exc

    try:
        # 1. Load the compiled module
        lib = tvm.runtime.load_module(filename)
        dev = tvm.cpu(0)
        module = graph_executor.GraphModule(lib["default"](dev))

        # 2. Build a random input matching the module's declared shapes
        #    get_input_info() returns (names, shapes, dtypes)
        input_info = module.get_input_info()
        inputs = {}
        for i, name in enumerate(input_info[0]):
            shape = input_info[1][i]
            shape = [s if s > 0 else 1 for s in shape]  # replace dynamic dims
            inputs[name] = np.random.rand(*shape).astype(np.float32)

        def _run():
            for name, arr in inputs.items():
                module.set_input(name, tvm.nd.array(arr, dev))
            module.run()

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


def make_input(session: rt.InferenceSession, batch_size: int = 1) -> dict:
    """Build random float32 inputs matching the session's declared shapes.

    Args:
      session : The loaded inference session
      batch_size : Value used for dynamic/unknown dimensions

    Output:
      dict : Mapping of input name -> numpy array
    """
    inputs = {}
    for inp in session.get_inputs():
        dims = [
            batch_size if not isinstance(d, int) or d <= 0 else d
            for d in inp.shape
        ]
        inputs[inp.name] = np.random.rand(*dims).astype(np.float32)
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


def benchmark(
    filename: str,
    input_data: dict | None = None,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
) -> str:
    """Benchmark a model and return a JSON report.

    Dispatches to the TVM benchmark path for .tar files and the ONNX Runtime
    path for .onnx files so the rest of the pipeline needs no changes.

    Args:
        filename       : Path to the .onnx or .tar model file.
        input_data     : Optional dict of inputs (ONNX path only).
        warmup_runs    : Number of ignored warm-up inferences.
        benchmark_runs : Number of measured inferences.

    Returns:
        str : JSON string with latency and memory statistics.
    """
    # ---------- Dispatch by file type ----------
    if filename.endswith(".tar"):
        return benchmark_tvm(filename, warmup_runs=warmup_runs, benchmark_runs=benchmark_runs)

    # ---------- ONNX Runtime path ----------
    try:
        session = inference_session(filename)
        inputs = input_data if input_data is not None else make_input(session)

        for _ in range(warmup_runs):
            run_inference(session, inputs)

        latencies = []
        process = psutil.Process()
        memory_before = process.memory_info().rss

        for _ in range(benchmark_runs):
            start = time.perf_counter()
            run_inference(session, inputs)
            latencies.append((time.perf_counter() - start) * 1000)

        memory_after = process.memory_info().rss

        report = {
            "model_path": filename,
            "provider": PROVIDER,
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
                "after": memory_after / (1024 * 1024),
                "delta": (memory_after - memory_before) / (1024 * 1024),
            },
            "success": True,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        report = {
            "model_path": filename,
            "provider": PROVIDER,
            "warmup_runs": warmup_runs,
            "benchmark_runs": benchmark_runs,
            "latency_ms": {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0},
            "memory_mb": {"before": 0.0, "after": 0.0, "delta": 0.0},
            "success": False,
            "error": str(exc),
        }

    return json.dumps(report, indent=2)
