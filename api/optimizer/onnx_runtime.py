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
    """Benchmark an ONNX model and return a JSON report.

    Args:
      filename : Path to the .onnx file
      input_data : Optional dict of inputs; random ones are generated if omitted
      warmup_runs : Number of ignored warmup inferences
      benchmark_runs : Number of measured inferences

    Output:
      str : JSON string with latency and memory statistics
    """
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
