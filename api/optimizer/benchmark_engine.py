"""
Benchmark Engine for EdgeForge / EdgeMason.

Measures model performance metrics on target hardware configurations:
- Latency percentiles (avg, min, max, std, p25, p50, p75, p90, p95, p99)
- Throughput (Inferences / Frames per second)
- Model size (bytes, KB, MB)
- Memory usage (RSS before, after, delta, peak RSS)
- Hardware target profiling and scaling estimates
- Machine-readable JSON output
"""

import json
import os
import platform
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil

# Optional framework imports
try:
    import onnxruntime as rt
except ImportError:
    rt = None

try:
    import tvm
    from tvm.contrib import graph_executor
except ImportError:
    tvm = None
    graph_executor = None


PROFILES_PATH = Path(__file__).resolve().parent.parent.parent / "profiles" / "hardware_profiles.json"
MAX_INPUT_DIMENSION = 4096
MAX_INPUT_ELEMENTS = 10_000_000
EXECUTION_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}


def resolve_execution_provider(requested: str) -> str:
    """Resolve auto/cpu/cuda to an installed ONNX Runtime provider."""
    if requested not in {"auto", *EXECUTION_PROVIDERS}:
        raise ValueError("execution_provider must be auto, cpu, or cuda")
    available = set(rt.get_available_providers()) if rt is not None else set()
    resolved = (
        "CUDAExecutionProvider"
        if requested == "auto" and "CUDAExecutionProvider" in available
        else EXECUTION_PROVIDERS.get(requested, "CPUExecutionProvider")
    )
    if resolved not in available:
        raise ValueError(f"{resolved} is not available")
    return resolved


def _preload_cuda_libraries(provider: str) -> None:
    """Load pip-installed NVIDIA libraries when supported by ONNX Runtime."""
    preload = getattr(rt, "preload_dlls", None)
    if provider == "CUDAExecutionProvider" and preload is not None:
        preload(directory="")


def load_hardware_profiles() -> Dict[str, Any]:
    """Load hardware configuration profiles from disk or fallback defaults."""
    default_profiles = {
        "rpi3": {
            "name": "Raspberry Pi 3 Model B+",
            "architecture": "aarch64",
            "cpu": "Cortex-A53",
            "cores": 4,
            "clock_ghz": 1.4,
            "ram_mb": 1024,
            "latency_scaling_factor": 18.5,
            "memory_overhead_scaling": 1.2,
        },
        "rpi4": {
            "name": "Raspberry Pi 4 Model B",
            "architecture": "aarch64",
            "cpu": "Cortex-A72",
            "cores": 4,
            "clock_ghz": 1.8,
            "ram_mb": 4096,
            "latency_scaling_factor": 11.8,
            "memory_overhead_scaling": 1.1,
        },
        "rpi5": {
            "name": "Raspberry Pi 5",
            "architecture": "aarch64",
            "cpu": "Cortex-A76",
            "cores": 4,
            "clock_ghz": 2.4,
            "ram_mb": 8192,
            "latency_scaling_factor": 4.5,
            "memory_overhead_scaling": 1.05,
        },
        "jetson_nano": {
            "name": "NVIDIA Jetson Nano",
            "architecture": "aarch64",
            "cpu": "Cortex-A57",
            "cores": 4,
            "clock_ghz": 1.43,
            "ram_mb": 4096,
            "latency_scaling_factor": 8.2,
            "memory_overhead_scaling": 1.15,
        },
        "x86_64": {
            "name": "Standard x86_64 Host / Server",
            "architecture": "x86_64",
            "cpu": "x86_64 CPU",
            "cores": 8,
            "clock_ghz": 3.0,
            "ram_mb": 16384,
            "latency_scaling_factor": 1.0,
            "memory_overhead_scaling": 1.0,
        },
    }
    if PROFILES_PATH.exists():
        try:
            with open(PROFILES_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                default_profiles.update(loaded)
        except Exception:
            pass
    return default_profiles


@dataclass
class LatencyMetrics:
    avg: float
    min: float
    max: float
    std: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float


@dataclass
class ThroughputMetrics:
    fps: float
    total_inferences: int
    total_time_ms: float
    batch_size: int = 1


@dataclass
class MemoryMetrics:
    before_mb: float
    after_mb: float
    delta_mb: float
    peak_rss_mb: float
    peak_tracemalloc_mb: float


@dataclass
class ModelSizeMetrics:
    bytes: int
    kb: float
    mb: float


@dataclass
class BenchmarkResult:
    model_path: str
    framework: str
    target_hardware: str
    warmup_runs: int
    benchmark_runs: int
    batch_size: int
    provider: str
    latency_ms: Dict[str, float]
    throughput: Dict[str, Any]
    memory_mb: Dict[str, float]
    model_size: Dict[str, Any]
    system_info: Dict[str, Any]
    hardware_estimate: Optional[Dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class BenchmarkEngine:
    """
    Core Benchmarking Engine for Edge AI Models.
    Measures latency percentiles, throughput, memory usage, and model size.
    """

    def __init__(
        self,
        hardware_id: str = "x86_64",
        execution_provider: str = "auto",
    ):
        self.hardware_id = hardware_id
        self.execution_provider = execution_provider
        self.hardware_profiles = load_hardware_profiles()
        self.profile = self.hardware_profiles.get(hardware_id, self.hardware_profiles.get("x86_64", {}))

    def get_model_size(self, model_path: str) -> ModelSizeMetrics:
        """Calculate model size in bytes, KB, and MB."""
        size_bytes = os.path.getsize(model_path) if os.path.exists(model_path) else 0
        return ModelSizeMetrics(
            bytes=size_bytes,
            kb=round(size_bytes / 1024.0, 3),
            mb=round(size_bytes / (1024.0 * 1024.0), 3),
        )

    def get_system_info(self) -> Dict[str, Any]:
        """Capture current host platform telemetry."""
        mem = psutil.virtual_memory()
        return {
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "total_ram_mb": round(mem.total / (1024 * 1024), 2),
            "available_ram_mb": round(mem.available / (1024 * 1024), 2),
            "python_version": platform.python_version(),
        }

    def _generate_onnx_inputs(self, session: Any, batch_size: int = 1) -> Dict[str, np.ndarray]:
        """Generate randomized float32 tensors matching declared input shapes."""
        inputs = {}
        for inp in session.get_inputs():
            shape = [
                batch_size if (not isinstance(d, int) or d <= 0) else d
                for d in inp.shape
            ]
            if any(d > MAX_INPUT_DIMENSION for d in shape):
                raise ValueError("model input dimension is too large")
            if int(np.prod(shape, dtype=np.int64)) > MAX_INPUT_ELEMENTS:
                raise ValueError("model input tensor is too large")
            inputs[inp.name] = np.random.rand(*shape).astype(np.float32)
        return inputs

    def _benchmark_onnx(
        self,
        model_path: str,
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
        batch_size: int = 1,
        input_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> BenchmarkResult:
        """Benchmark an ONNX model using ONNX Runtime."""
        if rt is None:
            raise ImportError("onnxruntime is not installed in the environment.")

        model_size = self.get_model_size(model_path)
        system_info = self.get_system_info()
        provider = resolve_execution_provider(self.execution_provider)
        _preload_cuda_libraries(provider)

        # Session creation
        def create_session(selected_provider: str):
            options = rt.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            if selected_provider == "CUDAExecutionProvider":
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            return rt.InferenceSession(
                model_path,
                sess_options=options,
                providers=[selected_provider],
            )

        try:
            session = create_session(provider)
        except Exception:
            if self.execution_provider != "auto" or provider != "CUDAExecutionProvider":
                raise
            provider = "CPUExecutionProvider"
            session = create_session(provider)
        if (
            provider == "CUDAExecutionProvider"
            and session.get_providers()[0] != "CUDAExecutionProvider"
        ):
            if self.execution_provider == "cuda":
                raise RuntimeError("CUDAExecutionProvider failed to initialize")
            provider = "CPUExecutionProvider"
            if provider not in session.get_providers():
                session = create_session(provider)
        for model_input in session.get_inputs():
            shape = [
                batch_size if (not isinstance(d, int) or d <= 0) else d
                for d in model_input.shape
            ]
            if any(d > MAX_INPUT_DIMENSION for d in shape):
                raise ValueError("model input dimension is too large")
            if int(np.prod(shape, dtype=np.int64)) > MAX_INPUT_ELEMENTS:
                raise ValueError("model input tensor is too large")
        inputs = input_data if input_data is not None else self._generate_onnx_inputs(session, batch_size)
        output_name = session.get_outputs()[0].name

        # Warmup phase
        for _ in range(warmup_runs):
            session.run([output_name], inputs)

        # Tracemalloc & psutil for memory tracking
        tracemalloc.start()
        process = psutil.Process()
        mem_before = process.memory_info().rss

        latencies_ms: List[float] = []
        start_total = time.perf_counter()

        for _ in range(benchmark_runs):
            t0 = time.perf_counter()
            session.run([output_name], inputs)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        total_time_sec = time.perf_counter() - start_total
        mem_after = process.memory_info().rss
        _, peak_tracemalloc = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        lat_arr = np.array(latencies_ms)
        avg_lat = float(np.mean(lat_arr))
        total_time_ms = total_time_sec * 1000.0
        fps = (benchmark_runs * batch_size) / total_time_sec if total_time_sec > 0 else 0.0

        latency_metrics = LatencyMetrics(
            avg=round(avg_lat, 4),
            min=round(float(np.min(lat_arr)), 4),
            max=round(float(np.max(lat_arr)), 4),
            std=round(float(np.std(lat_arr)), 4),
            p25=round(float(np.percentile(lat_arr, 25)), 4),
            p50=round(float(np.percentile(lat_arr, 50)), 4),
            p75=round(float(np.percentile(lat_arr, 75)), 4),
            p90=round(float(np.percentile(lat_arr, 90)), 4),
            p95=round(float(np.percentile(lat_arr, 95)), 4),
            p99=round(float(np.percentile(lat_arr, 99)), 4),
        )

        throughput_metrics = ThroughputMetrics(
            fps=round(fps, 2),
            total_inferences=benchmark_runs,
            total_time_ms=round(total_time_ms, 2),
            batch_size=batch_size,
        )

        memory_metrics = MemoryMetrics(
            before_mb=round(mem_before / (1024.0 * 1024.0), 3),
            after_mb=round(mem_after / (1024.0 * 1024.0), 3),
            delta_mb=round((mem_after - mem_before) / (1024.0 * 1024.0), 3),
            peak_rss_mb=round(mem_after / (1024.0 * 1024.0), 3),
            peak_tracemalloc_mb=round(peak_tracemalloc / (1024.0 * 1024.0), 3),
        )

        # Scaling estimation for target hardware
        scaling_factor = self.profile.get("latency_scaling_factor", 1.0)
        hardware_estimate = None
        if self.hardware_id != "x86_64":
            est_p50 = round(latency_metrics.p50 * scaling_factor, 2)
            est_avg = round(latency_metrics.avg * scaling_factor, 2)
            est_fps = round(fps / scaling_factor, 2) if scaling_factor > 0 else fps
            hardware_estimate = {
                "target_hardware": self.hardware_id,
                "target_name": self.profile.get("name", self.hardware_id),
                "scaling_factor": scaling_factor,
                "estimated_latency_p50_ms": est_p50,
                "estimated_latency_avg_ms": est_avg,
                "estimated_throughput_fps": est_fps,
            }

        return BenchmarkResult(
            model_path=model_path,
            framework="onnxruntime",
            target_hardware=self.hardware_id,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
            batch_size=batch_size,
            provider=provider,
            latency_ms=asdict(latency_metrics),
            throughput=asdict(throughput_metrics),
            memory_mb=asdict(memory_metrics),
            model_size=asdict(model_size),
            system_info=system_info,
            hardware_estimate=hardware_estimate,
            success=True,
            error=None,
        )

    def _benchmark_tvm(
        self,
        model_path: str,
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
        batch_size: int = 1,
    ) -> BenchmarkResult:
        """Benchmark a compiled TVM Relax VM module (.tar / .so)."""
        from api.optimizer.tvm_runtime import load_input_specs, load_vm, run_vm

        model_size = self.get_model_size(model_path)
        system_info = self.get_system_info()

        specs = load_input_specs(model_path)
        vm, dev = load_vm(model_path)

        inputs = {}
        for spec in specs:
            shape = [
                batch_size if (not isinstance(s, int) or s <= 0) else s
                for s in spec["shape"]
            ]
            inputs[spec["name"]] = np.random.rand(*shape).astype(np.float32)
        order = [spec["name"] for spec in specs]

        def _run():
            run_vm(vm, dev, inputs, order)

        for _ in range(warmup_runs):
            _run()

        tracemalloc.start()
        process = psutil.Process()
        mem_before = process.memory_info().rss

        latencies_ms: List[float] = []
        start_total = time.perf_counter()

        for _ in range(benchmark_runs):
            t0 = time.perf_counter()
            _run()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        total_time_sec = time.perf_counter() - start_total
        mem_after = process.memory_info().rss
        _, peak_tracemalloc = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        lat_arr = np.array(latencies_ms)
        avg_lat = float(np.mean(lat_arr))
        total_time_ms = total_time_sec * 1000.0
        fps = (benchmark_runs * batch_size) / total_time_sec if total_time_sec > 0 else 0.0

        latency_metrics = LatencyMetrics(
            avg=round(avg_lat, 4),
            min=round(float(np.min(lat_arr)), 4),
            max=round(float(np.max(lat_arr)), 4),
            std=round(float(np.std(lat_arr)), 4),
            p25=round(float(np.percentile(lat_arr, 25)), 4),
            p50=round(float(np.percentile(lat_arr, 50)), 4),
            p75=round(float(np.percentile(lat_arr, 75)), 4),
            p90=round(float(np.percentile(lat_arr, 90)), 4),
            p95=round(float(np.percentile(lat_arr, 95)), 4),
            p99=round(float(np.percentile(lat_arr, 99)), 4),
        )

        throughput_metrics = ThroughputMetrics(
            fps=round(fps, 2),
            total_inferences=benchmark_runs,
            total_time_ms=round(total_time_ms, 2),
            batch_size=batch_size,
        )

        memory_metrics = MemoryMetrics(
            before_mb=round(mem_before / (1024.0 * 1024.0), 3),
            after_mb=round(mem_after / (1024.0 * 1024.0), 3),
            delta_mb=round((mem_after - mem_before) / (1024.0 * 1024.0), 3),
            peak_rss_mb=round(mem_after / (1024.0 * 1024.0), 3),
            peak_tracemalloc_mb=round(peak_tracemalloc / (1024.0 * 1024.0), 3),
        )

        scaling_factor = self.profile.get("latency_scaling_factor", 1.0)
        hardware_estimate = None
        if self.hardware_id != "x86_64":
            est_p50 = round(latency_metrics.p50 * scaling_factor, 2)
            est_avg = round(latency_metrics.avg * scaling_factor, 2)
            est_fps = round(fps / scaling_factor, 2) if scaling_factor > 0 else fps
            hardware_estimate = {
                "target_hardware": self.hardware_id,
                "target_name": self.profile.get("name", self.hardware_id),
                "scaling_factor": scaling_factor,
                "estimated_latency_p50_ms": est_p50,
                "estimated_latency_avg_ms": est_avg,
                "estimated_throughput_fps": est_fps,
            }

        return BenchmarkResult(
            model_path=model_path,
            framework="tvm",
            target_hardware=self.hardware_id,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
            batch_size=batch_size,
            provider="TVM/CPUDevice",
            latency_ms=asdict(latency_metrics),
            throughput=asdict(throughput_metrics),
            memory_mb=asdict(memory_metrics),
            model_size=asdict(model_size),
            system_info=system_info,
            hardware_estimate=hardware_estimate,
            success=True,
            error=None,
        )

    def benchmark(
        self,
        model_path: str,
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
        batch_size: int = 1,
        input_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> BenchmarkResult:
        """
        Universal benchmark dispatcher:
        Automatically selects TVM (.tar/.so) or ONNX Runtime (.onnx).
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            if model_path.endswith(".tar") or model_path.endswith(".so"):
                return self._benchmark_tvm(
                    model_path=model_path,
                    warmup_runs=warmup_runs,
                    benchmark_runs=benchmark_runs,
                    batch_size=batch_size,
                )
            else:
                return self._benchmark_onnx(
                    model_path=model_path,
                    warmup_runs=warmup_runs,
                    benchmark_runs=benchmark_runs,
                    batch_size=batch_size,
                    input_data=input_data,
                )
        except Exception as exc:
            model_size = self.get_model_size(model_path)
            system_info = self.get_system_info()
            return BenchmarkResult(
                model_path=model_path,
                framework="unknown",
                target_hardware=self.hardware_id,
                warmup_runs=warmup_runs,
                benchmark_runs=benchmark_runs,
                batch_size=batch_size,
                provider=(
                    EXECUTION_PROVIDERS.get(
                        self.execution_provider,
                        "CUDAExecutionProvider"
                        if self.execution_provider == "auto"
                        and rt is not None
                        and "CUDAExecutionProvider" in rt.get_available_providers()
                        else "CPUExecutionProvider",
                    )
                ),
                latency_ms={"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0},
                throughput={"fps": 0.0, "total_inferences": 0, "total_time_ms": 0.0, "batch_size": batch_size},
                memory_mb={"before_mb": 0.0, "after_mb": 0.0, "delta_mb": 0.0, "peak_rss_mb": 0.0, "peak_tracemalloc_mb": 0.0},
                model_size=asdict(model_size),
                system_info=system_info,
                hardware_estimate=None,
                success=False,
                error=str(exc),
            )


def benchmark(
    filename: str,
    input_data: Optional[Dict[str, np.ndarray]] = None,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    hardware_id: str = "x86_64",
    batch_size: int = 1,
    execution_provider: str = "auto",
) -> str:
    """
    Standard functional interface returning machine-readable JSON string.
    Backwards-compatible drop-in replacement.
    """
    engine = BenchmarkEngine(
        hardware_id=hardware_id,
        execution_provider=execution_provider,
    )
    res = engine.benchmark(
        model_path=filename,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        batch_size=batch_size,
        input_data=input_data,
    )
    data = res.to_dict()
    # Provide backward-compatibility key mapping
    data["memory_mb"] = {
        "before": data["memory_mb"]["before_mb"],
        "after": data["memory_mb"]["after_mb"],
        "delta": data["memory_mb"]["delta_mb"],
        "peak_rss_mb": data["memory_mb"]["peak_rss_mb"],
        "peak_tracemalloc_mb": data["memory_mb"]["peak_tracemalloc_mb"],
    }
    return json.dumps(data, indent=2)
