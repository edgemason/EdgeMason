"""Measure this ONNX model on this device and write measured.json.

    python bench.py                       # defaults: model.onnx -> measured.json
    python bench.py --threads 4 --runs 200

Uses the same protocol as the EdgeMason server benchmark (warm-up share, timed
runs, minimum timed runs, time budget), so the numbers can be compared with the
server's estimate. Needs only onnxruntime, numpy and the standard library.
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # reported as a failed run below
    ort = None

try:
    import resource
except ImportError:  # not available on Windows
    resource = None

HERE = Path(__file__).resolve().parent

# Protocol constants. They must equal the server's values in benchmark_engine.py.
MIN_TIMED_RUNS = 10
WARMUP_BUDGET_SHARE = 0.25

# Module-level so tests can point them at other files.
THERMAL_PATH = "/sys/class/thermal/thermal_zone0/temp"
GOVERNOR_PATH = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
CLEAR_REFS_PATH = "/proc/self/clear_refs"
STATUS_PATH = "/proc/self/status"

MB = 1024.0 * 1024.0


# ---------------------------------------------------------------------------
# Small readers for files that may not exist
# ---------------------------------------------------------------------------

def _read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _status_kb(key):
    """Value in kB of a /proc/self/status line such as 'VmHWM:', or None."""
    try:
        with open(STATUS_PATH, encoding="ascii") as status:
            for line in status:
                if line.startswith(key):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def _ru_maxrss_bytes():
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _read_temp_c():
    raw = _read_text(THERMAL_PATH)
    if raw is None:
        return None
    try:
        return round(int(raw) / 1000.0, 1)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Inputs and timing
# ---------------------------------------------------------------------------

def build_inputs(shapes, seed):
    """One seeded float32 array per model input. Dynamic dims become 1."""
    rng = np.random.default_rng(seed)
    inputs = {}
    for name, shape in shapes.items():
        concrete = [d if isinstance(d, int) and d > 0 else 1 for d in shape]
        inputs[name] = rng.random(concrete, dtype=np.float32)
    return inputs


def make_session(model_path, threads, profile_prefix=None):
    if ort is None:
        raise RuntimeError("onnxruntime not installed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = profile_prefix
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def run_timed(run, warmup_runs, benchmark_runs, budget):
    """Warm up, then time each run. Returns (warm-ups done, latencies in ms, seconds).

    Same protocol as BenchmarkEngine._run_timed: warm-up stops after its share of
    the budget; timing stops when the budget is used, but never before
    min(MIN_TIMED_RUNS, benchmark_runs) timed runs.
    """
    warmups_done = 0
    start_warmup = time.perf_counter()
    while warmups_done < warmup_runs:
        run()
        warmups_done += 1
        if time.perf_counter() - start_warmup >= budget * WARMUP_BUDGET_SHARE:
            break

    min_runs = min(MIN_TIMED_RUNS, benchmark_runs)
    latencies_ms = []
    start_total = time.perf_counter()
    while len(latencies_ms) < benchmark_runs:
        t0 = time.perf_counter()
        run()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if len(latencies_ms) >= min_runs and time.perf_counter() - start_total >= budget:
            break
    return warmups_done, latencies_ms, time.perf_counter() - start_total


# ---------------------------------------------------------------------------
# Memory: peak RSS for the session and its runs
# ---------------------------------------------------------------------------

class _Memory:
    """Baseline is read before the session is created; peak is read afterwards."""

    def __init__(self):
        reset_ok = True
        try:
            with open(CLEAR_REFS_PATH, "w", encoding="ascii") as clear_refs:
                clear_refs.write("5")
        except OSError:
            reset_ok = False

        rss_kb = _status_kb("VmRSS:")
        if _status_kb("VmHWM:") is not None:
            self.method = "vmhwm_reset" if reset_ok else "vmhwm"
            self.baseline = rss_kb * 1024 if rss_kb is not None else None
        else:
            self.method = "ru_maxrss"
            self.baseline = _ru_maxrss_bytes()

    def finish(self):
        if self.method == "ru_maxrss":
            peak = _ru_maxrss_bytes()
        else:
            hwm_kb = _status_kb("VmHWM:")
            peak = hwm_kb * 1024 if hwm_kb is not None else _ru_maxrss_bytes()
        if peak is None or self.baseline is None:
            return {"baseline": None, "peak": None, "delta": None, "method": self.method}
        peak = max(peak, self.baseline)
        return {
            "baseline": round(self.baseline / MB, 3),
            "peak": round(peak / MB, 3),
            "delta": round((peak - self.baseline) / MB, 3),
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def _empty_latency():
    return {k: None for k in ("avg", "min", "max", "std", "p25", "p50", "p75", "p90", "p95", "p99")}


def _system_info():
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "python_version": platform.python_version(),
    }


def _latency_stats(latencies_ms):
    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "avg": round(float(np.mean(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
    }


def _new_result(args, model_path, out_path):
    size = model_path.stat().st_size if model_path.is_file() else None
    return {
        "model_path": str(model_path),
        "framework": "onnxruntime",
        "provider": "CPUExecutionProvider",
        "warmup_runs": 0,
        "benchmark_runs": 0,
        "batch_size": 1,
        "latency_ms": _empty_latency(),
        "throughput": {"fps": None, "total_inferences": 0, "total_time_ms": None, "batch_size": 1},
        "memory_mb": {"baseline": None, "peak": None, "delta": None, "method": None},
        "model_size": {
            "bytes": size,
            "kb": round(size / 1024.0, 3) if size is not None else None,
            "mb": round(size / MB, 3) if size is not None else None,
        },
        "system_info": _system_info(),
        "settings": {
            "threads": args.threads,
            "seed": args.seed,
            "warmup": args.warmup,
            "runs": args.runs,
            "budget_seconds": args.budget,
            "profile": args.profile,
        },
        "thermal": {"temp_c_before": None, "temp_c_after": None, "governor": None},
        "profile_path": None,
        "onnxruntime_version": ort.__version__ if ort is not None else None,
        # timezone.utc, not datetime.UTC: bench.py runs on device Pythons that may predate 3.11.
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
        "success": False,
        "error": None,
    }


def _benchmark(args, model_path, out_path, result):
    memory = _Memory()  # baseline before the session is created
    result["thermal"]["temp_c_before"] = _read_temp_c()
    result["thermal"]["governor"] = _read_text(GOVERNOR_PATH)

    profile_prefix = None
    if args.profile:
        profile_prefix = str(out_path.with_name(out_path.stem + "_profile"))
    session = make_session(model_path, args.threads, profile_prefix)
    shapes = {item.name: item.shape for item in session.get_inputs()}
    inputs = build_inputs(shapes, args.seed)

    def run():
        session.run(None, inputs)

    warmups, latencies_ms, total_seconds = run_timed(run, args.warmup, args.runs, args.budget)
    result["thermal"]["temp_c_after"] = _read_temp_c()
    result["memory_mb"] = memory.finish()
    if args.profile:
        result["profile_path"] = session.end_profiling()

    count = len(latencies_ms)
    result["warmup_runs"] = warmups
    result["benchmark_runs"] = count
    result["latency_ms"] = _latency_stats(latencies_ms)
    result["throughput"] = {
        "fps": round(count / total_seconds, 2) if total_seconds > 0 else 0.0,
        "total_inferences": count,
        "total_time_ms": round(total_seconds * 1000.0, 4),
        "batch_size": 1,
    }
    result["success"] = True
    return result


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Benchmark this ONNX model on this device.")
    parser.add_argument("--model", default=str(HERE / "model.onnx"), help="ONNX file to benchmark")
    parser.add_argument("--threads", type=int, default=1, help="intra-op threads")
    parser.add_argument("--warmup", type=int, default=10, help="warm-up runs (not timed)")
    parser.add_argument("--runs", type=int, default=100, help="timed runs")
    parser.add_argument("--budget", type=float, default=10.0, help="time budget in seconds")
    parser.add_argument("--seed", type=int, default=0, help="seed for the random input")
    parser.add_argument("--profile", action="store_true", help="turn on ONNX Runtime profiling")
    parser.add_argument("--out", default=str(HERE / "measured.json"), help="output JSON path")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    model_path = Path(args.model)
    out_path = Path(args.out)
    result = _new_result(args, model_path, out_path)

    try:
        if ort is None:
            raise RuntimeError("onnxruntime not installed")
        if not model_path.is_file():
            raise FileNotFoundError(f"model file not found: {model_path}")
        result = _benchmark(args, model_path, out_path, result)
        code = 0
    except Exception as exc:  # the JSON must be written even when the run fails
        result["success"] = False
        result["error"] = str(exc) or type(exc).__name__
        code = 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{'OK' if code == 0 else 'FAILED'}: wrote {out_path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
