"""Tests for the on-device benchmark template shipped in the ONNX deployment ZIP.

bench.py is a standalone script, so it is loaded by file path. Constants that
point at /proc and /sys are module-level so tests can replace them.
"""

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "api" / "packager" / "templates" / "bench.py"

JSON_KEYS = {
    "model_path": str,
    "framework": str,
    "provider": str,
    "warmup_runs": int,
    "benchmark_runs": int,
    "batch_size": int,
    "latency_ms": dict,
    "throughput": dict,
    "memory_mb": dict,
    "model_size": dict,
    "system_info": dict,
    "settings": dict,
    "thermal": dict,
    "profile_path": (str, type(None)),
    "onnxruntime_version": str,
    "timestamp": str,
    "success": bool,
    "error": (str, type(None)),
}
LATENCY_KEYS = {"avg", "min", "max", "std", "p25", "p50", "p75", "p90", "p95", "p99"}
THROUGHPUT_KEYS = {"fps", "total_inferences", "total_time_ms", "batch_size"}
MEMORY_KEYS = {"baseline", "peak", "delta", "method"}
MODEL_SIZE_KEYS = {"bytes", "kb", "mb"}
SYSTEM_INFO_KEYS = {
    "os", "os_release", "architecture", "processor", "cpu_count_logical", "python_version"
}
SETTINGS_KEYS = {"threads", "seed", "warmup", "runs", "budget_seconds", "profile"}
THERMAL_KEYS = {"temp_c_before", "temp_c_after", "governor"}


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench", BENCH_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {BENCH_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_bench(*args, cwd=None):
    return subprocess.run(
        [sys.executable, str(BENCH_PATH), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
    )


# 1. Same shapes and seed give identical inputs.
def test_build_inputs_is_deterministic_for_a_seed(bench):
    first = bench.build_inputs({"x": [1, 3, 224, 224]}, 0)
    second = bench.build_inputs({"x": [1, 3, 224, 224]}, 0)

    assert first.keys() == second.keys()
    assert (first["x"] == second["x"]).all()


# 2. Different seeds give different inputs.
def test_build_inputs_differs_between_seeds(bench):
    zero = bench.build_inputs({"x": [1, 3, 224, 224]}, 0)["x"]
    one = bench.build_inputs({"x": [1, 3, 224, 224]}, 1)["x"]

    assert not (zero == one).all()


# 3. Dynamic or non-positive dimensions become 1.
@pytest.mark.parametrize("shape", [["batch", 4], [-1, 4]])
def test_build_inputs_replaces_dynamic_dimensions_with_one(bench, shape):
    assert bench.build_inputs({"x": shape}, 0)["x"].shape == (1, 4)


# 4. --threads sets intra-op threads; inter-op is always 1.
def test_session_options_follow_threads(bench, tiny_onnx):
    session = bench.make_session(str(tiny_onnx), threads=2)
    options = session.get_session_options()

    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 1


# 5. Timing loop never stops before MIN_TIMED_RUNS timed runs, even over budget.
def test_timing_loop_records_min_timed_runs_when_over_budget(bench):
    def slow_run():
        time.sleep(0.02)

    warmups, latencies, total = bench.run_timed(
        slow_run, warmup_runs=10, benchmark_runs=100, budget=0.001
    )

    assert len(latencies) == bench.MIN_TIMED_RUNS == 10
    assert total > 0
    assert warmups >= 1


# 6. End-to-end run writes every key from §2.6 with success true.
def test_bench_writes_full_result_json(tmp_path, tiny_onnx):
    out = tmp_path / "measured.json"
    result = _run_bench("--model", str(tiny_onnx), "--out", str(out))

    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    assert set(data) == set(JSON_KEYS)
    for key, expected in JSON_KEYS.items():
        assert isinstance(data[key], expected), key
    assert set(data["latency_ms"]) == LATENCY_KEYS
    assert set(data["throughput"]) == THROUGHPUT_KEYS
    assert set(data["memory_mb"]) == MEMORY_KEYS
    assert set(data["model_size"]) == MODEL_SIZE_KEYS
    assert set(data["system_info"]) == SYSTEM_INFO_KEYS
    assert set(data["settings"]) == SETTINGS_KEYS
    assert set(data["thermal"]) == THERMAL_KEYS
    assert data["success"] is True
    assert data["error"] is None


# 7. Missing thermal and governor files give null values and still succeed.
def test_missing_thermal_and_governor_files_give_nulls(bench, tmp_path, tiny_onnx, monkeypatch):
    monkeypatch.setattr(bench, "THERMAL_PATH", str(tmp_path / "no_temp"))
    monkeypatch.setattr(bench, "GOVERNOR_PATH", str(tmp_path / "no_governor"))
    out = tmp_path / "measured.json"

    code = bench.main(
        ["--model", str(tiny_onnx), "--out", str(out), "--runs", "10", "--budget", "5"]
    )

    assert code == 0
    data = json.loads(out.read_text())
    assert data["success"] is True
    assert data["thermal"] == {"temp_c_before": None, "temp_c_after": None, "governor": None}


# 8. An unwritable clear_refs path falls back to vmhwm and still succeeds.
def test_unwritable_clear_refs_falls_back_to_vmhwm(bench, tmp_path, tiny_onnx, monkeypatch):
    monkeypatch.setattr(bench, "CLEAR_REFS_PATH", str(tmp_path / "missing_dir" / "clear_refs"))
    out = tmp_path / "measured.json"

    code = bench.main(
        ["--model", str(tiny_onnx), "--out", str(out), "--runs", "10", "--budget", "5"]
    )

    assert code == 0
    data = json.loads(out.read_text())
    assert data["success"] is True
    assert data["memory_mb"]["method"] in ("vmhwm", "ru_maxrss")


# 9. bench.py runs from an empty folder with only itself and the model: no api import.
def test_bench_runs_without_the_repo(tmp_path, tiny_onnx):
    folder = tmp_path / "isolated"
    folder.mkdir()
    shutil.copy(BENCH_PATH, folder / "bench.py")
    model = folder / "model.onnx"
    shutil.copy(tiny_onnx, model)
    out = folder / "measured.json"

    result = _run_bench(
        "--model", str(model), "--out", str(out), "--runs", "10", "--budget", "5", cwd=folder
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["success"] is True




def _write_weight_model(path, rows=1000, cols=10000):
    """MatMul(X, W) with a large float32 weight W and tiny input and output.

    Returns the weight size in bytes. The weight is random, not zeros, so its
    pages are actually resident once the session loads them.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    weights = np.random.default_rng(0).random((rows, cols), dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["X", "W"], ["Y"])],
        "big_weight",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, rows])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, cols])],
        initializer=[
            helper.make_tensor("W", TensorProto.FLOAT, weights.shape, weights.tobytes(), raw=True)
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return weights.nbytes


# The baseline must be taken before the session loads the weights. If it is taken after,
# the weights are counted in the baseline and memory_mb.delta under-reports them.
def test_memory_delta_includes_model_weights(tmp_path):
    model = tmp_path / "big_weight.onnx"
    weight_bytes = _write_weight_model(model)
    out = tmp_path / "measured.json"

    result = _run_bench(
        "--model", str(model), "--out", str(out), "--warmup", "1", "--runs", "10", "--budget", "1"
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    assert data["success"] is True
    if not sys.platform.startswith("linux") or data["memory_mb"]["method"] != "vmhwm_reset":
        pytest.skip("memory peak reset (vmhwm_reset) is only available on Linux")

    weight_mb = weight_bytes / (1024 * 1024)
    delta = data["memory_mb"]["delta"]
    assert delta >= 0.8 * weight_mb, (
        f"delta={delta} MB, weight_mb={weight_mb:.2f} MB, memory_mb={data['memory_mb']}"
    )
