import json
import subprocess
import sys
import zipfile
from importlib import metadata

import numpy as np
import pytest

from api.packager.generator import create_deployment_package

REPORT = {
    "latency_ms": {"p50": 12.3},
    "hardware_estimate": {"latency_p50_ms": 145.2, "memory_mb": 25.9},
    "accuracy": {"top1_accuracy": 0.575},
}
CANDIDATE = {"name": "onnx_static_int8", "framework": "onnxruntime", "quantization": "static_int8"}


def _package(model_path, tmp_path, hardware_id="rpi4"):
    output = create_deployment_package(
        model_path=str(model_path),
        benchmark_report=REPORT,
        hardware_id=hardware_id,
        output_path=str(tmp_path / "deploy.zip"),
        candidate=CANDIDATE,
    )
    folder = tmp_path / "unzipped"
    with zipfile.ZipFile(output) as zf:
        zf.extractall(folder)
    return folder


def _run(folder, *args):
    return subprocess.run(
        [sys.executable, "inference.py", *args],
        cwd=folder,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_create_deployment_package(tiny_onnx, tmp_path):
    folder = _package(tiny_onnx, tmp_path)

    assert {path.name for path in folder.iterdir()} == {
        "model.onnx",
        "inference.py",
        "requirements.txt",
        "Dockerfile",
        "benchmark_report.json",
        "README.md",
    }


def test_onnx_package_runs_out_of_the_box(tiny_onnx, tmp_path):
    folder = _package(tiny_onnx, tmp_path)

    self_test = _run(folder)
    assert self_test.returncode == 0, self_test.stderr
    assert "Output: shape [1, 8]" in self_test.stdout
    assert "OK" in self_test.stdout

    # Own input without the batch dimension: it is added automatically.
    np.save(folder / "input.npy", np.ones(4, dtype=np.float32))
    own_input = _run(folder, "input.npy")
    assert own_input.returncode == 0, own_input.stderr
    assert "Output: shape [1, 8]" in own_input.stdout


def test_package_can_be_used_from_python_code(tiny_onnx, tmp_path):
    folder = _package(tiny_onnx, tmp_path)
    script = (
        "import numpy as np\n"
        "from inference import Model\n"
        "outputs = Model().predict(np.ones((1, 4), dtype=np.float32))\n"
        "print(outputs[0].shape)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=folder, capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "(1, 8)"


def test_package_pins_tested_versions_and_uses_multi_arch_image(tiny_onnx, tmp_path):
    folder = _package(tiny_onnx, tmp_path)

    requirements = (folder / "requirements.txt").read_text().split()
    runtime = next(
        metadata.version(name)
        for name in ("onnxruntime", "onnxruntime-gpu")
        if _installed(name)
    )
    # The target gets the CPU package even if the server uses onnxruntime-gpu.
    assert requirements == [f"onnxruntime=={runtime}", f"numpy=={metadata.version('numpy')}"]

    dockerfile = (folder / "Dockerfile").read_text()
    assert "FROM python:3.11-slim" in dockerfile
    assert "arm64v8" not in dockerfile
    assert "pip install --no-cache-dir -r requirements.txt" in dockerfile


def test_readme_describes_input_and_expected_performance(tiny_onnx, tmp_path):
    readme = (_package(tiny_onnx, tmp_path) / "README.md").read_text()

    assert "**rpi4**" in readme
    assert "`X`" in readme and "`[1, 4]`" in readme
    assert "145.2 ms" in readme and "25.9 MB" in readme and "57.5%" in readme
    assert "static_int8 (onnxruntime)" in readme
    assert "{{" not in (tmp_path / "unzipped" / "inference.py").read_text()


def test_tvm_package_runs_out_of_the_box(tiny_onnx, tmp_path):
    from api.optimizer.tvm_runtime import compile_onnx, is_tvm_available

    if not is_tvm_available():
        pytest.skip("Apache TVM is not installed")
    tar_path = compile_onnx(tmp_path / "model.tar", tiny_onnx, {"kind": "llvm"})
    folder = _package(tar_path, tmp_path, hardware_id="x86_64")

    assert (folder / "model_tvm.tar").is_file()
    assert json.loads((folder / "tvm_inputs.json").read_text())["inputs"][0]["shape"] == [1, 4]
    assert (folder / "requirements.txt").read_text().startswith("apache-tvm==")

    result = _run(folder)
    assert result.returncode == 0, result.stderr
    assert "Output: shape [1, 8]" in result.stdout


def _installed(name):
    try:
        metadata.version(name)
        return True
    except metadata.PackageNotFoundError:
        return False

