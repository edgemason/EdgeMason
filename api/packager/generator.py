"""Build the deployment ZIP: a self-contained folder that runs on the target.

The ZIP has the same layout and commands whichever candidate won (ONNX Runtime
or TVM, FP32 or INT8): `python inference.py` runs a self-test, and
`Model().predict(array)` runs the model from the user's own code.
"""

import json
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Dict, Optional

# Multi-arch image: Docker pulls arm64 on a Raspberry Pi / Jetson (64-bit OS)
# and amd64 on an x86_64 machine, so one Dockerfile serves every target.
PYTHON_IMAGE = "python:3.11-slim"

DOCKERFILE = f"""# {PYTHON_IMAGE} is multi-arch: Docker picks arm64 on a Raspberry Pi or
# Jetson (64-bit OS) and amd64 on an x86_64 machine.
FROM {PYTHON_IMAGE}

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "inference.py"]
"""

# Plain text, never passed through str.format, so braces are literal Python.
INFERENCE_SCRIPT = '''"""Run the optimized model.

    python inference.py              # self-test with a random input
    python inference.py input.npy    # run on your own float32 input

From your own code:

    from inference import Model
    model = Model()
    outputs = model.predict(array)   # list of numpy arrays
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


class Model:
    """Loads whichever model file is in this folder (ONNX Runtime or TVM)."""

    def __init__(self, folder=HERE):
        folder = Path(folder)
        if (folder / "model.onnx").is_file():
            self._load_onnx(folder / "model.onnx")
        elif (folder / "model_tvm.tar").is_file():
            self._load_tvm(folder / "model_tvm.tar", folder / "tvm_inputs.json")
        else:
            raise FileNotFoundError("model.onnx or model_tvm.tar must be next to inference.py")

    def _load_onnx(self, path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3  # errors only; hides harmless graph-cleanup warnings
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        model_input = session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in model_input.shape]
        self._run = lambda x: session.run(None, {self.input_name: x})

    def _load_tvm(self, path, specs_path):
        import tvm
        from tvm import relax

        spec = json.loads(specs_path.read_text(encoding="utf-8"))["inputs"][0]
        self.input_name, self.input_shape = spec["name"], spec["shape"]
        device = tvm.cpu(0)
        vm = relax.VirtualMachine(tvm.runtime.load_module(str(path)), device)

        def run(x):
            output = vm["main"](tvm.runtime.tensor(x, device))
            if isinstance(output, (list, tuple)) or type(output).__name__ in ("Tuple", "Array"):
                items = list(output)
            else:
                items = [output]
            return [np.asarray(item.numpy() if hasattr(item, "numpy") else item) for item in items]

        self._run = run

    def predict(self, inputs):
        x = np.ascontiguousarray(inputs, dtype=np.float32)
        if x.ndim == len(self.input_shape) - 1:
            x = x[None]  # add the batch dimension
        return self._run(x)


def main(argv):
    model = Model()
    if len(argv) > 1:
        inputs = np.load(argv[1], allow_pickle=False)
        source = argv[1]
    else:
        inputs = np.random.rand(*model.input_shape).astype(np.float32)
        source = "random self-test input"

    model.predict(inputs)  # warm-up
    start = time.perf_counter()
    outputs = model.predict(inputs)
    latency_ms = (time.perf_counter() - start) * 1000

    print(f"Input:  {source}, shape {list(np.shape(inputs))}")
    print(f"Output: shape {list(outputs[0].shape)}")
    if outputs[0].ndim == 2 and outputs[0].shape[1] > 1:
        top5 = np.argsort(outputs[0][0])[::-1][:5]
        print(f"Top-5 classes: {top5.tolist()}")
    print(f"Latency: {latency_ms:.1f} ms")
    print("OK")


if __name__ == "__main__":
    main(sys.argv)
'''


def _installed_version(*names: str) -> Optional[str]:
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _requirements(is_tvm: bool) -> str:
    """Pin the versions the server tested with. The target always gets the
    CPU onnxruntime package, even when the server uses onnxruntime-gpu."""
    if is_tvm:
        runtime = _installed_version("apache-tvm")
        lines = [f"apache-tvm=={runtime}" if runtime else "apache-tvm"]
    else:
        runtime = _installed_version("onnxruntime", "onnxruntime-gpu")
        lines = [f"onnxruntime=={runtime}" if runtime else "onnxruntime"]
    numpy_version = _installed_version("numpy")
    lines.append(f"numpy=={numpy_version}" if numpy_version else "numpy")
    return "\n".join(lines) + "\n"


def _input_spec(model_path: str, is_tvm: bool) -> dict:
    if is_tvm:
        sidecar = Path(str(model_path) + ".inputs.json")
        return json.loads(sidecar.read_text(encoding="utf-8"))["inputs"][0]
    import onnx

    model = onnx.load(model_path, load_external_data=False)
    initializers = {item.name for item in model.graph.initializer}
    model_input = next(item for item in model.graph.input if item.name not in initializers)
    shape = [
        dim.dim_value if dim.dim_value > 0 else (dim.dim_param or "batch")
        for dim in model_input.type.tensor_type.shape.dim
    ]
    return {"name": model_input.name, "shape": shape}


def _fmt(value, unit: str) -> str:
    return f"{value:.1f} {unit}" if isinstance(value, (int, float)) else "n/a"


def _readme(
    hardware_id: str,
    model_file: str,
    input_spec: dict,
    report: Dict,
    candidate: Optional[Dict],
    is_tvm: bool,
) -> str:
    estimate = report.get("hardware_estimate") or {}
    accuracy = report.get("accuracy") or {}
    top1 = accuracy.get("top1_accuracy")
    method = (candidate or {}).get("quantization", "unknown")
    framework = (candidate or {}).get("framework", "tvm" if is_tvm else "onnxruntime")

    lines = [
        "# EdgeMason deployment package",
        "",
        f"Optimized model for **{hardware_id}**, ready to run.",
        "",
        "## Run it",
        "",
        "Python 3.11 on a 64-bit OS (on a Raspberry Pi: 64-bit Raspberry Pi OS).",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "python inference.py              # self-test with a random input",
        "python inference.py input.npy    # your own input",
        "```",
        "",
        "With Docker (the image picks the right CPU architecture by itself):",
        "",
        "```bash",
        "docker build -t edgemason-model .",
        "docker run --rm edgemason-model",
        "docker run --rm -v \"$PWD/input.npy:/app/input.npy\" edgemason-model python inference.py input.npy",
        "```",
        "",
        "From your own Python code:",
        "",
        "```python",
        "from inference import Model",
        "model = Model()",
        "outputs = model.predict(array)   # list of numpy arrays",
        "```",
        "",
        "## Input",
        "",
        f"- Name: `{input_spec['name']}`",
        f"- Shape: `{input_spec['shape']}` (the batch dimension is added if you leave it out)",
        "- Type: float32, preprocessed the same way as the calibration images",
        "",
        "## Expected performance",
        "",
        "| | |",
        "|---|---|",
        f"| Estimated latency on {hardware_id} | {_fmt(estimate.get('latency_p50_ms'), 'ms')} |",
        f"| Estimated memory on {hardware_id} | {_fmt(estimate.get('memory_mb'), 'MB')} |",
        f"| Top-1 accuracy (held-out images) | {f'{top1 * 100:.1f}%' if isinstance(top1, (int, float)) else 'n/a'} |",
        f"| Optimization | {method} ({framework}) |",
        "",
        "Estimates come from benchmarking on the server and scaling to the target.",
        "Full numbers are in `benchmark_report.json`.",
        "",
        "## Files",
        "",
        f"- `{model_file}`: the optimized model",
        "- `inference.py`: loads and runs it",
        "- `requirements.txt`: the exact versions it was tested with",
        "- `Dockerfile`: optional container build",
        "- `benchmark_report.json`: full benchmark results",
    ]
    if is_tvm:
        lines += ["- `tvm_inputs.json`: input name and shape for the TVM model"]
        if hardware_id != "x86_64":
            lines += [
                "",
                "> **Note:** this TVM model was compiled for the server's x86_64 CPU and",
                f"> will not load on {hardware_id} yet (ARM cross-compilation is pending).",
            ]
    return "\n".join(lines) + "\n"


def create_deployment_package(
    model_path: str,
    benchmark_report: Dict,
    hardware_id: str,
    output_path: str = "./edgemason-deployment.zip",
    candidate: Optional[Dict] = None,
) -> str:
    """Create a ZIP with the model, inference.py, requirements, Dockerfile,
    README and the benchmark report."""
    is_tvm = str(model_path).endswith((".tar", ".so"))
    model_file = "model_tvm.tar" if is_tvm else "model.onnx"
    input_spec = _input_spec(str(model_path), is_tvm)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, arcname=model_file)
        if is_tvm:
            zf.write(str(model_path) + ".inputs.json", arcname="tvm_inputs.json")
        zf.writestr("inference.py", INFERENCE_SCRIPT)
        zf.writestr("requirements.txt", _requirements(is_tvm))
        zf.writestr("Dockerfile", DOCKERFILE)
        zf.writestr("benchmark_report.json", json.dumps(benchmark_report, indent=2))
        zf.writestr(
            "README.md",
            _readme(hardware_id, model_file, input_spec, benchmark_report, candidate, is_tvm),
        )
    return output_path
