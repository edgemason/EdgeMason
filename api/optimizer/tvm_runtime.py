"""Shared TVM Relax helpers (apache-tvm >= 0.25, no Relay).

Single place for compiling ONNX -> Relax VM .tar and running it, so the
quantizer, benchmark engine and accuracy evaluation stay in sync.
"""

import json
from pathlib import Path

import numpy as np


def _require_tvm():
    try:
        import tvm
    except ImportError as exc:
        raise ImportError(
            "Apache TVM is not installed. Run: pip install apache-tvm"
        ) from exc
    return tvm


def is_tvm_available() -> bool:
    try:
        import tvm  # noqa: F401
        import tvm.relax  # noqa: F401
        from tvm.relax.frontend.onnx import from_onnx  # noqa: F401

        return True
    except ImportError:
        return False


def onnx_input_specs(model_path: str | Path) -> list[dict]:
    """Return [{name, shape}] for the ONNX model inputs.

    Dynamic dimensions fall back to 1 (single-sample compile).
    """
    import onnx

    model = onnx.load(str(model_path))
    initializer_names = {item.name for item in model.graph.initializer}
    specs = []
    for item in model.graph.input:
        if item.name in initializer_names:
            continue
        shape: list[int] = []
        for index, dim in enumerate(item.type.tensor_type.shape.dim):
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            elif index == 0:
                shape.append(1)
            else:
                raise ValueError("only the model batch dimension may be dynamic")
        specs.append({"name": item.name, "shape": shape})
    if len(specs) != 1:
        raise ValueError("TVM compilation supports single-input models only")
    return specs


def sidecar_path(tar_path: str | Path) -> Path:
    return Path(str(tar_path) + ".inputs.json")


def compile_onnx(tar_path: str | Path, model_path: str | Path, target: str | dict) -> str:
    """Compile an ONNX model to a Relax VM .tar plus a sidecar inputs file."""
    tvm = _require_tvm()
    from tvm.relax.frontend.onnx import from_onnx

    import onnx

    specs = onnx_input_specs(model_path)
    model = onnx.load(str(model_path))
    shape_dict = {spec["name"]: spec["shape"] for spec in specs}
    mod = from_onnx(model, shape_dict)
    executable = tvm.relax.build(mod, target=tvm.target.Target(target))
    tar_path = str(tar_path)
    executable.export_library(tar_path)
    with open(sidecar_path(tar_path), "w", encoding="utf-8") as handle:
        json.dump({"inputs": specs, "target": target}, handle)
    return tar_path


def load_input_specs(tar_path: str | Path) -> list[dict]:
    """Read the sidecar written by compile_onnx."""
    path = sidecar_path(tar_path)
    if not path.is_file():
        raise ValueError(
            f"missing TVM inputs sidecar for {tar_path}; "
            "recompile the artifact with Quantizer to regenerate it"
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["inputs"]


def load_vm(tar_path: str | Path):
    """Load a compiled .tar and return (vm, device)."""
    tvm = _require_tvm()
    module = tvm.runtime.load_module(str(tar_path))
    device = tvm.cpu(0)
    return tvm.relax.VirtualMachine(module, device), device


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def run_vm(vm, device, inputs: dict[str, np.ndarray], order: list[str]) -> list[np.ndarray]:
    """Run the Relax VM 'main' with named inputs, return outputs as numpy."""
    tvm = _require_tvm()
    args = [
        tvm.runtime.tensor(np.ascontiguousarray(inputs[name], dtype=np.float32), device)
        for name in order
    ]
    output = vm["main"](*args)
    if isinstance(output, (tuple, list)):
        return [_to_numpy(item) for item in output]
    if type(output).__module__.startswith("tvm") and type(output).__name__ in (
        "Tuple",
        "Array",
    ):
        return [_to_numpy(item) for item in list(output)]
    return [_to_numpy(output)]


def run_model(tar_path: str | Path, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Load (cached per call) and run a .tar artifact."""
    specs = load_input_specs(tar_path)
    vm, device = load_vm(tar_path)
    return run_vm(vm, device, inputs, [spec["name"] for spec in specs])
