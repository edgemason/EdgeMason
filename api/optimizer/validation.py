from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import onnxruntime as ort


MAX_VALIDATION_BYTES = 256 * 1024 * 1024
MAX_VALIDATION_SAMPLES = 256


@dataclass(frozen=True)
class ValidationData:
    inputs: np.ndarray
    labels: np.ndarray


def _validate_inputs(inputs: np.ndarray) -> np.ndarray:
    if inputs.dtype != np.float32 or inputs.ndim < 2:
        raise ValueError("inputs must be a float32 batch")
    if not 1 <= len(inputs) <= MAX_VALIDATION_SAMPLES:
        raise ValueError(f"validation data must contain 1-{MAX_VALIDATION_SAMPLES} samples")
    if inputs.nbytes > MAX_VALIDATION_BYTES:
        raise ValueError("expanded validation data is too large")
    if not np.isfinite(inputs).all():
        raise ValueError("inputs must contain only finite values")
    return inputs


def load_calibration_inputs(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        return load_validation_data(path).inputs
    if path.suffix.lower() != ".npy":
        raise ValueError("calibration data must be .npy or .npz")
    if path.stat().st_size > MAX_VALIDATION_BYTES:
        raise ValueError("calibration data is too large")
    inputs = np.load(path, allow_pickle=False, mmap_mode="r")
    return _validate_inputs(inputs)


def load_validation_data(path: str | Path) -> ValidationData:
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("validation data must be a labeled .npz file")
    if path.stat().st_size > MAX_VALIDATION_BYTES:
        raise ValueError("validation data is too large")

    with ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_VALIDATION_BYTES:
            raise ValueError("expanded validation data is too large")

    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"inputs", "labels"}:
            raise ValueError("validation .npz must contain only inputs and labels")
        inputs = np.asarray(data["inputs"])
        labels = np.asarray(data["labels"])

    inputs = _validate_inputs(inputs)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array")
    if len(inputs) != len(labels):
        raise ValueError("inputs and labels must contain the same number of samples")
    if np.any(labels < 0):
        raise ValueError("labels must be non-negative")

    return ValidationData(inputs=inputs, labels=labels.astype(np.int64, copy=False))


def evaluate_tvm(
    model_path: str | Path,
    validation: ValidationData,
) -> dict[str, int | float]:
    """Score a compiled TVM Relax .tar artifact (top1/top5).

    The artifact is compiled for a fixed batch (see tvm_runtime sidecar),
    so samples are evaluated in chunks of the compiled batch size.
    """
    from api.optimizer.tvm_runtime import load_input_specs, load_vm, run_vm

    specs = load_input_specs(model_path)
    if len(specs) != 1:
        raise ValueError("accuracy evaluation supports single-input models only")
    spec = specs[0]
    compiled_batch = spec["shape"][0]
    if not isinstance(compiled_batch, int) or compiled_batch <= 0:
        raise ValueError("TVM artifact has an invalid compiled batch size")

    vm, device = load_vm(model_path)
    top1_correct = 0
    top5_correct = 0
    sample_count = len(validation.inputs)

    for start in range(0, sample_count, compiled_batch):
        batch = validation.inputs[start : start + compiled_batch]
        labels = validation.labels[start : start + compiled_batch]
        if len(batch) != compiled_batch:
            # Relax artifacts are fixed-shape; evaluate leftovers one by one
            # only when the artifact was compiled for batch 1.
            if compiled_batch != 1:
                raise ValueError(
                    "validation sample count must divide the TVM compiled batch size"
                )
            for sample, label in zip(batch, labels):
                logits = run_vm(vm, device, {spec["name"]: sample[None]}, [spec["name"]])[0]
                top1_correct, top5_correct = _accumulate(
                    logits, label, top1_correct, top5_correct
                )
            continue
        logits = run_vm(vm, device, {spec["name"]: batch}, [spec["name"]])[0]
        if logits.ndim != 2 or logits.shape[0] != len(batch):
            raise ValueError("model output must have shape [batch, classes]")
        if np.any(labels >= logits.shape[1]):
            raise ValueError("validation label exceeds model class count")
        top1_correct += int(np.sum(np.argmax(logits, axis=1) == labels))
        top_k = min(5, logits.shape[1])
        top_indices = np.argpartition(logits, -top_k, axis=1)[:, -top_k:]
        top5_correct += int(np.sum(np.any(top_indices == labels[:, None], axis=1)))

    return {
        "sample_count": sample_count,
        "top1_accuracy": top1_correct / sample_count,
        "top5_accuracy": top5_correct / sample_count,
    }


def _accumulate(
    logits: np.ndarray, label: int, top1_correct: int, top5_correct: int
) -> tuple[int, int]:
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("model output must have shape [batch, classes]")
    if label >= logits.shape[1]:
        raise ValueError("validation label exceeds model class count")
    top1_correct += int(np.argmax(logits[0]) == label)
    top_k = min(5, logits.shape[1])
    top_indices = np.argpartition(logits[0], -top_k)[-top_k:]
    top5_correct += int(label in top_indices)
    return top1_correct, top5_correct


def evaluate_onnx(
    model_path: str | Path,
    validation: ValidationData,
    batch_size: int = 16,
) -> dict[str, int | float]:
    if not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be between 1 and 64")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    model_inputs = session.get_inputs()
    if len(model_inputs) != 1:
        raise ValueError("accuracy evaluation supports single-input models only")

    input_name = model_inputs[0].name
    model_batch_size = model_inputs[0].shape[0]
    if isinstance(model_batch_size, int) and model_batch_size > 0:
        batch_size = model_batch_size
    top1_correct = 0
    top5_correct = 0

    for start in range(0, len(validation.inputs), batch_size):
        batch = validation.inputs[start : start + batch_size]
        labels = validation.labels[start : start + batch_size]
        logits = session.run(None, {input_name: batch})[0]
        if logits.ndim != 2 or logits.shape[0] != len(batch):
            raise ValueError("model output must have shape [batch, classes]")
        if np.any(labels >= logits.shape[1]):
            raise ValueError("validation label exceeds model class count")

        top1_correct += int(np.sum(np.argmax(logits, axis=1) == labels))
        top_k = min(5, logits.shape[1])
        top_indices = np.argpartition(logits, -top_k, axis=1)[:, -top_k:]
        top5_correct += int(np.sum(np.any(top_indices == labels[:, None], axis=1)))

    sample_count = len(validation.inputs)
    return {
        "sample_count": sample_count,
        "top1_accuracy": top1_correct / sample_count,
        "top5_accuracy": top5_correct / sample_count,
    }
