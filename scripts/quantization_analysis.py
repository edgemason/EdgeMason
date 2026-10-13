"""Quantization analysis for the results slide.

For each model, on the same seeded calibration/test split the pipeline uses:
- FP32 vs static INT8 (per-channel, the shipped setting): accuracy and a 2x2
  agreement matrix (is each model right or wrong on the same test image)
- ablation 1 (Fix 4): static INT8 per-tensor vs per-channel
- ablation 2 (Fix 5): dynamic INT8 vs static INT8 latency and accuracy

Latency uses one thread, like the benchmark engine (10 warm-up + 50 timed runs).

Usage:
    uv run python scripts/quantization_analysis.py --out reports/data/quantization_analysis.json
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.optimizer.quantizer import Quantizer  # noqa: E402
from api.optimizer.validation import load_validation_data, split_calibration_and_test  # noqa: E402

MODELS = [
    "squeezenet1.1-13.onnx", "shufflenet-v2-13.onnx", "mobilenetv2.onnx",
    "resnet18-v1-13.onnx", "resnet50.onnx", "resnet152.onnx",
]
CALIBRATION = PROJECT_ROOT / "models" / "samples" / "imagenetv2_mobilenetv2_256.npz"


def _session(path: str, threads: int = 0) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    if threads:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


def predict(path: str, inputs: np.ndarray) -> np.ndarray:
    session = _session(path)
    model_input = session.get_inputs()[0]
    # Some models (e.g. ONNX Model Zoo) declare a fixed batch size of 1.
    batch = model_input.shape[0] if isinstance(model_input.shape[0], int) and model_input.shape[0] > 0 else 16
    logits = np.concatenate(
        [session.run(None, {model_input.name: inputs[i : i + batch]})[0] for i in range(0, len(inputs), batch)]
    )
    return logits.reshape(len(inputs), -1)


def accuracy(logits: np.ndarray, labels: np.ndarray) -> dict:
    top5 = np.argsort(logits, axis=1)[:, -5:]
    return {
        "top1": float(np.mean(np.argmax(logits, axis=1) == labels)),
        "top5": float(np.mean(np.any(top5 == labels[:, None], axis=1))),
    }


def latency_p50(path: str, sample: np.ndarray) -> float:
    session = _session(path, threads=1)
    feed = {session.get_inputs()[0].name: sample}
    for _ in range(10):
        session.run(None, feed)
    times = []
    for _ in range(50):
        start = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(times, 50))


def per_tensor_int8(model_path: str, calibration: np.ndarray, out_dir: Path) -> str:
    """The pre-Fix-4 setting: one weight scale per layer."""
    session = _session(model_path)
    name = session.get_inputs()[0].name

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.index = 0

        def get_next(self):
            if self.index >= len(calibration):
                return None
            self.index += 1
            return {name: calibration[self.index - 1 : self.index]}

    prepped = out_dir / "per_tensor_prepped.onnx"
    quant_pre_process(model_path, str(prepped), skip_symbolic_shape=True)
    out = out_dir / "per_tensor_int8.onnx"
    quantize_static(str(prepped), str(out), Reader(), weight_type=QuantType.QInt8,
                    activation_type=QuantType.QInt8, per_channel=False)
    return str(out)


def analyse(model_name: str) -> dict:
    split = split_calibration_and_test(load_validation_data(CALIBRATION))
    test_x, test_y = split.test.inputs, split.test.labels
    model_path = str(PROJECT_ROOT / "models" / model_name)
    sample = test_x[:1]
    with tempfile.TemporaryDirectory(prefix="edgemason-analysis-") as directory:
        work = Path(directory)
        quantizer = Quantizer(str(work))
        variants = {
            "fp32": model_path,
            "static_int8_per_channel": quantizer.create_static_int8(model_path, split.calibration),
            "static_int8_per_tensor": per_tensor_int8(model_path, split.calibration, work),
            "dynamic_int8": quantizer.create_dynamic_int8(model_path),
        }
        logits = {key: predict(path, test_x) for key, path in variants.items()}
        result = {
            "model": model_name,
            "calibration_samples": len(split.calibration),
            "test_samples": len(test_y),
            "variants": {
                key: accuracy(logits[key], test_y)
                | {"latency_p50_ms_1thread": latency_p50(path, sample),
                   "size_mb": Path(path).stat().st_size / 2**20}
                for key, path in variants.items()
            },
        }

    fp32_right = np.argmax(logits["fp32"], axis=1) == test_y
    int8_right = np.argmax(logits["static_int8_per_channel"], axis=1) == test_y
    result["agreement_fp32_vs_static_int8"] = {
        "both_right": int(np.sum(fp32_right & int8_right)),
        "fp32_right_int8_wrong": int(np.sum(fp32_right & ~int8_right)),
        "fp32_wrong_int8_right": int(np.sum(~fp32_right & int8_right)),
        "both_wrong": int(np.sum(~fp32_right & ~int8_right)),
        "same_top1_prediction": float(np.mean(
            np.argmax(logits["fp32"], axis=1) == np.argmax(logits["static_int8_per_channel"], axis=1)
        )),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantization analysis for the results slide.")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "data" / "quantization_analysis.json")
    args = parser.parse_args()
    results = []
    for model_name in MODELS:
        print(f"[{time.strftime('%H:%M:%S')}] {model_name} ...", flush=True)
        results.append(analyse(model_name))
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        variants = results[-1]["variants"]
        for key, value in variants.items():
            print(f"    {key:26} top1={value['top1'] * 100:5.1f}%  p50={value['latency_p50_ms_1thread']:7.1f} ms  "
                  f"size={value['size_mb']:6.1f} MB", flush=True)
        print(f"    agreement: {results[-1]['agreement_fp32_vs_static_int8']}", flush=True)


if __name__ == "__main__":
    main()
