"""How static INT8 accuracy changes with the size of the calibration set.

The 256 labeled images are split once (fixed seed) into a calibration pool of
100 and a fixed test set of 156 that never changes. For each calibration size,
several random subsets are drawn from the pool and the model is quantized
(per-channel, the shipped setting, and per-tensor, the old setting).

The calibration set plays the role of a "training" set (it is what the
quantizer learns its scales from) and the test set the "validation" set, so
accuracy is reported on both.

Usage:
    uv run python scripts/calibration_size_study.py --out reports/data/calibration_size_study.json
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from api.optimizer.quantizer import Quantizer  # noqa: E402
from api.optimizer.validation import load_validation_data  # noqa: E402
from quantization_analysis import accuracy, per_tensor_int8, predict  # noqa: E402

DATA = PROJECT_ROOT / "models" / "samples" / "imagenetv2_mobilenetv2_256.npz"
POOL_SIZE = 100
SIZES = [1, 2, 5, 10, 25, 50, 100]
DRAWS = 3
DEFAULT_MODELS = ["squeezenet1.1-13.onnx", "shufflenet-v2-13.onnx", "mobilenetv2.onnx", "resnet18-v1-13.onnx"]


def study(model_name: str) -> dict:
    data = load_validation_data(DATA)
    order = np.random.default_rng(0).permutation(len(data.inputs))
    pool_index, test_index = order[:POOL_SIZE], order[POOL_SIZE:]
    test_x, test_y = data.inputs[test_index], data.labels[test_index]
    model_path = str(PROJECT_ROOT / "models" / model_name)

    fp32_test = accuracy(predict(model_path, test_x), test_y)
    points = []
    with tempfile.TemporaryDirectory(prefix="edgemason-calib-") as directory:
        for size in SIZES:
            draws = 1 if size == POOL_SIZE else DRAWS
            for draw in range(draws):
                subset = np.sort(np.random.default_rng(1000 + draw).choice(pool_index, size, replace=False))
                calib_x, calib_y = data.inputs[subset], data.labels[subset]
                work = Path(directory) / f"{size}_{draw}"
                work.mkdir()
                per_channel = Quantizer(str(work)).create_static_int8(model_path, calib_x)
                per_tensor = per_tensor_int8(model_path, calib_x, work)
                points.append({
                    "calibration_size": size,
                    "draw": draw,
                    "per_channel_test": accuracy(predict(per_channel, test_x), test_y),
                    "per_tensor_test": accuracy(predict(per_tensor, test_x), test_y),
                    "per_channel_calibration": accuracy(predict(per_channel, calib_x), calib_y),
                    "fp32_calibration": accuracy(predict(model_path, calib_x), calib_y),
                })
                print(f"    size={size:3d} draw={draw}  per-channel test top1="
                      f"{points[-1]['per_channel_test']['top1'] * 100:5.1f}%  per-tensor="
                      f"{points[-1]['per_tensor_test']['top1'] * 100:5.1f}%", flush=True)
    return {"model": model_name, "test_samples": len(test_y), "pool_size": POOL_SIZE,
            "fp32_test": fp32_test, "points": points}


def main() -> None:
    parser = argparse.ArgumentParser(description="Static INT8 accuracy vs calibration-set size.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "data" / "calibration_size_study.json")
    args = parser.parse_args()
    results = []
    for model_name in args.models:
        print(f"[{time.strftime('%H:%M:%S')}] {model_name} ...", flush=True)
        results.append(study(model_name))
        print(f"    FP32 test top1={results[-1]['fp32_test']['top1'] * 100:.1f}%", flush=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
