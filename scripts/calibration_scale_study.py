"""How static INT8 accuracy changes as the calibration dataset grows.

Uses one large ImageNetV2 subset (default imagenetv2_1000.npz). The set is
shuffled once with a fixed seed: the last 700 images are a FIXED test set that
never changes, and the first 300 form the calibration pool. Calibration subsets
of 30, 77, 150, 225 and 300 images are drawn from the pool (several random draws
per size). These sizes equal 30% of total datasets of 100, 256, 500, 750 and
1000 images, so each point answers "a 30/70 split of a total dataset of this
size", while the fixed 700-image test set keeps every accuracy directly
comparable (±~3.7 points instead of ±7).

For each model and calibration size we quantize per-channel static INT8 (the
shipped setting) and score top-1/top-5 on the fixed test set. FP32 is scored
once per model as the reference.

Usage:
    uv run python scripts/calibration_scale_study.py \
        --data models/samples/imagenetv2_1000.npz \
        --out reports/data/calibration_scale_study.json
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
from quantization_analysis import _session, accuracy  # noqa: E402

MODELS = [
    "squeezenet1.1-13.onnx", "shufflenet-v2-13.onnx", "mobilenetv2.onnx",
    "resnet18-v1-13.onnx", "resnet50.onnx", "resnet152.onnx",
]
# Default: 1000-image run. Calibration sizes are 30% of totals 100..1000.
DEFAULT_TEST_SIZE = 700
DEFAULT_SIZES = [30, 77, 150, 225, 300]
DEFAULT_DRAWS = 3


def predict(path: str, inputs: np.ndarray, batch: int = 64) -> np.ndarray:
    session = _session(path)  # default: all CPU cores
    model_input = session.get_inputs()[0]
    step = model_input.shape[0] if isinstance(model_input.shape[0], int) and model_input.shape[0] > 0 else batch
    logits = np.concatenate(
        [session.run(None, {model_input.name: inputs[i : i + step]})[0] for i in range(0, len(inputs), step)]
    )
    return logits.reshape(len(inputs), -1)


def save(results, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def study(model_name, test_x, test_y, pool_x, sizes, max_draws, out_dir, results, out_path):
    model_path = str(PROJECT_ROOT / "models" / model_name)
    # Reuse the model's existing entry (partial or complete) for per-point resume.
    entry = next((r for r in results if r["model"] == model_name), None)
    if entry is None:
        fp32 = accuracy(predict(model_path, test_x), test_y)
        print(f"    FP32 top1={fp32['top1'] * 100:.1f}%", flush=True)
        entry = {"model": model_name, "test_samples": int(len(test_y)),
                 "pool_size": int(len(pool_x)), "fp32_test": fp32, "points": []}
        results.append(entry)
        save(results, out_path)

    done_points = {(p["calibration_size"], p["draw"]) for p in entry["points"]}
    for size in sizes:
        draws = 1 if size >= len(pool_x) else max_draws
        total = round(size / 0.3)
        for draw in range(draws):
            if (size, draw) in done_points:
                continue
            index = np.sort(np.random.default_rng(1000 + draw).choice(len(pool_x), size, replace=False))
            calib = pool_x[index]
            out = out_dir / f"{model_name}_{size}_{draw}"
            out.mkdir(parents=True, exist_ok=True)
            quantized = Quantizer(str(out)).create_static_int8(model_path, calib)
            acc = accuracy(predict(quantized, test_x), test_y)
            entry["points"].append({"calibration_size": int(size), "total_dataset": int(total),
                                    "draw": draw, "test": acc})
            save(results, out_path)  # checkpoint after every point
            print(f"    size={size:3d} (total {total:4d}) draw={draw} "
                  f"INT8 top1={acc['top1'] * 100:.1f}%", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Static INT8 accuracy vs total calibration dataset size.")
    parser.add_argument("--data", type=Path,
                        default=PROJECT_ROOT / "models" / "samples" / "imagenetv2_1000.npz")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "reports" / "data" / "calibration_scale_study.json")
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES),
                        help="comma-separated calibration sizes")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    args = parser.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]

    with np.load(args.data, allow_pickle=False) as data:
        inputs, labels = data["inputs"], data["labels"]
    if args.test_size + max(sizes) > len(inputs):
        raise SystemExit(f"need >= {args.test_size + max(sizes)} images, have {len(inputs)}")
    order = np.random.default_rng(0).permutation(len(inputs))
    test_idx = np.sort(order[:args.test_size])
    pool_idx = np.sort(order[args.test_size:])
    test_x, test_y = inputs[test_idx], labels[test_idx]
    pool_x = inputs[pool_idx]
    print(f"loaded {len(inputs)} images: fixed test {len(test_y)}, calibration pool {len(pool_x)}", flush=True)

    # Resume: load existing results; study() skips points already computed and
    # checkpoints after every point, so no progress is lost if the run is killed.
    results = json.loads(args.out.read_text(encoding="utf-8")) if args.out.is_file() else []
    if results:
        print(f"resuming: {len(results)} model entries present "
              f"({sum(len(r['points']) for r in results)} points)", flush=True)
    with tempfile.TemporaryDirectory(prefix="edgemason-calibscale-") as directory:
        for model_name in args.models:
            print(f"[{time.strftime('%H:%M:%S')}] {model_name} ...", flush=True)
            study(model_name, test_x, test_y, pool_x, sizes, args.draws, Path(directory), results, args.out)
    print(f"[{time.strftime('%H:%M:%S')}] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
