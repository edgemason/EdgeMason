"""TVM (Relax) benchmark for GitHub Actions runners.

Companion to run_ci_benchmark.py: same CI runners, same 5 models, same
Imagenette images, but measures Apache TVM instead of ONNX Runtime -- untuned
(default Relax codegen) and lightly MetaSchedule-tuned (CI-sized trial
budget; the full 1024-trial protocol from P2.2 runs on the physical Pi 5
separately, see profiles/tvm_tuning/).

Reuses the project's real pipeline (api.optimizer.tvm_runtime,
api.optimizer.quantizer.TVM_TARGETS) rather than reimplementing TVM calls, so
these numbers reflect the same code path the FastAPI /deploy endpoint uses.

Usage:
    python scripts/run_ci_benchmark_tvm.py --hardware-tag github_x86_64 --max-trials 32
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.optimizer.quantizer import TVM_TARGETS
from api.optimizer.tvm_runtime import compile_onnx, load_vm, run_vm
from scripts.run_ci_benchmark import (
    MODEL_CONSTRUCTORS,
    export_models,
    preprocess_image,
)

os.environ.setdefault("STRICT_TVM", "1")


def collect_image_paths(dataset_dir: str, num_images: int) -> list[str]:
    paths = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(root, f))
    if not paths:
        raise SystemExit(f"No images found under {dataset_dir}")
    return paths[:num_images]


def benchmark_candidate(tar_path: str, selected_paths: list[str]) -> dict:
    size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    vm, device = load_vm(tar_path)
    order = ["input"]

    warm_img = preprocess_image(Image.open(selected_paths[0]).convert("RGB"))
    for _ in range(15):
        run_vm(vm, device, {"input": warm_img}, order)

    latencies = []
    t_start = time.perf_counter()
    for p in selected_paths:
        img = Image.open(p).convert("RGB")
        x = preprocess_image(img)
        t0 = time.perf_counter()
        run_vm(vm, device, {"input": x}, order)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    total_sec = time.perf_counter() - t_start
    return {
        "model_size_mb": round(size_mb, 2),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 4),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 4),
        "throughput_fps": round(len(selected_paths) / total_sec, 2),
        "total_time_sec": round(total_sec, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TVM Relax benchmark for CI runners.")
    parser.add_argument("--hardware-tag", required=True)
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument(
        "--max-trials",
        type=int,
        default=32,
        help="MetaSchedule trial budget for the 'tuned' candidate",
    )
    parser.add_argument(
        "--skip-tuned",
        action="store_true",
        help="Skip the tuned candidate entirely. Currently needed: tune_relax's "
        "TuneContext constructor segfaults/TypeErrors in apache-tvm 0.25-0.27 "
        "regardless of inputs -- a confirmed upstream bug, not fixable here. "
        "A segfault bypasses Python's try/except, so it can take the whole job "
        "down; don't rely on per-candidate error handling to contain it.",
    )
    parser.add_argument(
        "--models", default="mobilenetv2,shufflenetv2,squeezenet,resnet18,resnet50"
    )
    parser.add_argument("--dataset-dir", default="dataset/imagenette2-160/val")
    args = parser.parse_args()

    os.makedirs("reports/data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("temp_models", exist_ok=True)

    target = TVM_TARGETS.get(args.hardware_tag)
    if target is None:
        raise SystemExit(f"No TVM_TARGETS entry for hardware_tag={args.hardware_tag!r}")
    print(f"TVM target for {args.hardware_tag}: {target}")

    print("[1/3] Exporting ONNX models (skipped if already present)...")
    export_models()

    selected_paths = collect_image_paths(args.dataset_dir, args.num_images)
    print(f"Benchmarking with {len(selected_paths)} images")

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    results = []

    for name in model_names:
        if name not in MODEL_CONSTRUCTORS:
            print(f"Skipping unknown model {name}")
            continue
        onnx_path = f"models/{name}.onnx"
        if not os.path.exists(onnx_path):
            print(f"Skipping {name}: {onnx_path} not found")
            continue

        print(f"\n=== {name}: compiling TVM untuned ===")
        untuned_tar = f"temp_models/{name}_tvm_untuned_{args.hardware_tag}.tar"
        t0 = time.time()
        try:
            compile_onnx(untuned_tar, onnx_path, target, tune=False)
            compile_sec = round(time.time() - t0, 1)
            row = {
                "hardware": args.hardware_tag,
                "model": f"{name}.onnx",
                "strategy": "tvm_untuned",
                "framework": "tvm",
                "tuning_trials": 0,
                "compile_sec": compile_sec,
            }
            row.update(benchmark_candidate(untuned_tar, selected_paths))
            print(
                f"[{name.upper()}] tvm_untuned | p50: {row['latency_p50_ms']:.2f} ms | "
                f"FPS: {row['throughput_fps']:.2f} | compile: {compile_sec}s"
            )
            results.append(row)
        except Exception as e:
            print(f"[{name.upper()}] tvm_untuned FAILED: {e}")
            results.append(
                {
                    "hardware": args.hardware_tag,
                    "model": f"{name}.onnx",
                    "strategy": "tvm_untuned",
                    "framework": "tvm",
                    "error": str(e),
                }
            )

        if args.skip_tuned:
            continue

        print(f"\n=== {name}: compiling TVM tuned ({args.max_trials} trials) ===")
        tuned_tar = f"temp_models/{name}_tvm_tuned_{args.hardware_tag}.tar"
        tuning_dir = f"profiles/tvm_tuning/ci_{args.hardware_tag}/{name}"
        t0 = time.time()
        try:
            compile_onnx(
                tuned_tar,
                onnx_path,
                target,
                tune=True,
                tuning_database_path=tuning_dir,
                max_tuning_trials=args.max_trials,
            )
            compile_sec = round(time.time() - t0, 1)
            row = {
                "hardware": args.hardware_tag,
                "model": f"{name}.onnx",
                "strategy": "tvm_tuned",
                "framework": "tvm",
                "tuning_trials": args.max_trials,
                "compile_sec": compile_sec,
            }
            row.update(benchmark_candidate(tuned_tar, selected_paths))
            print(
                f"[{name.upper()}] tvm_tuned | p50: {row['latency_p50_ms']:.2f} ms | "
                f"FPS: {row['throughput_fps']:.2f} | tuning+compile: {compile_sec}s"
            )
            results.append(row)
        except Exception as e:
            print(f"[{name.upper()}] tvm_tuned FAILED: {e}")
            results.append(
                {
                    "hardware": args.hardware_tag,
                    "model": f"{name}.onnx",
                    "strategy": "tvm_tuned",
                    "framework": "tvm",
                    "error": str(e),
                }
            )

    df = pd.DataFrame(results)
    out_csv = f"reports/data/benchmark_tvm_{args.hardware_tag}.csv"
    df.to_csv(out_csv, index=False)
    print("\n" + "=" * 70)
    print(f"TVM {args.hardware_tag.upper()} BENCHMARK COMPLETE -> {out_csv}")
    print("=" * 70)
    print(df.to_string())


if __name__ == "__main__":
    main()
