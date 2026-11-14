"""Run the benchmark pipeline across models on edge hardware (Raspberry Pi, Jetson, x86).

For each model it calls the deploy pipeline, records every candidate's telemetry
(p50, p95 latency, throughput FPS, memory delta, model size, and accuracy),
and saves both structured JSON and tabular CSV datasets.

Usage:
    python scripts/run_full_benchmark.py --hardware-id rpi5 --provider cpu --out reports/data/benchmark_rpi5.json
"""

import argparse
import io
import json
from pathlib import Path
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
SAMPLES_DIR = MODELS_DIR / "samples"

DEFAULT_5_MODELS = [
    ("mobilenetv2.onnx", "calibration_50.npz"),
    ("resnet18.onnx", "calibration_50.npz"),
    ("resnet50.onnx", "calibration_50.npz"),
    ("shufflenetv2.onnx", "calibration_50.npz"),
    ("squeezenet.onnx", "calibration_50.npz"),
]


def _candidate(result: dict) -> dict:
    accuracy = result.get("accuracy") or {}
    keys = [
        "name", "framework", "success", "provider", "error",
        "latency_p50_ms", "latency_p95_ms", "estimated_latency_p50_ms",
        "throughput_fps", "memory_delta_mb", "estimated_memory_mb",
        "model_size_mb", "warmup_runs", "timed_runs",
    ]
    return {key: result.get(key) for key in keys} | {
        "top1": accuracy.get("top1_accuracy"),
        "top5": accuracy.get("top5_accuracy"),
        "test_samples": accuracy.get("sample_count"),
    }


def export_csv(report: dict, csv_path: Path) -> None:
    rows = []
    hardware_id = report.get("hardware_id", "unknown")
    for m in report.get("models", []):
        for c in m.get("candidates", []):
            rows.append({
                "hardware": hardware_id,
                "model": m.get("model"),
                "model_size_mb": m.get("model_size_mb"),
                "parameters": m.get("parameters"),
                "model_family": m.get("model_family"),
                "strategy": c.get("name"),
                "framework": c.get("framework"),
                "success": c.get("success"),
                "latency_p50_ms": c.get("latency_p50_ms"),
                "latency_p95_ms": c.get("latency_p95_ms"),
                "throughput_fps": c.get("throughput_fps"),
                "memory_delta_mb": c.get("memory_delta_mb"),
                "top1_accuracy": c.get("top1"),
                "top5_accuracy": c.get("top5"),
                "winner": (m.get("winner") == c.get("name")) if m.get("winner") else False,
            })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Exported tabular dataset: {csv_path} ({len(df)} candidate records)", flush=True)


def run(
    hardware_id: str,
    provider: str,
    priority: str,
    out: Path,
    include_tvm: bool = True,
    enable_tvm_tuning: bool = False,
) -> None:
    client = TestClient(app)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_suffix(".csv")

    report = {
        "hardware_id": hardware_id,
        "execution_provider": provider,
        "priority": priority,
        "include_tvm": include_tvm,
        "enable_tvm_tuning": enable_tvm_tuning,
        "models": [],
    }

    for model_name, calibration_name in DEFAULT_5_MODELS:
        model_file = MODELS_DIR / model_name
        calib_file = SAMPLES_DIR / calibration_name

        if not model_file.is_file():
            print(f"Skipping {model_name}: file not found at {model_file}", flush=True)
            continue
        if not calib_file.is_file():
            print(f"Warning: calibration {calibration_name} not found at {calib_file}", flush=True)

        print(f"\n[{time.strftime('%H:%M:%S')}] Benchmarking {model_name} on {hardware_id} ...", flush=True)
        start = time.time()

        deploy_data = {
            "hardware_id": hardware_id,
            "priority": priority,
            "execution_provider": provider,
            "include_tvm": str(include_tvm).lower(),
            "enable_tvm_tuning": str(enable_tvm_tuning).lower(),
        }
        deploy_files = {
            "file": (model_name, model_file.read_bytes()),
        }
        if calib_file.is_file():
            deploy_files["calibration_file"] = (calibration_name, calib_file.read_bytes())

        response = client.post("/deploy", data=deploy_data, files=deploy_files)
        seconds = round(time.time() - start, 1)

        entry = {
            "model": model_name,
            "calibration": calibration_name if calib_file.is_file() else None,
            "status": response.status_code,
            "deploy_seconds": seconds,
        }

        if response.status_code == 200:
            body = response.json()
            entry |= {
                "model_family": body["analysis"]["model_family"],
                "model_size_mb": body["analysis"]["model_size_mb"],
                "parameters": body["analysis"]["total_parameters"],
                "data_split": body.get("data_split", {}),
                "warnings": body.get("warnings", []),
                "winner": body["winner"]["name"] if body.get("winner") else None,
                "candidates": [_candidate(result) for result in body.get("all_results", [])],
            }
        else:
            entry["error"] = response.text[:500]

        report["models"].append(entry)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        export_csv(report, csv_path)

        print(f"[{time.strftime('%H:%M:%S')}] {model_name} completed in {seconds}s | Winner: {entry.get('winner')}", flush=True)
        for cand in entry.get("candidates", []):
            top1 = cand["top1"]
            top1_str = f"{top1 * 100:.1f}%" if top1 is not None else "n/a"
            print(
                f"    -> {cand['name']:22} ok={str(cand['success']):<5} "
                f"p50={str(cand['latency_p50_ms']):>7} ms  "
                f"fps={str(cand['throughput_fps']):>6}  "
                f"ram_delta={str(cand['memory_delta_mb']):>6} MB  "
                f"size={str(cand['model_size_mb']):>6} MB  "
                f"top1={top1_str}",
                flush=True,
            )

    print(f"\nAll benchmark runs completed! JSON: {out} | CSV: {csv_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full pipeline for 5 models on hardware.")
    parser.add_argument("--hardware-id", default="rpi5")
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--priority", default="latency")
    parser.add_argument("--include-tvm", action="store_true", default=True)
    parser.add_argument("--enable-tvm-tuning", action="store_true", default=False)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "reports" / "data" / "benchmark_rpi5.json",
    )
    args = parser.parse_args()
    run(
        args.hardware_id,
        args.provider,
        args.priority,
        args.out,
        args.include_tvm,
        args.enable_tvm_tuning,
    )


if __name__ == "__main__":
    main()
