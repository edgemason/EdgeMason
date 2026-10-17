"""Run the full pipeline (ONNX Runtime + TVM + FP32 baseline) for several models.

For each model it calls /deploy in-process, records every candidate's results,
downloads the winning ZIP and runs its inference.py self-test. Results are
written to a JSON file after each model, so a long run can be followed and
partial results are kept.

Usage:
    uv run python scripts/run_full_benchmark.py --out reports/data/full_run_x86_64.json
"""

import argparse
import io
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

SAMPLES = PROJECT_ROOT / "models" / "samples"
DEFAULT_RUNS = [
    ("mobilenetv2.onnx", "imagenetv2_mobilenetv2_256.npz"),
    ("resnet50.onnx", "imagenetv2_mobilenetv2_256.npz"),
    # Untuned TVM needs several seconds per ResNet-152 image, so a smaller
    # file keeps accuracy scoring to minutes instead of about an hour.
    ("resnet152.onnx", "imagenetv2_mobilenetv2_50.npz"),
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


def _run_package(zip_bytes: bytes) -> dict:
    with tempfile.TemporaryDirectory(prefix="edgemason-pkg-") as directory:
        zipfile.ZipFile(io.BytesIO(zip_bytes)).extractall(directory)
        completed = subprocess.run(
            [sys.executable, "inference.py"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=600,
        )
        return {
            "files": sorted(zipfile.ZipFile(io.BytesIO(zip_bytes)).namelist()),
            "zip_mb": round(len(zip_bytes) / 2**20, 2),
            "self_test_ok": completed.returncode == 0 and "OK" in completed.stdout,
            "self_test_output": completed.stdout.strip().splitlines()[-4:],
        }


def run(hardware_id: str, provider: str, priority: str, out: Path) -> None:
    client = TestClient(app)
    report = {
        "hardware_id": hardware_id,
        "execution_provider": provider,
        "priority": priority,
        "include_tvm": True,
        "models": [],
    }
    for model_name, calibration_name in DEFAULT_RUNS:
        print(f"[{time.strftime('%H:%M:%S')}] {model_name} with {calibration_name} ...", flush=True)
        start = time.time()
        response = client.post(
            "/deploy",
            data={
                "hardware_id": hardware_id,
                "priority": priority,
                "execution_provider": provider,
                "include_tvm": "true",
            },
            files={
                "file": (model_name, (PROJECT_ROOT / "models" / model_name).read_bytes()),
                "calibration_file": (calibration_name, (SAMPLES / calibration_name).read_bytes()),
            },
        )
        seconds = round(time.time() - start, 1)
        entry = {"model": model_name, "calibration": calibration_name, "status": response.status_code,
                 "deploy_seconds": seconds}
        if response.status_code == 200:
            body = response.json()
            entry |= {
                "model_family": body["analysis"]["model_family"],
                "model_size_mb": body["analysis"]["model_size_mb"],
                "parameters": body["analysis"]["total_parameters"],
                "data_split": body["data_split"],
                "warnings": body["warnings"],
                "winner": body["winner"]["name"] if body["winner"] else None,
                "candidates": [_candidate(result) for result in body["all_results"]],
            }
            if body["download_url"]:
                entry["package"] = _run_package(client.get(body["download_url"]).content)
        else:
            entry["error"] = response.text[:500]
        report["models"].append(entry)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] {model_name} done in {seconds}s, winner={entry.get('winner')}", flush=True)
        for candidate in entry.get("candidates", []):
            top1 = candidate["top1"]
            print(
                f"    {candidate['name']:20} ok={candidate['success']!s:5} "
                f"p50={candidate['latency_p50_ms']!s:>9} ms  runs={candidate['warmup_runs']}+{candidate['timed_runs']}  "
                f"mem={candidate['memory_delta_mb']!s:>7} MB  size={candidate['model_size_mb']!s:>7} MB  "
                f"top1={'n/a' if top1 is None else f'{top1 * 100:.1f}%'}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full pipeline for several models.")
    parser.add_argument("--hardware-id", default="x86_64")
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--priority", default="latency")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "data" / "full_run_x86_64.json")
    args = parser.parse_args()
    run(args.hardware_id, args.provider, args.priority, args.out)


if __name__ == "__main__":
    main()
