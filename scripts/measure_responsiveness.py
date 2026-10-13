"""Measure how responsive the server stays while a deploy is running.

Starts the real API with uvicorn, sends one /deploy, and meanwhile requests
/health and / (the dashboard) every 0.25 s. Prints a JSON summary of how long
those requests took to answer.

Usage:
    uv run python scripts/measure_responsiveness.py \
        --model models/mobilenetv2.onnx \
        --calibration models/samples/imagenetv2_mobilenetv2_256.npz
"""

import argparse
import json
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLL_INTERVAL_SECONDS = 0.25
SLOW_SECONDS = 1.0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until_up(base_url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{base_url}/health", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise RuntimeError("server did not start")


def _summary(latencies: list[float]) -> dict:
    if not latencies:
        return {"requests": 0}
    return {
        "requests": len(latencies),
        "median_ms": round(statistics.median(latencies) * 1000, 1),
        "max_ms": round(max(latencies) * 1000, 1),
        "slower_than_1s": sum(latency > SLOW_SECONDS for latency in latencies),
    }


def measure(model: Path, calibration: Path, hardware_id: str) -> dict:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(port)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_until_up(base_url)
        deploy_result: dict = {}

        def run_deploy() -> None:
            start = time.perf_counter()
            with model.open("rb") as model_file, calibration.open("rb") as calibration_file:
                response = httpx.post(
                    f"{base_url}/deploy",
                    data={"hardware_id": hardware_id, "priority": "latency", "execution_provider": "cpu"},
                    files={"file": (model.name, model_file), "calibration_file": (calibration.name, calibration_file)},
                    timeout=1800,
                )
            deploy_result["status"] = response.status_code
            deploy_result["seconds"] = round(time.perf_counter() - start, 1)
            if response.status_code == 200:
                deploy_result["winner"] = response.json()["winner"]["name"]

        deploy_thread = threading.Thread(target=run_deploy)
        deploy_thread.start()
        time.sleep(0.5)  # let the upload start

        health, page = [], []
        with httpx.Client(base_url=base_url, timeout=1800) as client:
            while deploy_thread.is_alive():
                for path, bucket in (("/health", health), ("/", page)):
                    start = time.perf_counter()
                    client.get(path)
                    bucket.append(time.perf_counter() - start)
                time.sleep(POLL_INTERVAL_SECONDS)
        deploy_thread.join()

        return {
            "model": model.name,
            "calibration": calibration.name,
            "hardware_id": hardware_id,
            "deploy": deploy_result,
            "health_during_deploy": _summary(health),
            "dashboard_during_deploy": _summary(page),
        }
    finally:
        server.terminate()
        server.wait(timeout=30)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure server responsiveness during a deploy.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "mobilenetv2.onnx")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "models" / "samples" / "imagenetv2_mobilenetv2_256.npz",
    )
    parser.add_argument("--hardware-id", default="x86_64")
    args = parser.parse_args()
    print(json.dumps(measure(args.model, args.calibration, args.hardware_id), indent=2))


if __name__ == "__main__":
    main()
