import os
# Use all 4 Pi 5 cores for both backends so ORT-vs-TVM is head-to-head.
# Requires the official 5V/5A PSU + active cooling -- a phone charger will
# brownout at this thread count, which is why this used to be pinned to 2.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TVM_NUM_THREADS"] = "4"
os.environ["ORT_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_full_benchmark import run

print(f"[{time.strftime('%H:%M:%S')}] Starting full benchmark with TVM Tuning & ONNX Runtime on Raspberry Pi 5...", flush=True)

out_file = PROJECT_ROOT / "reports" / "data" / "benchmark_rpi5_all.json"

try:
    # Run full suite: include_tvm=True and enable_tvm_tuning=True
    run(
        hardware_id="rpi5",
        provider="cpu",
        priority="latency",
        out=out_file,
        include_tvm=True,
        enable_tvm_tuning=True,
    )
    print(f"[{time.strftime('%H:%M:%S')}] ALL BENCHMARKS (TVM + ONNX) COMPLETED SUCCESSFULLY ON RPI 5!", flush=True)
except Exception as e:
    import traceback
    print(f"[{time.strftime('%H:%M:%S')}] ERROR OCCURRED: {e}", flush=True)
    traceback.print_exc()
