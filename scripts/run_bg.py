import os
# Pin to single core and cap thread pools to prevent phone charger under-voltage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ORT_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_full_benchmark import run

print(f"[{time.strftime('%H:%M:%S')}] Starting stabilized phone-charger benchmark on Raspberry Pi 5...", flush=True)

out_file = PROJECT_ROOT / "reports" / "data" / "benchmark_rpi5.json"

try:
    # Run FP32, Dynamic INT8, and Static INT8
    run(
        hardware_id="rpi5",
        provider="cpu",
        priority="latency",
        out=out_file,
        include_tvm=False,
        enable_tvm_tuning=False,
    )
    print(f"[{time.strftime('%H:%M:%S')}] ALL 5 MODELS BENCHMARKED SUCCESSFULLY ON RPI 5!", flush=True)
except Exception as e:
    import traceback
    print(f"[{time.strftime('%H:%M:%S')}] ERROR OCCURRED: {e}", flush=True)
    traceback.print_exc()
