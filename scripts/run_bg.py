import os
# Pin thread pools to 2 cores to prevent phone-charger under-voltage brownout
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TVM_NUM_THREADS"] = "2"
os.environ["ORT_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

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
