import tempfile
from pathlib import Path

from api.optimizer.validation import MAX_VALIDATION_BYTES


MAX_MODEL_BYTES = 350 * 1024 * 1024
MAX_CALIBRATION_BYTES = MAX_VALIDATION_BYTES
MAX_MODEL_NODES = 10_000
MAX_INPUT_ELEMENTS = 10_000_000

ARTIFACT_TTL_SECONDS = 60 * 60
ARTIFACT_DIR = Path(tempfile.gettempdir()) / "edgemason-artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"

VALID_PRIORITIES = {"latency", "memory", "accuracy", "balanced"}
VALID_EXECUTION_PROVIDERS = {"auto", "cpu", "cuda"}
CALIBRATION_SUFFIXES = {".npy", ".npz"}

# Per candidate: up to 10 warm-up and 100 timed runs, stopping after about
# 10 s of timing (but never before 10 timed runs) so big models stay fast.
BENCHMARK_WARMUP_RUNS = 10
BENCHMARK_TIMED_RUNS = 100
BENCHMARK_TIME_BUDGET_SECONDS = 10.0
