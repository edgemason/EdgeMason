import copy
import difflib
import json
import os
import re
import tempfile
from pathlib import Path

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    r"|^[0-9a-f]{32}$",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(r"^[a-zA-Z0-9_.-]+==\d+\.\d+\.\d+")
_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$"
)


def _is_path(value):
    if value.startswith("/"):
        return True
    tmp = tempfile.gettempdir()
    return bool(tmp) and tmp in value


def _replace(obj):
    if isinstance(obj, dict):
        return {
            key: "<SYSTEM_INFO>" if key == "system_info" else _replace(value)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [_replace(item) for item in obj]
    if isinstance(obj, str):
        if _is_path(obj):
            return "<PATH>"
        if _UUID_RE.match(obj):
            return "<UUID>"
        if _VERSION_RE.match(obj):
            return "<VERSION>"
        if _TIMESTAMP_RE.match(obj):
            return "<TIMESTAMP>"
        return obj
    return obj


def normalise(obj):
    """Deep-copy obj with run-specific values replaced by placeholders."""
    return _replace(copy.deepcopy(obj))


GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def assert_matches_golden(name, data):
    """Compare ``data`` (normalised) against ``tests/golden/<name>.json``.

    Missing golden is written and the test fails asking for review.
    ``UPDATE_GOLDEN=1`` rewrites the golden and passes.
    """
    golden_path = GOLDEN_DIR / f"{name}.json"
    normalised = normalise(data)
    desired = json.dumps(normalised, indent=2, sort_keys=True)

    if os.environ.get("UPDATE_GOLDEN") == "1":
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(desired, encoding="utf-8")
        return None

    if not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(desired, encoding="utf-8")
        raise AssertionError(f"golden {name} created — review and commit it")

    actual_text = golden_path.read_text(encoding="utf-8")
    if actual_text == desired:
        return None

    diff = "".join(
        difflib.unified_diff(
            actual_text.splitlines(keepends=True),
            desired.splitlines(keepends=True),
            fromfile=str(golden_path),
            tofile=f"{name} (actual)",
        )
    )
    raise AssertionError(diff)


BENCHMARK_MEASUREMENT_KEYS = {
    # flat keys from deploy responses
    "throughput_fps",
    "latency_p50_ms", "latency_p95_ms", "latency_min_ms", "latency_max_ms",
    "latency_mean_ms", "latency_avg_ms", "latency_std_ms",
    "total_time_ms",
    "fps",
    # memory — both naming conventions (QUIRK)
    "baseline", "peak", "delta",
    "baseline_mb", "peak_mb", "delta_mb",
    "memory_delta_mb",
    # estimates derived from measurements
    "estimated_latency_p50_ms", "estimated_latency_avg_ms",
    "estimated_memory_mb", "estimated_throughput_fps",
    # latency_ms subkeys (benchmark engine nests percentiles here)
    "avg", "min", "max", "std",
    "p25", "p50", "p75", "p90", "p95", "p99",
    # model_size.kb can round differently
    "kb",
}


def strip_measurements(obj, *, _keys=BENCHMARK_MEASUREMENT_KEYS):
    """Deep-copy obj with live measurement values replaced by '<MEASUREMENT>'.

    Only replaces values whose key is in BENCHMARK_MEASUREMENT_KEYS and whose
    value is a number (int or float). Everything else passes through unchanged.
    """
    if isinstance(obj, dict):
        return {
            k: ("<MEASUREMENT>" if k in _keys and isinstance(v, (int, float)) else strip_measurements(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [strip_measurements(item) for item in obj]
    return obj
