#!/usr/bin/env python
"""Capture the P0.8 baseline from the provisioned assets.

Runs analysis, planning, quantization and accuracy on the real models and
writes them as JSON so a later spec can diff two runs byte for byte.

Nothing here measures latency or memory: timing is P0.9's business, and a
timing value would never be deterministic.

Usage:
    uv run python scripts/capture_baseline.py --out baseline/full
    uv run python scripts/capture_baseline.py --out baseline/quick --quick
    uv run python scripts/capture_baseline.py --out /tmp/one --models mobilenetv2
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.analyzer.model_analyzer import model_report  # noqa: E402
from api.optimizer.quantizer import Quantizer  # noqa: E402
from api.optimizer.validation import (  # noqa: E402
    evaluate_onnx,
    load_validation_data,
    split_calibration_and_test,
)
from api.planner.rule_based import (  # noqa: E402
    PlannerInput,
    generate_candidates,
)

ASSETS_MANIFEST = "assets.json"
MODEL_DIR = "models"
MODEL_SUFFIX = ".onnx"
# The labelled ImageNetV2 subset both the product and the research scripts use.
DATASET_PATH = "models/samples/imagenetv2_mobilenetv2_256.npz"

# The two small models the fast gate runs on.
QUICK_MODELS = ("squeezenet1.1-13", "mobilenetv2")

# The candidates the product can build for an ONNX Runtime model.
QUANTIZED_CANDIDATES = (
    "onnx_fp32_baseline",
    "onnx_dynamic_int8",
    "onnx_static_int8",
)

# Candidate -> the variant name reports/data/quantization_analysis.json scores
# the same artifact under. The two paths use different names for one artifact.
REPORT_VARIANT_BY_CANDIDATE = {
    "onnx_fp32_baseline": "fp32",
    "onnx_dynamic_int8": "dynamic_int8",
    "onnx_static_int8": "static_int8_per_channel",
}
# A variant the research script measured that the product cannot build, so
# there is no product artifact to reproduce it from.
UNREPRODUCIBLE_REPORT_VARIANT = "static_int8_per_tensor"
UNREPRODUCIBLE_REASON = (
    "the product's only static INT8 candidate is per-channel; "
    f"{UNREPRODUCIBLE_REPORT_VARIANT} was measured with per_channel=False by "
    "scripts/quantization_analysis.py, so no product artifact exists to compare"
)

COMMITTED_ACCURACY_SOURCE = "reports/data/quantization_analysis.json"


class CaptureError(Exception):
    """A requested asset is missing, so nothing can be captured."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture the EdgeMason baseline.")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--assets-root", type=Path, default=Path("."), help="root holding assets.json"
    )
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="NAME",
        help="bare model names to capture (default: every model in assets.json)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help=f"capture only {', '.join(QUICK_MODELS)}",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return ""
    for line in cpuinfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return ""


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _dependency_freeze() -> list[str]:
    """The installed distributions, as `pip freeze` would report them.

    `uv run pip` resolves to the system pip (CODING_STANDARDS, environment
    note), and the project venv has no pip of its own, so the venv is asked
    for explicitly via `uv pip freeze --python <this interpreter>`.
    """
    uv = shutil.which("uv")
    commands = [[sys.executable, "-m", "pip", "freeze"]]
    if uv:
        commands.append([uv, "pip", "freeze", "--python", sys.executable])
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return [line for line in result.stdout.splitlines() if line.strip()]
    return []


def _environment(assets_root: Path) -> dict:
    manifest = assets_root / ASSETS_MANIFEST
    return {
        "assets_json_sha256": _sha256(manifest) if manifest.is_file() else "",
        "cpu_model": _cpu_model(),
        "date_utc": datetime.now(UTC).isoformat(),
        "dependencies": _dependency_freeze(),
        "git_commit": _git_commit(),
        "python_version": sys.version,
    }


def _manifest_model_names(assets_root: Path) -> list[str]:
    """Every model registered in assets.json, as bare names, sorted."""
    manifest = json.loads((assets_root / ASSETS_MANIFEST).read_text(encoding="utf-8"))
    prefix = f"{MODEL_DIR}/"
    names = set()
    for asset in manifest.get("assets", []):
        relative = asset.get("path", "")
        stem = relative.removeprefix(prefix)
        if relative.startswith(prefix) and stem.endswith(MODEL_SUFFIX) and "/" not in stem:
            names.add(stem.removesuffix(MODEL_SUFFIX))
    return sorted(names)


def resolve_model_names(
    assets_root: Path,
    requested: Sequence[str] | None,
    quick: bool,
) -> list[str]:
    if requested:
        return list(requested)
    if quick:
        return list(QUICK_MODELS)
    return _manifest_model_names(assets_root)


def _require_assets(assets_root: Path, model_names: Sequence[str]) -> Path:
    """Fail before anything is written when an asset is missing."""
    required = [assets_root / MODEL_DIR / f"{name}{MODEL_SUFFIX}" for name in model_names]
    required.append(assets_root / DATASET_PATH)
    missing = [path.relative_to(assets_root).as_posix() for path in required if not path.is_file()]
    if missing:
        raise CaptureError(f"missing asset(s): {', '.join(missing)}")
    return assets_root / DATASET_PATH


def _analyse(model_path: Path) -> dict:
    report = json.loads(model_report(str(model_path)))
    # `filename` is the path this process happened to use; it must not vary.
    report.pop("filename", None)
    return report


def _plans(analysis: dict) -> dict:
    """The plan for both calibration settings, built exactly as run_deploy does."""
    plans = {}
    for has_calibration in (False, True):
        plans[str(has_calibration).lower()] = generate_candidates(
            PlannerInput(
                model_size_mb=analysis["model_size_mb"],
                max_latency_ms=None,
                has_calibration_data=has_calibration,
                include_tvm=False,
                operator_counts=analysis["operator_types"],
            )
        )
    return plans


def _variant_map() -> dict[str, dict[str, str | None]]:
    variants: dict[str, dict[str, str | None]] = {
        report_variant: {"candidate": candidate}
        for candidate, report_variant in REPORT_VARIANT_BY_CANDIDATE.items()
    }
    variants[UNREPRODUCIBLE_REPORT_VARIANT] = {
        "candidate": None,
        "reason": UNREPRODUCIBLE_REASON,
    }
    return variants


def _reproduction() -> dict:
    """The reproduction check lives in its own step; this file records its shape."""
    return {
        "committed_source": COMMITTED_ACCURACY_SOURCE,
        "comparison_fields": [
            "model", "variant", "field", "committed", "measured", "verdict", "reason",
        ],
        "variants": _variant_map(),
        "comparisons": [],
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def capture(assets_root: Path, model_names: Sequence[str], out_dir: Path) -> list[Path]:
    """Write the six baseline files. Raises CaptureError if an asset is missing."""
    dataset_path = _require_assets(assets_root, model_names)
    split = split_calibration_and_test(load_validation_data(dataset_path))

    analysis: dict[str, dict] = {}
    plans: dict[str, dict] = {}
    quantized: dict[str, dict] = {}
    accuracy: dict[str, dict] = {}

    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="quantized-", dir=out_dir))
    try:
        for name in model_names:
            model_path = assets_root / MODEL_DIR / f"{name}{MODEL_SUFFIX}"
            report = _analyse(model_path)
            analysis[name] = report
            plans[name] = _plans(report)

            quantizer = Quantizer(str(work_dir / name))
            quantized[name] = {}
            accuracy[name] = {}
            for candidate in QUANTIZED_CANDIDATES:
                quantized_path = quantizer.quantize_for_candidate(
                    str(model_path),
                    candidate,
                    calibration_data=split.calibration,
                )
                quantized[name][candidate] = _sha256(Path(quantized_path))
                accuracy[name][candidate] = evaluate_onnx(quantized_path, split.test)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    written = [
        ("environment.json", _environment(assets_root)),
        ("analysis.json", analysis),
        ("plans.json", plans),
        ("quantized.json", quantized),
        ("accuracy.json", accuracy),
        ("reproduction.json", _reproduction()),
    ]
    paths: list[Path] = []
    for filename, payload in written:
        path = out_dir / filename
        _write_json(path, payload)
        paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    assets_root = args.assets_root.resolve()
    try:
        model_names = resolve_model_names(assets_root, args.models, args.quick)
        if not model_names:
            raise CaptureError(f"no models registered in {ASSETS_MANIFEST}")
        written = capture(assets_root, model_names, args.out)
    except CaptureError as exc:
        print(f"capture_baseline: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
