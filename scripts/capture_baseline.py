#!/usr/bin/env python
"""Capture the P0.8 baseline from the provisioned assets.

Runs analysis, planning, quantization and accuracy on the real models and
writes them as JSON so a later spec can diff two runs byte for byte.

Nothing here measures latency or memory: timing is P0.9's business, and a
timing value would never be deterministic.

A candidate the product cannot build is **recorded, not fatal**. Some real
models (anything below opset 11) get an artifact ONNX Runtime cannot load;
rather than aborting the whole capture, the failure is written into
quantized.json and accuracy.json as an error entry and summarised on stderr,
so the baseline records the limitation instead of hiding it.

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
# The accuracy numbers the research scripts committed, compared against below.
COMMITTED_DATA = "reports/data/quantization_analysis.json"

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
    "the product `Quantizer` implements per-channel static INT8 only; the "
    "per-tensor path exists only in `scripts/quantization_analysis.py` and has "
    "no product counterpart to compare against"
)

# A block the research script derives; no product function computes it.
AGREEMENT_FIELD = "agreement_fp32_vs_static_int8"
AGREEMENT_REASON = (
    "no product function computes this block; it is derived in "
    "`scripts/quantization_analysis.py` only"
)

# Timing is measured by a different path entirely, and is never deterministic.
LATENCY_FIELD = "latency_p50_ms_1thread"
LATENCY_REASON = (
    "measured in-process by `scripts/quantization_analysis.py` (10 warm-up + 50 "
    "timed runs on the default thread pool) rather than by the product's "
    "benchmark engine, and it is a timing number, so it is never deterministic "
    "and cannot be compared against a stored value"
)

# Top-1 is computed the same way on both sides (argmax over the same logits).
TOP1_FIELD = "top1"
TOP1_MEASURED = "top1_accuracy"
# Top-5 is NOT: static INT8 quantisation collapses class groups onto identical
# logits, so the 5th-place class is ambiguous and the two scorers pick
# different members of the tied group.
TOP5_FIELD = "top5"
TOP5_MEASURED = "top5_accuracy"
TOP5_REASON = (
    "scored with different tie-breaking (product `evaluate_onnx` uses "
    "`np.argpartition`, `scripts/quantization_analysis.py` uses `np.argsort`) and "
    "different thread count (product pins `intra_op_num_threads=1`, research "
    "script uses all cores); static INT8 quantisation collapses class groups "
    "onto identical logits so 5th-place class membership is ambiguous - e.g. "
    "squeezenet1.1-13: argsort/1-thread 0.6033519553072626, argsort/N-thread "
    "0.6089385474860335 (the committed value), argpartition/1-thread "
    "0.6368715083798883, 123/179 images differ by exact ties"
)

# `size_mb` is the same file measured the same way on both sides.
SIZE_FIELD = "size_mb"

# Fields the committed data carries per (model, variant).
COMPARABLE_FIELDS = (TOP1_FIELD, TOP5_FIELD, SIZE_FIELD)

MISSING_MODEL_REASON = (
    f"no committed value in {COMMITTED_DATA} for this model"
)

# Placeholders that keep a recorded failure identical between two runs: the
# work directory is a fresh mkdtemp every time, so a raw path would make
# reproduction.json (and quantized.json) differ run to run.
WORK_DIR_TOKEN = "<work_dir>"
QUANTIZED_TOKEN = "<quantized>"
MODEL_TOKEN = "<model>"


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
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="abort on the first candidate the product cannot build, instead of "
             "recording the failure and continuing",
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


def _failure_message(exc: Exception, quantized_path: str, work_dir: Path, model_path: Path) -> str:
    """A stable one-line record of why a candidate could not be built.

    The raw ORT text names the file it failed to load, and the work directory
    is a fresh mkdtemp on every run, so the paths are replaced by fixed
    tokens. Everything that explains the cause is kept verbatim.
    """
    text = str(exc)
    for path, token in (
        (quantized_path, QUANTIZED_TOKEN),
        (str(work_dir), WORK_DIR_TOKEN),
        (str(model_path), MODEL_TOKEN),
    ):
        if path:
            text = text.replace(path, token)
    return f"{type(exc).__name__}: {text}"


def _committed_variants(assets_root: Path) -> dict[str, dict[str, dict]]:
    """The committed report, keyed by bare model name. Empty if absent."""
    path = assets_root / COMMITTED_DATA
    if not path.is_file():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    return {record["model"].removesuffix(MODEL_SUFFIX): record["variants"] for record in records}


def _comparison(model: str, variant: str, field: str, verdict: str, committed=None,
                measured=None, reason: str | None = None) -> dict:
    entry = {"model": model, "variant": variant, "field": field, "verdict": verdict}
    entry["committed"] = committed
    entry["measured"] = measured
    entry["reason"] = reason
    return entry


def _reproduction(assets_root: Path, measured: dict[str, dict[str, dict]]) -> dict:
    """Compare this run's numbers with the committed research data.

    Only fields computed the same way on both sides are compared; every other
    field is marked `not_comparable` with the reason. `measured` maps a model
    name to a candidate name to {"sha256", "size_mb", "accuracy", "error"}.
    """
    committed = _committed_variants(assets_root)
    comparisons: list[dict] = []

    for model in sorted(measured):
        for candidate, variant in sorted(REPORT_VARIANT_BY_CANDIDATE.items()):
            result = measured[model][candidate]
            error = result["error"]
            scores = result["accuracy"]
            report_variant = committed.get(model, {}).get(variant)
            if report_variant is None:
                comparisons.append(_comparison(
                    model, variant, "*", "not_comparable", reason=MISSING_MODEL_REASON,
                ))
                continue
            if error is not None or scores is None:
                comparisons.append(_comparison(
                    model, variant, "*", "not_comparable",
                    reason=f"the product could not build a loadable artifact for this "
                           f"candidate: {error}",
                ))
                continue
            for field in COMPARABLE_FIELDS:
                if field == TOP5_FIELD:
                    comparisons.append(_comparison(
                        model, variant, TOP5_FIELD, "not_comparable",
                        committed=report_variant[TOP5_FIELD],
                        measured=scores[TOP5_MEASURED], reason=TOP5_REASON,
                    ))
                    continue
                if field == TOP1_FIELD:
                    new, old = scores[TOP1_MEASURED], report_variant[TOP1_FIELD]
                else:
                    new, old = result["size_mb"], report_variant[SIZE_FIELD]
                comparisons.append(_comparison(
                    model, variant, field,
                    "reproduces" if new == old else "differs", committed=old, measured=new,
                ))
            comparisons.append(_comparison(
                model, variant, LATENCY_FIELD, "not_comparable",
                committed=report_variant[LATENCY_FIELD], reason=LATENCY_REASON,
            ))

    for model in sorted(committed):
        comparisons.append(_comparison(
            model, UNREPRODUCIBLE_REPORT_VARIANT, "*", "not_comparable",
            reason=UNREPRODUCIBLE_REASON,
        ))
        comparisons.append(_comparison(
            model, AGREEMENT_FIELD, "*", "not_comparable", reason=AGREEMENT_REASON,
        ))

    return {
        "committed_source": COMMITTED_DATA,
        "comparison_fields": [
            "model", "variant", "field", "committed", "measured", "verdict", "reason",
        ],
        "variants": _variant_map(),
        "comparisons": comparisons,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def capture(
    assets_root: Path,
    model_names: Sequence[str],
    out_dir: Path,
    fail_fast: bool = False,
) -> tuple[list[Path], list[tuple[str, str, str]]]:
    """Write the six baseline files.

    Returns the written paths and one (model, candidate, error) triple per
    candidate the product could not build. Raises CaptureError if an asset is
    missing.
    """
    dataset_path = _require_assets(assets_root, model_names)
    split = split_calibration_and_test(load_validation_data(dataset_path))

    analysis: dict[str, dict] = {}
    plans: dict[str, dict] = {}
    quantized: dict[str, dict] = {}
    accuracy: dict[str, dict] = {}
    measured: dict[str, dict[str, dict]] = {}
    failures: list[tuple[str, str, str]] = []

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
            measured[name] = {}
            for candidate in QUANTIZED_CANDIDATES:
                quantized_path = str(work_dir / name / f"{candidate}.onnx")
                try:
                    built = quantizer.quantize_for_candidate(
                        str(model_path),
                        candidate,
                        calibration_data=split.calibration,
                    )
                    quantized_path = built
                    digest = _sha256(Path(built))
                    size_mb = Path(built).stat().st_size / 2**20
                    scores = evaluate_onnx(built, split.test)
                except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                    message = _failure_message(exc, quantized_path, work_dir, model_path)
                    quantized[name][candidate] = {"sha256": None, "error": message}
                    accuracy[name][candidate] = {"error": message}
                    measured[name][candidate] = {
                        "sha256": None, "size_mb": None, "accuracy": None, "error": message,
                    }
                    failures.append((name, candidate, message))
                    if fail_fast:
                        raise
                    continue
                quantized[name][candidate] = {"sha256": digest, "error": None}
                accuracy[name][candidate] = scores
                measured[name][candidate] = {
                    "sha256": digest, "size_mb": size_mb, "accuracy": scores, "error": None,
                }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    written = [
        ("environment.json", _environment(assets_root)),
        ("analysis.json", analysis),
        ("plans.json", plans),
        ("quantized.json", quantized),
        ("accuracy.json", accuracy),
        ("reproduction.json", _reproduction(assets_root, measured)),
    ]
    paths: list[Path] = []
    for filename, payload in written:
        path = out_dir / filename
        _write_json(path, payload)
        paths.append(path)
    return paths, failures


def _report_failures(failures: Sequence[tuple[str, str, str]]) -> None:
    """Say plainly on stderr what the capture could not do."""
    if not failures:
        return
    print(
        f"capture_baseline: {len(failures)} candidate(s) could not be built by the "
        "product and were recorded as errors, not silently skipped:",
        file=sys.stderr,
    )
    for name, candidate, message in failures:
        print(f"  {name} / {candidate}: {message}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    assets_root = args.assets_root.resolve()
    try:
        model_names = resolve_model_names(assets_root, args.models, args.quick)
        if not model_names:
            raise CaptureError(f"no models registered in {ASSETS_MANIFEST}")
        written, failures = capture(assets_root, model_names, args.out, args.fail_fast)
    except CaptureError as exc:
        print(f"capture_baseline: {exc}", file=sys.stderr)
        return 1
    _report_failures(failures)
    for path in written:
        print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
