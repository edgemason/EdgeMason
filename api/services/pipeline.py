"""Deploy pipeline: plan candidates, quantize, benchmark, rank, package."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from api.analyzer.model_analyzer import model_report
from api.config import (
    BENCHMARK_TIME_BUDGET_SECONDS,
    BENCHMARK_TIMED_RUNS,
    BENCHMARK_WARMUP_RUNS,
)
from api.optimizer.benchmark_engine import benchmark
from api.optimizer.quantizer import Quantizer
from api.optimizer.validation import (
    ValidationData,
    data_warnings,
    evaluate_onnx,
    load_calibration_inputs,
    load_validation_data,
    split_calibration_and_test,
)
from api.packager.generator import create_deployment_package
from api.planner.rule_based import PlannerInput, generate_candidates
from api.services.artifacts import new_artifact_path
from api.services.hardware import estimate_on_target
from api.services.model_validation import validate_model
from api.services.ranking import rank_results


@dataclass
class DeployRequest:
    model_path: Path
    calibration_path: Optional[Path]
    hardware_id: str
    max_latency_ms: Optional[float]
    max_memory_mb: Optional[float]
    priority: str
    work_dir: Path
    requested_provider: str
    execution_provider: str
    include_tvm: bool


def _model_input_name(model_path: str) -> str:
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    return session.get_inputs()[0].name


def _evaluate_candidate(
    candidate: dict,
    request: DeployRequest,
    calibration_inputs: Optional[np.ndarray],
    validation: Optional[ValidationData],
) -> dict:
    try:
        quantized_path = Quantizer(str(request.work_dir / candidate["name"])).quantize_for_candidate(
            str(request.model_path),
            candidate["name"],
            calibration_data=calibration_inputs,
            hardware_id=request.hardware_id,
        )
        is_tvm = candidate.get("framework") == "tvm" or str(quantized_path).endswith(
            (".tar", ".so")
        )
        input_data = None
        if is_tvm:
            from api.optimizer.validation import evaluate_tvm

            evaluate = evaluate_tvm
        else:
            if calibration_inputs is not None:
                input_data = {
                    _model_input_name(quantized_path): np.asarray(calibration_inputs[:1])
                }
            evaluate = evaluate_onnx
        bench = json.loads(
            benchmark(
                quantized_path,
                input_data=input_data,
                warmup_runs=BENCHMARK_WARMUP_RUNS,
                benchmark_runs=BENCHMARK_TIMED_RUNS,
                time_budget_seconds=BENCHMARK_TIME_BUDGET_SECONDS,
                hardware_id=request.hardware_id,
                execution_provider=request.execution_provider,
            )
        )
        accuracy = evaluate(quantized_path, validation) if validation else None
        estimate = None
        if bench.get("success"):
            estimate = estimate_on_target(
                request.hardware_id,
                candidate.get("quantization", "fp32"),
                bench["latency_ms"]["p50"],
                bench["memory_mb"]["delta"],
            )
        return {
            "candidate": candidate,
            "benchmark": bench,
            "estimate": estimate,
            "accuracy": accuracy,
            "model_path": quantized_path,
            "provider": bench.get("provider"),
            "error": bench.get("error"),
        }
    except Exception as exc:
        return {
            "candidate": candidate,
            "benchmark": {"success": False},
            "estimate": None,
            "accuracy": None,
            "model_path": None,
            "provider": None,
            "error": str(exc),
        }


# ONNX Runtime's error when a model needs CPU-only operators but CPU fallback is off.
CPU_ONLY_ERROR_MARKERS = (
    "fallback to CPU EP has been explicitly disabled",
    "assigned to the default CPU EP",
)


def gpu_warnings(results: list[dict], execution_provider: str) -> list[str]:
    """Explain candidates that could not run because they need the CPU.

    On CUDA every candidate must run fully on the GPU, so the comparison stays
    fair. INT8 models often contain operators that only exist on the CPU.
    """
    if execution_provider != "cuda":
        return []
    return [
        f"{result['candidate']['name']} cannot run on the GPU: some of its "
        f"{result['candidate'].get('quantization', '')} operations are only supported "
        "on the CPU, so it was not compared. Choose the CPU execution provider to "
        "compare it."
        for result in results
        if not result["benchmark"].get("success")
        and any(marker in (result["error"] or "") for marker in CPU_ONLY_ERROR_MARKERS)
    ]


def _public_result(result: dict) -> dict:
    bench = result["benchmark"]
    estimate = result["estimate"] or {}
    return {
        "name": result["candidate"]["name"],
        "framework": result["candidate"].get("framework", "onnxruntime"),
        "success": bench.get("success", False),
        "error": result["error"],
        "latency_p50_ms": bench.get("latency_ms", {}).get("p50"),
        "latency_p95_ms": bench.get("latency_ms", {}).get("p95"),
        "warmup_runs": bench.get("warmup_runs"),
        "timed_runs": bench.get("benchmark_runs"),
        "latency_avg_ms": bench.get("latency_ms", {}).get("avg"),
        "throughput_fps": bench.get("throughput", {}).get("fps"),
        "memory_delta_mb": bench.get("memory_mb", {}).get("delta"),
        "model_size_mb": bench.get("model_size", {}).get("mb"),
        "estimated_latency_p50_ms": estimate.get("latency_p50_ms"),
        "estimated_memory_mb": estimate.get("memory_mb"),
        "latency_scaling_factor": estimate.get("latency_scaling_factor"),
        "provider": result["provider"],
        "accuracy": result["accuracy"],
    }


def run_deploy(request: DeployRequest) -> dict:
    validate_model(request.model_path)
    analysis = json.loads(model_report(str(request.model_path)))
    calibration_path = request.calibration_path
    calibration_inputs = None
    validation = None
    warnings: list[str] = []
    if calibration_path and calibration_path.suffix.lower() == ".npz":
        # Labeled data: calibrate on one part, score accuracy on the rest.
        split = split_calibration_and_test(load_validation_data(calibration_path))
        calibration_inputs, validation = split.calibration, split.test
        warnings = data_warnings(len(calibration_inputs), len(validation.inputs))
    elif calibration_path:
        # Unlabeled .npy: no accuracy to score, so all of it calibrates.
        calibration_inputs = load_calibration_inputs(calibration_path)
        warnings = data_warnings(len(calibration_inputs), None)
    if request.priority == "accuracy" and validation is None:
        raise ValueError("accuracy priority requires labeled .npz validation data")

    plan = generate_candidates(
        PlannerInput(
            model_size_mb=analysis["model_size_mb"],
            max_latency_ms=request.max_latency_ms,
            has_calibration_data=calibration_path is not None,
            include_tvm=request.include_tvm,
            operator_counts=analysis["operator_types"],
        )
    )
    warnings = plan["warnings"] + warnings
    results = [
        _evaluate_candidate(candidate, request, calibration_inputs, validation)
        for candidate in plan["candidates"]
    ]
    warnings += gpu_warnings(results, request.execution_provider)
    ranked = rank_results(
        results,
        request.priority,
        request.max_latency_ms,
        request.max_memory_mb,
    )
    winner = ranked[0] if ranked else None

    artifact_id = None
    if winner:
        artifact_id, artifact_path = new_artifact_path()
        create_deployment_package(
            model_path=winner["model_path"],
            benchmark_report={
                **winner["benchmark"],
                "hardware_estimate": winner["estimate"],
                "accuracy": winner["accuracy"],
            },
            hardware_id=request.hardware_id,
            output_path=str(artifact_path),
            candidate=winner["candidate"],
        )

    return {
        "hardware_target": request.hardware_id,
        "execution_provider": request.execution_provider,
        "constraints": {
            "max_latency_ms": request.max_latency_ms,
            "max_memory_mb": request.max_memory_mb,
            "priority": request.priority,
            "execution_provider": request.requested_provider,
            "include_tvm": request.include_tvm,
        },
        "analysis": {
            "model_size_mb": analysis["model_size_mb"],
            "total_parameters": analysis["total_parameters"],
            "model_family": plan["model_family"],
        },
        "plan": {
            "rules_applied": plan["rules_applied"],
            "skipped": plan["skipped"],
        },
        "data_split": {
            "calibration_samples": len(calibration_inputs) if calibration_inputs is not None else 0,
            "test_samples": len(validation.inputs) if validation is not None else 0,
        },
        "warnings": warnings,
        "candidates_explored": len(results),
        "all_results": [_public_result(result) for result in results],
        "winner": _public_result(winner) if winner else None,
        "download_url": f"/download/{artifact_id}" if artifact_id else None,
    }
