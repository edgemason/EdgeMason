"""
Rule-based planner for EdgeForge.

This file decides which ONNX deployment/quantization candidates should be tried.
It does NOT run ONNX Runtime.
It does NOT create an InferenceSession.
It does NOT benchmark anything.

Its only job:
model info + user constraints -> candidate strategy list
"""

from dataclasses import dataclass, asdict
from typing import Optional


# Strategy names — ONNX Runtime
STRATEGY_FP32_BASELINE = "onnx_fp32_baseline"
STRATEGY_DYNAMIC_INT8 = "onnx_dynamic_int8"
STRATEGY_STATIC_INT8 = "onnx_static_int8"

# Strategy names — TVM
STRATEGY_TVM_OPT3 = "tvm_opt_level_3"
STRATEGY_TVM_INT8 = "tvm_int8_quantized"


# Rule thresholds
SIZE_THRESHOLD_MB = 5.0
LATENCY_THRESHOLD_MS = 100.0


@dataclass
class PlannerInput:
    model_size_mb: float
    max_latency_ms: Optional[float] = None
    has_calibration_data: bool = False
    include_tvm: bool = False


@dataclass
class Candidate:
    name: str
    framework: str
    quantization: str
    reason: str
    requires_calibration: bool = False


def generate_candidates(planner_input: PlannerInput) -> dict:
    """
    Generate ONNX candidate strategies based on simple hardcoded rules.

    Rules:
    1. Always include FP32 baseline.
    2. If model size > 50MB, try INT8.
    3. If latency constraint < 100ms, try dynamic quantization.

    Args:
        planner_input: PlannerInput containing model size and user constraints.

    Returns:
        dict containing selected candidates and rules applied.
    """

    candidates: list[Candidate] = []
    rules_applied: list[str] = []
    skipped: list[dict] = []

    # Rule 1: Always include FP32 baseline
    candidates.append(
        Candidate(
            name=STRATEGY_FP32_BASELINE,
            framework="onnxruntime",
            quantization="fp32",
            reason="FP32 baseline is always included for comparison.",
            requires_calibration=False,
        )
    )
    rules_applied.append("Always include FP32 baseline.")

    # Calibration data enables a direct FP32/dynamic/static comparison.
    if planner_input.has_calibration_data:
        candidates.append(
            Candidate(
                name=STRATEGY_STATIC_INT8,
                framework="onnxruntime",
                quantization="static_int8",
                reason="Static INT8 is selected because calibration data is available.",
                requires_calibration=True,
            )
        )
        candidates.append(
            Candidate(
                name=STRATEGY_DYNAMIC_INT8,
                framework="onnxruntime",
                quantization="dynamic_int8",
                reason="Dynamic INT8 is included for comparison with static INT8.",
                requires_calibration=False,
            )
        )
        rules_applied.append("Calibration data provided -> compare dynamic and static INT8.")
    elif planner_input.model_size_mb > SIZE_THRESHOLD_MB:
        candidates.append(
            Candidate(
                name=STRATEGY_DYNAMIC_INT8,
                framework="onnxruntime",
                quantization="dynamic_int8",
                reason=(
                    f"Model size is {planner_input.model_size_mb:.2f}MB, "
                    f"which is greater than {SIZE_THRESHOLD_MB:.0f}MB. "
                    "Dynamic INT8 is selected because calibration data is unavailable."
                ),
                requires_calibration=False,
            )
        )
        rules_applied.append(
            f"model_size_mb > {SIZE_THRESHOLD_MB:.0f}MB -> try dynamic INT8."
        )

    # Rule 3: If latency constraint < 100ms, try dynamic quantization
    if (
        planner_input.max_latency_ms is not None
        and planner_input.max_latency_ms < LATENCY_THRESHOLD_MS
    ):
        already_added = any(c.name == STRATEGY_DYNAMIC_INT8 for c in candidates)

        if not already_added:
            candidates.append(
                Candidate(
                    name=STRATEGY_DYNAMIC_INT8,
                    framework="onnxruntime",
                    quantization="dynamic_int8",
                    reason=(
                        f"User latency constraint is {planner_input.max_latency_ms:.2f}ms, "
                        f"which is below {LATENCY_THRESHOLD_MS:.0f}ms. "
                        "Dynamic INT8 is selected as a low-cost latency-oriented candidate."
                    ),
                    requires_calibration=False,
                )
            )

        rules_applied.append(
            f"max_latency_ms < {LATENCY_THRESHOLD_MS:.0f}ms -> try dynamic INT8."
        )

    if planner_input.include_tvm:
        candidates.append(
            Candidate(
                name=STRATEGY_TVM_OPT3,
                framework="tvm",
                quantization="none",
                reason="TVM Relax compiled baseline was explicitly requested.",
                requires_calibration=False,
            )
        )
        rules_applied.append("TVM candidates explicitly enabled.")

        if planner_input.has_calibration_data:
            candidates.append(
                Candidate(
                    name=STRATEGY_TVM_INT8,
                    framework="tvm",
                    quantization="int8",
                    reason="TVM INT8 is included because calibration data is available.",
                    requires_calibration=True,
                )
            )
            rules_applied.append("Calibration data provided -> include TVM INT8 candidate.")
    else:
        skipped.append(
            {
                "strategy": STRATEGY_TVM_OPT3,
                "reason": "TVM candidates are opt-in because compilation and evaluation can be slow.",
            }
        )

    # If static INT8 was not selected, record why
    has_static_int8 = any(c.name == STRATEGY_STATIC_INT8 for c in candidates)

    if not has_static_int8:
        skipped.append(
            {
                "strategy": STRATEGY_STATIC_INT8,
                "reason": "Static INT8 was skipped because either the model was not large enough or calibration data was unavailable.",
            }
        )

    if not any(c.name == STRATEGY_TVM_INT8 for c in candidates):
        skipped.append(
            {
                "strategy": STRATEGY_TVM_INT8,
                "reason": (
                    "TVM INT8 was skipped because TVM was not enabled or "
                    "calibration data was unavailable."
                ),
            }
        )

    return {
        "planner": "rule_based",
        "input": asdict(planner_input),
        "candidate_count": len(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
        "rules_applied": rules_applied,
        "skipped": skipped,
    }
