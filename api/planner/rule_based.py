"""
Rule-based planner for EdgeForge.

This file decides which ONNX deployment/quantization candidates should be tried.
It does NOT run ONNX Runtime.
It does NOT create an InferenceSession.
It does NOT benchmark anything.

Its only job:
model info + user constraints -> candidate strategy list
"""

from dataclasses import asdict, dataclass, field
from typing import Optional


# Strategy names — ONNX Runtime
STRATEGY_FP32_BASELINE = "onnx_fp32_baseline"
STRATEGY_DYNAMIC_INT8 = "onnx_dynamic_int8"
STRATEGY_STATIC_INT8 = "onnx_static_int8"

# Strategy names — TVM
STRATEGY_TVM_OPT3 = "tvm_opt_level_3"
STRATEGY_TVM_INT8 = "tvm_int8_quantized"

# Quantization methods, shared by every framework
QUANT_FP32 = "fp32"
QUANT_DYNAMIC_INT8 = "dynamic_int8"
QUANT_STATIC_INT8 = "static_int8"


# Rule thresholds
SIZE_THRESHOLD_MB = 5.0
LATENCY_THRESHOLD_MS = 100.0

# Model families, decided from operator counts
FAMILY_CNN = "cnn"
FAMILY_MATMUL = "matmul"
FAMILY_UNKNOWN = "unknown"
CONV_OPS = {"Conv", "ConvTranspose"}
MATMUL_OPS = {"MatMul", "Gemm"}


@dataclass
class PlannerInput:
    model_size_mb: float
    max_latency_ms: Optional[float] = None
    has_calibration_data: bool = False
    include_tvm: bool = False
    # Operator type -> count, as reported by the model analyzer.
    operator_counts: dict = field(default_factory=dict)


@dataclass
class Candidate:
    name: str
    framework: str
    quantization: str
    reason: str
    requires_calibration: bool = False


def model_family(operator_counts: dict) -> str:
    """Conv-heavy models are CNNs; MatMul/Gemm-heavy ones are transformer-style.

    A CNN usually ends in one Gemm (the classifier), so the larger count wins.
    """
    conv = sum(operator_counts.get(op, 0) for op in CONV_OPS)
    matmul = sum(operator_counts.get(op, 0) for op in MATMUL_OPS)
    if conv == 0 and matmul == 0:
        return FAMILY_UNKNOWN
    return FAMILY_CNN if conv >= matmul else FAMILY_MATMUL


def generate_candidates(planner_input: PlannerInput) -> dict:
    """
    Generate candidate strategies (ONNX Runtime and, if enabled, TVM).

    Rules:
    1. Always include FP32 baseline.
    2. Calibration data -> static INT8 (and dynamic INT8 for comparison).
       No calibration and model size > 5MB -> dynamic INT8.
    3. Latency constraint < 100ms -> dynamic INT8.
    4. TVM candidates only when include_tvm is set.
    5. CNNs never get dynamic INT8, in any framework: quantizing activations
       at run time before every Conv makes it slower than FP32. ONNX Runtime
       recommends static INT8 for CNNs and dynamic INT8 for transformers.

    Args:
        planner_input: PlannerInput containing model info and user constraints.

    Returns:
        dict containing selected candidates, rules applied, skipped strategies,
        the detected model family and warnings for the user.
    """

    candidates: list[Candidate] = []
    rules_applied: list[str] = []
    skipped: list[dict] = []
    warnings: list[str] = []

    family = model_family(planner_input.operator_counts)
    allow_dynamic = family != FAMILY_CNN

    def add(candidate: Candidate) -> None:
        if not any(c.name == candidate.name for c in candidates):
            candidates.append(candidate)

    # Rule 1: Always include FP32 baseline
    add(
        Candidate(
            name=STRATEGY_FP32_BASELINE,
            framework="onnxruntime",
            quantization=QUANT_FP32,
            reason="FP32 baseline is always included for comparison.",
            requires_calibration=False,
        )
    )
    rules_applied.append("Always include FP32 baseline.")

    # Rule 2: Calibration data enables static INT8.
    if planner_input.has_calibration_data:
        add(
            Candidate(
                name=STRATEGY_STATIC_INT8,
                framework="onnxruntime",
                quantization=QUANT_STATIC_INT8,
                reason="Static INT8 is selected because calibration data is available.",
                requires_calibration=True,
            )
        )
        if allow_dynamic:
            add(
                Candidate(
                    name=STRATEGY_DYNAMIC_INT8,
                    framework="onnxruntime",
                    quantization=QUANT_DYNAMIC_INT8,
                    reason="Dynamic INT8 is included for comparison with static INT8.",
                    requires_calibration=False,
                )
            )
            rules_applied.append("Calibration data provided -> compare dynamic and static INT8.")
        else:
            rules_applied.append("Calibration data provided -> static INT8.")
    elif planner_input.model_size_mb > SIZE_THRESHOLD_MB and allow_dynamic:
        add(
            Candidate(
                name=STRATEGY_DYNAMIC_INT8,
                framework="onnxruntime",
                quantization=QUANT_DYNAMIC_INT8,
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

    # Rule 3: Tight latency constraint -> dynamic INT8 (not for CNNs).
    if (
        planner_input.max_latency_ms is not None
        and planner_input.max_latency_ms < LATENCY_THRESHOLD_MS
        and allow_dynamic
    ):
        add(
            Candidate(
                name=STRATEGY_DYNAMIC_INT8,
                framework="onnxruntime",
                quantization=QUANT_DYNAMIC_INT8,
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

    # Rule 4: TVM candidates are opt-in.
    if planner_input.include_tvm:
        add(
            Candidate(
                name=STRATEGY_TVM_OPT3,
                framework="tvm",
                quantization=QUANT_FP32,
                reason="TVM Relax compiled baseline was explicitly requested.",
                requires_calibration=False,
            )
        )
        rules_applied.append("TVM candidates explicitly enabled.")

        if planner_input.has_calibration_data:
            add(
                Candidate(
                    name=STRATEGY_TVM_INT8,
                    framework="tvm",
                    quantization=QUANT_STATIC_INT8,
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

    # Rule 5: CNNs get static INT8 only, whatever the framework.
    if not allow_dynamic:
        rules_applied.append(
            "Model is a CNN (Conv-heavy) -> no dynamic INT8; static INT8 is the INT8 option."
        )
        skipped.append(
            {
                "strategy": STRATEGY_DYNAMIC_INT8,
                "reason": (
                    "Dynamic INT8 was skipped because the model is a CNN: quantizing "
                    "activations at run time before every Conv makes it slower than FP32. "
                    "Static INT8 is used for CNNs instead."
                ),
            }
        )
        if not planner_input.has_calibration_data:
            warnings.append(
                "This model is a CNN, so INT8 needs calibration images. Upload a "
                "calibration file to get static INT8; without one only FP32 is available."
            )

    if not any(c.name == STRATEGY_STATIC_INT8 for c in candidates):
        skipped.append(
            {
                "strategy": STRATEGY_STATIC_INT8,
                "reason": "Static INT8 was skipped because calibration data was unavailable.",
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
        "model_family": family,
        "candidate_count": len(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
        "rules_applied": rules_applied,
        "skipped": skipped,
        "warnings": warnings,
    }
