from api.planner.rule_based import (
    LATENCY_THRESHOLD_MS,
    SIZE_THRESHOLD_MB,
    PlannerInput,
    generate_candidates,
)


def candidate_names(plan):
    return [c["name"] for c in plan["candidates"]]


def test_fp32_baseline_always_present():
    plan = generate_candidates(
        PlannerInput(model_size_mb=1.0, max_latency_ms=None)
    )
    assert "onnx_fp32_baseline" in candidate_names(plan)
    assert plan["candidate_count"] >= 1


def test_large_model_with_calibration_uses_static_int8():
    plan = generate_candidates(
        PlannerInput(
            model_size_mb=SIZE_THRESHOLD_MB + 1,
            has_calibration_data=True,
        )
    )
    names = candidate_names(plan)
    assert "onnx_static_int8" in names
    assert "onnx_dynamic_int8" in names


def test_large_model_without_calibration_uses_dynamic_int8():
    plan = generate_candidates(
        PlannerInput(
            model_size_mb=SIZE_THRESHOLD_MB + 1,
            has_calibration_data=False,
        )
    )
    names = candidate_names(plan)
    assert "onnx_dynamic_int8" in names
    assert "onnx_static_int8" not in names


def test_tight_latency_constraint_uses_dynamic_int8():
    plan = generate_candidates(
        PlannerInput(
            model_size_mb=1.0,
            max_latency_ms=LATENCY_THRESHOLD_MS - 1,
        )
    )
    assert "onnx_dynamic_int8" in candidate_names(plan)


def test_candidates_are_deduplicated():
    plan = generate_candidates(
        PlannerInput(
            model_size_mb=SIZE_THRESHOLD_MB + 1,
            max_latency_ms=LATENCY_THRESHOLD_MS - 1,
            has_calibration_data=False,
        )
    )
    names = candidate_names(plan)
    assert names.count("onnx_dynamic_int8") == 1


def test_candidate_shape():
    plan = generate_candidates(PlannerInput(model_size_mb=1.0))
    cand = plan["candidates"][0]
    assert set(cand) >= {"name", "framework", "quantization", "reason"}
    assert cand["framework"] == "onnxruntime"
    assert isinstance(plan["rules_applied"], list)
    assert "skipped" in plan


def test_default_plan_excludes_tvm_candidates():
    plan = generate_candidates(
        PlannerInput(model_size_mb=1.0, max_latency_ms=None)
    )
    names = candidate_names(plan)
    assert "tvm_opt_level_3" not in names
    assert "tvm_int8_quantized" not in names


def test_opted_in_plan_includes_tvm_candidates():
    plan = generate_candidates(
        PlannerInput(
            model_size_mb=SIZE_THRESHOLD_MB + 1,
            has_calibration_data=True,
            include_tvm=True,
        )
    )
    names = candidate_names(plan)
    assert "tvm_opt_level_3" in names
    assert "tvm_int8_quantized" in names
    tvm_cand = next(c for c in plan["candidates"] if c["name"] == "tvm_opt_level_3")
    assert tvm_cand["framework"] == "tvm"


def test_calibration_includes_tvm_int8():
    plan = generate_candidates(
        PlannerInput(
            model_size_mb=SIZE_THRESHOLD_MB + 1,
            has_calibration_data=True,
            include_tvm=True,
        )
    )
    assert "tvm_int8_quantized" in candidate_names(plan)


def test_no_calibration_skips_tvm_int8():
    plan = generate_candidates(
        PlannerInput(
            model_size_mb=SIZE_THRESHOLD_MB + 1,
            has_calibration_data=False,
        )
    )
    assert "tvm_int8_quantized" not in candidate_names(plan)


def test_small_fast_model_skips_static_int8():
    plan = generate_candidates(
        PlannerInput(model_size_mb=1.0, max_latency_ms=None)
    )
    assert "onnx_static_int8" not in candidate_names(plan)


CNN_OPS = {"Conv": 52, "Clip": 35, "Add": 10, "Gemm": 1}          # MobileNetV2-like
MATMUL_OPS = {"MatMul": 96, "Add": 120, "Softmax": 12, "Gemm": 2}  # transformer-like


def test_model_family_from_operator_counts():
    from api.planner.rule_based import model_family

    assert model_family(CNN_OPS) == "cnn"
    assert model_family(MATMUL_OPS) == "matmul"
    assert model_family({"Relu": 3}) == "unknown"
    assert model_family({}) == "unknown"


def test_cnn_with_calibration_gets_static_int8_but_no_dynamic():
    plan = generate_candidates(
        PlannerInput(model_size_mb=13.0, has_calibration_data=True, operator_counts=CNN_OPS)
    )

    assert plan["model_family"] == "cnn"
    assert candidate_names(plan) == ["onnx_fp32_baseline", "onnx_static_int8"]
    skipped = {item["strategy"]: item["reason"] for item in plan["skipped"]}
    assert "slower than FP32" in skipped["onnx_dynamic_int8"]
    assert plan["warnings"] == []


def test_cnn_never_gets_dynamic_even_for_large_model_and_tight_latency():
    plan = generate_candidates(
        PlannerInput(
            model_size_mb=SIZE_THRESHOLD_MB + 100,
            max_latency_ms=LATENCY_THRESHOLD_MS - 1,
            has_calibration_data=False,
            operator_counts=CNN_OPS,
        )
    )

    assert candidate_names(plan) == ["onnx_fp32_baseline"]
    assert "calibration" in plan["warnings"][0]


def test_cnn_rule_applies_to_tvm_candidates_by_method():
    plan = generate_candidates(
        PlannerInput(
            model_size_mb=13.0,
            has_calibration_data=True,
            include_tvm=True,
            operator_counts=CNN_OPS,
        )
    )

    methods = {c["name"]: c["quantization"] for c in plan["candidates"]}
    assert methods == {
        "onnx_fp32_baseline": "fp32",
        "onnx_static_int8": "static_int8",
        "tvm_opt_level_3": "fp32",
        "tvm_int8_quantized": "static_int8",
    }
    assert "dynamic_int8" not in methods.values()


def test_matmul_heavy_model_keeps_dynamic_int8():
    plan = generate_candidates(
        PlannerInput(
            model_size_mb=SIZE_THRESHOLD_MB + 1,
            max_latency_ms=LATENCY_THRESHOLD_MS - 1,
            has_calibration_data=True,
            operator_counts=MATMUL_OPS,
        )
    )

    assert plan["model_family"] == "matmul"
    assert set(candidate_names(plan)) == {
        "onnx_fp32_baseline",
        "onnx_static_int8",
        "onnx_dynamic_int8",
    }
