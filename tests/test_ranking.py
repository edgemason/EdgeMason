from api.services.hardware import estimate_on_target
from api.services.ranking import rank_results


def _result(name, quantization, hardware_id, host_p50_ms, memory_delta_mb):
    return {
        "candidate": {"name": name, "quantization": quantization},
        "benchmark": {
            "success": True,
            "latency_ms": {"p50": host_p50_ms},
            "memory_mb": {"delta": memory_delta_mb},
        },
        "estimate": estimate_on_target(hardware_id, quantization, host_p50_ms, memory_delta_mb),
        "accuracy": None,
    }


def _names(ranked):
    return [result["candidate"]["name"] for result in ranked]


def test_int8_uses_its_own_multiplier_on_rpi4():
    fp32 = estimate_on_target("rpi4", "fp32", 10.0, 10.0)
    int8 = estimate_on_target("rpi4", "static_int8", 10.0, 10.0)

    assert fp32["latency_scaling_factor"] == 11.8
    assert int8["latency_scaling_factor"] > fp32["latency_scaling_factor"]
    assert int8["latency_p50_ms"] == 153.0


def test_target_changes_the_latency_winner():
    # INT8 is a bit faster on the host, but gains less on RPi 4 (no fast INT8).
    def candidates(hardware_id):
        return [
            _result("fp32", "fp32", hardware_id, 10.0, 13.0),
            _result("int8", "static_int8", hardware_id, 8.5, 3.5),
        ]

    assert _names(rank_results(candidates("x86_64"), "latency", None, None))[0] == "int8"
    assert _names(rank_results(candidates("rpi4"), "latency", None, None))[0] == "fp32"


def test_latency_limit_is_checked_against_target_estimate():
    results = [_result("fp32", "fp32", "rpi4", 6.0, 13.0)]

    # 6 ms on the host is about 70 ms on an RPi 4, so a 20 ms limit rejects it.
    assert rank_results(results, "latency", 20.0, None) == []
    assert _names(rank_results(results, "latency", 100.0, None)) == ["fp32"]


def test_memory_priority_and_limit_use_estimated_target_memory():
    results = [
        _result("fp32", "fp32", "rpi4", 10.0, 40.0),
        _result("int8", "static_int8", "rpi4", 8.0, 12.0),
    ]

    assert _names(rank_results(results, "memory", None, None)) == ["int8", "fp32"]
    # 12 MB measured x 1.1 RPi 4 overhead = 13.2 MB
    assert _names(rank_results(results, "latency", None, 13.0)) == []
    assert _names(rank_results(results, "latency", None, 14.0)) == ["int8"]


def test_balanced_weighs_latency_and_memory_relative_to_best():
    results = [
        _result("fast_big", "fp32", "x86_64", 10.0, 100.0),
        _result("slow_small", "static_int8", "x86_64", 11.0, 25.0),
    ]

    assert _names(rank_results(results, "balanced", None, None))[0] == "slow_small"
