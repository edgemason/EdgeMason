"""Hardware targets and execution-provider selection.

Each profile in profiles/hardware_profiles.json lists the execution providers
the device can use, in order of preference. The provider is resolved once per
deploy so that every candidate is benchmarked on the same device.
"""

import onnxruntime as ort

from api.optimizer.benchmark_engine import load_hardware_profiles


ORT_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}


def hardware_ids() -> set[str]:
    return set(load_hardware_profiles())


def supported_providers(hardware_id: str) -> list[str]:
    profile = load_hardware_profiles()[hardware_id]
    return profile.get("execution_providers", ["cpu"])


def _is_installed(provider: str) -> bool:
    return ORT_PROVIDERS[provider] in ort.get_available_providers()


def select_execution_provider(
    hardware_id: str,
    requested: str,
    include_tvm: bool,
) -> str:
    """Resolve auto/cpu/cuda to one concrete provider ("cpu" or "cuda").

    auto picks the first provider in the profile that is installed on this
    host. TVM candidates are compiled for CPU, so auto stays on CPU when they
    are included to keep the comparison fair.
    """
    supported = supported_providers(hardware_id)
    if requested == "auto":
        for provider in supported:
            if provider == "cuda" and include_tvm:
                continue
            if _is_installed(provider):
                return provider
        return "cpu"

    if requested not in supported:
        raise ValueError(f"{hardware_id} does not support execution_provider={requested}")
    if requested == "cuda" and include_tvm:
        raise ValueError("include_tvm cannot be combined with execution_provider=cuda")
    if not _is_installed(requested):
        raise ValueError(f"{ORT_PROVIDERS[requested]} is not available")
    return requested


def estimate_on_target(
    hardware_id: str,
    quantization: str,
    latency_p50_ms: float,
    memory_mb: float,
) -> dict:
    """Estimate latency and memory on the target from the host benchmark.

    Latency: host p50 x the profile's multiplier for this quantization. Each
    model type gets its own multiplier because INT8 speeds up differently on
    each CPU; fp32 uses latency_scaling_factor.
    Memory: measured peak RSS delta (model load + runs, in a fresh process)
    x memory_overhead_scaling. Memory barely changes between host and device,
    so this factor is small.
    """
    profile = load_hardware_profiles()[hardware_id]
    factor = profile.get("latency_scaling_by_quantization", {}).get(
        quantization, profile.get("latency_scaling_factor", 1.0)
    )
    return {
        "target_hardware": hardware_id,
        "latency_scaling_factor": factor,
        "latency_p50_ms": round(latency_p50_ms * factor, 3),
        "memory_mb": round(memory_mb * profile.get("memory_overhead_scaling", 1.0), 3),
    }
