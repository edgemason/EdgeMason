"""Pick the winner using the estimate for the target device, not host numbers."""

from typing import Optional


def _latency(result: dict) -> float:
    return result["estimate"]["latency_p50_ms"]


def _memory(result: dict) -> float:
    return result["estimate"]["memory_mb"]


def rank_results(
    results: list[dict],
    priority: str,
    max_latency_ms: Optional[float],
    max_memory_mb: Optional[float],
) -> list[dict]:
    valid = [
        result
        for result in results
        if result["benchmark"].get("success") and result.get("estimate")
    ]
    if max_latency_ms is not None:
        valid = [result for result in valid if _latency(result) <= max_latency_ms]
    if max_memory_mb is not None:
        valid = [result for result in valid if _memory(result) <= max_memory_mb]

    if priority == "accuracy":
        return sorted(
            [result for result in valid if result["accuracy"] is not None],
            key=lambda result: (
                result["accuracy"]["top1_accuracy"],
                result["accuracy"]["top5_accuracy"],
            ),
            reverse=True,
        )
    if priority == "memory":
        return sorted(valid, key=_memory)
    if priority == "balanced" and valid:
        # Each metric relative to the best candidate, so ms and MB weigh equally.
        best_latency = max(min(map(_latency, valid)), 1e-9)
        best_memory = max(min(map(_memory, valid)), 1e-9)
        return sorted(
            valid,
            key=lambda result: (
                _latency(result) / best_latency + _memory(result) / best_memory
            ),
        )
    return sorted(valid, key=_latency)
