"""Run one benchmark in a fresh process (used by benchmark_engine.benchmark).

Usage: python -m api.optimizer.benchmark_worker <request.json> <result.json>
"""

import json
import sys
from pathlib import Path

import numpy as np

from api.optimizer.benchmark_engine import BenchmarkEngine


def main(request_path: str, result_path: str) -> None:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    input_data = None
    if request.get("inputs_path"):
        with np.load(request["inputs_path"]) as arrays:
            input_data = {
                name: arrays[f"arr_{index}"]
                for index, name in enumerate(request["input_names"])
            }

    engine = BenchmarkEngine(
        hardware_id=request["hardware_id"],
        execution_provider=request["execution_provider"],
        time_budget_seconds=request.get("time_budget_seconds"),
    )
    result = engine.benchmark(
        model_path=request["model_path"],
        warmup_runs=request["warmup_runs"],
        benchmark_runs=request["benchmark_runs"],
        batch_size=request["batch_size"],
        input_data=input_data,
    )
    Path(result_path).write_text(result.to_json(), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
