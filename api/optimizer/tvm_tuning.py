"""TVM Tuning Module for EdgeForge / EdgeMason.

Provides MetaSchedule auto-tuning, schedule database persistence, and optimization
pipeline passes for Apache TVM (Relax).

Why TVM needs tuning:
Unlike ONNX Runtime which uses pre-compiled, vendor-optimized libraries (oneDNN,
Eigen, MKL), TVM generates code from scratch using TensorIR. Without tuning, TVM
generates generic naive nested loops that run scalar and unvectorized on CPUs.
MetaSchedule searches the space of loop transformations (tiling, unrolling,
vectorization, thread binding, cache allocation) to find hardware-optimal
schedules that match or beat vendor libraries.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TuningConfig:
    """Configuration for TVM MetaSchedule tuning."""

    hardware_id: str = "x86_64"
    max_trials_global: int = 16
    num_trials_per_iter: int = 4
    strategy: str = "evolutionary"  # "evolutionary" or "random"
    work_dir: Optional[str] = None
    timeout_sec_per_trial: int = 30
    cost_model: str = "xgb"  # "xgb" or "random"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_default_tuning_dir(hardware_id: str = "x86_64") -> Path:
    """Return the default directory for persisting TVM tuning records."""
    base = Path(__file__).resolve().parent.parent.parent / "profiles" / "tvm_tuning" / hardware_id
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_tuning_database(work_dir: str | Path | None = None):
    """Load or initialize a MetaSchedule JSONDatabase.

    Returns:
        JSONDatabase instance if TVM is available and directory exists, else None.
    """
    try:
        import tvm
        from tvm import meta_schedule as ms

        work_path = Path(work_dir) if work_dir else get_default_tuning_dir()
        work_path.mkdir(parents=True, exist_ok=True)

        workload_path = work_path / "database_workload.json"
        record_path = work_path / "database_tuning_record.json"

        # Touch files if they don't exist
        if not workload_path.exists():
            workload_path.write_text("[]", encoding="utf-8")
        if not record_path.exists():
            record_path.write_text("[]", encoding="utf-8")

        return ms.database.JSONDatabase(
            path_workload=str(workload_path),
            path_tuning_record=str(record_path),
        )
    except Exception:
        return None


def tune_relax_module(
    mod: Any,
    target: Any,
    config: Optional[TuningConfig] = None,
) -> tuple[Any, Dict[str, Any]]:
    """Tune compute-heavy kernels in a Relax IRModule using MetaSchedule.

    Args:
        mod: TVM Relax IRModule.
        target: TVM Target instance or dict.
        config: Optional TuningConfig specifying trial budget and paths.

    Returns:
        tuple (database, tuning_report_dict)
    """
    try:
        import tvm
        from tvm import meta_schedule as ms
    except ImportError as exc:
        raise ImportError(
            "Apache TVM is not installed. Run: pip install apache-tvm"
        ) from exc

    cfg = config or TuningConfig()
    work_path = Path(cfg.work_dir) if cfg.work_dir else get_default_tuning_dir(cfg.hardware_id)
    work_path.mkdir(parents=True, exist_ok=True)

    tvm_target = target if isinstance(target, tvm.target.Target) else tvm.target.Target(target)

    report = {
        "hardware_id": cfg.hardware_id,
        "max_trials_global": cfg.max_trials_global,
        "work_dir": str(work_path),
        "status": "pending",
        "tasks_tuned": 0,
        "error": None,
    }

    try:
        # Extract tuning tasks from the Relax module
        database = ms.tune_relax(
            mod=mod,
            target=tvm_target,
            work_dir=str(work_path),
            max_trials_global=cfg.max_trials_global,
            num_trials_per_iter=min(cfg.num_trials_per_iter, cfg.max_trials_global),
            strategy=cfg.strategy,
        )
        report["status"] = "completed"
        report["database_workload"] = str(work_path / "database_workload.json")
        report["database_records"] = str(work_path / "database_tuning_record.json")
        return database, report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        # Attempt fallback to loading existing database if any
        existing_db = get_tuning_database(work_path)
        return existing_db, report


def apply_tuning_best(mod: Any, database: Any) -> Any:
    """Apply the best tuned schedules from a MetaSchedule database to a Relax IRModule."""
    if database is None:
        return mod
    try:
        from tvm import meta_schedule as ms

        if hasattr(ms, "relax") and hasattr(ms.relax, "ApplyHistoryBest"):
            return ms.relax.ApplyHistoryBest(database)(mod)
    except Exception:
        pass
    return mod


def configure_runtime_threads(num_threads: Optional[int] = None) -> int:
    """Ensure TVM uses all available physical CPU cores for parallel execution.

    Without this, TVM defaults to a single thread while ONNX Runtime uses all cores,
    leading to an artificial 4x-16x latency disadvantage.
    """
    import psutil

    threads = num_threads or psutil.cpu_count(logical=False) or 4
    os.environ["TVM_NUM_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    try:
        import tvm

        if hasattr(tvm.runtime, "set_num_threads"):
            tvm.runtime.set_num_threads(threads)
    except Exception:
        pass
    return threads
