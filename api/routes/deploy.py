import asyncio
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from api.config import (
    CALIBRATION_SUFFIXES,
    MAX_CALIBRATION_BYTES,
    MAX_MODEL_BYTES,
    VALID_EXECUTION_PROVIDERS,
    VALID_PRIORITIES,
)
from api.services.artifacts import cleanup_stale_artifacts
from api.services.hardware import hardware_ids, select_execution_provider
from api.services.pipeline import DeployRequest, run_deploy
from api.services.uploads import require_suffix, save_upload


router = APIRouter()
# One deploy at a time: candidates are benchmarked one after another, and a
# second deploy running alongside would distort their timings.
DEPLOY_SEMAPHORE = asyncio.Semaphore(1)


def _validate_form(
    hardware_id: str,
    priority: str,
    execution_provider: str,
    include_tvm: bool,
    max_latency_ms: Optional[float],
    max_memory_mb: Optional[float],
) -> str:
    """Validate deploy options and return the resolved execution provider."""
    if hardware_id not in hardware_ids():
        raise HTTPException(status_code=422, detail="invalid hardware_id")
    if priority not in VALID_PRIORITIES:
        raise HTTPException(status_code=422, detail="invalid priority")
    if execution_provider not in VALID_EXECUTION_PROVIDERS:
        raise HTTPException(status_code=422, detail="invalid execution_provider")
    if max_latency_ms is not None and max_latency_ms <= 0:
        raise HTTPException(status_code=422, detail="max_latency_ms must be positive")
    if max_memory_mb is not None and max_memory_mb <= 0:
        raise HTTPException(status_code=422, detail="max_memory_mb must be positive")
    try:
        return select_execution_provider(hardware_id, execution_provider, include_tvm)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/deploy")
async def deploy_model(
    file: UploadFile = File(...),
    hardware_id: str = Form("rpi4"),
    max_latency_ms: Optional[float] = Form(None),
    max_memory_mb: Optional[float] = Form(None),
    priority: str = Form("latency"),
    calibration_file: Optional[UploadFile] = File(None),
    execution_provider: str = Form("auto"),
    include_tvm: bool = Form(False),
    enable_tvm_tuning: bool = Form(False),
):
    require_suffix(file, {".onnx"}, "model must be an .onnx file")
    resolved_provider = _validate_form(
        hardware_id,
        priority,
        execution_provider,
        include_tvm,
        max_latency_ms,
        max_memory_mb,
    )
    calibration_suffix = None
    if calibration_file:
        calibration_suffix = require_suffix(
            calibration_file,
            CALIBRATION_SUFFIXES,
            "calibration file must be .npy or .npz",
        )

    def deploy_in_work_dir() -> dict:
        # Blocking work (file copies, quantization, benchmarks, packaging).
        # Runs on a worker thread so the event loop keeps serving requests.
        cleanup_stale_artifacts()
        with tempfile.TemporaryDirectory(prefix="edgemason-deploy-") as directory:
            work_dir = Path(directory)
            model_path = work_dir / "model.onnx"
            save_upload(file, model_path, MAX_MODEL_BYTES)
            calibration_path = None
            if calibration_file and calibration_suffix:
                calibration_path = work_dir / f"calibration{calibration_suffix}"
                save_upload(calibration_file, calibration_path, MAX_CALIBRATION_BYTES)
            return run_deploy(
                DeployRequest(
                    model_path=model_path,
                    calibration_path=calibration_path,
                    hardware_id=hardware_id,
                    max_latency_ms=max_latency_ms,
                    max_memory_mb=max_memory_mb,
                    priority=priority,
                    work_dir=work_dir,
                    requested_provider=execution_provider,
                    execution_provider=resolved_provider,
                    include_tvm=include_tvm,
                    enable_tvm_tuning=enable_tvm_tuning,
                )
            )

    try:
        async with DEPLOY_SEMAPHORE:
            return await run_in_threadpool(deploy_in_work_dir)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
