import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

import numpy as np
import onnx
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from starlette.background import BackgroundTask

from api.analyzer.model_analyzer import model_report
from api.optimizer.onnx_runtime import benchmark
from api.optimizer.quantizer import Quantizer
from api.optimizer.validation import (
    ValidationData,
    evaluate_onnx,
    load_calibration_inputs,
    load_validation_data,
)
from api.packager.generator import create_deployment_package
from api.planner.rule_based import PlannerInput, generate_candidates


app = FastAPI(title="EdgeMason", version="0.1.0")

MAX_MODEL_BYTES = 350 * 1024 * 1024
MAX_CALIBRATION_BYTES = 256 * 1024 * 1024
MAX_MODEL_NODES = 10_000
MAX_INPUT_ELEMENTS = 10_000_000
ARTIFACT_TTL_SECONDS = 60 * 60
ARTIFACT_DIR = Path(tempfile.gettempdir()) / "edgemason-artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
DEPLOY_SEMAPHORE = asyncio.Semaphore(1)
VALID_HARDWARE = {"rpi3", "rpi4", "rpi5", "jetson_nano", "x86_64"}
VALID_PRIORITIES = {"latency", "memory", "accuracy", "balanced"}
VALID_EXECUTION_PROVIDERS = {"auto", "cpu", "cuda"}


def _copy_upload(upload: UploadFile, path: Path, limit: int) -> None:
    size = 0
    try:
        with path.open("wb") as destination:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail="uploaded file is too large")
                destination.write(chunk)
    finally:
        upload.file.close()


async def _save_upload(upload: UploadFile, path: Path, limit: int) -> None:
    _copy_upload(upload, path, limit)


def _validate_model(path: Path) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    if len(model.graph.node) > MAX_MODEL_NODES:
        raise ValueError("model has too many nodes")

    initializer_names = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializer_names]
    if len(inputs) != 1:
        raise ValueError("the pipeline supports single-input models only")

    tensor_type = inputs[0].type.tensor_type
    if tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("the pipeline supports float32 model inputs only")

    elements = 1
    for index, dimension in enumerate(tensor_type.shape.dim):
        if dimension.dim_value > 0:
            value = dimension.dim_value
        elif index == 0:
            value = 1
        else:
            raise ValueError("only the model batch dimension may be dynamic")
        if value > 4096:
            raise ValueError("model input dimension is too large")
        elements *= value
    if elements > MAX_INPUT_ELEMENTS:
        raise ValueError("model input tensor is too large")


def _cleanup_stale_artifacts() -> None:
    cutoff = time.time() - ARTIFACT_TTL_SECONDS
    for path in ARTIFACT_DIR.glob("*.zip"):
        if path.is_file() and path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)


def _model_input_name(model_path: str) -> str:
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    return session.get_inputs()[0].name


def _candidate_result(
    candidate: dict,
    model_path: str,
    calibration_path: Optional[Path],
    calibration_inputs: Optional[np.ndarray],
    validation: Optional[ValidationData],
    hardware_id: str,
    output_dir: Path,
    execution_provider: str,
) -> dict:
    try:
        quantized_path = Quantizer(str(output_dir)).quantize_for_candidate(
            model_path,
            candidate["name"],
            calibration_data_path=str(calibration_path) if calibration_path else None,
            hardware_id=hardware_id,
        )
        is_tvm = candidate.get("framework") == "tvm" or str(quantized_path).endswith(
            (".tar", ".so")
        )
        if is_tvm:
            from api.optimizer.validation import evaluate_tvm

            bench = json.loads(
                benchmark(
                    quantized_path,
                    warmup_runs=2,
                    benchmark_runs=5,
                    hardware_id=hardware_id,
                    execution_provider=execution_provider,
                )
            )
            accuracy = evaluate_tvm(quantized_path, validation) if validation else None
        else:
            input_data = None
            if calibration_inputs is not None:
                input_data = {
                    _model_input_name(quantized_path): np.asarray(calibration_inputs[:1])
                }
            bench = json.loads(
                benchmark(
                    quantized_path,
                    input_data=input_data,
                    warmup_runs=2,
                    benchmark_runs=5,
                    hardware_id=hardware_id,
                    execution_provider=execution_provider,
                )
            )
            accuracy = evaluate_onnx(quantized_path, validation) if validation else None
        return {
            "candidate": candidate,
            "benchmark": bench,
            "accuracy": accuracy,
            "model_path": quantized_path,
            "provider": bench.get("provider"),
            "error": bench.get("error"),
        }
    except Exception as exc:
        return {
            "candidate": candidate,
            "benchmark": {"success": False},
            "accuracy": None,
            "model_path": None,
            "provider": None,
            "error": str(exc),
        }


def _rank_results(
    results: list[dict],
    priority: str,
    max_latency_ms: Optional[float],
    max_memory_mb: Optional[float],
) -> list[dict]:
    valid = [result for result in results if result["benchmark"].get("success")]
    if max_latency_ms is not None:
        valid = [
            result
            for result in valid
            if result["benchmark"]["latency_ms"]["p50"] <= max_latency_ms
        ]
    if max_memory_mb is not None:
        valid = [
            result
            for result in valid
            if result["benchmark"]["memory_mb"]["delta"] <= max_memory_mb
        ]

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
        return sorted(valid, key=lambda result: result["benchmark"]["memory_mb"]["delta"])
    if priority == "balanced":
        return sorted(
            valid,
            key=lambda result: (
                result["benchmark"]["latency_ms"]["p50"]
                + result["benchmark"]["memory_mb"]["delta"] * 2
            ),
        )
    return sorted(valid, key=lambda result: result["benchmark"]["latency_ms"]["p50"])


def _run_deploy(
    model_path: Path,
    calibration_path: Optional[Path],
    hardware_id: str,
    max_latency_ms: Optional[float],
    max_memory_mb: Optional[float],
    priority: str,
    work_dir: Path,
    execution_provider: str,
    include_tvm: bool,
) -> dict:
    _validate_model(model_path)
    analysis = json.loads(model_report(str(model_path)))
    calibration_inputs = (
        load_calibration_inputs(calibration_path) if calibration_path else None
    )
    validation = (
        load_validation_data(calibration_path)
        if calibration_path and calibration_path.suffix.lower() == ".npz"
        else None
    )
    if priority == "accuracy" and validation is None:
        raise ValueError("accuracy priority requires labeled .npz validation data")

    plan = generate_candidates(
        PlannerInput(
            model_size_mb=analysis["model_size_mb"],
            max_latency_ms=max_latency_ms,
            has_calibration_data=calibration_path is not None,
            include_tvm=include_tvm,
        )
    )
    results = [
        _candidate_result(
            candidate,
            str(model_path),
            calibration_path,
            calibration_inputs,
            validation,
            hardware_id,
            work_dir / candidate["name"],
            execution_provider,
        )
        for candidate in plan["candidates"]
    ]
    ranked = _rank_results(results, priority, max_latency_ms, max_memory_mb)
    winner = ranked[0] if ranked else None

    artifact_id = None
    if winner:
        artifact_id = uuid4().hex
        create_deployment_package(
            model_path=winner["model_path"],
            benchmark_report={
                **winner["benchmark"],
                "accuracy": winner["accuracy"],
            },
            hardware_id=hardware_id,
            output_path=str(ARTIFACT_DIR / f"{artifact_id}.zip"),
        )

    def public_result(result: dict) -> dict:
        bench = result["benchmark"]
        return {
            "name": result["candidate"]["name"],
            "framework": result["candidate"].get("framework", "onnxruntime"),
            "success": bench.get("success", False),
            "error": result["error"],
            "latency_p50_ms": bench.get("latency_ms", {}).get("p50"),
            "latency_avg_ms": bench.get("latency_ms", {}).get("avg"),
            "throughput_fps": bench.get("throughput", {}).get("fps"),
            "memory_delta_mb": bench.get("memory_mb", {}).get("delta"),
            "model_size_mb": bench.get("model_size", {}).get("mb"),
            "provider": result["provider"],
            "accuracy": result["accuracy"],
        }

    return {
        "hardware_target": hardware_id,
        "constraints": {
            "max_latency_ms": max_latency_ms,
            "max_memory_mb": max_memory_mb,
            "priority": priority,
            "execution_provider": execution_provider,
            "include_tvm": include_tvm,
        },
        "analysis": {
            "model_size_mb": analysis["model_size_mb"],
            "total_parameters": analysis["total_parameters"],
        },
        "candidates_explored": len(results),
        "all_results": [public_result(result) for result in results],
        "winner": public_result(winner) if winner else None,
        "download_url": f"/download/{artifact_id}" if artifact_id else None,
    }


STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def root():
    if STATIC_INDEX.exists():
        return HTMLResponse(content=STATIC_INDEX.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>EdgeForge API</h1>")


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    if STATIC_INDEX.exists():
        return HTMLResponse(content=STATIC_INDEX.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>EdgeForge UI</h1>")


@app.get("/api")
async def api_info():
    return {"message": "EdgeMason API"}


@app.get("/health")
async def health():
    return {"status": "ok"}



@app.post("/analyze")
async def analyze_model(file: UploadFile = File(...)):
    if Path(file.filename or "").suffix.lower() != ".onnx":
        raise HTTPException(status_code=422, detail="model must be an .onnx file")
    try:
        with tempfile.TemporaryDirectory(prefix="edgemason-analyze-") as directory:
            model_path = Path(directory) / "model.onnx"
            await _save_upload(file, model_path, MAX_MODEL_BYTES)
            _validate_model(model_path)
            report = model_report(str(model_path))
            return {"filename": file.filename, "analysis": json.loads(report)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/deploy")
async def deploy_model(
    file: UploadFile = File(...),
    hardware_id: str = Form("rpi4"),
    max_latency_ms: Optional[float] = Form(None),
    max_memory_mb: Optional[float] = Form(None),
    priority: str = Form("latency"),
    calibration_file: Optional[UploadFile] = File(None),
    execution_provider: str = Form("auto"),
    include_tvm: bool = Form(False),
):
    if Path(file.filename or "").suffix.lower() != ".onnx":
        raise HTTPException(status_code=422, detail="model must be an .onnx file")
    if hardware_id not in VALID_HARDWARE:
        raise HTTPException(status_code=422, detail="invalid hardware_id")
    if priority not in VALID_PRIORITIES:
        raise HTTPException(status_code=422, detail="invalid priority")
    if execution_provider not in VALID_EXECUTION_PROVIDERS:
        raise HTTPException(status_code=422, detail="invalid execution_provider")
    if execution_provider == "cuda" and include_tvm:
        raise HTTPException(
            status_code=422,
            detail="include_tvm cannot be combined with execution_provider=cuda",
        )
    if (
        execution_provider == "cuda"
        and "CUDAExecutionProvider" not in ort.get_available_providers()
    ):
        raise HTTPException(
            status_code=422,
            detail="CUDAExecutionProvider is not available",
        )
    if max_latency_ms is not None and max_latency_ms <= 0:
        raise HTTPException(status_code=422, detail="max_latency_ms must be positive")
    if max_memory_mb is not None and max_memory_mb <= 0:
        raise HTTPException(status_code=422, detail="max_memory_mb must be positive")

    calibration_suffix = None
    if calibration_file:
        calibration_suffix = Path(calibration_file.filename or "").suffix.lower()
        if calibration_suffix not in {".npy", ".npz"}:
            raise HTTPException(
                status_code=422,
                detail="calibration file must be .npy or .npz",
            )

    try:
        _cleanup_stale_artifacts()
        with tempfile.TemporaryDirectory(prefix="edgemason-deploy-") as directory:
            work_dir = Path(directory)
            model_path = work_dir / "model.onnx"
            await _save_upload(file, model_path, MAX_MODEL_BYTES)
            calibration_path = None
            if calibration_file and calibration_suffix:
                calibration_path = work_dir / f"calibration{calibration_suffix}"
                await _save_upload(
                    calibration_file,
                    calibration_path,
                    MAX_CALIBRATION_BYTES,
                )

            async with DEPLOY_SEMAPHORE:
                return _run_deploy(
                    model_path,
                    calibration_path,
                    hardware_id,
                    max_latency_ms,
                    max_memory_mb,
                    priority,
                    work_dir,
                    execution_provider,
                    include_tvm,
                )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/download/{artifact_id}")
async def download_file(artifact_id: str):
    try:
        parsed_id = UUID(hex=artifact_id).hex
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    artifact_path = ARTIFACT_DIR / f"{parsed_id}.zip"
    if not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")

    async def content():
        with artifact_path.open("rb") as artifact:
            while chunk := artifact.read(1024 * 1024):
                yield chunk

    async def cleanup() -> None:
        artifact_path.unlink(missing_ok=True)

    return StreamingResponse(
        content(),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="edgemason-deployment.zip"'
        },
        background=BackgroundTask(cleanup),
    )
