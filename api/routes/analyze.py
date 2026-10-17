import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from api.analyzer.model_analyzer import model_report
from api.config import MAX_MODEL_BYTES
from api.services.model_validation import validate_model
from api.services.uploads import require_suffix, save_upload


router = APIRouter()


@router.post("/analyze")
async def analyze_model(file: UploadFile = File(...)):
    require_suffix(file, {".onnx"}, "model must be an .onnx file")
    def analyze_in_work_dir() -> dict:
        # Blocking work; runs on a worker thread so the event loop stays free.
        with tempfile.TemporaryDirectory(prefix="edgemason-analyze-") as directory:
            model_path = Path(directory) / "model.onnx"
            save_upload(file, model_path, MAX_MODEL_BYTES)
            validate_model(model_path)
            report = model_report(str(model_path))
            return {"filename": file.filename, "analysis": json.loads(report)}

    try:
        return await run_in_threadpool(analyze_in_work_dir)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
