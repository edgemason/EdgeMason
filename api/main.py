import shutil
import tempfile
from pathlib import Path
from typing import Optional
import json

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse

try:
    from api.analyzer.model_analyzer import model_report
    from api.optimizer.onnx_runtime import benchmark
    from api.optimizer.quantizer import Quantizer
    from api.planner.rule_based import generate_candidates, PlannerInput
    from api.packager.generator import create_deployment_package
except ImportError:
    from analyzer.model_analyzer import model_report
    from optimizer.onnx_runtime import benchmark
    from optimizer.quantizer import Quantizer
    from planner.rule_based import generate_candidates, PlannerInput
    from packager.generator import create_deployment_package


app = FastAPI(title="EdgeMason", version="0.1.0")

@app.get("/")
def root():
    return "Message : Home page and root for the EdgeMason api"


@app.post("/analyze")
async def analyze_model(file: UploadFile = File(...)):
    """Step 1: Upload and analyze the ONNX model."""
    temp_dir = tempfile.mkdtemp()
    model_path = Path(temp_dir) / Path(file.filename or "model.onnx").name
    
    with open(model_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Run your analyzer
    report_json = model_report(str(model_path))
    import json
    report = json.loads(report_json)
    
    return {
        "filename": file.filename,
        "analysis": report,
    }



@app.post("/deploy")
async def deploy_model(
    file: UploadFile = File(...),
    hardware_id: str = Form("rpi4"),           # rpi4, rpi5, x86_64
    max_latency_ms: Optional[float] = Form(None),
    max_memory_mb: Optional[float] = Form(None),
    priority: str = Form("latency"),           # latency, memory, accuracy, balanced
    calibration_file: Optional[UploadFile] = File(None),
):
    """
    Full pipeline: analyze → plan → quantize → benchmark → rank → package.
    """
    # ---------- Step 1: Save model ----------
    temp_dir = tempfile.mkdtemp()
    model_path = Path(temp_dir) / Path(file.filename or "model.onnx").name
    
    with open(model_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # ---------- Step 2: Analyze ----------
    analysis = json.loads(model_report(str(model_path)))
    model_size = analysis["model_size_mb"]
    
    # ---------- Step 3: Save calibration data if provided ----------
    calib_path = None
    has_calibration = False
    if calibration_file:
        calib_path = Path(temp_dir) / Path(calibration_file.filename or "calibration.npy").name
        with open(calib_path, "wb") as f:
            shutil.copyfileobj(calibration_file.file, f)
        has_calibration = True
    
    # ---------- Step 4: Generate candidates via rule-based planner ----------
    planner_input = PlannerInput(
        model_size_mb=model_size,
        max_latency_ms=max_latency_ms,
        has_calibration_data=has_calibration,
    )
    plan = generate_candidates(planner_input)
    candidates = plan["candidates"]  # List of dicts with name, framework, quantization
    
    # ---------- Step 5: Quantize + Benchmark each candidate ----------
    quantizer = Quantizer(output_dir=str(Path(temp_dir) / "quantized"))
    results = []
    
    for cand in candidates:
        try:
            # Create the quantized model file
            quantized_path = quantizer.quantize_for_candidate(
                str(model_path),
                cand["name"],
                calibration_data_path=str(calib_path) if calib_path else None,
                hardware_id=hardware_id,
            )
            
            # Benchmark it
            bench_json = benchmark(quantized_path, warmup_runs=5, benchmark_runs=20)
            bench = json.loads(bench_json)
            
            results.append({
                "candidate": cand,
                "benchmark": bench,
                "model_path": quantized_path,
            })
            
        except Exception as e:
            results.append({
                "candidate": cand,
                "benchmark": {"success": False, "error": str(e)},
                "model_path": None,
            })
    
    # ---------- Step 6: Rank candidates ----------
    # Filter out failures
    valid_results = [r for r in results if r["benchmark"].get("success")]
    
    # Filter by constraints
    if max_latency_ms:
        valid_results = [
            r for r in valid_results 
            if r["benchmark"]["latency_ms"]["p50"] <= max_latency_ms
        ]
    
    if max_memory_mb:
        valid_results = [
            r for r in valid_results
            if r["benchmark"]["memory_mb"]["delta"] <= max_memory_mb
        ]
    
    # Sort by priority
    if priority == "latency":
        valid_results.sort(key=lambda x: x["benchmark"]["latency_ms"]["p50"])
    elif priority == "memory":
        valid_results.sort(key=lambda x: x["benchmark"]["memory_mb"]["delta"])
    else:
        # balanced: simple weighted score
        valid_results.sort(
            key=lambda x: (
                x["benchmark"]["latency_ms"]["p50"] + 
                x["benchmark"]["memory_mb"]["delta"] * 2
            )
        )
    
    # ---------- Step 7: Package winner ----------
    winner = valid_results[0] if valid_results else None
    zip_path = None
    
    if winner:
        zip_path = create_deployment_package(
            model_path=winner["model_path"],
            benchmark_report=winner["benchmark"],
            hardware_id=hardware_id,
            output_path=str(Path(temp_dir) / "edgeforge-deployment.zip"),
        )
    
    # ---------- Step 8: Return response ----------
    return {
        "hardware_target": hardware_id,
        "constraints": {
            "max_latency_ms": max_latency_ms,
            "max_memory_mb": max_memory_mb,
            "priority": priority,
        },
        "analysis": {
            "model_size_mb": model_size,
            "total_parameters": analysis["total_parameters"],
        },
        "candidates_explored": len(candidates),
        "all_results": [
            {
                "name": r["candidate"]["name"],
                "latency_p50_ms": r["benchmark"]["latency_ms"]["p50"],
                "memory_delta_mb": r["benchmark"]["memory_mb"]["delta"],
                "success": r["benchmark"]["success"],
            }
            for r in results
        ],
        "winner": {
            "name": winner["candidate"]["name"] if winner else None,
            "latency_p50_ms": winner["benchmark"]["latency_ms"]["p50"] if winner else None,
            "memory_delta_mb": winner["benchmark"]["memory_mb"]["delta"] if winner else None,
        },
        "download_url": "/download?path=" + zip_path if zip_path else None,
    }


@app.get("/download")
def download_file(path: str):
    """Download the generated deployment package."""
    return FileResponse(path, filename="edgemason-deployment.zip")
