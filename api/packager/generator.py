# packager/generator.py
import zipfile
import json
from pathlib import Path
from typing import Dict


DOCKERFILE_TEMPLATE = '''FROM arm64v8/python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir onnxruntime==1.18.0 numpy==1.26.4

COPY model.onnx .
COPY inference.py .
COPY requirements.txt .

CMD ["python", "inference.py"]
'''

INFERENCE_TEMPLATE = '''import numpy as np
import onnxruntime as rt

# Load model
session = rt.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Dummy inference (replace with real preprocessing)
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {{input_name: dummy_input}})

print("Inference successful. Output shape:", outputs[0].shape)
'''

REQUIREMENTS_TXT = '''onnxruntime==1.18.0
numpy==1.26.4
'''


def create_deployment_package(
    model_path: str,
    benchmark_report: Dict,
    hardware_id: str,
    output_path: str = "./edgemason-deployment.zip",
) -> str:
    """Create a ZIP file containing the model, Dockerfile, and inference script."""
    
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. The optimized model
        zf.write(model_path, arcname="model.onnx")
        
        # 2. Dockerfile (ARM64 ready)
        zf.writestr("Dockerfile", DOCKERFILE_TEMPLATE)
        
        # 3. Inference script
        zf.writestr("inference.py", INFERENCE_TEMPLATE)
        
        # 4. Requirements
        zf.writestr("requirements.txt", REQUIREMENTS_TXT)
        
        # 5. Benchmark report
        zf.writestr("benchmark_report.json", json.dumps(benchmark_report, indent=2))
        
        # 6. README
        readme = f"""# EdgeForge Deployment Package

                Hardware Target: {hardware_id}
                Optimized Model: model.onnx

                ## Quick Start
                ```bash
                docker build -t my-model .
                docker run my-model ```
                
                ## Benchmark Summary
                {json.dumps(benchmark_report, indent=2)}
                """
        zf.writestr("README.md", readme)
        return output_path