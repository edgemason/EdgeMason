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

TVM_DOCKERFILE_TEMPLATE = '''FROM arm64v8/python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir apache-tvm numpy==1.26.4

COPY model_tvm.tar .
COPY inference.py .
COPY requirements.txt .

CMD ["python", "inference.py"]
'''

TVM_INFERENCE_TEMPLATE = '''import numpy as np
import tvm
from tvm import relax

# Load compiled Relax VM module
module = tvm.runtime.load_module("model_tvm.tar")
dev = tvm.cpu(0)
vm = relax.VirtualMachine(module, dev)

# Dummy inference (replace with real preprocessing)
dummy_input = tvm.runtime.tensor(
    np.random.randn(1, 3, 224, 224).astype(np.float32), dev
)
outputs = vm["main"](dummy_input)

print("Inference successful. Output shape:", np.asarray(outputs).shape)
'''

TVM_REQUIREMENTS_TXT = '''apache-tvm
numpy==1.26.4
'''


def create_deployment_package(
    model_path: str,
    benchmark_report: Dict,
    hardware_id: str,
    output_path: str = "./edgemason-deployment.zip",
) -> str:
    """Create a ZIP file containing the model, Dockerfile, and inference script."""
    is_tvm = str(model_path).endswith((".tar", ".so"))
    model_arcname = "model_tvm.tar" if is_tvm else "model.onnx"

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. The optimized model
        zf.write(model_path, arcname=model_arcname)

        # Sidecar with TVM input specs, when present
        sidecar = Path(str(model_path) + ".inputs.json")
        if is_tvm and sidecar.is_file():
            zf.write(sidecar, arcname="tvm_inputs.json")

        # 2. Dockerfile (ARM64 ready)
        zf.writestr(
            "Dockerfile",
            TVM_DOCKERFILE_TEMPLATE if is_tvm else DOCKERFILE_TEMPLATE,
        )

        # 3. Inference script
        zf.writestr("inference.py", TVM_INFERENCE_TEMPLATE if is_tvm else INFERENCE_TEMPLATE)

        # 4. Requirements
        zf.writestr("requirements.txt", TVM_REQUIREMENTS_TXT if is_tvm else REQUIREMENTS_TXT)
        
        # 5. Benchmark report
        zf.writestr("benchmark_report.json", json.dumps(benchmark_report, indent=2))
        
        # 6. README
        readme = f"""# EdgeForge Deployment Package

                Hardware Target: {hardware_id}
                Optimized Model: {model_arcname}

                ## Quick Start
                ```bash
                docker build -t my-model .
                docker run my-model ```
                
                ## Benchmark Summary
                {json.dumps(benchmark_report, indent=2)}
                """
        zf.writestr("README.md", readme)
        return output_path