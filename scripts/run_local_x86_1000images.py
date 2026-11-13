import gc
import os
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torchvision.transforms as transforms
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from PIL import Image

os.makedirs("reports/data", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("[1/3] Loading pre-extracted 1,000 real images...")
image_paths = []
for root, _, files in os.walk("dataset/imagenette2-160/val"):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(root, f))

selected_paths = image_paths[:1000]
print(f"Total test images: {len(selected_paths)}")

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocess a PIL image into a (1, 3, 224, 224) float32 array.

    torchvision's Compose stub doesn't reflect that our pipeline (Resize ->
    ToTensor -> Normalize) always returns a Tensor, not an Image -- the
    explicit annotation below is what actually makes that true.
    """
    tensor: torch.Tensor = preprocess(img)  # type: ignore[assignment]
    return tensor.unsqueeze(0).numpy().astype(np.float32)


# Calibration subset
calib_tensors = []
for p in selected_paths[:100]:
    img = Image.open(p).convert("RGB")
    calib_tensors.append(preprocess_image(img))
calib_array = np.concatenate(calib_tensors, axis=0)

models_list = ["mobilenetv2", "shufflenetv2", "squeezenet", "resnet18", "resnet50"]

print("[2/3] Quantizing models to static INT8 using existing ONNX models...")


class TinyDataReader(CalibrationDataReader):
    def __init__(self, data):
        self.data = data
        self.idx = 0

    def get_next(self):
        if self.idx < len(self.data):
            s = {"input": self.data[self.idx : self.idx + 1]}
            self.idx += 1
            return s
        return None


for name in models_list:
    fp32_p = f"models/{name}.onnx"
    int8_p = f"models/{name}_int8.onnx"
    if not os.path.exists(int8_p):
        reader = TinyDataReader(calib_array)
        try:
            quantize_static(
                fp32_p,
                int8_p,
                reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
            )
        except Exception as e:
            print(f"Fallback for {name}: {e}")
            from onnxruntime.quantization import quantize_dynamic

            quantize_dynamic(fp32_p, int8_p, weight_type=QuantType.QInt8)
        print(f"  -> Quantized {name}_int8.onnx")

print("[3/3] Executing benchmark streaming 1,000 real images on Local Host x86_64...")
results = []
providers = ["CPUExecutionProvider"]

for name in models_list:
    for strat, path, prec in [
        ("onnx_fp32", f"models/{name}.onnx", "fp32"),
        ("onnx_int8", f"models/{name}_int8.onnx", "int8"),
    ]:
        if not os.path.exists(path):
            continue
        size_mb = os.path.getsize(path) / (1024 * 1024)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(path, opts, providers=providers)
        inp = sess.get_inputs()[0].name

        # Warmup
        warm_img = preprocess_image(Image.open(selected_paths[0]).convert("RGB"))
        for _ in range(15):
            sess.run(None, {inp: warm_img})

        latencies = []
        t_start = time.perf_counter()
        for p in selected_paths:
            img = Image.open(p).convert("RGB")
            x = preprocess_image(img)
            t0 = time.perf_counter()
            sess.run(None, {inp: x})
            latencies.append((time.perf_counter() - t0) * 1000.0)

        total_sec = time.perf_counter() - t_start
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        fps = len(selected_paths) / total_sec

        print(
            f"[{name.upper()}] {strat} ({prec}) | p50: {p50:.2f} ms | "
            f"Throughput: {fps:.2f} FPS | Size: {size_mb:.2f} MB"
        )
        results.append(
            {
                "hardware": "x86_64_intel_cpu",
                "model": f"{name}.onnx",
                "strategy": strat,
                "precision": prec,
                "model_size_mb": round(size_mb, 2),
                "num_images_tested": len(selected_paths),
                "latency_p50_ms": round(p50, 4),
                "latency_p95_ms": round(p95, 4),
                "throughput_fps": round(fps, 2),
                "total_time_sec": round(total_sec, 2),
            }
        )
        del sess
        gc.collect()

df = pd.DataFrame(results)
df.to_csv("reports/data/benchmark_local_x86_1000images.csv", index=False)
print("\n" + "=" * 70)
print("LOCAL x86_64 BENCHMARK COMPLETE!")
print("=" * 70)
print(df.to_string())
