"""Self-contained CPU benchmark for GitHub Actions runners.

Mirrors scripts/run_local_x86_1000images.py and setup_and_run_aws_graviton.sh
(same preprocessing, warmup/measurement protocol, and CSV schema) so results
are directly comparable across hardware. Parameterized by --hardware-tag so
the same script covers the x86_64, ARM64 Linux, and Apple Silicon CI runners.

Usage:
    python scripts/run_ci_benchmark.py --hardware-tag github_x86_64
"""

import argparse
import gc
import os
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from PIL import Image

MODEL_CONSTRUCTORS = {
    "mobilenetv2": lambda: models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    ),
    "shufflenetv2": lambda: models.shufflenet_v2_x1_0(
        weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT
    ),
    "squeezenet": lambda: models.squeezenet1_1(
        weights=models.SqueezeNet1_1_Weights.DEFAULT
    ),
    "resnet18": lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
    "resnet50": lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
}

PREPROCESS = transforms.Compose(
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
    tensor: torch.Tensor = PREPROCESS(img)  # type: ignore[assignment]
    return tensor.unsqueeze(0).numpy().astype(np.float32)


class TinyDataReader(CalibrationDataReader):
    def __init__(self, data):
        self.data = data
        self.idx = 0

    def get_next(self):
        if self.idx < len(self.data):
            sample = {"input": self.data[self.idx : self.idx + 1]}
            self.idx += 1
            return sample
        return None


def export_models():
    dummy_input = torch.randn(1, 3, 224, 224)
    for name, ctor in MODEL_CONSTRUCTORS.items():
        p = f"models/{name}.onnx"
        if os.path.exists(p):
            continue
        model = ctor()
        model.eval()
        torch.onnx.export(
            model,
            (dummy_input,),
            p,
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            dynamo=False,
        )
        print(f"  -> Exported {name}.onnx")


def quantize_models(calib_array):
    for name in MODEL_CONSTRUCTORS:
        fp32_p = f"models/{name}.onnx"
        int8_p = f"models/{name}_int8.onnx"
        if os.path.exists(int8_p):
            continue
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
            print(f"Fallback (dynamic) for {name}: {e}")
            quantize_dynamic(fp32_p, int8_p, weight_type=QuantType.QInt8)
        print(f"  -> Quantized {name}_int8.onnx")


def run_benchmark(hardware_tag, selected_paths, num_threads):
    results = []
    providers = ["CPUExecutionProvider"]
    for name in MODEL_CONSTRUCTORS:
        for strat, path, prec in [
            ("onnx_fp32", f"models/{name}.onnx", "fp32"),
            ("onnx_int8", f"models/{name}_int8.onnx", "int8"),
        ]:
            if not os.path.exists(path):
                continue
            size_mb = os.path.getsize(path) / (1024 * 1024)
            opts = ort.SessionOptions()
            if num_threads:
                opts.intra_op_num_threads = num_threads
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(path, opts, providers=providers)
            inp = sess.get_inputs()[0].name

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
                    "hardware": hardware_tag,
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
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Self-contained CPU benchmark for CI runners."
    )
    parser.add_argument(
        "--hardware-tag",
        required=True,
        help="e.g. github_x86_64, github_arm64_linux, github_macos_arm64",
    )
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="0 = onnxruntime default (use all cores)",
    )
    parser.add_argument("--dataset-dir", default="dataset/imagenette2-160/val")
    args = parser.parse_args()

    os.makedirs("reports/data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print(
        f"[1/4] Collecting up to {args.num_images} images from {args.dataset_dir} ..."
    )
    image_paths = []
    for root, _, files in os.walk(args.dataset_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, f))
    if not image_paths:
        raise SystemExit(f"No images found under {args.dataset_dir}")
    selected_paths = image_paths[: args.num_images]
    print(f"Total test images: {len(selected_paths)}")

    print("[2/4] Exporting 5 pretrained CNN models to ONNX (skip if present)...")
    export_models()

    calib_tensors = [
        preprocess_image(Image.open(p).convert("RGB")) for p in selected_paths[:100]
    ]
    calib_array = np.concatenate(calib_tensors, axis=0)

    print("[3/4] Quantizing models to static INT8 (skipped if already present)...")
    quantize_models(calib_array)

    print(f"[4/4] Benchmarking {len(selected_paths)} images on {args.hardware_tag} ...")
    results = run_benchmark(args.hardware_tag, selected_paths, args.num_threads)

    df = pd.DataFrame(results)
    out_csv = f"reports/data/benchmark_{args.hardware_tag}_1000images.csv"
    df.to_csv(out_csv, index=False)
    print("\n" + "=" * 70)
    print(f"{args.hardware_tag.upper()} BENCHMARK COMPLETE -> {out_csv}")
    print("=" * 70)
    print(df.to_string())


if __name__ == "__main__":
    main()
