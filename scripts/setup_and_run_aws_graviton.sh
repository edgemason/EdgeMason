#!/bin/bash
# ==============================================================================
# EdgeMason AWS Graviton ARM Benchmark Runner (Memory-Optimized for t4g.small)
# ==============================================================================
set -e

echo "[+] Creating 4GB Swap Space to prevent RAM OOM on t4g.small..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 4G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=4096
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo "Swap enabled!"
fi

cd ~/graviton_benchmark
source ~/graviton_venv/bin/activate

cat << 'EOF' > run_graviton.py
import os, time, gc, torch, onnx, torchvision
import numpy as np, pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

print("[1/4] Preparing image paths...")
image_paths = []
for root, _, files in os.walk('dataset/imagenette2-160/val'):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(root, f))

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

selected_paths = image_paths[:1000]
print(f"Total test images: {len(selected_paths)}")

# Prepare small calibration subset (100 images) to keep RAM usage tiny (<100MB)
calib_tensors = []
for p in selected_paths[:100]:
    img = Image.open(p).convert('RGB')
    calib_tensors.append(preprocess(img).unsqueeze(0).numpy().astype(np.float32))
calib_array = np.concatenate(calib_tensors, axis=0)

print("[2/4] Exporting 5 pretrained CNN models to ONNX...")
model_constructors = {
    'mobilenetv2': models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
    'shufflenetv2': models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT),
    'squeezenet': models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT),
    'resnet18': models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
    'resnet50': models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
}
dummy_input = torch.randn(1, 3, 224, 224)

for name, model in model_constructors.items():
    model.eval()
    p = f'models/{name}.onnx'
    if not os.path.exists(p):
        torch.onnx.export(
            model, dummy_input, p,
            input_names=['input'], output_names=['output'],
            opset_version=13, dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        print(f"  -> Exported {name}.onnx")

print("[3/4] Quantizing models to static INT8...")
class TinyDataReader(CalibrationDataReader):
    def __init__(self, data):
        self.data = data
        self.idx = 0
    def get_next(self):
        if self.idx < len(self.data):
            s = {'input': self.data[self.idx:self.idx+1]}
            self.idx += 1
            return s
        return None

for name in model_constructors.keys():
    fp32_p = f'models/{name}.onnx'
    int8_p = f'models/{name}_int8.onnx'
    if not os.path.exists(int8_p):
        reader = TinyDataReader(calib_array)
        try:
            quantize_static(
                fp32_p, int8_p, reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=True
            )
        except Exception:
            from onnxruntime.quantization import quantize_dynamic
            quantize_dynamic(fp32_p, int8_p, weight_type=QuantType.QInt8)
        print(f"  -> Quantized {name}_int8.onnx")

print("[4/4] Executing benchmark streaming 1,000 images on AWS Graviton2...")
results = []
providers = ['CPUExecutionProvider']

for name in model_constructors.keys():
    for strat, path, prec in [('onnx_fp32', f'models/{name}.onnx', 'fp32'), ('onnx_int8', f'models/{name}_int8.onnx', 'int8')]:
        if not os.path.exists(path):
            continue
        size_mb = os.path.getsize(path) / (1024 * 1024)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(path, opts, providers=providers)
        inp = sess.get_inputs()[0].name

        # Warmup
        warm_img = preprocess(Image.open(selected_paths[0]).convert('RGB')).unsqueeze(0).numpy().astype(np.float32)
        for _ in range(15):
            sess.run(None, {inp: warm_img})

        latencies = []
        t_start = time.perf_counter()
        for p in selected_paths:
            img = Image.open(p).convert('RGB')
            x = preprocess(img).unsqueeze(0).numpy().astype(np.float32)
            t0 = time.perf_counter()
            sess.run(None, {inp: x})
            latencies.append((time.perf_counter() - t0) * 1000.0)
            
        total_sec = time.perf_counter() - t_start
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        fps = len(selected_paths) / total_sec

        print(f"[{name.upper()}] {strat} ({prec}) | p50: {p50:.2f} ms | Throughput: {fps:.2f} FPS | Size: {size_mb:.2f} MB")
        results.append({
            'hardware': 'aws_graviton2_t4g',
            'model': f'{name}.onnx',
            'strategy': strat,
            'precision': prec,
            'model_size_mb': round(size_mb, 2),
            'num_images_tested': len(selected_paths),
            'latency_p50_ms': round(p50, 4),
            'latency_p95_ms': round(p95, 4),
            'throughput_fps': round(fps, 2),
            'total_time_sec': round(total_sec, 2)
        })
        del sess
        gc.collect()

df = pd.DataFrame(results)
df.to_csv('reports/data/benchmark_aws_graviton_1000images.csv', index=False)
print("\n" + "="*70)
print("AWS GRAVITON BENCHMARK COMPLETE!")
print("="*70)
print(df.to_string())
EOF

python3 run_graviton.py
