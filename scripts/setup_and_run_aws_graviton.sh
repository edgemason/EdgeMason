#!/bin/bash
# ==============================================================================
# EdgeMason AWS Graviton ARM Benchmark Runner (t4g.small / ARM Neoverse)
# ==============================================================================
set -e

echo "[+] Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv wget git

echo "[+] Setting up Python ARM virtual environment..."
python3 -m venv ~/graviton_venv
source ~/graviton_venv/bin/activate

echo "[+] Installing ARM64 wheels for ONNX Runtime, PyTorch, and dependencies..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime onnxscript pandas numpy pillow tqdm psutil

echo "[+] Creating benchmark workspace..."
mkdir -p ~/graviton_benchmark/models
mkdir -p ~/graviton_benchmark/dataset
mkdir -p ~/graviton_benchmark/reports/data
cd ~/graviton_benchmark

echo "[+] Downloading real 1,000-image dataset (Imagenette)..."
if [ ! -d "dataset/imagenette2-160" ]; then
    wget -nc -q https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
    tar -xzf imagenette2-160.tgz -C dataset/
fi

cat << 'EOF' > run_graviton.py
import os, time, torch, onnx, torchvision
import numpy as np, pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

print("[1/4] Preprocessing 1,000 real images...")
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

selected = image_paths[:1000]
tensors = []
for p in tqdm(selected):
    img = Image.open(p).convert('RGB')
    tensors.append(preprocess(img).unsqueeze(0).numpy().astype(np.float32))
images_data = np.concatenate(tensors, axis=0)
np.save('dataset/real_images_1000.npy', images_data)

print("[2/4] Exporting 5 pretrained CNN models to ONNX...")
constructors = {
    'mobilenetv2': models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
    'resnet18': models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
    'resnet50': models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
    'shufflenetv2': models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT),
    'squeezenet': models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT),
}
dummy_input = torch.randn(1, 3, 224, 224)

for name, model in constructors.items():
    model.eval()
    p = f'models/{name}.onnx'
    torch.onnx.export(
        model, dummy_input, p,
        input_names=['input'], output_names=['output'],
        opset_version=13, dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )

print("[3/4] Quantizing models to static INT8 with real calibration...")
class RealDataReader(CalibrationDataReader):
    def __init__(self, data, max_samples=200):
        self.data = data
        self.max_samples = min(len(data), max_samples)
        self.idx = 0
    def get_next(self):
        if self.idx < self.max_samples:
            s = {'input': self.data[self.idx:self.idx+1]}
            self.idx += 1
            return s
        return None

for name in constructors.keys():
    fp32_p = f'models/{name}.onnx'
    int8_p = f'models/{name}_int8.onnx'
    reader = RealDataReader(images_data, 200)
    try:
        quantize_static(
            fp32_p, int8_p, reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True
        )
    except Exception as e:
        print(f"Fallback dynamic for {name}: {e}")
        from onnxruntime.quantization import quantize_dynamic
        quantize_dynamic(fp32_p, int8_p, weight_type=QuantType.QInt8)

print("[4/4] Executing benchmark across 1,000 real images on AWS Graviton ARM...")
results = []
providers = ['CPUExecutionProvider']

for name in constructors.keys():
    for strat, path, prec in [('onnx_fp32', f'models/{name}.onnx', 'fp32'), ('onnx_int8', f'models/{name}_int8.onnx', 'int8')]:
        if not os.path.exists(path):
            continue
        size_mb = os.path.getsize(path) / (1024 * 1024)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(path, opts, providers=providers)
        inp = sess.get_inputs()[0].name

        # Warmup
        for i in range(25):
            sess.run(None, {inp: images_data[i:i+1]})

        latencies = []
        t_start = time.perf_counter()
        for i in range(len(images_data)):
            t0 = time.perf_counter()
            sess.run(None, {inp: images_data[i:i+1]})
            latencies.append((time.perf_counter() - t0) * 1000.0)
        total_sec = time.perf_counter() - t_start

        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        fps = len(images_data) / total_sec

        print(f"[{name.upper()}] {strat} ({prec}) | p50: {p50:.2f} ms | Throughput: {fps:.2f} FPS | Size: {size_mb:.2f} MB")
        results.append({
            'hardware': 'aws_graviton2_t4g',
            'model': f'{name}.onnx',
            'strategy': strat,
            'precision': prec,
            'model_size_mb': round(size_mb, 2),
            'num_images_tested': len(images_data),
            'latency_p50_ms': round(p50, 4),
            'latency_p95_ms': round(p95, 4),
            'throughput_fps': round(fps, 2),
            'total_time_sec': round(total_sec, 2)
        })

df = pd.DataFrame(results)
df.to_csv('reports/data/benchmark_aws_graviton_1000images.csv', index=False)
print("\n" + "="*70)
print("AWS GRAVITON BENCHMARK COMPLETE! Saved to reports/data/benchmark_aws_graviton_1000images.csv")
print("="*70)
print(df.to_string())
EOF

python3 run_graviton.py
