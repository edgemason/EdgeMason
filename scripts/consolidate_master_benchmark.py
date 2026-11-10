import pandas as pd, numpy as np

# Load datasets
gpu_df = pd.read_csv('reports/data/benchmark_gpu_t4_large_dataset.csv')
graviton_df = pd.read_csv('reports/data/benchmark_aws_graviton_1000images.csv')
x86_df = pd.read_csv('reports/data/benchmark_local_x86_1000images.csv')
rpi5_df = pd.read_csv('reports/data/benchmark_rpi5_all.csv')

# Standardize columns
cols = ['hardware', 'model', 'strategy', 'precision', 'model_size_mb', 'latency_p50_ms', 'latency_p95_ms', 'throughput_fps']

# Normalize RPi5 columns
rpi5_onnx = rpi5_df[rpi5_df['framework'] == 'onnxruntime'].copy()
rpi5_onnx['precision'] = rpi5_onnx['strategy'].apply(lambda x: 'fp32' if 'fp32' in x else 'int8')
rpi5_onnx = rpi5_onnx.rename(columns={'latency_p50_ms': 'latency_p50_ms', 'latency_p95_ms': 'latency_p95_ms', 'throughput_fps': 'throughput_fps'})

# Combine
all_df = pd.concat([
    gpu_df[cols],
    graviton_df[cols],
    x86_df[cols],
    rpi5_onnx[cols]
], ignore_index=True)

all_df.to_csv('reports/data/master_1000images_multihardware_benchmark.csv', index=False)
print("Consolidated master dataset written to reports/data/master_1000images_multihardware_benchmark.csv")
print(f"Total benchmark records: {len(all_df)}")
print(all_df.groupby(['hardware', 'precision'])['latency_p50_ms'].count())
