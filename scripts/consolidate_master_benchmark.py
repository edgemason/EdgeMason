import os

import pandas as pd

# Load datasets
gpu_df = pd.read_csv("reports/data/benchmark_gpu_t4_large_dataset.csv")
graviton_df = pd.read_csv("reports/data/benchmark_aws_graviton_1000images.csv")
x86_df = pd.read_csv("reports/data/benchmark_local_x86_1000images.csv")
rpi5_df = pd.read_csv("reports/data/benchmark_rpi5_all.csv")

# Standardize columns
cols = [
    "hardware",
    "model",
    "strategy",
    "precision",
    "model_size_mb",
    "latency_p50_ms",
    "latency_p95_ms",
    "throughput_fps",
]

# Normalize RPi5 columns
rpi5_onnx = rpi5_df[rpi5_df["framework"] == "onnxruntime"].copy()
rpi5_onnx["precision"] = rpi5_onnx["strategy"].apply(  # type: ignore[attr-defined]
    lambda x: "fp32" if "fp32" in x else "int8"
)

frames = [gpu_df[cols], graviton_df[cols], x86_df[cols], rpi5_onnx[cols]]

# Optional: GitHub Actions CI hardware (x86_64 / ARM64 Linux / Apple Silicon macOS),
# only included once their CSVs have been downloaded from the workflow run.
for tag in ["github_x86_64", "github_arm64_linux", "github_macos_arm64"]:
    ci_path = f"reports/data/benchmark_{tag}_1000images.csv"
    if os.path.exists(ci_path):
        frames.append(pd.read_csv(ci_path)[cols])
    else:
        print(f"Skipping {ci_path}: not downloaded yet")

# Combine
all_df = pd.concat(frames, ignore_index=True)

out_path = "reports/data/master_1000images_multihardware_benchmark.csv"
all_df.to_csv(out_path, index=False)
print(f"Consolidated master dataset written to {out_path}")
print(f"Total benchmark records: {len(all_df)}")
print(all_df.groupby(["hardware", "precision"])["latency_p50_ms"].count())
