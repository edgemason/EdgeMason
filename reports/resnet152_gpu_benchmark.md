# ResNet-152 GPU benchmark

Date: 2026-09-15

## Environment

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6 GiB
- Driver: 580.178.04
- Host CUDA: 13.0
- Benchmark runtime: ONNX Runtime GPU 1.27.0
- Input shape: `(1, 3, 224, 224)`
- Method: 10 warm-up runs followed by 100 measured runs
- CUDA measurements disable CPU execution-provider fallback

## Results

| Candidate | Provider | Build (s) | Size (MiB) | p50 (ms) | Average (ms) | p95 (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| FP32 | CUDA | 1.014 | 229.398 | 7.3904 | 7.5896 | 8.5094 | 131.70 |
| FP32 | CPU | 0.470 | 229.398 | 168.7225 | 166.8892 | 172.9076 | 5.99 |
| Dynamic INT8 | CPU | 1.813 | 57.868 | 164.4200 | 164.7827 | 169.6974 | 6.07 |
| Static INT8 | CPU | 9.707 | 57.854 | 97.7383 | 101.3138 | 108.7450 | 9.87 |

## CUDA compatibility

FP32 runs entirely with `CUDAExecutionProvider`. The dynamic and static INT8
graphs do not run with strict CUDA execution in this ONNX Runtime configuration:
ONNX Runtime assigns some graph nodes to its CPU provider. The benchmark rejects
that fallback so mixed CPU/GPU execution is not mislabeled as a GPU result.

For `execution_provider=auto`, EdgeMason tries strict CUDA first and truthfully
reports CPU when the graph cannot run entirely on CUDA. An explicit
`execution_provider=cuda` request returns a failed candidate instead.

Genuine GPU INT8 benchmarking requires a GPU backend with INT8 graph support,
such as a configured TensorRT execution provider and an appropriate calibration
and engine-build path.

## Deployment verification

The complete calibrated deployment finished in **49.771 seconds** with TVM
disabled. It evaluated three ONNX candidates and selected the FP32 CUDA model.
The original deployment exceeded the client timeout because its default plan
also evaluated two slow TVM candidates.
